"""Google Drive ingestion — OAuth 2.0 + Changes API delta sync.

Authentication:
  - Uses InstalledAppFlow for the desktop OAuth flow, same pattern as gmail.py.
  - Tokens are persisted in ~/.verra/oauth/<account>_drive_token.json.
  - credentials.json must live in ~/.verra/oauth/ (same file used by Gmail).

Usage:
  ingestor = DriveIngestor("you@gmail.com")
  if ingestor.authenticate():
      stats = ingest_drive(ingestor, metadata_store, vector_store)

Supported file types:
  - Regular: PDF, DOCX, TXT, MD, CSV, XLSX
  - Google Workspace: Docs (→ plain text), Sheets (→ CSV), Slides (→ PDF)
"""


from __future__ import annotations

import hashlib
import io
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# OAuth scopes — read-only access to Drive files
_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Exponential backoff config (mirrors gmail.py)
_MAX_RETRIES = 5
_INITIAL_BACKOFF = 1.0  # seconds

# ---------------------------------------------------------------------------
# MIME type mappings
# ---------------------------------------------------------------------------

# Google Workspace types that require export rather than direct download
_GOOGLE_EXPORT_MAP: dict[str, tuple[str, str]] = {
    # mime_type → (export_mime_type, file_extension)
    "application/vnd.google-apps.document": ("text/plain", ".txt"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.drawing": ("application/pdf", ".pdf"),
}

# Regular file types we support downloading directly
_SUPPORTED_MIME_TYPES: set[str] = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/plain",
    "text/csv",
    "text/markdown",
}

# Extension to use when writing a temp file for a given regular mime type
_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
}


# ---------------------------------------------------------------------------
# API call with exponential backoff (mirrors gmail.py helper)
# ---------------------------------------------------------------------------


def _api_call_with_backoff(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute fn(*args, **kwargs) with exponential backoff on HTTP 429/5xx."""
    delay = _INITIAL_BACKOFF
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "503" in exc_str or "500" in exc_str:
                logger.warning(
                    "Drive API rate limit / server error (attempt %d): %s",
                    attempt + 1,
                    exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60)
                last_exc = exc
            else:
                raise
    raise RuntimeError(f"Drive API call failed after {_MAX_RETRIES} retries") from last_exc


# ---------------------------------------------------------------------------
# DriveIngestor
# ---------------------------------------------------------------------------


class DriveIngestor:
    """Google Drive file ingestor with OAuth 2.0 and Changes API delta sync."""

    def __init__(
        self,
        account: str,
        credentials_dir: str | None = None,
    ) -> None:
        self.account = account

        from verra.config import VERRA_HOME

        self._oauth_dir = Path(credentials_dir) if credentials_dir else VERRA_HOME / "oauth"
        self._oauth_dir.mkdir(parents=True, exist_ok=True)

        # credentials.json is shared with Gmail (same Google Cloud project)
        self._credentials_path = self._oauth_dir / "credentials.json"
        # Per-account, per-service token to avoid collisions with Gmail token
        safe_account = re.sub(r"[^a-zA-Z0-9._-]", "_", account)
        self._token_path = self._oauth_dir / f"{safe_account}_drive_token.json"

        self._service: Any = None  # googleapiclient Resource, set after authenticate()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """Run the OAuth 2.0 desktop app flow and persist the refresh token.

        Returns True on success, False if credentials.json is missing.
        """
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise ImportError(
                "Google API packages are required: "
                "pip install google-auth-oauthlib google-api-python-client"
            ) from exc

        if not self._credentials_path.exists():
            self._print_setup_instructions()
            return False

        creds: Credentials | None = None

        # Load cached token if it exists
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self._token_path), _SCOPES)

        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                logger.warning("Drive token refresh failed (%s), re-authenticating.", exc)
                creds = None

        # First-time auth or refresh failed — run interactive flow
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self._credentials_path), _SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Persist token for future runs
        self._token_path.write_text(creds.to_json())
        # Restrict token file permissions — contains OAuth credentials
        try:
            self._token_path.chmod(0o600)
        except OSError:
            pass
        logger.info("Drive token saved to %s", self._token_path)

        self._service = build("drive", "v3", credentials=creds)
        return True

    def _print_setup_instructions(self) -> None:
        print(
            "\nTo connect Google Drive, create a Google Cloud project and download credentials:\n"
            "  1. Go to https://console.cloud.google.com/\n"
            "  2. Create a new project (or select an existing one)\n"
            "  3. Enable the Google Drive API  (APIs & Services > Enable APIs > search 'Drive API')\n"
            "  4. Create OAuth credentials  (APIs & Services > Credentials > Create Credentials\n"
            "     > OAuth client ID > Desktop app)\n"
            "  5. Download the JSON file and save it as:\n"
            f"     {self._credentials_path}\n"
            "  6. Re-run this command.\n"
            "\n"
            "  If you have already set up Gmail OAuth, use the same credentials.json —\n"
            "  just add the Drive API to the same project.\n"
        )

    def _require_service(self) -> Any:
        """Return the authenticated service or attempt silent re-auth from cache."""
        if self._service is None:
            if not self.authenticate():
                raise RuntimeError(
                    "Not authenticated. Call authenticate() before using the Drive API."
                )
        return self._service

    # ------------------------------------------------------------------
    # Listing files
    # ------------------------------------------------------------------

    def list_files(
        self,
        folder_id: str | None = None,
        mime_types: list[str] | None = None,
        max_results: int = 500,
    ) -> list[dict[str, Any]]:
        """List files in Drive, optionally filtered by folder and/or mime type.

        Parameters
        ----------
        folder_id:
            If provided, restrict listing to files whose parent is this folder.
            Pass None to search the entire Drive.
        mime_types:
            Optional list of MIME types to include.  Defaults to all supported
            types (regular downloads + Google Workspace exports).
        max_results:
            Maximum number of file records to return.

        Returns
        -------
        List of dicts with keys: id, name, mimeType, modifiedTime, size
        """
        svc = self._require_service()

        # Build query
        query_parts: list[str] = ["trashed = false"]

        if folder_id:
            # H-05: Validate folder ID to prevent Drive API query injection
            if not __import__("re").fullmatch(r"[a-zA-Z0-9_-]{10,60}", folder_id):
                raise ValueError(f"Invalid Drive folder ID: {folder_id!r}")
            query_parts.append(f"'{folder_id}' in parents")

        # Determine the set of mime types to include
        if mime_types is None:
            # Default: all supported regular types + all Google Workspace types
            all_supported = list(_SUPPORTED_MIME_TYPES) + list(_GOOGLE_EXPORT_MAP.keys())
            mime_types = all_supported

        if mime_types:
            type_clauses = " or ".join(
                f"mimeType = '{mt}'" for mt in mime_types
            )
            query_parts.append(f"({type_clauses})")

        query = " and ".join(query_parts)

        files: list[dict[str, Any]] = []
        page_token: str | None = None

        while len(files) < max_results:
            request_kwargs: dict[str, Any] = {
                "pageSize": min(100, max_results - len(files)),
                "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                "q": query,
            }
            if page_token:
                request_kwargs["pageToken"] = page_token

            resp = _api_call_with_backoff(
                svc.files().list(**request_kwargs).execute
            )
            batch = resp.get("files", [])
            files.extend(batch)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        logger.info("Drive list_files found %d files (account=%s)", len(files), self.account)
        return files[:max_results]

    # ------------------------------------------------------------------
    # Downloading files
    # ------------------------------------------------------------------

    def download_file(self, file_id: str, mime_type: str) -> bytes | None:
        """Download a file's content as bytes.

        Google Workspace files (Docs, Sheets, Slides) are exported to a
        compatible open format before download.  Regular files are fetched
        directly.

        Returns None if the file is empty or download fails.
        """
        svc = self._require_service()

        try:
            if mime_type in _GOOGLE_EXPORT_MAP:
                # Export Google Workspace file to an open format
                export_mime, _ = _GOOGLE_EXPORT_MAP[mime_type]
                request = svc.files().export_media(fileId=file_id, mimeType=export_mime)
            else:
                # Direct binary download for regular files
                request = svc.files().get_media(fileId=file_id)

            buf = io.BytesIO()
            # Use MediaIoBaseDownload for chunked streaming (handles large files)
            from googleapiclient.http import MediaIoBaseDownload

            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = _api_call_with_backoff(downloader.next_chunk)

            data = buf.getvalue()
            return data if data else None

        except Exception as exc:
            logger.warning("Failed to download file %s (type=%s): %s", file_id, mime_type, exc)
            return None

    # ------------------------------------------------------------------
    # Changes API (delta sync)
    # ------------------------------------------------------------------

    def get_start_page_token(self) -> str:
        """Fetch the current start page token for the Changes feed."""
        svc = self._require_service()
        resp = _api_call_with_backoff(
            svc.changes().getStartPageToken().execute
        )
        return str(resp.get("startPageToken", ""))

    def get_changes(
        self, start_page_token: str | None = None
    ) -> tuple[list[dict[str, Any]], str]:
        """Fetch changed files since the last sync using the Changes API.

        Parameters
        ----------
        start_page_token:
            The page token from the previous sync.  Pass None to fall back to
            a full listing on the next ingest_drive() call.

        Returns
        -------
        (changed_files, new_page_token)
            changed_files is a list of file metadata dicts (same shape as
            list_files() output) for files that were added or modified.
            new_page_token should be persisted for the next delta sync.
        """
        svc = self._require_service()

        if not start_page_token:
            # No cursor — caller should do a full listing instead
            token = self.get_start_page_token()
            return [], token

        changed_files: list[dict[str, Any]] = []
        page_token: str = start_page_token
        new_page_token: str = start_page_token

        while True:
            try:
                resp = _api_call_with_backoff(
                    svc.changes().list(
                        pageToken=page_token,
                        fields=(
                            "nextPageToken, newStartPageToken, "
                            "changes(removed, file(id, name, mimeType, modifiedTime, size, trashed))"
                        ),
                    ).execute
                )
            except Exception as exc:
                logger.warning(
                    "Drive changes.list failed (token=%s): %s — will fall back to full sync",
                    page_token,
                    exc,
                )
                # Return a fresh token so the next call does a full listing
                new_page_token = self.get_start_page_token()
                return [], new_page_token

            for change in resp.get("changes", []):
                if change.get("removed"):
                    continue  # Deleted files — skip (we can't re-download them)
                file_meta = change.get("file")
                if not file_meta:
                    continue
                if file_meta.get("trashed"):
                    continue
                mime = file_meta.get("mimeType", "")
                if mime in _GOOGLE_EXPORT_MAP or mime in _SUPPORTED_MIME_TYPES:
                    changed_files.append(file_meta)

            new_page_token = resp.get("newStartPageToken", new_page_token)
            next_page = resp.get("nextPageToken")
            if not next_page:
                break
            page_token = next_page

        logger.info(
            "Drive get_changes found %d changed files (account=%s)",
            len(changed_files),
            self.account,
        )
        return changed_files, new_page_token


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extension_for_file(mime_type: str, file_name: str) -> str:
    """Return the appropriate temp file extension for a given mime type."""
    if mime_type in _GOOGLE_EXPORT_MAP:
        _, ext = _GOOGLE_EXPORT_MAP[mime_type]
        return ext
    # Use MIME mapping, fall back to the original file's extension
    if mime_type in _MIME_TO_EXT:
        return _MIME_TO_EXT[mime_type]
    return Path(file_name).suffix or ".bin"


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Public pipeline function
# ---------------------------------------------------------------------------


def ingest_drive(
    ingestor: DriveIngestor,
    metadata_store: Any,
    vector_store: Any,
    folder_id: str | None = None,
    max_results: int = 500,
    force_reindex: bool = False,
) -> Any:
    """Ingest files from Google Drive into Verra.

    Steps
    -----
    1. Check sync state for a stored page token (delta sync) or do a full list.
    2. For each file, download its content.
    3. Write content to a temp file and extract text via ContentExtractor.
    4. Chunk the extracted text.
    5. Store chunks in metadata_store and vector_store.
    6. Persist the new Changes API page token for the next delta sync.

    Parameters
    ----------
    ingestor:
        An authenticated DriveIngestor instance.
    metadata_store:
        MetadataStore instance.
    vector_store:
        VectorStore instance.
    folder_id:
        Optional Drive folder ID to restrict ingestion scope.
    max_results:
        Maximum number of files to process per run.
    force_reindex:
        Re-process files even when the content hash is unchanged.

    Returns
    -------
    IngestStats
    """
    import time as _time

    from verra.ingest.chunking import chunk_document
    from verra.ingest.extractors import detect_and_extract
    from verra.ingest.pipeline import IngestStats

    stats = IngestStats()
    t0 = _time.monotonic()

    source_key = f"drive:{ingestor.account}"
    sync_state = metadata_store.get_sync_state(source_key)
    page_token: str | None = sync_state.get("cursor") if sync_state else None

    # Decide: delta fetch or full listing
    if page_token and not force_reindex:
        logger.info(
            "Drive delta sync from page_token=%s... for %s",
            page_token[:12],
            ingestor.account,
        )
        metadata_store.upsert_sync_state(
            source=source_key,
            cursor=page_token,
            items_processed=sync_state.get("items_processed", 0) if sync_state else 0,
            status="syncing",
        )
        try:
            files, new_page_token = ingestor.get_changes(page_token)
        except Exception as exc:
            stats.errors.append(f"get_changes failed: {exc}")
            metadata_store.upsert_sync_state(
                source=source_key,
                cursor=page_token,
                items_processed=sync_state.get("items_processed", 0) if sync_state else 0,
                status="error",
            )
            stats.elapsed_seconds = _time.monotonic() - t0
            return stats
    else:
        logger.info("Drive full listing for %s (folder_id=%s)", ingestor.account, folder_id)
        metadata_store.upsert_sync_state(
            source=source_key, cursor=None, items_processed=0, status="syncing",
        )
        try:
            files = ingestor.list_files(folder_id=folder_id, max_results=max_results)
            new_page_token = ingestor.get_start_page_token()
        except Exception as exc:
            stats.errors.append(f"list_files failed: {exc}")
            metadata_store.upsert_sync_state(
                source=source_key, cursor=None, items_processed=0, status="error",
            )
            stats.elapsed_seconds = _time.monotonic() - t0
            return stats

    stats.files_found = len(files)

    for file_meta in files:
        file_id: str = file_meta.get("id", "")
        file_name: str = file_meta.get("name", "unknown")
        mime_type: str = file_meta.get("mimeType", "")
        modified_time: str = file_meta.get("modifiedTime", "")

        if not file_id or not mime_type:
            stats.files_skipped += 1
            continue

        try:
            # Download file content
            data = ingestor.download_file(file_id, mime_type)
            if not data:
                logger.debug("Skipping empty file: %s (%s)", file_name, file_id)
                stats.files_skipped += 1
                continue

            # Skip unchanged content (content-addressable dedup)
            content_hash = _content_hash(data)
            if not force_reindex:
                existing = metadata_store.get_document_by_hash(content_hash)
                if existing is not None:
                    logger.debug("Skipping unchanged file: %s", file_name)
                    stats.files_skipped += 1
                    continue

            # Remove stale document entry for this Drive file (re-index case)
            drive_path = f"drive:{ingestor.account}:{file_id}"
            old_doc = metadata_store.get_document_by_path(drive_path)
            if old_doc is not None:
                vector_store.delete_by_document_id(old_doc["id"])
                metadata_store.delete_document(old_doc["id"])

            # Write to a temp file so existing extractors can handle it
            ext = _extension_for_file(mime_type, file_name)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="wb") as tmp:
                import os as _os
                _os.chmod(tmp.name, 0o600)
                tmp.write(data)
                tmp_path = Path(tmp.name)

            try:
                extracted = detect_and_extract(tmp_path)
            except Exception as exc:
                logger.warning("Text extraction failed for %s: %s", file_name, exc)
                stats.errors.append(f"{file_name}: extraction failed: {exc}")
                stats.files_skipped += 1
                tmp_path.unlink(missing_ok=True)
                continue
            finally:
                tmp_path.unlink(missing_ok=True)

            if not extracted.content.strip():
                logger.debug("Skipping file with no extractable text: %s", file_name)
                stats.files_skipped += 1
                continue

            # Register document in metadata store
            doc_id = metadata_store.add_document(
                file_path=drive_path,
                file_name=file_name[:255],
                source_type="drive",
                format=extracted.format,
                content_hash=content_hash,
                page_count=extracted.page_count,
                extra_metadata={
                    "file_id": file_id,
                    "account": ingestor.account,
                    "mime_type": mime_type,
                    "modified_time": modified_time,
                    "folder_id": folder_id,
                },
            )

            # Chunk the extracted text
            chunks = chunk_document(
                extracted.content,
                metadata={
                    "document_id": doc_id,
                    "file_name": file_name,
                    "file_path": drive_path,
                    "format": extracted.format,
                    "source_type": "drive",
                    "file_id": file_id,
                    "account": ingestor.account,
                    "mime_type": mime_type,
                },
            )

            chunk_ids = metadata_store.add_chunks(doc_id, chunks)
            vector_store.add_chunks(chunk_ids, chunks)

            stats.files_processed += 1
            stats.chunks_created += len(chunks)
            logger.debug(
                "Ingested Drive file %s → %d chunks (doc_id=%s)",
                file_name,
                len(chunks),
                doc_id,
            )

        except Exception as exc:
            logger.exception("Error ingesting Drive file %s (%s)", file_name, file_id)
            stats.errors.append(f"{file_name}:{file_id}: {exc}")
            stats.files_skipped += 1

    stats.elapsed_seconds = _time.monotonic() - t0

    # Persist sync cursor for next delta run
    metadata_store.upsert_sync_state(
        source=source_key,
        cursor=new_page_token,
        items_processed=stats.files_processed,
        status="idle",
    )

    logger.info(
        "Drive ingest complete for %s: %d processed, %d skipped, %d errors in %.1fs",
        ingestor.account,
        stats.files_processed,
        stats.files_skipped,
        len(stats.errors),
        stats.elapsed_seconds,
    )
    return stats
