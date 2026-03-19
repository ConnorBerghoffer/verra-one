"""Gmail ingestion — OAuth 2.0 + history-based delta sync.

Authentication:
  - Uses InstalledAppFlow for the desktop OAuth flow.
  - Tokens are persisted in ~/.verra/oauth/<account>_token.json.
  - credentials.json must live in ~/.verra/oauth/ (shared with Drive — same Google Cloud project).

Usage:
  ingestor = GmailIngestor("you@gmail.com")
  if ingestor.authenticate():
      stats = ingest_gmail(ingestor, metadata_store, vector_store)
"""


from __future__ import annotations

import base64
import hashlib
import logging
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Gmail API OAuth scopes — read-only is sufficient
_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Exponential backoff limits for API rate-limit handling
_MAX_RETRIES = 5
_INITIAL_BACKOFF = 1.0  # seconds


# ---------------------------------------------------------------------------
# HTML → plain-text helper
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper that preserves newlines at block boundaries."""

    _BLOCK_TAGS = {
        "p", "div", "br", "li", "tr", "h1", "h2", "h3",
        "h4", "h5", "h6", "blockquote", "pre",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self._parts.append("\n")

    def get_text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return readable plain text."""
    stripper = _HTMLStripper()
    stripper.feed(html)
    text = stripper.get_text()
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# MIME body extraction helpers
# ---------------------------------------------------------------------------


def _decode_body(data: str) -> str:
    """Base64url-decode a Gmail message body part and return UTF-8 text."""
    try:
        raw = base64.urlsafe_b64decode(data + "==")  # padding tolerance
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body(payload: dict[str, Any]) -> str:
    """Recursively walk the MIME payload tree.

    Prefers text/plain; falls back to text/html (converted to plain text).
    Returns empty string when no readable part is found.
    """
    mime_type: str = payload.get("mimeType", "")
    body: dict[str, Any] = payload.get("body", {})
    parts: list[dict[str, Any]] = payload.get("parts", [])

    if mime_type == "text/plain":
        data = body.get("data", "")
        return _decode_body(data) if data else ""

    if mime_type == "text/html":
        data = body.get("data", "")
        if data:
            return _html_to_text(_decode_body(data))
        return ""

    # multipart/* — try text/plain subtrees first, then text/html
    plain_texts: list[str] = []
    html_texts: list[str] = []
    for part in parts:
        part_mime = part.get("mimeType", "")
        extracted = _extract_body(part)
        if extracted:
            if "html" in part_mime:
                html_texts.append(extracted)
            else:
                plain_texts.append(extracted)

    if plain_texts:
        return "\n".join(plain_texts)
    if html_texts:
        return "\n".join(html_texts)
    return ""


def _extract_header(headers: list[dict[str, str]], name: str) -> str:
    """Return the value of the first header matching *name* (case-insensitive)."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


# ---------------------------------------------------------------------------
# API call with exponential backoff
# ---------------------------------------------------------------------------


def _api_call_with_backoff(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute *fn(*args, **kwargs)* with exponential backoff on HTTP 429/5xx."""
    delay = _INITIAL_BACKOFF
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            # googleapiclient.errors.HttpError carries status in its string repr
            exc_str = str(exc)
            if "429" in exc_str or "503" in exc_str or "500" in exc_str:
                logger.warning("Gmail API rate limit / server error (attempt %d): %s", attempt + 1, exc)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                last_exc = exc
            else:
                raise
    raise RuntimeError(f"Gmail API call failed after {_MAX_RETRIES} retries") from last_exc


# ---------------------------------------------------------------------------
# GmailIngestor
# ---------------------------------------------------------------------------


class GmailIngestor:
    """Full Gmail ingestor with OAuth 2.0 and history-based delta sync."""

    def __init__(
        self,
        account: str,
        credentials_dir: str | None = None,
    ) -> None:
        self.account = account

        from verra.config import VERRA_HOME

        self._oauth_dir = Path(credentials_dir) if credentials_dir else VERRA_HOME / "oauth"
        self._oauth_dir.mkdir(parents=True, exist_ok=True)

        # Derived paths — credentials.json is shared with Drive (same Google Cloud project)
        self._credentials_path = self._oauth_dir / "credentials.json"
        # Per-account token so multiple accounts can coexist
        safe_account = re.sub(r"[^a-zA-Z0-9._-]", "_", account)
        self._token_path = self._oauth_dir / f"{safe_account}_token.json"

        self._service: Any = None  # googleapiclient Resource, set after authenticate()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> bool:
        """Run the OAuth 2.0 desktop app flow and persist the refresh token.

        Returns True on success, False if credentials file is missing.
        """
        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
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

        # Refresh or re-authenticate
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                logger.warning("Token refresh failed (%s), re-authenticating.", exc)
                creds = None

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self._credentials_path), _SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Persist the token for future runs
        self._token_path.write_text(creds.to_json())
        # Restrict token file permissions — contains OAuth credentials
        try:
            self._token_path.chmod(0o600)
        except OSError:
            pass
        logger.info("Token saved to %s", self._token_path)

        self._service = build("gmail", "v1", credentials=creds)
        return True

    def _print_setup_instructions(self) -> None:
        print(
            "\n  Gmail requires OAuth credentials. One-time setup:\n"
            "\n"
            "  1. Go to console.cloud.google.com\n"
            "  2. Create a project (or use existing)\n"
            "  3. Enable the Gmail API\n"
            "  4. Create OAuth 2.0 credentials -> Desktop application\n"
            "  5. Download the JSON file\n"
            f"  6. Save it as: {self._credentials_path}\n"
            "\n"
            f"  Then run: verra gmail {self.account}\n"
        )

    def _require_service(self) -> Any:
        """Return the authenticated service or raise if authenticate() hasn't been called."""
        if self._service is None:
            # Try a silent re-auth from cached token
            if not self.authenticate():
                raise RuntimeError(
                    "Not authenticated. Call authenticate() before fetching emails."
                )
        return self._service

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch_threads(
        self,
        since: str | None = None,
        labels: list[str] | None = None,
        max_results: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch and hydrate email threads from Gmail.

        Parameters
        ----------
        since:
            ISO-8601 date string (e.g. "2024-01-01").  Converted to a
            Gmail ``after:`` query.
        labels:
            Optional list of Gmail label IDs to restrict the search.
        max_results:
            Maximum number of messages to retrieve.

        Returns
        -------
        List of thread dicts, each with keys:
          thread_id, subject, messages (list of message dicts)
        """
        svc = self._require_service()

        query_parts: list[str] = []
        if since:
            # Gmail's after: operator expects YYYY/MM/DD
            date_str = since.replace("-", "/")
            query_parts.append(f"after:{date_str}")
        query = " ".join(query_parts) if query_parts else ""

        # List messages (paged)
        message_ids: list[str] = []
        page_token: str | None = None
        while len(message_ids) < max_results:
            request_kwargs: dict[str, Any] = {
                "userId": "me",
                "maxResults": min(500, max_results - len(message_ids)),
            }
            if query:
                request_kwargs["q"] = query
            if labels:
                request_kwargs["labelIds"] = labels
            if page_token:
                request_kwargs["pageToken"] = page_token

            resp = _api_call_with_backoff(
                svc.users().messages().list(**request_kwargs).execute
            )
            batch = resp.get("messages", [])
            message_ids.extend(m["id"] for m in batch)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        # Group by thread
        threads_map: dict[str, list[dict[str, Any]]] = {}
        for msg_id in message_ids:
            msg = _api_call_with_backoff(
                svc.users().messages().get(
                    userId="me",
                    id=msg_id,
                    format="full",
                ).execute
            )
            thread_id: str = msg.get("threadId", msg_id)
            threads_map.setdefault(thread_id, []).append(msg)

        return [
            self._hydrate_thread(thread_id, msgs)
            for thread_id, msgs in threads_map.items()
        ]

    def _hydrate_thread(
        self, thread_id: str, raw_messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Convert raw Gmail API message dicts into a normalised thread dict."""
        # Sort oldest-first by internalDate
        sorted_msgs = sorted(
            raw_messages,
            key=lambda m: int(m.get("internalDate", 0)),
        )

        messages: list[dict[str, Any]] = []
        for raw in sorted_msgs:
            payload = raw.get("payload", {})
            headers = payload.get("headers", [])
            body = _extract_body(payload)

            messages.append(
                {
                    "message_id": raw.get("id", ""),
                    "thread_id": thread_id,
                    "from": _extract_header(headers, "From"),
                    "to": _extract_header(headers, "To"),
                    "cc": _extract_header(headers, "Cc"),
                    "date": _extract_header(headers, "Date"),
                    "subject": _extract_header(headers, "Subject"),
                    "body": body,
                    "labels": raw.get("labelIds", []),
                    "internal_date": raw.get("internalDate", ""),
                }
            )

        subject = messages[0]["subject"] if messages else ""
        return {
            "thread_id": thread_id,
            "subject": subject,
            "messages": messages,
        }

    # ------------------------------------------------------------------
    # Delta sync
    # ------------------------------------------------------------------

    def delta_fetch(self, history_id: str) -> tuple[list[dict[str, Any]], str]:
        """Fetch messages added since *history_id*.

        Uses Gmail's history.list API which is far more efficient than a
        full re-fetch for polling.

        Returns
        -------
        (new_threads, new_history_id)
        """
        svc = self._require_service()

        new_message_ids: list[str] = []
        page_token: str | None = None

        while True:
            kwargs: dict[str, Any] = {
                "userId": "me",
                "startHistoryId": history_id,
                "historyTypes": ["messageAdded"],
            }
            if page_token:
                kwargs["pageToken"] = page_token

            try:
                resp = _api_call_with_backoff(
                    svc.users().history().list(**kwargs).execute
                )
            except Exception as exc:
                # historyId too old → fall back to a full fetch
                if "404" in str(exc) or "invalid" in str(exc).lower():
                    logger.warning(
                        "historyId %s expired, falling back to full fetch", history_id
                    )
                    new_history_id = self.get_latest_history_id()
                    return [], new_history_id
                raise

            for record in resp.get("history", []):
                for added in record.get("messagesAdded", []):
                    new_message_ids.append(added["message"]["id"])

            page_token = resp.get("nextPageToken")
            new_history_id = resp.get("historyId", history_id)
            if not page_token:
                break

        if not new_message_ids:
            return [], new_history_id

        # Hydrate new messages into thread dicts
        threads_map: dict[str, list[dict[str, Any]]] = {}
        for msg_id in new_message_ids:
            msg = _api_call_with_backoff(
                svc.users().messages().get(
                    userId="me",
                    id=msg_id,
                    format="full",
                ).execute
            )
            thread_id = msg.get("threadId", msg_id)
            threads_map.setdefault(thread_id, []).append(msg)

        threads = [
            self._hydrate_thread(tid, msgs)
            for tid, msgs in threads_map.items()
        ]
        return threads, new_history_id

    def get_latest_history_id(self) -> str:
        """Return the current mailbox historyId from the user's profile."""
        svc = self._require_service()
        profile = _api_call_with_backoff(
            svc.users().getProfile(userId="me").execute
        )
        return str(profile.get("historyId", ""))


# ---------------------------------------------------------------------------
# Attachment text extraction
# ---------------------------------------------------------------------------


def _extract_attachment_text(data: bytes, filename: str) -> str:
    """Best-effort text extraction from an email attachment.

    Supports PDF (via PyMuPDF) and DOCX (via python-docx).
    Returns empty string for unsupported types or on error.
    """
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".pdf":
            import fitz  # PyMuPDF
            import io
            doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(p.strip() for p in pages if p.strip())

        if suffix == ".docx":
            from docx import Document
            import io
            doc = Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
    except Exception as exc:
        logger.debug("Could not extract text from attachment %s: %s", filename, exc)
    return ""


# ---------------------------------------------------------------------------
# Public pipeline function
# ---------------------------------------------------------------------------


def ingest_gmail(
    ingestor: GmailIngestor,
    metadata_store: Any,
    vector_store: Any,
    since: str | None = None,
    labels: list[str] | None = None,
    max_results: int = 500,
) -> Any:
    """Fetch threads → chunk → embed → store.

    Mirrors the shape of ``ingest_folder()`` in pipeline.py.

    Parameters
    ----------
    ingestor:
        An authenticated GmailIngestor.
    metadata_store:
        MetadataStore instance.
    vector_store:
        VectorStore instance.
    since:
        Only fetch messages after this ISO date string.
    labels:
        Optional Gmail label filter list.
    max_results:
        Cap on messages to retrieve.

    Returns
    -------
    IngestStats
    """
    import time as _time

    from verra.ingest.chunking import chunk_email_thread
    from verra.ingest.email_cleaner import clean_email_body
    from verra.ingest.pipeline import IngestStats

    stats = IngestStats()
    t0 = _time.monotonic()

    source_key = f"gmail:{ingestor.account}"

    # Decide: full fetch or delta
    sync_state = metadata_store.get_sync_state(source_key)
    history_id: str | None = sync_state.get("cursor") if sync_state else None

    if history_id:
        logger.info("Gmail delta fetch from historyId=%s for %s", history_id, ingestor.account)
        metadata_store.upsert_sync_state(
            source=source_key, cursor=history_id,
            items_processed=sync_state.get("items_processed", 0),
            status="syncing",
        )
        try:
            threads, new_history_id = ingestor.delta_fetch(history_id)
        except Exception as exc:
            stats.errors.append(f"delta_fetch failed: {exc}")
            metadata_store.upsert_sync_state(
                source=source_key, cursor=history_id,
                items_processed=sync_state.get("items_processed", 0) if sync_state else 0,
                status="error",
            )
            stats.elapsed_seconds = _time.monotonic() - t0
            return stats
    else:
        logger.info("Gmail full fetch for %s (since=%s)", ingestor.account, since)
        metadata_store.upsert_sync_state(
            source=source_key, cursor=None, items_processed=0, status="syncing",
        )
        try:
            threads = ingestor.fetch_threads(since=since, labels=labels, max_results=max_results)
            new_history_id = ingestor.get_latest_history_id()
        except Exception as exc:
            stats.errors.append(f"fetch_threads failed: {exc}")
            metadata_store.upsert_sync_state(
                source=source_key, cursor=None, items_processed=0, status="error",
            )
            stats.elapsed_seconds = _time.monotonic() - t0
            return stats

    stats.files_found = len(threads)

    for thread in threads:
        thread_id: str = thread["thread_id"]
        messages: list[dict[str, Any]] = thread.get("messages", [])
        if not messages:
            stats.files_skipped += 1
            continue

        try:
            # Use the full thread body as a stable content hash
            combined = thread_id + "".join(m.get("internal_date", "") for m in messages)
            content_hash = hashlib.sha256(combined.encode()).hexdigest()

            # Skip if already indexed (same thread, no new messages)
            existing = metadata_store.get_document_by_hash(content_hash)
            if existing is not None:
                stats.files_skipped += 1
                continue

            # Remove stale document for this thread if re-ingesting
            thread_path = f"gmail:{ingestor.account}:{thread_id}"
            old_doc = metadata_store.get_document_by_path(thread_path)
            if old_doc is not None:
                vector_store.delete_by_document_id(old_doc["id"])
                metadata_store.delete_document(old_doc["id"])

            first_msg = messages[0]
            subject = first_msg.get("subject", "(no subject)")

            # Register the thread as a document
            doc_id = metadata_store.add_document(
                file_path=thread_path,
                file_name=subject[:255],
                source_type="email",
                format="email",
                content_hash=content_hash,
                page_count=len(messages),
                extra_metadata={
                    "thread_id": thread_id,
                    "account": ingestor.account,
                    "message_count": len(messages),
                },
            )

            # Chunk the thread
            # chunk_email_thread expects dicts with: from, to, date, subject, body
            # Clean each message body before chunking
            chunk_inputs = [
                {
                    "from": m.get("from", ""),
                    "to": m.get("to", ""),
                    "cc": m.get("cc", ""),
                    "date": m.get("date", ""),
                    "subject": m.get("subject", ""),
                    "body": clean_email_body(m.get("body", "")),
                    "thread_id": thread_id,
                }
                for m in messages
            ]
            chunks = chunk_email_thread(chunk_inputs)

            # Attach document_id to each chunk's metadata
            for chunk in chunks:
                chunk.metadata["document_id"] = doc_id
                chunk.metadata["account"] = ingestor.account
                chunk.metadata["thread_id"] = thread_id
                chunk.metadata["source_type"] = "email"
                # parent_email_id / attachment_filename are set on attachment
                # chunks only; thread chunks leave these absent.
                chunk.metadata.setdefault("parent_email_id", None)
                chunk.metadata.setdefault("attachment_filename", None)

            chunk_ids = metadata_store.add_chunks(doc_id, chunks)
            vector_store.add_chunks(chunk_ids, chunks)

            # Store per-message email metadata, linking to the first chunk
            first_chunk_id = chunk_ids[0] if chunk_ids else None
            for msg in messages:
                metadata_store.add_email(
                    thread_id=thread_id,
                    message_id=msg.get("message_id"),
                    from_addr=msg.get("from"),
                    to_addr=msg.get("to"),
                    cc_addr=msg.get("cc"),
                    subject=msg.get("subject"),
                    date=msg.get("date"),
                    labels=msg.get("labels", []),
                    chunk_id=first_chunk_id,
                )

            stats.files_processed += 1
            stats.chunks_created += len(chunks)

        except Exception as exc:
            logger.exception("Error ingesting thread %s", thread_id)
            stats.errors.append(f"thread:{thread_id}: {exc}")
            stats.files_skipped += 1

    stats.elapsed_seconds = _time.monotonic() - t0

    # Persist new sync cursor
    metadata_store.upsert_sync_state(
        source=source_key,
        cursor=new_history_id or history_id,
        items_processed=stats.files_processed,
        status="idle",
    )

    return stats
