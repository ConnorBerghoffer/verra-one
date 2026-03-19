"""Background sync daemon.

Runs two independent sync strategies in a single background thread:

1. Folder watching (watchdog)
   - One Observer per configured folder source.
   - File created/modified  → re-ingest that file.
   - File deleted           → remove document + vector chunks.

2. Periodic polling (Gmail, Drive, Outlook)
   - Every config.sync.interval seconds (default 300 s / 5 min).
   - Gmail uses delta_fetch() with the stored historyId so only new mail is
     fetched.
   - Drive uses the Changes API page token for delta sync.
   - Outlook uses the since parameter to fetch only new messages.

Usage:
    daemon = SyncDaemon(config, metadata_store, vector_store)
    daemon.start()
    ...
    daemon.stop()
"""


from __future__ import annotations

import fnmatch
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Folder event handler
# ---------------------------------------------------------------------------


class _FolderEventHandler:
    """Handles watchdog filesystem events for a single watched folder.

    Constructed with a reference to the daemon so it can call back into the
    re-ingest helpers.
    """

    def __init__(
        self,
        folder_path: Path,
        daemon: "SyncDaemon",
        ignore_patterns: list[str],
    ) -> None:
        self._folder = folder_path
        self._daemon = daemon
        self._ignore_patterns = ignore_patterns

    # watchdog calls dispatch() which routes to on_created / on_modified / on_deleted
    def dispatch(self, event: Any) -> None:
        """Route watchdog events to the appropriate handler."""
        if event.is_directory:
            return
        etype = type(event).__name__
        if etype in ("FileCreatedEvent", "FileModifiedEvent"):
            self._on_file_changed(Path(event.src_path))
        elif etype == "FileMovedEvent":
            # Treat a move as: delete old, create new
            self._on_file_deleted(Path(event.src_path))
            self._on_file_changed(Path(event.dest_path))
        elif etype == "FileDeletedEvent":
            self._on_file_deleted(Path(event.src_path))

    def _is_ignored(self, path: Path) -> bool:
        name = path.name
        for pattern in self._ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _on_file_changed(self, path: Path) -> None:
        if self._is_ignored(path):
            return
        logger.info("[daemon] file changed: %s — re-ingesting", path)
        self._daemon._reingest_file(path)

    def _on_file_deleted(self, path: Path) -> None:
        if self._is_ignored(path):
            return
        logger.info("[daemon] file deleted: %s — removing from index", path)
        self._daemon._remove_file(path)


# ---------------------------------------------------------------------------
# SyncDaemon
# ---------------------------------------------------------------------------


class SyncDaemon:
    """Background sync daemon for folder watching and Gmail/Drive/Outlook polling.

    Parameters
    ----------
    config:
        VerraConfig — provides source list and sync.interval.
    metadata_store:
        MetadataStore instance (must be thread-safe; SQLite WAL mode is).
    vector_store:
        VectorStore instance.
    """

    def __init__(
        self,
        config: Any,
        metadata_store: Any,
        vector_store: Any,
    ) -> None:
        self._config = config
        self._metadata_store = metadata_store
        self._vector_store = vector_store

        self._running = False
        self._thread: threading.Thread | None = None

        # Serialise all metadata/vector write operations so that concurrent
        # watchdog events cannot interleave SQLite writes.
        self._write_lock = threading.Lock()

        # Track active watchdog Observers so we can stop them cleanly
        self._observers: list[Any] = []

        # Default ignore patterns — mirrors folder.py defaults
        self._ignore_patterns: list[str] = [
            ".git",
            "__pycache__",
            "*.pyc",
            "node_modules",
            ".DS_Store",
            "*.egg-info",
            ".venv",
            "venv",
        ]

    # ------------------------------------------------------------------
    # Public control interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sync thread (non-blocking)."""
        if self._running:
            logger.warning("[daemon] already running")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._sync_loop,
            name="verra-sync-daemon",
            daemon=True,  # dies with the main process automatically
        )
        self._thread.start()
        logger.info("[daemon] started (interval=%ds)", self._config.sync.interval)

    def stop(self) -> None:
        """Signal the daemon to stop and wait for the thread to exit."""
        self._running = False
        # Stop all watchdog observers
        for observer in self._observers:
            try:
                observer.stop()
            except Exception:
                pass
        for observer in self._observers:
            try:
                observer.join(timeout=5)
            except Exception:
                pass
        self._observers.clear()

        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        logger.info("[daemon] stopped")

    def status(self) -> dict[str, Any]:
        """Return current sync status for all known sources."""
        states = self._metadata_store.list_sync_states()
        return {
            "running": self._running,
            "sources": states,
        }

    # ------------------------------------------------------------------
    # Sync loop
    # ------------------------------------------------------------------

    def _sync_loop(self) -> None:
        """Main loop: start folder watchers, then poll remote sources periodically."""
        self._start_folder_watchers()

        interval = self._config.sync.interval
        last_poll = 0.0

        while self._running:
            now = time.monotonic()

            # Poll all remote sources (Gmail, Drive, Outlook) every interval seconds
            if now - last_poll >= interval:
                self._poll_gmail_sources()
                self._poll_drive_sources()
                self._poll_outlook_sources()
                last_poll = time.monotonic()

            # Sleep in short increments so stop() is responsive
            time.sleep(min(5, interval))

    # ------------------------------------------------------------------
    # Folder watching
    # ------------------------------------------------------------------

    def _start_folder_watchers(self) -> None:
        """Create a watchdog Observer for each configured folder source."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler  # noqa: F401
        except ImportError:
            logger.warning("[daemon] watchdog not installed — folder watching disabled")
            return

        for source in self._config.sources:
            if source.type != "folder" or not source.path:
                continue

            folder = Path(source.path)
            if not folder.exists():
                logger.warning("[daemon] folder source does not exist: %s", folder)
                continue

            # Read .verraignore if present
            patterns = list(self._ignore_patterns)
            ignore_file = folder / ".verraignore"
            if ignore_file.exists():
                for line in ignore_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)

            handler = _WatchdogBridge(
                _FolderEventHandler(folder, self, patterns)
            )
            observer = Observer()
            observer.schedule(handler, str(folder), recursive=True)
            observer.start()
            self._observers.append(observer)
            logger.info("[daemon] watching folder: %s", folder)

    # ------------------------------------------------------------------
    # Gmail polling
    # ------------------------------------------------------------------

    def _poll_gmail_sources(self) -> None:
        """Run a delta fetch for each configured Gmail source."""
        for source in self._config.sources:
            if source.type != "gmail" or not source.account:
                continue
            try:
                self._poll_one_gmail_account(source)
            except Exception as exc:
                logger.exception("[daemon] Gmail poll error for %s: %s", source.account, exc)

    def _poll_one_gmail_account(self, source: Any) -> None:
        """Poll a single Gmail account using delta_fetch if a historyId is stored."""
        try:
            from verra.ingest.gmail import GmailIngestor, ingest_gmail
        except ImportError as exc:
            logger.error("[daemon] gmail module unavailable: %s", exc)
            return

        account = source.account
        ingestor = GmailIngestor(account=account)

        # authenticate() will silently re-use the cached token
        if not ingestor.authenticate():
            logger.warning("[daemon] Gmail auth failed for %s — skipping", account)
            return

        logger.info("[daemon] polling Gmail for %s", account)
        stats = ingest_gmail(
            ingestor=ingestor,
            metadata_store=self._metadata_store,
            vector_store=self._vector_store,
            since=source.since,
            labels=source.labels if source.labels else None,
        )
        logger.info(
            "[daemon] Gmail poll for %s: %d threads processed, %d chunks, %d skipped, %d errors",
            account,
            stats.files_processed,
            stats.chunks_created,
            stats.files_skipped,
            len(stats.errors),
        )
        for err in stats.errors:
            logger.warning("[daemon] Gmail error: %s", err)

    # ------------------------------------------------------------------
    # Drive polling
    # ------------------------------------------------------------------

    def _poll_drive_sources(self) -> None:
        """Run a delta sync for each configured Drive source."""
        for source in self._config.sources:
            if source.type != "drive" or not source.account:
                continue
            try:
                self._poll_one_drive_account(source)
            except Exception as exc:
                logger.exception("[daemon] Drive poll error for %s: %s", source.account, exc)

    def _poll_one_drive_account(self, source: Any) -> None:
        """Poll a single Google Drive account using the Changes API for delta sync."""
        try:
            from verra.ingest.drive import DriveIngestor, ingest_drive
        except ImportError as exc:
            logger.error("[daemon] drive module unavailable: %s", exc)
            return

        account = source.account
        ingestor = DriveIngestor(account=account)

        if not ingestor.authenticate():
            logger.warning("[daemon] Drive auth failed for %s — skipping", account)
            return

        logger.info("[daemon] polling Drive for %s", account)
        # folder_id comes from source.path when a Drive folder URL/ID is configured
        folder_id: str | None = getattr(source, "folder_id", None) or source.path or None
        stats = ingest_drive(
            ingestor=ingestor,
            metadata_store=self._metadata_store,
            vector_store=self._vector_store,
            folder_id=folder_id,
        )
        logger.info(
            "[daemon] Drive poll for %s: %d files processed, %d chunks, %d skipped, %d errors",
            account,
            stats.files_processed,
            stats.chunks_created,
            stats.files_skipped,
            len(stats.errors),
        )
        for err in stats.errors:
            logger.warning("[daemon] Drive error: %s", err)

    # ------------------------------------------------------------------
    # Outlook polling
    # ------------------------------------------------------------------

    def _poll_outlook_sources(self) -> None:
        """Run a delta fetch for each configured Outlook source."""
        for source in self._config.sources:
            if source.type != "outlook" or not source.account:
                continue
            try:
                self._poll_one_outlook_account(source)
            except Exception as exc:
                logger.exception("[daemon] Outlook poll error for %s: %s", source.account, exc)

    def _poll_one_outlook_account(self, source: Any) -> None:
        """Poll a single Outlook/365 account, fetching only messages since last sync."""
        try:
            from verra.ingest.outlook import OutlookIngestor, ingest_outlook
        except ImportError as exc:
            logger.error("[daemon] outlook module unavailable: %s", exc)
            return

        account = source.account
        client_id: str | None = getattr(source, "client_id", None)
        ingestor = OutlookIngestor(account=account, client_id=client_id)

        if not ingestor.authenticate():
            logger.warning("[daemon] Outlook auth failed for %s — skipping", account)
            return

        logger.info("[daemon] polling Outlook for %s", account)
        stats = ingest_outlook(
            ingestor=ingestor,
            metadata_store=self._metadata_store,
            vector_store=self._vector_store,
            since=source.since,
        )
        logger.info(
            "[daemon] Outlook poll for %s: %d messages processed, %d chunks, %d skipped, %d errors",
            account,
            stats.files_processed,
            stats.chunks_created,
            stats.files_skipped,
            len(stats.errors),
        )
        for err in stats.errors:
            logger.warning("[daemon] Outlook error: %s", err)

    # ------------------------------------------------------------------
    # File-level re-ingest / removal helpers
    # ------------------------------------------------------------------

    def _reingest_file(self, path: Path) -> None:
        """Re-ingest a single file that was created or modified.

        The write lock ensures that concurrent watchdog events (e.g. rapid
        save + rename) do not interleave their SQLite writes and produce
        duplicate or orphaned rows.
        """
        from verra.ingest.extractors import SUPPORTED_EXTENSIONS, detect_and_extract
        from verra.ingest.chunking import chunk_document
        import hashlib
        import time as _time

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        try:
            doc = detect_and_extract(path)
        except Exception as exc:
            logger.warning("[daemon] extract failed for %s: %s", path, exc)
            return

        content_hash = hashlib.sha256(doc.content.encode()).hexdigest()

        with self._write_lock:
            # Skip if unchanged — check inside the lock so two concurrent
            # events for the same file don't both proceed past this gate.
            existing = self._metadata_store.get_document_by_hash(content_hash)
            if existing is not None:
                return

            # Remove old version if present (delete-then-insert within one lock hold)
            old = self._metadata_store.get_document_by_path(str(path))
            if old is not None:
                self._vector_store.delete_by_document_id(old["id"])
                self._metadata_store.delete_document(old["id"])

            doc_id = self._metadata_store.add_document(
                file_path=str(path),
                file_name=path.name,
                source_type="folder",
                format=doc.format,
                content_hash=content_hash,
                page_count=doc.page_count,
                extra_metadata=doc.metadata,
            )

            chunks = chunk_document(
                doc.content,
                metadata={
                    "document_id": doc_id,
                    "file_name": path.name,
                    "file_path": str(path),
                    "format": doc.format,
                    "source_type": "folder",
                },
            )
            chunk_ids = self._metadata_store.add_chunks(doc_id, chunks)
            self._vector_store.add_chunks(chunk_ids, chunks)

            # Update sync cursor for the containing folder source
            for source in self._config.sources:
                if source.type == "folder" and source.path:
                    folder = Path(source.path).resolve()
                    if str(path).startswith(str(folder)):
                        self._metadata_store.upsert_sync_state(
                            source=f"folder:{folder}",
                            cursor=str(_time.time()),
                            items_processed=1,
                            status="idle",
                        )
                        break

        logger.info("[daemon] re-indexed %s (%d chunks)", path.name, len(chunks))

    def _remove_file(self, path: Path) -> None:
        """Remove index entries for a deleted file.

        Held under the write lock so a concurrent re-ingest of the same path
        (e.g. rapid delete + recreate) cannot observe a partially removed state.
        """
        with self._write_lock:
            old = self._metadata_store.get_document_by_path(str(path))
            if old is None:
                return
            self._vector_store.delete_by_document_id(old["id"])
            self._metadata_store.delete_document(old["id"])
        logger.info("[daemon] removed %s from index", path.name)


# ---------------------------------------------------------------------------
# Watchdog bridge
# ---------------------------------------------------------------------------


class _WatchdogBridge:
    """Thin shim that adapts our _FolderEventHandler to the watchdog API.

    watchdog's Observer calls ``on_any_event`` on registered handlers.
    We route all events through our own dispatch() method so that we don't
    need to inherit from watchdog's FileSystemEventHandler (avoids the hard
    import at module level when watchdog might not be installed).
    """

    def __init__(self, inner: _FolderEventHandler) -> None:
        self._inner = inner

    # watchdog calls this for every event
    def dispatch(self, event: Any) -> None:
        self._inner.dispatch(event)

    # watchdog expects these attributes for scheduling
    @property
    def patterns(self) -> list[str] | None:
        return None

    @property
    def ignore_patterns(self) -> list[str] | None:
        return None

    @property
    def ignore_directories(self) -> bool:
        return True

    @property
    def case_sensitive(self) -> bool:
        return True
