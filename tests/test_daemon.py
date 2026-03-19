"""Tests for sync/daemon.py — SyncDaemon and _FolderEventHandler.

The daemon itself runs in a background thread so tests use short sleep
windows and mocked stores/ingestors to avoid real filesystem or network I/O.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from verra.config import VerraConfig, SourceConfig, SyncConfig
from verra.store.metadata import MetadataStore
from verra.sync.daemon import SyncDaemon, _FolderEventHandler, _WatchdogBridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    sources: list[SourceConfig] | None = None,
    interval: int = 1,
    enabled: bool = True,
) -> VerraConfig:
    return VerraConfig(
        sources=sources or [],
        sync=SyncConfig(interval=interval, enabled=enabled),
    )


# ---------------------------------------------------------------------------
# Fake watchdog event classes (type(evt).__name__ must match exactly)
# ---------------------------------------------------------------------------


class FileCreatedEvent:
    def __init__(self, src_path: str, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.is_directory = is_directory


class FileModifiedEvent:
    def __init__(self, src_path: str, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.is_directory = is_directory


class FileDeletedEvent:
    def __init__(self, src_path: str, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.is_directory = is_directory


class FileMovedEvent:
    def __init__(self, src_path: str, dest_path: str, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = is_directory


# ---------------------------------------------------------------------------
# _FolderEventHandler tests
# ---------------------------------------------------------------------------


class TestFolderEventHandler:
    def _make_handler(self, folder: Path = Path("/tmp")) -> tuple[_FolderEventHandler, MagicMock]:
        daemon = MagicMock()
        handler = _FolderEventHandler(folder, daemon, ["*.pyc", "__pycache__"])
        return handler, daemon

    def test_file_created_triggers_reingest(self) -> None:
        handler, daemon = self._make_handler()
        handler.dispatch(FileCreatedEvent("/tmp/report.txt"))
        daemon._reingest_file.assert_called_once_with(Path("/tmp/report.txt"))

    def test_file_deleted_triggers_removal(self) -> None:
        handler, daemon = self._make_handler()
        handler.dispatch(FileDeletedEvent("/tmp/report.txt"))
        daemon._remove_file.assert_called_once_with(Path("/tmp/report.txt"))

    def test_directory_events_ignored(self) -> None:
        handler, daemon = self._make_handler()
        handler.dispatch(FileCreatedEvent("/tmp/subdir", is_directory=True))
        daemon._reingest_file.assert_not_called()
        daemon._remove_file.assert_not_called()

    def test_ignored_pattern_skips_reingest(self) -> None:
        handler, daemon = self._make_handler()
        handler.dispatch(FileCreatedEvent("/tmp/module.pyc"))  # matches *.pyc
        daemon._reingest_file.assert_not_called()

    def test_move_event_deletes_old_creates_new(self) -> None:
        handler, daemon = self._make_handler()
        handler.dispatch(FileMovedEvent("/tmp/old.txt", "/tmp/new.txt"))
        daemon._remove_file.assert_called_once_with(Path("/tmp/old.txt"))
        daemon._reingest_file.assert_called_once_with(Path("/tmp/new.txt"))


# ---------------------------------------------------------------------------
# _WatchdogBridge tests
# ---------------------------------------------------------------------------


class TestWatchdogBridge:
    def test_dispatch_delegates_to_inner(self) -> None:
        inner = MagicMock()
        bridge = _WatchdogBridge(inner)
        evt = MagicMock()
        bridge.dispatch(evt)
        inner.dispatch.assert_called_once_with(evt)

    def test_ignore_directories_property(self) -> None:
        bridge = _WatchdogBridge(MagicMock())
        assert bridge.ignore_directories is True


# ---------------------------------------------------------------------------
# SyncDaemon lifecycle tests
# ---------------------------------------------------------------------------


class TestSyncDaemonLifecycle:
    def test_start_and_stop(self, metadata_store: MetadataStore, tmp_path: Path) -> None:
        config = _make_config(interval=1)
        vector_store = MagicMock()
        daemon = SyncDaemon(config, metadata_store, vector_store)

        daemon.start()
        assert daemon._running is True
        assert daemon._thread is not None
        assert daemon._thread.is_alive()

        daemon.stop()
        assert daemon._running is False
        # Thread should have exited within a reasonable timeout
        assert not (daemon._thread and daemon._thread.is_alive())

    def test_double_start_is_safe(self, metadata_store: MetadataStore) -> None:
        config = _make_config(interval=60)
        daemon = SyncDaemon(config, metadata_store, MagicMock())
        daemon.start()
        thread_before = daemon._thread
        daemon.start()  # Should log warning but not crash
        assert daemon._thread is thread_before
        daemon.stop()

    def test_stop_before_start_is_safe(self, metadata_store: MetadataStore) -> None:
        config = _make_config(interval=60)
        daemon = SyncDaemon(config, metadata_store, MagicMock())
        daemon.stop()  # Must not raise


# ---------------------------------------------------------------------------
# SyncDaemon.status() test
# ---------------------------------------------------------------------------


class TestSyncDaemonStatus:
    def test_status_returns_running_flag_and_sources(
        self, metadata_store: MetadataStore
    ) -> None:
        config = _make_config(interval=60)
        daemon = SyncDaemon(config, metadata_store, MagicMock())

        status = daemon.status()
        assert "running" in status
        assert "sources" in status
        assert status["running"] is False

        daemon.start()
        status_running = daemon.status()
        assert status_running["running"] is True
        daemon.stop()

    def test_status_reflects_sync_state_rows(self, metadata_store: MetadataStore) -> None:
        metadata_store.upsert_sync_state("folder:/tmp/docs", "abc", 10, "idle")
        config = _make_config(interval=60)
        daemon = SyncDaemon(config, metadata_store, MagicMock())
        status = daemon.status()
        sources = {s["source"]: s for s in status["sources"]}
        assert "folder:/tmp/docs" in sources


# ---------------------------------------------------------------------------
# SyncDaemon._reingest_file / _remove_file tests
# ---------------------------------------------------------------------------


class TestReingestAndRemove:
    def _make_daemon(
        self, metadata_store: MetadataStore, tmp_path: Path
    ) -> tuple[SyncDaemon, MagicMock]:
        config = _make_config(
            sources=[SourceConfig(type="folder", path=str(tmp_path))],
            interval=60,
        )
        vector_store = MagicMock()
        daemon = SyncDaemon(config, metadata_store, vector_store)
        return daemon, vector_store

    def test_reingest_unsupported_extension_is_noop(
        self, metadata_store: MetadataStore, tmp_path: Path
    ) -> None:
        daemon, vector_store = self._make_daemon(metadata_store, tmp_path)
        unknown_file = tmp_path / "image.png"
        unknown_file.write_bytes(b"\x89PNG")
        daemon._reingest_file(unknown_file)
        vector_store.add_chunks.assert_not_called()

    def test_reingest_txt_file_indexes_document(
        self, metadata_store: MetadataStore, tmp_path: Path
    ) -> None:
        daemon, vector_store = self._make_daemon(metadata_store, tmp_path)
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("These are some important notes about a client meeting.")

        daemon._reingest_file(txt_file)

        docs = metadata_store.list_documents(source_type="folder")
        assert len(docs) == 1
        assert docs[0]["file_name"] == "notes.txt"
        vector_store.add_chunks.assert_called_once()

    def test_reingest_unchanged_file_is_noop(
        self, metadata_store: MetadataStore, tmp_path: Path
    ) -> None:
        daemon, vector_store = self._make_daemon(metadata_store, tmp_path)
        txt_file = tmp_path / "stable.txt"
        txt_file.write_text("Stable content that never changes.")

        # First ingest
        daemon._reingest_file(txt_file)
        call_count_after_first = vector_store.add_chunks.call_count

        # Second ingest — same content hash → should skip
        daemon._reingest_file(txt_file)
        assert vector_store.add_chunks.call_count == call_count_after_first

    def test_reingest_modified_file_replaces_old_document(
        self, metadata_store: MetadataStore, tmp_path: Path
    ) -> None:
        daemon, vector_store = self._make_daemon(metadata_store, tmp_path)
        txt_file = tmp_path / "mutable.txt"
        txt_file.write_text("Original content.")
        daemon._reingest_file(txt_file)

        # Modify content
        txt_file.write_text("Updated content with entirely different text.")
        daemon._reingest_file(txt_file)

        # Still only one document for this path
        docs = metadata_store.list_documents(source_type="folder")
        assert len(docs) == 1
        # vector_store.delete_by_document_id should have been called for the old doc
        vector_store.delete_by_document_id.assert_called_once()

    def test_remove_file_deletes_known_document(
        self, metadata_store: MetadataStore, tmp_path: Path
    ) -> None:
        daemon, vector_store = self._make_daemon(metadata_store, tmp_path)
        txt_file = tmp_path / "to_delete.txt"
        txt_file.write_text("Content to be deleted.")
        daemon._reingest_file(txt_file)

        docs_before = metadata_store.list_documents()
        assert len(docs_before) == 1
        doc_id = docs_before[0]["id"]

        daemon._remove_file(txt_file)

        docs_after = metadata_store.list_documents()
        assert len(docs_after) == 0
        vector_store.delete_by_document_id.assert_called_with(doc_id)

    def test_remove_unknown_file_is_noop(
        self, metadata_store: MetadataStore, tmp_path: Path
    ) -> None:
        daemon, vector_store = self._make_daemon(metadata_store, tmp_path)
        daemon._remove_file(tmp_path / "ghost.txt")  # Never indexed
        vector_store.delete_by_document_id.assert_not_called()


# ---------------------------------------------------------------------------
# Gmail polling integration test
# ---------------------------------------------------------------------------


class TestGmailPolling:
    """Gmail polling patches target the module where the names are imported *from*,
    because _poll_one_gmail_account uses local imports inside the method body."""

    def test_poll_gmail_sources_calls_ingest_gmail(
        self, metadata_store: MetadataStore
    ) -> None:
        config = _make_config(
            sources=[SourceConfig(type="gmail", account="user@gmail.com")],
            interval=60,
        )
        daemon = SyncDaemon(config, metadata_store, MagicMock())

        mock_ingestor_instance = MagicMock()
        mock_ingestor_instance.authenticate.return_value = True

        from verra.ingest.pipeline import IngestStats

        mock_stats = IngestStats(files_processed=3, chunks_created=10)

        with (
            patch("verra.ingest.gmail.GmailIngestor", return_value=mock_ingestor_instance),
            patch("verra.ingest.gmail.ingest_gmail", return_value=mock_stats) as mock_ingest,
        ):
            daemon._poll_gmail_sources()
            mock_ingest.assert_called_once()
            call_kwargs = mock_ingest.call_args
            assert call_kwargs.kwargs["ingestor"] is mock_ingestor_instance
            assert call_kwargs.kwargs["metadata_store"] is metadata_store

    def test_poll_skips_non_gmail_sources(self, metadata_store: MetadataStore) -> None:
        config = _make_config(
            sources=[SourceConfig(type="folder", path="/tmp/docs")],
            interval=60,
        )
        daemon = SyncDaemon(config, metadata_store, MagicMock())

        with patch("verra.ingest.gmail.ingest_gmail") as mock_ingest:
            daemon._poll_gmail_sources()
            mock_ingest.assert_not_called()

    def test_auth_failure_skips_without_crashing(
        self, metadata_store: MetadataStore
    ) -> None:
        config = _make_config(
            sources=[SourceConfig(type="gmail", account="user@gmail.com")],
            interval=60,
        )
        daemon = SyncDaemon(config, metadata_store, MagicMock())

        mock_ingestor_instance = MagicMock()
        mock_ingestor_instance.authenticate.return_value = False  # auth failed

        with (
            patch("verra.ingest.gmail.GmailIngestor", return_value=mock_ingestor_instance),
            patch("verra.ingest.gmail.ingest_gmail") as mock_ingest,
        ):
            daemon._poll_gmail_sources()
            mock_ingest.assert_not_called()
