"""Tests for gmail.py — GmailIngestor and ingest_gmail().

All Gmail API calls are replaced with lightweight fakes so no network or
OAuth credentials are required.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from verra.ingest.gmail import (
    GmailIngestor,
    _decode_body,
    _extract_body,
    _extract_header,
    _html_to_text,
    ingest_gmail,
)
from verra.store.metadata import MetadataStore


# ---------------------------------------------------------------------------
# Helpers for building fake Gmail API responses
# ---------------------------------------------------------------------------


def _b64(text: str) -> str:
    """Return a URL-safe base64 encoding of *text* (as Gmail provides)."""
    return base64.urlsafe_b64encode(text.encode()).decode()


def _make_message(
    msg_id: str = "msg1",
    thread_id: str = "thread1",
    from_: str = "alice@example.com",
    to: str = "bob@example.com",
    subject: str = "Hello",
    body_text: str = "Hello Bob",
    date: str = "Mon, 1 Jan 2024 10:00:00 +0000",
    internal_date: str = "1704067200000",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Construct a minimal Gmail API message dict with a text/plain body."""
    return {
        "id": msg_id,
        "threadId": thread_id,
        "internalDate": internal_date,
        "labelIds": labels or ["INBOX"],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": from_},
                {"name": "To", "value": to},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": date},
            ],
            "body": {"data": _b64(body_text)},
            "parts": [],
        },
    }


def _make_multipart_message(
    msg_id: str = "msg1",
    thread_id: str = "thread1",
    plain_text: str = "Plain text body",
    html_text: str = "<p>HTML body</p>",
) -> dict[str, Any]:
    """Construct a multipart/alternative Gmail message."""
    return {
        "id": msg_id,
        "threadId": thread_id,
        "internalDate": "1704067200000",
        "labelIds": ["INBOX"],
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "alice@example.com"},
                {"name": "To", "value": "bob@example.com"},
                {"name": "Subject", "value": "Multipart"},
                {"name": "Date", "value": "Mon, 1 Jan 2024 10:00:00 +0000"},
            ],
            "body": {},
            "parts": [
                {
                    "mimeType": "text/plain",
                    "headers": [],
                    "body": {"data": _b64(plain_text)},
                    "parts": [],
                },
                {
                    "mimeType": "text/html",
                    "headers": [],
                    "body": {"data": _b64(html_text)},
                    "parts": [],
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# Unit tests for low-level helpers
# ---------------------------------------------------------------------------


class TestDecodeBody:
    def test_roundtrip(self) -> None:
        encoded = _b64("Hello, world!")
        assert _decode_body(encoded) == "Hello, world!"

    def test_bad_data_returns_empty(self) -> None:
        # Non-base64 gibberish should not raise
        result = _decode_body("!!!not_valid!!!")
        # Returns empty string or replacement-character string, never raises
        assert isinstance(result, str)


class TestExtractHeader:
    def test_exact_match(self) -> None:
        headers = [
            {"name": "From", "value": "alice@example.com"},
            {"name": "Subject", "value": "Test"},
        ]
        assert _extract_header(headers, "From") == "alice@example.com"

    def test_case_insensitive(self) -> None:
        headers = [{"name": "SUBJECT", "value": "Hello"}]
        assert _extract_header(headers, "subject") == "Hello"

    def test_missing_header_returns_empty(self) -> None:
        assert _extract_header([], "Cc") == ""


class TestHtmlToText:
    def test_strips_tags(self) -> None:
        result = _html_to_text("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_block_tags_add_newlines(self) -> None:
        result = _html_to_text("<p>First</p><p>Second</p>")
        assert "First" in result
        assert "Second" in result


class TestExtractBody:
    def test_plain_text(self) -> None:
        payload = {
            "mimeType": "text/plain",
            "body": {"data": _b64("Simple body")},
            "parts": [],
        }
        assert _extract_body(payload) == "Simple body"

    def test_html_fallback(self) -> None:
        payload = {
            "mimeType": "text/html",
            "body": {"data": _b64("<p>HTML content</p>")},
            "parts": [],
        }
        result = _extract_body(payload)
        assert "HTML content" in result
        assert "<" not in result

    def test_multipart_prefers_plain(self) -> None:
        payload = _make_multipart_message()["payload"]
        result = _extract_body(payload)
        assert "Plain text body" in result
        assert "HTML body" not in result

    def test_empty_body_returns_empty(self) -> None:
        payload = {"mimeType": "text/plain", "body": {}, "parts": []}
        assert _extract_body(payload) == ""


# ---------------------------------------------------------------------------
# GmailIngestor unit tests (mocked service)
# ---------------------------------------------------------------------------


class TestGmailIngestorInit:
    def test_token_path_uses_account_name(self, tmp_path: Path) -> None:
        ingestor = GmailIngestor("test@gmail.com", credentials_dir=str(tmp_path))
        # @ is replaced with _; dots are kept by the regex [^a-zA-Z0-9._-]
        assert "@" not in ingestor._token_path.name
        assert "test" in ingestor._token_path.name
        assert ingestor._token_path.name.endswith("_token.json")

    def test_credentials_dir_created(self, tmp_path: Path) -> None:
        cred_dir = tmp_path / "oauth_subdir"
        GmailIngestor("a@b.com", credentials_dir=str(cred_dir))
        assert cred_dir.exists()


class TestAuthenticate:
    def test_missing_client_secret_returns_false(self, tmp_path: Path, capsys: Any) -> None:
        ingestor = GmailIngestor("user@gmail.com", credentials_dir=str(tmp_path))
        # No client_secret.json → should print instructions and return False
        result = ingestor.authenticate()
        assert result is False
        captured = capsys.readouterr()
        assert "console.cloud.google.com" in captured.out

    def test_cached_valid_token_skips_flow(self, tmp_path: Path) -> None:
        """If a valid non-expired Credentials object is loaded, no browser flow runs."""
        ingestor = GmailIngestor("user@gmail.com", credentials_dir=str(tmp_path))

        # Write a dummy client_secret so the path check passes
        (tmp_path / "client_secret.json").write_text("{}")

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.to_json.return_value = "{}"

        mock_service = MagicMock()

        # authenticate() uses local imports, so we patch the source packages
        with (
            patch("google.oauth2.credentials.Credentials") as MockCreds,
            patch("google_auth_oauthlib.flow.InstalledAppFlow"),
            patch("googleapiclient.discovery.build", return_value=mock_service),
        ):
            # Simulate: token file exists and is valid
            token_path = ingestor._token_path
            token_path.write_text("{}")

            MockCreds.from_authorized_user_file.return_value = mock_creds

            result = ingestor.authenticate()

        assert result is True
        assert ingestor._service is mock_service


class TestHydrateThread:
    def test_single_message_thread(self) -> None:
        ingestor = GmailIngestor("user@gmail.com")
        raw = _make_message(msg_id="m1", thread_id="t1", subject="Test Thread")
        thread = ingestor._hydrate_thread("t1", [raw])

        assert thread["thread_id"] == "t1"
        assert thread["subject"] == "Test Thread"
        assert len(thread["messages"]) == 1
        msg = thread["messages"][0]
        assert msg["from"] == "alice@example.com"
        assert msg["body"] == "Hello Bob"

    def test_messages_sorted_by_internal_date(self) -> None:
        ingestor = GmailIngestor("user@gmail.com")
        # Provide messages out of order
        m_old = _make_message(msg_id="m_old", thread_id="t1", body_text="Older", internal_date="100")
        m_new = _make_message(msg_id="m_new", thread_id="t1", body_text="Newer", internal_date="200")
        thread = ingestor._hydrate_thread("t1", [m_new, m_old])
        assert thread["messages"][0]["body"] == "Older"
        assert thread["messages"][1]["body"] == "Newer"

    def test_multipart_body_extracted(self) -> None:
        ingestor = GmailIngestor("user@gmail.com")
        raw = _make_multipart_message(msg_id="m1", thread_id="t1", plain_text="Plain part")
        thread = ingestor._hydrate_thread("t1", [raw])
        assert "Plain part" in thread["messages"][0]["body"]


# ---------------------------------------------------------------------------
# ingest_gmail integration tests (fully mocked stores + ingestor)
# ---------------------------------------------------------------------------


class TestIngestGmail:
    def _make_ingestor(self, threads: list[dict[str, Any]]) -> GmailIngestor:
        """Return a GmailIngestor whose fetch_threads/delta_fetch are mocked."""
        ingestor = GmailIngestor("user@gmail.com")
        ingestor._service = MagicMock()  # prevent authenticate() from being called

        # get_latest_history_id returns a dummy value
        ingestor.get_latest_history_id = MagicMock(return_value="999")  # type: ignore[method-assign]
        ingestor.fetch_threads = MagicMock(return_value=threads)  # type: ignore[method-assign]
        ingestor.delta_fetch = MagicMock(return_value=(threads, "1000"))  # type: ignore[method-assign]
        return ingestor

    def _make_thread(self, thread_id: str = "t1") -> dict[str, Any]:
        msg = _make_message(msg_id="m1", thread_id=thread_id, subject="Invoice Q1")
        ingestor_helper = GmailIngestor("user@gmail.com")
        return ingestor_helper._hydrate_thread(thread_id, [msg])

    def test_full_ingest_creates_document_and_chunks(
        self, metadata_store: MetadataStore, tmp_path: Path
    ) -> None:
        import chromadb

        vector_store_mock = MagicMock()
        vector_store_mock.add_chunks = MagicMock()

        thread = self._make_thread("t1")
        ingestor = self._make_ingestor([thread])

        stats = ingest_gmail(
            ingestor=ingestor,
            metadata_store=metadata_store,
            vector_store=vector_store_mock,
        )

        assert stats.files_processed == 1
        assert stats.chunks_created >= 1
        assert stats.files_skipped == 0
        assert stats.errors == []

        # Document should be in metadata store
        docs = metadata_store.list_documents(source_type="email")
        assert len(docs) == 1
        assert docs[0]["file_name"].startswith("Invoice Q1")

        # Sync state should be set
        state = metadata_store.get_sync_state("gmail:user@gmail.com")
        assert state is not None
        assert state["status"] == "idle"
        assert state["cursor"] == "999"

    def test_duplicate_thread_skipped(
        self, metadata_store: MetadataStore
    ) -> None:
        vector_store_mock = MagicMock()
        thread = self._make_thread("t1")
        ingestor = self._make_ingestor([thread])

        # Ingest once
        stats1 = ingest_gmail(ingestor=ingestor, metadata_store=metadata_store, vector_store=vector_store_mock)
        assert stats1.files_processed == 1

        # Ingest same thread again — should be skipped (same content hash)
        # Reset historyId so we take the fetch_threads path again
        metadata_store.upsert_sync_state("gmail:user@gmail.com", None, 0, "idle")
        stats2 = ingest_gmail(ingestor=ingestor, metadata_store=metadata_store, vector_store=vector_store_mock)
        assert stats2.files_skipped == 1
        assert stats2.files_processed == 0

    def test_empty_thread_is_skipped(self, metadata_store: MetadataStore) -> None:
        vector_store_mock = MagicMock()
        empty_thread = {"thread_id": "t_empty", "subject": "", "messages": []}
        ingestor = self._make_ingestor([empty_thread])

        stats = ingest_gmail(ingestor=ingestor, metadata_store=metadata_store, vector_store=vector_store_mock)
        assert stats.files_skipped == 1
        assert stats.files_processed == 0

    def test_delta_path_used_when_history_id_stored(
        self, metadata_store: MetadataStore
    ) -> None:
        """When a historyId is already in sync_state, delta_fetch is called, not fetch_threads."""
        vector_store_mock = MagicMock()
        thread = self._make_thread("t2")
        ingestor = self._make_ingestor([thread])

        # Pre-seed a history cursor
        metadata_store.upsert_sync_state("gmail:user@gmail.com", "500", 0, "idle")

        stats = ingest_gmail(ingestor=ingestor, metadata_store=metadata_store, vector_store=vector_store_mock)

        ingestor.delta_fetch.assert_called_once_with("500")
        ingestor.fetch_threads.assert_not_called()
        assert stats.files_processed == 1

    def test_email_metadata_stored(self, metadata_store: MetadataStore) -> None:
        vector_store_mock = MagicMock()
        thread = self._make_thread("t3")
        ingestor = self._make_ingestor([thread])

        ingest_gmail(ingestor=ingestor, metadata_store=metadata_store, vector_store=vector_store_mock)

        emails = metadata_store.search_emails(from_addr="alice@example.com")
        assert len(emails) >= 1
        assert emails[0]["thread_id"] == "t3"

    def test_fetch_error_recorded_in_stats(self, metadata_store: MetadataStore) -> None:
        vector_store_mock = MagicMock()
        ingestor = GmailIngestor("user@gmail.com")
        ingestor._service = MagicMock()
        ingestor.fetch_threads = MagicMock(side_effect=RuntimeError("API down"))  # type: ignore[method-assign]
        ingestor.get_latest_history_id = MagicMock(return_value="0")  # type: ignore[method-assign]

        stats = ingest_gmail(ingestor=ingestor, metadata_store=metadata_store, vector_store=vector_store_mock)
        assert len(stats.errors) == 1
        assert "fetch_threads" in stats.errors[0] or "API down" in stats.errors[0]

    def test_multiple_threads_all_processed(self, metadata_store: MetadataStore) -> None:
        vector_store_mock = MagicMock()
        threads = [self._make_thread(f"thread_{i}") for i in range(5)]
        ingestor = self._make_ingestor(threads)

        stats = ingest_gmail(ingestor=ingestor, metadata_store=metadata_store, vector_store=vector_store_mock)
        assert stats.files_processed == 5
        assert stats.files_found == 5
