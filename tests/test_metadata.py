"""Tests for SQLite metadata store."""

from __future__ import annotations

from pathlib import Path

import pytest

from verra.ingest.chunking import Chunk
from verra.store.metadata import MetadataStore


class TestDocuments:
    def test_add_and_retrieve_by_hash(self, metadata_store: MetadataStore) -> None:
        doc_id = metadata_store.add_document(
            file_path="/data/test.txt",
            file_name="test.txt",
            source_type="folder",
            format="txt",
            content_hash="abc123",
            page_count=1,
        )
        assert doc_id > 0
        doc = metadata_store.get_document_by_hash("abc123")
        assert doc is not None
        assert doc["file_name"] == "test.txt"

    def test_retrieve_by_path(self, metadata_store: MetadataStore) -> None:
        metadata_store.add_document(
            file_path="/data/doc.md",
            file_name="doc.md",
            source_type="folder",
            format="md",
            content_hash="xyz789",
        )
        doc = metadata_store.get_document_by_path("/data/doc.md")
        assert doc is not None
        assert doc["content_hash"] == "xyz789"

    def test_missing_hash_returns_none(self, metadata_store: MetadataStore) -> None:
        result = metadata_store.get_document_by_hash("nonexistent")
        assert result is None

    def test_delete_document(self, metadata_store: MetadataStore) -> None:
        doc_id = metadata_store.add_document(
            file_path="/tmp/del.txt",
            file_name="del.txt",
            source_type="folder",
            format="txt",
            content_hash="del_hash",
        )
        metadata_store.delete_document(doc_id)
        assert metadata_store.get_document_by_hash("del_hash") is None

    def test_list_documents_by_source_type(self, metadata_store: MetadataStore) -> None:
        metadata_store.add_document(
            file_path="/folder/a.txt",
            file_name="a.txt",
            source_type="folder",
            format="txt",
            content_hash="hash_a",
        )
        metadata_store.add_document(
            file_path="/email/b",
            file_name="b",
            source_type="email",
            format="email",
            content_hash="hash_b",
        )
        folder_docs = metadata_store.list_documents(source_type="folder")
        assert len(folder_docs) == 1
        assert folder_docs[0]["file_name"] == "a.txt"

    def test_extra_metadata_stored_as_json(self, metadata_store: MetadataStore) -> None:
        metadata_store.add_document(
            file_path="/data/pdf.pdf",
            file_name="pdf.pdf",
            source_type="folder",
            format="pdf",
            content_hash="pdf_hash",
            extra_metadata={"author": "Alice", "pages": 10},
        )
        doc = metadata_store.get_document_by_hash("pdf_hash")
        assert doc is not None
        import json
        extra = json.loads(doc["extra_metadata"])
        assert extra["author"] == "Alice"


class TestChunks:
    def test_add_and_retrieve_chunks(self, metadata_store: MetadataStore) -> None:
        doc_id = metadata_store.add_document(
            file_path="/data/test.txt",
            file_name="test.txt",
            source_type="folder",
            format="txt",
            content_hash="chunk_test",
        )
        chunks = [
            Chunk(text="First chunk of content.", metadata={"doc": "test"}),
            Chunk(text="Second chunk of content.", metadata={"doc": "test"}),
        ]
        ids = metadata_store.add_chunks(doc_id, chunks)
        assert len(ids) == 2
        assert all(i > 0 for i in ids)

        stored = metadata_store.get_chunks_for_document(doc_id)
        assert len(stored) == 2
        assert stored[0]["position"] == 0
        assert stored[1]["position"] == 1

    def test_get_chunk_by_id(self, metadata_store: MetadataStore) -> None:
        doc_id = metadata_store.add_document(
            file_path="/x.txt",
            file_name="x.txt",
            source_type="folder",
            format="txt",
            content_hash="get_chunk_hash",
        )
        ids = metadata_store.add_chunks(doc_id, [Chunk(text="Hello", metadata={})])
        chunk = metadata_store.get_chunk(ids[0])
        assert chunk is not None
        assert chunk["token_count"] > 0

    def test_cascade_delete_removes_chunks(self, metadata_store: MetadataStore) -> None:
        doc_id = metadata_store.add_document(
            file_path="/del2.txt",
            file_name="del2.txt",
            source_type="folder",
            format="txt",
            content_hash="cascade_hash",
        )
        ids = metadata_store.add_chunks(doc_id, [Chunk(text="data", metadata={})])
        metadata_store.delete_document(doc_id)
        chunk = metadata_store.get_chunk(ids[0])
        assert chunk is None


class TestEmails:
    def test_add_and_search_email(self, metadata_store: MetadataStore) -> None:
        metadata_store.add_email(
            thread_id="t001",
            message_id="m001",
            from_addr="jake@acme.com",
            to_addr="me@company.com",
            cc_addr=None,
            subject="Proposal for Q2",
            date="2024-01-10",
            labels=["inbox"],
        )
        results = metadata_store.search_emails(from_addr="jake")
        assert len(results) == 1
        assert results[0]["subject"] == "Proposal for Q2"

    def test_search_by_subject(self, metadata_store: MetadataStore) -> None:
        metadata_store.add_email(
            thread_id="t002",
            message_id="m002",
            from_addr="alice@corp.com",
            to_addr="me@company.com",
            cc_addr=None,
            subject="Contract renewal terms",
            date="2024-01-11",
        )
        results = metadata_store.search_emails(subject_contains="renewal")
        assert len(results) >= 1
        assert "renewal" in results[0]["subject"].lower()

    def test_no_results_returns_empty(self, metadata_store: MetadataStore) -> None:
        results = metadata_store.search_emails(from_addr="nobody@nowhere.com")
        assert results == []


class TestSyncState:
    def test_upsert_and_retrieve(self, metadata_store: MetadataStore) -> None:
        metadata_store.upsert_sync_state(
            source="folder:/Users/test/docs",
            cursor="1234567890",
            items_processed=42,
            status="idle",
        )
        state = metadata_store.get_sync_state("folder:/Users/test/docs")
        assert state is not None
        assert state["items_processed"] == 42
        assert state["status"] == "idle"

    def test_upsert_updates_existing(self, metadata_store: MetadataStore) -> None:
        source = "gmail:test@example.com"
        metadata_store.upsert_sync_state(source=source, cursor="100", items_processed=10, status="idle")
        metadata_store.upsert_sync_state(source=source, cursor="200", items_processed=20, status="syncing")
        state = metadata_store.get_sync_state(source)
        assert state["cursor"] == "200"
        assert state["items_processed"] == 20

    def test_list_sync_states(self, metadata_store: MetadataStore) -> None:
        metadata_store.upsert_sync_state("src_a", "a", 1)
        metadata_store.upsert_sync_state("src_b", "b", 2)
        states = metadata_store.list_sync_states()
        sources = {s["source"] for s in states}
        assert "src_a" in sources
        assert "src_b" in sources
