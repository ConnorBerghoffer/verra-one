"""Tests for the ChromaDB vector store wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from verra.ingest.chunking import Chunk
from verra.store.vector import VectorStore


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    """A fresh VectorStore backed by a temporary directory."""
    vs = VectorStore(tmp_path / "chroma")
    yield vs
    vs.reset()


class TestVectorStore:
    def test_initial_count_is_zero(self, vector_store: VectorStore) -> None:
        assert vector_store.count() == 0

    def test_add_chunks_increases_count(self, vector_store: VectorStore) -> None:
        chunks = [
            Chunk(text="The refund policy allows 30-day returns.", metadata={"source_type": "folder"}),
            Chunk(text="Contact support at support@example.com.", metadata={"source_type": "folder"}),
        ]
        vector_store.add_chunks([1, 2], chunks)
        assert vector_store.count() == 2

    def test_search_returns_results(self, vector_store: VectorStore) -> None:
        chunks = [
            Chunk(text="The refund policy allows full returns within 30 days.", metadata={"source_type": "folder", "file_name": "policy.md"}),
            Chunk(text="Our pricing starts at $29 per month for the Starter plan.", metadata={"source_type": "folder", "file_name": "pricing.md"}),
            Chunk(text="Contact the support team for billing issues.", metadata={"source_type": "folder", "file_name": "support.md"}),
        ]
        vector_store.add_chunks([10, 11, 12], chunks)

        results = vector_store.search("refund policy", n_results=2)
        assert len(results) >= 1
        # The refund chunk should score well
        top_texts = [r["document"] for r in results]
        assert any("refund" in t.lower() for t in top_texts)

    def test_search_result_fields(self, vector_store: VectorStore) -> None:
        chunks = [Chunk(text="Sample text about invoices.", metadata={"source_type": "folder"})]
        vector_store.add_chunks([99], chunks)
        results = vector_store.search("invoice", n_results=1)
        assert len(results) == 1
        result = results[0]
        assert "id" in result
        assert "document" in result
        assert "metadata" in result
        assert "distance" in result

    def test_upsert_is_idempotent(self, vector_store: VectorStore) -> None:
        chunk = Chunk(text="Idempotent content.", metadata={"source_type": "folder"})
        vector_store.add_chunks([1], [chunk])
        vector_store.add_chunks([1], [chunk])  # same ID, upsert
        assert vector_store.count() == 1  # should not duplicate

    def test_delete_by_document_id(self, vector_store: VectorStore) -> None:
        chunks = [
            Chunk(text="Doc A chunk 1.", metadata={"source_type": "folder", "document_id": "5"}),
            Chunk(text="Doc A chunk 2.", metadata={"source_type": "folder", "document_id": "5"}),
            Chunk(text="Doc B chunk 1.", metadata={"source_type": "folder", "document_id": "6"}),
        ]
        vector_store.add_chunks([1, 2, 3], chunks)
        assert vector_store.count() == 3

        vector_store.delete_by_document_id(5)
        # The two doc-A chunks should be gone; doc-B should remain
        assert vector_store.count() == 1

    def test_search_with_empty_store(self, vector_store: VectorStore) -> None:
        # Should not raise; returns empty list
        results = vector_store.search("any query", n_results=5)
        assert isinstance(results, list)

    def test_reset_clears_all_chunks(self, vector_store: VectorStore) -> None:
        chunks = [Chunk(text="Some text.", metadata={"source_type": "folder"})]
        vector_store.add_chunks([1], chunks)
        assert vector_store.count() == 1
        vector_store.reset()
        assert vector_store.count() == 0
