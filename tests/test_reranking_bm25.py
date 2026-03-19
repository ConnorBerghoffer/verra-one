"""Tests for cross-encoder reranking and BM25/FTS5 search.

Covers:
  - rerank() falls back gracefully when reranker is unavailable
  - rerank() normalises scores to [0, 1]
  - rerank() returns top_n results sorted by score descending
  - MetadataStore.search_fts() returns BM25 results
  - MetadataStore.ensure_fts_populated() backfills from metadata JSON
  - MetadataStore.index_chunk_text() upserts single chunks
  - _bm25_search() integrates with vector store to enrich results
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from verra.retrieval.router import ClassifiedQuery, QueryType
from verra.retrieval.search import SearchResult, rerank, _bm25_search
from verra.store.metadata import MetadataStore


# ---------------------------------------------------------------------------
# rerank()
# ---------------------------------------------------------------------------


class TestRerank:
    def test_fallback_when_reranker_unavailable(self) -> None:
        """rerank() returns first top_n results in input order when reranker is None."""
        results = [
            SearchResult(chunk_id=str(i), text=f"doc {i}", score=float(i))
            for i in range(5)
        ]
        with patch("verra.retrieval.search._get_reranker", return_value=None):
            out = rerank("test query", results, top_n=3)

        assert len(out) == 3
        # Input order preserved (fallback path)
        assert [r.chunk_id for r in out] == ["0", "1", "2"]

    def test_empty_results_returns_empty(self) -> None:
        with patch("verra.retrieval.search._get_reranker", return_value=None):
            out = rerank("query", [], top_n=5)
        assert out == []

    def test_scores_normalised_to_0_1(self) -> None:
        """Reranker output scores must be in [0, 1] regardless of raw logit range."""
        mock_ranker = MagicMock()
        # Simulate typical ms-marco logit range: very negative values
        mock_ranker.predict.return_value = [-12.5, -8.0, -15.0, -3.5]

        results = [
            SearchResult(chunk_id=str(i), text=f"doc {i}", score=0.5)
            for i in range(4)
        ]
        with patch("verra.retrieval.search._get_reranker", return_value=mock_ranker):
            out = rerank("query", results, top_n=4)

        assert all(0.0 <= r.score <= 1.0 for r in out), (
            f"Scores out of [0,1]: {[r.score for r in out]}"
        )

    def test_sorted_descending_by_score(self) -> None:
        """Results must be sorted best-first after reranking."""
        mock_ranker = MagicMock()
        mock_ranker.predict.return_value = [-10.0, -2.0, -7.0]

        results = [
            SearchResult(chunk_id="low", text="low relevance", score=0.5),
            SearchResult(chunk_id="high", text="high relevance", score=0.5),
            SearchResult(chunk_id="mid", text="mid relevance", score=0.5),
        ]
        with patch("verra.retrieval.search._get_reranker", return_value=mock_ranker):
            out = rerank("query", results, top_n=3)

        assert out[0].chunk_id == "high"  # raw score -2.0 → highest after normalisation
        assert out[1].chunk_id == "mid"   # raw score -7.0
        assert out[2].chunk_id == "low"   # raw score -10.0

    def test_top_n_respected(self) -> None:
        mock_ranker = MagicMock()
        mock_ranker.predict.return_value = [1.0, 2.0, 3.0, 4.0, 5.0]

        results = [
            SearchResult(chunk_id=str(i), text=f"doc {i}", score=0.5)
            for i in range(5)
        ]
        with patch("verra.retrieval.search._get_reranker", return_value=mock_ranker):
            out = rerank("query", results, top_n=2)

        assert len(out) == 2

    def test_all_identical_scores_no_crash(self) -> None:
        """When all raw scores are identical, normalisation must not divide by zero."""
        mock_ranker = MagicMock()
        mock_ranker.predict.return_value = [5.0, 5.0, 5.0]

        results = [
            SearchResult(chunk_id=str(i), text=f"doc {i}", score=0.5)
            for i in range(3)
        ]
        with patch("verra.retrieval.search._get_reranker", return_value=mock_ranker):
            out = rerank("query", results, top_n=3)

        assert all(r.score == 1.0 for r in out)


# ---------------------------------------------------------------------------
# MetadataStore FTS5
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MetadataStore:
    return MetadataStore(tmp_path / "test.db")


class TestMetadataStoreFTS:
    def test_search_fts_returns_matching_chunk(self, store: MetadataStore) -> None:
        """index_chunk_text + search_fts should find the indexed chunk."""
        store.index_chunk_text(42, "Jennifer Walsh is VP of Operations at Meridian Health.")
        results = store.search_fts("Jennifer Walsh")
        assert len(results) >= 1
        chunk_ids = [r["chunk_id"] for r in results]
        assert 42 in chunk_ids

    def test_search_fts_empty_on_no_match(self, store: MetadataStore) -> None:
        store.index_chunk_text(1, "The quick brown fox jumps.")
        results = store.search_fts("Jennifer Walsh Operations")
        assert results == []

    def test_search_fts_returns_snippet(self, store: MetadataStore) -> None:
        store.index_chunk_text(7, "Contract renewal is due on 2025-12-31 for Acme Corp.")
        results = store.search_fts("contract renewal")
        assert len(results) >= 1
        assert results[0]["snippet"]  # snippet should be non-empty

    def test_index_chunk_text_upsert(self, store: MetadataStore) -> None:
        """Re-indexing the same chunk_id should replace the previous entry."""
        store.index_chunk_text(99, "original content about finance")
        store.index_chunk_text(99, "updated content about operations")
        # Old content should not match; new content should
        assert store.search_fts("finance") == []
        assert len(store.search_fts("operations")) >= 1

    def test_search_fts_malformed_query_no_crash(self, store: MetadataStore) -> None:
        """FTS5 MATCH errors are caught and return empty list."""
        store.index_chunk_text(1, "some text content")
        # FTS5 MATCH syntax error — should not raise
        result = store.search_fts('AND OR "')  # malformed FTS5 query
        assert isinstance(result, list)

    def test_ensure_fts_populated_backfills_from_metadata(self, store: MetadataStore) -> None:
        """ensure_fts_populated reads text from the metadata JSON blob in chunks."""
        # Insert a document and chunk with text in the JSON metadata blob
        doc_id = store.add_document(
            file_path="/test/file.txt",
            file_name="file.txt",
            source_type="folder",
            format="txt",
            content_hash="abc123",
        )

        # Manually insert a chunk with text in the metadata JSON blob
        store._conn.execute(
            """
            INSERT INTO chunks (document_id, position, token_count, metadata, authority_weight)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, 0, 10, json.dumps({"text": "backfill test content about revenue"}), 50),
        )
        store._conn.commit()

        count = store.ensure_fts_populated()
        assert count == 1

        results = store.search_fts("revenue")
        assert len(results) >= 1

    def test_ensure_fts_populated_skips_already_indexed(self, store: MetadataStore) -> None:
        """ensure_fts_populated should not double-index already present chunks."""
        doc_id = store.add_document(
            file_path="/test/file2.txt",
            file_name="file2.txt",
            source_type="folder",
            format="txt",
            content_hash="def456",
        )
        store._conn.execute(
            """
            INSERT INTO chunks (document_id, position, token_count, metadata, authority_weight)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, 0, 5, json.dumps({"text": "already indexed text"}), 50),
        )
        store._conn.commit()

        # First population
        first = store.ensure_fts_populated()
        assert first == 1

        # Second call should find nothing new
        second = store.ensure_fts_populated()
        assert second == 0


# ---------------------------------------------------------------------------
# _bm25_search()
# ---------------------------------------------------------------------------


class TestBM25Search:
    def test_bm25_search_returns_empty_when_fts_empty(self, store: MetadataStore) -> None:
        """When the FTS index has no matches, _bm25_search returns []."""
        vs = MagicMock()
        q = ClassifiedQuery(
            raw="Jennifer Walsh VP Operations",
            semantic_text="Jennifer Walsh VP Operations",
            query_type=QueryType.SEMANTIC,
        )
        results = _bm25_search(q, store, vs, n_results=5)
        assert results == []

    def test_bm25_search_enriches_with_vector_metadata(self, store: MetadataStore) -> None:
        """_bm25_search fetches chunk text from ChromaDB and returns SearchResult objects."""
        store.index_chunk_text(1, "Jennifer Walsh is VP of Operations at Meridian Health.")

        vs = MagicMock()
        vs._collection.get.return_value = {
            "ids": ["1"],
            "documents": ["Jennifer Walsh is VP of Operations at Meridian Health."],
            "metadatas": [{"authority_weight": 75, "file_name": "org_chart.md"}],
        }

        q = ClassifiedQuery(
            raw="Who is Jennifer Walsh?",
            semantic_text="Who is Jennifer Walsh?",
            query_type=QueryType.SEMANTIC,
        )
        results = _bm25_search(q, store, vs, n_results=5)

        assert len(results) == 1
        assert results[0].chunk_id == "1"
        assert results[0].authority_weight == 75
        assert 0.0 <= results[0].score <= 1.0

    def test_bm25_search_scores_normalised(self, store: MetadataStore) -> None:
        """Scores returned by _bm25_search must be in [0, 1]."""
        for i, text in enumerate([
            "Revenue for Q3 increased by 15 percent year over year.",
            "Q3 revenue performance exceeded expectations.",
            "Annual revenue figures are tracked quarterly.",
        ], start=1):
            store.index_chunk_text(i, text)

        vs = MagicMock()
        vs._collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "documents": [
                "Revenue for Q3 increased by 15 percent year over year.",
                "Q3 revenue performance exceeded expectations.",
                "Annual revenue figures are tracked quarterly.",
            ],
            "metadatas": [{"authority_weight": 50}] * 3,
        }

        q = ClassifiedQuery(
            raw="Q3 revenue",
            semantic_text="Q3 revenue",
            query_type=QueryType.SEMANTIC,
        )
        results = _bm25_search(q, store, vs, n_results=5)

        assert all(0.0 <= r.score <= 1.0 for r in results), (
            f"Scores not in [0,1]: {[r.score for r in results]}"
        )
