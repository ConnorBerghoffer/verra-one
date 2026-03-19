"""Tests for authority-weighted ranking and entity-based retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from verra.retrieval.router import ClassifiedQuery, QueryType
from verra.retrieval.search import SearchResult, rank_by_authority, _entity_search


# ---------------------------------------------------------------------------
# rank_by_authority
# ---------------------------------------------------------------------------


class TestRankByAuthority:
    def test_higher_authority_ranks_first_when_scores_equal(self) -> None:
        """With identical similarity scores, higher authority wins."""
        low = SearchResult(chunk_id="1", text="low authority", score=0.7, authority_weight=30)
        high = SearchResult(chunk_id="2", text="high authority", score=0.7, authority_weight=90)
        mid = SearchResult(chunk_id="3", text="mid authority", score=0.7, authority_weight=60)

        ranked = rank_by_authority([low, high, mid])

        assert ranked[0].chunk_id == "2"  # authority 90
        assert ranked[1].chunk_id == "3"  # authority 60
        assert ranked[2].chunk_id == "1"  # authority 30

    def test_high_score_beats_high_authority_when_relevance_dominant(self) -> None:
        """Highly relevant low-authority doc should beat low-relevance high-authority doc."""
        # 0.70*0.95 + 0.25*0.30 = 0.665+0.075 = 0.740 (high score, low auth)
        # 0.70*0.40 + 0.25*0.90 = 0.280+0.225 = 0.505 (low score, high auth)
        high_score_low_auth = SearchResult(chunk_id="A", text="very relevant", score=0.95, authority_weight=30)
        low_score_high_auth = SearchResult(chunk_id="B", text="less relevant", score=0.40, authority_weight=90)

        ranked = rank_by_authority([low_score_high_auth, high_score_low_auth])

        assert ranked[0].chunk_id == "A"

    def test_same_authority_same_date_score_breaks_tie(self) -> None:
        high_score = SearchResult(chunk_id="x", text="high score", score=0.9, authority_weight=70, valid_from="2023-01-01")
        low_score = SearchResult(chunk_id="y", text="low score", score=0.4, authority_weight=70, valid_from="2023-01-01")

        ranked = rank_by_authority([low_score, high_score])

        assert ranked[0].chunk_id == "x"

    def test_same_authority_newer_valid_from_gives_small_boost(self) -> None:
        """Recency boost (0.05) tips the tie when authority and score are equal."""
        older = SearchResult(chunk_id="a", text="older", score=0.7, authority_weight=50, valid_from=None)
        newer = SearchResult(chunk_id="b", text="newer", score=0.7, authority_weight=50, valid_from="2024-06-15")

        ranked = rank_by_authority([older, newer])

        # newer has recency boost so its composite is slightly higher
        assert ranked[0].chunk_id == "b"

    def test_empty_list_returns_empty(self) -> None:
        assert rank_by_authority([]) == []

    def test_single_item_unchanged(self) -> None:
        r = SearchResult(chunk_id="z", text="only one", score=0.5, authority_weight=50)
        assert rank_by_authority([r]) == [r]

    def test_higher_score_wins_when_authority_equal(self) -> None:
        """Without recency difference, higher score is decisive."""
        lower_score = SearchResult(chunk_id="d", text="lower score", score=0.5, authority_weight=50)
        higher_score = SearchResult(chunk_id="u", text="higher score", score=0.9, authority_weight=50)

        ranked = rank_by_authority([lower_score, higher_score])
        assert ranked[0].chunk_id == "u"

    def test_authority_weight_in_results_field(self) -> None:
        result = SearchResult(chunk_id="r", text="test", score=0.5, authority_weight=75)
        assert result.authority_weight == 75

    def test_default_authority_weight_is_50(self) -> None:
        result = SearchResult(chunk_id="r", text="test", score=0.5)
        assert result.authority_weight == 50

    def test_default_valid_from_is_none(self) -> None:
        result = SearchResult(chunk_id="r", text="test", score=0.5)
        assert result.valid_from is None


# ---------------------------------------------------------------------------
# entity-based search
# ---------------------------------------------------------------------------


class TestEntityBasedSearch:
    def _make_classified(self, text: str) -> ClassifiedQuery:
        return ClassifiedQuery(
            raw=text,
            query_type=QueryType.SEMANTIC,
            semantic_text=text,
        )

    def test_entity_search_returns_empty_when_no_match(self) -> None:
        entity_store = MagicMock()
        entity_store.resolve.return_value = None  # no entity found

        metadata_store = MagicMock()
        vector_store = MagicMock()

        query = self._make_classified("unknown entity xyz")
        results = _entity_search(query, entity_store, metadata_store, vector_store, n_results=5)

        assert results == []

    def test_entity_search_returns_empty_when_no_chunks_linked(self) -> None:
        entity_store = MagicMock()
        entity_store.resolve.return_value = {"id": 1, "canonical_name": "Acme Corp", "entity_type": "company"}
        entity_store.get_chunks_for_entity.return_value = []  # entity exists but no chunks

        metadata_store = MagicMock()
        vector_store = MagicMock()

        query = self._make_classified("Acme Corp revenue")
        results = _entity_search(query, entity_store, metadata_store, vector_store, n_results=5)

        assert results == []

    def test_entity_search_returns_ranked_results(self) -> None:
        entity_store = MagicMock()
        entity_store.resolve.return_value = {"id": 1, "canonical_name": "Acme", "entity_type": "company"}
        entity_store.get_chunks_for_entity.return_value = [42, 43]

        vector_store = MagicMock()
        vector_store.search.return_value = [
            {"id": "42", "document": "Acme signed a $100k contract.", "metadata": {"file_name": "acme_contract.txt", "authority_weight": 80}, "distance": 0.1},
            {"id": "43", "document": "Acme renewal terms.", "metadata": {"file_name": "acme_renewal.txt", "authority_weight": 60}, "distance": 0.3},
            {"id": "99", "document": "Unrelated doc.", "metadata": {"file_name": "other.txt"}, "distance": 0.5},
        ]

        metadata_store = MagicMock()
        query = self._make_classified("Acme contract value")
        results = _entity_search(query, entity_store, metadata_store, vector_store, n_results=5)

        # Should only include chunks linked to the entity (42, 43), not 99
        chunk_ids = [r.chunk_id for r in results]
        assert "42" in chunk_ids
        assert "43" in chunk_ids
        assert "99" not in chunk_ids

    def test_entity_search_re_ranks_by_vector_similarity(self) -> None:
        entity_store = MagicMock()
        entity_store.resolve.return_value = {"id": 5, "canonical_name": "Bob Smith", "entity_type": "person"}
        entity_store.get_chunks_for_entity.return_value = [10, 11]

        vector_store = MagicMock()
        # chunk 11 has smaller distance = higher similarity
        vector_store.search.return_value = [
            {"id": "10", "document": "Bob discussed the project.", "metadata": {"authority_weight": 50}, "distance": 0.4},
            {"id": "11", "document": "Bob is the project lead.", "metadata": {"authority_weight": 50}, "distance": 0.1},
        ]

        metadata_store = MagicMock()
        query = self._make_classified("Bob Smith role")
        results = _entity_search(query, entity_store, metadata_store, vector_store, n_results=5)

        # After authority rank (both 50), higher score (chunk 11, dist 0.1) should rank first
        assert results[0].chunk_id == "11"
        assert results[1].chunk_id == "10"
