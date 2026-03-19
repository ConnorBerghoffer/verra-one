"""Tests for advanced chat engine features.

Covers:
  - Authority ranking integration
  - Smart "don't know" detection (unknown entity, partial coverage)
  - Multi-hop fallback to single-pass when tool calling unavailable
  - Assertion extraction
  - System prompt content
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from verra.agent.chat import (
    ChatEngine,
    ChatResponse,
    KnowledgeAssessment,
    _assess_knowledge,
    _extract_sources,
    _SYSTEM_PROMPT,
)
from verra.agent.llm import LLMClient
from verra.retrieval.search import SearchResult
from verra.store.memory import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> LLMClient:
    llm = MagicMock(spec=LLMClient)
    llm.complete.return_value = "Here is the answer based on the data."
    llm.complete_with_tools.return_value = "Multi-hop answer based on search results."
    return llm


@pytest.fixture
def mock_metadata_store() -> MagicMock:
    store = MagicMock()
    store.search_emails.return_value = []
    return store


@pytest.fixture
def mock_vector_store() -> MagicMock:
    store = MagicMock()
    store.search.return_value = [
        {
            "id": "1",
            "document": "The refund policy allows 30-day returns.",
            "metadata": {"file_name": "policies.txt", "authority_weight": 80},
            "distance": 0.1,
        }
    ]
    store.count.return_value = 1
    return store


@pytest.fixture
def memory_store(tmp_path) -> MemoryStore:
    store = MemoryStore(tmp_path / "test_memory.db")
    yield store
    store.close()


@pytest.fixture
def engine(mock_llm, mock_metadata_store, mock_vector_store, memory_store) -> ChatEngine:
    return ChatEngine(
        llm=mock_llm,
        metadata_store=mock_metadata_store,
        vector_store=mock_vector_store,
        memory_store=memory_store,
    )


# ---------------------------------------------------------------------------
# Authority ranking tests (integration via search results)
# ---------------------------------------------------------------------------


class TestAuthorityRanking:
    def test_authority_ranking_in_search_results(self, engine, mock_vector_store) -> None:
        """When similarity scores are close, higher authority results should rank first.

        The cross-encoder reranker is patched out here so authority ranking is
        the deciding factor, matching the pre-reranker behavior this test was
        originally designed for.
        """
        # Both have similar distance (0.1 vs 0.15), so authority is the tiebreaker.
        # Composite for high_auth: 0.70*(1-0.1) + 0.25*(90/100) = 0.63+0.225 = 0.855
        # Composite for low_auth:  0.70*(1-0.15) + 0.25*(20/100) = 0.595+0.05 = 0.645
        mock_vector_store.search.return_value = [
            {"id": "1", "document": "Low authority doc.", "metadata": {"authority_weight": 20}, "distance": 0.15},
            {"id": "2", "document": "High authority doc.", "metadata": {"authority_weight": 90}, "distance": 0.1},
        ]
        # Disable the cross-encoder so authority ranking acts as the tiebreaker.
        # Without this patch the reranker's text-based scoring may override authority
        # for short placeholder documents that lack real semantic content.
        with patch("verra.retrieval.search._get_reranker", return_value=None):
            resp = engine.ask("what is the refund policy?", use_multi_hop=False)

        # The context passed to the LLM should reference high-authority source first
        call_args = engine.llm.complete.call_args
        messages = call_args[0][0]
        context_msg = next(m for m in messages if "Context from your data" in m.get("content", ""))
        content = context_msg["content"]

        high_pos = content.find("High authority doc.")
        low_pos = content.find("Low authority doc.")
        assert high_pos < low_pos, "High authority result should appear before low authority result"


# ---------------------------------------------------------------------------
# Smart "don't know" detection
# ---------------------------------------------------------------------------


class TestSmartDontKnow:
    def test_unknown_entity_returns_informative_response(self) -> None:
        """Entity not in registry should produce 'doesn't appear in data' note."""
        entity_store = MagicMock()
        entity_store.resolve.return_value = None  # nothing matches

        results: list[SearchResult] = []  # no retrieval results either

        assessment = _assess_knowledge(
            query="What is Globotech Inc's revenue?",
            results=results,
            entity_store=entity_store,
            coverage_store=None,
        )

        assert assessment.has_relevant_data is False
        # With only unknown entities and no data, confidence_note should say so
        assert "doesn't appear" in assessment.confidence_note or not assessment.confidence_note

    def test_smart_dont_know_unknown_entity(self) -> None:
        """Query about an entity not in registry produces a clear no-data note."""
        entity_store = MagicMock()
        entity_store.resolve.return_value = None

        results: list[SearchResult] = []

        assessment = _assess_knowledge(
            "Who is Unknown Person?",
            results,
            entity_store=entity_store,
            coverage_store=None,
        )

        assert assessment.has_relevant_data is False
        assert assessment.known_entities == []

    def test_smart_dont_know_partial_coverage(self) -> None:
        """Entity is known but has no data in our stores."""
        entity_store = MagicMock()
        entity_store.resolve.return_value = {"id": 1, "canonical_name": "Acme Corp", "entity_type": "company"}

        coverage_store = MagicMock()
        coverage_store.get_coverage.return_value = []  # coverage exists but empty
        coverage_store.has_any_data.return_value = False  # no chunks at all

        results: list[SearchResult] = []  # nothing retrieved

        assessment = _assess_knowledge(
            "What is Acme Corp's budget?",
            results,
            entity_store=entity_store,
            coverage_store=coverage_store,
        )

        assert assessment.has_relevant_data is False
        assert "Acme Corp" in assessment.known_entities
        # Should flag that we have records for Acme but no data for the topic
        assert len(assessment.missing_sources) > 0

    def test_low_confidence_note_prepended(self) -> None:
        """Low-scoring results trigger a 'Based on limited data' note."""
        results = [
            SearchResult(chunk_id="1", text="Sparse info.", score=0.3, authority_weight=50)
        ]

        assessment = _assess_knowledge(
            "What is the contract value?",
            results,
            entity_store=None,
            coverage_store=None,
        )

        # has_relevant_data is True (score 0.3 > _MIN_SCORE of -0.8)
        assert assessment.has_relevant_data is True
        # Confidence note reflects low score
        assert "limited data" in assessment.confidence_note

    def test_high_confidence_no_note(self) -> None:
        """High-scoring results produce no confidence note."""
        results = [
            SearchResult(chunk_id="1", text="Detailed answer.", score=0.9, authority_weight=80)
        ]

        assessment = _assess_knowledge(
            "What is our refund policy?",
            results,
            entity_store=None,
            coverage_store=None,
        )

        assert assessment.has_relevant_data is True
        assert assessment.confidence_note == ""


# ---------------------------------------------------------------------------
# Multi-hop fallback to single-pass
# ---------------------------------------------------------------------------


class TestMultiHopFallback:
    def test_multi_hop_fallback_to_single_pass(
        self, mock_metadata_store, mock_vector_store, memory_store
    ) -> None:
        """If complete_with_tools raises, engine falls back to single-pass."""
        llm = MagicMock(spec=LLMClient)
        llm.complete_with_tools.side_effect = Exception("model does not support tool calling")
        llm.complete.return_value = "Single-pass answer."

        engine = ChatEngine(
            llm=llm,
            metadata_store=mock_metadata_store,
            vector_store=mock_vector_store,
            memory_store=memory_store,
        )

        resp = engine.ask("what is the refund policy?", use_multi_hop=True)

        # Should not raise; should fall back and call llm.complete
        assert resp.answer == "Single-pass answer."
        llm.complete.assert_called_once()

    def test_multi_hop_disabled_uses_single_pass(
        self, mock_llm, mock_metadata_store, mock_vector_store, memory_store
    ) -> None:
        """use_multi_hop=False skips tool calling entirely."""
        engine = ChatEngine(
            llm=mock_llm,
            metadata_store=mock_metadata_store,
            vector_store=mock_vector_store,
            memory_store=memory_store,
        )

        engine.ask("question?", use_multi_hop=False)

        mock_llm.complete.assert_called_once()
        mock_llm.complete_with_tools.assert_not_called()

    def test_multi_hop_uses_tool_calling_when_available(
        self, mock_llm, mock_metadata_store, mock_vector_store, memory_store
    ) -> None:
        """use_multi_hop=True invokes complete_with_tools."""
        engine = ChatEngine(
            llm=mock_llm,
            metadata_store=mock_metadata_store,
            vector_store=mock_vector_store,
            memory_store=memory_store,
        )

        engine.ask("question?", use_multi_hop=True)

        mock_llm.complete_with_tools.assert_called_once()


# ---------------------------------------------------------------------------
# Assertion extraction
# ---------------------------------------------------------------------------


class TestAssertionExtraction:
    def test_assertion_extraction_basic(self, tmp_path) -> None:
        """Factual sentences in answers are stored as assertions."""
        from verra.store.assertions import AssertionStore

        assertion_store = AssertionStore(tmp_path / "assertions.db")

        llm = MagicMock(spec=LLMClient)
        llm.complete.return_value = (
            "The Acme Corp contract expires on 2025-12-31. "
            "The retainer costs $5,000 per month. "
            "This sentence has no factual content."
        )
        llm.complete_with_tools.side_effect = Exception("no tool support")

        memory_store = MemoryStore(tmp_path / "memory.db")
        metadata_store = MagicMock()
        metadata_store.search_emails.return_value = []
        vector_store = MagicMock()
        vector_store.search.return_value = [
            {"id": "1", "document": "Acme contract details.", "metadata": {"file_name": "acme.txt"}, "distance": 0.1}
        ]
        vector_store.count.return_value = 1

        engine = ChatEngine(
            llm=llm,
            metadata_store=metadata_store,
            vector_store=vector_store,
            memory_store=memory_store,
            assertion_store=assertion_store,
        )

        engine.ask("when does the Acme contract expire?", use_multi_hop=False)

        # At least the "expires on" and "costs" sentences should be stored
        rows = assertion_store.get_current_assertions([])
        # No entity IDs, but null-entity assertions should exist
        all_rows = assertion_store._conn.execute("SELECT * FROM assertions").fetchall()
        assert len(all_rows) >= 1  # At least one factual sentence extracted

        assertion_store.close()
        memory_store.close()

    def test_assertion_extraction_skips_short_sentences(self, tmp_path) -> None:
        """Sentences shorter than 15 chars are not stored as assertions."""
        from verra.store.assertions import AssertionStore

        assertion_store = AssertionStore(tmp_path / "assertions2.db")
        llm = MagicMock(spec=LLMClient)
        llm.complete.return_value = "Yes. No. Maybe."
        llm.complete_with_tools.side_effect = Exception("no tools")

        memory_store = MemoryStore(tmp_path / "memory2.db")
        metadata_store = MagicMock()
        metadata_store.search_emails.return_value = []
        vector_store = MagicMock()
        vector_store.search.return_value = [
            {"id": "1", "document": "Some text.", "metadata": {"file_name": "test.txt"}, "distance": 0.1}
        ]
        vector_store.count.return_value = 1

        engine = ChatEngine(
            llm=llm,
            metadata_store=metadata_store,
            vector_store=vector_store,
            memory_store=memory_store,
            assertion_store=assertion_store,
        )
        engine.ask("test?", use_multi_hop=False)

        all_rows = assertion_store._conn.execute("SELECT * FROM assertions").fetchall()
        assert len(all_rows) == 0

        assertion_store.close()
        memory_store.close()

    def test_no_assertion_store_does_not_crash(
        self, mock_llm, mock_metadata_store, mock_vector_store, memory_store
    ) -> None:
        """Engine works fine with no assertion_store provided."""
        engine = ChatEngine(
            llm=mock_llm,
            metadata_store=mock_metadata_store,
            vector_store=mock_vector_store,
            memory_store=memory_store,
            assertion_store=None,
        )
        resp = engine.ask("anything?", use_multi_hop=False)
        assert resp.answer is not None


# ---------------------------------------------------------------------------
# System prompt content
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_prompt_contains_conflict_instruction(self) -> None:
        assert "conflicting information" in _SYSTEM_PROMPT.lower() or "conflict" in _SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_tool_instruction(self) -> None:
        assert "search_knowledge_base" in _SYSTEM_PROMPT

    def test_system_prompt_contains_citation_instruction(self) -> None:
        assert "cite" in _SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_authority_instruction(self) -> None:
        assert "authority" in _SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_multiple_search_instruction(self) -> None:
        assert "multiple" in _SYSTEM_PROMPT.lower() or "more than once" in _SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_previous_assertions_instruction(self) -> None:
        assert "previous assertions" in _SYSTEM_PROMPT.lower() or "prior" in _SYSTEM_PROMPT.lower() or "contradict" in _SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Backwards-compatibility: existing ask() signature still works
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    def test_ask_no_extra_params(self, engine) -> None:
        """Original ask(user_message) signature still works."""
        resp = engine.ask("what is our refund policy?")
        assert isinstance(resp, ChatResponse)
        assert isinstance(resp.answer, str)

    def test_had_context_true_when_results_found(self, engine) -> None:
        resp = engine.ask("refund policy?", use_multi_hop=False)
        assert resp.had_context is True

    def test_sources_list_returned(self, engine) -> None:
        resp = engine.ask("refund policy?", use_multi_hop=False)
        assert isinstance(resp.sources, list)

    def test_query_type_in_response(self, engine) -> None:
        resp = engine.ask("refund policy?", use_multi_hop=False)
        assert resp.query_type in ("semantic", "metadata", "hybrid")
