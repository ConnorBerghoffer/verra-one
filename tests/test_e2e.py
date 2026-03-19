"""End-to-end integration tests using the test-data/ business dataset.

These tests verify the full pipeline: ingest → retrieve → chat
using realistic business documents (policies, contracts, invoices,
meeting notes, emails, team/client rosters).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from verra.agent.chat import ChatEngine
from verra.agent.llm import LLMClient
from verra.ingest.pipeline import ingest_folder
from verra.retrieval.router import classify_query, parse_query, QueryType
from verra.retrieval.search import search
from verra.store.memory import MemoryStore
from verra.store.metadata import MetadataStore
from verra.store.vector import VectorStore

# Path to the test business data
TEST_DATA_DIR = Path(__file__).parent.parent / "test-data"


@pytest.fixture(scope="module")
def stores():
    """Set up a temporary data dir, ingest test-data/, yield stores, cleanup."""
    tmp = tempfile.mkdtemp(prefix="verra_e2e_")
    tmp_path = Path(tmp)

    ms = MetadataStore(tmp_path / "metadata.db")
    vs = VectorStore(tmp_path / "chroma")
    mem = MemoryStore(tmp_path / "memory.db")

    # Ingest the full test dataset
    stats = ingest_folder(
        folder_path=TEST_DATA_DIR,
        metadata_store=ms,
        vector_store=vs,
    )

    assert stats.files_processed >= 8, f"Expected at least 8 files, got {stats.files_processed}"
    assert stats.chunks_created >= 8, f"Expected at least 8 chunks, got {stats.chunks_created}"

    yield {"ms": ms, "vs": vs, "mem": mem, "stats": stats}

    ms.close()
    mem.close()
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def engine(stores):
    """Create a ChatEngine with a mock LLM that echoes context."""
    llm = LLMClient(model="mock")

    def mock_complete(messages):
        for m in messages:
            content = m.get("content", "")
            if "Context from your data" in content:
                # Extract source labels
                sources = []
                i = 1
                while f"[Source {i}:" in content:
                    start = content.find(f"[Source {i}: ") + len(f"[Source {i}: ")
                    end = content.find("]", start)
                    sources.append(content[start:end])
                    i += 1
                # Extract actual context text for verification
                q_idx = content.rfind("Question: ")
                question = content[q_idx + 10 :] if q_idx >= 0 else "?"
                return f"SOURCES: {', '.join(sources[:5])}\nCONTEXT AVAILABLE: yes\nQUESTION: {question[:80]}"
        return "No context provided"

    llm.complete = mock_complete

    return ChatEngine(
        llm=llm,
        metadata_store=stores["ms"],
        vector_store=stores["vs"],
        memory_store=stores["mem"],
    )


# -------------------------------------------------------------------------
# Ingestion tests
# -------------------------------------------------------------------------


class TestIngestion:
    def test_all_files_ingested(self, stores):
        assert stores["stats"].files_processed >= 8

    def test_chunks_created(self, stores):
        assert stores["stats"].chunks_created >= 8

    def test_no_errors(self, stores):
        assert len(stores["stats"].errors) == 0

    def test_vector_count_matches(self, stores):
        assert stores["vs"].count() == stores["stats"].chunks_created

    def test_reindex_skips_unchanged(self, stores):
        """Re-ingesting the same folder should skip all files."""
        stats2 = ingest_folder(
            folder_path=TEST_DATA_DIR,
            metadata_store=stores["ms"],
            vector_store=stores["vs"],
        )
        assert stats2.files_skipped >= 8
        assert stats2.files_processed == 0


# -------------------------------------------------------------------------
# Retrieval quality tests
# -------------------------------------------------------------------------


class TestRetrievalQuality:
    """Verify that the right documents rank first for specific queries."""

    def test_refund_policy_finds_policies(self, stores):
        q = parse_query("what is our refund policy?")
        results = search(q, stores["ms"], stores["vs"], n_results=3)
        top_source = results[0].metadata.get("file_name", "")
        assert "policies" in top_source.lower(), f"Expected policies doc, got {top_source}"

    def test_acme_pricing_finds_contract(self, stores):
        q = parse_query("how much do we charge Acme Corp?")
        results = search(q, stores["ms"], stores["vs"], n_results=3)
        top_source = results[0].metadata.get("file_name", "")
        assert "acme" in top_source.lower(), f"Expected Acme contract, got {top_source}"

    def test_overdue_invoices_finds_relevant(self, stores):
        q = parse_query("what invoices are overdue?")
        results = search(q, stores["ms"], stores["vs"], n_results=3)
        sources = [r.metadata.get("file_name", "") for r in results]
        has_invoices_or_emails = any(
            "invoice" in s.lower() or "email" in s.lower() for s in sources
        )
        assert has_invoices_or_emails, f"Expected invoice/email source, got {sources}"

    def test_acme_expiry_finds_contract(self, stores):
        q = parse_query("when does the Acme contract expire?")
        results = search(q, stores["ms"], stores["vs"], n_results=3)
        top_source = results[0].metadata.get("file_name", "")
        assert "acme" in top_source.lower(), f"Expected Acme contract, got {top_source}"

    def test_team_roster_finds_roster(self, stores):
        q = parse_query("who is on our team?")
        results = search(q, stores["ms"], stores["vs"], n_results=3)
        top_source = results[0].metadata.get("file_name", "")
        assert "team" in top_source.lower() or "roster" in top_source.lower(), f"Expected team roster, got {top_source}"

    def test_standup_finds_meeting_notes(self, stores):
        q = parse_query("what action items came from the March standup?")
        results = search(q, stores["ms"], stores["vs"], n_results=3)
        top_source = results[0].metadata.get("file_name", "")
        assert "standup" in top_source.lower() or "2024-03" in top_source.lower(), f"Expected standup notes, got {top_source}"

    def test_all_queries_return_results(self, stores):
        """Every reasonable business query should return at least 1 result."""
        queries = [
            "what is our refund policy?",
            "how much do we charge Acme Corp?",
            "who is Lisa Park?",
            "what are the overdue invoices?",
            "when does the Acme contract expire?",
            "who is on our team?",
            "what did Jake Mitchell want?",
            "what is Priya working on?",
        ]
        for q_text in queries:
            q = parse_query(q_text)
            results = search(q, stores["ms"], stores["vs"], n_results=3)
            assert len(results) > 0, f"No results for: {q_text}"


# -------------------------------------------------------------------------
# Chat engine tests
# -------------------------------------------------------------------------


class TestChatEngine:
    """Verify the full chat pipeline: query → retrieve → LLM → response."""

    def test_refund_policy_has_context(self, engine):
        resp = engine.ask("what is our refund policy?")
        assert resp.had_context is True
        assert len(resp.sources) > 0

    def test_refund_policy_cites_policies_doc(self, engine):
        resp = engine.ask("what is our refund policy?")
        source_labels = [s["label"] for s in resp.sources]
        assert any("policies" in l.lower() for l in source_labels), f"Sources: {source_labels}"

    def test_acme_pricing_has_context(self, engine):
        resp = engine.ask("how much is the Acme Corp retainer?")
        assert resp.had_context is True

    def test_overdue_invoices_has_context(self, engine):
        resp = engine.ask("which invoices are overdue?")
        assert resp.had_context is True

    def test_team_info_has_context(self, engine):
        resp = engine.ask("who works at Berghoffer Digital?")
        assert resp.had_context is True

    def test_meeting_actions_has_context(self, engine):
        resp = engine.ask("team standup meeting notes action items Connor Marcus Priya")
        assert resp.had_context is True

    def test_conversation_memory_persists(self, engine):
        """Ask two related questions — the second should work in context."""
        engine.ask("tell me about the Acme contract")
        resp2 = engine.ask("when does it expire?")
        # The engine should still find relevant context
        assert resp2.had_context is True

    def test_irrelevant_query_still_returns(self, engine):
        """Even weird queries shouldn't crash, just return low-confidence."""
        resp = engine.ask("what is the airspeed velocity of an unladen swallow?")
        # Should still return something (either context or no-context response)
        assert resp.answer is not None
        assert isinstance(resp.answer, str)


# -------------------------------------------------------------------------
# Query router tests with real queries
# -------------------------------------------------------------------------


class TestQueryRouter:
    def test_refund_is_semantic(self):
        assert classify_query("what is our refund policy?") == QueryType.SEMANTIC

    def test_emails_from_person_is_metadata_or_hybrid(self):
        qt = classify_query("emails from John last month")
        assert qt in (QueryType.METADATA, QueryType.HYBRID)

    def test_what_did_person_say_is_hybrid(self):
        assert classify_query("what did Lisa say about pricing?") == QueryType.HYBRID

    def test_general_question_is_semantic(self):
        assert classify_query("how much do we charge?") == QueryType.SEMANTIC
