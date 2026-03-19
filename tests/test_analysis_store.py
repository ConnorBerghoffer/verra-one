"""Tests for the analysis store (chunk_analysis, conflicts, commitments, summaries, coverage)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from verra.store.analysis import AnalysisStore


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        s = AnalysisStore(f.name)
        yield s
        s.close()


class TestChunkAnalysis:
    def test_set_and_get_status(self, store):
        store.set_chunk_status(1, "pending")
        result = store.get_chunk_analysis(1)
        assert result is not None
        assert result["analysis_status"] == "pending"

    def test_save_full_analysis(self, store):
        store.save_chunk_analysis(
            chunk_id=1,
            sentiment="positive",
            staleness_risk=0.3,
            topics=["pricing", "contract"],
            assertions_count=5,
        )
        result = store.get_chunk_analysis(1)
        assert result["sentiment"] == "positive"
        assert result["staleness_risk"] == 0.3
        assert result["topics"] == ["pricing", "contract"]
        assert result["assertions_extracted"] == 5
        assert result["analysis_status"] == "analysed"

    def test_get_pending_chunks(self, store):
        store.set_chunk_status(1, "pending")
        store.set_chunk_status(2, "pending")
        store.set_chunk_status(3, "analysed")
        pending = store.get_pending_chunks()
        assert set(pending) == {1, 2}

    def test_pending_limit(self, store):
        for i in range(20):
            store.set_chunk_status(i, "pending")
        assert len(store.get_pending_chunks(limit=5)) == 5

    def test_missing_chunk_returns_none(self, store):
        assert store.get_chunk_analysis(999) is None


class TestConflicts:
    def test_add_and_retrieve(self, store):
        cid = store.add_conflict(
            assertion_a="Project is in phase 2",
            assertion_b="Project is in phase 3",
            entity_id=42,
        )
        conflicts = store.get_unresolved_conflicts(entity_id=42)
        assert len(conflicts) == 1
        assert conflicts[0]["assertion_a"] == "Project is in phase 2"

    def test_resolve_conflict(self, store):
        cid = store.add_conflict("old claim", "new claim")
        store.resolve_conflict(cid, notes="New info is correct")
        assert len(store.get_unresolved_conflicts()) == 0

    def test_filter_by_entity(self, store):
        store.add_conflict("a", "b", entity_id=1)
        store.add_conflict("c", "d", entity_id=2)
        assert len(store.get_unresolved_conflicts(entity_id=1)) == 1
        assert len(store.get_unresolved_conflicts(entity_id=2)) == 1
        assert len(store.get_unresolved_conflicts()) == 2


class TestCommitments:
    def test_add_and_retrieve(self, store):
        store.add_commitment(
            who_name="Connor",
            what="Send the proposal by Friday",
            due_date="2024-03-22",
        )
        commits = store.get_open_commitments()
        assert len(commits) == 1
        assert commits[0]["who_name"] == "Connor"
        assert commits[0]["what"] == "Send the proposal by Friday"

    def test_update_status(self, store):
        cid = store.add_commitment(who_name="Jake", what="Review contract")
        store.update_commitment_status(cid, "completed")
        assert len(store.get_open_commitments()) == 0

    def test_filter_by_entity(self, store):
        store.add_commitment(who_name="A", what="task 1", who_entity_id=10)
        store.add_commitment(who_name="B", what="task 2", who_entity_id=20)
        assert len(store.get_open_commitments(entity_id=10)) == 1


class TestEntitySummaries:
    def test_save_and_retrieve(self, store):
        store.save_entity_summary(
            entity_id=1,
            summary_text="Acme Corp is a key client with a $12.5K monthly retainer.",
            chunk_count=15,
            based_on_chunks=[1, 2, 3],
        )
        result = store.get_entity_summary(1)
        assert result is not None
        assert "Acme Corp" in result["summary_text"]
        assert result["chunk_count"] == 15
        assert result["based_on_chunks"] == [1, 2, 3]

    def test_upsert_overwrites(self, store):
        store.save_entity_summary(1, "v1", chunk_count=5)
        store.save_entity_summary(1, "v2", chunk_count=10)
        result = store.get_entity_summary(1)
        assert result["summary_text"] == "v2"
        assert result["chunk_count"] == 10

    def test_missing_returns_none(self, store):
        assert store.get_entity_summary(999) is None


class TestDocumentCoverage:
    def test_update_and_retrieve(self, store):
        store.update_document_coverage(entity_id=1, document_type="contract")
        store.update_document_coverage(entity_id=1, document_type="email")
        coverage = store.get_entity_coverage(1)
        assert len(coverage) == 2
        types = {c["document_type"] for c in coverage}
        assert types == {"contract", "email"}

    def test_coverage_gaps(self, store):
        store.update_document_coverage(entity_id=1, document_type="contract")
        store.update_document_coverage(entity_id=1, document_type="email")
        gaps = store.get_coverage_gaps(1, ["contract", "email", "sla", "financial"])
        assert set(gaps) == {"sla", "financial"}

    def test_no_gaps_when_complete(self, store):
        store.update_document_coverage(1, "contract")
        store.update_document_coverage(1, "email")
        assert store.get_coverage_gaps(1, ["contract", "email"]) == []

    def test_upsert_updates_timestamp(self, store):
        store.update_document_coverage(1, "contract")
        store.update_document_coverage(1, "contract")  # should not error
        assert len(store.get_entity_coverage(1)) == 1
