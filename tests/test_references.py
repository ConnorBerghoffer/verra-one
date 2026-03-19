"""Tests for cross-document reference detection (verra.ingest.references)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from verra.ingest.references import extract_references, resolve_references
from verra.store.metadata import MetadataStore


# ---------------------------------------------------------------------------
# extract_references
# ---------------------------------------------------------------------------


def test_detect_see_the_sla():
    text = "Please review the terms — see the SLA document for uptime guarantees."
    refs = extract_references(text)
    types = {r["reference_type"] for r in refs}
    texts = [r["reference_text"] for r in refs]
    assert "document" in types
    assert any("SLA" in t for t in texts)


def test_detect_jira_ticket():
    text = "This was tracked in JIRA-4521 and resolved last week."
    refs = extract_references(text)
    assert len(refs) >= 1
    assert any(r["reference_type"] == "ticket" for r in refs)
    assert any("JIRA-4521" in r["reference_text"] for r in refs)


def test_detect_as_discussed_in():
    text = "As discussed in the Q3 planning meeting, the deadline moves to October."
    refs = extract_references(text)
    assert len(refs) >= 1
    assert any(r["reference_type"] == "discussion" for r in refs)


def test_detect_attached_document():
    text = "Attached herewith is the renewal proposal for your review."
    refs = extract_references(text)
    assert len(refs) >= 1
    assert any(r["reference_type"] == "document" for r in refs)


def test_no_references_in_normal_text():
    text = (
        "We shipped the new feature to production yesterday and the metrics look great. "
        "Team velocity was 42 points this sprint."
    )
    refs = extract_references(text)
    assert refs == []


def test_pr_ticket_pattern():
    text = "The fix was merged in PR#198 after code review."
    refs = extract_references(text)
    assert any("PR" in r["reference_text"] for r in refs)
    assert any(r["reference_type"] == "ticket" for r in refs)


def test_multiple_references():
    text = (
        "Per the SLA document, uptime must be 99.9%. "
        "See JIRA-100 for historical context. "
        "As discussed in the onboarding plan, this applies to new customers."
    )
    refs = extract_references(text)
    assert len(refs) >= 2


# ---------------------------------------------------------------------------
# resolve_references
# ---------------------------------------------------------------------------


@pytest.fixture
def store_with_docs(tmp_path: Path) -> MetadataStore:
    store = MetadataStore(tmp_path / "meta.db")
    store.add_document(
        file_path="/docs/sla_agreement.pdf",
        file_name="sla_agreement.pdf",
        source_type="folder",
        format="pdf",
        content_hash="abc123",
    )
    store.add_document(
        file_path="/docs/renewal_proposal.docx",
        file_name="renewal_proposal.docx",
        source_type="folder",
        format="docx",
        content_hash="def456",
    )
    yield store
    store.close()


def test_resolve_known_document(store_with_docs: MetadataStore):
    refs = [
        {"reference_text": "see the SLA document", "reference_type": "document", "target_hint": "SLA"},
    ]
    resolved = resolve_references(refs, store_with_docs)
    assert len(resolved) == 1
    assert resolved[0]["resolved_document_id"] is not None
    assert resolved[0]["confidence"] > 0.0


def test_resolve_unresolvable_reference(store_with_docs: MetadataStore):
    refs = [
        {"reference_text": "JIRA-9999", "reference_type": "ticket", "target_hint": "JIRA-9999"},
    ]
    resolved = resolve_references(refs, store_with_docs)
    assert len(resolved) == 1
    assert resolved[0]["resolved_document_id"] is None
