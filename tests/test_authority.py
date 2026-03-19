"""Tests for classify_document_authority() in verra.store.metadata."""

from __future__ import annotations

import pytest

from verra.store.metadata import classify_document_authority


# ---------------------------------------------------------------------------
# Policy / handbook
# ---------------------------------------------------------------------------


def test_policy_by_filename() -> None:
    doc_type, weight = classify_document_authority("remote-work-policy.pdf", "/docs/remote-work-policy.pdf", "")
    assert doc_type == "policy"
    assert weight == 90


def test_policies_in_path() -> None:
    doc_type, weight = classify_document_authority("document.docx", "/hr/policies/leave.docx", "")
    assert doc_type == "policy"
    assert weight == 90


def test_handbook_in_filename() -> None:
    doc_type, weight = classify_document_authority("employee_handbook.pdf", "/onboarding/employee_handbook.pdf", "")
    assert doc_type == "policy"
    assert weight == 90


def test_guidelines_in_content() -> None:
    content = "This document sets out the company guidelines for expense reimbursement."
    doc_type, weight = classify_document_authority("expenses.txt", "/docs/expenses.txt", content)
    assert doc_type == "policy"
    assert weight == 90


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_contract_by_filename() -> None:
    doc_type, weight = classify_document_authority("services-contract.pdf", "/legal/services-contract.pdf", "")
    assert doc_type == "contract"
    assert weight == 85


def test_msa_filename() -> None:
    doc_type, weight = classify_document_authority("msa_acme.pdf", "/legal/msa_acme.pdf", "")
    assert doc_type == "contract"
    assert weight == 85


def test_nda_in_filename() -> None:
    doc_type, weight = classify_document_authority("nda_signed.pdf", "/legal/nda_signed.pdf", "")
    assert doc_type == "contract"
    assert weight == 85


def test_agreement_in_content() -> None:
    content = "This is a service agreement between Acme Inc and the Provider."
    doc_type, weight = classify_document_authority("doc.txt", "/files/doc.txt", content)
    assert doc_type == "contract"
    assert weight == 85


def test_proposal_filename() -> None:
    doc_type, weight = classify_document_authority("proposal_q1.pdf", "/sales/proposal_q1.pdf", "")
    assert doc_type == "contract"
    assert weight == 80


def test_scope_in_filename() -> None:
    doc_type, weight = classify_document_authority("scope_of_work.docx", "/projects/scope_of_work.docx", "")
    assert doc_type == "contract"
    assert weight == 80


# ---------------------------------------------------------------------------
# Executive
# ---------------------------------------------------------------------------


def test_executive_path_segment() -> None:
    doc_type, weight = classify_document_authority("memo.txt", "/executive/q3_memo.txt", "A strategic memo.")
    assert doc_type == "executive"
    assert weight == 80


def test_ceo_in_path() -> None:
    doc_type, weight = classify_document_authority("update.docx", "/ceo/weekly_update.docx", "")
    assert doc_type == "executive"
    assert weight == 80


def test_director_in_path() -> None:
    doc_type, weight = classify_document_authority("notes.txt", "/director/board/notes.txt", "")
    assert doc_type == "executive"
    assert weight == 80


# ---------------------------------------------------------------------------
# Financial
# ---------------------------------------------------------------------------


def test_invoice_filename() -> None:
    doc_type, weight = classify_document_authority("invoice_001.pdf", "/billing/invoice_001.pdf", "")
    assert doc_type == "financial"
    assert weight == 75


def test_receipt_in_filename() -> None:
    doc_type, weight = classify_document_authority("receipt_may.pdf", "/receipts/receipt_may.pdf", "")
    assert doc_type == "financial"
    assert weight == 75


# ---------------------------------------------------------------------------
# Team / meeting docs
# ---------------------------------------------------------------------------


def test_meeting_notes_filename() -> None:
    doc_type, weight = classify_document_authority("meeting_notes.txt", "/team/meeting_notes.txt", "")
    assert doc_type == "team"
    assert weight == 60


def test_standup_in_filename() -> None:
    doc_type, weight = classify_document_authority("standup_2024-03-01.md", "/notes/standup_2024-03-01.md", "")
    assert doc_type == "team"
    assert weight == 60


def test_retro_in_path() -> None:
    doc_type, weight = classify_document_authority("notes.md", "/sprints/retro/notes.md", "")
    assert doc_type == "team"
    assert weight == 60


# ---------------------------------------------------------------------------
# Email detection
# ---------------------------------------------------------------------------


def test_email_content_heuristic() -> None:
    content = "From: alice@example.com\nTo: bob@example.com\nHi Bob, see attached."
    doc_type, weight = classify_document_authority("thread.txt", "/exports/thread.txt", content)
    assert doc_type == "email"
    assert weight == 50


# ---------------------------------------------------------------------------
# Management
# ---------------------------------------------------------------------------


def test_okr_in_content() -> None:
    content = "Q2 OKR review: we hit 80% of our key results this quarter."
    doc_type, weight = classify_document_authority("okr_q2.txt", "/docs/okr_q2.txt", content)
    assert doc_type == "management"
    assert weight == 70


def test_roadmap_in_filename() -> None:
    doc_type, weight = classify_document_authority("product_roadmap.md", "/strategy/product_roadmap.md", "")
    assert doc_type == "management"
    assert weight == 70


# ---------------------------------------------------------------------------
# Informal (very short content)
# ---------------------------------------------------------------------------


def test_informal_short_content() -> None:
    doc_type, weight = classify_document_authority("scratch.txt", "/notes/scratch.txt", "todo: fix the thing")
    assert doc_type == "informal"
    assert weight == 30


# ---------------------------------------------------------------------------
# General fallback
# ---------------------------------------------------------------------------


def test_general_fallback() -> None:
    # Craft content that is long enough (>= 300 chars) and avoids every
    # keyword pattern used by the classifier.
    content = (
        "The dashboard rendering engine uses a Python backend. "
        "Charts are drawn with a JavaScript library on the frontend. "
        "Data flows from the SQLite database through a REST endpoint. "
        "Authentication uses OAuth tokens stored in the browser session. "
        "The build toolchain compiles TypeScript and bundles assets with Vite. "
        "Deployment targets a Linux VPS running Nginx as a reverse proxy. "
        "The whole stack is containerised and orchestrated via Docker Compose."
    )
    assert len(content) >= 300
    doc_type, weight = classify_document_authority("dashboard_tech.txt", "/docs/dashboard_tech.txt", content)
    assert doc_type == "general"
    assert weight == 50


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------


def test_returns_tuple_of_str_and_int() -> None:
    result = classify_document_authority("foo.txt", "/foo.txt", "bar")
    assert isinstance(result, tuple)
    assert len(result) == 2
    doc_type, weight = result
    assert isinstance(doc_type, str)
    assert isinstance(weight, int)
    assert 0 <= weight <= 100
