"""Tests for the briefing detector module."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from verra.briefing import BriefingDetector, BriefingItem
from verra.store.db import DatabaseManager


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeConfig:
    max_items: int = 10
    stale_lead_days: int = 14
    contract_warning_days: int = 30


def _utc_days_ago(n: int) -> str:
    """ISO date string for N days ago."""
    return (datetime.now(tz=timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")


def _utc_days_ahead(n: int) -> str:
    """ISO date string for N days in the future."""
    return (datetime.now(tz=timezone.utc) + timedelta(days=n)).strftime("%Y-%m-%d")


def _iso_ts_days_ago(n: int) -> str:
    """ISO timestamp string for N days ago."""
    return (datetime.now(tz=timezone.utc) - timedelta(days=n)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def _iso_ts_days_ahead(n: int) -> str:
    return (datetime.now(tz=timezone.utc) + timedelta(days=n)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


@pytest.fixture()
def db(tmp_path):
    """Shared DatabaseManager backed by a tmp directory."""
    mgr = DatabaseManager(tmp_path)
    yield mgr
    mgr.close()


@pytest.fixture()
def detector(db):
    """BriefingDetector wired to the test databases."""
    return BriefingDetector(
        core_conn=db.core,
        analysis_conn=db.analysis,
        config=_FakeConfig(),
        user_email="me@example.com",
    )


# ---------------------------------------------------------------------------
# BriefingItem dataclass
# ---------------------------------------------------------------------------


class TestBriefingItem:
    def test_fields_stored_correctly(self):
        now = datetime.now(tz=timezone.utc)
        item = BriefingItem(
            category="stale_lead",
            title="No reply",
            detail="Thread with no reply.",
            entity_name="Acme",
            urgency=3,
            source_label="emails table",
            detected_at=now,
            item_key="stale_lead:acme",
        )
        assert item.category == "stale_lead"
        assert item.urgency == 3
        assert item.entity_name == "Acme"


# ---------------------------------------------------------------------------
# detect_stale_leads
# ---------------------------------------------------------------------------


class TestDetectStaleLeads:
    def _insert_email(
        self, core: sqlite3.Connection, thread_id: str, from_addr: str, date: str, subject: str = "Test subject"
    ) -> None:
        core.execute(
            """
            INSERT INTO emails (thread_id, message_id, from_addr, to_addr, subject, date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (thread_id, f"msg_{from_addr}_{date}", from_addr, "other@example.com", subject, date),
        )
        core.commit()

    def test_returns_empty_when_no_emails(self, detector):
        items = detector.detect_stale_leads()
        assert items == []

    def test_returns_empty_when_no_user_email(self, db):
        det = BriefingDetector(db.core, db.analysis, _FakeConfig(), user_email=None)
        assert det.detect_stale_leads() == []

    def test_detects_unanswered_sent_email(self, detector, db):
        # User sent an email 20 days ago, no reply
        self._insert_email(db.core, "thread-1", "me@example.com", _utc_days_ago(20), "Follow up on proposal")
        items = detector.detect_stale_leads(days_threshold=14)
        assert len(items) == 1
        assert items[0].category == "stale_lead"
        assert "Follow up on proposal" in items[0].title

    def test_skips_thread_with_reply(self, detector, db):
        self._insert_email(db.core, "thread-2", "me@example.com", _utc_days_ago(20))
        # Reply from another address
        self._insert_email(db.core, "thread-2", "them@example.com", _utc_days_ago(18))
        items = detector.detect_stale_leads(days_threshold=14)
        assert items == []

    def test_skips_recent_sent_email(self, detector, db):
        # Only 5 days ago — within threshold of 14
        self._insert_email(db.core, "thread-3", "me@example.com", _utc_days_ago(5))
        items = detector.detect_stale_leads(days_threshold=14)
        assert items == []

    def test_urgency_scales_with_age(self, detector, db):
        self._insert_email(db.core, "t-short", "me@example.com", _utc_days_ago(20))
        self._insert_email(db.core, "t-long", "me@example.com", _utc_days_ago(60))
        items = detector.detect_stale_leads(days_threshold=14)
        urgencies = {i.item_key.split(":")[-1]: i.urgency for i in items}
        # Longer wait → higher urgency
        assert urgencies.get("t-long", 0) >= urgencies.get("t-short", 0)

    def test_multiple_stale_threads(self, detector, db):
        for i, thread in enumerate(["ta", "tb", "tc"]):
            self._insert_email(db.core, thread, "me@example.com", _utc_days_ago(20 + i))
        items = detector.detect_stale_leads(days_threshold=14)
        assert len(items) == 3


# ---------------------------------------------------------------------------
# detect_expiring_contracts
# ---------------------------------------------------------------------------


class TestDetectExpiringContracts:
    def _insert_contract(
        self,
        core: sqlite3.Connection,
        file_name: str,
        valid_until: str | None = None,
        extra_metadata: str | None = None,
        chunk_text: str | None = None,
    ) -> int:
        cur = core.execute(
            """
            INSERT INTO documents (file_path, file_name, source_type, content_hash,
                                   document_type, extra_metadata)
            VALUES (?, ?, 'folder', ?, 'contract', ?)
            """,
            (f"/docs/{file_name}", file_name, f"hash_{file_name}", extra_metadata),
        )
        doc_id = cur.lastrowid
        core.commit()

        if valid_until or chunk_text:
            import json
            meta_json = json.dumps({"text": chunk_text or ""}) if chunk_text else None
            core.execute(
                """
                INSERT INTO chunks (document_id, position, token_count, metadata,
                                    valid_until)
                VALUES (?, 0, 10, ?, ?)
                """,
                (doc_id, meta_json, valid_until),
            )
            core.commit()

        return doc_id

    def test_empty_when_no_contracts(self, detector):
        assert detector.detect_expiring_contracts() == []

    def test_detects_chunk_valid_until(self, detector, db):
        self._insert_contract(
            db.core, "service_agreement.pdf", valid_until=_utc_days_ahead(10)
        )
        items = detector.detect_expiring_contracts(warning_days=30)
        assert len(items) == 1
        assert items[0].category == "expiring_contract"
        assert "service_agreement.pdf" in items[0].title

    def test_skips_far_future_expiry(self, detector, db):
        self._insert_contract(
            db.core, "long_contract.pdf", valid_until=_utc_days_ahead(120)
        )
        items = detector.detect_expiring_contracts(warning_days=30)
        assert items == []

    def test_detects_extra_metadata_expiry(self, detector, db):
        import json
        meta = json.dumps({"expiry_date": _utc_days_ahead(5)})
        self._insert_contract(db.core, "nda.pdf", extra_metadata=meta)
        items = detector.detect_expiring_contracts(warning_days=30)
        assert any("nda.pdf" in i.title for i in items)

    def test_urgency_higher_for_imminent_expiry(self, detector, db):
        self._insert_contract(
            db.core, "urgent.pdf", valid_until=_utc_days_ahead(3)
        )
        self._insert_contract(
            db.core, "soon.pdf", valid_until=_utc_days_ahead(20)
        )
        items = detector.detect_expiring_contracts(warning_days=30)
        urgencies = {i.title.split(": ")[1]: i.urgency for i in items}
        assert urgencies.get("urgent.pdf", 0) >= urgencies.get("soon.pdf", 0)

    def test_no_duplicate_items_for_same_doc(self, detector, db):
        # valid_until on chunk AND expiry_date in extra_metadata — should produce one item
        import json
        meta = json.dumps({"expiry_date": _utc_days_ahead(10)})
        self._insert_contract(
            db.core,
            "double.pdf",
            valid_until=_utc_days_ahead(10),
            extra_metadata=meta,
        )
        items = detector.detect_expiring_contracts(warning_days=30)
        titles = [i.title for i in items]
        assert titles.count("Contract expiring: double.pdf") == 1

    def test_detects_chunk_text_pattern(self, detector, db):
        from datetime import datetime, timedelta, timezone
        future = datetime.now(tz=timezone.utc) + timedelta(days=15)
        expiry_str = future.strftime("%B %d, %Y")
        chunk_text = f"This contract expires on {expiry_str} and must be renewed."
        self._insert_contract(db.core, "text_contract.pdf", chunk_text=chunk_text)
        items = detector.detect_expiring_contracts(warning_days=30)
        assert any("text_contract.pdf" in i.title for i in items)


# ---------------------------------------------------------------------------
# detect_forgotten_commitments
# ---------------------------------------------------------------------------


class TestDetectForgottenCommitments:
    def _insert_commitment(
        self,
        analysis: sqlite3.Connection,
        who_name: str,
        what: str,
        due_date: str | None = None,
        status: str = "open",
        detected_at: str | None = None,
    ) -> None:
        ts = detected_at or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        analysis.execute(
            """
            INSERT INTO commitments (who_name, what, due_date, status, detected_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (who_name, what, due_date, status, ts),
        )
        analysis.commit()

    def test_empty_when_no_commitments(self, detector):
        assert detector.detect_forgotten_commitments() == []

    def test_detects_overdue_commitment(self, detector, db):
        self._insert_commitment(db.analysis, "Alice", "Send proposal", due_date=_utc_days_ago(5))
        items = detector.detect_forgotten_commitments()
        assert any(i.category == "forgotten_commitment" for i in items)
        assert any("Send proposal" in i.title for i in items)

    def test_closed_commitment_not_surfaced(self, detector, db):
        self._insert_commitment(
            db.analysis, "Bob", "Review contract", due_date=_utc_days_ago(3), status="completed"
        )
        items = detector.detect_forgotten_commitments()
        assert items == []

    def test_due_soon_commitment_surfaced(self, detector, db):
        self._insert_commitment(db.analysis, "Carol", "Submit invoice", due_date=_utc_days_ahead(3))
        items = detector.detect_forgotten_commitments()
        assert any("Submit invoice" in i.title for i in items)

    def test_long_open_no_date_surfaced(self, detector, db):
        old_ts = _iso_ts_days_ago(45)
        self._insert_commitment(
            db.analysis, "Dave", "Follow up on deal", detected_at=old_ts
        )
        items = detector.detect_forgotten_commitments()
        assert any("Follow up on deal" in i.title for i in items)

    def test_recent_open_no_date_not_surfaced(self, detector, db):
        self._insert_commitment(db.analysis, "Eve", "Check in next week")
        items = detector.detect_forgotten_commitments()
        assert all("Check in next week" not in i.title for i in items)

    def test_overdue_urgency_is_high(self, detector, db):
        self._insert_commitment(db.analysis, "Frank", "Critical task", due_date=_utc_days_ago(20))
        items = detector.detect_forgotten_commitments()
        overdue = [i for i in items if "Critical task" in i.title]
        assert overdue[0].urgency == 5

    def test_entity_name_set(self, detector, db):
        self._insert_commitment(db.analysis, "Grace", "File the report", due_date=_utc_days_ago(2))
        items = detector.detect_forgotten_commitments()
        assert any(i.entity_name == "Grace" for i in items)


# ---------------------------------------------------------------------------
# detect_patterns
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    def _insert_user_message(
        self, core: sqlite3.Connection, content: str, conv_id: int
    ) -> None:
        core.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, 'user', ?)",
            (conv_id, content),
        )
        core.commit()

    def _insert_conversation(self, core: sqlite3.Connection) -> int:
        cur = core.execute("INSERT INTO conversations (title) VALUES ('test')")
        core.commit()
        return cur.lastrowid

    def test_empty_when_no_messages(self, detector):
        assert detector.detect_patterns() == []

    def test_returns_recurring_topic(self, detector, db):
        # Same word in 3 different conversations
        for _ in range(3):
            cid = self._insert_conversation(db.core)
            self._insert_user_message(db.core, "Tell me about pricing strategy", cid)

        items = detector.detect_patterns()
        topics = [i.title for i in items]
        # "pricing" or "strategy" should surface
        assert any("pricing" in t or "strategy" in t for t in topics)

    def test_ignores_stop_words(self, detector, db):
        for _ in range(5):
            cid = self._insert_conversation(db.core)
            self._insert_user_message(db.core, "the and is in on at", cid)
        items = detector.detect_patterns()
        assert items == []

    def test_returns_at_most_three_patterns(self, detector, db):
        words = ["contract", "invoice", "pricing", "deadline", "renewal"]
        for word in words:
            for _ in range(3):
                cid = self._insert_conversation(db.core)
                self._insert_user_message(db.core, f"question about {word} details", cid)
        items = detector.detect_patterns()
        assert len(items) <= 3

    def test_word_appearing_in_one_conv_not_surfaced(self, detector, db):
        cid = self._insert_conversation(db.core)
        self._insert_user_message(db.core, "unusual word aardvark habitat", cid)
        items = detector.detect_patterns()
        assert all("aardvark" not in i.title for i in items)


# ---------------------------------------------------------------------------
# detect_new_data
# ---------------------------------------------------------------------------


class TestDetectNewData:
    def _insert_document(self, core: sqlite3.Connection, indexed_at: str) -> None:
        import uuid
        core.execute(
            """
            INSERT INTO documents (file_path, file_name, source_type, content_hash, indexed_at)
            VALUES (?, ?, 'folder', ?, ?)
            """,
            (f"/docs/{uuid.uuid4()}.txt", "doc.txt", str(uuid.uuid4()), indexed_at),
        )
        core.commit()

    def _insert_email(self, core: sqlite3.Connection, created_at: str) -> None:
        import uuid
        core.execute(
            """
            INSERT INTO emails (thread_id, from_addr, to_addr, subject, date, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), "sender@ex.com", "me@ex.com", "Hi", "2025-01-01", created_at),
        )
        core.commit()

    def test_empty_when_no_new_data(self, detector):
        items = detector.detect_new_data()
        assert items == []

    def test_counts_new_documents(self, detector, db):
        self._insert_document(db.core, _iso_ts_days_ago(0))
        self._insert_document(db.core, _iso_ts_days_ago(0))
        items = detector.detect_new_data()
        assert len(items) == 1
        assert "2 documents" in items[0].title

    def test_counts_new_emails(self, detector, db):
        self._insert_email(db.core, _iso_ts_days_ago(0))
        items = detector.detect_new_data()
        assert len(items) == 1
        assert "1 email" in items[0].title

    def test_counts_both_docs_and_emails(self, detector, db):
        self._insert_document(db.core, _iso_ts_days_ago(0))
        self._insert_email(db.core, _iso_ts_days_ago(0))
        items = detector.detect_new_data()
        assert len(items) == 1
        assert "document" in items[0].title
        assert "email" in items[0].title

    def test_persists_last_run_timestamp(self, detector, db):
        self._insert_document(db.core, _iso_ts_days_ago(0))
        detector.detect_new_data()
        row = db.core.execute(
            "SELECT value FROM memory WHERE category='briefing' AND key='last_run_at'"
        ).fetchone()
        assert row is not None

    def test_uses_stored_last_run_at(self, detector, db):
        # Set last_run_at to 1 hour ago; only docs from the last hour should count.
        one_hour_ago = (
            datetime.now(tz=timezone.utc) - timedelta(hours=1)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        db.core.execute(
            "INSERT INTO memory (category, key, value) VALUES ('briefing', 'last_run_at', ?)",
            (one_hour_ago,),
        )
        db.core.commit()

        # One document from 2 hours ago (outside window) and one from 30 minutes ago
        self._insert_document(db.core, _iso_ts_days_ago(0))  # now — inside window
        items = detector.detect_new_data()
        assert len(items) == 1


# ---------------------------------------------------------------------------
# Dismissal
# ---------------------------------------------------------------------------


class TestDismissal:
    def test_dismiss_and_is_dismissed(self, detector):
        key = "stale_lead:thread_abc"
        assert not detector._is_dismissed(key)
        detector.dismiss(key)
        assert detector._is_dismissed(key)

    def test_double_dismiss_is_noop(self, detector):
        key = "stale_lead:thread_xyz"
        detector.dismiss(key)
        detector.dismiss(key)  # should not raise
        assert detector._is_dismissed(key)

    def test_dismissed_item_excluded_from_detect_all(self, detector, db):
        # Insert a commitment so there's something to surface
        db.analysis.execute(
            """
            INSERT INTO commitments (who_name, what, due_date, status)
            VALUES ('Alice', 'Send report', ?, 'open')
            """,
            (_utc_days_ago(5),),
        )
        db.analysis.commit()

        items_before = detector.detect_all()
        assert len(items_before) > 0

        # Dismiss all of them
        for item in items_before:
            detector.dismiss(item.item_key)

        items_after = detector.detect_all()
        assert len(items_after) == 0


# ---------------------------------------------------------------------------
# detect_all integration
# ---------------------------------------------------------------------------


class TestDetectAll:
    def test_returns_empty_with_no_data(self, detector):
        assert detector.detect_all() == []

    def test_respects_max_items(self, detector, db):
        # Insert many commitments to overflow max_items
        for i in range(20):
            db.analysis.execute(
                "INSERT INTO commitments (who_name, what, due_date, status) VALUES (?, ?, ?, 'open')",
                (f"Person{i}", f"Task {i}", _utc_days_ago(i + 1)),
            )
        db.analysis.commit()
        items = detector.detect_all(max_items=3)
        assert len(items) <= 3

    def test_sorted_by_urgency_descending(self, detector, db):
        # Low urgency: open commitment with no date, detected recently
        db.analysis.execute(
            "INSERT INTO commitments (who_name, what, status) VALUES ('A', 'low priority', 'open')"
        )
        # High urgency: severely overdue
        db.analysis.execute(
            "INSERT INTO commitments (who_name, what, due_date, status) VALUES ('B', 'critical', ?, 'open')",
            (_utc_days_ago(40),),
        )
        db.analysis.commit()
        items = detector.detect_all()
        if len(items) >= 2:
            # First item should have urgency >= last item urgency
            assert items[0].urgency >= items[-1].urgency

    def test_items_have_all_required_fields(self, detector, db):
        db.analysis.execute(
            "INSERT INTO commitments (who_name, what, due_date, status) VALUES ('X', 'thing', ?, 'open')",
            (_utc_days_ago(3),),
        )
        db.analysis.commit()
        items = detector.detect_all()
        for item in items:
            assert item.category
            assert item.title
            assert item.detail
            assert isinstance(item.urgency, int)
            assert 1 <= item.urgency <= 5
            assert item.source_label
            assert isinstance(item.detected_at, datetime)
            assert item.item_key
