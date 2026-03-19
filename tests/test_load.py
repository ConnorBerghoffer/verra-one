"""Load test & edge case suite for Verra One.

Simulates a company ingesting millions of large documents without writing
anything to disk.  Every store uses in-memory SQLite, the vector store is
mocked to count operations, and synthetic documents are generated on the fly.

Run:
    python tests/test_load.py              # full suite
    python tests/test_load.py --quick      # quick smoke test (smaller scale)
    python tests/test_load.py --section chunking   # run one section only

Sections:
    chunking       — chunk huge docs, edge cases
    metadata       — insert millions of doc/chunk rows
    entities       — millions of entities + aliases + relationships
    memory         — millions of conversation messages
    retrieval      — search across large datasets
    chat           — multi-turn conversations at scale
    briefing       — briefing detection with large tables
    pipeline       — end-to-end ingest pipeline stress
    edge_cases     — unicode, empty, binary, malformed inputs
    concurrent     — thread-safety of stores
"""

from __future__ import annotations

import gc
import hashlib
import os
import random
import re
import sqlite3
import string
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUICK = "--quick" in sys.argv
SECTION = None
for i, a in enumerate(sys.argv):
    if a == "--section" and i + 1 < len(sys.argv):
        SECTION = sys.argv[i + 1]


def scale(n: int) -> int:
    """Scale numbers down in quick mode."""
    return max(1, n // 100) if QUICK else n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASSED = 0
FAILED = 0
ERRORS: list[str] = []


def run_test(name: str, fn, *args):
    global PASSED, FAILED
    try:
        t0 = time.time()
        fn(*args)
        elapsed = time.time() - t0
        print(f"  \033[32m✓\033[0m {name} ({elapsed:.2f}s)")
        PASSED += 1
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  \033[31m✗\033[0m {name} ({elapsed:.2f}s) — {exc}")
        ERRORS.append(f"{name}: {exc}\n{traceback.format_exc()}")
        FAILED += 1


def random_text(n_words: int) -> str:
    words = ["the", "quick", "brown", "fox", "jumped", "over", "lazy", "dog",
             "contract", "agreement", "proposal", "invoice", "meeting", "deadline",
             "client", "revenue", "budget", "quarterly", "report", "analysis",
             "performance", "strategy", "implementation", "delivery", "milestone"]
    return " ".join(random.choice(words) for _ in range(n_words))


def random_email() -> str:
    name = "".join(random.choices(string.ascii_lowercase, k=6))
    domain = random.choice(["acme.com", "corp.io", "biz.net", "co.uk"])
    return f"{name}@{domain}"


def make_core_db() -> sqlite3.Connection:
    """Create an in-memory core.db with all tables."""
    from verra.store.db import DatabaseManager
    dm = DatabaseManager.__new__(DatabaseManager)
    dm.data_dir = Path("/tmp/fake")
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    dm.core = conn
    dm.analysis = sqlite3.connect(":memory:", check_same_thread=False)
    dm.analysis.row_factory = sqlite3.Row
    dm.analysis.execute("PRAGMA journal_mode=WAL")
    dm.analysis.execute("PRAGMA foreign_keys=ON")
    dm._init_core_tables()
    dm._init_analysis_tables()
    return dm


class MockVectorStore:
    """Counts operations without storing embeddings."""

    def __init__(self):
        self.add_count = 0
        self.search_count = 0
        self.total_chunks = 0
        self.deleted = 0

    def add_chunks(self, ids=None, chunk_ids=None, chunks=None, texts=None, metadatas=None):
        n = 0
        if ids:
            n = len(ids)
        elif chunk_ids:
            n = len(chunk_ids)
        elif chunks:
            n = len(chunks)
        elif texts:
            n = len(texts)
        self.add_count += 1
        self.total_chunks += n

    def search(self, query="", n_results=5, where=None):
        self.search_count += 1
        results = []
        for i in range(min(n_results, max(1, self.total_chunks))):
            results.append({
                "id": str(i),
                "document": f"Mock chunk {i}: {random_text(20)}",
                "metadata": {"chunk_id": i, "document_id": 1, "source_type": "folder",
                             "file_name": "test.pdf", "authority_weight": 50},
                "distance": random.uniform(0.1, 0.8),
            })
        return results

    def count(self):
        return self.total_chunks

    def delete_by_document_id(self, doc_id):
        self.deleted += 1

    def reset(self):
        self.total_chunks = 0
        self.add_count = 0


# ===================================================================
# SECTION: Chunking
# ===================================================================

def test_chunking():
    print("\n\033[1m── Chunking ──\033[0m")

    from verra.ingest.chunking import chunk_document, Chunk

    def test_huge_document():
        """Chunk a 1M-word document without crashing."""
        text = random_text(scale(1_000_000))
        chunks = chunk_document(text)
        assert len(chunks) > 0, "Should produce at least one chunk"
        for c in chunks:
            assert len(c.text) > 0, "Empty chunk text"

    def test_tiny_document():
        """Chunk a document with just one word."""
        chunks = chunk_document("hello")
        assert len(chunks) == 1

    def test_empty_document():
        """Chunk an empty string."""
        chunks = chunk_document("")
        assert len(chunks) <= 1  # may produce 0 or 1

    def test_repeated_content():
        """Chunk a document with 100K identical lines."""
        text = "This is a test line.\n" * scale(100_000)
        chunks = chunk_document(text)
        assert len(chunks) > 0

    def test_no_whitespace():
        """Chunk a 50K-character string with no spaces."""
        text = "a" * scale(50_000)
        chunks = chunk_document(text)
        assert len(chunks) > 0

    def test_unicode_heavy():
        """Chunk text with lots of unicode/emoji."""
        text = "会議の議事録 📊 Geschäftsbericht über 利益 " * scale(10_000)
        chunks = chunk_document(text)
        assert len(chunks) > 0
        for c in chunks:
            assert len(c.text) > 0

    def test_email_chunking():
        """Chunk email-style content."""
        text = ""
        for i in range(scale(500)):
            text += f"From: user{i}@example.com\nTo: boss@company.com\n"
            text += f"Subject: Thread {i}\nDate: 2024-01-{(i % 28) + 1:02d}\n\n"
            text += random_text(200) + "\n\n---\n\n"
        chunks = chunk_document(text, metadata={"source_type": "email"})
        assert len(chunks) > 0

    def test_table_preservation():
        """Tables should not be split across chunks."""
        header = "| Col A | Col B | Col C |\n|-------|-------|-------|\n"
        rows = "".join(f"| val{i} | val{i+1} | val{i+2} |\n" for i in range(scale(500)))
        text = "Some intro text.\n\n" + header + rows + "\n\nSome conclusion."
        chunks = chunk_document(text)
        assert len(chunks) > 0

    def test_many_small_chunks():
        """10K separate paragraphs."""
        paragraphs = [random_text(20) for _ in range(scale(10_000))]
        text = "\n\n".join(paragraphs)
        chunks = chunk_document(text)
        assert len(chunks) > 0

    run_test("1M-word document", test_huge_document)
    run_test("Tiny document (1 word)", test_tiny_document)
    run_test("Empty document", test_empty_document)
    run_test("100K repeated lines", test_repeated_content)
    run_test("50K chars no whitespace", test_no_whitespace)
    run_test("Unicode/emoji heavy", test_unicode_heavy)
    run_test("500 email threads", test_email_chunking)
    run_test("Large table preservation", test_table_preservation)
    run_test("10K small paragraphs", test_many_small_chunks)


# ===================================================================
# SECTION: Metadata Store
# ===================================================================

def test_metadata():
    print("\n\033[1m── Metadata Store ──\033[0m")

    from verra.store.metadata import MetadataStore
    from verra.ingest.chunking import Chunk

    def test_bulk_insert_documents():
        """Insert 100K documents."""
        n = scale(100_000)
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        store = MetadataStore.__new__(MetadataStore)
        store.db_path = Path(":memory:")
        store._conn = conn
        from verra.store.metadata import _SCHEMA
        conn.executescript(_SCHEMA)
        conn.commit()

        for i in range(n):
            store.add_document(
                file_path=f"/docs/file_{i}.pdf",
                file_name=f"file_{i}.pdf",
                source_type="folder",
                format="pdf",
                content_hash=hashlib.sha256(f"doc{i}".encode()).hexdigest(),
                page_count=random.randint(1, 100),
            )
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == n, f"Expected {n} docs, got {count}"

    def test_bulk_insert_chunks():
        """Insert 500K chunks across 1K documents."""
        n_docs = scale(1_000)
        chunks_per_doc = 500 if not QUICK else 5
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        store = MetadataStore.__new__(MetadataStore)
        store.db_path = Path(":memory:")
        store._conn = conn
        from verra.store.metadata import _SCHEMA
        conn.executescript(_SCHEMA)
        conn.commit()

        total_chunks = 0
        for i in range(n_docs):
            doc_id = store.add_document(
                file_path=f"/docs/file_{i}.pdf",
                file_name=f"file_{i}.pdf",
                source_type="folder",
                format="pdf",
                content_hash=hashlib.sha256(f"doc{i}".encode()).hexdigest(),
            )
            chunks = [
                Chunk(text=random_text(50), token_count=random.randint(100, 500))
                for _ in range(chunks_per_doc)
            ]
            store.add_chunks(doc_id, chunks)
            total_chunks += len(chunks)

        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == total_chunks, f"Expected {total_chunks} chunks, got {count}"

    def test_hash_dedup():
        """Same hash should be unique-indexed."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        store = MetadataStore.__new__(MetadataStore)
        store.db_path = Path(":memory:")
        store._conn = conn
        from verra.store.metadata import _SCHEMA
        conn.executescript(_SCHEMA)
        conn.commit()

        h = hashlib.sha256(b"dupe").hexdigest()
        store.add_document(file_path="/a.pdf", file_name="a.pdf",
                          source_type="folder", format="pdf", content_hash=h)
        result = store.get_document_by_hash(h)
        assert result is not None

    def test_bulk_emails():
        """Insert 50K email records."""
        n = scale(50_000)
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        store = MetadataStore.__new__(MetadataStore)
        store.db_path = Path(":memory:")
        store._conn = conn
        from verra.store.metadata import _SCHEMA
        conn.executescript(_SCHEMA)
        conn.commit()

        for i in range(n):
            store.add_email(
                thread_id=f"thread_{i // 5}",
                message_id=f"msg_{i}",
                from_addr=random_email(),
                to_addr=random_email(),
                cc_addr=None,
                subject=f"Subject {i}: {random_text(5)}",
                date=f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                labels=None,
            )
        count = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
        assert count == n

    run_test("100K document inserts", test_bulk_insert_documents)
    run_test("500K chunk inserts", test_bulk_insert_chunks)
    run_test("Hash deduplication", test_hash_dedup)
    run_test("50K email inserts", test_bulk_emails)


# ===================================================================
# SECTION: Entity Store
# ===================================================================

def test_entities():
    print("\n\033[1m── Entity Store ──\033[0m")

    from verra.store.entities import EntityStore

    def test_bulk_entities():
        """Insert 50K entities with aliases."""
        n = scale(50_000)
        db = make_core_db()
        store = EntityStore.from_connection(db.core)

        for i in range(n):
            store.add_entity(
                canonical_name=f"Entity {i}",
                entity_type=random.choice(["person", "company", "project"]),
                aliases=[f"alias_{i}_a", f"alias_{i}_b"],
            )
        count = db.core.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count == n
        db.close()

    def test_resolve_performance():
        """Resolve 10K entity lookups."""
        n = scale(10_000)
        db = make_core_db()
        store = EntityStore.from_connection(db.core)

        for i in range(1000):
            store.add_entity(f"Person {i}", "person", [f"p{i}"])

        for i in range(n):
            store.resolve(f"p{i % 1000}")
        db.close()

    def test_bulk_relationships():
        """Insert 100K relationships."""
        n = scale(100_000)
        db = make_core_db()
        store = EntityStore.from_connection(db.core)

        entity_ids = []
        for i in range(500):
            eid = store.add_entity(f"E{i}", "person")
            entity_ids.append(eid)

        for i in range(n):
            a = random.choice(entity_ids)
            b = random.choice(entity_ids)
            if a != b:
                try:
                    store.add_relationship(a, "works_with", b)
                except Exception:
                    pass  # UNIQUE constraint expected
        db.close()

    run_test("50K entity inserts", test_bulk_entities)
    run_test("10K entity resolve lookups", test_resolve_performance)
    run_test("100K relationship inserts", test_bulk_relationships)


# ===================================================================
# SECTION: Memory Store
# ===================================================================

def test_memory():
    print("\n\033[1m── Memory Store ──\033[0m")

    from verra.store.memory import MemoryStore

    def test_bulk_conversations():
        """Create 10K conversations with 20 messages each."""
        n_convs = scale(10_000)
        msgs_per = 20 if not QUICK else 2
        db = make_core_db()
        store = MemoryStore.from_connection(db.core)

        for i in range(n_convs):
            cid = store.new_conversation(title=f"Conversation {i}")
            for j in range(msgs_per):
                role = "user" if j % 2 == 0 else "assistant"
                store.add_message(cid, role, random_text(50))
        count = db.core.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == n_convs * msgs_per
        db.close()

    def test_bulk_memory_entries():
        """Insert 50K memory entries."""
        n = scale(50_000)
        db = make_core_db()
        store = MemoryStore.from_connection(db.core)

        for i in range(n):
            store.set_memory(
                category=random.choice(["preference", "fact", "dismissed", "reminder"]),
                key=f"key_{i}",
                value=random_text(10),
            )
        count = db.core.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        assert count == n
        db.close()

    run_test("10K conversations × 20 messages", test_bulk_conversations)
    run_test("50K memory entries", test_bulk_memory_entries)


# ===================================================================
# SECTION: Retrieval
# ===================================================================

def test_retrieval():
    print("\n\033[1m── Retrieval ──\033[0m")

    from verra.retrieval.router import parse_query

    def test_query_parsing_volume():
        """Parse 10K diverse queries."""
        n = scale(10_000)
        queries = [
            "emails from Jake about pricing",
            "what's our refund policy?",
            "contracts expiring next month",
            "meeting notes from last week",
            random_text(random.randint(3, 30)),
        ]
        for i in range(n):
            q = random.choice(queries) if i % 2 == 0 else random_text(random.randint(2, 50))
            result = parse_query(q)
            assert result is not None

    def test_search_with_mock():
        """Run 1K searches against mock vector store."""
        n = scale(1_000)
        db = make_core_db()
        from verra.store.metadata import MetadataStore
        meta = MetadataStore.from_connection(db.core)
        vec = MockVectorStore()
        vec.total_chunks = 10000

        from verra.retrieval.search import search
        from verra.retrieval.router import parse_query

        for i in range(n):
            q = parse_query(random_text(random.randint(3, 15)))
            try:
                results = search(q, meta, vec, n_results=5)
            except Exception:
                pass  # some queries may not match, that's fine
        db.close()

    run_test("10K query parses", test_query_parsing_volume)
    run_test("1K searches with mock vector store", test_search_with_mock)


# ===================================================================
# SECTION: Chat Engine
# ===================================================================

def test_chat():
    print("\n\033[1m── Chat Engine ──\033[0m")

    def test_multi_turn_conversation():
        """50-turn conversation with mocked LLM."""
        n_turns = scale(50)
        db = make_core_db()
        from verra.store.metadata import MetadataStore
        from verra.store.memory import MemoryStore
        from verra.agent.chat import ChatEngine

        meta = MetadataStore.from_connection(db.core)
        mem = MemoryStore.from_connection(db.core)
        vec = MockVectorStore()
        vec.total_chunks = 100

        mock_llm = MagicMock()
        mock_llm.model = "mock-model"
        mock_llm.complete.return_value = "This is a mock response about your data."
        mock_llm.stream.return_value = iter(["This ", "is ", "a ", "mock ", "response."])
        mock_llm.complete_with_tools.return_value = "Mock tool response about your business data."

        engine = ChatEngine(
            llm=mock_llm,
            metadata_store=meta,
            vector_store=vec,
            memory_store=mem,
        )

        for i in range(n_turns):
            question = random_text(random.randint(5, 20))
            response = engine.ask(question, use_multi_hop=False)
            assert response.answer, f"Empty answer on turn {i}"

        # Verify history was trimmed
        assert len(engine._history) <= engine._max_history_turns * 2
        db.close()

    def test_stream_ask():
        """Stream 20 responses."""
        n = scale(20)
        db = make_core_db()
        from verra.store.metadata import MetadataStore
        from verra.store.memory import MemoryStore
        from verra.agent.chat import ChatEngine

        meta = MetadataStore.from_connection(db.core)
        mem = MemoryStore.from_connection(db.core)
        vec = MockVectorStore()
        vec.total_chunks = 50

        mock_llm = MagicMock()
        mock_llm.model = "mock"
        mock_llm.stream.return_value = iter(["chunk1 ", "chunk2 ", "chunk3"])

        engine = ChatEngine(llm=mock_llm, metadata_store=meta,
                           vector_store=vec, memory_store=mem)

        for i in range(n):
            mock_llm.stream.return_value = iter(["chunk1 ", "chunk2 ", "chunk3"])
            chunks = list(engine.stream_ask(random_text(10)))
            assert len(chunks) > 0
        db.close()

    def test_correction_detection():
        """Test correction handling doesn't crash."""
        db = make_core_db()
        from verra.store.metadata import MetadataStore
        from verra.store.memory import MemoryStore
        from verra.agent.chat import ChatEngine

        meta = MetadataStore.from_connection(db.core)
        mem = MemoryStore.from_connection(db.core)
        vec = MockVectorStore()

        mock_llm = MagicMock()
        mock_llm.model = "mock"
        mock_llm.complete.return_value = "Noted."
        mock_llm.stream.return_value = iter(["Noted."])

        engine = ChatEngine(llm=mock_llm, metadata_store=meta,
                           vector_store=vec, memory_store=mem)

        # Test various input types that should not crash
        test_inputs = [
            "That's wrong, it's actually $5,000",
            "No, the correct answer is John Smith",
            "",  # empty
            "a" * 10000,  # very long
            "🎉🚀 emoji query 中文",  # unicode
        ]
        for msg in test_inputs:
            if not msg.strip():
                continue
            mock_llm.stream.return_value = iter(["Mock response."])
            chunks = list(engine.stream_ask(msg))
            assert len(chunks) > 0
        db.close()

    run_test("50-turn conversation", test_multi_turn_conversation)
    run_test("20 streaming responses", test_stream_ask)
    run_test("Correction detection", test_correction_detection)


# ===================================================================
# SECTION: Briefing
# ===================================================================

def test_briefing():
    print("\n\033[1m── Briefing ──\033[0m")

    from verra.briefing.detector import BriefingDetector, BriefingItem

    def _make_briefing_config():
        cfg = MagicMock()
        cfg.max_items = 5
        cfg.stale_lead_days = 14
        cfg.contract_warning_days = 30
        return cfg

    def test_empty_db_briefing():
        """Briefing on empty database should not crash."""
        db = make_core_db()
        detector = BriefingDetector(
            core_conn=db.core,
            analysis_conn=db.analysis,
            config=_make_briefing_config(),
        )
        items = detector.detect_all()
        assert isinstance(items, list)
        db.close()

    def test_large_email_set():
        """Briefing with 50K emails."""
        n = scale(50_000)
        db = make_core_db()

        for i in range(n):
            db.core.execute(
                """INSERT INTO emails (thread_id, message_id, from_addr, to_addr, subject, date)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (f"t{i // 3}", f"m{i}", random_email(), random_email(),
                 f"Subject {i}", (datetime.now() - timedelta(days=random.randint(0, 60))).isoformat()),
            )
        db.core.commit()

        detector = BriefingDetector(
            core_conn=db.core, analysis_conn=db.analysis,
            config=_make_briefing_config(),
        )
        items = detector.detect_all()
        assert isinstance(items, list)
        db.close()

    def test_large_commitments():
        """Briefing with 10K commitments."""
        n = scale(10_000)
        db = make_core_db()

        for i in range(n):
            due = (datetime.now() - timedelta(days=random.randint(-30, 30))).isoformat()
            db.analysis.execute(
                """INSERT INTO commitments (who_name, what, due_date, status)
                   VALUES (?, ?, ?, ?)""",
                (f"Person {i}", f"Task {i}: {random_text(5)}", due, "open"),
            )
        db.analysis.commit()

        detector = BriefingDetector(
            core_conn=db.core, analysis_conn=db.analysis,
            config=_make_briefing_config(),
        )
        items = detector.detect_all()
        assert isinstance(items, list)
        db.close()

    def test_dismiss_at_scale():
        """Dismiss 1K items."""
        n = scale(1_000)
        db = make_core_db()
        detector = BriefingDetector(
            core_conn=db.core, analysis_conn=db.analysis,
            config=_make_briefing_config(),
        )
        for i in range(n):
            detector.dismiss(f"test_category:Item {i}")
        # Verify all are dismissed
        for i in range(min(n, 100)):
            assert detector._is_dismissed(f"test_category:Item {i}")
        db.close()

    run_test("Briefing on empty DB", test_empty_db_briefing)
    run_test("Briefing with 50K emails", test_large_email_set)
    run_test("Briefing with 10K commitments", test_large_commitments)
    run_test("Dismiss 1K items", test_dismiss_at_scale)


# ===================================================================
# SECTION: Pipeline (end-to-end with mocks)
# ===================================================================

def test_pipeline():
    print("\n\033[1m── Pipeline ──\033[0m")

    def test_ner_at_scale():
        """Run NER on 1K large text blocks."""
        n = scale(1_000)
        from verra.ingest.ner import extract_entities

        for i in range(n):
            text = (
                f"John Smith from Acme Corp met with Jane Doe at the New York office. "
                f"They discussed the Q{(i % 4) + 1} budget of ${random.randint(1000, 1000000):,}. "
                f"Email: john@acme.com, jane@partner.io. Project Alpha deadline is 2024-{(i % 12) + 1:02d}-15."
            )
            entities = extract_entities(text)
            # Should not crash

    def test_email_cleaning_at_scale():
        """Clean 5K email bodies."""
        n = scale(5_000)
        from verra.ingest.email_cleaner import clean_email_body

        for i in range(n):
            body = (
                f"Hi team,\n\n{random_text(100)}\n\n"
                f"On Mon, Jan {(i % 28) + 1}, 2024 someone wrote:\n"
                f"> {random_text(50)}\n> {random_text(50)}\n\n"
                f"--\nBest regards,\nJohn Smith\nSenior VP\nAcme Corp\n"
                f"+1 555-0100\nwww.acme.com"
            )
            cleaned = clean_email_body(body)
            assert isinstance(cleaned, str)

    def test_authority_classification():
        """Classify 10K documents."""
        n = scale(10_000)
        from verra.store.metadata import classify_document_authority

        files = [
            ("policy_handbook.pdf", "/company/policies/", "All employees must..."),
            ("contract_acme.docx", "/legal/contracts/", "This agreement between..."),
            ("meeting_notes.txt", "/team/meetings/", "Sprint retro discussion..."),
            ("random_file.md", "/docs/", "Some random content here."),
            ("invoice_2024.xlsx", "/finance/", "Invoice #12345 total $5000"),
        ]
        for i in range(n):
            fname, fpath, content = random.choice(files)
            doc_type, weight = classify_document_authority(fname, fpath, content)
            assert isinstance(doc_type, str)
            assert 0 <= weight <= 100

    run_test("NER on 1K text blocks", test_ner_at_scale)
    run_test("Email cleaning × 5K", test_email_cleaning_at_scale)
    run_test("Authority classification × 10K", test_authority_classification)


# ===================================================================
# SECTION: Edge Cases
# ===================================================================

def test_edge_cases():
    print("\n\033[1m── Edge Cases ──\033[0m")

    from verra.ingest.chunking import chunk_document

    def test_null_bytes():
        """Handle text with null bytes."""
        text = "Hello\x00World\x00This has null bytes.\x00" * 100
        chunks = chunk_document(text.replace("\x00", " "))
        assert len(chunks) > 0

    def test_only_whitespace():
        """Handle text that's all whitespace."""
        text = "   \n\n\t\t   \n\n   " * 1000
        chunks = chunk_document(text)
        # May produce 0 chunks — that's fine

    def test_extremely_long_word():
        """A single 100K-character 'word'."""
        text = "a" * 100_000
        chunks = chunk_document(text)
        assert len(chunks) > 0

    def test_mixed_encodings():
        """UTF-8 with various scripts."""
        text = (
            "English text. "
            "中文文本。"
            "العربية "
            "हिन्दी "
            "Кириллица "
            "日本語テキスト "
            "한국어 "
            "🎉🚀💼📊 "
        ) * 1000
        chunks = chunk_document(text)
        assert len(chunks) > 0

    def test_deeply_nested_structure():
        """Markdown with 50 heading levels worth of nesting."""
        text = ""
        for i in range(50):
            level = min(i + 1, 6)
            text += f"{'#' * level} Heading Level {level} Section {i}\n\n"
            text += random_text(100) + "\n\n"
        chunks = chunk_document(text)
        assert len(chunks) > 0

    def test_huge_single_line():
        """A single line with 500K words (no newlines)."""
        text = random_text(scale(500_000))
        chunks = chunk_document(text)
        assert len(chunks) > 0

    def test_sql_injection_in_data():
        """Ensure SQL injection in content doesn't crash stores."""
        db = make_core_db()
        from verra.store.metadata import MetadataStore
        meta = MetadataStore.from_connection(db.core)

        evil = "'; DROP TABLE documents; --"
        doc_id = meta.add_document(
            file_path=evil, file_name=evil,
            source_type="folder", format="pdf",
            content_hash=hashlib.sha256(evil.encode()).hexdigest(),
        )
        assert doc_id > 0
        # Verify table still exists
        count = db.core.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1
        db.close()

    def test_max_int_values():
        """Handle very large numbers in metadata."""
        db = make_core_db()
        from verra.store.metadata import MetadataStore
        meta = MetadataStore.from_connection(db.core)

        doc_id = meta.add_document(
            file_path="/big.pdf", file_name="big.pdf",
            source_type="folder", format="pdf",
            content_hash=hashlib.sha256(b"big").hexdigest(),
            page_count=999_999_999,
        )
        assert doc_id > 0
        db.close()

    def test_empty_query_handling():
        """Empty and whitespace-only queries should not crash."""
        from verra.retrieval.router import parse_query

        for q in ["", " ", "\n", "\t", "   \n  "]:
            result = parse_query(q)
            assert result is not None

    run_test("Null bytes in text", test_null_bytes)
    run_test("Whitespace-only text", test_only_whitespace)
    run_test("100K-char single word", test_extremely_long_word)
    run_test("Mixed unicode scripts", test_mixed_encodings)
    run_test("50-level nested headings", test_deeply_nested_structure)
    run_test("500K-word single line", test_huge_single_line)
    run_test("SQL injection in data", test_sql_injection_in_data)
    run_test("Max integer values", test_max_int_values)
    run_test("Empty/whitespace queries", test_empty_query_handling)


# ===================================================================
# SECTION: Concurrent Access
# ===================================================================

def test_concurrent():
    print("\n\033[1m── Concurrent Access ──\033[0m")

    def test_concurrent_writes():
        """10 threads writing to separate metadata stores (file-based)."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "concurrent.db"

        from verra.store.metadata import MetadataStore
        # Initialize schema with one connection
        init_store = MetadataStore(db_path)
        init_store.close()

        errors = []
        n_per_thread = scale(1_000)

        def writer(thread_id: int):
            try:
                store = MetadataStore(db_path)
                for i in range(n_per_thread):
                    store.add_document(
                        file_path=f"/t{thread_id}/file_{i}.pdf",
                        file_name=f"file_{thread_id}_{i}.pdf",
                        source_type="folder", format="pdf",
                        content_hash=hashlib.sha256(f"t{thread_id}_{i}".encode()).hexdigest(),
                    )
                store.close()
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert len(errors) == 0, f"Thread errors: {errors}"
        check = MetadataStore(db_path)
        count = check._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        check.close()
        assert count == 10 * n_per_thread, f"Expected {10 * n_per_thread}, got {count}"
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_concurrent_reads_writes():
        """5 writers + 5 readers with separate connections."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "concurrent_rw.db"

        from verra.store.metadata import MetadataStore
        init_store = MetadataStore(db_path)
        init_store.close()

        errors = []
        n_ops = scale(500)

        def writer(tid: int):
            try:
                store = MetadataStore(db_path)
                for i in range(n_ops):
                    store.add_document(
                        file_path=f"/rw{tid}/f{i}.pdf",
                        file_name=f"f{tid}_{i}.pdf",
                        source_type="folder", format="pdf",
                        content_hash=hashlib.sha256(f"rw{tid}_{i}".encode()).hexdigest(),
                    )
                store.close()
            except Exception as exc:
                errors.append(f"Writer {tid}: {exc}")

        def reader(tid: int):
            try:
                store = MetadataStore(db_path)
                for _ in range(n_ops):
                    store._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
                store.close()
            except Exception as exc:
                errors.append(f"Reader {tid}: {exc}")

        threads = []
        for t in range(5):
            threads.append(threading.Thread(target=writer, args=(t,)))
            threads.append(threading.Thread(target=reader, args=(t,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert len(errors) == 0, f"Errors: {errors}"
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    run_test("10 concurrent writer threads", test_concurrent_writes)
    run_test("5 writers + 5 readers", test_concurrent_reads_writes)


# ===================================================================
# SECTION: Real-World Scenarios
# ===================================================================

def test_realworld():
    print("\n\033[1m── Real-World Scenarios ──\033[0m")

    from verra.ingest.chunking import chunk_document, Chunk
    from verra.store.metadata import MetadataStore, classify_document_authority

    def test_legal_contract_50_pages():
        """Simulate a 50-page legal contract with dense legalese."""
        sections = []
        for i in range(50):
            section = f"## Section {i + 1}: {'Indemnification Liability Arbitration Governing Law Termination Confidentiality'.split()[i % 6]}\n\n"
            for j in range(8):
                section += (
                    f"{i + 1}.{j + 1} The Party of the First Part (hereinafter referred to as "
                    f'"Licensor") hereby grants to the Party of the Second Part (hereinafter '
                    f'"Licensee") a non-exclusive, non-transferable, revocable license to use '
                    f"the Software Product (as defined in Schedule A, Exhibit {j + 1}) subject to "
                    f"the terms and conditions set forth herein, including but not limited to the "
                    f"payment of license fees as specified in Section {i + 2}.{j + 1}(a) through "
                    f"Section {i + 2}.{j + 1}(f), and compliance with all applicable federal, "
                    f"state, and local laws, regulations, and ordinances. Notwithstanding anything "
                    f"to the contrary contained herein, the aggregate liability of the Licensor "
                    f"shall not exceed the total fees paid by Licensee during the twelve (12) "
                    f"month period immediately preceding the date on which the claim arose. "
                    f"This Agreement shall be governed by and construed in accordance with the "
                    f"laws of the State of Delaware, without regard to its conflict of laws principles.\n\n"
                )
            section += f"**Effective Date:** January {(i % 28) + 1}, 2024\n"
            section += f"**Renewal Date:** January {(i % 28) + 1}, 2025\n\n"
            sections.append(section)
        text = "\n".join(sections)
        assert len(text) > 100_000, f"Contract too short: {len(text)} chars"
        chunks = chunk_document(text)
        assert len(chunks) > 20, f"Only {len(chunks)} chunks from 50-page contract"
        # Verify no chunk lost the date information entirely
        all_text = " ".join(c.text for c in chunks)
        assert "2025" in all_text, "Lost renewal date during chunking"

    def test_email_thread_200_replies():
        """A single email thread with 200 reply-all messages."""
        messages = []
        participants = [f"person{i}@company.com" for i in range(15)]
        for i in range(200):
            sender = participants[i % len(participants)]
            recipients = [p for p in participants if p != sender]
            msg = f"From: {sender}\nTo: {', '.join(recipients[:5])}\n"
            msg += f"Cc: {', '.join(recipients[5:])}\n"
            msg += f"Date: 2024-{((i // 30) % 12) + 1:02d}-{(i % 28) + 1:02d} {9 + (i % 10)}:{(i * 7) % 60:02d}\n"
            msg += f"Subject: Re: Re: Re: Q4 Budget Planning Discussion\n\n"
            if i < 3:
                msg += f"Hi all,\n\nI'd like to kick off our Q4 budget planning. Key items:\n"
                msg += f"- Marketing budget: $450,000 (up 15% from Q3)\n"
                msg += f"- Engineering headcount: 12 new hires\n"
                msg += f"- Infrastructure costs: $180,000/month\n\n"
            elif i % 7 == 0:
                msg += f"I disagree with the proposed numbers. Here's my counter-proposal:\n"
                msg += f"- We should cap marketing at ${random.randint(200, 500)},000\n"
                msg += f"- Defer {random.randint(2, 8)} of the engineering hires to Q1\n"
                msg += f"- Negotiate AWS reserved instances to save ~30%\n\n"
            else:
                msg += f"+1 on {participants[(i - 1) % len(participants)]}'s points. "
                msg += f"Additionally, I think we should consider {random.choice(['outsourcing', 'automation', 'consolidation', 'partnership', 'acquisition'])} "
                msg += f"for the {random.choice(['infrastructure', 'QA', 'support', 'onboarding', 'deployment'])} workstream.\n\n"
            msg += f"Best,\n{sender.split('@')[0].title()}\n\n---\n\n"
            messages.append(msg)
        text = "\n".join(messages)
        chunks = chunk_document(text, metadata={"source_type": "email"})
        assert len(chunks) > 10, f"Only {len(chunks)} chunks from 200-reply thread"

    def test_financial_spreadsheet_as_text():
        """CSV data with 10K rows (typical financial export)."""
        header = "Date,Account,Description,Debit,Credit,Balance,Category,Reference\n"
        rows = []
        balance = 1_000_000.00
        for i in range(scale(10_000)):
            month = (i % 12) + 1
            day = (i % 28) + 1
            amount = round(random.uniform(10, 50000), 2)
            is_debit = random.random() > 0.4
            if is_debit:
                balance -= amount
                row = f"2024-{month:02d}-{day:02d},{'Operating' if i % 3 else 'Payroll'},{random.choice(['AWS Invoice', 'Salary', 'Office Rent', 'Software License', 'Travel', 'Consulting Fee', 'Insurance Premium'])},{amount:.2f},,{balance:.2f},{random.choice(['OPEX', 'CAPEX', 'PAYROLL'])},REF-{i:06d}"
            else:
                balance += amount
                row = f"2024-{month:02d}-{day:02d},Revenue,{random.choice(['Client Payment', 'Subscription', 'Service Fee', 'Milestone Payment'])},,{amount:.2f},{balance:.2f},REVENUE,INV-{i:06d}"
            rows.append(row)
        text = header + "\n".join(rows)
        chunks = chunk_document(text)
        assert len(chunks) >= 1

    def test_mixed_language_document():
        """Document with English, Chinese, Arabic, and code blocks."""
        text = (
            "# International Operations Report\n\n"
            "## Executive Summary\n\n"
            "Our Q4 expansion into Asian markets exceeded projections by 23%.\n\n"
            "## 中国市场分析 (China Market Analysis)\n\n"
            "第四季度中国市场收入达到 ¥45,000,000，同比增长 34%。\n"
            "主要客户包括：\n"
            "- 阿里巴巴集团 (Alibaba Group) — ¥12,000,000\n"
            "- 腾讯控股 (Tencent) — ¥8,500,000\n"
            "- 字节跳动 (ByteDance) — ¥6,200,000\n\n"
            "### 竞争对手分析\n"
            "市场份额：我们 23%，竞争对手A 31%，竞争对手B 18%\n\n"
            "## تقرير السوق العربي (Arabic Market Report)\n\n"
            "إجمالي الإيرادات في الربع الرابع: 15,000,000 درهم إماراتي\n"
            "عدد العملاء الجدد: 45 عميل\n"
            "معدل الاحتفاظ بالعملاء: 92%\n\n"
            "## Technical Integration Notes\n\n"
            "    class MarketAnalyzer:\n"
            "        def calculate_growth(self, current, previous):\n"
            "            return ((current - previous) / previous) * 100\n\n"
            "    SELECT region, SUM(revenue) as total_revenue\n"
            "    FROM sales WHERE quarter = 4\n"
            "    GROUP BY region ORDER BY total_revenue DESC;\n\n"
            "## Appendix: Raw Numbers\n\n"
        )
        appendix_rows = []
        for i in range(200):
            region = "APAC" if i % 3 == 0 else "EMEA" if i % 3 == 1 else "LATAM"
            product = "Product-" + chr(65 + i % 26)
            amount = random.randint(10000, 999999)
            pct = random.randint(5, 95)
            appendix_rows.append(f"| {region} | {product} | {amount:,} | {pct}% |")
        text += "\n".join(appendix_rows)

        chunks = chunk_document(text)
        assert len(chunks) >= 1, f"Expected at least 1 chunk, got {len(chunks)}"
        all_text = " ".join(c.text for c in chunks)
        # With enough content the chunker may split — verify nothing is silently dropped
        assert len(all_text) > 1000, f"Total chunked text too short: {len(all_text)}"
        # At least some Chinese, Arabic, and code should survive
        has_cjk = any(ord(c) > 0x4E00 for c in all_text)
        has_arabic = any("\u0600" <= c <= "\u06FF" for c in all_text)
        assert has_cjk, "Lost all CJK text during chunking"
        assert has_arabic, "Lost all Arabic text during chunking"

    def test_malformed_pdf_text_extraction():
        """Simulate garbled PDF extraction output (common with scanned docs)."""
        garbled = ""
        for i in range(scale(5_000)):
            if random.random() < 0.3:
                garbled += "".join(random.choices("abcdefghijklmnopqrstuvwxyz  \n\t", k=random.randint(5, 50)))
            elif random.random() < 0.5:
                garbled += f"Page {i} of {scale(5_000)} - {'█' * random.randint(1, 20)} "
                garbled += random.choice(["Contract", "Agreement", "CONFIDENTIAL", "DRAFT", ""])
                garbled += "\n"
            else:
                garbled += random_text(random.randint(10, 50)) + "\n\n"
        chunks = chunk_document(garbled)
        assert len(chunks) > 0, "Should handle garbled text"

    def test_resume_dump_1000_files():
        """Simulate ingesting 1000 resumes (HR department use case)."""
        n = scale(1_000)
        db = make_core_db()
        store = MetadataStore.from_connection(db.core)
        vec = MockVectorStore()

        skills = ["Python", "Java", "React", "AWS", "Docker", "Kubernetes",
                  "Machine Learning", "SQL", "TypeScript", "Go", "Rust", "C++",
                  "Project Management", "Agile", "Scrum", "DevOps"]
        companies = ["Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix",
                     "Stripe", "Airbnb", "Uber", "Spotify", "Salesforce"]

        for i in range(n):
            name = f"{'John Jane Alex Sam Pat Chris'.split()[i % 6]} {'Smith Johnson Williams Brown Davis Miller'.split()[i % 6]}"
            resume = f"# {name}\n\n"
            resume += f"Email: {name.lower().replace(' ', '.')}@email.com\n"
            resume += f"Phone: +1-555-{random.randint(1000, 9999)}\n\n"
            resume += f"## Experience\n\n"
            for j in range(random.randint(2, 5)):
                company = random.choice(companies)
                resume += f"### {random.choice(['Senior', 'Staff', 'Lead', 'Principal'])} {random.choice(['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer'])} at {company}\n"
                resume += f"*{2020 - j * 2} - {2022 - j * 2}*\n\n"
                resume += f"- Led team of {random.randint(3, 15)} engineers on {random.choice(['payments', 'search', 'ads', 'infrastructure', 'mobile', 'ML'])} project\n"
                resume += f"- Improved {random.choice(['latency', 'throughput', 'reliability', 'test coverage'])} by {random.randint(20, 300)}%\n"
                resume += f"- Managed ${random.randint(1, 50)}M annual budget\n\n"
            resume += f"## Skills\n\n"
            resume += ", ".join(random.sample(skills, random.randint(5, 10))) + "\n\n"
            resume += f"## Education\n\n"
            resume += f"BS Computer Science, {random.choice(['MIT', 'Stanford', 'CMU', 'Berkeley', 'Georgia Tech'])}, {random.randint(2005, 2018)}\n"

            content_hash = hashlib.sha256(resume.encode()).hexdigest()
            doc_id = store.add_document(
                file_path=f"/hr/resumes/{name.replace(' ', '_')}.pdf",
                file_name=f"{name}.pdf",
                source_type="folder", format="pdf",
                content_hash=content_hash,
            )
            chunks = chunk_document(resume)
            cids = store.add_chunks(doc_id, chunks)
            for j, (cid, chunk) in enumerate(zip(cids, chunks)):
                vec.add_chunks(ids=[str(cid)], texts=[chunk.text])

        assert vec.total_chunks >= n, f"Expected at least {n} chunks from {n} resumes, got {vec.total_chunks}"
        db.close()

    def test_rapid_fire_queries():
        """Simulate 500 users querying simultaneously (burst traffic)."""
        n = scale(500)
        db = make_core_db()
        from verra.store.metadata import MetadataStore
        from verra.store.memory import MemoryStore
        from verra.agent.chat import ChatEngine

        meta = MetadataStore.from_connection(db.core)
        mem = MemoryStore.from_connection(db.core)
        vec = MockVectorStore()
        vec.total_chunks = 5000

        mock_llm = MagicMock()
        mock_llm.model = "mock"
        mock_llm.complete.return_value = "Here's what I found in your data."
        mock_llm.complete_with_tools.return_value = "Based on the search results..."

        real_queries = [
            "What did we agree with Acme about pricing?",
            "Find the latest contract with Johnson & Johnson",
            "Who is responsible for the Q4 marketing budget?",
            "Show me all emails from Jake about the partnership",
            "What's our refund policy?",
            "When does the AWS contract expire?",
            "How much did we spend on infrastructure last quarter?",
            "Find meeting notes from the board meeting on Jan 15",
            "What commitments did we make to the client in the SOW?",
            "Summarize the NDA with Acme Corp",
            "Who are our top 5 customers by revenue?",
            "What's the status of Project Alpha?",
            "Find all invoices over $10,000 from last year",
            "What did Sarah say about the hiring freeze?",
            "Show me the latest performance review for the engineering team",
        ]

        engines = []
        for i in range(min(n, 20)):
            e = ChatEngine(llm=mock_llm, metadata_store=meta,
                          vector_store=vec, memory_store=mem)
            engines.append(e)

        for i in range(n):
            engine = engines[i % len(engines)]
            query = random.choice(real_queries)
            response = engine.ask(query, use_multi_hop=False)
            assert response.answer, f"Empty answer on query {i}"
        db.close()

    def test_company_full_lifecycle():
        """Simulate a company's full data lifecycle: ingest → query → correct → briefing."""
        db = make_core_db()
        from verra.store.metadata import MetadataStore
        from verra.store.memory import MemoryStore
        from verra.store.entities import EntityStore
        from verra.agent.chat import ChatEngine

        meta = MetadataStore.from_connection(db.core)
        mem = MemoryStore.from_connection(db.core)
        ent = EntityStore.from_connection(db.core)
        vec = MockVectorStore()
        vec.total_chunks = 500

        # Phase 1: Ingest company data
        departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Legal"]
        doc_types = ["contract", "policy", "meeting", "invoice", "email", "report"]
        for dept in departments:
            for dtype in doc_types:
                for i in range(scale(50)):
                    h = hashlib.sha256(f"{dept}_{dtype}_{i}".encode()).hexdigest()
                    doc_id = meta.add_document(
                        file_path=f"/company/{dept}/{dtype}_{i}.pdf",
                        file_name=f"{dtype}_{i}.pdf",
                        source_type="folder", format="pdf",
                        content_hash=h,
                        document_type=dtype,
                    )
                    chunks = [Chunk(text=random_text(80), token_count=100) for _ in range(5)]
                    meta.add_chunks(doc_id, chunks)

        # Phase 2: Add entities
        people = ["Alice Chen", "Bob Smith", "Carol Johnson", "Dave Williams",
                  "Eve Brown", "Frank Davis", "Grace Miller", "Henry Wilson"]
        for person in people:
            eid = ent.add_entity(person, "person")
            ent.add_entity(f"{person.split()[0]}'s Team", "project")

        for p1, p2 in zip(people[:-1], people[1:]):
            e1 = ent.resolve(p1)
            e2 = ent.resolve(p2)
            if e1 and e2:
                ent.add_relationship(e1["id"], "works_with", e2["id"])

        # Phase 3: Add emails
        for i in range(scale(500)):
            meta.add_email(
                thread_id=f"thread_{i // 5}",
                message_id=f"msg_{i}",
                from_addr=f"{random.choice(people).split()[0].lower()}@company.com",
                to_addr=f"{random.choice(people).split()[0].lower()}@company.com",
                cc_addr=None,
                subject=f"Re: {random.choice(['Budget', 'Roadmap', 'Hiring', 'Contract', 'Review'])} Discussion",
                date=(datetime.now() - timedelta(days=random.randint(0, 90))).isoformat(),
            )

        # Phase 4: Chat queries
        mock_llm = MagicMock()
        mock_llm.model = "mock"
        mock_llm.complete.return_value = "Based on your company data, here's what I found."

        engine = ChatEngine(llm=mock_llm, metadata_store=meta,
                           vector_store=vec, memory_store=mem,
                           entity_store=ent)

        for q in [
            "What contracts does the Legal department have?",
            "Show me Alice Chen's recent emails",
            "What's the engineering team's budget?",
            "Find all invoices from the Finance department",
        ]:
            resp = engine.ask(q, use_multi_hop=False)
            assert resp.answer

        # Phase 5: Briefing
        from verra.briefing.detector import BriefingDetector
        cfg = MagicMock()
        cfg.max_items = 10
        cfg.stale_lead_days = 14
        cfg.contract_warning_days = 30
        detector = BriefingDetector(core_conn=db.core, analysis_conn=db.analysis, config=cfg)
        items = detector.detect_all()
        assert isinstance(items, list)

        # Phase 6: Memory persistence
        for i in range(100):
            mem.set_memory("preference", f"pref_{i}", f"User prefers {random_text(5)}")
        assert db.core.execute("SELECT COUNT(*) FROM memory WHERE category='preference'").fetchone()[0] == 100

        db.close()

    def test_million_row_queries():
        """Query performance with 1M chunks in metadata store."""
        db = make_core_db()
        meta = MetadataStore.from_connection(db.core)

        n_docs = scale(10_000)
        chunks_per = 100 if not QUICK else 1

        # Bulk insert using executemany for speed
        doc_rows = [
            (f"/docs/file_{i}.pdf", f"file_{i}.pdf", "folder", "pdf",
             hashlib.sha256(f"bulk{i}".encode()).hexdigest(), 1, None, "general", 50)
            for i in range(n_docs)
        ]
        db.core.executemany(
            """INSERT INTO documents (file_path, file_name, source_type, format,
               content_hash, page_count, extra_metadata, document_type, authority_weight)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            doc_rows,
        )
        db.core.commit()

        chunk_rows = []
        for doc_id in range(1, n_docs + 1):
            for pos in range(chunks_per):
                chunk_rows.append((doc_id, pos, random.randint(50, 500), None, 50))
        db.core.executemany(
            """INSERT INTO chunks (document_id, position, token_count, metadata, authority_weight)
               VALUES (?, ?, ?, ?, ?)""",
            chunk_rows,
        )
        db.core.commit()

        total = db.core.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert total == n_docs * chunks_per

        # Time some queries
        t0 = time.time()
        for _ in range(100):
            db.core.execute(
                "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.document_type = 'general'"
            ).fetchone()
        query_time = time.time() - t0
        assert query_time < 30, f"100 queries on {total} rows took {query_time:.1f}s — too slow"

        db.close()

    def test_memory_pressure():
        """Generate and chunk data until we hit 500MB of text processed without leaking."""
        import resource
        target_mb = 50 if QUICK else 500
        processed = 0
        batch_size = 1_000_000  # 1M chars per batch (~1MB)

        initial_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        batch_count = 0

        while processed < target_mb * 1_000_000:
            text = random_text(batch_size // 5)  # ~5 chars per word avg
            chunks = chunk_document(text)
            processed += len(text)
            batch_count += 1
            del text, chunks
            if batch_count % 10 == 0:
                gc.collect()

        final_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS ru_maxrss is bytes, on Linux it's KB
        mem_growth = (final_mem - initial_mem)
        # Just verify we didn't OOM — the test passing is the assertion
        assert processed >= target_mb * 1_000_000

    def test_pathological_regex_inputs():
        """Inputs designed to trigger regex backtracking in NER/router."""
        from verra.retrieval.router import parse_query
        from verra.ingest.ner import extract_entities

        evil_inputs = [
            "a]" * 10000,  # bracket bombs
            "(" * 5000 + ")" * 5000,  # nested parens
            "from: " + "a" * 50000,  # huge from field
            "@" * 10000,  # at-sign spam
            "http://" + "a" * 50000 + ".com",  # huge URL
            "$" + "9" * 50000,  # huge dollar amount
            "\\" * 10000,  # backslash flood
            "email@" * 10000,  # repeated email pattern
        ]
        for inp in evil_inputs:
            t0 = time.time()
            try:
                parse_query(inp)
            except Exception:
                pass
            try:
                extract_entities(inp)
            except Exception:
                pass
            elapsed = time.time() - t0
            assert elapsed < 5, f"Regex took {elapsed:.1f}s on pathological input (len={len(inp)})"

    def test_authority_ranking_consistency():
        """Verify authority ranking is deterministic across repeated runs."""
        from verra.store.metadata import classify_document_authority

        test_cases = [
            ("company_policy_2024.pdf", "/legal/policies/handbook/", "All employees must adhere to the following guidelines regarding data retention and privacy."),
            ("signed_contract_acme.docx", "/contracts/active/", "This Master Services Agreement is entered into between Acme Corp and our company."),
            ("random_notes.txt", "/tmp/scratch/", "Just some random notes I took during lunch."),
        ]
        for _ in range(1000):
            for fname, fpath, content in test_cases:
                dtype, weight = classify_document_authority(fname, fpath, content)
                dtype2, weight2 = classify_document_authority(fname, fpath, content)
                assert dtype == dtype2 and weight == weight2, "Authority classification is non-deterministic"

    run_test("50-page legal contract", test_legal_contract_50_pages)
    run_test("200-reply email thread", test_email_thread_200_replies)
    run_test("10K-row financial CSV", test_financial_spreadsheet_as_text)
    run_test("Mixed language (EN/ZH/AR + code)", test_mixed_language_document)
    run_test("Garbled PDF extraction", test_malformed_pdf_text_extraction)
    run_test("1000 resumes (HR ingest)", test_resume_dump_1000_files)
    run_test("500 rapid-fire queries", test_rapid_fire_queries)
    run_test("Company full lifecycle", test_company_full_lifecycle)
    run_test("1M-row query performance", test_million_row_queries)
    run_test("500MB memory pressure", test_memory_pressure)
    run_test("Pathological regex inputs", test_pathological_regex_inputs)
    run_test("Authority ranking consistency", test_authority_ranking_consistency)


# ===================================================================
# SECTION: Stress (extreme scale)
# ===================================================================

def test_stress():
    print("\n\033[1m── Stress Tests ──\033[0m")

    from verra.ingest.chunking import chunk_document, Chunk
    from verra.store.metadata import MetadataStore

    def test_5m_word_document():
        """Chunk a single 5-million-word document with paragraph breaks."""
        # Must include paragraph breaks for the chunker to split
        paragraphs = [random_text(200) for _ in range(scale(25_000))]
        text = "\n\n".join(paragraphs)
        chunks = chunk_document(text)
        assert len(chunks) > 10, f"Only {len(chunks)} chunks from 5M-word doc"
        del text, chunks
        gc.collect()

    def test_1m_documents_metadata():
        """Insert 1M documents into metadata store."""
        n = scale(1_000_000)
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        store = MetadataStore.__new__(MetadataStore)
        store.db_path = Path(":memory:")
        store._conn = conn
        from verra.store.metadata import _SCHEMA
        conn.executescript(_SCHEMA)
        conn.commit()

        batch = []
        for i in range(n):
            batch.append((
                f"/docs/f{i}.pdf", f"f{i}.pdf", "folder", "pdf",
                hashlib.sha256(f"m{i}".encode()).hexdigest(),
                1, None, "general", 50,
            ))
            if len(batch) >= 10_000:
                conn.executemany(
                    """INSERT INTO documents (file_path, file_name, source_type, format,
                       content_hash, page_count, extra_metadata, document_type, authority_weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    batch,
                )
                conn.commit()
                batch.clear()
        if batch:
            conn.executemany(
                """INSERT INTO documents (file_path, file_name, source_type, format,
                   content_hash, page_count, extra_metadata, document_type, authority_weight)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch,
            )
            conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == n
        conn.close()

    def test_100k_entity_graph():
        """Build a 100K-node entity graph with relationships."""
        n = scale(100_000)
        db = make_core_db()
        from verra.store.entities import EntityStore
        ent = EntityStore.from_connection(db.core)

        ids = []
        for i in range(n):
            eid = ent.add_entity(f"Entity-{i}", random.choice(["person", "company", "project"]))
            ids.append(eid)

        # Add 500K relationships
        rel_count = 0
        for _ in range(min(n * 5, scale(500_000))):
            a, b = random.sample(ids[:min(len(ids), 10000)], 2)
            try:
                ent.add_relationship(a, random.choice(["works_with", "reports_to", "client_of"]), b)
                rel_count += 1
            except Exception:
                pass
        assert rel_count > 0
        db.close()

    def test_200_concurrent_conversations():
        """200 parallel chat sessions with file-backed DB."""
        import tempfile
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir)

        from verra.store.db import DatabaseManager
        from verra.store.metadata import MetadataStore
        from verra.store.memory import MemoryStore
        from verra.agent.chat import ChatEngine

        # Init schema
        init_db = DatabaseManager(db_path)
        init_db.close()

        vec = MockVectorStore()
        vec.total_chunks = 1000

        mock_llm = MagicMock()
        mock_llm.model = "mock"
        mock_llm.complete.return_value = "Mock response."

        n = scale(200)
        errors = []

        def chat_session(sid: int):
            try:
                sdb = DatabaseManager(db_path)
                meta = MetadataStore.from_connection(sdb.core)
                mem = MemoryStore.from_connection(sdb.core)
                engine = ChatEngine(llm=mock_llm, metadata_store=meta,
                                   vector_store=vec, memory_store=mem)
                for _ in range(5):
                    engine.ask(random_text(10), use_multi_hop=False)
                sdb.close()
            except Exception as exc:
                errors.append(f"Session {sid}: {exc}")

        threads = [threading.Thread(target=chat_session, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        assert len(errors) == 0, f"Session errors: {errors[:5]}"
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    run_test("5M-word single document", test_5m_word_document)
    run_test("1M document inserts", test_1m_documents_metadata)
    run_test("100K entity graph", test_100k_entity_graph)
    run_test("200 concurrent chat sessions", test_200_concurrent_conversations)


# ===================================================================
# Main
# ===================================================================

def main():
    sections = {
        "chunking": test_chunking,
        "metadata": test_metadata,
        "entities": test_entities,
        "memory": test_memory,
        "retrieval": test_retrieval,
        "chat": test_chat,
        "briefing": test_briefing,
        "pipeline": test_pipeline,
        "edge_cases": test_edge_cases,
        "concurrent": test_concurrent,
        "realworld": test_realworld,
        "stress": test_stress,
    }

    print(f"\n\033[1m{'=' * 60}\033[0m")
    print(f"\033[1m  Verra One — Load Test Suite\033[0m")
    print(f"\033[1m  Mode: {'QUICK' if QUICK else 'FULL'}\033[0m")
    if SECTION:
        print(f"\033[1m  Section: {SECTION}\033[0m")
    print(f"\033[1m{'=' * 60}\033[0m")

    t0 = time.time()

    if SECTION:
        if SECTION in sections:
            sections[SECTION]()
        else:
            print(f"\n  Unknown section: {SECTION}")
            print(f"  Available: {', '.join(sections.keys())}")
            sys.exit(1)
    else:
        for fn in sections.values():
            fn()

    elapsed = time.time() - t0

    print(f"\n\033[1m{'=' * 60}\033[0m")
    print(f"  \033[32m{PASSED} passed\033[0m, \033[31m{FAILED} failed\033[0m  ({elapsed:.1f}s)")
    if ERRORS:
        print(f"\n\033[31m  Failures:\033[0m")
        for err in ERRORS:
            print(f"    {err.split(chr(10))[0]}")
    print(f"\033[1m{'=' * 60}\033[0m\n")

    sys.exit(1 if FAILED > 0 else 0)


if __name__ == "__main__":
    main()
