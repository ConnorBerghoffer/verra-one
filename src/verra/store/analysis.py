"""Chunk analysis, conflicts, commitments, and entity summaries storage.

Stores the output of the LLM-powered ingestion analyser:
- chunk_analysis: per-chunk analysis metadata (sentiment, topics, staleness)
- conflicts: contradictions between assertions
- commitments: extracted action items and promises
- entity_summaries: auto-generated entity profiles
- document_coverage: what document types exist per entity
"""


from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class AnalysisStore:
    """SQLite-backed storage for ingestion analysis results."""

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "AnalysisStore":
        """Create an AnalysisStore that shares an existing SQLite connection.

        Used by DatabaseManager so all analysis stores share analysis.db.
        The caller is responsible for connection lifecycle.
        """
        instance = object.__new__(cls)
        instance._conn = conn
        instance._create_tables()
        return instance

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunk_analysis (
                chunk_id INTEGER PRIMARY KEY,
                analysis_status TEXT DEFAULT 'pending',
                sentiment TEXT,
                staleness_risk REAL DEFAULT 0.0,
                topics TEXT DEFAULT '[]',
                assertions_extracted INTEGER DEFAULT 0,
                analysed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assertion_a TEXT NOT NULL,
                assertion_b TEXT NOT NULL,
                entity_id INTEGER,
                source_chunk_a INTEGER,
                source_chunk_b INTEGER,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved INTEGER DEFAULT 0,
                resolution_notes TEXT
            );

            CREATE TABLE IF NOT EXISTS commitments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                who_entity_id INTEGER,
                who_name TEXT NOT NULL,
                what TEXT NOT NULL,
                due_date TEXT,
                status TEXT DEFAULT 'open',
                source_chunk_id INTEGER,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS entity_summaries (
                entity_id INTEGER PRIMARY KEY,
                summary_text TEXT NOT NULL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunk_count INTEGER DEFAULT 0,
                based_on_chunks TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS document_coverage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                document_type TEXT NOT NULL,
                present INTEGER DEFAULT 1,
                last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source_chunk_id INTEGER,
                UNIQUE(entity_id, document_type)
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_analysis_status
                ON chunk_analysis(analysis_status);
            CREATE INDEX IF NOT EXISTS idx_commitments_status
                ON commitments(status);
            CREATE INDEX IF NOT EXISTS idx_conflicts_resolved
                ON conflicts(resolved);
            CREATE INDEX IF NOT EXISTS idx_doc_coverage_entity
                ON document_coverage(entity_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Chunk Analysis
    # ------------------------------------------------------------------

    def set_chunk_status(self, chunk_id: int, status: str) -> None:
        self._conn.execute(
            """INSERT INTO chunk_analysis (chunk_id, analysis_status)
               VALUES (?, ?)
               ON CONFLICT(chunk_id) DO UPDATE SET analysis_status = ?""",
            (chunk_id, status, status),
        )
        self._conn.commit()

    def save_chunk_analysis(
        self,
        chunk_id: int,
        sentiment: str | None = None,
        staleness_risk: float = 0.0,
        topics: list[str] | None = None,
        assertions_count: int = 0,
    ) -> None:
        self._conn.execute(
            """INSERT INTO chunk_analysis
               (chunk_id, analysis_status, sentiment, staleness_risk, topics, assertions_extracted, analysed_at)
               VALUES (?, 'analysed', ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(chunk_id) DO UPDATE SET
                 analysis_status = 'analysed',
                 sentiment = ?, staleness_risk = ?,
                 topics = ?, assertions_extracted = ?,
                 analysed_at = CURRENT_TIMESTAMP""",
            (
                chunk_id, sentiment, staleness_risk,
                json.dumps(topics or []), assertions_count,
                sentiment, staleness_risk,
                json.dumps(topics or []), assertions_count,
            ),
        )
        self._conn.commit()

    def get_pending_chunks(self, limit: int = 100) -> list[int]:
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunk_analysis WHERE analysis_status = 'pending' LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    def get_chunk_analysis(self, chunk_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM chunk_analysis WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["topics"] = json.loads(d.get("topics") or "[]")
            return d
        return None

    # ------------------------------------------------------------------
    # Conflicts
    # ------------------------------------------------------------------

    def add_conflict(
        self,
        assertion_a: str,
        assertion_b: str,
        entity_id: int | None = None,
        source_chunk_a: int | None = None,
        source_chunk_b: int | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO conflicts
               (assertion_a, assertion_b, entity_id, source_chunk_a, source_chunk_b)
               VALUES (?, ?, ?, ?, ?)""",
            (assertion_a, assertion_b, entity_id, source_chunk_a, source_chunk_b),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_unresolved_conflicts(self, entity_id: int | None = None) -> list[dict]:
        if entity_id:
            rows = self._conn.execute(
                "SELECT * FROM conflicts WHERE resolved = 0 AND entity_id = ? ORDER BY detected_at DESC",
                (entity_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM conflicts WHERE resolved = 0 ORDER BY detected_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_conflict(self, conflict_id: int, notes: str = "") -> None:
        self._conn.execute(
            "UPDATE conflicts SET resolved = 1, resolution_notes = ? WHERE id = ?",
            (notes, conflict_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Commitments
    # ------------------------------------------------------------------

    def add_commitment(
        self,
        who_name: str,
        what: str,
        who_entity_id: int | None = None,
        due_date: str | None = None,
        source_chunk_id: int | None = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO commitments
               (who_entity_id, who_name, what, due_date, source_chunk_id)
               VALUES (?, ?, ?, ?, ?)""",
            (who_entity_id, who_name, what, due_date, source_chunk_id),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_open_commitments(self, entity_id: int | None = None) -> list[dict]:
        if entity_id:
            rows = self._conn.execute(
                "SELECT * FROM commitments WHERE status = 'open' AND who_entity_id = ? ORDER BY due_date",
                (entity_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM commitments WHERE status = 'open' ORDER BY due_date"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_commitment_status(self, commitment_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE commitments SET status = ? WHERE id = ?", (status, commitment_id)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Entity Summaries
    # ------------------------------------------------------------------

    def save_entity_summary(
        self,
        entity_id: int,
        summary_text: str,
        chunk_count: int = 0,
        based_on_chunks: list[int] | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO entity_summaries
               (entity_id, summary_text, chunk_count, based_on_chunks, generated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(entity_id) DO UPDATE SET
                 summary_text = ?, chunk_count = ?,
                 based_on_chunks = ?, generated_at = CURRENT_TIMESTAMP""",
            (
                entity_id, summary_text, chunk_count,
                json.dumps(based_on_chunks or []),
                summary_text, chunk_count,
                json.dumps(based_on_chunks or []),
            ),
        )
        self._conn.commit()

    def get_entity_summary(self, entity_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM entity_summaries WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["based_on_chunks"] = json.loads(d.get("based_on_chunks") or "[]")
            return d
        return None

    # ------------------------------------------------------------------
    # Document Coverage
    # ------------------------------------------------------------------

    def update_document_coverage(
        self,
        entity_id: int,
        document_type: str,
        source_chunk_id: int | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO document_coverage
               (entity_id, document_type, present, source_chunk_id, last_verified)
               VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(entity_id, document_type) DO UPDATE SET
                 present = 1, last_verified = CURRENT_TIMESTAMP,
                 source_chunk_id = COALESCE(?, document_coverage.source_chunk_id)""",
            (entity_id, document_type, source_chunk_id, source_chunk_id),
        )
        self._conn.commit()

    def get_entity_coverage(self, entity_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM document_coverage WHERE entity_id = ? ORDER BY document_type",
            (entity_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_coverage_gaps(
        self, entity_id: int, expected_types: list[str]
    ) -> list[str]:
        existing = {
            r["document_type"]
            for r in self._conn.execute(
                "SELECT document_type FROM document_coverage WHERE entity_id = ? AND present = 1",
                (entity_id,),
            ).fetchall()
        }
        return [t for t in expected_types if t not in existing]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
