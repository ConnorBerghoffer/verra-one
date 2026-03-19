"""SQLite metadata store.

Tables
------
documents       — one row per ingested file (path, hash, format, etc.)
chunks          — one row per chunk (document_id FK, position, token_count)
emails          — email-specific metadata (from, to, subject, thread_id, etc.)
sync_state      — last sync cursor per source
chunk_hierarchy — parent/child relationships between chunks (sections, paragraphs)
"""


from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verra.ingest.chunking import Chunk


# Authority classification

# Keyword lists used for document authority classification.
# Each tuple is (keywords_to_match, doc_type, authority_weight).
_AUTHORITY_RULES: list[tuple[list[str], str, int]] = [
    (["policy", "policies", "handbook", "guidelines"], "policy", 90),
    (["contract", "agreement", "msa", "nda", "sow"], "contract", 85),
    (["proposal", "scope"], "contract", 80),
    (["invoice", "receipt", "billing"], "financial", 75),
    (["meeting", "standup", "sprint", "retro"], "team", 60),
]

# Path-segment keywords that indicate an executive-level document.
_EXECUTIVE_PATH_KEYWORDS = {"executive", "board", "ceo", "cto", "director"}


def classify_document_authority(
    file_name: str,
    file_path: str,
    content: str,
) -> tuple[str, int]:
    """Classify a document's type and authority weight.

    Examines the file name, its directory path, and a snippet of the
    document content to decide which authority tier the document belongs
    to.

    Returns
    -------
    (document_type, authority_weight)
        document_type is one of: 'policy', 'contract', 'executive',
        'financial', 'management', 'team', 'email', 'informal', 'general'.
        authority_weight is an integer on the 0-100 scale where higher
        numbers mean more authoritative.
    """
    # Combine the signals we can inspect without tokenising the full content.
    name_lower = file_name.lower()
    path_lower = file_path.lower()
    # Only look at the first 2000 characters of content to keep it cheap.
    content_lower = content[:2000].lower()

    combined = f"{name_lower} {path_lower} {content_lower}"

    # Check keyword rules in order of descending authority weight.
    for keywords, doc_type, weight in _AUTHORITY_RULES:
        if any(kw in combined for kw in keywords):
            return doc_type, weight

    # Path-segment check for executive/board/director documents.
    path_parts = set(re.split(r"[\\/\s_-]", path_lower))
    if path_parts & _EXECUTIVE_PATH_KEYWORDS:
        return "executive", 80

    # Email content heuristic: email threads have From/To headers.
    if re.search(r"\bfrom:\s+\S", content_lower) and re.search(r"\bto:\s+\S", content_lower):
        return "email", 50

    # Management documents: words commonly found in manager-authored docs.
    management_kws = {"directive", "okr", "roadmap", "performance", "review", "1:1", "one-on-one"}
    if any(kw in combined for kw in management_kws):
        return "management", 70

    # Informal: very short files, personal notes, markdown scratchpads.
    if len(content.strip()) < 300:
        return "informal", 30

    return "general", 50


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path      TEXT    NOT NULL,
    file_name      TEXT    NOT NULL,
    source_type    TEXT    NOT NULL DEFAULT 'folder',
    format         TEXT,
    content_hash   TEXT    NOT NULL,
    page_count     INTEGER NOT NULL DEFAULT 1,
    extra_metadata TEXT,                        -- JSON blob
    document_type  TEXT    NOT NULL DEFAULT 'general',
    authority_weight INTEGER NOT NULL DEFAULT 50,
    indexed_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash
    ON documents(content_hash);

CREATE INDEX IF NOT EXISTS idx_documents_path
    ON documents(file_path);

CREATE TABLE IF NOT EXISTS chunks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position         INTEGER NOT NULL,
    token_count      INTEGER NOT NULL,
    metadata         TEXT,                      -- JSON blob
    authority_weight INTEGER NOT NULL DEFAULT 50,
    valid_from       TIMESTAMP,
    valid_until      TIMESTAMP,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON chunks(document_id);

CREATE TABLE IF NOT EXISTS chunk_hierarchy (
    chunk_id       INTEGER PRIMARY KEY,
    parent_chunk_id INTEGER,
    level          INTEGER NOT NULL DEFAULT 0,  -- 0=document, 1=section, 2=subsection, 3=paragraph
    position       INTEGER NOT NULL DEFAULT 0,  -- order within parent
    heading        TEXT                         -- section heading if applicable
);

CREATE TABLE IF NOT EXISTS emails (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT    NOT NULL,
    message_id  TEXT,
    from_addr   TEXT,
    to_addr     TEXT,
    cc_addr     TEXT,
    subject     TEXT,
    date        TEXT,
    labels      TEXT,             -- JSON array
    chunk_id    INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_emails_thread ON emails(thread_id);
CREATE INDEX IF NOT EXISTS idx_emails_from   ON emails(from_addr);
CREATE INDEX IF NOT EXISTS idx_emails_date   ON emails(date);

CREATE TABLE IF NOT EXISTS chunk_near_duplicates (
    chunk_id         INTEGER NOT NULL,
    near_duplicate_of INTEGER NOT NULL,
    similarity_score  REAL NOT NULL,
    PRIMARY KEY (chunk_id, near_duplicate_of)
);

CREATE INDEX IF NOT EXISTS idx_near_dup_chunk
    ON chunk_near_duplicates(chunk_id);

CREATE TABLE IF NOT EXISTS chunk_references (
    source_chunk_id INTEGER,
    reference_text TEXT,
    target_document_id INTEGER,  -- NULL if unresolved
    target_chunk_id INTEGER,     -- NULL if unresolved
    reference_type TEXT,         -- 'document', 'ticket', 'discussion'
    confidence REAL DEFAULT 0.5,
    PRIMARY KEY (source_chunk_id, reference_text)
);

CREATE INDEX IF NOT EXISTS idx_chunk_refs_source
    ON chunk_references(source_chunk_id);

CREATE TABLE IF NOT EXISTS sync_state (
    source           TEXT PRIMARY KEY,
    last_sync_at     TEXT NOT NULL DEFAULT (datetime('now')),
    cursor           TEXT,
    items_processed  INTEGER NOT NULL DEFAULT 0,
    status           TEXT    NOT NULL DEFAULT 'idle'
);

-- FTS5 virtual table for BM25 full-text search across chunk text.
-- We store text directly here (not a content table) because the text
-- lives in the JSON metadata blob, not a plain text column in chunks.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    text,
    tokenize = 'porter ascii'
);
"""


class MetadataStore:
    """Thin wrapper around a SQLite database for document and chunk metadata."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "MetadataStore":
        """Create a MetadataStore that shares an existing SQLite connection.

        Used by DatabaseManager so all core stores share core.db.
        The caller is responsible for connection lifecycle.
        """
        instance = object.__new__(cls)
        instance.db_path = Path(":memory:")  # sentinel — not a real path
        instance._conn = conn
        # Schema already created by DatabaseManager — skip redundant executescript
        return instance

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def add_document(
        self,
        file_path: str,
        file_name: str,
        source_type: str,
        format: str,
        content_hash: str,
        page_count: int = 1,
        extra_metadata: dict[str, Any] | None = None,
        document_type: str = "general",
        authority_weight: int = 50,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO documents
                (file_path, file_name, source_type, format, content_hash,
                 page_count, extra_metadata, document_type, authority_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_path,
                file_name,
                source_type,
                format,
                content_hash,
                page_count,
                json.dumps(extra_metadata) if extra_metadata else None,
                document_type,
                authority_weight,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return dict(row) if row else None

    def get_document_by_path(self, file_path: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE file_path = ?", (file_path,)
        ).fetchone()
        return dict(row) if row else None

    def delete_document(self, document_id: int) -> None:
        self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self._conn.commit()

    def list_documents(self, source_type: str | None = None) -> list[dict[str, Any]]:
        if source_type:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE source_type = ? ORDER BY indexed_at DESC",
                (source_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM documents ORDER BY indexed_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        document_id: int,
        chunks: list[Chunk],
        authority_weight: int = 50,
    ) -> list[int]:
        """Insert chunks and return their SQLite IDs.

        Parameters
        ----------
        document_id:
            Parent document row ID.
        chunks:
            Chunk objects from the chunker.
        authority_weight:
            Authority score inherited from the parent document (0-100).
        """
        ids: list[int] = []
        for position, chunk in enumerate(chunks):
            cur = self._conn.execute(
                """
                INSERT INTO chunks (document_id, position, token_count, metadata, authority_weight)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    position,
                    chunk.token_count,
                    json.dumps(chunk.metadata),
                    authority_weight,
                ),
            )
            ids.append(cur.lastrowid)  # type: ignore[arg-type]
        self._conn.commit()
        return ids

    def index_chunk_text(self, chunk_id: int, text: str) -> None:
        """Insert or replace a chunk's text in the FTS5 index.

        Called at ingest time after the chunk is stored in ChromaDB so that
        the same text is searchable via BM25 without a round-trip to the
        vector store.
        """
        # DELETE + INSERT pattern for upsert on FTS5 tables (no ON CONFLICT support)
        self._conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
        )
        self._conn.execute(
            "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
            (chunk_id, text),
        )
        self._conn.commit()

    def search_fts(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """BM25 full-text search across chunk text via SQLite FTS5.

        Returns a list of dicts with keys: chunk_id, snippet, rank.
        Results are ordered by BM25 rank (most relevant first).
        """
        try:
            rows = self._conn.execute(
                """
                SELECT
                    chunk_id,
                    snippet(chunks_fts, 1, '', '', '...', 32) AS snippet,
                    rank
                FROM chunks_fts
                WHERE text MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            # FTS5 MATCH can raise if the query string is malformed; degrade gracefully.
            return []

    def ensure_fts_populated(self) -> int:
        """Backfill the FTS5 index from the metadata JSON blob in the chunks table.

        Should be called once after an upgrade to populate an existing database.
        Skips chunks that are already indexed.  Returns the number of newly
        indexed chunks.
        """
        already: set[int] = {
            row[0]
            for row in self._conn.execute("SELECT chunk_id FROM chunks_fts").fetchall()
        }
        pairs = self.get_all_chunk_texts()
        count = 0
        for chunk_id, text in pairs:
            if chunk_id in already:
                continue
            self._conn.execute(
                "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                (chunk_id, text),
            )
            count += 1
        if count:
            self._conn.commit()
        return count

    def get_chunk(self, chunk_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_chunks_for_document(self, document_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY position",
            (document_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Chunk hierarchy
    # ------------------------------------------------------------------

    def add_chunk_hierarchy(
        self,
        chunk_id: int,
        parent_chunk_id: int | None,
        level: int = 0,
        position: int = 0,
        heading: str | None = None,
    ) -> None:
        """Insert or replace a hierarchy record for a chunk.

        Parameters
        ----------
        chunk_id:
            The chunk whose position in the hierarchy is being recorded.
        parent_chunk_id:
            The enclosing chunk (None for top-level / document-level chunks).
        level:
            0=document, 1=section, 2=subsection, 3=paragraph.
        position:
            Zero-based order within the parent.
        heading:
            Optional heading text extracted from the source.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO chunk_hierarchy
                (chunk_id, parent_chunk_id, level, position, heading)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chunk_id, parent_chunk_id, level, position, heading),
        )
        self._conn.commit()

    def get_sibling_chunks(self, chunk_id: int) -> list[int]:
        """Return the IDs of all chunks that share the same parent as chunk_id.

        The given chunk_id itself is excluded from the result.
        Returns an empty list if the chunk has no hierarchy record or no parent.
        """
        row = self._conn.execute(
            "SELECT parent_chunk_id FROM chunk_hierarchy WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None or row["parent_chunk_id"] is None:
            return []
        parent_id = row["parent_chunk_id"]
        rows = self._conn.execute(
            """
            SELECT chunk_id FROM chunk_hierarchy
            WHERE parent_chunk_id = ? AND chunk_id != ?
            ORDER BY position
            """,
            (parent_id, chunk_id),
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    def get_parent_chunk(self, chunk_id: int) -> int | None:
        """Return the parent chunk ID, or None if this is a top-level chunk."""
        row = self._conn.execute(
            "SELECT parent_chunk_id FROM chunk_hierarchy WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return row["parent_chunk_id"]

    def get_chunk_by_id(self, chunk_id: int) -> dict[str, Any] | None:
        """Return a chunk row including its metadata JSON, or None."""
        return self.get_chunk(chunk_id)

    def get_all_chunk_texts(self) -> list[tuple[int, str]]:
        """Return (chunk_id, text) for every chunk stored in metadata.

        The text is pulled from the JSON metadata blob written by the chunker.
        Chunks with no stored text are silently skipped.
        """
        rows = self._conn.execute(
            "SELECT id, metadata FROM chunks ORDER BY id"
        ).fetchall()
        result: list[tuple[int, str]] = []
        for row in rows:
            meta_raw = row["metadata"]
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            text = meta.get("text", "")
            if text:
                result.append((row["id"], text))
        return result

    # ------------------------------------------------------------------
    # Near-duplicate relationships
    # ------------------------------------------------------------------

    def add_near_duplicate(
        self,
        chunk_id: int,
        near_duplicate_of: int,
        similarity_score: float,
    ) -> None:
        """Record that chunk_id is a near-duplicate of near_duplicate_of."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO chunk_near_duplicates
                (chunk_id, near_duplicate_of, similarity_score)
            VALUES (?, ?, ?)
            """,
            (chunk_id, near_duplicate_of, similarity_score),
        )
        self._conn.commit()

    def get_near_duplicates(self, chunk_id: int) -> list[dict[str, Any]]:
        """Return all near-duplicate relationships for chunk_id."""
        rows = self._conn.execute(
            "SELECT * FROM chunk_near_duplicates WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cross-document references
    # ------------------------------------------------------------------

    def add_chunk_reference(
        self,
        source_chunk_id: int,
        reference_text: str,
        reference_type: str,
        target_document_id: int | None = None,
        target_chunk_id: int | None = None,
        confidence: float = 0.5,
    ) -> None:
        """Store a cross-document reference extracted from a chunk."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO chunk_references
                (source_chunk_id, reference_text, reference_type,
                 target_document_id, target_chunk_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source_chunk_id,
                reference_text,
                reference_type,
                target_document_id,
                target_chunk_id,
                confidence,
            ),
        )
        self._conn.commit()

    def get_chunk_references(self, source_chunk_id: int) -> list[dict[str, Any]]:
        """Return all references originating from a chunk."""
        rows = self._conn.execute(
            "SELECT * FROM chunk_references WHERE source_chunk_id = ?",
            (source_chunk_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_child_chunks(self, chunk_id: int) -> list[int]:
        """Return the IDs of all direct children of chunk_id, ordered by position."""
        rows = self._conn.execute(
            """
            SELECT chunk_id FROM chunk_hierarchy
            WHERE parent_chunk_id = ?
            ORDER BY position
            """,
            (chunk_id,),
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    # ------------------------------------------------------------------
    # Emails
    # ------------------------------------------------------------------

    def add_email(
        self,
        thread_id: str,
        message_id: str | None,
        from_addr: str | None,
        to_addr: str | None,
        cc_addr: str | None,
        subject: str | None,
        date: str | None,
        labels: list[str] | None = None,
        chunk_id: int | None = None,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO emails
                (thread_id, message_id, from_addr, to_addr, cc_addr,
                 subject, date, labels, chunk_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                message_id,
                from_addr,
                to_addr,
                cc_addr,
                subject,
                date,
                json.dumps(labels or []),
                chunk_id,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def search_emails(
        self,
        from_addr: str | None = None,
        subject_contains: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Simple metadata filter for emails."""
        clauses: list[str] = []
        params: list[Any] = []

        if from_addr:
            clauses.append("from_addr LIKE ?")
            params.append(f"%{from_addr}%")
        if subject_contains:
            clauses.append("subject LIKE ?")
            params.append(f"%{subject_contains}%")
        if since:
            clauses.append("date >= ?")
            params.append(since)
        if until:
            clauses.append("date <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM emails {where} ORDER BY date DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Sync state
    # ------------------------------------------------------------------

    def upsert_sync_state(
        self,
        source: str,
        cursor: str | None,
        items_processed: int,
        status: str = "idle",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO sync_state (source, cursor, items_processed, status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                last_sync_at    = datetime('now'),
                cursor          = excluded.cursor,
                items_processed = excluded.items_processed,
                status          = excluded.status
            """,
            (source, cursor, items_processed, status),
        )
        self._conn.commit()

    def get_sync_state(self, source: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM sync_state WHERE source = ?", (source,)
        ).fetchone()
        return dict(row) if row else None

    def list_sync_states(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM sync_state").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MetadataStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
