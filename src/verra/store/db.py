"""Database manager — consolidates SQLite into core.db + analysis.db."""


from __future__ import annotations

import sqlite3
from pathlib import Path


class DatabaseManager:
    """Manages consolidated SQLite databases for Verra.

    Two databases:
      core.db     — entities, state, metadata, memory (hot path, queried on every chat)
      analysis.db — analysis, provenance, analytics (enrichment data, queried less frequently)

    Parameters
    ----------
    data_dir:
        Directory where core.db and analysis.db will be created.
        The chroma/ subdirectory for vector embeddings is left unaffected.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety: check_same_thread=False is required because the sync
        # daemon and the main CLI thread may share a DatabaseManager instance.
        # WAL mode allows concurrent readers safely.  Writes are serialized by
        # SQLite's internal locking.  If concurrent write paths are added in
        # future, a threading.Lock should guard write operations.
        self.core = sqlite3.connect(
            str(self.data_dir / "core.db"),
            check_same_thread=False,
        )
        self.core.row_factory = sqlite3.Row
        self.core.execute("PRAGMA journal_mode=WAL")
        self.core.execute("PRAGMA foreign_keys=ON")

        self.analysis = sqlite3.connect(
            str(self.data_dir / "analysis.db"),
            check_same_thread=False,
        )
        self.analysis.row_factory = sqlite3.Row
        self.analysis.execute("PRAGMA journal_mode=WAL")
        self.analysis.execute("PRAGMA foreign_keys=ON")

        # Initialise all table schemas in their respective databases
        self._init_core_tables()
        self._init_analysis_tables()

    # ------------------------------------------------------------------
    # Schema initialisation — core.db
    # ------------------------------------------------------------------

    def _init_core_tables(self) -> None:
        """Create all core.db tables (metadata, entities, state, memory)."""
        # MetadataStore tables
        self.core.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path      TEXT    NOT NULL,
                file_name      TEXT    NOT NULL,
                source_type    TEXT    NOT NULL DEFAULT 'folder',
                format         TEXT,
                content_hash   TEXT    NOT NULL,
                page_count     INTEGER NOT NULL DEFAULT 1,
                extra_metadata TEXT,
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
                metadata         TEXT,
                authority_weight INTEGER NOT NULL DEFAULT 50,
                valid_from       TIMESTAMP,
                valid_until      TIMESTAMP,
                created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id);

            CREATE TABLE IF NOT EXISTS chunk_hierarchy (
                chunk_id        INTEGER PRIMARY KEY,
                parent_chunk_id INTEGER,
                level           INTEGER NOT NULL DEFAULT 0,
                position        INTEGER NOT NULL DEFAULT 0,
                heading         TEXT
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
                labels      TEXT,
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
                target_document_id INTEGER,
                target_chunk_id INTEGER,
                reference_type TEXT,
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
        """)

        # EntityStore tables
        self.core.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS entity_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alias TEXT NOT NULL COLLATE NOCASE,
                entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                UNIQUE(alias, entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_alias_lookup
                ON entity_aliases(alias COLLATE NOCASE);

            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                PRIMARY KEY (chunk_id, entity_id)
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_a INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                relationship_type TEXT NOT NULL,
                entity_b INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                source_chunk_id INTEGER,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity_a, relationship_type, entity_b)
            );
        """)

        # MemoryStore tables
        self.core.executescript("""
            CREATE TABLE IF NOT EXISTS memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                category   TEXT NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                source     TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_key ON memory(category, key);

            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
        """)

        self.core.commit()

    # ------------------------------------------------------------------
    # Schema initialisation — analysis.db
    # ------------------------------------------------------------------

    def _init_analysis_tables(self) -> None:
        """Create all analysis.db tables (analysis, provenance, coverage, assertions)."""
        # AnalysisStore tables
        self.analysis.executescript("""
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

        # ProvenanceStore tables
        self.analysis.executescript("""
            CREATE TABLE IF NOT EXISTS chunk_provenance (
                chunk_id INTEGER PRIMARY KEY,
                source_file_path TEXT NOT NULL,
                source_type TEXT NOT NULL,
                start_location TEXT,
                end_location TEXT,
                page_start INTEGER,
                page_end INTEGER,
                paragraph_start INTEGER,
                paragraph_end INTEGER,
                message_id TEXT,
                thread_id TEXT,
                position_in_thread INTEGER,
                row_start INTEGER,
                row_end INTEGER,
                extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS source_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                content_hash TEXT,
                last_ingested_batch_id INTEGER,
                original_encoding TEXT DEFAULT 'utf-8',
                language TEXT DEFAULT 'en',
                file_size_bytes INTEGER,
                first_ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ingestion_batches (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                source_description TEXT,
                chunk_count INTEGER DEFAULT 0,
                file_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running',
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS superseded_chunks (
                original_chunk_id INTEGER NOT NULL,
                original_text TEXT,
                original_metadata TEXT,
                superseded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                superseded_by_batch_id INTEGER,
                retention_until TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                entity_name TEXT,
                gap_type TEXT NOT NULL,
                description TEXT,
                detected_from_chunk_id INTEGER,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved INTEGER DEFAULT 0,
                resolved_at TIMESTAMP,
                resolution_notes TEXT
            );

            CREATE TABLE IF NOT EXISTS communication_profiles (
                entity_id INTEGER PRIMARY KEY,
                entity_name TEXT,
                avg_sentiment REAL DEFAULT 0.0,
                avg_response_time_hours REAL,
                typical_message_length INTEGER,
                message_count INTEGER DEFAULT 0,
                positive_pct REAL DEFAULT 0.0,
                negative_pct REAL DEFAULT 0.0,
                escalation_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                entity_ids TEXT DEFAULT '[]',
                description TEXT,
                timestamp TIMESTAMP,
                source_chunk_id INTEGER,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS emergent_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT NOT NULL,
                entity_ids TEXT DEFAULT '[]',
                direction TEXT,
                confidence REAL DEFAULT 0.0,
                supporting_metrics TEXT DEFAULT '{}',
                description TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_provenance_file
                ON chunk_provenance(source_file_path);
            CREATE INDEX IF NOT EXISTS idx_batches_status
                ON ingestion_batches(status);
            CREATE INDEX IF NOT EXISTS idx_gaps_entity
                ON knowledge_gaps(entity_id);
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_insights_signal
                ON emergent_insights(signal_type);
        """)

        self.analysis.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close both database connections."""
        self.core.close()
        self.analysis.close()

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
