"""Entity registry and relationship storage.

Stores resolved entities (people, companies, projects), their aliases,
chunk-entity associations, and inter-entity relationships in SQLite.

Schema:
  entities       — canonical entity records (id, name, type)
  entity_aliases — variant names mapping to entity IDs
  chunk_entities — junction table linking chunks to entities
  relationships  — entity-to-entity links with type and provenance
"""


from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class EntityStore:
    """SQLite-backed entity registry with alias resolution and relationships."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "EntityStore":
        """Create an EntityStore that shares an existing SQLite connection.

        Used by DatabaseManager so all core stores share core.db.
        The caller is responsible for connection lifecycle.
        """
        instance = object.__new__(cls)
        instance._db_path = ":memory:"
        instance._conn = conn
        instance._create_tables()
        return instance

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,  -- 'person', 'company', 'project', 'product', 'location'
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS entity_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alias TEXT NOT NULL COLLATE NOCASE,
                entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                UNIQUE(alias, entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_alias_lookup ON entity_aliases(alias COLLATE NOCASE);

            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                PRIMARY KEY (chunk_id, entity_id)
            );

            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_a INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                relationship_type TEXT NOT NULL,  -- 'works_at', 'manages', 'client_of', 'reports_to', 'involved_in'
                entity_b INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                source_chunk_id INTEGER,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity_a, relationship_type, entity_b)
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def add_entity(self, canonical_name: str, entity_type: str, aliases: list[str] | None = None) -> int:
        """Create an entity and register its aliases. Returns the entity ID.

        If an entity with this canonical name and type already exists, returns
        the existing ID and merges any new aliases.
        """
        existing = self.resolve(canonical_name)
        if existing:
            eid = existing["id"]
        else:
            cur = self._conn.execute(
                "INSERT INTO entities (canonical_name, entity_type) VALUES (?, ?)",
                (canonical_name, entity_type),
            )
            eid = cur.lastrowid
            # Add the canonical name as an alias too
            self._conn.execute(
                "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
                (canonical_name, eid),
            )

        # Add extra aliases
        if aliases:
            for alias in aliases:
                self._conn.execute(
                    "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
                    (alias.strip(), eid),
                )

        self._conn.commit()
        return eid

    def resolve(self, name: str) -> dict[str, Any] | None:
        """Look up an entity by any of its aliases. Returns entity dict or None."""
        row = self._conn.execute(
            """
            SELECT e.id, e.canonical_name, e.entity_type
            FROM entities e
            JOIN entity_aliases a ON a.entity_id = e.id
            WHERE a.alias = ? COLLATE NOCASE
            LIMIT 1
            """,
            (name.strip(),),
        ).fetchone()
        return dict(row) if row else None

    def get_entity(self, entity_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, canonical_name, entity_type FROM entities WHERE id = ?",
            (entity_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_aliases(self, entity_id: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()
        return [r["alias"] for r in rows]

    def list_entities(self, entity_type: str | None = None) -> list[dict[str, Any]]:
        if entity_type:
            rows = self._conn.execute(
                "SELECT id, canonical_name, entity_type FROM entities WHERE entity_type = ? ORDER BY canonical_name",
                (entity_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, canonical_name, entity_type FROM entities ORDER BY entity_type, canonical_name"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Chunk-Entity links
    # ------------------------------------------------------------------

    def link_chunk(self, chunk_id: int, entity_id: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
            (chunk_id, entity_id),
        )
        self._conn.commit()

    def link_chunk_batch(self, chunk_id: int, entity_ids: list[int]) -> None:
        self._conn.executemany(
            "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity_id) VALUES (?, ?)",
            [(chunk_id, eid) for eid in entity_ids],
        )
        self._conn.commit()

    def get_chunks_for_entity(self, entity_id: int) -> list[int]:
        """Return all chunk IDs tagged with this entity."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunk_entities WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()
        return [r["chunk_id"] for r in rows]

    def get_entities_for_chunk(self, chunk_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT e.id, e.canonical_name, e.entity_type
            FROM entities e
            JOIN chunk_entities ce ON ce.entity_id = e.id
            WHERE ce.chunk_id = ?
            """,
            (chunk_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        entity_a: int,
        relationship_type: str,
        entity_b: int,
        source_chunk_id: int | None = None,
    ) -> None:
        """Add or update a relationship between two entities."""
        self._conn.execute(
            """
            INSERT INTO relationships (entity_a, relationship_type, entity_b, source_chunk_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_a, relationship_type, entity_b) DO UPDATE SET
                last_seen = CURRENT_TIMESTAMP,
                source_chunk_id = COALESCE(excluded.source_chunk_id, relationships.source_chunk_id)
            """,
            (entity_a, relationship_type, entity_b, source_chunk_id),
        )
        self._conn.commit()

    def get_relationships(self, entity_id: int) -> list[dict[str, Any]]:
        """Get all relationships where entity_id is either side."""
        rows = self._conn.execute(
            """
            SELECT r.id, r.entity_a, r.relationship_type, r.entity_b,
                   ea.canonical_name as name_a, eb.canonical_name as name_b,
                   ea.entity_type as type_a, eb.entity_type as type_b
            FROM relationships r
            JOIN entities ea ON ea.id = r.entity_a
            JOIN entities eb ON eb.id = r.entity_b
            WHERE r.entity_a = ? OR r.entity_b = ?
            ORDER BY r.last_seen DESC
            """,
            (entity_id, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_related_entities(self, entity_id: int) -> list[dict[str, Any]]:
        """Get all entities related to the given entity (deduplicated)."""
        rows = self._conn.execute(
            """
            SELECT DISTINCT e.id, e.canonical_name, e.entity_type, r.relationship_type
            FROM relationships r
            JOIN entities e ON (e.id = r.entity_b AND r.entity_a = ?)
                            OR (e.id = r.entity_a AND r.entity_b = ?)
            WHERE e.id != ?
            """,
            (entity_id, entity_id, entity_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_entity(self, entity_id: int) -> None:
        """Delete an entity and cascade to aliases, links, and relationships."""
        self._conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        self._conn.commit()

    def unlink_chunks_for_document(self, chunk_ids: list[int]) -> None:
        """Remove all entity links for a set of chunk IDs (used when re-ingesting)."""
        if not chunk_ids:
            return
        placeholders = ",".join("?" * len(chunk_ids))
        self._conn.execute(
            f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
