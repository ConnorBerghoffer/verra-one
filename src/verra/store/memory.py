"""Persistent memory store (SQLite).

Tracks:
  - memory     — agent-learned facts, preferences, dismissed items
  - conversations — chat session metadata
  - messages   — individual turns within a conversation
"""


from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memory (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    category   TEXT NOT NULL,   -- 'preference' | 'fact' | 'dismissed' | 'context'
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    source     TEXT,            -- conversation_id or action that created this
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT             -- NULL = permanent
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
    role            TEXT    NOT NULL,  -- 'user' | 'assistant' | 'system'
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    feedback        TEXT                -- 'positive' | 'negative' | NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
"""

# Migration: add feedback column to existing databases that pre-date the schema change.
_MIGRATE_MESSAGES_FEEDBACK = """
ALTER TABLE messages ADD COLUMN feedback TEXT;
"""


class MemoryStore:
    """Persistent memory: learned facts, preferences, conversation history."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._apply_migrations()
        self._conn.commit()

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "MemoryStore":
        """Create a MemoryStore that shares an existing SQLite connection.

        Used by DatabaseManager so all core stores share core.db.
        The caller is responsible for connection lifecycle.
        """
        instance = object.__new__(cls)
        instance.db_path = Path(":memory:")  # sentinel — not a real path
        instance._conn = conn
        instance._conn.executescript(_SCHEMA)
        instance._apply_migrations()
        instance._conn.commit()
        return instance

    def _apply_migrations(self) -> None:
        """Apply incremental schema migrations gracefully."""
        # Add feedback column if it doesn't exist yet.
        try:
            self._conn.execute(_MIGRATE_MESSAGES_FEEDBACK)
        except Exception:
            pass  # column already exists — that's fine

    # ------------------------------------------------------------------
    # Memory (facts, preferences, etc.)
    # ------------------------------------------------------------------

    def set_memory(
        self,
        category: str,
        key: str,
        value: str,
        source: str | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Upsert a memory entry."""
        self._conn.execute(
            """
            INSERT INTO memory (category, key, value, source, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET
                value      = excluded.value,
                source     = excluded.source,
                expires_at = excluded.expires_at
            """,
            (
                category,
                key,
                value,
                source,
                expires_at.isoformat() if expires_at else None,
            ),
        )
        self._conn.commit()

    def get_memory(self, category: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM memory WHERE category = ? AND key = ?",
            (category, key),
        ).fetchone()
        return row["value"] if row else None

    def list_memory(self, category: str | None = None) -> list[dict[str, Any]]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM memory WHERE category = ? ORDER BY created_at DESC",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory ORDER BY category, created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, category: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM memory WHERE category = ? AND key = ?", (category, key)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def new_conversation(self, title: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title,)
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_conversations(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self, conversation_id: int, role: str, content: str
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        # Update conversation updated_at
        self._conn.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_messages(
        self, conversation_id: int, limit: int = 50
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_feedback(self, conversation_id: int, rating: str) -> None:
        """Record user feedback on the last assistant message in a conversation.

        Parameters
        ----------
        conversation_id:
            The conversation to update.
        rating:
            'positive' or 'negative'.
        """
        try:
            self._conn.execute(
                """
                UPDATE messages
                SET feedback = ?
                WHERE id = (
                    SELECT id FROM messages
                    WHERE conversation_id = ? AND role = 'assistant'
                    ORDER BY created_at DESC
                    LIMIT 1
                )
                """,
                (rating, conversation_id),
            )
            self._conn.commit()
        except Exception:
            pass  # non-critical — never crash the chat loop

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
