"""Scheduled batch analysis of the full dataset.

Computes aggregates and stores them in SQLite for fast retrieval:
  - Communication frequency per entity per month
  - Entity mention counts across all chunks
  - Source type distribution per entity
  - Influence graph (who communicates with whom, scored)
  - Trajectory projections (trend direction + threshold crossing dates)

All results are stored in a single analytics_results table with a
(metric_type, entity_id, period) key so new metric types can be added
without schema migrations.  Influence edges and trajectory projections
have their own dedicated tables in the same analytics.db file.

Usage
-----
    from verra.analytics.batch import BatchAnalytics
    analytics = BatchAnalytics(metadata_store, entity_store, vector_store)
    analytics.run_all()
    results = analytics.get_analytics("entity_mention_counts")
"""


from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from verra.store.entities import EntityStore
from verra.store.metadata import MetadataStore
from verra.store.vector import VectorStore


_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS analytics_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT    NOT NULL,
    entity_id   INTEGER,            -- NULL for system-wide metrics
    period      TEXT,               -- e.g. '2024-03', NULL for all-time
    value_json  TEXT    NOT NULL,   -- JSON-encoded result value
    computed_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_analytics_metric  ON analytics_results(metric_type);
CREATE INDEX IF NOT EXISTS idx_analytics_entity  ON analytics_results(entity_id);
CREATE INDEX IF NOT EXISTS idx_analytics_period  ON analytics_results(period);
"""


class BatchAnalytics:
    """Scheduled batch analysis of the full Verra dataset.

    Computes aggregates and stores pre-computed results in a dedicated
    SQLite database for fast retrieval.

    Parameters
    ----------
    metadata_store:
        The metadata SQLite store (documents, chunks, emails tables).
    entity_store:
        The entity registry (entities, chunk_entities tables).
    vector_store:
        The ChromaDB vector store (used for counts only, not searched here).
    db_path:
        Path for the analytics SQLite database.  Defaults to a file called
        'analytics.db' alongside the metadata database.
    """

    def __init__(
        self,
        metadata_store: MetadataStore,
        entity_store: EntityStore,
        vector_store: VectorStore,
        db_path: Path | str | None = None,
    ) -> None:
        self._metadata_store = metadata_store
        self._entity_store = entity_store
        self._vector_store = vector_store

        # Derive a default db_path from the metadata store's path when not given.
        if db_path is None:
            meta_path = Path(str(metadata_store.db_path))
            db_path = meta_path.parent / "analytics.db"

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # Ensure influence table exists.
        from verra.analytics.influence import ensure_influence_table
        ensure_influence_table(self._conn)

    # ------------------------------------------------------------------
    # Top-level runner
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        """Run every available analytics computation and persist results.

        Call this via the CLI command `verra analytics run` or from a
        scheduled job.  Existing results are replaced on each run so the
        analytics table always reflects the current state of the data.

        Runs:
          - entity_mention_counts
          - communication_frequency
          - source_distribution
          - influence_analysis  (influence_edges table)
          - trajectory_analysis (trajectory_projections table)
        """
        self.compute_entity_mention_counts()
        self.compute_communication_frequency()
        self.compute_source_distribution()
        self.run_influence_analysis()

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def compute_entity_mention_counts(self) -> None:
        """Count total chunk mentions per entity and persist results.

        Queries the chunk_entities junction table and groups by entity_id.
        Stores one analytics row per entity with value_json = {"count": N}.
        """
        # Use the entity store's own connection which already has chunk_entities.
        rows = self._entity_store._conn.execute(
            """
            SELECT ce.entity_id, COUNT(*) as cnt
            FROM chunk_entities ce
            GROUP BY ce.entity_id
            ORDER BY cnt DESC
            """
        ).fetchall()

        # Clear previous results for this metric type.
        self._conn.execute(
            "DELETE FROM analytics_results WHERE metric_type = 'entity_mention_counts'"
        )

        self._conn.executemany(
            """
            INSERT INTO analytics_results (metric_type, entity_id, period, value_json)
            VALUES ('entity_mention_counts', ?, NULL, ?)
            """,
            [(r["entity_id"], json.dumps({"count": r["cnt"]})) for r in rows],
        )
        self._conn.commit()

    def compute_communication_frequency(self) -> None:
        """Count chunks per entity per calendar month.

        Joins chunk_entities → chunks and groups by (entity_id, month).
        Stores one row per (entity_id, period) with value_json = {"count": N}.
        """
        # We need both the entity store connection (chunk_entities) and the
        # metadata store connection (chunks with created_at).
        # Build a temporary ATTACH-free query using Python-side join.

        # Fetch all (chunk_id, entity_id) links.
        ce_rows = self._entity_store._conn.execute(
            "SELECT chunk_id, entity_id FROM chunk_entities"
        ).fetchall()

        if not ce_rows:
            return

        # Fetch chunk created_at for the relevant chunk_ids.
        chunk_ids = list({r["chunk_id"] for r in ce_rows})
        placeholders = ",".join("?" * len(chunk_ids))
        chunk_rows = self._metadata_store._conn.execute(
            f"SELECT id, created_at FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        chunk_date: dict[int, str] = {r["id"]: r["created_at"][:7] for r in chunk_rows}  # 'YYYY-MM'

        # Aggregate in Python.
        freq: dict[tuple[int, str], int] = {}
        for row in ce_rows:
            cid = row["chunk_id"]
            eid = row["entity_id"]
            month = chunk_date.get(cid)
            if month is None:
                continue
            key = (eid, month)
            freq[key] = freq.get(key, 0) + 1

        self._conn.execute(
            "DELETE FROM analytics_results WHERE metric_type = 'communication_frequency'"
        )
        self._conn.executemany(
            """
            INSERT INTO analytics_results (metric_type, entity_id, period, value_json)
            VALUES ('communication_frequency', ?, ?, ?)
            """,
            [
                (eid, month, json.dumps({"count": cnt}))
                for (eid, month), cnt in sorted(freq.items())
            ],
        )
        self._conn.commit()

    def compute_source_distribution(self) -> None:
        """For each entity, compute what percentage of chunks come from each source type.

        Stores one row per entity with value_json = {"folder": 0.6, "gmail": 0.4, ...}.
        """
        # Get all (chunk_id, entity_id) links.
        ce_rows = self._entity_store._conn.execute(
            "SELECT chunk_id, entity_id FROM chunk_entities"
        ).fetchall()

        if not ce_rows:
            return

        chunk_ids = list({r["chunk_id"] for r in ce_rows})
        placeholders = ",".join("?" * len(chunk_ids))

        # Pull source_type from the chunks → documents join.
        chunk_source_rows = self._metadata_store._conn.execute(
            f"""
            SELECT c.id AS chunk_id, d.source_type
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
        chunk_source: dict[int, str] = {r["chunk_id"]: r["source_type"] for r in chunk_source_rows}

        # Aggregate source counts per entity.
        from collections import defaultdict
        entity_sources: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in ce_rows:
            cid = row["chunk_id"]
            eid = row["entity_id"]
            source = chunk_source.get(cid, "unknown")
            entity_sources[eid][source] += 1

        self._conn.execute(
            "DELETE FROM analytics_results WHERE metric_type = 'source_distribution'"
        )
        rows_to_insert: list[tuple[int, None, str]] = []
        for eid, source_counts in entity_sources.items():
            total = sum(source_counts.values())
            distribution = {src: round(cnt / total, 4) for src, cnt in source_counts.items()}
            rows_to_insert.append((eid, None, json.dumps(distribution)))

        self._conn.executemany(
            """
            INSERT INTO analytics_results (metric_type, entity_id, period, value_json)
            VALUES ('source_distribution', ?, ?, ?)
            """,
            rows_to_insert,
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Influence graph
    # ------------------------------------------------------------------

    def run_influence_analysis(self) -> int:
        """Compute influence graph from communication patterns.

        Returns the number of influence edges computed.
        """
        from verra.analytics.influence import (
            compute_influence_graph,
            persist_influence_edges,
        )

        edges = compute_influence_graph(
            entity_store=self._entity_store,
            metadata_store=self._metadata_store,
            analysis_store=None,  # type: ignore[arg-type]
        )
        persist_influence_edges(self._conn, edges)
        return len(edges)

    def get_influence_edges(self) -> list[dict[str, Any]]:
        """Retrieve all persisted influence edges."""
        from verra.analytics.influence import get_influence_edges
        return get_influence_edges(self._conn)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_analytics(
        self,
        metric_type: str,
        entity_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve pre-computed analytics results.

        Parameters
        ----------
        metric_type:
            One of 'entity_mention_counts', 'communication_frequency',
            'source_distribution'.
        entity_id:
            If provided, return only rows for this entity.  If None,
            return all rows for the metric.

        Returns
        -------
        List of result dicts with keys: id, metric_type, entity_id,
        period, value_json (parsed as a Python object), computed_at.
        """
        if entity_id is not None:
            rows = self._conn.execute(
                """
                SELECT id, metric_type, entity_id, period, value_json, computed_at
                FROM analytics_results
                WHERE metric_type = ? AND entity_id = ?
                ORDER BY period
                """,
                (metric_type, entity_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, metric_type, entity_id, period, value_json, computed_at
                FROM analytics_results
                WHERE metric_type = ?
                ORDER BY entity_id, period
                """,
                (metric_type,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["value_json"] = json.loads(d["value_json"])
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BatchAnalytics":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
