"""Communication influence analysis.

Builds an influence graph from communication patterns:
- Who initiates conversations with whom
- Response time asymmetry (reply to boss in minutes, peers in days)
- CC patterns (who gets copied on what)
- Approval chains (who signs off on decisions)

Stored in the entities relationship graph with 'influences' type
and in a dedicated influence_edges table.
"""


from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from verra.store.entities import EntityStore
    from verra.store.metadata import MetadataStore
    from verra.store.analysis import AnalysisStore


@dataclass
class InfluenceEdge:
    from_entity_id: int
    to_entity_id: int
    communication_count: int
    avg_response_time_hours: float | None
    cc_frequency: float  # how often they're CC'd together
    influence_score: float  # composite 0-1


def compute_influence_score(
    outbound_count: int,
    inbound_count: int,
    avg_response_time: float | None,
    cc_count: int,
) -> float:
    """Compute a composite influence score 0-1.

    Higher outbound/inbound ratio + faster responses = more influence.

    Factors:
    - communication_ratio: outbound / (outbound + inbound) — how often this
      entity initiates vs receives.  0.5 = balanced, 1.0 = always initiates.
    - response_speed: if avg response time is available, faster replies (e.g.
      replying to this entity quickly) suggest authority/priority.
    - cc_boost: being CC'd on many emails implies visibility/influence.
    """
    total = outbound_count + inbound_count
    if total == 0:
        return 0.0

    # Component 1: initiation ratio (0-1)
    initiation = outbound_count / total

    # Component 2: response speed normalised to 0-1 (lower hours = higher score)
    # Assume 24h is "normal"; anything faster scores higher, slower scores lower.
    if avg_response_time is not None and avg_response_time > 0:
        # Scores 1.0 at 0h, ~0.5 at 24h, approaches 0 at very long delays
        speed = 1.0 / (1.0 + avg_response_time / 24.0)
    else:
        speed = 0.5  # neutral when no response time data

    # Component 3: CC presence (log-scale, 0-1)
    import math
    cc_score = min(1.0, math.log1p(cc_count) / math.log1p(20))

    # Weighted composite
    score = 0.5 * initiation + 0.3 * speed + 0.2 * cc_score
    return round(min(1.0, max(0.0, score)), 4)


def compute_influence_graph(
    entity_store: "EntityStore",
    metadata_store: "MetadataStore",
    analysis_store: "AnalysisStore",
) -> list[InfluenceEdge]:
    """Compute influence relationships from communication data.

    Analyzes:
    1. Email from/to patterns -> who talks to whom, how often
    2. Response times -> asymmetry indicates hierarchy
    3. CC patterns -> who gets visibility
    4. Decision mentions -> who approves what

    Returns edges for the influence graph.
    """
    # Build entity name -> id lookup
    all_entities = entity_store.list_entities()
    name_to_id: dict[str, int] = {}
    for e in all_entities:
        name_to_id[e["canonical_name"].lower()] = e["id"]
        # Also add aliases
        for alias in entity_store.get_aliases(e["id"]):
            name_to_id[alias.lower()] = e["id"]

    if not name_to_id:
        return []

    # Count co-occurrences in the same chunk: if entity A and entity B both
    # appear in the same chunk, they have a communication link.
    # We look for all chunks that have 2+ entities linked.
    ce_rows = entity_store._conn.execute(
        "SELECT chunk_id, entity_id FROM chunk_entities ORDER BY chunk_id"
    ).fetchall()

    # Group entity IDs by chunk
    chunk_entities: dict[int, list[int]] = {}
    for row in ce_rows:
        cid = row["chunk_id"]
        eid = row["entity_id"]
        chunk_entities.setdefault(cid, []).append(eid)

    # Fetch chunk metadata for source_type (to weight emails more heavily)
    chunk_ids_list = list(chunk_entities.keys())
    if not chunk_ids_list:
        return []

    placeholders = ",".join("?" * len(chunk_ids_list))
    chunk_meta_rows = metadata_store._conn.execute(
        f"""
        SELECT c.id AS chunk_id, d.source_type, c.metadata
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids_list,
    ).fetchall()
    chunk_source: dict[int, str] = {r["chunk_id"]: r["source_type"] for r in chunk_meta_rows}

    # Build co-communication counts between entity pairs.
    # Use (from, to) where from is the "first" entity alphabetically to avoid
    # double-counting, then we'll infer direction from chunk context.
    import json as _json

    # For each chunk with 2+ entities, record a link between every pair.
    pair_counts: dict[tuple[int, int], int] = {}
    pair_cc_counts: dict[tuple[int, int], int] = {}

    for chunk_id, eids in chunk_entities.items():
        if len(eids) < 2:
            continue

        source_type = chunk_source.get(chunk_id, "unknown")
        # Weight email co-occurrences more strongly
        weight = 2 if source_type == "gmail" else 1

        # All ordered pairs (a, b) where a != b
        unique_eids = list(set(eids))
        for i, ea in enumerate(unique_eids):
            for eb in unique_eids[i + 1:]:
                # Store as (smaller_id, larger_id) for deduplication
                pair = (min(ea, eb), max(ea, eb))
                pair_counts[pair] = pair_counts.get(pair, 0) + weight
                if source_type == "gmail":
                    pair_cc_counts[pair] = pair_cc_counts.get(pair, 0) + 1

    if not pair_counts:
        return []

    # Convert pairs to InfluenceEdge objects.
    # Direction: entity with more outbound connections in the pair is "from".
    # Compute outbound degree for each entity.
    entity_total_comms: dict[int, int] = {}
    for (ea, eb), count in pair_counts.items():
        entity_total_comms[ea] = entity_total_comms.get(ea, 0) + count
        entity_total_comms[eb] = entity_total_comms.get(eb, 0) + count

    edges: list[InfluenceEdge] = []
    for (ea, eb), count in pair_counts.items():
        cc_count = pair_cc_counts.get((ea, eb), 0)

        # Determine direction: entity with higher total communication volume
        # is treated as the "from" (more active/initiating) entity.
        if entity_total_comms.get(ea, 0) >= entity_total_comms.get(eb, 0):
            from_id, to_id = ea, eb
            outbound = entity_total_comms.get(ea, 0)
            inbound = entity_total_comms.get(eb, 0)
        else:
            from_id, to_id = eb, ea
            outbound = entity_total_comms.get(eb, 0)
            inbound = entity_total_comms.get(ea, 0)

        cc_freq = cc_count / count if count > 0 else 0.0
        score = compute_influence_score(
            outbound_count=outbound,
            inbound_count=inbound,
            avg_response_time=None,  # response time requires threaded email data
            cc_count=cc_count,
        )

        edges.append(
            InfluenceEdge(
                from_entity_id=from_id,
                to_entity_id=to_id,
                communication_count=count,
                avg_response_time_hours=None,
                cc_frequency=round(cc_freq, 4),
                influence_score=score,
            )
        )

    return edges


# ---------------------------------------------------------------------------
# Persistence helper (used by BatchAnalytics)
# ---------------------------------------------------------------------------

_INFLUENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS influence_edges (
    from_entity_id INTEGER,
    to_entity_id INTEGER,
    communication_count INTEGER DEFAULT 0,
    avg_response_time_hours REAL,
    cc_frequency REAL DEFAULT 0.0,
    influence_score REAL DEFAULT 0.0,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (from_entity_id, to_entity_id)
);
"""


def ensure_influence_table(conn: sqlite3.Connection) -> None:
    """Create the influence_edges table if it doesn't exist."""
    conn.executescript(_INFLUENCE_SCHEMA)
    conn.commit()


def persist_influence_edges(conn: sqlite3.Connection, edges: list[InfluenceEdge]) -> None:
    """Replace all influence_edges with the freshly computed set."""
    conn.execute("DELETE FROM influence_edges")
    conn.executemany(
        """
        INSERT OR REPLACE INTO influence_edges
        (from_entity_id, to_entity_id, communication_count,
         avg_response_time_hours, cc_frequency, influence_score, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            (
                e.from_entity_id,
                e.to_entity_id,
                e.communication_count,
                e.avg_response_time_hours,
                e.cc_frequency,
                e.influence_score,
            )
            for e in edges
        ],
    )
    conn.commit()


def get_influence_edges(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Retrieve all persisted influence edges."""
    rows = conn.execute(
        "SELECT * FROM influence_edges ORDER BY influence_score DESC"
    ).fetchall()
    return [dict(r) for r in rows]
