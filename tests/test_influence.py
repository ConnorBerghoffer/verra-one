"""Tests for verra.analytics.influence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from verra.analytics.influence import (
    InfluenceEdge,
    compute_influence_score,
    ensure_influence_table,
    persist_influence_edges,
    get_influence_edges,
    compute_influence_graph,
)
from verra.store.entities import EntityStore
from verra.store.metadata import MetadataStore
from verra.store.vector import VectorStore
from verra.ingest.chunking import Chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def entity_store(tmp_path: Path) -> EntityStore:
    s = EntityStore(tmp_path / "entities.db")
    yield s
    s.close()


@pytest.fixture
def metadata_store(tmp_path: Path) -> MetadataStore:
    s = MetadataStore(tmp_path / "metadata.db")
    yield s
    s.close()


@pytest.fixture
def analytics_conn(tmp_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_path / "analytics.db"), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_influence_table(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# compute_influence_score
# ---------------------------------------------------------------------------


def test_compute_influence_score_high_activity() -> None:
    """High outbound, fast responses, many CCs -> score near 1."""
    score = compute_influence_score(
        outbound_count=100,
        inbound_count=10,
        avg_response_time=1.0,  # 1 hour — very fast
        cc_count=50,
    )
    assert score > 0.7, f"Expected high score, got {score}"
    assert 0.0 <= score <= 1.0


def test_compute_influence_score_low_activity() -> None:
    """Low outbound, slow responses, no CCs -> score near 0."""
    score = compute_influence_score(
        outbound_count=1,
        inbound_count=100,
        avg_response_time=72.0,  # 3 days — very slow
        cc_count=0,
    )
    assert score < 0.4, f"Expected low score, got {score}"
    assert 0.0 <= score <= 1.0


def test_compute_influence_score_balanced() -> None:
    """Balanced communication -> mid-range score."""
    score = compute_influence_score(
        outbound_count=50,
        inbound_count=50,
        avg_response_time=24.0,
        cc_count=5,
    )
    assert 0.2 <= score <= 0.8


def test_compute_influence_score_zero_total() -> None:
    """No communication -> score = 0."""
    score = compute_influence_score(
        outbound_count=0,
        inbound_count=0,
        avg_response_time=None,
        cc_count=0,
    )
    assert score == 0.0


def test_compute_influence_score_no_response_time() -> None:
    """None response time is handled gracefully with neutral speed."""
    score = compute_influence_score(
        outbound_count=10,
        inbound_count=5,
        avg_response_time=None,
        cc_count=2,
    )
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# InfluenceEdge creation
# ---------------------------------------------------------------------------


def test_influence_edge_creation() -> None:
    edge = InfluenceEdge(
        from_entity_id=1,
        to_entity_id=2,
        communication_count=15,
        avg_response_time_hours=8.0,
        cc_frequency=0.3,
        influence_score=0.65,
    )
    assert edge.from_entity_id == 1
    assert edge.to_entity_id == 2
    assert edge.communication_count == 15
    assert edge.avg_response_time_hours == 8.0
    assert edge.cc_frequency == 0.3
    assert edge.influence_score == 0.65


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persist_and_retrieve_edges(analytics_conn: sqlite3.Connection) -> None:
    edges = [
        InfluenceEdge(1, 2, 10, 4.0, 0.2, 0.75),
        InfluenceEdge(2, 3, 5, None, 0.0, 0.4),
    ]
    persist_influence_edges(analytics_conn, edges)
    retrieved = get_influence_edges(analytics_conn)
    assert len(retrieved) == 2
    scores = {(r["from_entity_id"], r["to_entity_id"]): r["influence_score"] for r in retrieved}
    assert scores[(1, 2)] == pytest.approx(0.75)
    assert scores[(2, 3)] == pytest.approx(0.4)


def test_persist_replaces_old_edges(analytics_conn: sqlite3.Connection) -> None:
    """Re-running should replace all edges, not append."""
    edges1 = [InfluenceEdge(1, 2, 10, None, 0.0, 0.5)]
    persist_influence_edges(analytics_conn, edges1)

    edges2 = [InfluenceEdge(3, 4, 7, None, 0.0, 0.6)]
    persist_influence_edges(analytics_conn, edges2)

    retrieved = get_influence_edges(analytics_conn)
    assert len(retrieved) == 1
    assert retrieved[0]["from_entity_id"] == 3


# ---------------------------------------------------------------------------
# compute_influence_graph (integration)
# ---------------------------------------------------------------------------


def _populate_for_influence(
    metadata_store: MetadataStore,
    entity_store: EntityStore,
) -> tuple[int, int]:
    """Create two entities sharing chunks (simulating communication)."""
    eid_a = entity_store.add_entity("Alice", "person")
    eid_b = entity_store.add_entity("Bob", "person")

    doc_id = metadata_store.add_document(
        file_path="/emails/thread.eml",
        file_name="thread.eml",
        source_type="folder",
        format="eml",
        content_hash="abc123",
    )
    chunks = [Chunk(text=f"message {i}", token_count=5) for i in range(5)]
    chunk_ids = metadata_store.add_chunks(doc_id, chunks)

    # Link both entities to all chunks (they co-appear -> communication link)
    for cid in chunk_ids:
        entity_store.link_chunk(cid, eid_a)
        entity_store.link_chunk(cid, eid_b)

    return eid_a, eid_b


def test_empty_data_returns_empty(
    entity_store: EntityStore,
    metadata_store: MetadataStore,
) -> None:
    """No entities, no edges."""
    edges = compute_influence_graph(
        entity_store=entity_store,
        metadata_store=metadata_store,
        analysis_store=None,  # type: ignore
    )
    assert edges == []


def test_influence_graph_finds_edge(
    entity_store: EntityStore,
    metadata_store: MetadataStore,
) -> None:
    """Two entities co-appearing in chunks should produce an influence edge."""
    _populate_for_influence(metadata_store, entity_store)
    edges = compute_influence_graph(
        entity_store=entity_store,
        metadata_store=metadata_store,
        analysis_store=None,  # type: ignore
    )
    assert len(edges) == 1
    edge = edges[0]
    assert edge.communication_count > 0
    assert 0.0 <= edge.influence_score <= 1.0


def test_influence_graph_no_self_edges(
    entity_store: EntityStore,
    metadata_store: MetadataStore,
) -> None:
    """No edge should have from_entity_id == to_entity_id."""
    _populate_for_influence(metadata_store, entity_store)
    edges = compute_influence_graph(
        entity_store=entity_store,
        metadata_store=metadata_store,
        analysis_store=None,  # type: ignore
    )
    for edge in edges:
        assert edge.from_entity_id != edge.to_entity_id
