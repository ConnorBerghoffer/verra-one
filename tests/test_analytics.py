"""Tests for BatchAnalytics in verra.analytics.batch."""

from __future__ import annotations

import pytest
from pathlib import Path

from verra.analytics.batch import BatchAnalytics
from verra.ingest.chunking import Chunk
from verra.store.entities import EntityStore
from verra.store.metadata import MetadataStore
from verra.store.vector import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metadata_store(tmp_path: Path) -> MetadataStore:
    s = MetadataStore(tmp_path / "metadata.db")
    yield s
    s.close()


@pytest.fixture
def entity_store(tmp_path: Path) -> EntityStore:
    s = EntityStore(tmp_path / "entities.db")
    yield s
    s.close()


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    s = VectorStore(tmp_path / "chroma")
    yield s


@pytest.fixture
def batch(
    metadata_store: MetadataStore,
    entity_store: EntityStore,
    vector_store: VectorStore,
    tmp_path: Path,
) -> BatchAnalytics:
    ba = BatchAnalytics(
        metadata_store=metadata_store,
        entity_store=entity_store,
        vector_store=vector_store,
        db_path=tmp_path / "analytics.db",
    )
    yield ba
    ba.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate(
    metadata_store: MetadataStore,
    entity_store: EntityStore,
    num_docs: int = 2,
    chunks_per_doc: int = 3,
) -> dict:
    """Populate the stores with synthetic data and return useful IDs."""
    entity_id = entity_store.add_entity("Alice", "person")
    entity_id2 = entity_store.add_entity("Acme Corp", "company")

    all_chunk_ids: list[int] = []
    for i in range(num_docs):
        doc_id = metadata_store.add_document(
            file_path=f"/docs/doc{i}.txt",
            file_name=f"doc{i}.txt",
            source_type="folder",
            format="txt",
            content_hash=f"hash{i}",
        )
        chunks = [Chunk(text=f"doc{i} chunk{j}", token_count=5) for j in range(chunks_per_doc)]
        chunk_ids = metadata_store.add_chunks(doc_id, chunks)
        all_chunk_ids.extend(chunk_ids)

        # Link entity to all chunks of first doc; entity2 to second.
        if i == 0:
            for cid in chunk_ids:
                entity_store.link_chunk(cid, entity_id)
        else:
            for cid in chunk_ids:
                entity_store.link_chunk(cid, entity_id2)

    return {
        "entity_id": entity_id,
        "entity_id2": entity_id2,
        "chunk_ids": all_chunk_ids,
    }


# ---------------------------------------------------------------------------
# compute_entity_mention_counts
# ---------------------------------------------------------------------------


def test_entity_mention_counts_basic(batch, metadata_store, entity_store) -> None:
    ids = _populate(metadata_store, entity_store)
    batch.compute_entity_mention_counts()

    results = batch.get_analytics("entity_mention_counts", entity_id=ids["entity_id"])
    assert len(results) == 1
    assert results[0]["value_json"]["count"] == 3  # 1 doc × 3 chunks


def test_entity_mention_counts_all(batch, metadata_store, entity_store) -> None:
    _populate(metadata_store, entity_store, num_docs=2, chunks_per_doc=3)
    batch.compute_entity_mention_counts()

    all_results = batch.get_analytics("entity_mention_counts")
    assert len(all_results) == 2  # one row per entity
    totals = {r["entity_id"]: r["value_json"]["count"] for r in all_results}
    assert all(v == 3 for v in totals.values())


def test_entity_mention_counts_no_data(batch) -> None:
    batch.compute_entity_mention_counts()
    assert batch.get_analytics("entity_mention_counts") == []


def test_entity_mention_counts_replaced_on_rerun(batch, metadata_store, entity_store) -> None:
    ids = _populate(metadata_store, entity_store)
    batch.compute_entity_mention_counts()
    batch.compute_entity_mention_counts()  # second run should not duplicate rows
    results = batch.get_analytics("entity_mention_counts")
    # Should be exactly 2 rows (one per entity), not 4.
    assert len(results) == 2


# ---------------------------------------------------------------------------
# compute_communication_frequency
# ---------------------------------------------------------------------------


def test_communication_frequency_basic(batch, metadata_store, entity_store) -> None:
    ids = _populate(metadata_store, entity_store)
    batch.compute_communication_frequency()

    results = batch.get_analytics("communication_frequency", entity_id=ids["entity_id"])
    assert len(results) >= 1
    total = sum(r["value_json"]["count"] for r in results)
    assert total == 3  # 3 chunks linked to entity 1


def test_communication_frequency_no_data(batch) -> None:
    batch.compute_communication_frequency()
    assert batch.get_analytics("communication_frequency") == []


def test_communication_frequency_replaced_on_rerun(batch, metadata_store, entity_store) -> None:
    _populate(metadata_store, entity_store)
    batch.compute_communication_frequency()
    batch.compute_communication_frequency()
    results = batch.get_analytics("communication_frequency")
    # Running again should not duplicate rows; each (entity, period) is unique.
    seen = set()
    for r in results:
        key = (r["entity_id"], r["period"])
        assert key not in seen, f"Duplicate row: {key}"
        seen.add(key)


# ---------------------------------------------------------------------------
# compute_source_distribution
# ---------------------------------------------------------------------------


def test_source_distribution_basic(batch, metadata_store, entity_store) -> None:
    ids = _populate(metadata_store, entity_store)
    batch.compute_source_distribution()

    results = batch.get_analytics("source_distribution", entity_id=ids["entity_id"])
    assert len(results) == 1
    dist = results[0]["value_json"]
    assert "folder" in dist
    assert abs(dist["folder"] - 1.0) < 0.01  # 100% from folder


def test_source_distribution_no_data(batch) -> None:
    batch.compute_source_distribution()
    assert batch.get_analytics("source_distribution") == []


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


def test_run_all_populates_all_metrics(batch, metadata_store, entity_store) -> None:
    _populate(metadata_store, entity_store)
    batch.run_all()

    assert len(batch.get_analytics("entity_mention_counts")) > 0
    assert len(batch.get_analytics("communication_frequency")) > 0
    assert len(batch.get_analytics("source_distribution")) > 0


# ---------------------------------------------------------------------------
# get_analytics
# ---------------------------------------------------------------------------


def test_get_analytics_returns_parsed_value_json(batch, metadata_store, entity_store) -> None:
    ids = _populate(metadata_store, entity_store)
    batch.compute_entity_mention_counts()

    results = batch.get_analytics("entity_mention_counts", entity_id=ids["entity_id"])
    assert isinstance(results[0]["value_json"], dict)


def test_get_analytics_unknown_metric(batch) -> None:
    assert batch.get_analytics("nonexistent_metric") == []


# ---------------------------------------------------------------------------
# Default db_path derivation
# ---------------------------------------------------------------------------


def test_default_db_path_derived_from_metadata_store(
    metadata_store: MetadataStore,
    entity_store: EntityStore,
    vector_store: VectorStore,
    tmp_path: Path,
) -> None:
    """BatchAnalytics should create analytics.db next to the metadata db."""
    ba = BatchAnalytics(metadata_store, entity_store, vector_store, db_path=None)
    expected = Path(str(metadata_store.db_path)).parent / "analytics.db"
    assert ba._db_path == expected
    ba.close()
