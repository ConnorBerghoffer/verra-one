"""Tests for chunk hierarchy methods on MetadataStore."""

from __future__ import annotations

import pytest
from pathlib import Path

from verra.store.metadata import MetadataStore
from verra.ingest.chunking import Chunk


@pytest.fixture
def store(tmp_path: Path) -> MetadataStore:
    s = MetadataStore(tmp_path / "test_hierarchy.db")
    yield s
    s.close()


def _add_doc_and_chunks(store: MetadataStore, n: int = 4) -> tuple[int, list[int]]:
    """Helper: add a document and n chunks, return (doc_id, chunk_ids)."""
    doc_id = store.add_document(
        file_path="/test/doc.txt",
        file_name="doc.txt",
        source_type="folder",
        format="txt",
        content_hash=f"hash{n}",
    )
    chunks = [Chunk(text=f"chunk {i}", token_count=5) for i in range(n)]
    chunk_ids = store.add_chunks(doc_id, chunks)
    return doc_id, chunk_ids


# ---------------------------------------------------------------------------
# add_chunk_hierarchy / get_parent_chunk
# ---------------------------------------------------------------------------


def test_add_and_get_parent(store: MetadataStore) -> None:
    _, chunk_ids = _add_doc_and_chunks(store, 3)
    parent, child1, child2 = chunk_ids
    store.add_chunk_hierarchy(child1, parent_chunk_id=parent, level=1, position=0)
    assert store.get_parent_chunk(child1) == parent


def test_get_parent_none_for_top_level(store: MetadataStore) -> None:
    _, chunk_ids = _add_doc_and_chunks(store, 2)
    root = chunk_ids[0]
    store.add_chunk_hierarchy(root, parent_chunk_id=None, level=0, position=0)
    assert store.get_parent_chunk(root) is None


def test_get_parent_none_for_missing_record(store: MetadataStore) -> None:
    assert store.get_parent_chunk(9999) is None


def test_heading_stored_and_retrievable(store: MetadataStore) -> None:
    _, chunk_ids = _add_doc_and_chunks(store, 2)
    store.add_chunk_hierarchy(chunk_ids[0], None, level=0, position=0, heading="Introduction")
    row = store._conn.execute(
        "SELECT heading FROM chunk_hierarchy WHERE chunk_id = ?", (chunk_ids[0],)
    ).fetchone()
    assert row["heading"] == "Introduction"


# ---------------------------------------------------------------------------
# get_sibling_chunks
# ---------------------------------------------------------------------------


def test_get_siblings_basic(store: MetadataStore) -> None:
    _, ids = _add_doc_and_chunks(store, 4)
    parent, c1, c2, c3 = ids
    store.add_chunk_hierarchy(c1, parent_chunk_id=parent, level=1, position=0)
    store.add_chunk_hierarchy(c2, parent_chunk_id=parent, level=1, position=1)
    store.add_chunk_hierarchy(c3, parent_chunk_id=parent, level=1, position=2)

    siblings_of_c1 = store.get_sibling_chunks(c1)
    assert set(siblings_of_c1) == {c2, c3}
    assert c1 not in siblings_of_c1


def test_get_siblings_single_child(store: MetadataStore) -> None:
    _, ids = _add_doc_and_chunks(store, 2)
    parent, child = ids
    store.add_chunk_hierarchy(child, parent_chunk_id=parent, level=1, position=0)
    assert store.get_sibling_chunks(child) == []


def test_get_siblings_no_hierarchy_record(store: MetadataStore) -> None:
    assert store.get_sibling_chunks(9999) == []


def test_get_siblings_top_level_returns_empty(store: MetadataStore) -> None:
    _, ids = _add_doc_and_chunks(store, 1)
    store.add_chunk_hierarchy(ids[0], parent_chunk_id=None, level=0, position=0)
    assert store.get_sibling_chunks(ids[0]) == []


# ---------------------------------------------------------------------------
# get_child_chunks
# ---------------------------------------------------------------------------


def test_get_children_basic(store: MetadataStore) -> None:
    _, ids = _add_doc_and_chunks(store, 4)
    parent, c1, c2, c3 = ids
    store.add_chunk_hierarchy(c1, parent_chunk_id=parent, level=1, position=0)
    store.add_chunk_hierarchy(c2, parent_chunk_id=parent, level=1, position=1)
    store.add_chunk_hierarchy(c3, parent_chunk_id=parent, level=1, position=2)

    children = store.get_child_chunks(parent)
    # Should be ordered by position.
    assert children == [c1, c2, c3]


def test_get_children_empty(store: MetadataStore) -> None:
    _, ids = _add_doc_and_chunks(store, 1)
    assert store.get_child_chunks(ids[0]) == []


def test_get_children_ordering_by_position(store: MetadataStore) -> None:
    _, ids = _add_doc_and_chunks(store, 4)
    parent, c1, c2, c3 = ids
    # Insert out of order.
    store.add_chunk_hierarchy(c3, parent_chunk_id=parent, level=1, position=2)
    store.add_chunk_hierarchy(c1, parent_chunk_id=parent, level=1, position=0)
    store.add_chunk_hierarchy(c2, parent_chunk_id=parent, level=1, position=1)

    assert store.get_child_chunks(parent) == [c1, c2, c3]


# ---------------------------------------------------------------------------
# Replace on re-insert
# ---------------------------------------------------------------------------


def test_replace_hierarchy_record(store: MetadataStore) -> None:
    _, ids = _add_doc_and_chunks(store, 3)
    parent, child, other = ids
    store.add_chunk_hierarchy(child, parent_chunk_id=parent, level=1, position=0)
    # Re-insert with different parent — should replace cleanly.
    store.add_chunk_hierarchy(child, parent_chunk_id=other, level=2, position=1)
    assert store.get_parent_chunk(child) == other
