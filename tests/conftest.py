"""Shared pytest fixtures for Atlas tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from verra.store.metadata import MetadataStore
from verra.store.memory import MemoryStore


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """A temporary directory that is cleaned up after each test."""
    return tmp_path


@pytest.fixture
def metadata_store(tmp_path: Path) -> MetadataStore:
    """An in-process MetadataStore backed by a temporary SQLite database."""
    store = MetadataStore(tmp_path / "test_metadata.db")
    yield store
    store.close()


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    """An in-process MemoryStore backed by a temporary SQLite database."""
    store = MemoryStore(tmp_path / "test_memory.db")
    yield store
    store.close()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
