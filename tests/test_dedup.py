"""Tests for semantic near-duplicate detection (verra.ingest.dedup)."""

from __future__ import annotations

from verra.ingest.chunking import Chunk
from verra.ingest.dedup import (
    cluster_related_chunks,
    compute_similarity,
    find_near_duplicates,
)


# ---------------------------------------------------------------------------
# compute_similarity
# ---------------------------------------------------------------------------


def test_exact_match_similarity_is_1():
    text = "The quick brown fox jumps over the lazy dog"
    assert compute_similarity(text, text) == 1.0


def test_different_text_low_similarity():
    a = "revenue growth exceeded expectations in Q3 fiscal year"
    b = "the deployment pipeline failed due to a missing environment variable"
    score = compute_similarity(a, b)
    assert score < 0.3


def test_similar_texts_above_threshold():
    a = "The SLA requires 99.9% uptime for all production services."
    b = "Our SLA mandates 99.9% uptime across all production services."
    score = compute_similarity(a, b)
    assert score > 0.4


def test_empty_strings():
    assert compute_similarity("", "") == 1.0


def test_one_empty_string():
    assert compute_similarity("hello world", "") == 0.0


# ---------------------------------------------------------------------------
# find_near_duplicates
# ---------------------------------------------------------------------------


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, metadata={})


def test_near_duplicate_detection():
    new_chunks = [
        _chunk("The contract renewal is scheduled for Q4 2024 at the same rate."),
    ]
    existing = [
        (101, "Contract renewal scheduled for Q4 2024 at the existing rate."),
        (102, "Completely unrelated text about deployment pipelines and Docker."),
    ]
    pairs = find_near_duplicates(new_chunks, existing, threshold=0.4)
    chunk_indices = [p[0] for p in pairs]
    existing_ids = [p[1] for p in pairs]
    assert 0 in chunk_indices
    assert 101 in existing_ids
    assert 102 not in existing_ids


def test_no_duplicates_in_unique_chunks():
    new_chunks = [
        _chunk("The deployment pipeline uses GitHub Actions and runs on ubuntu-latest."),
    ]
    existing = [
        (10, "Quarterly financial review shows 15% YoY revenue growth."),
        (11, "The onboarding checklist requires completion within the first week."),
    ]
    pairs = find_near_duplicates(new_chunks, existing, threshold=0.7)
    assert len(pairs) == 0


def test_threshold_filtering():
    a = "The project deadline is end of month and all deliverables must be ready."
    b = "The project deadline is end of month and deliverables should be complete."
    new_chunks = [_chunk(a)]
    existing = [(55, b)]

    # Low threshold → match
    pairs_low = find_near_duplicates(new_chunks, existing, threshold=0.3)
    assert len(pairs_low) > 0

    # Very high threshold → no match
    pairs_high = find_near_duplicates(new_chunks, existing, threshold=0.99)
    assert len(pairs_high) == 0


# ---------------------------------------------------------------------------
# cluster_related_chunks
# ---------------------------------------------------------------------------


def test_cluster_identical_chunks():
    ids = [1, 2, 3]
    text = "The SLA requires 99.9% uptime for production services."
    texts = [text, text, text]
    clusters = cluster_related_chunks(ids, texts, threshold=0.9)
    # All three should be in one cluster
    flat = [cid for cluster in clusters for cid in cluster]
    assert set(flat) == {1, 2, 3}
    assert len(clusters) == 1


def test_cluster_distinct_chunks():
    ids = [1, 2]
    texts = [
        "The contract renewal is scheduled for Q4 with the same pricing.",
        "Deployment failed due to a missing Kubernetes secret in production.",
    ]
    clusters = cluster_related_chunks(ids, texts, threshold=0.6)
    # Expect two separate singleton clusters
    assert len(clusters) == 2
    for cluster in clusters:
        assert len(cluster) == 1


def test_cluster_empty_input():
    assert cluster_related_chunks([], [], threshold=0.6) == []
