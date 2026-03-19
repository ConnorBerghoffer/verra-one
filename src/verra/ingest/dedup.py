"""Semantic near-duplicate detection across sources.

Content hash catches exact copies. This catches the same information
described differently across email, meeting notes, tickets, etc.
"""


from __future__ import annotations

import re

from verra.ingest.chunking import Chunk


def _tokenise(text: str) -> set[str]:
    """Lower-case word tokens, stripping punctuation."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute text similarity using token overlap (Jaccard).

    Fast, no embedding needed.  Returns a value in [0.0, 1.0].
    """
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def find_near_duplicates(
    new_chunks: list[Chunk],
    existing_chunk_texts: list[tuple[int, str]],  # (chunk_id, text)
    threshold: float = 0.7,
) -> list[tuple[int, int, float]]:
    """Find chunks that are semantically near-duplicates.

    Compares every new chunk against every existing chunk using Jaccard
    similarity.  Pairs that meet or exceed ``threshold`` are returned.

    Parameters
    ----------
    new_chunks:
        Freshly produced chunks not yet in the store.
    existing_chunk_texts:
        (chunk_id, text) pairs already present in the store.
    threshold:
        Minimum Jaccard similarity to consider a pair a near-duplicate.

    Returns
    -------
    List of (new_chunk_index, existing_chunk_id, similarity_score).
    """
    results: list[tuple[int, int, float]] = []
    for new_idx, new_chunk in enumerate(new_chunks):
        for existing_id, existing_text in existing_chunk_texts:
            score = compute_similarity(new_chunk.text, existing_text)
            if score >= threshold:
                results.append((new_idx, existing_id, score))
    return results


def cluster_related_chunks(
    chunk_ids: list[int],
    chunk_texts: list[str],
    threshold: float = 0.6,
) -> list[list[int]]:
    """Group chunk IDs into clusters of near-duplicates.

    Uses a greedy union-find approach: for each pair whose Jaccard similarity
    meets ``threshold``, merge them into the same cluster.

    Returns list of clusters, each cluster is a list of chunk_ids sorted
    ascending.  Chunks with no near-duplicate partner appear in singleton
    clusters.
    """
    n = len(chunk_ids)
    if n == 0:
        return []

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            score = compute_similarity(chunk_texts[i], chunk_texts[j])
            if score >= threshold:
                union(i, j)

    # Build clusters
    cluster_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        cluster_map.setdefault(root, []).append(chunk_ids[i])

    return [sorted(members) for members in cluster_map.values()]
