"""Hybrid retrieval: SQL metadata filter + ChromaDB vector search.

Three paths:
  metadata → SQLite only (fast, exact)
  semantic  → ChromaDB only (approximate, semantic)
  hybrid    → SQLite filter to get candidate chunk IDs, then re-rank by vector
  entity    → Entity registry lookup → fetch linked chunks → re-rank by vector

Authority ranking:
  When multiple results address the same topic, prefer higher authority_weight.
  Among equal authority, prefer newer valid_from dates.
  Similarity score breaks remaining ties.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verra.retrieval.router import ClassifiedQuery, QueryType
from verra.store.metadata import MetadataStore
from verra.store.vector import VectorStore


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # lower distance = higher relevance for ChromaDB cosine
    authority_weight: int = 50  # 0–100; higher = more authoritative
    valid_from: str | None = None  # ISO date string; None means no temporal bound


def rank_by_authority(results: list[SearchResult]) -> list[SearchResult]:
    """Re-rank results by blending normalized similarity score with authority and recency.

    Scores from ChromaDB can range from about -1.0 to 1.0 (computed as
    ``1.0 - L2_distance``). A naive blend of raw scores with authority weights
    breaks when scores are negative, because the always-positive authority term
    can dominate. Instead we normalise scores to [0, 1] first.

    Composite formula (after normalisation):
        composite = 0.75 * norm_score
                  + 0.20 * (authority_weight / 100)
                  + 0.05 * recency_boost

    Where:
      norm_score    = (score - global_min) / (global_max - global_min + ε)
      recency_boost = 1.0 if valid_from is set, else 0.0

    This keeps a highly-relevant low-authority document above a less-relevant
    high-authority one while letting authority and recency tip close calls.

    The sort is stable so equal composites retain their original relative order.
    """
    if not results:
        return results

    # Normalise raw scores to [0, 1] within this result set
    scores = [r.score for r in results]
    s_min = min(scores)
    s_max = max(scores)
    score_range = s_max - s_min

    def _norm_score(s: float) -> float:
        if score_range < 1e-9:
            return 1.0  # all identical — treat all as perfect
        return (s - s_min) / score_range

    def _composite(r: SearchResult) -> float:
        recency = 1.0 if r.valid_from else 0.0
        return 0.75 * _norm_score(r.score) + 0.20 * (r.authority_weight / 100.0) + 0.05 * recency

    return sorted(results, key=_composite, reverse=True)


def search(
    query: ClassifiedQuery,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    n_results: int = 5,
    entity_store: Any | None = None,  # EntityStore, optional to avoid circular import
) -> list[SearchResult]:
    """Run hybrid retrieval and return ranked results.

    If entity_store is provided and the query text mentions known entities,
    entity-based retrieval is tried first. Falls back to standard strategies
    if no entity matches are found.
    """
    # Try entity-based retrieval when an entity store is available
    if entity_store is not None:
        entity_results = _entity_search(query, entity_store, metadata_store, vector_store, n_results)
        if entity_results:
            return entity_results

    if query.query_type == QueryType.METADATA:
        return _metadata_search(query, metadata_store, n_results)
    elif query.query_type == QueryType.SEMANTIC:
        results = _semantic_search(query, vector_store, n_results)
        # Always supplement with keyword search — embeddings miss specific
        # names, acronyms, and terms that keyword matching catches reliably.
        keyword_hits = _keyword_fallback(query, vector_store, n_results)
        if keyword_hits:
            existing_ids = {r.chunk_id for r in results}
            for kh in keyword_hits:
                if kh.chunk_id not in existing_ids:
                    results.append(kh)
                    existing_ids.add(kh.chunk_id)
            results = rank_by_authority(results)[:n_results]
        return results
    else:
        return _hybrid_search(query, metadata_store, vector_store, n_results)


def _keyword_fallback(
    query: ClassifiedQuery,
    vector_store: VectorStore,
    n_results: int,
) -> list[SearchResult]:
    """Search all chunks by keyword overlap when embedding search fails.

    This catches cases where important content is buried in large chunks
    and the embedding doesn't capture the specific keywords well.
    """
    import re as _re

    _stop = {"what", "is", "the", "our", "how", "many", "much", "who", "does",
             "do", "are", "were", "was", "can", "will", "a", "an", "and", "or",
             "for", "in", "on", "at", "to", "of", "by", "from", "with", "about",
             "this", "that", "its", "has", "have", "had", "been", "not"}
    keywords = [
        w.lower() for w in _re.findall(r"\w+", query.semantic_text)
        if len(w) >= 3 and w.lower() not in _stop
    ]
    if not keywords:
        return []

    try:
        all_data = vector_store._collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    scored: list[tuple[float, SearchResult]] = []
    for chunk_id, text, meta in zip(
        all_data.get("ids", []),
        all_data.get("documents", []),
        all_data.get("metadatas", []),
    ):
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches == 0:
            continue
        ratio = matches / len(keywords)
        # Score based on keyword match ratio — high enough to compete
        # with embedding results, since keyword hits are often more precise
        score = ratio * 0.8  # max +0.8 for full match

        # Penalize CSV chunks same as in semantic search
        if (meta or {}).get("format") == "csv":
            score -= 0.4

        if score <= 0:
            continue

        scored.append((score, SearchResult(
            chunk_id=chunk_id,
            text=text,
            metadata=meta or {},
            score=score,
            authority_weight=int((meta or {}).get("authority_weight", 50)),
        )))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Document diversity: max 2 per file
    diverse: list[SearchResult] = []
    doc_counts: dict[str, int] = {}
    for _, sr in scored:
        fname = sr.metadata.get("file_name", "")
        count = doc_counts.get(fname, 0)
        if count < 2:
            diverse.append(sr)
            doc_counts[fname] = count + 1
        if len(diverse) >= n_results:
            break

    return diverse


# Internal strategies


def _metadata_search(
    query: ClassifiedQuery,
    metadata_store: MetadataStore,
    n_results: int,
) -> list[SearchResult]:
    """Query the SQLite emails table and return matching rows as SearchResult."""
    email_rows = metadata_store.search_emails(
        from_addr=query.from_address,
        since=query.since_date,
        until=query.until_date,
        limit=n_results,
    )
    results: list[SearchResult] = []
    for row in email_rows:
        text = f"Subject: {row.get('subject', '')}\nFrom: {row.get('from_addr', '')}\nDate: {row.get('date', '')}"
        results.append(
            SearchResult(
                chunk_id=str(row.get("chunk_id") or row["id"]),
                text=text,
                metadata=row,
                score=1.0,  # exact metadata match
                authority_weight=row.get("authority_weight", 50),
                valid_from=row.get("date"),
            )
        )
    # Metadata results are already date-filtered; still apply authority ranking
    return rank_by_authority(results)


def _semantic_search(
    query: ClassifiedQuery,
    vector_store: VectorStore,
    n_results: int,
) -> list[SearchResult]:
    """Pure vector similarity search."""
    where: dict[str, Any] | None = None
    if query.source_type:
        where = {"source_type": query.source_type}

    # Fetch many more candidates than needed — when the corpus is dominated
    # by CSV chunks we need to dig deep to find the relevant prose docs.
    fetch_n = max(n_results * 5, 60)
    hits = vector_store.search(query.semantic_text, n_results=fetch_n, where=where)

    # Fall back to unfiltered search if source_type filter returned nothing
    if not hits and where is not None:
        hits = vector_store.search(query.semantic_text, n_results=fetch_n, where=None)

    # Extract significant query keywords for keyword-boost scoring
    _stop = {"what", "is", "the", "our", "how", "many", "much", "who", "does",
             "do", "are", "was", "were", "can", "will", "a", "an", "and", "or",
             "for", "in", "on", "at", "to", "of", "by", "from", "with", "about"}
    query_keywords = [
        w.lower() for w in query.semantic_text.split()
        if len(w) >= 3 and w.lower() not in _stop
    ]

    results = []
    seen_texts: set[str] = set()
    for h in hits:
        text = h["document"]
        text_key = text[:200]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        score = 1.0 - (h["distance"] or 0.0)

        # Keyword overlap boost — if query terms appear literally in the chunk,
        # boost the score. This catches cases where the embedding misses a
        # specific term buried in a large chunk.
        if query_keywords:
            text_lower = text.lower()
            matches = sum(1 for kw in query_keywords if kw in text_lower)
            keyword_ratio = matches / len(query_keywords)
            score += keyword_ratio * 0.3  # up to +0.3 for full keyword overlap

        # Boost short, specific documents — they tend to be highly targeted
        doc_len = len(text)
        if doc_len < 500:
            score += 0.15
        elif doc_len < 1500:
            score += 0.05

        # Penalize CSV chunks — they're repetitive tabular data that
        # overwhelm prose documents when the corpus has large CSVs.
        doc_format = h["metadata"].get("format", "")
        if doc_format == "csv":
            score -= 0.4

        results.append(SearchResult(
            chunk_id=h["id"],
            text=text,
            metadata=h["metadata"],
            score=score,
            authority_weight=int(h["metadata"].get("authority_weight", 50)),
            valid_from=h["metadata"].get("valid_from") or h["metadata"].get("date"),
        ))

    # Document diversity: limit to max 2 chunks per source file
    # to prevent large files from dominating results.
    if len(results) > n_results:
        diverse: list[SearchResult] = []
        doc_counts: dict[str, int] = {}
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            fname = r.metadata.get("file_name", "")
            count = doc_counts.get(fname, 0)
            if count < 2:
                diverse.append(r)
                doc_counts[fname] = count + 1
        results = diverse

    ranked = rank_by_authority(results)
    return ranked[:n_results]


def _hybrid_search(
    query: ClassifiedQuery,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    n_results: int,
) -> list[SearchResult]:
    """SQL filter on email metadata, then vector re-rank on the result set."""
    # First get a broader set of candidate chunks from SQL
    email_rows = metadata_store.search_emails(
        from_addr=query.from_address,
        since=query.since_date,
        until=query.until_date,
        limit=n_results * 3,  # wider net for re-ranking
    )
    candidate_ids = [str(row["chunk_id"]) for row in email_rows if row.get("chunk_id")]

    if not candidate_ids:
        # Fall back to pure semantic if SQL finds nothing
        return _semantic_search(query, vector_store, n_results)

    # Vector search scoped to the candidate IDs
    where: dict[str, Any] | None = None
    if query.source_type:
        where = {"source_type": query.source_type}

    hits = vector_store.search(query.semantic_text, n_results=n_results, where=where)

    # Prefer hits that overlap with SQL candidates
    in_candidates = [h for h in hits if h["id"] in candidate_ids]
    not_in = [h for h in hits if h["id"] not in candidate_ids]
    reranked_hits = in_candidates + not_in

    results = [
        SearchResult(
            chunk_id=h["id"],
            text=h["document"],
            metadata=h["metadata"],
            score=1.0 - (h["distance"] or 0.0),
            authority_weight=int(h["metadata"].get("authority_weight", 50)),
            valid_from=h["metadata"].get("valid_from") or h["metadata"].get("date"),
        )
        for h in reranked_hits[:n_results]
    ]
    return rank_by_authority(results)


def _entity_search(
    query: ClassifiedQuery,
    entity_store: Any,  # EntityStore
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    n_results: int,
) -> list[SearchResult]:
    """Pull chunks by entity ID, then re-rank by vector similarity.

    Steps:
    1. Tokenize the query into candidate entity names (single words and
       two-word phrases).
    2. Look each candidate up in the entity registry.
    3. Collect all chunk IDs linked to the matched entities.
    4. Fetch those chunks from the vector store by re-querying with the
       same query text (ChromaDB cannot efficiently fetch by ID list, so
       we do a broad search and filter).
    5. Re-rank by similarity score, then apply authority ranking.
    6. Return top n_results.
    """
    raw = query.raw

    # Build candidate name tokens: single words + two-word phrases
    words = raw.split()
    candidates: list[str] = list(words)
    for i in range(len(words) - 1):
        candidates.append(f"{words[i]} {words[i + 1]}")

    # Resolve against entity registry
    matched_entity_ids: list[int] = []
    matched_names: list[str] = []
    seen_ids: set[int] = set()
    for candidate in candidates:
        entity = entity_store.resolve(candidate)
        if entity and entity["id"] not in seen_ids:
            seen_ids.add(entity["id"])
            matched_entity_ids.append(entity["id"])
            matched_names.append(entity["canonical_name"])

    if not matched_entity_ids:
        return []

    # Gather chunk IDs linked to these entities
    linked_chunk_ids: set[str] = set()
    for eid in matched_entity_ids:
        chunk_ids = entity_store.get_chunks_for_entity(eid)
        linked_chunk_ids.update(str(cid) for cid in chunk_ids)

    if not linked_chunk_ids:
        return []

    # Broad vector search then filter to linked chunks
    broad_n = min(n_results * 4, 50)
    hits = vector_store.search(query.semantic_text, n_results=broad_n)

    # Filter to entity-linked chunks, fall back to all hits if none overlap
    filtered = [h for h in hits if h["id"] in linked_chunk_ids]
    if not filtered:
        # Entity chunks exist but none surfaced in the top broad_n — fall back
        return []

    results = [
        SearchResult(
            chunk_id=h["id"],
            text=h["document"],
            metadata=h["metadata"],
            score=1.0 - (h["distance"] or 0.0),
            authority_weight=int(h["metadata"].get("authority_weight", 50)),
            valid_from=h["metadata"].get("valid_from") or h["metadata"].get("date"),
        )
        for h in filtered[:n_results]
    ]
    return rank_by_authority(results)
