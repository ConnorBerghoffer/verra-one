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

# ---------------------------------------------------------------------------
# Cross-encoder reranker (lazy-loaded on first use)
# ---------------------------------------------------------------------------

_reranker: Any = None  # CrossEncoder instance, False if unavailable, None if not yet loaded


def _get_reranker() -> Any:
    """Lazy-load the cross-encoder reranker.

    Returns a CrossEncoder instance, or None if sentence-transformers is not
    installed.  The result is cached so the model is only loaded once per
    process.
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _reranker = False  # sentinel: unavailable
    return _reranker if _reranker is not False else None


def rerank(query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
    """Rerank results using a cross-encoder for better precision.

    The cross-encoder scores each (query, passage) pair independently, giving
    much more accurate relevance judgements than bi-encoder similarity alone.
    Falls back to input order if the model is unavailable.

    Raw cross-encoder logits (ms-marco-MiniLM) use a different scale than
    ChromaDB similarity scores — they can be very negative for irrelevant
    passages and highly positive for strong matches.  We normalise them to
    [0, 1] within the result set so downstream thresholds (e.g. _MIN_SCORE
    in chat.py) continue to work correctly.
    """
    ranker = _get_reranker()
    if ranker is None or not results:
        return results[:top_n]

    pairs = [(query, r.text) for r in results]
    raw_scores = ranker.predict(pairs)

    # Normalise logits to [0, 1] within this candidate set
    s_min = float(min(raw_scores))
    s_max = float(max(raw_scores))
    s_range = s_max - s_min

    for r, raw in zip(results, raw_scores):
        if s_range > 1e-9:
            r.score = (float(raw) - s_min) / s_range
        else:
            r.score = 1.0  # all identical

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]


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


def _hyde_expand(query_text: str, llm_client: Any = None) -> str:
    """Generate a hypothetical document that would answer the query.

    This hypothetical text is then embedded for search, which produces
    embeddings much closer to actual document embeddings than the raw
    question would.
    """
    if llm_client is None:
        return query_text

    try:
        response = llm_client.complete([
            {"role": "system", "content": (
                "Write a short paragraph (3-5 sentences) that would be found in a "
                "business document answering the following question. Write it as if "
                "it's an excerpt from the actual document, not as an answer. "
                "Include specific details, names, numbers where plausible. "
                "Output ONLY the paragraph, nothing else."
            )},
            {"role": "user", "content": query_text},
        ])
        return response.strip()
    except Exception:
        return query_text


def search(
    query: ClassifiedQuery,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    n_results: int = 5,
    entity_store: Any | None = None,  # EntityStore, optional to avoid circular import
    llm_client: Any | None = None,    # LLMClient, optional; enables HyDE when provided
) -> list[SearchResult]:
    """Run hybrid retrieval and return ranked results.

    If entity_store is provided and the query text mentions known entities,
    entity-based retrieval is tried first. Falls back to standard strategies
    if no entity matches are found.

    If llm_client is provided, HyDE (Hypothetical Document Embedding) is used
    for the semantic search path: a hypothetical answer is generated and its
    embedding is used for vector search instead of the raw query embedding.
    The original query text is still used for keyword/BM25/filename search and
    for the LLM generation step.  HyDE adds ~2s latency (one extra LLM call).
    """
    # Try entity-based retrieval when an entity store is available
    if entity_store is not None:
        entity_results = _entity_search(query, entity_store, metadata_store, vector_store, n_results)
        if entity_results:
            return entity_results

    if query.query_type == QueryType.METADATA:
        return _metadata_search(query, metadata_store, n_results)
    elif query.query_type in (QueryType.SEMANTIC, QueryType.COMPARATIVE,
                               QueryType.TEMPORAL_TREND, QueryType.HYPOTHETICAL,
                               QueryType.META, QueryType.GAP, QueryType.MULTI_HOP,
                               QueryType.STATE_LOOKUP):
        # Generate HyDE text for better embedding search.
        # HyDE text is used ONLY for the vector search embedding; original query
        # text is used for keyword/BM25/filename search.
        hyde_text: str | None = None
        if llm_client is not None:
            hyde_text = _hyde_expand(query.semantic_text, llm_client)

        # Gather candidates from all retrieval strategies.
        # _semantic_search returns a larger candidate set without final reranking.
        embedding_results = _semantic_search(query, vector_store, n_results, hyde_text=hyde_text)

        # Supplement with keyword search — catches specific names, acronyms
        keyword_hits = _keyword_fallback(query, vector_store, n_results)

        # Guarantee: if a file's name contains significant query keywords,
        # include it. Catches "PTO" → pto_log_2025.csv etc.
        fname_hits = _filename_search(query, vector_store, n_results=3)

        # BM25 full-text search from SQLite FTS5
        bm25_hits = _bm25_search(query, metadata_store, vector_store, n_results)

        # Reserve filename hits — if a file is literally named after the
        # query topic (e.g. "PTO" → pto_log.csv), it MUST appear in results
        # regardless of what the reranker thinks.
        reserved: list[SearchResult] = []
        reserved_ids: set[str] = set()
        for r in fname_hits:
            reserved.append(r)
            reserved_ids.add(r.chunk_id)

        # Merge remaining candidates for reranking, excluding reserved.
        merged: list[SearchResult] = []
        seen_ids: set[str] = set(reserved_ids)
        for r in embedding_results + bm25_hits + keyword_hits:
            if r.chunk_id not in seen_ids:
                merged.append(r)
                seen_ids.add(r.chunk_id)

        # Cross-encoder reranking adjudicates between retrieval strategies.
        rerank_n = n_results * 4
        reranked = rerank(query.semantic_text, merged, rerank_n)

        # Combine: reserved filename hits first, then reranked results.
        # Trim to n_results total.
        combined = reserved + reranked
        results = rank_by_authority(combined)[:n_results]
        return results
    else:
        return _hybrid_search(query, metadata_store, vector_store, n_results)


def _filename_search(
    query: ClassifiedQuery,
    vector_store: VectorStore,
    n_results: int = 3,
) -> list[SearchResult]:
    """Find chunks from files whose names match query keywords.

    This catches the common case where a file like "pto_log_2025.csv" or
    "tickets_2025.csv" is the obvious answer but neither embedding nor
    keyword text matching surfaces it strongly enough.
    """
    import re as _re

    _stop = {"what", "is", "the", "our", "how", "many", "much", "who", "does",
             "do", "did", "are", "were", "was", "can", "will", "a", "an", "and", "or",
             "for", "in", "on", "at", "to", "of", "by", "from", "with", "about",
             "this", "that", "its", "has", "have", "had", "been", "not", "take",
             "find", "all", "any", "get", "show", "tell", "give", "list"}
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

    # Score each chunk by how many keywords appear in its file name
    scored: list[tuple[float, SearchResult]] = []
    for chunk_id, text, meta in zip(
        all_data.get("ids", []),
        all_data.get("documents", []),
        all_data.get("metadatas", []),
    ):
        fname = (meta or {}).get("file_name", "")
        if not fname:
            continue
        fname_lower = fname.lower()
        fname_matches = sum(1 for kw in keywords if kw in fname_lower)
        if fname_matches < 1:
            continue

        # Score heavily weighted to file name matches
        score = 0.7 + (0.15 * fname_matches)

        scored.append((score, SearchResult(
            chunk_id=chunk_id,
            text=text,
            metadata=meta or {},
            score=score,
            authority_weight=int((meta or {}).get("authority_weight", 50)),
        )))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Max 1 chunk per file
    seen_files: set[str] = set()
    results: list[SearchResult] = []
    for _, sr in scored:
        fname = sr.metadata.get("file_name", "")
        if fname in seen_files:
            continue
        seen_files.add(fname)
        results.append(sr)
        if len(results) >= n_results:
            break

    return results


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
             "do", "did", "are", "were", "was", "can", "will", "a", "an", "and", "or",
             "for", "in", "on", "at", "to", "of", "by", "from", "with", "about",
             "this", "that", "its", "has", "have", "had", "been", "not", "take",
             "find", "all", "any", "get", "show", "tell", "give", "list"}
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
        fname_lower = (meta or {}).get("file_name", "").lower()
        # Match against both chunk text AND file name
        matches = sum(1 for kw in keywords if kw in text_lower or kw in fname_lower)
        if matches == 0:
            continue
        ratio = matches / len(keywords)
        # Score based on keyword match ratio — high enough to compete
        # with embedding results, since keyword hits are often more precise
        score = ratio * 0.8  # max +0.8 for full match
        # Strong bonus when keywords appear in the file name — a file named
        # "pto_log" matching query "PTO" is a very high-confidence signal.
        fname_matches = sum(1 for kw in keywords if kw in fname_lower)
        if fname_matches > 0:
            score += 0.15 * fname_matches

        scored.append((score, SearchResult(
            chunk_id=chunk_id,
            text=text,
            metadata=meta or {},
            score=score,
            authority_weight=int((meta or {}).get("authority_weight", 50)),
        )))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Diversity: max 2 per file, max 3 CSV total
    diverse: list[SearchResult] = []
    doc_counts: dict[str, int] = {}
    csv_count = 0
    for _, sr in scored:
        fname = sr.metadata.get("file_name", "")
        is_csv = sr.metadata.get("format") == "csv"

        file_count = doc_counts.get(fname, 0)
        if file_count >= 2:
            continue
        if is_csv and csv_count >= 3:
            continue

        diverse.append(sr)
        doc_counts[fname] = file_count + 1
        if is_csv:
            csv_count += 1
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


def _bm25_search(
    query: ClassifiedQuery,
    metadata_store: MetadataStore,
    vector_store: VectorStore,
    n_results: int,
) -> list[SearchResult]:
    """BM25 full-text search via SQLite FTS5.

    Uses the MetadataStore FTS5 index to find chunks matching the query.
    Enriches each hit with chunk metadata from the vector store so the
    result can be merged with embedding hits.
    """
    fts_hits = metadata_store.search_fts(query.semantic_text, limit=n_results * 2)
    if not fts_hits:
        return []

    # Build a map of chunk_id -> metadata from ChromaDB for the hit IDs
    hit_ids = [str(h["chunk_id"]) for h in fts_hits]

    # Fetch just those chunks from the vector store
    try:
        chroma_data = vector_store._collection.get(
            ids=hit_ids,
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    id_to_doc: dict[str, tuple[str, dict]] = {}
    for cid, doc, meta in zip(
        chroma_data.get("ids", []),
        chroma_data.get("documents", []),
        chroma_data.get("metadatas", []),
    ):
        id_to_doc[cid] = (doc, meta or {})

    # FTS5 rank is negative (lower = better); normalise to a 0-1 score
    ranks = [h["rank"] for h in fts_hits]
    r_min = min(ranks) if ranks else -1.0
    r_max = max(ranks) if ranks else 0.0
    r_range = r_max - r_min

    results: list[SearchResult] = []
    for h in fts_hits:
        cid = str(h["chunk_id"])
        if cid not in id_to_doc:
            continue
        doc_text, meta = id_to_doc[cid]
        # Normalise BM25 rank to [0, 1]; ranks are negative so we invert.
        norm = (h["rank"] - r_min) / r_range if r_range > 1e-9 else 1.0
        score = 1.0 - norm  # higher score = better rank

        results.append(SearchResult(
            chunk_id=cid,
            text=doc_text,
            metadata=meta,
            score=score,
            authority_weight=int(meta.get("authority_weight", 50)),
            valid_from=meta.get("valid_from") or meta.get("date"),
        ))

    return results


def _semantic_search(
    query: ClassifiedQuery,
    vector_store: VectorStore,
    n_results: int,
    hyde_text: str | None = None,
) -> list[SearchResult]:
    """Pure vector similarity search returning a larger candidate pool.

    Returns up to ``n_results * 2`` diversity-filtered results so the
    caller (search()) can merge with other retrieval strategies before
    cross-encoder reranking.  Final trimming to n_results happens there.

    If hyde_text is provided it is used as the embedding query instead of the
    raw query text.  Keyword scoring still uses query.semantic_text so HyDE
    does not corrupt non-embedding signals.
    """
    where: dict[str, Any] | None = None
    if query.source_type:
        where = {"source_type": query.source_type}

    # HyDE: use the hypothetical document text for embedding search when available.
    # The original query text is kept for keyword-overlap scoring below.
    search_text = hyde_text if hyde_text is not None else query.semantic_text

    # Fetch many more candidates than needed — when the corpus is dominated
    # by CSV chunks we need to dig deep to find the relevant prose docs.
    fetch_n = max(n_results * 5, 60)
    hits = vector_store.search(search_text, n_results=fetch_n, where=where)

    # Fall back to unfiltered search if source_type filter returned nothing
    if not hits and where is not None:
        hits = vector_store.search(search_text, n_results=fetch_n, where=None)

    # Extract significant query keywords for keyword-boost scoring.
    # Always derived from the original query, not the HyDE expansion.
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

        results.append(SearchResult(
            chunk_id=h["id"],
            text=text,
            metadata=h["metadata"],
            score=score,
            authority_weight=int(h["metadata"].get("authority_weight", 50)),
            valid_from=h["metadata"].get("valid_from") or h["metadata"].get("date"),
        ))

    # Diversity filtering: sort by score, then enforce two caps:
    #  1. Max 2 chunks per source file (prevents one big file dominating)
    #  2. Max 3 CSV chunks total (CSV data can appear but can't drown prose)
    # This lets CSV data surface when genuinely relevant without overwhelming.
    results.sort(key=lambda x: x.score, reverse=True)
    diverse: list[SearchResult] = []
    doc_counts: dict[str, int] = {}
    csv_count = 0
    max_csv = 3
    for r in results:
        fname = r.metadata.get("file_name", "")
        is_csv = r.metadata.get("format") == "csv"

        # Per-file cap
        file_count = doc_counts.get(fname, 0)
        if file_count >= 2:
            continue

        # Format cap for CSV
        if is_csv and csv_count >= max_csv:
            continue

        diverse.append(r)
        doc_counts[fname] = file_count + 1
        if is_csv:
            csv_count += 1

    results = diverse

    # Return a larger candidate pool (up to 2x n_results) for the caller to
    # merge with BM25/keyword/filename results before cross-encoder reranking.
    # We still apply authority ranking here so the embedding ordering is
    # meaningful when the reranker is unavailable.
    ranked = rank_by_authority(results)
    return ranked[: n_results * 2]


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
