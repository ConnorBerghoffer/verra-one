"""Query classifier / router.

Classifies a user query into one of three retrieval strategies:
  - metadata: structured filter (SQL only) — "emails from Jake last month"
  - semantic: vector similarity only — "what's our refund policy"
  - hybrid: SQL filter + vector search — "what did Jake say about pricing"

The classifier uses lightweight keyword heuristics so it works with no LLM call.
"""


from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    METADATA = "metadata"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    COMPARATIVE = "comparative"
    TEMPORAL_TREND = "temporal_trend"
    HYPOTHETICAL = "hypothetical"
    META = "meta"
    GAP = "gap"
    MULTI_HOP = "multi_hop"
    STATE_LOOKUP = "state_lookup"


# Patterns that suggest structured / metadata intent
_METADATA_PATTERNS = [
    r"\bfrom\s+\w+",                  # "from Jake"
    r"\bemails?\s+(from|to|by)\b",    # "emails from", "email to"
    r"\blast\s+(week|month|year)\b",  # "last month"
    r"\bsince\s+\d{4}",               # "since 2023"
    r"\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\bsubject\s*(:|contains|is)\b",
    r"\blabel\s*:",
]

# Patterns that suggest a combination of person/entity + topic
_HYBRID_SIGNALS = [
    r"\bwhat\s+did\s+\w+\s+say\b",   # "what did Jake say about..."
    r"\bdiscuss(ed|ion)?\b.*\bwith\b",
    r"\bconversation\b.*\babout\b",
    r"\b(agreement|proposal|quote)\b.*\bfrom\b",
]


_COMPARATIVE_PATTERNS = [
    r"\bcompar[ei]",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bdifference\s+between\b",
    r"\bhow\s+does\s+\w+\s+(?:compare|stack\s+up|differ)",
]

_TEMPORAL_PATTERNS = [
    r"\b(improv|declin|increas|decreas|grow|shrink|chang|trend|evolv)(ing|ed|e|s)?\b",
    r"\bover\s+(?:the\s+)?(?:past|last|previous)\s+\d+",
    r"\bover\s+time\b",
    r"\btrend\b",
    r"\bhistor(?:y|ical)\b",
]

_HYPOTHETICAL_PATTERNS = [
    r"\bif\s+we\b",
    r"\bwhat\s+(?:would|could|might)\s+happen\b",
    r"\bwhat\s+if\b",
    r"\bhypothetical\b",
    r"\bimpact\s+of\s+(?:losing|adding|removing|changing)\b",
]

_META_PATTERNS = [
    r"\b(?:most|least)\s+(?:outdated|stale|recent)\s+(?:info|information|data|document)",
    r"\bwhat\s+(?:do|does)\s+(?:the|our|my)\s+(?:system|knowledge\s*base|data)\s+(?:know|have|contain)",
    r"\bhow\s+many\s+(?:documents|files|emails|chunks)\b",
    r"\bwhat\s+sources?\b.*\bingested\b",
    r"\bsystem\s+status\b",
]

_GAP_PATTERNS = [
    r"\bwhat\s+(?:don't|do\s*n't|doesn't)\s+we\s+(?:have|know)\b",
    r"\bwhat(?:'s|\s+is|\s+\w+\s+is)\s+missing\b",
    r"\bmissing\s+(?:from|in)\s+(?:our|the|my)\b",
    r"\bwhat\s+(?:gaps?|holes?)\b",
    r"\bundocumented\b",
    r"\bno\s+record\s+of\b",
    r"\bwhat\s+(?:are\s+we|am\s+I)\s+missing\b",
]

_MULTI_HOP_PATTERNS = [
    r"\bwho\s+manages?\s+the\s+team\s+that\b",
    r"\bwhat\s+.*\band\s+then\b.*\?",
    r"\bfind\s+.*\bthen\s+(?:check|look|search|find)\b",
    r"\bchain\b.*\bof\b",
]


def classify_query(query: str) -> QueryType:
    """Return the retrieval strategy for a query string.

    Expanded classification: metadata, semantic, hybrid, comparative,
    temporal_trend, hypothetical, meta, gap, multi_hop.

    Uses rule-based matching — no LLM call required.
    """
    q = query.lower()

    # State lookup: current status/state questions
    _STATE_LOOKUP_PATTERNS = [
        r"\b(?:is|are|does)\s+\w+\s+(?:still|currently|active|subscribed|employed|on the)\b",
        r"\bcurrent\s+(?:status|state|tier|plan|phase|contact|role)\b",
        r"\bwhat\s+(?:tier|plan|phase|status)\s+is\b",
        r"\bwho\s+is\s+the\s+(?:current|primary)\s+contact\b",
        r"\b(?:still|currently)\s+(?:active|working|employed|subscribed|on)\b",
    ]
    for pattern in _STATE_LOOKUP_PATTERNS:
        if re.search(pattern, q):
            return QueryType.STATE_LOOKUP

    # Check expanded types first (most specific → least specific)
    for pattern in _META_PATTERNS:
        if re.search(pattern, q):
            return QueryType.META

    for pattern in _GAP_PATTERNS:
        if re.search(pattern, q):
            return QueryType.GAP

    for pattern in _HYPOTHETICAL_PATTERNS:
        if re.search(pattern, q):
            return QueryType.HYPOTHETICAL

    for pattern in _COMPARATIVE_PATTERNS:
        if re.search(pattern, q):
            return QueryType.COMPARATIVE

    for pattern in _TEMPORAL_PATTERNS:
        if re.search(pattern, q):
            return QueryType.TEMPORAL_TREND

    for pattern in _MULTI_HOP_PATTERNS:
        if re.search(pattern, q):
            return QueryType.MULTI_HOP

    # Check hybrid signals (entity + semantic intent)
    for pattern in _HYBRID_SIGNALS:
        if re.search(pattern, q):
            return QueryType.HYBRID

    # Count metadata pattern matches
    meta_hits = sum(1 for p in _METADATA_PATTERNS if re.search(p, q))

    if meta_hits >= 2:
        # Strongly structured query
        return QueryType.METADATA
    elif meta_hits == 1:
        # Some structure but likely also needs semantic understanding
        return QueryType.HYBRID
    else:
        return QueryType.SEMANTIC


@dataclass
class ClassifiedQuery:
    raw: str
    query_type: QueryType

    # Extracted filters (for metadata / hybrid paths)
    from_address: str | None = None
    since_date: str | None = None
    until_date: str | None = None
    subject_contains: str | None = None
    source_type: str | None = None  # "email" | "folder"

    # Semantic search text (possibly different from the raw query)
    semantic_text: str = ""

    def __post_init__(self) -> None:
        if not self.semantic_text:
            self.semantic_text = self.raw


def parse_query(query: str) -> ClassifiedQuery:
    """Classify and extract structured filters from a query string."""
    query_type = classify_query(query)
    q = query.lower()

    # Simple extraction of "from <name>" — stop before time words
    _TIME_WORDS = r"(?:last|this|next|since|in|on|during|yesterday|today|ago)"
    from_match = re.search(rf"\bfrom\s+([a-z]+(?:\s+[a-z]+)?)\s+{_TIME_WORDS}\b", q)
    if not from_match:
        from_match = re.search(r"\bfrom\s+([a-z]+)", q)
    from_addr = from_match.group(1) if from_match else None

    # Detect explicit source type
    source_type = None
    if re.search(r"\b(email|gmail|inbox|message)\b", q):
        source_type = "email"
    elif re.search(r"\b(document|file|pdf|doc|folder)\b", q):
        source_type = "folder"

    return ClassifiedQuery(
        raw=query,
        query_type=query_type,
        from_address=from_addr,
        source_type=source_type,
        semantic_text=query,
    )
