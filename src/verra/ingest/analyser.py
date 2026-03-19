"""LLM-powered ingestion analyser.

Examines each chunk and extracts structured metadata:
- Assertions/facts
- Sentiment
- Commitments/action items
- Topics
- Staleness risk
- Contradiction detection against existing assertions

Three modes:
  fast    — skip analysis, just mark chunks as pending (for bulk ingestion)
  deep    — background batch job, processes pending chunks (for initial load)
  realtime — analyse inline during ingestion (for ongoing sync, low volume)
"""


from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from verra.store.analysis import AnalysisStore
from verra.store.entities import EntityStore


# ---------------------------------------------------------------------------
# Analysis result model
# ---------------------------------------------------------------------------

@dataclass
class ChunkAnalysis:
    """Structured output from analysing a single chunk."""
    sentiment: str = "neutral"          # positive, neutral, negative, escalation
    topics: list[str] | None = None
    staleness_risk: float = 0.0         # 0-1
    assertions: list[str] | None = None
    commitments: list[dict[str, str]] | None = None  # [{who, what, due_date}]
    contradictions: list[dict[str, str]] | None = None  # [{old, new}]


# ---------------------------------------------------------------------------
# Heuristic analyser (no LLM required — fast, deterministic)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "happy", "great", "excellent", "love", "wonderful", "perfect",
    "pleased", "thrilled", "fantastic", "appreciate", "thank",
    "congratulations", "congrats", "well done", "awesome",
}

_NEGATIVE_WORDS = {
    "issue", "problem", "complaint", "frustrated", "disappointed",
    "broken", "failure", "failed", "error", "bug", "wrong",
    "urgent", "critical", "escalat", "overdue", "late", "delay",
    "concern", "worried", "unfortunately", "sorry",
}

_ESCALATION_WORDS = {
    "escalat", "urgent", "immediately", "asap", "critical",
    "unacceptable", "demand", "legal", "lawyer", "attorney",
}

_COMMITMENT_PATTERNS = [
    # "I'll send X by Friday"
    re.compile(
        r"(?:I(?:'ll| will)|we(?:'ll| will))\s+(.{10,80}?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "Action item: X"
    re.compile(
        r"(?:action item|todo|to do|task):\s*(.{10,120}?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "[ ] Name: do something"
    re.compile(
        r"\[\s*\]\s*(\w+(?:\s+\w+)?:\s*.{10,120}?)(?:\n|$)",
    ),
    # "Need to X by Y"
    re.compile(
        r"(?:need to|must|should)\s+(.{10,80}?)(?:\.|$)",
        re.IGNORECASE,
    ),
]

_DATE_REFS = re.compile(
    r"\b(20\d{2}|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|Q[1-4]|version\s+\d|v\d+\.\d+)\b",
    re.IGNORECASE,
)

_STALENESS_KEYWORDS = re.compile(
    r"\b(deprecated|legacy|old|outdated|previous|former|was|used to|no longer)\b",
    re.IGNORECASE,
)

_ASSERTION_PATTERNS = [
    # "X is Y" — simple factual claim
    re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|was|were)\s+(.{5,80}?)(?:\.|$)"),
    # "X costs $Y"
    re.compile(r"(.{5,40}?)\s+(?:costs?|charges?|priced?\s+at)\s+(.{3,40}?)(?:\.|$)", re.IGNORECASE),
    # "X expires on Y"
    re.compile(r"(.{5,40}?)\s+(?:expires?|ends?|terminates?)\s+(?:on|at|by)?\s*(.{5,40}?)(?:\.|$)", re.IGNORECASE),
    # "$X/month" or "$X per user"
    re.compile(r"(\$[\d,]+(?:\.\d{2})?)\s*/\s*(month|year|user|seat|hour)", re.IGNORECASE),
]


def analyse_chunk_heuristic(
    chunk_text: str,
    chunk_metadata: dict[str, Any] | None = None,
) -> ChunkAnalysis:
    """Analyse a chunk using regex/heuristic methods (no LLM call).

    Fast and deterministic. Used as fallback when LLM is unavailable
    or for bulk ingestion where speed matters.
    """
    text_lower = chunk_text.lower()
    words = set(text_lower.split())

    # -- Sentiment --
    pos_count = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
    esc_count = sum(1 for w in _ESCALATION_WORDS if w in text_lower)

    if esc_count >= 2:
        sentiment = "escalation"
    elif neg_count > pos_count and neg_count >= 2:
        sentiment = "negative"
    elif pos_count > neg_count and pos_count >= 2:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    # -- Topics --
    topics = _extract_topics(text_lower)

    # -- Staleness risk --
    date_refs = len(_DATE_REFS.findall(chunk_text))
    stale_keywords = len(_STALENESS_KEYWORDS.findall(chunk_text))
    staleness = min(1.0, (stale_keywords * 0.3) + (0.1 if date_refs > 0 else 0))

    # -- Assertions --
    assertions = []
    for pattern in _ASSERTION_PATTERNS:
        for match in pattern.finditer(chunk_text):
            assertion = match.group(0).strip()
            if len(assertion) > 10 and len(assertion) < 200:
                assertions.append(assertion)

    # -- Commitments --
    commitments = []
    for pattern in _COMMITMENT_PATTERNS:
        for match in pattern.finditer(chunk_text):
            text = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if len(text) > 10:
                # Try to extract who from context
                who = _extract_commitment_owner(chunk_text, match.start())
                commitments.append({
                    "who": who or "unknown",
                    "what": text,
                    "due_date": _extract_due_date(text) or "",
                })

    return ChunkAnalysis(
        sentiment=sentiment,
        topics=topics,
        staleness_risk=staleness,
        assertions=assertions,
        commitments=commitments,
    )


def analyse_chunk_llm(
    chunk_text: str,
    chunk_metadata: dict[str, Any] | None = None,
    existing_assertions: list[str] | None = None,
    llm_client: Any = None,
) -> ChunkAnalysis:
    """Analyse a chunk using an LLM for higher accuracy.

    Falls back to heuristic analysis if LLM is unavailable.
    """
    if llm_client is None:
        return analyse_chunk_heuristic(chunk_text, chunk_metadata)

    prompt = _build_analysis_prompt(chunk_text, existing_assertions)

    try:
        response = llm_client.complete([
            {"role": "system", "content": "You are a document analyser. Extract structured information from text. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt},
        ])
        return _parse_llm_analysis(response)
    except Exception:
        # Fall back to heuristic on any LLM failure
        return analyse_chunk_heuristic(chunk_text, chunk_metadata)


def check_contradictions(
    new_assertions: list[str],
    existing_assertions: list[str],
) -> list[dict[str, str]]:
    """Compare new assertions against existing ones for contradictions.

    Uses simple heuristic: if two assertions mention the same entity/subject
    but have different values (numbers, dates, statuses), flag as conflict.
    """
    contradictions = []

    for new in new_assertions:
        new_lower = new.lower()
        for existing in existing_assertions:
            existing_lower = existing.lower()

            # Check if they're about the same subject (share 3+ words)
            new_words = set(new_lower.split()) - {"is", "are", "was", "the", "a", "an", "in", "on", "at", "to", "for"}
            existing_words = set(existing_lower.split()) - {"is", "are", "was", "the", "a", "an", "in", "on", "at", "to", "for"}
            overlap = new_words & existing_words

            if len(overlap) >= 2 and new_lower != existing_lower:
                # Check if they have different values (numbers, dates, statuses)
                new_nums = set(re.findall(r'\d+', new))
                exist_nums = set(re.findall(r'\d+', existing))

                if new_nums and exist_nums and new_nums != exist_nums:
                    contradictions.append({"old": existing, "new": new})
                elif _has_status_conflict(new_lower, existing_lower) or _has_status_conflict(existing_lower, new_lower):
                    contradictions.append({"old": existing, "new": new})

    return contradictions


def process_analysis_results(
    chunk_id: int,
    analysis: ChunkAnalysis,
    analysis_store: AnalysisStore,
    entity_store: EntityStore | None = None,
    entity_ids: list[int] | None = None,
    existing_assertions: list[str] | None = None,
) -> None:
    """Store analysis results in the database.

    Saves chunk analysis, commitments, conflicts, and coverage updates.
    """
    # Save chunk analysis
    analysis_store.save_chunk_analysis(
        chunk_id=chunk_id,
        sentiment=analysis.sentiment,
        staleness_risk=analysis.staleness_risk,
        topics=analysis.topics,
        assertions_count=len(analysis.assertions or []),
    )

    # Save commitments
    if analysis.commitments:
        for commit in analysis.commitments:
            who_entity_id = None
            if entity_store and commit.get("who"):
                resolved = entity_store.resolve(commit["who"])
                if resolved:
                    who_entity_id = resolved["id"]

            analysis_store.add_commitment(
                who_name=commit.get("who", "unknown"),
                what=commit.get("what", ""),
                who_entity_id=who_entity_id,
                due_date=commit.get("due_date"),
                source_chunk_id=chunk_id,
            )

    # Check for contradictions
    if analysis.assertions and existing_assertions:
        contradictions = check_contradictions(analysis.assertions, existing_assertions)
        for conflict in contradictions:
            primary_entity = entity_ids[0] if entity_ids else None
            analysis_store.add_conflict(
                assertion_a=conflict["old"],
                assertion_b=conflict["new"],
                entity_id=primary_entity,
                source_chunk_b=chunk_id,
            )

    # Update document coverage for linked entities
    if entity_ids and analysis.topics:
        doc_type = _infer_document_type(analysis.topics)
        if doc_type:
            for eid in entity_ids:
                analysis_store.update_document_coverage(
                    entity_id=eid,
                    document_type=doc_type,
                    source_chunk_id=chunk_id,
                )


# ---------------------------------------------------------------------------
# Background deep analysis job
# ---------------------------------------------------------------------------

def run_deep_analysis(
    analysis_store: AnalysisStore,
    entity_store: EntityStore,
    metadata_store: Any,
    vector_store: Any,
    llm_client: Any = None,
    batch_size: int = 50,
) -> dict[str, int]:
    """Process all pending chunks through the analysis pipeline.

    Used as a background job after bulk fast-ingest.
    Returns stats: {analysed, conflicts_found, commitments_found}.
    """
    stats = {"analysed": 0, "conflicts_found": 0, "commitments_found": 0}

    pending = analysis_store.get_pending_chunks(limit=batch_size)
    if not pending:
        return stats

    for chunk_id in pending:
        # Get chunk text from metadata store
        chunk_data = metadata_store.get_chunk_by_id(chunk_id)
        if not chunk_data:
            analysis_store.set_chunk_status(chunk_id, "skipped")
            continue

        chunk_text = chunk_data.get("text", "")
        if not chunk_text:
            # Try getting from vector store
            analysis_store.set_chunk_status(chunk_id, "skipped")
            continue

        # Get entity IDs linked to this chunk
        entity_ids = entity_store.get_entities_for_chunk(chunk_id)
        eid_list = [e["id"] for e in entity_ids]

        # Get existing assertions for these entities
        existing = []
        # (would pull from assertions store when available)

        # Run analysis
        if llm_client:
            analysis = analyse_chunk_llm(chunk_text, llm_client=llm_client, existing_assertions=existing)
        else:
            analysis = analyse_chunk_heuristic(chunk_text)

        # Store results
        process_analysis_results(
            chunk_id=chunk_id,
            analysis=analysis,
            analysis_store=analysis_store,
            entity_store=entity_store,
            entity_ids=eid_list,
            existing_assertions=existing,
        )

        stats["analysed"] += 1
        stats["conflicts_found"] += len(analysis.contradictions or [])
        stats["commitments_found"] += len(analysis.commitments or [])

    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS = {
    "pricing": ["price", "pricing", "cost", "fee", "rate", "charge", "billing", "invoice"],
    "contract": ["contract", "agreement", "sow", "msa", "terms", "renewal"],
    "security": ["security", "vulnerability", "ssl", "certificate", "auth", "encryption"],
    "performance": ["performance", "latency", "speed", "slow", "optimization", "cache"],
    "hiring": ["hiring", "recruit", "candidate", "interview", "job", "position", "role"],
    "infrastructure": ["infrastructure", "server", "aws", "cloud", "deploy", "docker", "kubernetes"],
    "support": ["support", "ticket", "issue", "bug", "fix", "patch", "incident"],
    "meeting": ["meeting", "standup", "retro", "sync", "call", "agenda"],
    "project": ["project", "phase", "milestone", "timeline", "deadline", "deliverable"],
    "onboarding": ["onboarding", "training", "documentation", "guide", "walkthrough"],
    "compliance": ["compliance", "audit", "regulation", "policy", "gdpr", "pci"],
    "migration": ["migration", "migrate", "upgrade", "transition", "legacy", "modernize"],
}


def _extract_topics(text_lower: str) -> list[str]:
    topics = []
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            topics.append(topic)
    return topics[:5]  # max 5 topics


def _extract_commitment_owner(text: str, position: int) -> str | None:
    """Try to find who made the commitment from surrounding context."""
    # Look backwards for a name pattern
    before = text[max(0, position - 100):position]
    # Check for "Name:" or "Name —" patterns
    match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*(?::|—|-)\s*$', before)
    if match:
        return match.group(1)
    # Check for "From: Name" in email context
    match = re.search(r'From:\s*([a-zA-Z.]+@[a-zA-Z.]+|[A-Z][a-z]+\s+[A-Z][a-z]+)', before)
    if match:
        return match.group(1)
    return None


def _extract_due_date(text: str) -> str | None:
    """Try to extract a due date from commitment text."""
    patterns = [
        r'by\s+((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,?\s+\d{4})?)',
        r'by\s+(\d{4}-\d{2}-\d{2})',
        r'by\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))',
        r'by\s+(end of (?:week|month|quarter|year))',
        r'by\s+(EOD|EOW|EOM)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _has_status_conflict(text_a: str, text_b: str) -> bool:
    """Check if two texts have conflicting status words."""
    status_pairs = [
        ({"active", "running", "live", "open"}, {"inactive", "stopped", "closed", "completed"}),
        ({"approved", "accepted"}, {"rejected", "denied", "declined"}),
        ({"paid"}, {"unpaid", "overdue", "outstanding"}),
        ({"phase 1", "phase 2", "phase 3", "phase 4"}, {"phase 1", "phase 2", "phase 3", "phase 4"}),
    ]
    for set_a, set_b in status_pairs:
        a_has = any(s in text_a for s in set_a)
        b_has = any(s in text_b for s in set_b)
        if a_has and b_has:
            a_status = next((s for s in set_a if s in text_a), "")
            b_status = next((s for s in set_b if s in text_b), "")
            if a_status != b_status:
                return True
    return False


def _infer_document_type(topics: list[str]) -> str | None:
    """Map topics to a document type for coverage tracking."""
    topic_to_type = {
        "contract": "contract",
        "pricing": "financial",
        "compliance": "compliance",
        "support": "support",
        "meeting": "meeting_notes",
        "project": "project",
        "onboarding": "onboarding",
    }
    for topic in topics:
        if topic in topic_to_type:
            return topic_to_type[topic]
    return None


def _build_analysis_prompt(
    chunk_text: str,
    existing_assertions: list[str] | None = None,
) -> str:
    """Build the LLM prompt for chunk analysis."""
    prompt = f"""Analyse the following text and extract structured information.

TEXT:
{chunk_text[:3000]}

Extract the following as JSON:
{{
  "sentiment": "positive" | "neutral" | "negative" | "escalation",
  "topics": ["list", "of", "topics"],
  "assertions": ["list of factual claims in the text"],
  "commitments": [{{"who": "person name", "what": "commitment", "due_date": "date or null"}}],
  "staleness_risk": 0.0 to 1.0 (how likely this info is outdated)
}}"""

    if existing_assertions:
        prompt += f"""

EXISTING ASSERTIONS (check for contradictions):
{json.dumps(existing_assertions[:20])}

If any extracted assertion contradicts an existing one, add:
  "contradictions": [{{"old": "existing assertion", "new": "new conflicting assertion"}}]
"""

    return prompt


def _parse_llm_analysis(response: str) -> ChunkAnalysis:
    """Parse LLM JSON response into ChunkAnalysis."""
    try:
        # Extract JSON from response (may have markdown wrapping)
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
        else:
            return ChunkAnalysis()

        return ChunkAnalysis(
            sentiment=data.get("sentiment", "neutral"),
            topics=data.get("topics", []),
            staleness_risk=float(data.get("staleness_risk", 0.0)),
            assertions=data.get("assertions", []),
            commitments=data.get("commitments", []),
            contradictions=data.get("contradictions", []),
        )
    except (json.JSONDecodeError, ValueError):
        return ChunkAnalysis()


# ---------------------------------------------------------------------------
# Knowledge gap detection
# ---------------------------------------------------------------------------

_OUTCOME_WITHOUT_RATIONALE = [
    re.compile(r"(?:renewed|approved|rejected|cancelled|terminated|extended)\s+(?:at|for|with)\s+(.{10,80})", re.IGNORECASE),
    re.compile(r"(?:discount|increase|decrease|change)\s+(?:of|to)\s+(.{5,50})", re.IGNORECASE),
    re.compile(r"(?:agreed|decided|chose|selected)\s+(?:to|on)\s+(.{10,80})", re.IGNORECASE),
]

_RATIONALE_INDICATORS = [
    r"\bbecause\b", r"\bdue to\b", r"\breason\b", r"\bsince\b",
    r"\bas a result\b", r"\bjustification\b", r"\bapproved by\b",
    r"\bper\s+(?:discussion|meeting|request)\b",
]


def detect_knowledge_gaps(
    chunk_text: str,
    entity_ids: list[int] | None = None,
) -> list[dict[str, str]]:
    """Detect outcomes/decisions without documented rationale."""
    gaps = []
    text_lower = chunk_text.lower()

    has_rationale = any(re.search(p, text_lower) for p in _RATIONALE_INDICATORS)

    if not has_rationale:
        for pattern in _OUTCOME_WITHOUT_RATIONALE:
            for match in pattern.finditer(chunk_text):
                gaps.append({
                    "gap_type": "missing_rationale",
                    "description": f"Decision/outcome without documented reason: {match.group(0)[:100]}",
                })

    return gaps[:3]  # max 3 gaps per chunk


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

_EVENT_PATTERNS = [
    (r"\b(?:deployed|released|launched|shipped)\s+(.{5,60})", "deployment"),
    (r"\b(?:hired|onboarded|joined)\s+(.{5,40})", "hiring"),
    (r"\b(?:incident|outage|downtime)\b.{0,40}", "incident"),
    (r"\b(?:contract|agreement)\s+(?:signed|renewed|expired|terminated)", "contract_change"),
    (r"\bprice\s+(?:change|increase|decrease|updated)", "pricing_change"),
    (r"\b(?:restructur|reorganiz)", "org_change"),
]


def extract_events(chunk_text: str) -> list[dict[str, str]]:
    """Extract discrete business events from text."""
    events = []
    for pattern, event_type in _EVENT_PATTERNS:
        for match in re.finditer(pattern, chunk_text, re.IGNORECASE):
            events.append({
                "event_type": event_type,
                "description": match.group(0).strip()[:120],
            })
    return events[:5]


# ---------------------------------------------------------------------------
# Corroboration scoring
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# State change extraction
# ---------------------------------------------------------------------------

_STATE_PATTERNS = [
    # Status changes
    (re.compile(r"(\w+(?:\s+\w+){0,3})\s+(?:has been|was|is now|has)\s+(cancelled|canceled|terminated|suspended|paused|activated|renewed|expired|completed|closed)", re.IGNORECASE),
     "status"),
    # Role changes
    (re.compile(r"(\w+(?:\s+\w+){0,2})\s+(?:has been|was|is now)\s+(promoted|transferred|reassigned|departed|resigned|hired|fired|terminated)", re.IGNORECASE),
     "role"),
    # Tier/plan changes
    (re.compile(r"(?:upgraded?|downgraded?|moved?|switched?|changed?)\s+(?:to|from)\s+(\w+)\s+(?:plan|tier|level)", re.IGNORECASE),
     "plan_tier"),
    # Contact changes
    (re.compile(r"(?:new|updated|changed)\s+(?:primary\s+)?contact\s*(?:is|:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", re.IGNORECASE),
     "primary_contact"),
    # Phase changes
    (re.compile(r"(?:project|phase)\s+(?:moved?|advanced?|progressed?)\s+(?:to|into)\s+(phase\s*\d+|stage\s*\d+|production|staging|development)", re.IGNORECASE),
     "project_phase"),
    # Payment status
    (re.compile(r"(?:invoice|payment)\s+\S+\s+(?:is|has been|was)\s+(paid|overdue|outstanding|pending|refunded)", re.IGNORECASE),
     "payment_status"),
]

_CONFIDENCE_MAP = {
    # Source type → base confidence for state changes
    "contract": 0.95,
    "policy": 0.90,
    "executive": 0.85,
    "financial": 0.85,
    "email": 0.70,
    "team": 0.60,
    "meeting": 0.55,
    "general": 0.50,
}


def extract_state_changes(
    chunk_text: str,
    chunk_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract state change signals from text.

    Returns list of dicts: {attribute, new_value, confidence, context}
    """
    changes = []
    doc_type = (chunk_metadata or {}).get("document_type", "general")
    base_confidence = _CONFIDENCE_MAP.get(doc_type, 0.50)

    for pattern, attr_type in _STATE_PATTERNS:
        for match in pattern.finditer(chunk_text):
            groups = match.groups()
            if len(groups) >= 2:
                subject = groups[0].strip()
                new_value = groups[1].strip().lower()
            elif len(groups) == 1:
                subject = ""
                new_value = groups[0].strip().lower()
            else:
                continue

            # Adjust confidence based on language strength
            confidence = base_confidence
            context = match.group(0).strip()

            # Lower confidence for conditional/tentative language
            lower_context = context.lower()
            if any(w in lower_context for w in ["considering", "might", "may", "could", "potentially", "possibly", "thinking about"]):
                confidence *= 0.5

            # Higher confidence for definitive language
            if any(w in lower_context for w in ["confirmed", "officially", "effective immediately", "as of today"]):
                confidence = min(1.0, confidence * 1.2)

            changes.append({
                "attribute": attr_type,
                "subject": subject,
                "new_value": new_value,
                "confidence": round(confidence, 2),
                "context": context[:150],
            })

    return changes


# ---------------------------------------------------------------------------
# Conversation ingestion
# ---------------------------------------------------------------------------


def prepare_conversation_for_ingestion(
    conversation_id: int,
    messages: list[dict[str, str]],
    summary: str | None = None,
) -> dict[str, Any]:
    """Prepare a completed conversation for ingestion back into the knowledge base.

    Returns a dict ready for chunking and embedding:
    {content, metadata, entity_mentions}
    """
    # Build conversation text
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Verra: {content}")

    conversation_text = "\n\n".join(parts)

    # Use summary if available, otherwise use full text
    indexable_text = summary or conversation_text

    return {
        "content": indexable_text,
        "full_conversation": conversation_text,
        "metadata": {
            "source_type": "agent_conversation",
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "document_type": "conversation",
            "authority_weight": 40,  # lower than documents, higher than nothing
        },
    }


# ---------------------------------------------------------------------------
# Communication tone analysis
# ---------------------------------------------------------------------------

_DIPLOMATIC_PATTERNS = [
    re.compile(r"\b(?:interesting|great)\s+(?:approach|idea|point|suggestion)\b", re.IGNORECASE),
    re.compile(r"\bI\s+(?:appreciate|understand|see)\s+(?:your|the)\b", re.IGNORECASE),
    re.compile(r"\b(?:however|that said|on the other hand|alternatively)\b", re.IGNORECASE),
    re.compile(r"\b(?:perhaps|maybe)\s+(?:we|you)\s+(?:could|should|might)\b", re.IGNORECASE),
]


def analyse_communication_tone(
    chunk_text: str,
    sentiment: str,
    thread_sentiment: str | None = None,
) -> str:
    """Classify communication tone beyond basic sentiment.

    Returns: 'straightforward', 'potentially_diplomatic',
             'potentially_sarcastic', 'mixed_signals'
    """
    diplomatic_hits = sum(1 for p in _DIPLOMATIC_PATTERNS if p.search(chunk_text))

    # Positive words in a negative thread → possible sarcasm
    if sentiment == "positive" and thread_sentiment == "negative":
        return "potentially_sarcastic"

    # Diplomatic language + subsequent contradiction → diplomatic
    if diplomatic_hits >= 2:
        return "potentially_diplomatic"

    # Mixed positive and negative signals in same chunk
    text_lower = chunk_text.lower()
    pos = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
    neg = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
    if pos >= 2 and neg >= 2:
        return "mixed_signals"

    return "straightforward"


def compute_corroboration(
    assertion: str,
    all_assertions: list[dict[str, Any]],
) -> int:
    """Count how many independent chunks support this assertion.

    Uses word overlap to find similar assertions from different sources.
    """
    assertion_words = set(assertion.lower().split()) - {"is", "are", "the", "a", "an", "was", "to", "for", "in", "on"}
    if len(assertion_words) < 3:
        return 0

    count = 0
    for other in all_assertions:
        other_text = other.get("claim_text", other.get("text", ""))
        other_words = set(other_text.lower().split()) - {"is", "are", "the", "a", "an", "was", "to", "for", "in", "on"}
        overlap = assertion_words & other_words
        if len(overlap) >= max(3, len(assertion_words) * 0.5):
            # Check different source
            if other.get("source_chunk_id") != other.get("_current_chunk_id"):
                count += 1

    return count
