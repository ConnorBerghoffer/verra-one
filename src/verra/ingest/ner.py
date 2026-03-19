"""Lightweight Named Entity Recognition for business documents.

Extracts people, companies, email addresses, projects, and locations
from text using regex patterns and heuristics. No SpaCy dependency
required (compatible with Python 3.14+).

The extractor is designed for business documents — emails, contracts,
invoices, meeting notes — not general-purpose NER.
"""


from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ExtractedEntity:
    """A single extracted entity mention."""
    text: str               # the raw mention as found in text
    entity_type: str        # 'person', 'company', 'email', 'project', 'location', 'money'
    confidence: float = 0.8 # rough confidence (0-1)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Email addresses
_EMAIL_RE = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')

# Money amounts: $12,500.00 or AUD 12,500 or $345,000
_MONEY_RE = re.compile(
    r'(?:(?:USD|AUD|EUR|GBP|CAD)\s*)?'
    r'\$[\d,]+(?:\.\d{2})?'
    r'|(?:USD|AUD|EUR|GBP|CAD)\s+[\d,]+(?:\.\d{2})?',
    re.IGNORECASE,
)

# Company suffixes that indicate an organization
_COMPANY_SUFFIXES = re.compile(
    r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*'
    r'\s+(?:Inc|Corp|Corporation|LLC|Ltd|Pty|Co|Group|Solutions|Technologies|Systems|Partners|Services|Consulting)\.?)\b'
)

# Project/product names — "Project X" or "Phase 1"
_PROJECT_RE = re.compile(
    r'\b((?:Project|Phase|Sprint|Module|Service|Platform|System)\s+[A-Z0-9][a-zA-Z0-9\s]*?)(?:[,.\n]|$)',
    re.IGNORECASE,
)

# Person names — heuristic: "FirstName LastName" patterns
# Looks for 2-3 capitalized words that aren't common non-name words
_COMMON_NON_NAMES = {
    'the', 'and', 'for', 'from', 'with', 'this', 'that', 'will', 'have', 'been',
    'are', 'was', 'were', 'not', 'but', 'all', 'can', 'had', 'her', 'his',
    'our', 'their', 'its', 'has', 'him', 'how', 'may', 'new', 'now', 'old',
    'see', 'way', 'who', 'did', 'get', 'let', 'say', 'she', 'too', 'use',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
    'january', 'february', 'march', 'april', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'key', 'action', 'items', 'next', 'steps', 'notes', 'agenda', 'meeting',
    'table', 'total', 'date', 'subject', 'status', 'name', 'email', 'phone',
    'monthly', 'annual', 'weekly', 'daily', 'custom', 'plan', 'starter',
    'professional', 'enterprise', 'scope', 'services', 'pricing', 'payment',
    'terms', 'effective', 'expiration', 'auto', 'renewal', 'parties',
    'provider', 'client', 'contact', 'executive', 'summary', 'proposed',
    'approach', 'team', 'duration', 'cost', 'remote', 'work', 'policy',
    'leave', 'expense', 'refund', 'active', 'inactive', 'pending', 'paid',
    'overdue', 'stale', 'lead', 'company', 'contract', 'start', 'end',
    'last', 'first', 'best', 'full', 'free', 'open', 'core', 'data',
    'source', 'type', 'format', 'page', 'section', 'role', 'title',
    'description', 'value', 'amount', 'currency', 'issued', 'due',
    'invoice', 'number', 'consulting', 'infrastructure', 'audit',
    'architecture', 'decision', 'record', 'migration', 'roadmap',
    'assessment', 'deliverable', 'running', 'complete', 'completed',
    'proposed', 'outstanding', 'senior', 'junior', 'lead', 'module',
    'service', 'platform', 'system', 'phase', 'sprint', 'project',
    'finalize', 'schedule', 'share', 'digital', 'pty', 'ltd', 'inc',
    'corp', 'group', 'google', 'meet', 'zoom', 'slack', 'teams',
    'nova', 'tech', 'solutions', 'architect', 'melbourne', 'sydney',
    'brisbane', 'perth', 'adelaide', 'london', 'york', 'san', 'francisco',
    'los', 'angeles', 'pre', 'post', 'mid', 'net', 'per', 'sub',
    'set', 'run', 'add', 'get', 'put', 'use', 'fix', 'cut',
    'note', 'notes', 'item', 'task', 'review', 'report', 'update',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
    'quick', 'question', 'response', 'discussion', 'follow',
}

# Title patterns that precede person names
_TITLE_PATTERNS = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|CEO|CTO|CFO|COO|VP|Director|Manager|Lead|Senior|Junior)\b\.?\s*',
    re.IGNORECASE,
)

# "Name, Title" or "Name (Title)" patterns common in business docs
_NAME_TITLE_RE = re.compile(
    r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[,\(]\s*'
    r'(?:CEO|CTO|CFO|COO|VP|Director|Manager|Lead\s+\w+|Senior\s+\w+|Engineer|Designer|Founder)',
    re.IGNORECASE,
)

# "From: Name" or "To: Name" in email contexts
_EMAIL_NAME_RE = re.compile(
    r'(?:From|To|Cc|Attendees?):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
)

# Explicit person references: "Contact: Name" or "Prepared by: Name"
_EXPLICIT_PERSON_RE = re.compile(
    r'(?:Contact|Prepared\s+by|Attendees?|Author|Assigned\s+to|Reported\s+by):\s*'
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    re.IGNORECASE,
)


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract named entities from text using regex + heuristics.

    Returns deduplicated entities sorted by type then name.
    """
    entities: list[ExtractedEntity] = []
    seen: set[str] = set()

    def _add(text_val: str, etype: str, conf: float = 0.8) -> None:
        key = (text_val.strip().lower(), etype)
        if key not in seen and len(text_val.strip()) > 1:
            seen.add(key)
            entities.append(ExtractedEntity(text=text_val.strip(), entity_type=etype, confidence=conf))

    # 1. Email addresses (highest confidence)
    for match in _EMAIL_RE.finditer(text):
        _add(match.group(), "email", 0.99)

    # 2. Money amounts
    for match in _MONEY_RE.finditer(text):
        _add(match.group(), "money", 0.95)

    # 3. Companies (by suffix)
    for match in _COMPANY_SUFFIXES.finditer(text):
        _add(match.group(1), "company", 0.9)

    # 4. Projects/phases
    for match in _PROJECT_RE.finditer(text):
        name = match.group(1).strip()
        if len(name) > 5:  # skip very short matches
            _add(name, "project", 0.7)

    # 5. People — multiple strategies

    # 5a. Name + Title pattern (highest confidence for people)
    for match in _NAME_TITLE_RE.finditer(text):
        _add(match.group(1), "person", 0.95)

    # 5b. Email context names
    for match in _EMAIL_NAME_RE.finditer(text):
        _add(match.group(1), "person", 0.9)

    # 5c. Explicit references
    for match in _EXPLICIT_PERSON_RE.finditer(text):
        _add(match.group(1), "person", 0.9)

    # 5d. General capitalized name heuristic (lower confidence)
    # Look for "Firstname Lastname" patterns not already captured
    for match in re.finditer(r'\b([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})\b', text):
        first, last = match.group(1), match.group(2)
        full = f"{first} {last}"

        # Skip if either word is a common non-name
        if first.lower() in _COMMON_NON_NAMES or last.lower() in _COMMON_NON_NAMES:
            continue

        # Skip if it looks like a section heading (all previous chars are whitespace/newline)
        pos = match.start()
        line_start = text.rfind('\n', 0, pos) + 1
        prefix = text[line_start:pos].strip()
        if prefix in ('', '#', '##', '###', '-', '*', '|'):
            # Could be a heading or table header — check if it's followed by a title/role
            after = text[match.end():match.end() + 30].strip()
            if not re.match(r'[,\(]?\s*(?:CEO|CTO|VP|Director|Manager|Lead|Engineer|Developer|Designer|Founder)', after, re.IGNORECASE):
                continue

        _add(full, "person", 0.6)

    # Sort by type then name for consistency
    entities.sort(key=lambda e: (e.entity_type, e.text.lower()))
    return entities


def extract_email_username(email: str) -> str | None:
    """Extract a likely person name from an email address.

    'john.smith@company.com' → 'John Smith'
    'jake_mitchell@co.com' → 'Jake Mitchell'
    'info@company.com' → None (not a person)
    """
    local = email.split("@")[0]

    # Skip generic addresses
    generic = {"info", "admin", "support", "sales", "billing", "hello", "contact",
               "noreply", "no-reply", "team", "help", "office"}
    if local.lower() in generic:
        return None

    # Split on . _ - and capitalize
    parts = re.split(r'[._\-]', local)
    if len(parts) >= 2:
        name = " ".join(p.capitalize() for p in parts if len(p) > 1)
        return name

    return None


def resolve_entities_to_registry(
    extracted: list[ExtractedEntity],
    entity_store: "EntityStore",
) -> list[int]:
    """Resolve extracted entities against the entity registry.

    For each extracted entity:
    1. Check if any alias matches an existing entity → use that ID
    2. If not, create a new entity with the extracted text as canonical name
    3. For email addresses, also try to derive a person name and merge

    Returns list of entity IDs that were linked.
    """
    entity_ids: list[int] = []
    seen_ids: set[int] = set()

    for ent in extracted:
        # Skip money amounts — we don't store these as entities
        if ent.entity_type == "money":
            continue

        # Try to resolve against existing aliases
        existing = entity_store.resolve(ent.text)

        if existing:
            eid = existing["id"]
        else:
            # For emails, try to create a linked person entity
            if ent.entity_type == "email":
                person_name = extract_email_username(ent.text)
                if person_name:
                    # Check if person already exists
                    person = entity_store.resolve(person_name)
                    if person:
                        # Add email as alias to existing person
                        entity_store.add_entity(
                            person["canonical_name"],
                            person["entity_type"],
                            aliases=[ent.text],
                        )
                        eid = person["id"]
                    else:
                        # Create person with email as alias
                        eid = entity_store.add_entity(
                            person_name,
                            "person",
                            aliases=[ent.text, person_name],
                        )
                else:
                    # Generic email, store as-is
                    eid = entity_store.add_entity(ent.text, "email", aliases=[ent.text])
            else:
                # Create new entity
                eid = entity_store.add_entity(ent.text, ent.entity_type, aliases=[ent.text])

        if eid not in seen_ids:
            entity_ids.append(eid)
            seen_ids.add(eid)

    return entity_ids


def extract_relationships(
    chunk_entity_ids: list[int],
    entity_store: "EntityStore",
    source_chunk_id: int | None = None,
) -> list[tuple[int, str, int]]:
    """Infer relationships from co-occurring entities in the same chunk.

    Simple heuristic: if a person and a company appear in the same chunk,
    infer a 'works_at' or 'client_of' relationship. If a person and a
    project co-occur, infer 'involved_in'.

    Returns list of (entity_a, relationship_type, entity_b) tuples.
    """
    if len(chunk_entity_ids) < 2:
        return []

    relationships: list[tuple[int, str, int]] = []
    entities = []
    for eid in chunk_entity_ids:
        ent = entity_store.get_entity(eid)
        if ent:
            entities.append(ent)

    people = [e for e in entities if e["entity_type"] == "person"]
    companies = [e for e in entities if e["entity_type"] == "company"]
    projects = [e for e in entities if e["entity_type"] == "project"]

    # Person ↔ Company → "associated_with"
    for person in people:
        for company in companies:
            relationships.append((person["id"], "associated_with", company["id"]))

    # Person ↔ Project → "involved_in"
    for person in people:
        for project in projects:
            relationships.append((person["id"], "involved_in", project["id"]))

    # Company ↔ Project → "related_to"
    for company in companies:
        for project in projects:
            relationships.append((company["id"], "related_to", project["id"]))

    return relationships
