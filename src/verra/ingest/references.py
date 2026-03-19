"""Detect and store cross-document references.

When a document mentions another document ("see the SLA", "per JIRA-4521",
"as discussed in the Q3 review"), create explicit links between chunks.
"""


from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verra.store.metadata import MetadataStore

# ---------------------------------------------------------------------------
# Reference patterns
# ---------------------------------------------------------------------------

_REFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"(?:see|refer to|per|as per|according to|in the)\s+"
            r"(?:the\s+)?([A-Z][A-Za-z\s]+(?:document|report|policy|agreement|"
            r"SLA|contract|proposal|plan))",
            re.IGNORECASE,
        ),
        "document",
    ),
    (
        re.compile(
            r"(?:JIRA|TICKET|ISSUE|PR|MR)[-#]\d+",
            re.IGNORECASE,
        ),
        "ticket",
    ),
    (
        re.compile(
            r"(?:attached|enclosed|included)\s+(?:is|are|herewith)\s+(?:the\s+)?(.{5,60})",
            re.IGNORECASE,
        ),
        "document",
    ),
    (
        re.compile(
            r"(?:as discussed|as agreed|as noted)\s+(?:in|during|at)\s+(?:the\s+)?(.{5,60})",
            re.IGNORECASE,
        ),
        "discussion",
    ),
]


def extract_references(text: str) -> list[dict]:
    """Extract cross-document references from text.

    Returns a list of dicts with keys:
    - ``reference_text``: the matched raw text
    - ``reference_type``: one of ``'document'``, ``'ticket'``, ``'discussion'``
    - ``target_hint``: the capture group value that hints at the target name
      (may be the same as ``reference_text`` for patterns without a named group)
    """
    refs: list[dict] = []
    seen_texts: set[str] = set()

    for pattern, ref_type in _REFERENCE_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(0).strip()
            if raw in seen_texts:
                continue
            seen_texts.add(raw)

            # Prefer first capture group as the target hint, else use the full match.
            if m.lastindex and m.lastindex >= 1:
                hint = m.group(1).strip()
            else:
                hint = raw

            refs.append(
                {
                    "reference_text": raw,
                    "reference_type": ref_type,
                    "target_hint": hint,
                }
            )

    return refs


def resolve_references(
    references: list[dict],
    metadata_store: "MetadataStore",
) -> list[dict]:
    """Try to match references to actual ingested documents.

    For each reference, searches the metadata store for a document whose
    file name or path contains the target hint (case-insensitive).

    Returns a list of dicts with keys:
    - ``reference_text``
    - ``resolved_document_id`` (int or None if unresolved)
    - ``confidence`` (float 0.0-1.0)
    """
    results: list[dict] = []
    all_docs = metadata_store.list_documents()

    for ref in references:
        hint = ref.get("target_hint", "").lower().strip()
        resolved_id: int | None = None
        confidence = 0.0

        if hint:
            for doc in all_docs:
                file_name = (doc.get("file_name") or "").lower()
                file_path = (doc.get("file_path") or "").lower()
                # Exact substring match in name or path
                if hint in file_name or hint in file_path:
                    resolved_id = doc["id"]
                    confidence = 0.8
                    break
                # Partial word-level overlap
                hint_words = set(re.findall(r"[a-z0-9]+", hint))
                name_words = set(re.findall(r"[a-z0-9]+", file_name))
                if hint_words and len(hint_words & name_words) / len(hint_words) >= 0.5:
                    resolved_id = doc["id"]
                    confidence = 0.5
                    # Don't break — keep looking for a better match

        results.append(
            {
                "reference_text": ref["reference_text"],
                "resolved_document_id": resolved_id,
                "confidence": confidence,
            }
        )

    return results
