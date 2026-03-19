"""Agentic tools — LLM-callable functions for actions and calculations."""


from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from verra.retrieval.router import parse_query
from verra.retrieval.search import search


# Tool definitions — OpenAI function-calling format

DRAFT_EMAIL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "draft_email",
        "description": (
            "Compose a professional email draft to a specified recipient. "
            "Searches the knowledge base for relevant context about the recipient "
            "or topic before writing the draft."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "The recipient's name or email address.",
                },
                "subject": {
                    "type": "string",
                    "description": "The email subject line.",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "What the email should be about — the topic, purpose, "
                        "or key points to include."
                    ),
                },
            },
            "required": ["to", "subject", "context"],
        },
    },
}

SUMMARIZE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "summarize_thread",
        "description": (
            "Search for emails or messages matching a query and produce a "
            "concise summary of the thread or conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The topic, subject, or participant to search for. "
                        "E.g. 'invoice discussion with Acme Corp' or 'project kickoff emails'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

FIND_RELATED_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "find_related",
        "description": (
            "Find all ingested data related to a specific person, company, or topic. "
            "Returns a structured summary of matching documents, emails, and known "
            "relationships from the entity registry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "The name of the person, company, or topic to look up.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of related documents to return (default 10).",
                },
            },
            "required": ["entity_name"],
        },
    },
}

CREATE_NOTE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "create_note",
        "description": (
            "Save a note or piece of information to persistent memory. "
            "Useful for storing preferences, decisions, or facts the user wants to remember."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "The category for organising the note, e.g. 'preference', "
                        "'fact', 'context', 'task'."
                    ),
                },
                "key": {
                    "type": "string",
                    "description": "A short unique identifier for this note within its category.",
                },
                "value": {
                    "type": "string",
                    "description": "The content of the note.",
                },
            },
            "required": ["category", "key", "value"],
        },
    },
}

SET_REMINDER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_reminder",
        "description": (
            "Store a reminder for a future date. "
            "The 'when' field accepts natural language like 'tomorrow', "
            "'next Friday', 'in 3 days', or absolute dates like '2026-04-01'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "description": "What the reminder is about.",
                },
                "when": {
                    "type": "string",
                    "description": (
                        "When the reminder should fire. Accepts natural language "
                        "(e.g. 'tomorrow', 'next week', 'in 5 days') or ISO dates."
                    ),
                },
            },
            "required": ["what", "when"],
        },
    },
}

CALCULATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression and return the exact result. "
            "Use this whenever you need to add, subtract, multiply, divide, "
            "sum a list of numbers, compute percentages, or do any arithmetic. "
            "ALWAYS use this tool instead of doing mental math."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A Python math expression to evaluate. Examples: "
                        "'29500 + 9000 + 6000', '450000 - 320000', "
                        "'37500 * 12', '(180000 - 140000) / 180000 * 100', "
                        "'sum([42000, 44500, 46000])'"
                    ),
                },
            },
            "required": ["expression"],
        },
    },
}


# All agentic tools in one list — pass to LLMClient.complete_with_tools()
AGENTIC_TOOLS: list[dict[str, Any]] = [
    DRAFT_EMAIL_TOOL,
    SUMMARIZE_TOOL,
    FIND_RELATED_TOOL,
    CREATE_NOTE_TOOL,
    SET_REMINDER_TOOL,
    CALCULATE_TOOL,
]


# Date parsing helper

# Patterns for "in N <unit>" — e.g. "in 3 days", "in 2 weeks"
_IN_N_UNITS = re.compile(
    r"\bin\s+(\d+)\s+(day|days|week|weeks|month|months)\b",
    re.IGNORECASE,
)

# "next <weekday>" — Monday=0 … Sunday=6
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_NEXT_WEEKDAY = re.compile(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def _parse_when(when: str) -> datetime:
    """Parse a natural-language or ISO date string into a UTC datetime.

    Heuristics handled (in order):
      - "tomorrow"
      - "next week"
      - "next <weekday>"  (e.g. "next Friday")
      - "in N days/weeks/months"
      - Absolute ISO date via dateutil.parser (with stdlib fallback)

    Falls back to 24 hours from now if nothing matches.
    """
    now = datetime.now(tz=timezone.utc)
    text = when.strip().lower()

    if text in ("tomorrow", "next day"):
        return now + timedelta(days=1)

    if text in ("next week",):
        return now + timedelta(weeks=1)

    m = _IN_N_UNITS.search(text)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("day"):
            return now + timedelta(days=n)
        if unit.startswith("week"):
            return now + timedelta(weeks=n)
        if unit.startswith("month"):
            return now + timedelta(days=n * 30)

    m = _NEXT_WEEKDAY.search(text)
    if m:
        target_dow = _WEEKDAYS[m.group(1).lower()]
        current_dow = now.weekday()
        days_ahead = (target_dow - current_dow) % 7 or 7  # always at least 1 day
        return now + timedelta(days=days_ahead)

    # Try dateutil for absolute dates
    try:
        from dateutil import parser as dateutil_parser  # type: ignore[import-untyped]

        parsed = dateutil_parser.parse(when, dayfirst=False)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        pass

    # Try stdlib strptime for common ISO formats
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(when, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    # Ultimate fallback: 24 hours from now
    return now + timedelta(days=1)


# ToolHandler


class ToolHandler:
    """Route LLM tool invocations to their implementations.

    Each handler method performs a focused task using the stores and LLM
    client supplied at construction time.  All methods return plain strings
    that are fed back to the LLM as tool results.

    Parameters
    ----------
    llm:
        An LLMClient instance used for sub-calls (draft, summarise).
    metadata_store:
        MetadataStore — used indirectly through the search helper.
    vector_store:
        VectorStore — used indirectly through the search helper.
    memory_store:
        MemoryStore — used by create_note and set_reminder.
    entity_store:
        EntityStore (optional) — used by find_related for relationship lookups.
    """

    def __init__(
        self,
        llm: Any,
        metadata_store: Any,
        vector_store: Any,
        memory_store: Any,
        entity_store: Any | None = None,
    ) -> None:
        self.llm = llm
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.entity_store = entity_store

        # Dispatch table — maps tool name to bound method
        self._handlers: dict[str, Any] = {
            "draft_email": self._draft_email,
            "summarize_thread": self._summarize_thread,
            "find_related": self._find_related,
            "create_note": self._create_note,
            "set_reminder": self._set_reminder,
            "calculate": self._calculate,
        }

    # ------------------------------------------------------------------
    # Public routing entry point
    # ------------------------------------------------------------------

    def handle(self, tool_name: str, args: dict[str, Any]) -> str:
        """Route a tool call to the correct handler.

        Returns the result string that will be fed back to the LLM.
        Unknown tool names return an error string rather than raising so
        that the LLM can continue the conversation gracefully.
        """
        handler = self._handlers.get(tool_name)
        if handler is None:
            return f"[Unknown tool: {tool_name!r}. Available: {', '.join(self._handlers)}]"
        try:
            return handler(**args)
        except TypeError as exc:
            # Wrong / missing arguments from the LLM
            return f"[Tool argument error for {tool_name!r}: {exc}]"
        except Exception as exc:
            return f"[Tool error in {tool_name!r}: {exc}]"

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _draft_email(self, to: str, subject: str, context: str) -> str:
        """Search for relevant context then ask the LLM to write an email draft."""
        # Search for background information about the recipient/topic
        search_query = f"{to} {context}"
        classified = parse_query(search_query)
        hits = search(
            classified,
            self.metadata_store,
            self.vector_store,
            n_results=5,
            entity_store=self.entity_store,
        )

        if hits:
            context_block = "\n\n".join(
                f"[{i + 1}] {r.text[:500]}" for i, r in enumerate(hits[:5])
            )
            knowledge_section = (
                f"\n\nRelevant background from the knowledge base:\n{context_block}"
            )
        else:
            knowledge_section = "\n\nNo specific background was found in the knowledge base."

        prompt = (
            f"Draft a professional email with the following details:\n\n"
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"Purpose / context: {context}"
            f"{knowledge_section}\n\n"
            f"Write only the email body (not headers). "
            f"Be concise, professional, and directly address the stated purpose. "
            f"Sign off with a placeholder like '[Your Name]'."
        )

        draft = self.llm.complete([{"role": "user", "content": prompt}])
        return f"EMAIL DRAFT\n{'=' * 40}\nTo: {to}\nSubject: {subject}\n\n{draft.strip()}"

    def _summarize_thread(self, query: str) -> str:
        """Retrieve email chunks matching the query and summarise them."""
        # Try with an email source-type filter first
        classified = parse_query(query)
        classified.source_type = "email"

        hits = search(
            classified,
            self.metadata_store,
            self.vector_store,
            n_results=10,
            entity_store=self.entity_store,
        )

        # If the email filter yields nothing, fall back to unfiltered search
        if not hits:
            classified_unfiltered = parse_query(query)
            hits = search(
                classified_unfiltered,
                self.metadata_store,
                self.vector_store,
                n_results=10,
                entity_store=self.entity_store,
            )

        if not hits:
            return f"No emails or messages found matching '{query}'."

        chunks_text = "\n\n---\n\n".join(
            f"[Source: {r.metadata.get('subject') or r.metadata.get('file_name', 'unknown')}]\n{r.text[:800]}"
            for r in hits
        )

        prompt = (
            f"The user asked to summarise an email thread about: {query!r}\n\n"
            f"Here are the relevant email excerpts retrieved from the knowledge base:\n\n"
            f"{chunks_text}\n\n"
            f"Produce a concise summary (3–6 bullet points) covering:\n"
            f"- The main topic and participants\n"
            f"- Key decisions or outcomes\n"
            f"- Any open action items or follow-ups\n"
            f"Base the summary strictly on the excerpts above."
        )

        summary = self.llm.complete([{"role": "user", "content": prompt}])
        return f"THREAD SUMMARY — '{query}'\n{'=' * 40}\n{summary.strip()}"

    def _find_related(self, entity_name: str, max_results: int = 10) -> str:
        """Gather all knowledge about an entity and format a structured report."""
        lines: list[str] = [f"RELATED DATA — '{entity_name}'", "=" * 40]

        # --- Entity registry lookup ---
        entity_info: dict[str, Any] | None = None
        relationships: list[dict[str, Any]] = []
        related_entities: list[dict[str, Any]] = []

        if self.entity_store is not None:
            entity_info = self.entity_store.resolve(entity_name)
            if entity_info:
                lines.append(
                    f"\nEntity: {entity_info['canonical_name']} "
                    f"(type: {entity_info['entity_type']}, id: {entity_info['id']})"
                )
                relationships = self.entity_store.get_relationships(entity_info["id"])
                related_entities = self.entity_store.get_related_entities(entity_info["id"])
            else:
                lines.append(f"\n('{entity_name}' is not in the entity registry — searching by text)")

        # --- Relationships ---
        if relationships:
            lines.append("\nRelationships:")
            for rel in relationships[:10]:
                # Show the relationship from the perspective of the queried entity
                if rel["entity_a"] == entity_info["id"]:  # type: ignore[index]
                    lines.append(
                        f"  {rel['name_a']} --[{rel['relationship_type']}]--> {rel['name_b']}"
                    )
                else:
                    lines.append(
                        f"  {rel['name_b']} --[{rel['relationship_type']}]--> {rel['name_a']}"
                    )

        if related_entities:
            lines.append("\nRelated entities:")
            for ent in related_entities[:5]:
                lines.append(f"  • {ent['canonical_name']} ({ent['entity_type']})")

        # --- Vector search for document/email chunks ---
        classified = parse_query(entity_name)
        hits = search(
            classified,
            self.metadata_store,
            self.vector_store,
            n_results=max_results,
            entity_store=self.entity_store,
        )

        if hits:
            lines.append(f"\nMatching documents / emails ({len(hits)} found):")
            seen_sources: set[str] = set()
            for r in hits:
                source = (
                    r.metadata.get("subject")
                    or r.metadata.get("file_name")
                    or r.metadata.get("source_type", "unknown")
                )
                source_type = r.metadata.get("source_type", "")
                date = r.metadata.get("date") or r.metadata.get("valid_from", "")
                label = f"{source} [{source_type}]"
                if date:
                    label += f" ({date[:10]})"
                if label not in seen_sources:
                    seen_sources.add(label)
                    # Include a brief excerpt
                    excerpt = r.text[:200].replace("\n", " ")
                    lines.append(f"  • {label}\n    {excerpt}…")
        else:
            lines.append("\nNo documents or emails found in the knowledge base.")

        return "\n".join(lines)

    def _create_note(self, category: str, key: str, value: str) -> str:
        """Persist a note to MemoryStore and return a confirmation string."""
        self.memory_store.set_memory(
            category=category,
            key=key,
            value=value,
            source="tool:create_note",
        )
        return (
            f"Note saved.\n"
            f"  Category : {category}\n"
            f"  Key      : {key}\n"
            f"  Value    : {value}"
        )

    def _set_reminder(self, what: str, when: str) -> str:
        """Parse a natural-language date, store the reminder, and confirm."""
        try:
            expires_at = _parse_when(when)
        except Exception:
            # _parse_when has its own fallback but guard defensively
            expires_at = datetime.now(tz=timezone.utc) + timedelta(days=1)

        expires_iso = expires_at.strftime("%Y-%m-%d %H:%M UTC")

        self.memory_store.set_memory(
            category="reminder",
            key=what,
            value=when,
            source="tool:set_reminder",
            expires_at=expires_at,
        )

        return (
            f"Reminder set.\n"
            f"  What    : {what}\n"
            f"  When    : {when}\n"
            f"  Stored as: {expires_iso}"
        )

    def _calculate(self, expression: str) -> str:
        """Safely evaluate a math expression and return the result.

        Only allows basic arithmetic, built-in math functions, and sum/min/max.
        No access to imports, file system, or arbitrary code execution.
        """
        import ast
        import math
        import operator

        # Allowed operations
        _OPS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        # Allowed built-in functions
        _FUNCS = {
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "len": len,
            "sqrt": math.sqrt,
            "ceil": math.ceil,
            "floor": math.floor,
        }

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {node.value!r}")
            elif isinstance(node, ast.UnaryOp):
                op = _OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
                return op(_eval(node.operand))
            elif isinstance(node, ast.BinOp):
                op = _OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
                return op(_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
                    args = [_eval(a) for a in node.args]
                    return _FUNCS[node.func.id](*args)
                raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
            elif isinstance(node, ast.List):
                return [_eval(e) for e in node.elts]
            elif isinstance(node, ast.Tuple):
                return tuple(_eval(e) for e in node.elts)
            else:
                raise ValueError(f"Unsupported expression: {type(node).__name__}")

        try:
            # Strip dollar signs and commas inside numbers (e.g. "$1,000" -> "1000")
            # Only strip commas between digits to preserve function argument commas
            clean = expression.replace("$", "")
            clean = re.sub(r"(\d),(\d)", r"\1\2", clean)
            tree = ast.parse(clean, mode="eval")
            result = _eval(tree)
            # Format nicely
            if isinstance(result, float) and result == int(result):
                result = int(result)
            if isinstance(result, (int, float)):
                formatted = f"{result:,.2f}" if isinstance(result, float) else f"{result:,}"
                return f"{formatted}"
            return str(result)
        except Exception as exc:
            return f"[Calculation error: {exc}. Expression: {expression!r}]"
