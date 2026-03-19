"""Chat engine — retrieval-augmented generation with source attribution."""


from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from verra.agent.llm import LLMClient, RETRIEVAL_TOOL
from verra.agent.tools import AGENTIC_TOOLS, ToolHandler
from verra.retrieval.router import parse_query
from verra.retrieval.search import SearchResult, search
from verra.store.memory import MemoryStore
from verra.store.metadata import MetadataStore
from verra.store.vector import VectorStore


_SYSTEM_PROMPT = """You are Verra, a private AI assistant that helps the user understand their business data.

Rules:
- Base your answers ONLY on the provided context and search results.
- You have a search_knowledge_base tool. Use it to find relevant information.
- Search multiple times if the first search doesn't fully answer the question.
- You also have action tools: draft_email, summarize_thread, find_related, create_note, set_reminder.
- You have a calculate tool. You MUST use it whenever a question involves numbers, totals, sums, differences, averages, percentages, or any arithmetic. Call calculate with a Python expression like "29500 + 9000 + 6000" or "sum([42000, 44500, 46000])". NEVER attempt mental math — always use the tool.
- When context contains conflicting information, prefer the source with the most recent date. Flag the conflict.
- If you don't have enough information, say "I don't have that information in your data" clearly.
- Always cite sources by file name or email subject.
- Be thorough — list ALL relevant items, don't truncate lists.
- Be concise. Use bullet points.
"""

_STREAM_SYSTEM_PROMPT = """You are Verra, a private AI assistant that helps the user understand their business data.

Rules:
- Base your answers ONLY on the provided context below.
- When context contains conflicting information, prefer the source with the most recent date. Flag the conflict.
- If the context doesn't contain enough information, say "I don't have that information in your data."
- Cite sources by file name (e.g. "according to Q4_2025_Financial_Summary.txt").
- Be thorough — list ALL relevant items, don't truncate lists.
- Be concise. Use bullet points.
- When doing math, show your work step by step.
"""

_NO_ANSWER = (
    "I don't have enough information in your data to answer that question confidently.\n"
    "Try ingesting more documents or rephrasing your question."
)

_MIN_SCORE = -1.2  # ChromaDB L2 distance threshold — relaxed to avoid rejecting relevant results


# Knowledge assessment


@dataclass
class KnowledgeAssessment:
    """Result of assessing what Verra knows about a query."""

    has_relevant_data: bool
    known_entities: list[str] = field(default_factory=list)
    unknown_entities: list[str] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)
    confidence_note: str = ""  # prepended to the answer when non-empty


def _assess_knowledge(
    query: str,
    results: list[SearchResult],
    entity_store: Any | None,
) -> KnowledgeAssessment:
    """Assess what we know and don't know about a query."""
    has_data = bool(results) and max((r.score for r in results), default=0.0) >= _MIN_SCORE
    note_parts: list[str] = []

    if entity_store is None:
        if has_data and results:
            best = max(r.score for r in results)
            if best < 0.5:
                note_parts.append("Based on limited data, the following may be incomplete:")
        return KnowledgeAssessment(
            has_relevant_data=has_data,
            confidence_note=" ".join(note_parts),
        )

    words = query.split()
    candidates: list[str] = list(words)
    for i in range(len(words) - 1):
        candidates.append(f"{words[i]} {words[i + 1]}")

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for c in candidates:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            unique_candidates.append(c)

    known: list[str] = []
    unknown: list[str] = []

    for candidate in unique_candidates:
        if len(candidate) <= 2:
            continue
        entity = entity_store.resolve(candidate)
        if entity:
            known.append(entity["canonical_name"])
        elif len(candidate) >= 4 and candidate[0].isupper():
            unknown.append(candidate)

    if not has_data and unknown and not known:
        note_parts.append("This entity doesn't appear in any ingested data.")
    elif not has_data and known:
        note_parts.append(f"I have records for {', '.join(known)} but nothing about this topic.")

    if has_data and results:
        best = max(r.score for r in results)
        if best < 0.5:
            note_parts.append("Based on limited data, the following may be incomplete:")

    return KnowledgeAssessment(
        has_relevant_data=has_data,
        known_entities=known,
        unknown_entities=unknown,
        confidence_note=" ".join(note_parts),
    )


# Response dataclass


@dataclass
class ChatResponse:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    query_type: str = "semantic"
    had_context: bool = True
    assessment: KnowledgeAssessment | None = None


# Activity callback type

ActivityCallback = Callable[[str, dict], None] | None


# Chat engine


class ChatEngine:
    """Orchestrates retrieval + LLM to answer user questions."""

    def __init__(
        self,
        llm: LLMClient,
        metadata_store: MetadataStore,
        vector_store: VectorStore,
        memory_store: MemoryStore,
        n_results: int = 12,
        conversation_id: int | None = None,
        entity_store: Any | None = None,       # EntityStore
    ) -> None:
        self.llm = llm
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.entity_store = entity_store
        self.n_results = n_results

        # Conversation session
        if conversation_id is None:
            self.conversation_id = self.memory_store.new_conversation()
        else:
            self.conversation_id = conversation_id

        # In-session message history for multi-turn context (last N turns)
        self._history: list[dict[str, str]] = []
        self._max_history_turns = 6  # 3 pairs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        user_message: str,
        use_multi_hop: bool = True,
        on_activity: ActivityCallback = None,
    ) -> ChatResponse:
        """Process a user question and return a ChatResponse.

        Parameters
        ----------
        user_message:
            The user's question.
        use_multi_hop:
            If True (default) and the LLM supports tool calling, use the
            agentic multi-hop retrieval path. Falls back to single-pass on
            any exception.
        on_activity:
            Optional callback invoked at each processing stage to power the
            live activity panel. Signature: (category: str, data: dict) -> None.
            Categories emitted: "classify", "retrieval", "results",
            "llm_start", "llm_done".
        """
        def _emit(category: str, data: dict) -> None:
            if on_activity is not None:
                try:
                    on_activity(category, data)
                except Exception:
                    pass

        classified = parse_query(user_message)
        _emit("classify", {"query": user_message, "type": classified.query_type.value})

        if use_multi_hop:
            try:
                return self._ask_multi_hop(user_message, classified, _emit)
            except Exception:
                # Model doesn't support tool calling or another transient error —
                # fall through to the reliable single-pass path
                pass

        return self._ask_single_pass(user_message, classified, _emit)

    def retrieve(self, user_message: str) -> tuple[list[SearchResult], Any]:
        """Run retrieval only, return results and classified query.

        Call this first to get search results, then pass results to
        stream_with_context() for token-by-token LLM streaming.
        """
        classified = parse_query(user_message)
        results = search(
            classified,
            self.metadata_store,
            self.vector_store,
            n_results=self.n_results + 4,
            entity_store=self.entity_store,
        )
        return results, classified

    def stream_with_context(
        self, user_message: str, results: list[SearchResult]
    ) -> Iterator[str]:
        """Stream LLM response given pre-fetched retrieval results.

        Persists the conversation after streaming completes.
        Yields each text chunk as it arrives from the LLM.
        """
        best_score = max((r.score for r in results), default=0.0)
        if not results or best_score < _MIN_SCORE:
            yield _NO_ANSWER
            self._persist(user_message, _NO_ANSWER)
            self._set_conversation_title(user_message)
            return

        context_blocks = _format_context(results)
        messages = _build_messages(
            system=_STREAM_SYSTEM_PROMPT,
            history=self._history,
            context=context_blocks,
            user_message=user_message,
        )

        full_answer: list[str] = []
        for chunk in self.llm.stream(messages):
            full_answer.append(chunk)
            yield chunk

        answer = "".join(full_answer)
        self._persist(user_message, answer)
        self._update_history(user_message, answer)
        self._set_conversation_title(user_message)

    def stream_ask(self, user_message: str) -> Iterator[str]:
        """Stream the answer token by token.

        Yields each text chunk as it arrives from the LLM.
        Note: sources are NOT yielded; call ask() if you need them.
        Multi-hop is not supported in streaming mode; uses single-pass.
        Prefer retrieve() + stream_with_context() for richer status UX.
        """
        results, _classified = self.retrieve(user_message)
        yield from self.stream_with_context(user_message, results)

    # ------------------------------------------------------------------
    # Multi-hop retrieval path
    # ------------------------------------------------------------------

    def _ask_multi_hop(
        self,
        user_message: str,
        classified: Any,
        emit: Callable[[str, dict], None],
    ) -> ChatResponse:
        """Use agentic tool-calling for iterative retrieval."""
        # Accumulated results across all tool calls
        all_results: list[SearchResult] = []

        emit("retrieval", {"type": classified.query_type.value, "mode": "multi-hop"})

        # Initialize agentic tool handler for non-search tools
        agentic_handler = ToolHandler(
            llm=self.llm,
            metadata_store=self.metadata_store,
            vector_store=self.vector_store,
            memory_store=self.memory_store,
            entity_store=self.entity_store,
        )

        def tool_handler(tool_name: str, args: dict[str, Any]) -> str:
            if tool_name != "search_knowledge_base":
                return agentic_handler.handle(tool_name, args)

            query_text = args.get("query", user_message)
            source_type_filter = args.get("source_type")
            entity_filter = args.get("entity_name")

            # Build a classified query from the tool's requested search
            inner_query = parse_query(query_text)
            if source_type_filter and source_type_filter != "any":
                inner_query.source_type = source_type_filter
            if entity_filter:
                inner_query.raw = f"{inner_query.raw} {entity_filter}"

            hits = search(
                inner_query,
                self.metadata_store,
                self.vector_store,
                n_results=self.n_results,
                entity_store=self.entity_store,
            )
            all_results.extend(hits)

            emit("results", {
                "count": len(hits),
                "top_score": round(hits[0].score, 3) if hits else 0,
                "sources": list({
                    h.metadata.get("file_name", h.metadata.get("source_type", "?"))
                    for h in hits[:3]
                }),
            })

            if not hits:
                return "No results found for this search."

            return _format_context(hits)

        messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_message})

        emit("llm_start", {"model": getattr(self.llm, "model", "unknown"), "mode": "multi-hop"})

        answer = self.llm.complete_with_tools(
            messages=messages,
            tools=[RETRIEVAL_TOOL] + AGENTIC_TOOLS,
            tool_handler=tool_handler,
            max_rounds=3,
        )

        emit("llm_done", {"tokens_approx": len(answer.split()), "sources": len(all_results)})

        # Assess knowledge coverage using accumulated results
        assessment = _assess_knowledge(
            user_message, all_results, self.entity_store
        )

        if assessment.confidence_note:
            answer = f"{assessment.confidence_note}\n\n{answer}"

        sources = _extract_sources(all_results)

        self._persist(user_message, answer)
        self._update_history(user_message, answer)
        self._set_conversation_title(user_message)

        return ChatResponse(
            answer=answer,
            sources=sources,
            query_type=classified.query_type.value,
            had_context=bool(all_results),
            assessment=assessment,
        )

    # ------------------------------------------------------------------
    # Single-pass fallback path
    # ------------------------------------------------------------------

    def _ask_single_pass(
        self,
        user_message: str,
        classified: Any,
        emit: Callable[[str, dict], None],
    ) -> ChatResponse:
        """Original single-pass retrieval path (always works)."""
        emit("retrieval", {"type": classified.query_type.value, "mode": "single-pass"})

        results = search(
            classified,
            self.metadata_store,
            self.vector_store,
            n_results=self.n_results,
            entity_store=self.entity_store,
        )

        emit("results", {
            "count": len(results),
            "top_score": round(results[0].score, 3) if results else 0,
            "sources": list({
                r.metadata.get("file_name", r.metadata.get("source_type", "?"))
                for r in results[:3]
            }),
        })

        assessment = _assess_knowledge(
            user_message, results, self.entity_store
        )

        if not assessment.has_relevant_data:
            no_answer = assessment.confidence_note or _NO_ANSWER
            self._persist(user_message, no_answer)
            self._set_conversation_title(user_message)
            emit("llm_done", {"tokens_approx": 0, "skipped": True})
            return ChatResponse(
                answer=no_answer,
                sources=[],
                query_type=classified.query_type.value,
                had_context=False,
                assessment=assessment,
            )

        context_blocks = _format_context(results)

        messages = _build_messages(
            system=_SYSTEM_PROMPT,
            history=self._history,
            context=context_blocks,
            user_message=user_message,
        )

        # Inject a math hint into the prompt if the question involves numbers
        math_words = {"total", "sum", "how much", "spend", "cost", "increase",
                      "decrease", "difference", "average", "combined", "add up"}
        if any(w in user_message.lower() for w in math_words):
            # Extract numbers from the context so the LLM has them ready
            import re as _re
            numbers = _re.findall(r"\$?([\d,]+(?:\.\d+)?)", context_blocks)
            if numbers:
                math_hint = (
                    "\n\nIMPORTANT: This question requires arithmetic. "
                    "Show your calculation step by step. "
                    "Add the specific numbers from the sources."
                )
                messages[-1]["content"] += math_hint

        emit("llm_start", {"model": getattr(self.llm, "model", "unknown"), "mode": "single-pass"})
        answer = self.llm.complete(messages)
        emit("llm_done", {"tokens_approx": len(answer.split()), "sources": len(results)})

        if assessment.confidence_note:
            answer = f"{assessment.confidence_note}\n\n{answer}"

        sources = _extract_sources(results)

        self._persist(user_message, answer)
        self._update_history(user_message, answer)
        self._set_conversation_title(user_message)

        return ChatResponse(
            answer=answer,
            sources=sources,
            query_type=classified.query_type.value,
            had_context=True,
            assessment=assessment,
        )


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist(self, user_message: str, answer: str) -> None:
        self.memory_store.add_message(self.conversation_id, "user", user_message)
        self.memory_store.add_message(self.conversation_id, "assistant", answer)

    def _set_conversation_title(self, user_message: str) -> None:
        """Set conversation title from the first user message if not already set."""
        try:
            row = self.memory_store._conn.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (self.conversation_id,),
            ).fetchone()
            if row and not row[0]:
                # Truncate to ~60 chars, trim at a word boundary
                title = user_message.strip()
                if len(title) > 60:
                    title = title[:57].rsplit(" ", 1)[0] + "..."
                self.memory_store._conn.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (title, self.conversation_id),
                )
                self.memory_store._conn.commit()
        except Exception:
            pass  # non-critical — never crash the chat loop

    def _update_history(self, user_message: str, answer: str) -> None:
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": answer})
        max_msgs = self._max_history_turns * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def _entity_ids_for_query(self, query: str) -> list[int]:
        """Return entity IDs mentioned in the query, using entity_store."""
        if self.entity_store is None:
            return []
        words = query.split()
        candidates: list[str] = list(words)
        for i in range(len(words) - 1):
            candidates.append(f"{words[i]} {words[i + 1]}")
        seen: set[int] = set()
        ids: list[int] = []
        for c in candidates:
            entity = self.entity_store.resolve(c)
            if entity and entity["id"] not in seen:
                seen.add(entity["id"])
                ids.append(entity["id"])
        return ids



# Module-level helpers


def _format_context(results: list[SearchResult]) -> str:
    """Turn retrieved chunks into a context block for the prompt."""
    parts: list[str] = []
    for i, r in enumerate(results):
        source_label = _source_label(r.metadata)
        authority_note = f" [authority: {r.authority_weight}]" if r.authority_weight != 50 else ""
        parts.append(f"[Source {i + 1}: {source_label}{authority_note}]\n{r.text}")
    return "\n\n---\n\n".join(parts)


def _build_messages(
    system: str,
    history: list[dict[str, Any]],
    context: str,
    user_message: str,
) -> list[dict[str, Any]]:
    """Assemble the messages list for the LLM call."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context from your data:\n\n{context}\n\nQuestion: {user_message}",
        }
    )
    return messages


def _extract_sources(results: list[SearchResult]) -> list[dict[str, Any]]:
    """Build a deduplicated list of source citations from search results."""
    seen: set[str] = set()
    sources: list[dict[str, Any]] = []
    for r in results:
        label = _source_label(r.metadata)
        if label not in seen:
            seen.add(label)
            sources.append({"label": label, "metadata": r.metadata, "score": round(r.score, 3)})
    return sources


def _source_label(metadata: dict[str, Any]) -> str:
    """Return a human-readable source label from chunk metadata."""
    if metadata.get("file_name"):
        name = metadata["file_name"]
        if metadata.get("format") == "pdf":
            return f"{name} (PDF)"
        return name
    if metadata.get("subject"):
        return f"Email: {metadata['subject']}"
    return metadata.get("source_type", "unknown source")
