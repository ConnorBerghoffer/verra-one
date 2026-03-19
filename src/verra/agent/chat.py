"""Chat engine — retrieval-augmented generation with source attribution."""


from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
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
- Cite sources inline using [1], [2], etc. based on the source numbers in the context (e.g. "Revenue was $1.2M [1]").
- Every factual claim should have a numbered source reference.
- Be thorough — list ALL relevant items, don't truncate lists.
- Be concise. Use bullet points.
"""

_STREAM_SYSTEM_PROMPT = """You are Verra, a private AI assistant that helps the user understand their business data.

Rules:
- Base your answers ONLY on the provided context below.
- When context contains conflicting information, prefer the source with the most recent date. Flag the conflict.
- If the context doesn't contain enough information, say "I don't have that information in your data."
- Cite sources inline using [1], [2], etc. based on the source numbers in the context (e.g. "Revenue was $1.2M [1]").
- Every factual claim should have a numbered source reference.
- Be thorough — list ALL relevant items, don't truncate lists.
- Be concise. Use bullet points.
- When doing math, show your work step by step.
"""

_NO_ANSWER = (
    "I don't have enough information in your data to answer that question confidently.\n"
    "Try ingesting more documents or rephrasing your question."
)

_MIN_SCORE = -1.2  # ChromaDB L2 distance threshold — relaxed to avoid rejecting relevant results


# Confidence scoring


class ConfidenceLevel(str, Enum):
    HIGH = "high"      # top score > 0.7, 3+ diverse sources
    MEDIUM = "medium"  # top score > 0.3, 2+ sources
    LOW = "low"        # top score > 0, any sources
    NONE = "none"      # no relevant results


def compute_confidence(results: list[SearchResult]) -> ConfidenceLevel:
    """Derive a confidence level from retrieval results."""
    if not results:
        return ConfidenceLevel.NONE

    max_score = max(r.score for r in results)
    unique_files = len({
        r.metadata.get("file_name", "")
        for r in results
        if r.metadata.get("file_name")
    })

    if max_score > 0.7 and unique_files >= 3:
        return ConfidenceLevel.HIGH
    if max_score > 0.3 and unique_files >= 2:
        return ConfidenceLevel.MEDIUM
    if results:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.NONE


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
    confidence: ConfidenceLevel = ConfidenceLevel.NONE


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
        n_results: int = 10,
        conversation_id: int | None = None,
        entity_store: Any | None = None,       # EntityStore
        tabular_store: Any | None = None,      # TabularStore
    ) -> None:
        self.llm = llm
        self.metadata_store = metadata_store
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.entity_store = entity_store
        self.tabular_store = tabular_store
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

        If the query contains coreference signals and there is conversation
        history, the query is rewritten to be self-contained before retrieval.
        The original user_message is preserved for the LLM prompt.

        For multi-part questions (comparison/multi-topic signals), the query is
        decomposed into sub-queries that are each searched independently, and the
        results are merged and re-ranked.

        Includes an agentic relevance-grading step: if the first results do not
        cover the query terms well, the query is expanded/rewritten and a second
        search is issued.  Both result sets are merged and re-ranked by score.

        HyDE (Hypothetical Document Embedding) is enabled automatically when
        running semantic search — the LLM client is passed to search() so it can
        generate a hypothetical answer whose embedding is used for vector search.
        """
        retrieval_query = self._rewrite_query(user_message)

        # Query decomposition: for multi-part questions, search each sub-query
        # separately and merge results.
        sub_queries = self._decompose_query(retrieval_query)

        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()
        classified = None

        for sq in sub_queries:
            c = parse_query(sq)
            if classified is None:
                classified = c
            sub_results = search(
                c,
                self.metadata_store,
                self.vector_store,
                n_results=self.n_results + 4,
                entity_store=self.entity_store,
                llm_client=None,
            )
            for r in sub_results:
                if r.chunk_id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.chunk_id)

        # classified is always set (sub_queries has at least one element)
        assert classified is not None
        results = all_results

        # Sort merged results by score and trim
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[: self.n_results + 4]

        # Grade relevance and retry with a rewritten query if coverage is poor
        is_relevant, rewritten = self._grade_relevance(user_message, results)
        if not is_relevant and rewritten != user_message:
            classified2 = parse_query(rewritten)
            results2 = search(
                classified2,
                self.metadata_store,
                self.vector_store,
                n_results=self.n_results + 4,
                entity_store=self.entity_store,
                llm_client=None,
            )
            # Merge: deduplicate by chunk_id, prefer results2 for new chunks
            seen = {r.chunk_id for r in results}
            for r in results2:
                if r.chunk_id not in seen:
                    results.append(r)
                    seen.add(r.chunk_id)

            # Re-sort by score and trim to budget
            results.sort(key=lambda r: r.score, reverse=True)
            results = results[: self.n_results + 4]

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

        context_blocks = _format_context_with_full_docs(results)

        # Attempt to answer tabular questions via SQL before sending to LLM
        sql_result = self._try_sql_answer(user_message, results)
        if sql_result is not None:
            # Put the SQL answer PROMINENTLY at the top with a clear instruction
            sql_context = (
                "IMPORTANT: A SQL query was run against your structured data and returned "
                "the following result. Use this data to answer the question directly.\n\n"
                + sql_result
            )
            context_blocks = sql_context + "\n\n---\n\n" + context_blocks

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
            tabular_store=self.tabular_store,
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
        confidence = compute_confidence(all_results)

        self._persist(user_message, answer)
        self._update_history(user_message, answer)
        self._set_conversation_title(user_message)

        return ChatResponse(
            answer=answer,
            sources=sources,
            query_type=classified.query_type.value,
            had_context=bool(all_results),
            assessment=assessment,
            confidence=confidence,
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
                confidence=ConfidenceLevel.NONE,
            )

        context_blocks = _format_context_with_full_docs(results)

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
        confidence = compute_confidence(results)

        self._persist(user_message, answer)
        self._update_history(user_message, answer)
        self._set_conversation_title(user_message)

        return ChatResponse(
            answer=answer,
            sources=sources,
            query_type=classified.query_type.value,
            had_context=True,
            assessment=assessment,
            confidence=confidence,
        )


    # ------------------------------------------------------------------
    # Relevance grading
    # ------------------------------------------------------------------

    _GRADE_STOP: frozenset[str] = frozenset({
        "what", "is", "the", "our", "how", "many", "much", "who", "does",
        "do", "did", "are", "were", "was", "can", "will", "a", "an", "and",
        "or", "for", "in", "on", "at", "to", "of", "by", "from", "with",
        "about", "this", "that", "has", "have", "been", "not", "which",
        "best", "most", "top", "all", "any",
    })

    _GRADE_REWRITE_MAP: dict[str, str] = {
        "win": "won closed",
        "rate": "percentage ratio",
        "spend": "cost expense",
        "trend": "over time monthly",
        "pto": "vacation leave time off days",
        "p1": "priority critical urgent",
    }

    def _grade_relevance(
        self, query: str, results: list[SearchResult]
    ) -> tuple[bool, str]:
        """Quick check: do the retrieved chunks actually answer the query?

        Returns (is_relevant, rewritten_query_if_not).
        Uses a lightweight keyword-coverage heuristic.  If coverage is low,
        expands query terms using a synonym map and returns the expansion as
        the rewritten query so retrieve() can make a second pass.
        """
        if not results:
            return False, query

        # Extract meaningful terms from the query
        query_terms = [
            w.lower()
            for w in re.findall(r"\w+", query)
            if len(w) >= 3 and w.lower() not in self._GRADE_STOP
        ]

        if not query_terms:
            return True, query  # Nothing to grade — assume relevant

        # Check how many query terms appear across the top results
        combined_text = " ".join(r.text.lower() for r in results[:5])
        matches = sum(1 for t in query_terms if t in combined_text)
        coverage = matches / len(query_terms)

        if coverage >= 0.5:
            return True, query  # Good enough

        # Low coverage — expand terms with synonyms for a second-pass search
        expanded: list[str] = []
        for t in query_terms:
            expanded.append(t)
            if t in self._GRADE_REWRITE_MAP:
                expanded.extend(self._GRADE_REWRITE_MAP[t].split())

        rewritten = " ".join(expanded)
        return False, rewritten

    _COREF_SIGNALS: frozenset[str] = frozenset({
        'it', 'its', 'they', 'them', 'their', 'this', 'that',
        'those', 'these', 'he', 'she', 'his', 'her', 'the same',
        'above', 'mentioned', 'previous',
    })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _DECOMPOSE_SIGNALS: frozenset[str] = frozenset({
        "compare", "vs", "versus", "difference between",
        "and also", "additionally", "as well as",
        "both", "each", "respectively",
    })

    def _decompose_query(self, user_message: str) -> list[str]:
        """Decompose a multi-part question into sub-queries using heuristics.

        No LLM call — splits on comparison patterns and conjunctions.
        Returns a list of 1-3 queries.
        """
        lower = user_message.lower()
        if not any(s in lower for s in self._DECOMPOSE_SIGNALS):
            return [user_message]

        # Split on "compare X and Y" / "X vs Y" / "difference between X and Y"
        import re
        # "Compare Q3 and Q4 2025 financial performance"
        m = re.search(r"compare\s+(.+?)\s+and\s+(.+?)(?:\s+(?:performance|data|results|numbers|stats))?$",
                       user_message, re.IGNORECASE)
        if m:
            return [f"What is the {m.group(1).strip()} performance?",
                    f"What is the {m.group(2).strip()} performance?"]

        # "X vs Y"
        m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", user_message, re.IGNORECASE)
        if m:
            return [m.group(1).strip() + "?", m.group(2).strip() + "?"]

        # "difference between X and Y"
        m = re.search(r"difference\s+between\s+(.+?)\s+and\s+(.+)", user_message, re.IGNORECASE)
        if m:
            return [f"What is {m.group(1).strip()}?", f"What is {m.group(2).strip()}?"]

        return [user_message]

    def _rewrite_query(self, user_message: str) -> str:
        """Rewrite a follow-up question using conversation history (no LLM call).

        Replaces pronouns like 'it', 'they', 'them' with the most recent
        entity/topic from the conversation. Fast heuristic — no LLM needed.
        """
        if not self._history:
            return user_message

        words = set(user_message.lower().split())
        if not words & self._COREF_SIGNALS:
            return user_message

        # Find the most recent topic from the last user message
        last_user_msgs = [m["content"] for m in self._history if m["role"] == "user"]
        if not last_user_msgs:
            return user_message

        # Extract likely topic: longest capitalized phrase or noun from last question
        import re
        last_q = last_user_msgs[-1]
        # Look for capitalized phrases (proper nouns / entity names)
        names = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', last_q)
        topic = max(names, key=len) if names else ""

        if not topic:
            # Fall back to significant words from last question
            _stop = {"what", "is", "the", "our", "how", "who", "does", "do",
                     "are", "was", "can", "will", "and", "or", "for", "about"}
            sig_words = [w for w in last_q.split() if w.lower() not in _stop and len(w) > 3]
            topic = " ".join(sig_words[:3]) if sig_words else ""

        if not topic:
            return user_message

        # Replace pronouns with the topic
        rewritten = user_message
        for pronoun in ['it', 'its', 'they', 'them', 'their', 'this', 'that']:
            rewritten = re.sub(
                rf'\b{pronoun}\b',
                topic,
                rewritten,
                flags=re.IGNORECASE,
                count=1,
            )

        return rewritten

    # Keywords that suggest a tabular / analytical question
    _TABULAR_SIGNALS: frozenset[str] = frozenset({
        "which", "best", "worst", "most", "least", "top", "bottom",
        "count", "total", "average", "avg", "sum", "rank", "ranking",
        "compare", "comparison", "highest", "lowest", "max", "min",
        "rate", "ratio", "percent", "percentage", "how many",
    })

    def _try_sql_answer(
        self, user_message: str, results: list[SearchResult]
    ) -> str | None:
        """If the question looks tabular and we have relevant CSV tables, run SQL.

        Asks the LLM to generate a SELECT query, executes it, and returns a
        formatted block with the SQL and its results.  Returns None on any
        failure so the caller can proceed without SQL context.
        """
        if self.tabular_store is None:
            return None

        tables = self.tabular_store.list_tables()
        if not tables:
            return None

        # Only trigger when the retrieved sources include at least one CSV
        csv_sources = [r for r in results if r.metadata.get("format") == "csv"]
        if not csv_sources:
            return None

        # Only trigger if the question contains analytical signal words
        lower_msg = user_message.lower()
        if not any(signal in lower_msg for signal in self._TABULAR_SIGNALS):
            return None

        # Only include tables whose names are relevant to the query
        # (sending 31 table schemas confuses the model and wastes tokens)
        import re as _re
        query_words = set(_re.findall(r'\w+', user_message.lower()))
        relevant_tables = []
        for t in tables:
            tn = t["table_name"].lower()
            # Include table if any query word appears in the table name
            if any(w in tn for w in query_words if len(w) >= 3):
                relevant_tables.append(t)
        # If nothing matched, include all tables with < 10 columns (skip giant ones)
        if not relevant_tables:
            relevant_tables = [t for t in tables if len(t["columns"]) <= 10][:8]

        # Build schema with samples and pre-computed summaries
        table_schemas: list[str] = []
        for t in relevant_tables:
            cols = ", ".join(
                f"{c['name']} ({c['type']})" for c in t["columns"]
            )
            schema_line = f"Table '{t['table_name']}': {cols}"
            # Include sample rows so the LLM knows actual column values
            samples = self.tabular_store.get_sample_rows(t["table_name"], limit=2)
            if samples:
                sample_strs = []
                for s in samples:
                    vals = ", ".join(f"{k}={v!r}" for k, v in list(s.items())[:6])
                    sample_strs.append(f"  Example: {vals}")
                schema_line += "\n" + "\n".join(sample_strs)
            # Include pre-computed summaries for value distributions
            summaries = self.tabular_store.get_summaries(t["table_name"])
            for key, val in summaries.items():
                if key.startswith("value_counts:"):
                    col = key.split(":", 1)[1]
                    schema_line += f"\n  Distinct values for {col}: {val}"
            table_schemas.append(schema_line)

        sql_prompt: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Generate a single SQL SELECT query to answer the user's question. "
                    "Output ONLY the SQL query, nothing else. No markdown, no explanation. "
                    "Available tables:\n" + "\n".join(table_schemas)
                ),
            },
            {"role": "user", "content": user_message},
        ]

        try:
            sql = self.llm.complete(sql_prompt).strip()
            # Strip markdown code fences if present
            sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
            sql = sql.replace("```", "").strip()
            if not sql.upper().startswith("SELECT"):
                return None
            query_results = self.tabular_store.query(sql)
            if not query_results:
                return None
            headers = list(query_results[0].keys())
            lines = [" | ".join(headers)]
            lines.append("-" * len(lines[0]))
            for row in query_results[:20]:
                lines.append(
                    " | ".join(
                        str(row[h]) if row[h] is not None else "" for h in headers
                    )
                )
            result_table = "\n".join(lines)
            return (
                f"[SQL Query executed against tabular data]\n"
                f"Query: {sql}\n\n"
                f"Results:\n{result_table}"
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("SQL query failed: %s", exc)
            return None

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


_FULL_DOC_SIZE_LIMIT = 8_000  # bytes — pull the entire file when smaller than this


def _format_context_with_full_docs(results: list[SearchResult]) -> str:
    """Format context, pulling full documents for small files.

    For each result, if the source file exists and is below the size limit we
    read the entire file and substitute it for the chunk text.  When two chunks
    originate from the same file we include the full-doc text only once,
    skipping the duplicate chunk.  This gives the LLM complete context for
    small documents (org charts, policy files, config files, etc.) rather than
    just an 800-character fragment.
    """
    from pathlib import Path

    parts: list[str] = []
    seen_files: set[str] = set()
    source_index = 1  # we track our own counter because we may skip entries

    for r in results:
        source_label = _source_label(r.metadata)
        authority_note = (
            f" [authority: {r.authority_weight}]" if r.authority_weight != 50 else ""
        )

        file_path = r.metadata.get("file_path", "")
        file_name = r.metadata.get("file_name", "") or file_path

        text = r.text
        using_full_doc = False

        if file_path:
            if file_name in seen_files:
                # Full document already included — skip this duplicate chunk
                continue
            try:
                p = Path(file_path)
                if p.exists() and p.stat().st_size < _FULL_DOC_SIZE_LIMIT:
                    text = p.read_text(errors="replace")
                    seen_files.add(file_name)
                    using_full_doc = True
            except Exception:
                pass  # Fall back to chunk text on any I/O error

        # When not using a full-doc fetch, prefer the stored parent context
        # (surrounding section) over the raw chunk fragment.  Parent text is
        # set at ingest time by _add_parent_context() and provides ~2 KB of
        # surrounding paragraphs without inflating the context to the whole file.
        if not using_full_doc:
            parent = r.metadata.get("parent_text", "")
            if parent and len(parent) > len(text):
                text = parent

        parts.append(f"[Source {source_index}: {source_label}{authority_note}]\n{text}")
        source_index += 1

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
