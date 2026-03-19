"""Semantic + structure-aware chunking.

Design rules:
  - Target 500-1000 tokens per chunk; hard cap at 1200.
  - 10 % overlap between adjacent chunks.
  - Never split a table (detected as a block of lines sharing a '|' prefix).
  - Never split mid-paragraph (split at blank lines / heading boundaries).
  - Email threads: keep entire thread in one chunk when < 2000 tokens;
    otherwise split at message boundaries.
  - Every chunk carries a copy of the source metadata.
"""


from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

# ---------------------------------------------------------------------------
# Token counting (cached encoder)
# ---------------------------------------------------------------------------

_enc: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        # cl100k_base is the tokeniser for GPT-4 / Claude-compatible counts
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = count_tokens(self.text)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TARGET_TOKENS = 750
_MAX_TOKENS = 1200
_OVERLAP_RATIO = 0.10

# A "table block" is 3+ consecutive lines that all contain at least one '|'
_TABLE_RE = re.compile(r"^.*\|.*$", re.MULTILINE)


def _is_table_line(line: str) -> bool:
    return "|" in line


def _heading_level(line: str) -> int:
    """Return 1-6 if the line is a Markdown heading, else 0."""
    m = re.match(r"^(#{1,6})\s", line)
    return len(m.group(1)) if m else 0


def _split_into_blocks(text: str) -> list[str]:
    """Split text into semantic blocks: tables, headings, paragraphs.

    A block is either:
    - A contiguous run of table lines (preserves table integrity).
    - A Markdown heading line (always starts a new block).
    - A paragraph separated from the next by one or more blank lines.
    """
    lines = text.splitlines()
    blocks: list[str] = []
    current_lines: list[str] = []
    in_table = False

    for line in lines:
        table_line = _is_table_line(line)
        heading = _heading_level(line)

        if heading and not in_table:
            # Flush current paragraph, start heading as its own block
            if current_lines:
                blocks.append("\n".join(current_lines))
                current_lines = []
            blocks.append(line)
        elif table_line:
            if not in_table and current_lines:
                # Flush paragraph before table
                blocks.append("\n".join(current_lines))
                current_lines = []
            in_table = True
            current_lines.append(line)
        else:
            if in_table:
                # End of table block
                blocks.append("\n".join(current_lines))
                current_lines = []
                in_table = False
            if line.strip() == "":
                # Blank line → paragraph boundary
                if current_lines:
                    blocks.append("\n".join(current_lines))
                    current_lines = []
            else:
                current_lines.append(line)

    if current_lines:
        blocks.append("\n".join(current_lines))

    return [b for b in blocks if b.strip()]


def _make_overlap(chunks: list[Chunk]) -> list[Chunk]:
    """Prepend the tail of the previous chunk to the current chunk."""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        curr = chunks[i]
        overlap_tokens = int(prev.token_count * _OVERLAP_RATIO)
        if overlap_tokens > 0:
            # Grab last N tokens worth of text from previous chunk
            enc = _get_encoder()
            prev_ids = enc.encode(prev.text)
            tail_ids = prev_ids[-overlap_tokens:]
            tail_text = enc.decode(tail_ids)
            new_text = tail_text.strip() + "\n\n" + curr.text
            result.append(Chunk(text=new_text, metadata=curr.metadata))
        else:
            result.append(curr)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _apply_code_awareness(chunks: list[Chunk]) -> list[Chunk]:
    """Post-process chunks: detect code blocks and adjust embedding text.

    For each chunk that contains code blocks:
    - Replace the raw code with a natural-language summary in the embedding text
      (stored as ``Chunk.text`` which ChromaDB embeds).
    - Preserve the original text in ``metadata["original_text"]``.
    - Set ``metadata["has_code"] = True``.
    """
    # Import here to avoid circular imports at module level
    from verra.ingest.extractors import extract_code_blocks, summarize_code_block

    result: list[Chunk] = []
    for chunk in chunks:
        prose, code_blocks = extract_code_blocks(chunk.text)
        if not code_blocks:
            result.append(chunk)
            continue

        # Build summarised text: prose + summaries of each code block
        summaries = [
            summarize_code_block(cb["code"], cb.get("language", ""))
            for cb in code_blocks
        ]
        embedding_text = prose
        if summaries:
            embedding_text = (embedding_text + "\n\n" + "\n".join(summaries)).strip()

        new_meta = dict(chunk.metadata)
        new_meta["has_code"] = True
        new_meta["original_text"] = chunk.text
        result.append(Chunk(text=embedding_text, metadata=new_meta))

    return result


def _add_parent_context(
    chunks: list[Chunk],
    full_text: str,
    parent_window: int = 2000,
) -> None:
    """Add parent context to each chunk's metadata.

    For each chunk, locates it within the full document text and extracts a
    larger surrounding window. This parent text is stored in
    ``chunk.metadata["parent_text"]`` so the LLM receives richer context while
    the smaller chunk text is still used for embedding / semantic search.

    The window expands outward from the chunk position: one quarter of
    ``parent_window`` is taken from before the chunk and the remainder (plus
    the chunk itself) from after, then both ends are snapped to the nearest
    sentence boundary. The result is capped at ``parent_window`` characters.

    Falls back to the chunk text itself when the chunk cannot be located in
    the full document (e.g. after overlap stitching or code summarisation).
    """
    for chunk in chunks:
        # Use the first 100 chars of the chunk text as a locator. For code-
        # summarised chunks we fall back to the original_text if present so
        # we still anchor to actual document content.
        anchor = chunk.metadata.get("original_text", chunk.text)
        anchor_prefix = anchor[:100]
        pos = full_text.find(anchor_prefix)
        if pos == -1:
            chunk.metadata["parent_text"] = chunk.text
            continue

        # Determine the span of the chunk within the full text
        chunk_len = len(anchor)

        # Carve out a window: 25 % before, rest after + chunk length
        before = parent_window // 4
        after = parent_window - before

        start = max(0, pos - before)
        end = min(len(full_text), pos + chunk_len + after)

        # Snap start backward to the nearest sentence / paragraph boundary
        while start > 0 and full_text[start] not in ".!?\n":
            start -= 1

        # Snap end forward to the nearest sentence / paragraph boundary
        while end < len(full_text) and full_text[end] not in ".!?\n":
            end += 1

        parent = full_text[start:end].strip()

        # Hard cap
        if len(parent) > parent_window:
            parent = parent[:parent_window]

        chunk.metadata["parent_text"] = parent


def chunk_document(
    text: str,
    metadata: dict[str, Any] | None = None,
    attachment_context: str | None = None,
) -> list[Chunk]:
    """Split a document into semantically coherent chunks.

    Parameters
    ----------
    text:
        Raw extracted text of the document.
    metadata:
        Source metadata (file path, format, etc.) copied into every chunk.
    attachment_context:
        Optional excerpt from the parent email body that referenced this
        attachment. When provided, it is prepended to the first chunk so
        that retrieval can use the referring context.  The metadata of
        every chunk will also carry an ``attachment_context_included``
        flag on the first chunk.

    Returns
    -------
    List of Chunk objects.
    """
    meta = metadata or {}
    blocks = _split_into_blocks(text)

    chunks: list[Chunk] = []
    current_blocks: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block)

        # Single block exceeds the hard cap — emit it alone, possibly split further
        if block_tokens > _MAX_TOKENS:
            # Flush accumulated
            if current_blocks:
                chunks.append(Chunk(text="\n\n".join(current_blocks), metadata=dict(meta)))
                current_blocks = []
                current_tokens = 0
            # Split the oversized block by sentences (best-effort)
            sentences = re.split(r"(?<=[.!?])\s+", block)
            sent_buf: list[str] = []
            sent_tokens = 0
            for sent in sentences:
                st = count_tokens(sent)
                if sent_tokens + st > _MAX_TOKENS and sent_buf:
                    chunks.append(Chunk(text=" ".join(sent_buf), metadata=dict(meta)))
                    sent_buf = []
                    sent_tokens = 0
                sent_buf.append(sent)
                sent_tokens += st
            if sent_buf:
                chunks.append(Chunk(text=" ".join(sent_buf), metadata=dict(meta)))
            continue

        # Would adding this block exceed the target?
        if current_tokens + block_tokens > _TARGET_TOKENS and current_blocks:
            chunks.append(Chunk(text="\n\n".join(current_blocks), metadata=dict(meta)))
            current_blocks = []
            current_tokens = 0

        current_blocks.append(block)
        current_tokens += block_tokens

    # Flush remainder
    if current_blocks:
        chunks.append(Chunk(text="\n\n".join(current_blocks), metadata=dict(meta)))

    chunks_with_overlap = _make_overlap(chunks)
    final_chunks = _apply_code_awareness(chunks_with_overlap)

    # Add parent context to every chunk so the LLM sees surrounding text.
    # Called before the attachment prefix so anchors still match the raw doc.
    _add_parent_context(final_chunks, text)

    # Prepend attachment context to the first chunk so retrieval can use
    # the referring email's content when ranking attachment chunks.
    if attachment_context and final_chunks:
        first = final_chunks[0]
        context_prefix = (
            f"[Attachment context from parent email]\n{attachment_context.strip()}\n\n"
            f"[Attachment content]\n"
        )
        new_text = context_prefix + first.text
        new_meta = dict(first.metadata)
        new_meta["attachment_context_included"] = True
        final_chunks[0] = Chunk(text=new_text, metadata=new_meta)

    return final_chunks


def chunk_email_thread(messages: list[dict[str, Any]]) -> list[Chunk]:
    """Pack an email thread into one or more chunks.

    Each message dict should have keys:
      from, to, date, subject, body

    If the entire thread fits within 2000 tokens, it becomes one chunk.
    Otherwise it's split at message boundaries.

    Bodies are cleaned (signatures, disclaimers, quoted replies stripped)
    before chunking via email_cleaner.clean_email_body().
    """
    if not messages:
        return []

    # Import here to avoid circular imports at module level
    try:
        from verra.ingest.email_cleaner import clean_email_body as _clean
    except ImportError:
        def _clean(t: str) -> str:  # type: ignore[misc]
            return t

    def format_message(msg: dict[str, Any], index: int) -> str:
        header = (
            f"--- Message {index + 1} ---\n"
            f"From: {msg.get('from', '')}\n"
            f"To: {msg.get('to', '')}\n"
            f"Date: {msg.get('date', '')}\n"
            f"Subject: {msg.get('subject', '')}\n"
        )
        # Apply full cleaning then legacy quoted-text stripping as belt-and-braces
        raw_body = msg.get("body", "")
        body = _clean(raw_body)
        body = _strip_quoted_text(body)
        return f"{header}\n{body.strip()}"

    formatted = [format_message(m, i) for i, m in enumerate(messages)]

    # Build common metadata from the first message
    first = messages[0]
    thread_meta: dict[str, Any] = {
        "source_type": "email",
        "subject": first.get("subject", ""),
        "thread_id": first.get("thread_id", ""),
        "participants": _extract_participants(messages),
    }

    # Try to fit the whole thread in one chunk
    full_text = "\n\n".join(formatted)
    if count_tokens(full_text) <= 2000:
        return [Chunk(text=full_text, metadata=thread_meta)]

    # Split at message boundaries
    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    for part in formatted:
        part_tokens = count_tokens(part)
        if current_tokens + part_tokens > 2000 and current_parts:
            chunks.append(Chunk(text="\n\n".join(current_parts), metadata=dict(thread_meta)))
            current_parts = []
            current_tokens = 0
        current_parts.append(part)
        current_tokens += part_tokens

    if current_parts:
        chunks.append(Chunk(text="\n\n".join(current_parts), metadata=dict(thread_meta)))

    return chunks


def _strip_quoted_text(body: str) -> str:
    """Remove lines that are email quotes (start with '>') and common signature patterns."""
    lines = body.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip quoted lines
        if stripped.startswith(">"):
            continue
        # Stop at common signature delimiters
        if stripped in ("--", "---", "________________________________"):
            break
        # Skip "On [date], [person] wrote:" patterns
        if re.match(r"^On .+ wrote:$", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _extract_participants(messages: list[dict[str, Any]]) -> list[str]:
    """Collect unique participants across all messages in a thread."""
    seen: set[str] = set()
    for msg in messages:
        for field in ("from", "to", "cc"):
            val = msg.get(field, "")
            if val:
                seen.add(val.strip())
    return sorted(seen)
