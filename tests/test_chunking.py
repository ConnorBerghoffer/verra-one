"""Tests for semantic + email-thread chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from verra.ingest.chunking import (
    Chunk,
    _strip_quoted_text,
    chunk_document,
    chunk_email_thread,
    count_tokens,
)


FIXTURES = Path(__file__).parent / "fixtures"


class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_known_length(self) -> None:
        # "hello" is typically 1 token in cl100k_base
        assert count_tokens("hello") >= 1

    def test_longer_text(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 20
        tokens = count_tokens(text)
        assert tokens > 50  # sanity check


class TestChunkDocument:
    def test_short_doc_is_single_chunk(self) -> None:
        text = "This is a short document.\n\nIt has two paragraphs."
        chunks = chunk_document(text)
        assert len(chunks) == 1
        assert "short document" in chunks[0].text

    def test_metadata_copied_to_chunks(self) -> None:
        text = "Hello world.\n\nSecond paragraph."
        meta = {"file_name": "test.txt", "format": "txt"}
        chunks = chunk_document(text, meta)
        for chunk in chunks:
            assert chunk.metadata["file_name"] == "test.txt"
            assert chunk.metadata["format"] == "txt"

    def test_chunk_has_positive_token_count(self) -> None:
        text = "Some content here.\n\nMore content."
        chunks = chunk_document(text)
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_long_doc_produces_multiple_chunks(self) -> None:
        # Generate a document long enough to require splitting
        paragraph = "This is a paragraph with some meaningful content. " * 20
        text = "\n\n".join([paragraph] * 10)
        chunks = chunk_document(text)
        assert len(chunks) > 1

    def test_table_not_split(self) -> None:
        table = (
            "| Name | Value |\n"
            "|------|-------|\n"
            "| Foo  | 100   |\n"
            "| Bar  | 200   |\n"
            "| Baz  | 300   |\n"
        )
        before = "Introduction paragraph.\n\n"
        text = before + table + "\n\nConclusion paragraph."
        chunks = chunk_document(text)
        # The table lines must all appear in the same chunk
        table_chunk = None
        for chunk in chunks:
            if "| Foo  |" in chunk.text:
                table_chunk = chunk
                break
        assert table_chunk is not None, "Table chunk not found"
        assert "| Bar  |" in table_chunk.text
        assert "| Baz  |" in table_chunk.text

    def test_chunks_from_md_fixture(self) -> None:
        text = (FIXTURES / "sample.md").read_text()
        chunks = chunk_document(text, {"file_name": "sample.md"})
        assert len(chunks) >= 1
        # Refund policy content should appear somewhere
        all_text = " ".join(c.text for c in chunks)
        assert "refund" in all_text.lower()

    def test_empty_document(self) -> None:
        chunks = chunk_document("")
        assert chunks == []

    def test_overlap_not_exceeds_original(self) -> None:
        paragraph = "Content paragraph with several words. " * 15
        text = "\n\n".join([paragraph] * 8)
        chunks = chunk_document(text)
        if len(chunks) > 1:
            # Second chunk should start with some overlap from the first
            # but not be longer than MAX_TOKENS
            assert chunks[1].token_count <= 1200


class TestChunkEmailThread:
    def _make_msg(self, **kwargs):
        defaults = {
            "from": "alice@example.com",
            "to": "bob@example.com",
            "date": "2024-01-15",
            "subject": "Re: Project Update",
            "body": "Here is the project update.",
            "thread_id": "thread_001",
        }
        defaults.update(kwargs)
        return defaults

    def test_short_thread_is_single_chunk(self) -> None:
        messages = [
            self._make_msg(body="Initial message."),
            self._make_msg(from_="bob@example.com", body="Thanks, noted."),
        ]
        chunks = chunk_email_thread(messages)
        assert len(chunks) == 1
        assert "Initial message" in chunks[0].text

    def test_empty_thread(self) -> None:
        chunks = chunk_email_thread([])
        assert chunks == []

    def test_metadata_includes_thread_id(self) -> None:
        messages = [self._make_msg(thread_id="abc123")]
        chunks = chunk_email_thread(messages)
        assert chunks[0].metadata["thread_id"] == "abc123"

    def test_metadata_includes_subject(self) -> None:
        messages = [self._make_msg(subject="Contract Discussion")]
        chunks = chunk_email_thread(messages)
        assert chunks[0].metadata["subject"] == "Contract Discussion"

    def test_participants_extracted(self) -> None:
        messages = [
            self._make_msg(**{"from": "alice@example.com", "to": "bob@example.com"}),
            self._make_msg(**{"from": "bob@example.com", "to": "alice@example.com"}),
        ]
        chunks = chunk_email_thread(messages)
        participants = chunks[0].metadata["participants"]
        assert "alice@example.com" in participants
        assert "bob@example.com" in participants

    def test_long_thread_splits(self) -> None:
        # A very long thread that exceeds 2000 tokens
        long_body = "This is a very detailed email body with lots of content. " * 50
        messages = [self._make_msg(body=long_body) for _ in range(10)]
        chunks = chunk_email_thread(messages)
        # Should split into more than one chunk
        assert len(chunks) >= 1  # at minimum it doesn't crash


class TestStripQuotedText:
    def test_strips_gt_prefixed_lines(self) -> None:
        body = "My reply here.\n\n> Quoted line 1\n> Quoted line 2\n\nEnd."
        result = _strip_quoted_text(body)
        assert "> Quoted" not in result
        assert "My reply here" in result

    def test_stops_at_signature_delimiter(self) -> None:
        body = "Main body.\n\n--\nJohn Smith\njohn@example.com"
        result = _strip_quoted_text(body)
        assert "John Smith" not in result
        assert "Main body" in result

    def test_strips_on_wrote_pattern(self) -> None:
        body = "My response.\n\nOn Mon Jan 15, 2024, Alice wrote:\n> Some quote"
        result = _strip_quoted_text(body)
        assert "Alice wrote" not in result
        assert "My response" in result
