"""Tests for query router and retrieval logic."""

from __future__ import annotations

import pytest

from verra.retrieval.router import QueryType, classify_query, parse_query


class TestClassifyQuery:
    def test_semantic_for_policy_query(self) -> None:
        assert classify_query("what is our refund policy") == QueryType.SEMANTIC

    def test_semantic_for_general_question(self) -> None:
        assert classify_query("how do we handle customer complaints") == QueryType.SEMANTIC

    def test_metadata_for_strong_filter(self) -> None:
        result = classify_query("emails from Jake last month")
        # Should be METADATA or HYBRID — both are valid given strong metadata signals
        assert result in (QueryType.METADATA, QueryType.HYBRID)

    def test_hybrid_for_person_plus_topic(self) -> None:
        result = classify_query("what did Jake say about pricing")
        assert result == QueryType.HYBRID

    def test_hybrid_discussion_with(self) -> None:
        result = classify_query("discussion about contracts with Alice")
        assert result in (QueryType.HYBRID, QueryType.SEMANTIC)

    def test_case_insensitive(self) -> None:
        # Should handle mixed case
        result1 = classify_query("Emails FROM Jake")
        result2 = classify_query("emails from jake")
        assert result1 == result2


class TestParseQuery:
    def test_extracts_from_address(self) -> None:
        parsed = parse_query("emails from jake last month")
        assert parsed.from_address == "jake"

    def test_semantic_text_preserved(self) -> None:
        parsed = parse_query("what is our pricing model")
        assert "pricing" in parsed.semantic_text

    def test_source_type_email_detected(self) -> None:
        parsed = parse_query("email from alice about the invoice")
        assert parsed.source_type == "email"

    def test_source_type_folder_detected(self) -> None:
        parsed = parse_query("find the document about the refund policy")
        assert parsed.source_type == "folder"

    def test_no_source_type_when_ambiguous(self) -> None:
        parsed = parse_query("what is our pricing")
        # No strong signal for either email or folder
        assert parsed.source_type is None

    def test_query_type_propagated(self) -> None:
        parsed = parse_query("what did alice say about the contract")
        assert parsed.query_type == QueryType.HYBRID
