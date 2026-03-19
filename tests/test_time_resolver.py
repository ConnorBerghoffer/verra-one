"""Tests for verra.ingest.time_resolver."""

from __future__ import annotations

from datetime import date

import pytest

from verra.ingest.time_resolver import extract_document_date, resolve_time_references


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fixed reference date for all tests so assertions don't drift
REF = date(2024, 3, 15)  # Friday, Q1 2024


# ---------------------------------------------------------------------------
# resolve_time_references
# ---------------------------------------------------------------------------


def test_no_changes_when_no_references() -> None:
    text = "The project is on schedule."
    assert resolve_time_references(text, REF) == text


def test_yesterday_resolves() -> None:
    result = resolve_time_references("We met yesterday.", REF)
    # 2024-03-15 - 1 = 2024-03-14
    assert "2024-03-14" in result


def test_today_resolves() -> None:
    result = resolve_time_references("Call today at 3pm.", REF)
    assert "2024-03-15" in result


def test_tomorrow_resolves() -> None:
    result = resolve_time_references("Ship tomorrow.", REF)
    assert "2024-03-16" in result


def test_last_week_resolves() -> None:
    # REF is 2024-03-15 (Friday, week Mon 11 – Sun 17)
    # Last week: Mon 2024-03-04 – Sun 2024-03-10
    result = resolve_time_references("As agreed last week.", REF)
    assert "2024-03-04" in result
    assert "2024-03-10" in result


def test_last_month_resolves() -> None:
    # REF is 2024-03-15 → last month = Feb 2024
    result = resolve_time_references("Revenue last month was strong.", REF)
    assert "2024-02-01" in result
    assert "2024-02-29" in result  # 2024 is a leap year


def test_last_quarter_resolves() -> None:
    # REF is 2024-03-15 (Q1) → last quarter = Q4 2023
    result = resolve_time_references("We exceeded targets last quarter.", REF)
    assert "Q4 2023" in result
    assert "2023-10-01" in result
    assert "2023-12-31" in result


def test_last_year_resolves() -> None:
    result = resolve_time_references("Last year was record-breaking.", REF)
    assert "2023" in result
    assert "2023-01-01" in result
    assert "2023-12-31" in result


def test_this_quarter_resolves() -> None:
    # REF is Q1 2024
    result = resolve_time_references("This quarter is on track.", REF)
    assert "Q1 2024" in result
    assert "2024-01-01" in result
    assert "2024-03-31" in result


def test_next_friday_resolves() -> None:
    # REF is 2024-03-15 (Friday) → next Friday = 2024-03-22
    result = resolve_time_references("Let's finalise next Friday.", REF)
    assert "2024-03-22" in result


def test_two_weeks_ago_resolves() -> None:
    # 2024-03-15 - 14 days = 2024-03-01
    result = resolve_time_references("The decision was made 2 weeks ago.", REF)
    assert "2024-03-01" in result


def test_end_of_month_resolves() -> None:
    # March 2024 ends on 31st
    result = resolve_time_references("Deliver by end of month.", REF)
    assert "2024-03-31" in result


def test_end_of_quarter_resolves() -> None:
    # Q1 ends 31 March 2024
    result = resolve_time_references("Target by end of quarter.", REF)
    assert "2024-03-31" in result


def test_end_of_year_resolves() -> None:
    result = resolve_time_references("Due by end of year.", REF)
    assert "2024-12-31" in result


def test_uses_document_date_as_reference() -> None:
    doc_date = date(2023, 7, 1)  # Q3 2023
    result = resolve_time_references("Revenue last quarter was strong.", doc_date)
    # Last quarter relative to 2023-07-01 = Q2 2023
    assert "Q2 2023" in result
    assert "2023-04-01" in result


def test_multiple_references_in_one_text() -> None:
    text = "Yesterday we reviewed last month's numbers and plan to ship next Friday."
    result = resolve_time_references(text, REF)
    # yesterday = 2024-03-14
    assert "2024-03-14" in result
    # last month = Feb 2024
    assert "2024-02-01" in result
    # next Friday = 2024-03-22
    assert "2024-03-22" in result


def test_original_phrase_preserved() -> None:
    """The original phrase must still appear alongside the resolved date."""
    result = resolve_time_references("See you tomorrow.", REF)
    assert "tomorrow" in result
    assert "2024-03-16" in result


def test_n_days_ago_resolves() -> None:
    result = resolve_time_references("Signed 5 days ago.", REF)
    # 2024-03-10
    assert "2024-03-10" in result


def test_n_months_ago_resolves() -> None:
    result = resolve_time_references("Contract signed 3 months ago.", REF)
    # 2024-03-15 minus 3 months = 2023-12-15
    assert "2023-12-15" in result


# ---------------------------------------------------------------------------
# extract_document_date
# ---------------------------------------------------------------------------


def test_extract_document_date_from_content() -> None:
    text = "Date: 2024-01-20\n\nThis is the meeting notes."
    result = extract_document_date(text, "meeting.md")
    assert result == date(2024, 1, 20)


def test_extract_document_date_from_filename() -> None:
    result = extract_document_date("Some content here.", "2024-03-15-standup.md")
    assert result == date(2024, 3, 15)


def test_extract_document_date_from_compact_filename() -> None:
    result = extract_document_date("Content.", "20240315-notes.txt")
    assert result == date(2024, 3, 15)


def test_extract_document_date_returns_none_for_no_date() -> None:
    result = extract_document_date("No dates here at all.", "untitled.md")
    assert result is None


def test_extract_document_date_last_updated_pattern() -> None:
    text = "Last updated: January 5, 2024\n\nSome content."
    result = extract_document_date(text, "doc.md")
    assert result == date(2024, 1, 5)


def test_extract_document_date_prefers_content_over_filename() -> None:
    text = "Date: 2023-06-01\n\nContent."
    # Filename suggests a different date — content should win
    result = extract_document_date(text, "2022-01-01-old.md")
    assert result == date(2023, 6, 1)
