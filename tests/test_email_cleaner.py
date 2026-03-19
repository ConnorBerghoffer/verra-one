"""Tests for verra.ingest.email_cleaner."""

from __future__ import annotations

import pytest

from verra.ingest.email_cleaner import (
    clean_email_body,
    normalize_whitespace,
    strip_disclaimers,
    strip_quoted_replies,
    strip_signature,
)


# ---------------------------------------------------------------------------
# strip_signature
# ---------------------------------------------------------------------------


def test_strips_sent_from_iphone() -> None:
    text = "Hi, see you tomorrow.\n\nSent from my iPhone"
    result = strip_signature(text)
    assert "Sent from my iPhone" not in result
    assert "see you tomorrow" in result


def test_strips_sent_from_android() -> None:
    text = "Thanks for the update.\n\nSent from my Android"
    result = strip_signature(text)
    assert "Sent from my Android" not in result
    assert "Thanks for the update" in result


def test_strips_get_outlook_for_ios() -> None:
    text = "Let me know.\n\nGet Outlook for iOS"
    result = strip_signature(text)
    assert "Get Outlook for iOS" not in result
    assert "Let me know" in result


def test_strips_signature_delimiter() -> None:
    text = "Great meeting today.\n\n--\nJane Smith\nCEO, Acme Corp\n+1-555-0100"
    result = strip_signature(text)
    assert "Jane Smith" not in result
    assert "Great meeting today" in result


def test_strips_triple_dash_delimiter() -> None:
    text = "Please review the attached.\n\n---\nBob Jones | Senior VP"
    result = strip_signature(text)
    assert "Bob Jones" not in result
    assert "Please review" in result


def test_strips_underline_delimiter() -> None:
    text = "Confirmed for Thursday.\n\n____________________________\nAlice"
    result = strip_signature(text)
    assert "Alice" not in result
    assert "Confirmed for Thursday" in result


def test_preserves_content_before_signature() -> None:
    text = (
        "Hello,\n\n"
        "I wanted to follow up on the proposal.\n"
        "Please let me know your thoughts.\n\n"
        "--\n"
        "Regards,\n"
        "John Doe"
    )
    result = strip_signature(text)
    assert "follow up on the proposal" in result
    assert "Please let me know your thoughts" in result
    assert "John Doe" not in result


def test_empty_input_returns_empty() -> None:
    assert strip_signature("") == ""
    assert strip_disclaimers("") == ""
    assert strip_quoted_replies("") == ""
    assert normalize_whitespace("") == ""
    assert clean_email_body("") == ""


# ---------------------------------------------------------------------------
# strip_disclaimers
# ---------------------------------------------------------------------------


def test_strips_legal_disclaimer() -> None:
    text = (
        "Please find the report attached.\n\n"
        "This email is confidential and intended solely for the use of "
        "the individual or entity to whom it is addressed."
    )
    result = strip_disclaimers(text)
    assert "confidential" not in result
    assert "Please find the report attached" in result


def test_strips_disclaimer_keyword() -> None:
    text = "Thanks.\n\nDISCLAIMER: This message contains privileged information."
    result = strip_disclaimers(text)
    assert "DISCLAIMER" not in result
    assert "Thanks" in result


def test_strips_not_intended_recipient() -> None:
    text = (
        "See you at 3pm.\n\n"
        "If you are not the intended recipient, please notify the sender immediately."
    )
    result = strip_disclaimers(text)
    assert "not the intended recipient" not in result
    assert "See you at 3pm" in result


def test_strips_confidentiality_notice() -> None:
    text = "Agreed.\n\nConfidentiality notice: This communication is privileged."
    result = strip_disclaimers(text)
    assert "Confidentiality notice" not in result
    assert "Agreed" in result


# ---------------------------------------------------------------------------
# strip_quoted_replies
# ---------------------------------------------------------------------------


def test_strips_greater_than_quote_lines() -> None:
    text = "My answer is yes.\n> On Mon, Jan 1 wrote:\n> Sure, let's do it."
    result = strip_quoted_replies(text)
    assert "> " not in result
    assert "My answer is yes" in result


def test_strips_original_message_block() -> None:
    text = (
        "See inline.\n\n"
        "-----Original Message-----\n"
        "From: alice@example.com\n"
        "To: bob@example.com\n"
        "Subject: Follow-up\n\n"
        "Original content here."
    )
    result = strip_quoted_replies(text)
    assert "Original Message" not in result
    assert "Original content here" not in result
    assert "See inline" in result


def test_strips_outlook_quote_headers() -> None:
    text = (
        "Sounds good.\n\n"
        "On Mon, 1 Jan 2024 at 10:00, Alice Smith <alice@example.com> wrote:\n"
        "> Let's schedule a call."
    )
    result = strip_quoted_replies(text)
    assert "alice@example.com" not in result
    assert "Sounds good" in result


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


def test_normalizes_whitespace() -> None:
    text = "Hello.\n\n\n\n\nWorld."
    result = normalize_whitespace(text)
    assert "\n\n\n" not in result
    assert "Hello" in result
    assert "World" in result


def test_strips_trailing_spaces_on_lines() -> None:
    text = "Line one   \nLine two  "
    result = normalize_whitespace(text)
    for line in result.splitlines():
        assert line == line.rstrip()


# ---------------------------------------------------------------------------
# clean_email_body — composite
# ---------------------------------------------------------------------------


def test_handles_multiple_cleaning_passes() -> None:
    """Full pipeline removes quotes, disclaimers, and signatures together."""
    text = (
        "Hi,\n\n"
        "The contract is ready for review.\n\n"
        "On Mon, Jan 1 2024, Alice wrote:\n"
        "> Please send the draft.\n\n"
        "This email is confidential and intended solely for the addressee.\n\n"
        "--\n"
        "Bob\n"
        "CEO"
    )
    result = clean_email_body(text)
    assert "contract is ready for review" in result
    assert "> " not in result
    assert "confidential" not in result
    assert "CEO" not in result


def test_clean_email_body_preserves_main_content() -> None:
    text = (
        "Team,\n\n"
        "Please review the Q3 numbers before Friday.\n"
        "Key highlights:\n"
        "- Revenue up 12%\n"
        "- Churn down to 3%\n\n"
        "Sent from my iPhone"
    )
    result = clean_email_body(text)
    assert "Revenue up 12%" in result
    assert "Churn down to 3%" in result
    assert "iPhone" not in result
