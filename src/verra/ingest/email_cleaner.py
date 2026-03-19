"""Email body cleaning utilities.

Removes signatures, legal disclaimers, and quoted reply chains from
email bodies so that chunking and embedding operate on real content only.

Public API
----------
clean_email_body(text)      — run all cleaning passes in order
strip_signature(text)       — remove trailing signatures
strip_disclaimers(text)     — remove legal boilerplate
strip_quoted_replies(text)  — remove quoted reply chains
normalize_whitespace(text)  — collapse excessive blank lines / spaces
"""


from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Individual cleaning passes
# ---------------------------------------------------------------------------


def strip_signature(text: str) -> str:
    """Remove email signatures from the end of a message body.

    Handles:
    - '--' or '---' delimiter (RFC 3676 sig separator)
    - 'Sent from my iPhone / Android / iPad'
    - 'Get Outlook for iOS / Android'
    - Name / title / phone / address blocks at end (heuristic)
    """
    if not text:
        return text

    lines = text.splitlines()

    # Definitive delimiter patterns — everything from here onwards is sig
    _SIG_DELIMITERS = re.compile(
        r"^(--\s*|---+\s*|_{4,}\s*|={4,}\s*)$"
    )

    # Device / client footers (case-insensitive, anywhere in the line)
    _DEVICE_FOOTERS = re.compile(
        r"sent\s+from\s+(my\s+)?(iphone|android|ipad|blackberry|samsung|pixel|mail"
        r"|outlook|yahoo\s+mail|gmail|apple\s+mail)",
        re.IGNORECASE,
    )
    _CLIENT_FOOTERS = re.compile(
        r"get\s+outlook\s+for\s+(ios|android|mac|windows)",
        re.IGNORECASE,
    )

    # Find the last sig delimiter or device footer working backwards
    cut_at: int | None = None

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if _SIG_DELIMITERS.match(stripped):
            cut_at = i
            break
        if _DEVICE_FOOTERS.search(stripped) or _CLIENT_FOOTERS.search(stripped):
            # Device footers appear at the very end; cut from the nearest
            # preceding blank line or from this line itself.
            cut_at = i
            # Walk back to include any blank line before it
            while cut_at > 0 and not lines[cut_at - 1].strip():
                cut_at -= 1
            break

    if cut_at is not None:
        lines = lines[:cut_at]

    return "\n".join(lines).rstrip()


def strip_disclaimers(text: str) -> str:
    """Remove legal disclaimers that appear in email bodies.

    Handles multi-line blocks starting with common disclaimer signals.
    Everything from the first matched trigger line to the end of the
    message is removed (disclaimers almost always trail all real content).
    """
    if not text:
        return text

    _DISCLAIMER_STARTS = re.compile(
        r"(this\s+(e[\-]?mail|message|communication)\s+(is\s+)?(confidential|privileged|intended)"
        r"|disclaimer\s*[:：]"
        r"|if\s+you\s+(are\s+not\s+the\s+intended\s+recipient|have\s+received\s+this)"
        r"|the\s+information\s+(contained|in\s+this)\s+(e[\-]?mail|message)"
        r"|this\s+transmission\s+(may\s+contain|is\s+intended)"
        r"|confidentiality\s+notice"
        r"|legal\s+notice\s*[:：]"
        r"|privileged\s+(and\s+)?confidential)",
        re.IGNORECASE,
    )

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _DISCLAIMER_STARTS.search(line):
            # Remove blank lines immediately before the disclaimer
            cut = i
            while cut > 0 and not lines[cut - 1].strip():
                cut -= 1
            lines = lines[:cut]
            break

    return "\n".join(lines).rstrip()


def strip_quoted_replies(text: str) -> str:
    """Remove quoted reply chains from an email body.

    Handles:
    - Lines starting with '>' (standard quoting)
    - 'On [date], [person] wrote:' preamble + following quoted block
    - '-----Original Message-----' blocks (Outlook style)
    - 'From: ... Sent: ... To: ... Subject:' header blocks
    """
    if not text:
        return text

    # Patterns that introduce a quoted block — everything from the match
    # line onward is removed.
    _QUOTE_BLOCK_STARTS = re.compile(
        r"^(-{4,}\s*original\s+message\s*-{4,}"
        r"|\s*on\s+.{5,80}wrote\s*:\s*$"
        r"|from\s*:\s*\S.{0,100}\n\s*(sent|date)\s*:.{0,100}\n\s*to\s*:)"
        r"",
        re.IGNORECASE | re.MULTILINE,
    )

    # Drop lines beginning with '>'
    lines = text.splitlines()
    clean_lines: list[str] = []
    for line in lines:
        if line.strip().startswith(">"):
            continue
        clean_lines.append(line)

    rejoined = "\n".join(clean_lines)

    # Now cut at block-level quote introductions
    match = _QUOTE_BLOCK_STARTS.search(rejoined)
    if match:
        cut = match.start()
        # Trim trailing blank lines before the cut
        rejoined = rejoined[:cut].rstrip()

    return rejoined


def normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines and trailing whitespace.

    - More than two consecutive blank lines → two blank lines
    - Trailing whitespace on each line stripped
    - Leading/trailing whitespace on the whole message stripped
    """
    if not text:
        return text

    # Strip trailing spaces on each line
    lines = [l.rstrip() for l in text.splitlines()]
    rejoined = "\n".join(lines)

    # Collapse 3+ blank lines to 2
    rejoined = re.sub(r"\n{3,}", "\n\n", rejoined)

    return rejoined.strip()


# ---------------------------------------------------------------------------
# Composite cleaner
# ---------------------------------------------------------------------------


def clean_email_body(text: str) -> str:
    """Run all cleaning passes on an email body in the correct order.

    Order:
    1. strip_quoted_replies — removes the largest blocks first
    2. strip_disclaimers    — removes legal boilerplate
    3. strip_signature      — removes trailing signatures
    4. normalize_whitespace — tidies up spacing
    """
    if not text:
        return text

    text = strip_quoted_replies(text)
    text = strip_disclaimers(text)
    text = strip_signature(text)
    text = normalize_whitespace(text)
    return text
