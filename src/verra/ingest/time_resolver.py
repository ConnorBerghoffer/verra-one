"""Implicit time reference resolution for ingested documents.

Converts relative date expressions (e.g. "last quarter", "next Friday")
into absolute date strings using the document's own date as the reference
point. Falls back to today when the document date cannot be determined.

Public API
----------
resolve_time_references(text, reference_date) → str
    Replace relative references in *text* with absolute date strings.

extract_document_date(text, file_path) → date | None
    Infer a document's date from its content or filename.
"""


from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WEEKDAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Quarter boundaries: (start_month, end_month)
_QUARTER_MONTHS: dict[int, tuple[int, int]] = {
    1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12),
}


def _quarter_for(d: date) -> int:
    """Return the calendar quarter (1-4) that *d* falls in."""
    return (d.month - 1) // 3 + 1


def _quarter_label(q: int, year: int) -> str:
    start_month, end_month = _QUARTER_MONTHS[q]
    start_name = date(year, start_month, 1).strftime("%b")
    end_name = date(year, end_month, 1).strftime("%b")
    return f"Q{q} {year} ({start_name}–{end_name} {year})"


def _next_weekday(ref: date, weekday: int) -> date:
    """Return the next occurrence of *weekday* (0=Mon) after *ref* (exclusive)."""
    days_ahead = weekday - ref.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return ref + timedelta(days=days_ahead)


def _prev_weekday(ref: date, weekday: int) -> date:
    """Return the most recent past occurrence of *weekday* before *ref*."""
    days_behind = ref.weekday() - weekday
    if days_behind <= 0:
        days_behind += 7
    return ref - timedelta(days=days_behind)


def _end_of_month(d: date) -> date:
    last_day = monthrange(d.year, d.month)[1]
    return date(d.year, d.month, last_day)


def _end_of_quarter(d: date) -> date:
    q = _quarter_for(d)
    _, end_month = _QUARTER_MONTHS[q]
    last_day = monthrange(d.year, end_month)[1]
    return date(d.year, end_month, last_day)


def _end_of_year(d: date) -> date:
    return date(d.year, 12, 31)


# ---------------------------------------------------------------------------
# Replacement rules
#
# Each rule is a (pattern, replacement_fn) pair.
# replacement_fn(match, ref_date) → str
# ---------------------------------------------------------------------------

_RULES: list[tuple[re.Pattern[str], object]] = []


def _rule(pattern: str, flags: int = re.IGNORECASE) -> object:
    """Decorator that registers a replacement rule."""
    def decorator(fn: object) -> object:
        _RULES.append((re.compile(pattern, flags), fn))
        return fn
    return decorator


# ---- Simple relatives: yesterday / today / tomorrow ----

@_rule(r"\byesterday\b")
def _yesterday(m: re.Match[str], ref: date) -> str:
    return (ref - timedelta(days=1)).isoformat()


@_rule(r"\btoday\b")
def _today(m: re.Match[str], ref: date) -> str:
    return ref.isoformat()


@_rule(r"\btomorrow\b")
def _tomorrow(m: re.Match[str], ref: date) -> str:
    return (ref + timedelta(days=1)).isoformat()


# ---- N days/weeks/months ago ----

@_rule(r"\b(\d+)\s+days?\s+ago\b")
def _n_days_ago(m: re.Match[str], ref: date) -> str:
    n = int(m.group(1))
    return (ref - timedelta(days=n)).isoformat()


@_rule(r"\b(\d+)\s+weeks?\s+ago\b")
def _n_weeks_ago(m: re.Match[str], ref: date) -> str:
    n = int(m.group(1))
    return (ref - timedelta(weeks=n)).isoformat()


@_rule(r"\b(\d+)\s+months?\s+ago\b")
def _n_months_ago(m: re.Match[str], ref: date) -> str:
    n = int(m.group(1))
    month = ref.month - n
    year = ref.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    day = min(ref.day, monthrange(year, month)[1])
    return date(year, month, day).isoformat()


# ---- last week / last month / last quarter / last year ----

@_rule(r"\blast\s+week\b")
def _last_week(m: re.Match[str], ref: date) -> str:
    # ISO week that contains the Monday 7 days before ref's Monday
    monday = ref - timedelta(days=ref.weekday())  # this Monday
    prev_monday = monday - timedelta(weeks=1)
    prev_sunday = prev_monday + timedelta(days=6)
    return f"{prev_monday.isoformat()} to {prev_sunday.isoformat()}"


@_rule(r"\blast\s+month\b")
def _last_month(m: re.Match[str], ref: date) -> str:
    first_of_this = ref.replace(day=1)
    last_of_prev = first_of_this - timedelta(days=1)
    first_of_prev = last_of_prev.replace(day=1)
    return f"{first_of_prev.isoformat()} to {last_of_prev.isoformat()}"


@_rule(r"\blast\s+quarter\b")
def _last_quarter(m: re.Match[str], ref: date) -> str:
    q = _quarter_for(ref)
    prev_q = q - 1 if q > 1 else 4
    year = ref.year if q > 1 else ref.year - 1
    start_month, end_month = _QUARTER_MONTHS[prev_q]
    last_day = monthrange(year, end_month)[1]
    start = date(year, start_month, 1)
    end = date(year, end_month, last_day)
    return f"{_quarter_label(prev_q, year)} ({start.isoformat()} to {end.isoformat()})"


@_rule(r"\blast\s+year\b")
def _last_year(m: re.Match[str], ref: date) -> str:
    y = ref.year - 1
    return f"{y} ({date(y,1,1).isoformat()} to {date(y,12,31).isoformat()})"


# ---- this quarter / this week / this month / this year ----

@_rule(r"\bthis\s+quarter\b")
def _this_quarter(m: re.Match[str], ref: date) -> str:
    q = _quarter_for(ref)
    start_month, end_month = _QUARTER_MONTHS[q]
    start = date(ref.year, start_month, 1)
    last_day = monthrange(ref.year, end_month)[1]
    end = date(ref.year, end_month, last_day)
    return f"{_quarter_label(q, ref.year)} ({start.isoformat()} to {end.isoformat()})"


@_rule(r"\bthis\s+week\b")
def _this_week(m: re.Match[str], ref: date) -> str:
    monday = ref - timedelta(days=ref.weekday())
    sunday = monday + timedelta(days=6)
    return f"{monday.isoformat()} to {sunday.isoformat()}"


@_rule(r"\bthis\s+month\b")
def _this_month(m: re.Match[str], ref: date) -> str:
    start = ref.replace(day=1)
    end = _end_of_month(ref)
    return f"{start.isoformat()} to {end.isoformat()}"


@_rule(r"\bthis\s+year\b")
def _this_year(m: re.Match[str], ref: date) -> str:
    y = ref.year
    return f"{y} ({date(y,1,1).isoformat()} to {date(y,12,31).isoformat()})"


# ---- next <weekday> ----

@_rule(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
)
def _next_named_weekday(m: re.Match[str], ref: date) -> str:
    wday = _WEEKDAY_NAMES[m.group(1).lower()]
    target = _next_weekday(ref, wday)
    return target.isoformat()


# ---- last <weekday> ----

@_rule(
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
)
def _last_named_weekday(m: re.Match[str], ref: date) -> str:
    wday = _WEEKDAY_NAMES[m.group(1).lower()]
    target = _prev_weekday(ref, wday)
    return target.isoformat()


# ---- end of month / end of quarter / end of year ----

@_rule(r"\bend\s+of\s+(the\s+)?month\b")
def _end_of_month_rule(m: re.Match[str], ref: date) -> str:
    return _end_of_month(ref).isoformat()


@_rule(r"\bend\s+of\s+(the\s+)?quarter\b")
def _end_of_quarter_rule(m: re.Match[str], ref: date) -> str:
    return _end_of_quarter(ref).isoformat()


@_rule(r"\bend\s+of\s+(the\s+)?year\b")
def _end_of_year_rule(m: re.Match[str], ref: date) -> str:
    return _end_of_year(ref).isoformat()


# ---- next week / next month / next quarter / next year ----

@_rule(r"\bnext\s+week\b")
def _next_week(m: re.Match[str], ref: date) -> str:
    monday = ref - timedelta(days=ref.weekday()) + timedelta(weeks=1)
    sunday = monday + timedelta(days=6)
    return f"{monday.isoformat()} to {sunday.isoformat()}"


@_rule(r"\bnext\s+month\b")
def _next_month(m: re.Match[str], ref: date) -> str:
    month = ref.month + 1 if ref.month < 12 else 1
    year = ref.year + (1 if ref.month == 12 else 0)
    start = date(year, month, 1)
    end = _end_of_month(start)
    return f"{start.isoformat()} to {end.isoformat()}"


@_rule(r"\bnext\s+quarter\b")
def _next_quarter(m: re.Match[str], ref: date) -> str:
    q = _quarter_for(ref)
    next_q = q + 1 if q < 4 else 1
    year = ref.year + (1 if q == 4 else 0)
    start_month, end_month = _QUARTER_MONTHS[next_q]
    start = date(year, start_month, 1)
    end = date(year, end_month, monthrange(year, end_month)[1])
    return f"{_quarter_label(next_q, year)} ({start.isoformat()} to {end.isoformat()})"


@_rule(r"\bnext\s+year\b")
def _next_year(m: re.Match[str], ref: date) -> str:
    y = ref.year + 1
    return f"{y} ({date(y,1,1).isoformat()} to {date(y,12,31).isoformat()})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_time_references(
    text: str,
    reference_date: date | None = None,
) -> str:
    """Replace relative time references in *text* with absolute date strings.

    Parameters
    ----------
    text:
        The document text to process.
    reference_date:
        The date to resolve relative references against. When *None*, the
        current date (today) is used.

    Returns
    -------
    The text with relative references replaced.  Each replacement is
    annotated with the original phrase in brackets so the change is
    transparent, e.g. ``"last week [2024-01-08 to 2024-01-14]"``.
    """
    if not text:
        return text

    ref = reference_date or date.today()

    result = text
    for pattern, fn in _RULES:
        def _replacer(m: re.Match[str], _fn: object = fn, _ref: date = ref) -> str:
            resolved = _fn(m, _ref)  # type: ignore[call-arg]
            return f"{m.group(0)} [{resolved}]"

        result = pattern.sub(_replacer, result)

    return result


# ---------------------------------------------------------------------------
# Document date extraction
# ---------------------------------------------------------------------------

# Patterns checked against the first 3000 characters of a document
_DATE_PATTERNS_IN_CONTENT: list[re.Pattern[str]] = [
    # ISO date: 2024-03-15
    re.compile(r"\bdate\s*[:\-]\s*(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
    # "Last updated: March 15, 2024"
    re.compile(
        r"last\s+updated\s*[:\-]\s*(\w+\s+\d{1,2},?\s*\d{4})", re.IGNORECASE
    ),
    # RFC 2822 / email date headers: "Date: Mon, 15 Mar 2024 ..."
    re.compile(
        r"^date\s*:\s*\w{3},\s*(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE | re.MULTILINE
    ),
]

# Filename patterns: 2024-03-15-*, *-2024-03-15.*, 20240315-*
_DATE_IN_FILENAME = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")
_COMPACT_DATE_IN_FILENAME = re.compile(r"(\d{4})(\d{2})(\d{2})")

_MONTH_ABBREVS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_flexible_date(text: str) -> date | None:
    """Try to parse a date from a human-readable string."""
    # ISO: 2024-03-15
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", text.strip())
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # "March 15 2024" or "15 March 2024" or "Mar 15, 2024"
    m = re.match(
        r"(\w+)\s+(\d{1,2}),?\s*(\d{4})|(\d{1,2})\s+(\w+)\s+(\d{4})",
        text.strip(),
        re.IGNORECASE,
    )
    if m:
        try:
            if m.group(1):  # "Month Day Year"
                month_str = m.group(1).lower()[:3]
                month = _MONTH_ABBREVS.get(month_str) or _MONTH_NAMES.get(m.group(1).lower())
                if month:
                    return date(int(m.group(3)), month, int(m.group(2)))
            else:  # "Day Month Year"
                month_str = m.group(5).lower()[:3]
                month = _MONTH_ABBREVS.get(month_str) or _MONTH_NAMES.get(m.group(5).lower())
                if month:
                    return date(int(m.group(6)), month, int(m.group(4)))
        except ValueError:
            pass

    return None


def extract_document_date(text: str, file_path: str) -> date | None:
    """Try to determine the document's date from its content or filename.

    Checks (in order):
    1. 'Date: YYYY-MM-DD' or 'Date: Mon, 15 Mar 2024' patterns in content.
    2. 'Last updated: ...' patterns in content.
    3. Filename patterns like '2024-03-15-meeting-notes.md'.

    Parameters
    ----------
    text:
        Document text (first few thousand characters are sufficient).
    file_path:
        File path — the filename component is inspected for date patterns.

    Returns
    -------
    A ``date`` object when a date can be confidently extracted, or *None*.
    """
    # Check content first (first 3000 chars)
    snippet = text[:3000]
    for pattern in _DATE_PATTERNS_IN_CONTENT:
        m = pattern.search(snippet)
        if m:
            parsed = _parse_flexible_date(m.group(1))
            if parsed:
                return parsed

    # Check filename
    filename = Path(file_path).name
    m2 = _DATE_IN_FILENAME.search(filename)
    if m2:
        try:
            return date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        except ValueError:
            pass

    # Compact date in filename (20240315)
    m3 = _COMPACT_DATE_IN_FILENAME.search(filename)
    if m3:
        try:
            d = date(int(m3.group(1)), int(m3.group(2)), int(m3.group(3)))
            # Sanity check: year must be plausible
            if 2000 <= d.year <= date.today().year + 5:
                return d
        except ValueError:
            pass

    return None
