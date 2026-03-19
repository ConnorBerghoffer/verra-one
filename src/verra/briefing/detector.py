"""Briefing detector — surfaces actionable insights from ingested data.

Each detect_* method runs a targeted SQL query and returns a list of
BriefingItem objects.  detect_all() fans them all out, deduplicates
against the dismissed set, then returns the top N by urgency.

Design decisions
----------------
- All queries go directly through the sqlite3.Connection objects supplied
  at construction time; no ORM, no store wrappers.  This keeps the module
  self-contained and avoids circular imports.
- Every query is wrapped in a try/except so a missing or empty table never
  crashes the briefing — it just produces no items for that detector.
- Dismissals are stored in the existing ``memory`` table with
  category='dismissed' and key=<item_key>.  This piggybacks on the already-
  committed memory table rather than adding a new table.
- The ``config`` parameter accepts the ``BriefingConfig`` Pydantic model or
  any object with the same attributes (``max_items``, ``stale_lead_days``,
  ``contract_warning_days``).
"""


from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class BriefingItem:
    """A single actionable insight surfaced by the briefing detector."""

    category: str
    """One of: 'stale_lead', 'expiring_contract', 'forgotten_commitment',
    'pattern', 'new_data'."""

    title: str
    """Short, human-readable summary (one line)."""

    detail: str
    """Longer explanation of why this item was flagged."""

    entity_name: str | None
    """The person, organisation, or document most relevant to this item."""

    urgency: int
    """1–5, higher is more urgent.  Used to rank items in detect_all()."""

    source_label: str
    """Where this was detected, e.g. 'emails table', 'commitments table'."""

    detected_at: datetime
    """When this BriefingItem was generated (not when the underlying event
    occurred)."""

    # Derived key used for dismissal lookups — not part of the public API but
    # stored on the object so callers can call detector.dismiss(item.item_key).
    item_key: str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Patterns used to find expiry-like phrases in chunk text.
_EXPIRY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"expires?\s+on\s+([\w\s,]+\d{4})", re.IGNORECASE),
    re.compile(r"expir(?:y|ation)\s+date[:\s]+([\w\s,]+\d{4})", re.IGNORECASE),
    re.compile(r"renewal\s+date[:\s]+([\w\s,]+\d{4})", re.IGNORECASE),
    re.compile(r"due\s+by[:\s]+([\w\s,]+\d{4})", re.IGNORECASE),
    re.compile(r"valid\s+until[:\s]+([\w\s,]+\d{4})", re.IGNORECASE),
    re.compile(r"termination\s+date[:\s]+([\w\s,]+\d{4})", re.IGNORECASE),
]

# Common English stop-words filtered out during pattern detection.
_STOP_WORDS: frozenset[str] = frozenset(
    [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "it", "its", "this", "that", "these",
        "those", "i", "me", "my", "we", "our", "you", "your", "he", "she",
        "they", "them", "their", "what", "which", "who", "how", "when",
        "where", "why", "just", "about", "verra", "tell", "show", "find",
    ]
)


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_key(category: str, identifier: str) -> str:
    """Build a stable dismissal key for a briefing item."""
    # Normalise to avoid whitespace issues
    clean = re.sub(r"\s+", "_", identifier.strip().lower())
    return f"{category}:{clean}"


def _safe_rows(
    conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()
) -> list[sqlite3.Row]:
    """Execute a query and return rows, or an empty list on any error."""
    try:
        return conn.execute(sql, params).fetchall()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class BriefingDetector:
    """Analyses ingested data to surface actionable insights.

    Parameters
    ----------
    core_conn:
        sqlite3.Connection to core.db (documents, chunks, emails, memory,
        conversations, messages tables).
    analysis_conn:
        sqlite3.Connection to analysis.db (commitments, chunk_analysis tables).
    config:
        BriefingConfig (or duck-typed equivalent) providing ``max_items``,
        ``stale_lead_days``, and ``contract_warning_days``.
    user_email:
        The user's own email address.  Used to distinguish sent mail from
        received mail in stale lead detection.  If None, stale lead detection
        is skipped.
    """

    def __init__(
        self,
        core_conn: sqlite3.Connection,
        analysis_conn: sqlite3.Connection,
        config: Any,
        user_email: str | None = None,
    ) -> None:
        self._core = core_conn
        self._analysis = analysis_conn
        self._config = config
        self._user_email = user_email

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all(self, max_items: int | None = None) -> list[BriefingItem]:
        """Run all detectors and return the top items sorted by urgency desc.

        Items that have been dismissed are excluded.  If ``max_items`` is None
        the value from the config is used.
        """
        limit = max_items if max_items is not None else self._config.max_items

        items: list[BriefingItem] = []

        items.extend(
            self.detect_stale_leads(
                days_threshold=self._config.stale_lead_days
            )
        )
        items.extend(
            self.detect_expiring_contracts(
                warning_days=self._config.contract_warning_days
            )
        )
        items.extend(self.detect_forgotten_commitments())
        items.extend(self.detect_patterns())
        items.extend(self.detect_new_data())

        # Filter dismissed items
        active = [i for i in items if not self._is_dismissed(i.item_key)]

        # Sort by urgency descending, then by detected_at descending as tiebreak
        active.sort(key=lambda i: (i.urgency, i.detected_at), reverse=True)

        return active[:limit]

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def detect_stale_leads(self, days_threshold: int = 14) -> list[BriefingItem]:
        """Find threads where the user sent a message and got no reply.

        A "stale lead" thread is one where:
          - the most recent email FROM the user was sent more than
            ``days_threshold`` days ago, AND
          - no email from anyone *else* exists in the same thread after
            that sent message.

        If ``user_email`` was not provided at construction time, returns [].
        """
        if not self._user_email:
            return []

        cutoff = (_now_utc() - timedelta(days=days_threshold)).strftime(
            "%Y-%m-%d"
        )

        # Find threads where the last email from the user has no later reply
        # from another address.  We do this in two queries to keep the SQL
        # readable: first get the per-thread latest "sent" date, then check
        # that no later email in the thread has a different from_addr.
        sent_rows = _safe_rows(
            self._core,
            """
            SELECT thread_id,
                   subject,
                   MAX(date) AS last_sent_date
              FROM emails
             WHERE from_addr LIKE ?
               AND date < ?
             GROUP BY thread_id
            """,
            (f"%{self._user_email}%", cutoff),
        )

        items: list[BriefingItem] = []
        for row in sent_rows:
            thread_id = row["thread_id"]
            last_sent = row["last_sent_date"]
            subject = row["subject"] or "(no subject)"

            # Check if anyone else replied after the user's last sent email
            reply_rows = _safe_rows(
                self._core,
                """
                SELECT COUNT(*) AS cnt
                  FROM emails
                 WHERE thread_id = ?
                   AND from_addr NOT LIKE ?
                   AND date > ?
                """,
                (thread_id, f"%{self._user_email}%", last_sent),
            )

            reply_count = reply_rows[0]["cnt"] if reply_rows else 0
            if reply_count > 0:
                continue

            # Calculate days since last sent
            try:
                sent_dt = datetime.fromisoformat(last_sent.replace("Z", "+00:00"))
                days_ago = (_now_utc() - sent_dt.replace(tzinfo=timezone.utc if sent_dt.tzinfo is None else sent_dt.tzinfo)).days
            except (ValueError, TypeError, AttributeError):
                days_ago = days_threshold  # fallback

            # Urgency scales with how long it's been waiting
            urgency = 3
            if days_ago > days_threshold * 3:
                urgency = 5
            elif days_ago > days_threshold * 2:
                urgency = 4

            key = _make_key("stale_lead", thread_id)
            item = BriefingItem(
                category="stale_lead",
                title=f"No reply: {subject}",
                detail=(
                    f"You sent an email in thread '{subject}' "
                    f"{days_ago} days ago and have not received a reply."
                ),
                entity_name=None,
                urgency=urgency,
                source_label="emails table",
                detected_at=_now_utc(),
                item_key=key,
            )
            items.append(item)

        return items

    def detect_expiring_contracts(
        self, warning_days: int = 30
    ) -> list[BriefingItem]:
        """Find contracts with approaching expiry dates.

        Two sources are checked:
        1. The ``chunks.valid_until`` column (set during ingestion when a
           temporal boundary is detected).
        2. The ``chunks.metadata`` JSON blob (text may contain expiry phrases
           matched by ``_EXPIRY_PATTERNS``).
        3. The ``documents.extra_metadata`` JSON blob for 'expiry_date' or
           'valid_until' keys.

        Only documents classified as 'contract' in the documents table are
        considered.
        """
        now = _now_utc()
        warning_horizon = (now + timedelta(days=warning_days)).strftime(
            "%Y-%m-%d"
        )
        today_str = now.strftime("%Y-%m-%d")

        items: list[BriefingItem] = []
        seen_doc_ids: set[int] = set()

        # --- Source 1: chunks.valid_until -----------------------------------
        valid_until_rows = _safe_rows(
            self._core,
            """
            SELECT d.id           AS doc_id,
                   d.file_name,
                   d.extra_metadata,
                   c.valid_until
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE d.document_type = 'contract'
               AND c.valid_until IS NOT NULL
               AND c.valid_until BETWEEN ? AND ?
             ORDER BY c.valid_until ASC
            """,
            (today_str, warning_horizon),
        )

        for row in valid_until_rows:
            doc_id = row["doc_id"]
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            try:
                expiry_dt = datetime.fromisoformat(str(row["valid_until"]).replace("Z", "+00:00"))
                days_left = (expiry_dt.replace(tzinfo=timezone.utc if expiry_dt.tzinfo is None else expiry_dt.tzinfo) - now).days
            except (ValueError, TypeError):
                days_left = warning_days

            entity_name = _extract_entity_from_metadata(row["extra_metadata"])
            urgency = _expiry_urgency(days_left, warning_days)
            key = _make_key("expiring_contract", f"doc_{doc_id}")

            items.append(
                BriefingItem(
                    category="expiring_contract",
                    title=f"Contract expiring: {row['file_name']}",
                    detail=(
                        f"'{row['file_name']}' has a valid_until date of "
                        f"{row['valid_until']} ({days_left} days from now)."
                    ),
                    entity_name=entity_name,
                    urgency=urgency,
                    source_label="chunks.valid_until",
                    detected_at=now,
                    item_key=key,
                )
            )

        # --- Source 2: documents.extra_metadata JSON ------------------------
        meta_rows = _safe_rows(
            self._core,
            """
            SELECT id, file_name, extra_metadata
              FROM documents
             WHERE document_type = 'contract'
               AND extra_metadata IS NOT NULL
            """,
        )

        for row in meta_rows:
            doc_id = row["id"]
            if doc_id in seen_doc_ids:
                continue

            expiry_str = _extract_expiry_from_json_metadata(row["extra_metadata"])
            if not expiry_str:
                continue

            try:
                expiry_dt = datetime.fromisoformat(expiry_str)
                days_left = (expiry_dt.replace(tzinfo=timezone.utc if expiry_dt.tzinfo is None else expiry_dt.tzinfo) - now).days
            except (ValueError, TypeError):
                continue

            if days_left < 0 or days_left > warning_days:
                continue

            seen_doc_ids.add(doc_id)
            entity_name = _extract_entity_from_metadata(row["extra_metadata"])
            urgency = _expiry_urgency(days_left, warning_days)
            key = _make_key("expiring_contract", f"doc_{doc_id}")

            items.append(
                BriefingItem(
                    category="expiring_contract",
                    title=f"Contract expiring: {row['file_name']}",
                    detail=(
                        f"'{row['file_name']}' metadata shows an expiry date of "
                        f"{expiry_str} ({days_left} days from now)."
                    ),
                    entity_name=entity_name,
                    urgency=urgency,
                    source_label="documents.extra_metadata",
                    detected_at=now,
                    item_key=key,
                )
            )

        # --- Source 3: chunk text patterns ----------------------------------
        # Only scan chunks for contract documents not already captured above.
        text_rows = _safe_rows(
            self._core,
            """
            SELECT d.id AS doc_id, d.file_name, d.extra_metadata, c.metadata AS chunk_meta
              FROM chunks c
              JOIN documents d ON d.id = c.document_id
             WHERE d.document_type = 'contract'
            """,
        )

        for row in text_rows:
            doc_id = row["doc_id"]
            if doc_id in seen_doc_ids:
                continue

            chunk_text = _extract_text_from_chunk_meta(row["chunk_meta"])
            if not chunk_text:
                continue

            for pattern in _EXPIRY_PATTERNS:
                match = pattern.search(chunk_text)
                if not match:
                    continue
                date_str = match.group(1).strip()
                try:
                    # Try common date formats
                    expiry_dt = _parse_loose_date(date_str)
                    if expiry_dt is None:
                        continue
                    days_left = (expiry_dt - now.replace(tzinfo=None)).days
                except (ValueError, TypeError):
                    continue

                if days_left < 0 or days_left > warning_days:
                    break  # found a pattern but date is out of window

                seen_doc_ids.add(doc_id)
                entity_name = _extract_entity_from_metadata(row["extra_metadata"])
                urgency = _expiry_urgency(days_left, warning_days)
                key = _make_key("expiring_contract", f"doc_{doc_id}")

                items.append(
                    BriefingItem(
                        category="expiring_contract",
                        title=f"Contract expiring: {row['file_name']}",
                        detail=(
                            f"'{row['file_name']}' contains expiry language: "
                            f"'{match.group(0).strip()}' ({days_left} days from now)."
                        ),
                        entity_name=entity_name,
                        urgency=urgency,
                        source_label="chunk text patterns",
                        detected_at=now,
                        item_key=key,
                    )
                )
                break  # one item per document

        return items

    def detect_forgotten_commitments(self) -> list[BriefingItem]:
        """Find open commitments in analysis.db that are overdue or approaching.

        Commitments with ``status = 'open'`` and a ``due_date`` in the past
        are marked urgent.  Those due within the next 7 days are also surfaced
        at lower urgency.
        """
        now = _now_utc()
        today_str = now.strftime("%Y-%m-%d")
        week_ahead = (now + timedelta(days=7)).strftime("%Y-%m-%d")

        # Overdue: past due date
        overdue_rows = _safe_rows(
            self._analysis,
            """
            SELECT id, who_name, what, due_date, detected_at
              FROM commitments
             WHERE status = 'open'
               AND due_date IS NOT NULL
               AND due_date < ?
             ORDER BY due_date ASC
            """,
            (today_str,),
        )

        # No due date — never expires but still open (lower urgency)
        open_no_date_rows = _safe_rows(
            self._analysis,
            """
            SELECT id, who_name, what, due_date, detected_at
              FROM commitments
             WHERE status = 'open'
               AND due_date IS NULL
             ORDER BY detected_at ASC
            """,
        )

        # Due soon: within 7 days
        due_soon_rows = _safe_rows(
            self._analysis,
            """
            SELECT id, who_name, what, due_date, detected_at
              FROM commitments
             WHERE status = 'open'
               AND due_date IS NOT NULL
               AND due_date BETWEEN ? AND ?
             ORDER BY due_date ASC
            """,
            (today_str, week_ahead),
        )

        items: list[BriefingItem] = []

        for row in overdue_rows:
            days_overdue = _days_ago(row["due_date"], now)
            key = _make_key("forgotten_commitment", str(row["id"]))
            items.append(
                BriefingItem(
                    category="forgotten_commitment",
                    title=f"Overdue: {row['what'][:60]}",
                    detail=(
                        f"Commitment by {row['who_name']}: '{row['what']}' "
                        f"was due on {row['due_date']} ({days_overdue} days ago)."
                    ),
                    entity_name=row["who_name"],
                    urgency=5 if days_overdue > 14 else 4,
                    source_label="commitments table",
                    detected_at=now,
                    item_key=key,
                )
            )

        for row in due_soon_rows:
            days_left = _days_until(row["due_date"], now)
            key = _make_key("forgotten_commitment", str(row["id"]))
            items.append(
                BriefingItem(
                    category="forgotten_commitment",
                    title=f"Due soon: {row['what'][:60]}",
                    detail=(
                        f"Commitment by {row['who_name']}: '{row['what']}' "
                        f"is due on {row['due_date']} ({days_left} days from now)."
                    ),
                    entity_name=row["who_name"],
                    urgency=3,
                    source_label="commitments table",
                    detected_at=now,
                    item_key=key,
                )
            )

        for row in open_no_date_rows:
            # Only surface if it was detected more than 30 days ago and still open.
            detected_days_ago = _days_ago(row["detected_at"], now)
            if detected_days_ago < 30:
                continue
            key = _make_key("forgotten_commitment", str(row["id"]))
            items.append(
                BriefingItem(
                    category="forgotten_commitment",
                    title=f"Long-standing open: {row['what'][:60]}",
                    detail=(
                        f"Commitment by {row['who_name']}: '{row['what']}' "
                        f"has been open for {detected_days_ago} days with no due date."
                    ),
                    entity_name=row["who_name"],
                    urgency=2,
                    source_label="commitments table",
                    detected_at=now,
                    item_key=key,
                )
            )

        return items

    def detect_patterns(self) -> list[BriefingItem]:
        """Detect topics that recur frequently in recent user chat messages.

        Looks at user messages from the last 14 days across all conversations.
        Words that appear in at least 3 distinct conversations are treated as
        a recurring pattern worth surfacing.

        Returns at most 3 pattern items to avoid noise.
        """
        now = _now_utc()
        since = (now - timedelta(days=14)).strftime("%Y-%m-%d")

        # Fetch recent user messages
        msg_rows = _safe_rows(
            self._core,
            """
            SELECT m.content, m.conversation_id
              FROM messages m
             WHERE m.role = 'user'
               AND m.created_at >= ?
            """,
            (since,),
        )

        if not msg_rows:
            return []

        # Map word -> set of conversation_ids where it appeared
        word_convs: dict[str, set[int]] = {}
        for row in msg_rows:
            conv_id = row["conversation_id"]
            words = re.findall(r"\b[a-zA-Z]{4,}\b", row["content"].lower())
            for word in words:
                if word in _STOP_WORDS:
                    continue
                word_convs.setdefault(word, set()).add(conv_id)

        # Find words that appear in >= 3 distinct conversations
        recurring = [
            (word, convs)
            for word, convs in word_convs.items()
            if len(convs) >= 3
        ]

        # Sort by frequency descending
        recurring.sort(key=lambda x: len(x[1]), reverse=True)

        items: list[BriefingItem] = []
        for word, convs in recurring[:3]:
            key = _make_key("pattern", word)
            items.append(
                BriefingItem(
                    category="pattern",
                    title=f"Recurring topic: '{word}'",
                    detail=(
                        f"You've asked about '{word}' in {len(convs)} separate "
                        f"conversations over the last 14 days.  This may indicate "
                        f"an ongoing concern worth capturing in a document."
                    ),
                    entity_name=None,
                    urgency=2,
                    source_label="messages table",
                    detected_at=now,
                    item_key=key,
                )
            )

        return items

    def detect_new_data(self) -> list[BriefingItem]:
        """Summarise what has been ingested since the last briefing.

        Reads the timestamp of the last briefing from the memory table
        (category='briefing', key='last_run_at').  If no prior briefing
        exists, uses 24 hours ago as the window.

        Counts documents and emails ingested in that window.
        """
        now = _now_utc()

        # Determine window start
        last_run_row = _safe_rows(
            self._core,
            "SELECT value FROM memory WHERE category = 'briefing' AND key = 'last_run_at'",
        )
        if last_run_row:
            try:
                window_start = last_run_row[0]["value"]
            except (KeyError, IndexError):
                window_start = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        else:
            window_start = (now - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")

        # Count new documents
        doc_rows = _safe_rows(
            self._core,
            "SELECT COUNT(*) AS cnt FROM documents WHERE indexed_at > ?",
            (window_start,),
        )
        new_docs = doc_rows[0]["cnt"] if doc_rows else 0

        # Count new emails
        email_rows = _safe_rows(
            self._core,
            "SELECT COUNT(*) AS cnt FROM emails WHERE created_at > ?",
            (window_start,),
        )
        new_emails = email_rows[0]["cnt"] if email_rows else 0

        # Persist last run time so the next briefing uses an accurate window
        try:
            self._core.execute(
                """
                INSERT INTO memory (category, key, value)
                VALUES ('briefing', 'last_run_at', ?)
                ON CONFLICT(category, key) DO UPDATE SET value = excluded.value
                """,
                (now.strftime("%Y-%m-%dT%H:%M:%S"),),
            )
            self._core.commit()
        except Exception:
            pass  # Non-fatal — briefing window will just reset next time

        if new_docs == 0 and new_emails == 0:
            return []

        parts: list[str] = []
        if new_docs:
            parts.append(f"{new_docs} document{'s' if new_docs != 1 else ''}")
        if new_emails:
            parts.append(f"{new_emails} email{'s' if new_emails != 1 else ''}")

        summary = " and ".join(parts) + " ingested since last briefing."

        key = _make_key("new_data", now.strftime("%Y-%m-%d"))
        return [
            BriefingItem(
                category="new_data",
                title=f"New data: {summary}",
                detail=(
                    f"Since {window_start}: {summary}  "
                    f"You may want to review recently added content."
                ),
                entity_name=None,
                urgency=1,
                source_label="documents + emails tables",
                detected_at=now,
                item_key=key,
            )
        ]

    # ------------------------------------------------------------------
    # Dismissal helpers
    # ------------------------------------------------------------------

    def _is_dismissed(self, item_key: str) -> bool:
        """Return True if this item has been dismissed in the memory store."""
        rows = _safe_rows(
            self._core,
            "SELECT 1 FROM memory WHERE category = 'dismissed' AND key = ?",
            (item_key,),
        )
        return len(rows) > 0

    def dismiss(self, item_key: str) -> None:
        """Mark a briefing item as dismissed so it won't appear again.

        Stores a row in the ``memory`` table with category='dismissed'.
        Calling dismiss() on an already-dismissed key is a no-op.
        """
        try:
            self._core.execute(
                """
                INSERT INTO memory (category, key, value)
                VALUES ('dismissed', ?, 'dismissed')
                ON CONFLICT(category, key) DO NOTHING
                """,
                (item_key,),
            )
            self._core.commit()
        except Exception:
            pass  # Non-fatal — worst case the item reappears next briefing


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _expiry_urgency(days_left: int, warning_days: int) -> int:
    """Map days remaining to a 1-5 urgency score."""
    if days_left <= 0:
        return 5
    if days_left <= 7:
        return 5
    if days_left <= 14:
        return 4
    if days_left <= warning_days // 2:
        return 3
    return 2


def _days_ago(date_str: str | None, now: datetime) -> int:
    """Return how many days ago ``date_str`` was, relative to ``now``."""
    if not date_str:
        return 0
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        delta = now.replace(tzinfo=timezone.utc) - dt.replace(
            tzinfo=timezone.utc if dt.tzinfo is None else dt.tzinfo
        )
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 0


def _days_until(date_str: str | None, now: datetime) -> int:
    """Return how many days until ``date_str``, relative to ``now``."""
    if not date_str:
        return 0
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        delta = dt.replace(tzinfo=timezone.utc if dt.tzinfo is None else dt.tzinfo) - now.replace(tzinfo=timezone.utc)
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 0


def _extract_entity_from_metadata(extra_metadata_json: str | None) -> str | None:
    """Pull an entity or counterparty name from extra_metadata JSON."""
    if not extra_metadata_json:
        return None
    try:
        meta = json.loads(extra_metadata_json)
    except (json.JSONDecodeError, TypeError):
        return None
    # Try common keys in order of specificity
    for key in ("counterparty", "client", "vendor", "party", "entity", "company"):
        if key in meta and meta[key]:
            return str(meta[key])
    return None


def _extract_expiry_from_json_metadata(extra_metadata_json: str | None) -> str | None:
    """Extract an expiry date string from extra_metadata JSON."""
    if not extra_metadata_json:
        return None
    try:
        meta = json.loads(extra_metadata_json)
    except (json.JSONDecodeError, TypeError):
        return None
    for key in ("expiry_date", "valid_until", "expiration_date", "end_date", "renewal_date"):
        if key in meta and meta[key]:
            return str(meta[key])
    return None


def _extract_text_from_chunk_meta(chunk_meta_json: str | None) -> str:
    """Extract the text field from a chunk's metadata JSON blob."""
    if not chunk_meta_json:
        return ""
    try:
        meta = json.loads(chunk_meta_json)
        return str(meta.get("text", ""))
    except (json.JSONDecodeError, TypeError):
        return ""


def _parse_loose_date(date_str: str) -> datetime | None:
    """Try to parse a date string in several common formats.

    Returns a naive datetime (no tzinfo) or None if parsing fails.
    """
    # Normalise: remove day-of-week prefixes like "Monday, "
    date_str = re.sub(r"^[A-Za-z]+,\s*", "", date_str.strip())

    formats = [
        "%B %d, %Y",     # January 15, 2025
        "%B %d %Y",      # January 15 2025
        "%d %B %Y",      # 15 January 2025
        "%d/%m/%Y",      # 15/01/2025
        "%m/%d/%Y",      # 01/15/2025
        "%Y-%m-%d",      # 2025-01-15
        "%d-%m-%Y",      # 15-01-2025
        "%b %d, %Y",     # Jan 15, 2025
        "%b %d %Y",      # Jan 15 2025
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None
