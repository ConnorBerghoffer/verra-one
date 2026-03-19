"""TabularStore — manages CSV data as queryable SQLite tables."""

from __future__ import annotations

import csv
import io
import re
import sqlite3
from pathlib import Path
from typing import Any


# Columns whose names suggest numeric content — used to auto-detect types.
_NUMERIC_HINTS = re.compile(
    r"\b(count|total|amount|rate|score|pct|percent|revenue|price|cost|"
    r"qty|quantity|number|num|id|age|year|month|day|value|ratio|avg|average)\b",
    re.IGNORECASE,
)


def _sanitize_table_name(file_name: str) -> str:
    """Convert a file name to a valid SQLite identifier.

    Examples:
      'win_loss_2025.csv' -> 'win_loss_2025'
      'Sales Rep Data.csv' -> 'sales_rep_data'
    """
    stem = Path(file_name).stem
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("_")
    # Prefix with t_ if it starts with a digit
    if name and name[0].isdigit():
        name = "t_" + name
    return name.lower() or "table"


def _infer_column_type(header: str, sample_values: list[str]) -> str:
    """Return 'REAL' for likely-numeric columns, else 'TEXT'."""
    # Header name hint
    if _NUMERIC_HINTS.search(header):
        return "REAL"
    # Sample the first non-empty values
    numeric_count = 0
    total_count = 0
    for v in sample_values[:20]:
        v = v.strip().replace(",", "").replace("%", "").replace("$", "")
        if not v:
            continue
        total_count += 1
        try:
            float(v)
            numeric_count += 1
        except ValueError:
            pass
    if total_count > 0 and numeric_count / total_count >= 0.8:
        return "REAL"
    return "TEXT"


class TabularStore:
    """Manages CSV data as queryable SQLite tables.

    Each ingested CSV becomes a table named after the file stem.
    A ``_tables`` metadata table maps original file names to table names
    and serialises column info as pipe-delimited strings.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file (e.g. ``~/.verra/tabular.db``).
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_meta_table()

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_meta_table(self) -> None:
        """Create the metadata table if it doesn't exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _tables (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name    TEXT NOT NULL,
                table_name   TEXT NOT NULL UNIQUE,
                columns_json TEXT NOT NULL,
                row_count    INTEGER NOT NULL DEFAULT 0,
                ingested_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_csv(self, file_name: str, file_path: str, content: str) -> str:
        """Load CSV content into a SQLite table.

        If a table already exists for this *file_name* it is dropped and
        recreated so re-ingest always reflects the latest file content.

        Parameters
        ----------
        file_name:
            Original file name (e.g. ``'win_loss_2025.csv'``).
        file_path:
            Absolute path to the source file (stored for reference only).
        content:
            Raw CSV text content.

        Returns
        -------
        str
            The SQLite table name the data was loaded into.
        """
        table_name = _sanitize_table_name(file_name)

        # Parse CSV
        reader = csv.DictReader(io.StringIO(content))
        headers = reader.fieldnames or []
        if not headers:
            raise ValueError(f"CSV {file_name!r} has no headers.")

        rows = list(reader)

        # Infer column types from sample values
        col_types: dict[str, str] = {}
        for h in headers:
            sample = [r.get(h, "") for r in rows[:30]]
            col_types[h] = _infer_column_type(h, sample)

        # Sanitize column names (keep originals for display, create safe SQL names)
        safe_cols: dict[str, str] = {}  # original -> safe
        for h in headers:
            safe = re.sub(r"[^A-Za-z0-9_]", "_", h).strip("_")
            safe = re.sub(r"_+", "_", safe)
            if safe and safe[0].isdigit():
                safe = "c_" + safe
            safe_cols[h] = safe or f"col_{len(safe_cols)}"

        # Drop existing table + metadata entry if re-ingesting
        self._conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        self._conn.execute(
            "DELETE FROM _tables WHERE table_name = ?", (table_name,)
        )

        # CREATE TABLE
        col_defs = ", ".join(
            f'"{safe_cols[h]}" {col_types[h]}' for h in headers
        )
        self._conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

        # INSERT rows
        placeholders = ", ".join("?" for _ in headers)
        col_names = ", ".join(f'"{safe_cols[h]}"' for h in headers)
        insert_sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'

        def _coerce(value: str, col_type: str) -> Any:
            if col_type == "REAL":
                v = value.strip().replace(",", "").replace("%", "").replace("$", "")
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            return value

        row_data: list[tuple[Any, ...]] = []
        for row in rows:
            row_data.append(
                tuple(_coerce(row.get(h, ""), col_types[h]) for h in headers)
            )

        self._conn.executemany(insert_sql, row_data)

        # Persist metadata (columns stored as JSON-ish: "name:type|name:type")
        columns_json = "|".join(
            f"{safe_cols[h]}:{col_types[h]}" for h in headers
        )
        self._conn.execute(
            """
            INSERT INTO _tables (file_name, table_name, columns_json, row_count)
            VALUES (?, ?, ?, ?)
            """,
            (file_name, table_name, columns_json, len(row_data)),
        )
        self._conn.commit()

        return table_name

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a read-only SELECT query and return results as a list of dicts.

        Only SELECT statements are permitted.  Any attempt to run DDL or DML
        raises a ``ValueError`` before touching the database.

        Parameters
        ----------
        sql:
            A SELECT query string.
        params:
            Optional bound parameters (passed to ``cursor.execute``).

        Returns
        -------
        list[dict]
            Each row as a ``{column: value}`` dict.
        """
        stripped = sql.strip()
        if not stripped.upper().startswith("SELECT"):
            raise ValueError(
                f"Only SELECT queries are permitted. Got: {stripped[:80]!r}"
            )

        cursor = self._conn.execute(stripped, params)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def list_tables(self) -> list[dict[str, Any]]:
        """Return metadata for all ingested tables.

        Returns
        -------
        list[dict]
            Each dict has keys: ``table_name``, ``file_name``, ``row_count``,
            ``columns`` (list of ``{"name": str, "type": str}``).
        """
        try:
            rows = self._conn.execute(
                "SELECT file_name, table_name, columns_json, row_count FROM _tables ORDER BY table_name"
            ).fetchall()
        except Exception:
            return []

        result: list[dict[str, Any]] = []
        for row in rows:
            columns = _parse_columns_json(row["columns_json"])
            result.append({
                "file_name": row["file_name"],
                "table_name": row["table_name"],
                "row_count": row["row_count"],
                "columns": columns,
            })
        return result

    def get_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Return column names and types for a specific table.

        Parameters
        ----------
        table_name:
            The SQLite table name (not the original file name).

        Returns
        -------
        list[dict]
            Each dict has ``{"name": str, "type": str}``.
            Returns an empty list if the table is not found.
        """
        try:
            row = self._conn.execute(
                "SELECT columns_json FROM _tables WHERE table_name = ?",
                (table_name,),
            ).fetchone()
        except Exception:
            return []

        if row is None:
            return []
        return _parse_columns_json(row["columns_json"])

    def get_sample_rows(self, table_name: str, limit: int = 3) -> list[dict[str, Any]]:
        """Return a few sample rows from a table for schema hints."""
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "", table_name)
        try:
            rows = self._conn.execute(
                f"SELECT * FROM [{safe_name}] LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _parse_columns_json(columns_json: str) -> list[dict[str, str]]:
    """Parse the 'name:type|name:type' column encoding back to a list of dicts."""
    result: list[dict[str, str]] = []
    for part in columns_json.split("|"):
        if ":" in part:
            name, col_type = part.split(":", 1)
            result.append({"name": name, "type": col_type})
        elif part:
            result.append({"name": part, "type": "TEXT"})
    return result
