"""Shared SQL safety validation for any tool that lets an LLM write and execute SQL itself —
used by both the fixed NovaTech products retriever and the per-session uploaded-CSV retriever."""
import re
import sqlite3
from pathlib import Path

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|create|replace|vacuum)\b",
    re.IGNORECASE,
)


def validate_select(sql: str) -> None:
    """Raises ValueError unless `sql` is a single, read-only SELECT statement."""
    stripped = sql.strip().rstrip(";")
    if not re.match(r"^\s*select\b", stripped, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    if ";" in stripped:
        raise ValueError("Multiple statements are not allowed.")
    if _FORBIDDEN.search(stripped):
        raise ValueError("Query contains a forbidden keyword.")


def run_validated_select(db_path: Path, sql: str, max_rows: int = 50) -> str:
    """Validates `sql` as SELECT-only, executes it against `db_path`, and formats the results
    as a pipe-delimited table string (or an ERROR string on failure)."""
    try:
        validate_select(sql)
    except ValueError as e:
        return f"ERROR: {e}"

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql)
            rows = cur.fetchall()
    except sqlite3.Error as e:
        return f"ERROR executing query: {e}"

    if not rows:
        return "No matching rows found."

    columns = rows[0].keys()
    lines = [" | ".join(columns)]
    for row in rows[:max_rows]:
        lines.append(" | ".join(str(row[c]) for c in columns))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows truncated)")
    return "\n".join(lines)
