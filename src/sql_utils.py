import re
from dataclasses import dataclass
from typing import Literal

import duckdb

_parser_conn: duckdb.DuckDBPyConnection | None = None


def get_parser_conn() -> duckdb.DuckDBPyConnection:
    """Lazy singleton in-memory connection for SQL parsing."""
    global _parser_conn
    if _parser_conn is None:
        _parser_conn = duckdb.connect(":memory:")
    return _parser_conn


def split_sql_statements(sql_text: str) -> list[str]:
    """Split SQL text into individual statements using DuckDB's parser."""
    sql_text = (sql_text or "").strip()
    if not sql_text:
        return []
    try:
        statements = get_parser_conn().extract_statements(sql_text)
    except duckdb.Error:
        # Fallback for unparseable garbage: return as single statement
        return [sql_text]
    return [s.query.strip() for s in statements if s.query.strip()]


class SqlParseError(ValueError):
    """Raised when DuckDB cannot parse SQL text."""


@dataclass(frozen=True)
class ParsedStatement:
    """A single parsed SQL statement."""

    sql: str
    stmt_type: object


def parse_one_statement(sql_text: str) -> ParsedStatement:
    """Parse *sql_text* into exactly one statement.

    Enforces the same constraints as the agent SQL tool:
    - non-empty
    - exactly one statement
    """
    sql_text = (sql_text or "").strip()
    if not sql_text:
        raise SqlParseError("Empty query")

    try:
        statements = get_parser_conn().extract_statements(sql_text)
    except Exception as e:  # duckdb.Error, but keep broad for safety
        raise SqlParseError(f"SQL parse error: {e}") from e

    if not statements:
        raise SqlParseError("Empty query")
    if len(statements) > 1:
        raise SqlParseError("Only one statement allowed at a time")

    stmt = statements[0]
    return ParsedStatement(sql=stmt.query.strip(), stmt_type=stmt.type)


@dataclass(frozen=True)
class DdlTarget:
    """DDL target extracted from a CREATE/DROP statement."""

    action: Literal["create", "drop"]
    kind: Literal["view", "macro"]
    name: str | None


# Regexes to extract object name from CREATE/DROP VIEW|MACRO statements.
# These are intentionally conservative: they only extract simple identifier
# names (\w+), optionally double-quoted.

DDL_CREATE_TARGET_RE = re.compile(
    r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?"
    r"(?P<kind>VIEW|MACRO)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?"
    r'"?(?P<name>\w+)"?(?=\s|\(|$)',
    re.IGNORECASE,
)

DDL_DROP_TARGET_RE = re.compile(
    r"^\s*DROP\s+"
    r"(?P<kind>MACRO\s+TABLE|VIEW|MACRO)\s+"
    r"(?:IF\s+EXISTS\s+)?"
    r'"?(?P<name>\w+)"?(?=\s|;|$)',
    re.IGNORECASE,
)


def extract_ddl_target(sql: str) -> DdlTarget | None:
    """Extract the target from a CREATE/DROP VIEW|MACRO statement.

    Returns None if *sql* is not a CREATE/DROP of a VIEW or MACRO.
    The returned target's *name* may be None if extraction fails.
    """
    m = DDL_CREATE_TARGET_RE.match(sql or "")
    if m:
        kind = (m.group("kind") or "").upper()
        return DdlTarget(
            action="create",
            kind="view" if kind == "VIEW" else "macro",
            name=m.group("name"),
        )

    m = DDL_DROP_TARGET_RE.match(sql or "")
    if m:
        kind_raw = (m.group("kind") or "").upper()
        kind = "view" if kind_raw == "VIEW" else "macro"
        return DdlTarget(action="drop", kind=kind, name=m.group("name"))

    return None


def extract_create_name(sql: str) -> str | None:
    """Extract the name of the view or macro being created in a SQL statement.

    Kept for backward compatibility; prefer :func:`extract_ddl_target`.
    """
    target = extract_ddl_target(sql)
    if target and target.action == "create":
        return target.name
    return None


# Shared regex strings for the derived _view_definitions view.
# These are embedded into SQL via single-quoted literals.
VIEW_TRACE_CREATE_RE = r"(?i)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?"
VIEW_TRACE_DROP_RE = r"(?i)^\s*DROP\s+VIEW\s+(?:IF\s+EXISTS\s+)?"
VIEW_TRACE_NAME_EXTRACT_RE = r'(?i)VIEW\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?"?(\w+)"?'


def get_column_names(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    """Get the set of column names for a table or view."""
    rows = conn.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchall()
    return {row[0] for row in rows}


def get_column_schema(
    conn: duckdb.DuckDBPyConnection, table_name: str
) -> list[tuple[str, str]]:
    """Get the column names and data types for a table or view."""
    rows = conn.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = ?
        ORDER BY ordinal_position
        """,
        [table_name],
    ).fetchall()
    return [(row[0], row[1]) for row in rows]
