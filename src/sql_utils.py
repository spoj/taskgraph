import re
import duckdb
from typing import Any

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


# Regex to extract view/macro name from CREATE [OR REPLACE] [TEMP] VIEW/MACRO <name> ...
CREATE_NAME_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?"
    r"(?:VIEW|MACRO)\s+"
    r"\"?(\w+)\"?",
    re.IGNORECASE,
)


def extract_create_name(sql: str) -> str | None:
    """Extract the name of the view or macro being created in a SQL statement."""
    match = CREATE_NAME_RE.search(sql)
    return match.group(1) if match else None


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
