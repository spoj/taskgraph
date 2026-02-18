"""DuckDB catalog helpers.

These helpers centralize common patterns for querying DuckDB's catalog
(duckdb_views/duckdb_tables) and safely counting rows.
"""

from __future__ import annotations

from typing import Iterable

import duckdb


def quote_ident(name: str) -> str:
    """Quote an identifier for DuckDB SQL."""
    return '"' + name.replace('"', '""') + '"'


def _excluded(name: str, exclude_prefixes: Iterable[str]) -> bool:
    for p in exclude_prefixes:
        if p and name.startswith(p):
            return True
    return False


def list_views(
    conn: duckdb.DuckDBPyConnection,
    *,
    include_internal: bool = False,
    exclude_prefixes: Iterable[str] = (),
) -> list[str]:
    """Return view names from the catalog."""
    where = "" if include_internal else "WHERE internal = false"
    rows = conn.execute(
        f"SELECT view_name FROM duckdb_views() {where} ORDER BY view_name"
    ).fetchall()
    names = [r[0] for r in rows]
    if exclude_prefixes:
        names = [n for n in names if not _excluded(n, exclude_prefixes)]
    return names


def list_views_with_sql(
    conn: duckdb.DuckDBPyConnection,
    *,
    include_internal: bool = False,
    exclude_prefixes: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """Return (view_name, sql) rows from the catalog."""
    where = "" if include_internal else "WHERE internal = false"
    rows = conn.execute(
        f"SELECT view_name, sql FROM duckdb_views() {where} ORDER BY view_name"
    ).fetchall()
    out: list[tuple[str, str]] = []
    for view_name, sql in rows:
        if exclude_prefixes and _excluded(view_name, exclude_prefixes):
            continue
        out.append((view_name, sql or ""))
    return out


def view_exists(
    conn: duckdb.DuckDBPyConnection,
    view_name: str,
    *,
    include_internal: bool = False,
) -> bool:
    """Return True if a view exists."""
    where = "" if include_internal else "AND internal = false"
    rows = conn.execute(
        f"SELECT 1 FROM duckdb_views() WHERE view_name = ? {where}",
        [view_name],
    ).fetchall()
    return bool(rows)


def list_tables(
    conn: duckdb.DuckDBPyConnection,
    *,
    include_internal: bool = False,
    exclude_prefixes: Iterable[str] = (),
) -> list[str]:
    """Return table names from the catalog."""
    where = "" if include_internal else "WHERE internal = false"
    rows = conn.execute(
        f"SELECT table_name FROM duckdb_tables() {where} ORDER BY table_name"
    ).fetchall()
    names = [r[0] for r in rows]
    if exclude_prefixes:
        names = [n for n in names if not _excluded(n, exclude_prefixes)]
    return names


def count_rows(conn: duckdb.DuckDBPyConnection, relation_name: str) -> int | None:
    """Return COUNT(*) for a table/view, or None on error."""
    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {quote_ident(relation_name)}"
        ).fetchone()
    except duckdb.Error:
        return None
    if not row:
        return 0
    try:
        return int(row[0])
    except Exception:
        return None


def count_rows_display(conn: duckdb.DuckDBPyConnection, relation_name: str) -> str:
    """Return a display-friendly row count (or 'error')."""
    n = count_rows(conn, relation_name)
    return str(n) if n is not None else "error"
