"""Workspace infrastructure tables.

Centralizes creation/migration and persistence for workspace-internal tables:
- _trace (+ _trace_seq, _view_definitions)
- _node_meta
- _workspace_meta
- _changes
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
from typing import Any, Iterable

import duckdb

from .sql_utils import (
    VIEW_TRACE_CREATE_RE,
    VIEW_TRACE_DROP_RE,
    VIEW_TRACE_NAME_EXTRACT_RE,
    get_column_schema,
)


def init_infra(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure all workspace infra tables exist."""
    ensure_trace(conn)
    ensure_node_meta(conn)
    ensure_workspace_meta(conn)
    ensure_changes(conn)


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


def ensure_trace(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the _trace table and derived views."""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS _trace_seq")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _trace (
            id INTEGER DEFAULT nextval('_trace_seq'),
            timestamp TIMESTAMP DEFAULT current_timestamp,
            node VARCHAR,
            source VARCHAR,
            query VARCHAR NOT NULL,
            success BOOLEAN NOT NULL,
            error VARCHAR,
            row_count INTEGER,
            elapsed_ms DOUBLE
        )
        """
    )

    # Derived view definitions for lineage inspection.
    conn.execute(
        rf"""
        CREATE OR REPLACE VIEW _view_definitions AS
        WITH actions AS (
            SELECT
                id,
                node,
                query,
                NULLIF(regexp_extract(query, '{VIEW_TRACE_NAME_EXTRACT_RE}', 1), '') AS view_name,
                CASE
                    WHEN regexp_matches(query, '{VIEW_TRACE_DROP_RE}') THEN 'drop'
                    ELSE 'create'
                END AS action
            FROM _trace
            WHERE success = true
              AND (
                regexp_matches(query, '{VIEW_TRACE_CREATE_RE}')
                OR regexp_matches(query, '{VIEW_TRACE_DROP_RE}')
              )
        ),
        latest AS (
            SELECT
                *,
                row_number() OVER (PARTITION BY view_name ORDER BY id DESC) AS rn
            FROM actions
            WHERE view_name IS NOT NULL
        )
        SELECT node, view_name, query AS sql
        FROM latest
        WHERE rn = 1 AND action = 'create'
        """
    )


def log_trace(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    success: bool,
    *,
    error: str | None = None,
    row_count: int | None = None,
    elapsed_ms: float | None = None,
    node_name: str | None = None,
    source: str | None = None,
) -> None:
    """Log a SQL query execution to the _trace table."""
    conn.execute(
        """
        INSERT INTO _trace (node, source, query, success, error, row_count, elapsed_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [node_name, source, query, success, error, row_count, elapsed_ms],
    )


# ---------------------------------------------------------------------------
# Node metadata
# ---------------------------------------------------------------------------


def ensure_node_meta(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _node_meta (
            node VARCHAR PRIMARY KEY,
            meta_json VARCHAR NOT NULL
        )
        """
    )


def persist_node_meta(
    conn: duckdb.DuckDBPyConnection, node_name: str, meta: dict[str, Any]
) -> None:
    """Persist per-node run metadata in _node_meta (overwrite per node)."""
    ensure_node_meta(conn)
    conn.execute("DELETE FROM _node_meta WHERE node = ?", [node_name])
    conn.execute(
        "INSERT INTO _node_meta (node, meta_json) VALUES (?, ?)",
        [node_name, json.dumps(meta, sort_keys=True)],
    )


# ---------------------------------------------------------------------------
# Workspace metadata
# ---------------------------------------------------------------------------


def ensure_workspace_meta(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _workspace_meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
        """
    )


def persist_workspace_meta(
    conn: duckdb.DuckDBPyConnection,
    model: str,
    nodes: list[Any],
    *,
    reasoning_effort: str | None = None,
    max_iterations: int | None = None,
    source_row_counts: dict[str, int] | None = None,
    spec_module: str | None = None,
) -> None:
    """Write workspace-level metadata to _workspace_meta."""
    ensure_workspace_meta(conn)
    conn.execute("DELETE FROM _workspace_meta")

    try:
        tg_version = importlib.metadata.version("taskgraph")
    except Exception:
        tg_version = "unknown"

    rows: list[tuple[str, str]] = [
        ("meta_version", "2"),
        ("created_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ("taskgraph_version", tg_version),
        ("python_version", sys.version.split()[0]),
        ("platform", platform.platform()),
        (
            "node_prompts",
            json.dumps(
                {n.name: n.prompt for n in (nodes or []) if getattr(n, "prompt", "")},
                sort_keys=True,
            ),
        ),
        ("llm_model", model),
        ("run", json.dumps({"mode": "run"}, sort_keys=True)),
    ]

    if reasoning_effort:
        rows.append(("llm_reasoning_effort", reasoning_effort))
    if max_iterations is not None:
        rows.append(("llm_max_iterations", str(max_iterations)))

    if source_row_counts:
        source_schemas = {
            table: [
                {"name": r[0], "type": r[1]} for r in get_column_schema(conn, table)
            ]
            for table in sorted(source_row_counts)
        }
        rows.append(
            ("inputs_row_counts", json.dumps(source_row_counts, sort_keys=True))
        )
        rows.append(("inputs_schema", json.dumps(source_schemas, sort_keys=True)))

    if spec_module:
        rows.append(("spec", json.dumps({"module": spec_module}, sort_keys=True)))

    conn.executemany("INSERT INTO _workspace_meta (key, value) VALUES (?, ?)", rows)


def read_workspace_meta(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Read workspace metadata. Returns empty dict if table doesn't exist."""
    try:
        return dict(conn.execute("SELECT key, value FROM _workspace_meta").fetchall())
    except duckdb.Error:
        return {}


def upsert_workspace_meta(
    conn: duckdb.DuckDBPyConnection, rows: list[tuple[str, str]]
) -> None:
    """Upsert additional workspace metadata rows."""
    ensure_workspace_meta(conn)
    conn.executemany(
        """
        INSERT INTO _workspace_meta (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        rows,
    )


# ---------------------------------------------------------------------------
# View changes
# ---------------------------------------------------------------------------


def ensure_changes(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _changes (
            node        VARCHAR,
            view_name   VARCHAR,
            kind        VARCHAR,
            sql_before  VARCHAR,
            sql_after   VARCHAR,
            cols_before VARCHAR,
            cols_after  VARCHAR,
            rows_before INTEGER,
            rows_after  INTEGER
        )
        """
    )


def persist_changes(
    conn: duckdb.DuckDBPyConnection, node_name: str, changes: list[Any]
) -> None:
    """Append view changes to the _changes table."""
    ensure_changes(conn)
    if not changes:
        return

    rows: list[tuple[Any, ...]] = []
    for c in changes:
        rows.append(
            (
                node_name,
                c.view_name,
                c.kind,
                c.sql_before,
                c.sql_after,
                json.dumps([list(t) for t in c.cols_before]) if c.cols_before else None,
                json.dumps([list(t) for t in c.cols_after]) if c.cols_after else None,
                c.rows_before,
                c.rows_after,
            )
        )

    conn.executemany(
        "INSERT INTO _changes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
