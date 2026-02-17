"""View catalog diffing for change reporting.

Captures before/after snapshots of DuckDB view definitions and produces
structured diffs showing what each task changed: new views, dropped views,
and modified view SQL definitions.

The diff operates on SQL definitions stored in duckdb_views(), not on data.
This makes it cheap (no table scans) and gives precise visibility into what
the agent actually wrote.
"""

import difflib
import json
import logging
import duckdb
from dataclasses import dataclass
from .sql_utils import get_column_schema

log = logging.getLogger(__name__)

# Maximum changed lines to show inline per modified view
MAX_DIFF_LINES = 10


@dataclass
class ViewSnapshot:
    """Snapshot of a single view's definition and metadata."""

    name: str
    sql: str  # Full CREATE VIEW ... AS ... from duckdb_views()
    columns: list[tuple[str, str]]  # [(col_name, data_type), ...]
    row_count: int


@dataclass
class ViewChange:
    """A single view change detected between two catalog snapshots."""

    view_name: str
    kind: str  # 'created' | 'dropped' | 'modified'
    sql_before: str | None  # None for created
    sql_after: str | None  # None for dropped
    cols_before: list[tuple[str, str]] | None  # None for created
    cols_after: list[tuple[str, str]] | None  # None for dropped
    rows_before: int | None  # None for created
    rows_after: int | None  # None for dropped


def snapshot_views(conn: duckdb.DuckDBPyConnection) -> dict[str, ViewSnapshot]:
    """Snapshot all user view definitions, schemas, and row counts.

    Excludes ``_``-prefixed infrastructure views (e.g. ``_view_definitions``)
    which are internal to the workspace machinery.
    """
    snapshots: dict[str, ViewSnapshot] = {}

    try:
        views = conn.execute(
            "SELECT view_name, sql FROM duckdb_views() "
            "WHERE internal = false AND view_name[1] != '_'"
        ).fetchall()
    except duckdb.Error:
        return snapshots

    for view_name, sql in views:
        # Get column info
        columns = get_column_schema(conn, view_name)

        # Get row count (cheap — DuckDB optimizes COUNT(*) on views)
        try:
            row = conn.execute(f'SELECT COUNT(*) FROM "{view_name}"').fetchone()
            row_count = int(row[0]) if row else 0
        except duckdb.Error:
            row_count = 0

        snapshots[view_name] = ViewSnapshot(
            name=view_name,
            sql=sql or "",
            columns=columns,
            row_count=row_count,
        )

    return snapshots


def diff_snapshots(
    before: dict[str, ViewSnapshot],
    after: dict[str, ViewSnapshot],
) -> list[ViewChange]:
    """Diff two catalog snapshots. Returns only changed views.

    Compares SQL definitions to detect modifications. Views with
    identical SQL are omitted from the result.
    """
    changes: list[ViewChange] = []

    all_names = sorted(set(before.keys()) | set(after.keys()))

    for name in all_names:
        b = before.get(name)
        a = after.get(name)

        if b is None and a is not None:
            # Created
            changes.append(
                ViewChange(
                    view_name=name,
                    kind="created",
                    sql_before=None,
                    sql_after=a.sql,
                    cols_before=None,
                    cols_after=a.columns,
                    rows_before=None,
                    rows_after=a.row_count,
                )
            )
        elif b is not None and a is None:
            # Dropped
            changes.append(
                ViewChange(
                    view_name=name,
                    kind="dropped",
                    sql_before=b.sql,
                    sql_after=None,
                    cols_before=b.columns,
                    cols_after=None,
                    rows_before=b.row_count,
                    rows_after=None,
                )
            )
        elif b is not None and a is not None:
            # Check if SQL changed
            if _normalize_sql(b.sql) != _normalize_sql(a.sql):
                changes.append(
                    ViewChange(
                        view_name=name,
                        kind="modified",
                        sql_before=b.sql,
                        sql_after=a.sql,
                        cols_before=b.columns,
                        cols_after=a.columns,
                        rows_before=b.row_count,
                        rows_after=a.row_count,
                    )
                )

    return changes


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: strip whitespace, collapse spaces."""
    return " ".join(sql.split())


def _compact_sql_diff(sql_before: str, sql_after: str) -> list[str]:
    """Produce a compact diff of two SQL definitions.

    Returns only the changed lines (prefixed with - / +), capped at
    MAX_DIFF_LINES. If more lines changed, appends a summary.
    """
    before_lines = sql_before.splitlines(keepends=False)
    after_lines = sql_after.splitlines(keepends=False)

    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            lineterm="",
        )
    )

    # Extract only +/- lines (skip --- +++ and @@ headers)
    changed: list[str] = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            changed.append(line)

    if len(changed) > MAX_DIFF_LINES:
        truncated = changed[:MAX_DIFF_LINES]
        remaining = len(changed) - MAX_DIFF_LINES
        truncated.append(f"  ... {remaining} more lines changed")
        return truncated

    return changed


def _fmt_cols(cols: list[tuple[str, str]]) -> str:
    """Format column count."""
    n = len(cols)
    return f"{n} col{'s' if n != 1 else ''}"


def _fmt_rows(count: int) -> str:
    """Format row count."""
    return f"{count} row{'s' if count != 1 else ''}"


def format_changes(task_name: str, changes: list[ViewChange]) -> str:
    """Format task changes for terminal display.

    Returns a multi-line string ready for click.echo(). Returns empty
    string if there are no changes.
    """
    if not changes:
        return ""

    lines: list[str] = []
    lines.append(f"  {task_name}:")

    for c in changes:
        if c.kind == "created":
            cols_s = _fmt_cols(c.cols_after or [])
            rows_s = _fmt_rows(c.rows_after or 0)
            lines.append(f"    + {c.view_name:<24s} {cols_s}, {rows_s}")

        elif c.kind == "dropped":
            lines.append(f"    - {c.view_name}")

        elif c.kind == "modified":
            # Schema change summary
            parts: list[str] = []
            if c.cols_before is not None and c.cols_after is not None:
                before_names = {n for n, _ in c.cols_before}
                after_names = {n for n, _ in c.cols_after}
                added = sorted(after_names - before_names)
                dropped = sorted(before_names - after_names)
                if added or dropped:
                    col_parts: list[str] = []
                    if added:
                        col_parts.append(f"+{', '.join(added)}")
                    if dropped:
                        col_parts.append(f"-{', '.join(dropped)}")
                    parts.append(f"cols: {' '.join(col_parts)}")
                elif len(c.cols_before) != len(c.cols_after):
                    parts.append(
                        f"{_fmt_cols(c.cols_before)} -> {_fmt_cols(c.cols_after)}"
                    )

            if c.rows_before is not None and c.rows_after is not None:
                if c.rows_before != c.rows_after:
                    parts.append(
                        f"{_fmt_rows(c.rows_before)} -> {_fmt_rows(c.rows_after)}"
                    )

            meta = f"  {', '.join(parts)}" if parts else ""
            lines.append(f"    ~ {c.view_name:<24s}{meta}")

            # Compact SQL diff
            if c.sql_before and c.sql_after:
                diff_lines = _compact_sql_diff(c.sql_before, c.sql_after)
                for dl in diff_lines:
                    lines.append(f"      {dl}")

    return "\n".join(lines)


def format_all_changes(task_changes: list[tuple[str, list[ViewChange]]]) -> str:
    """Format changes for all tasks into a single terminal block.

    Args:
        task_changes: List of (task_name, changes) pairs in execution order.

    Returns formatted string, or empty string if no changes anywhere.
    """
    sections: list[str] = []
    for task_name, changes in task_changes:
        section = format_changes(task_name, changes)
        if section:
            sections.append(section)

    if not sections:
        return ""

    return "Changes:\n" + "\n".join(sections)


def persist_changes(
    conn: duckdb.DuckDBPyConnection,
    task_name: str,
    changes: list[ViewChange],
) -> None:
    """Write view changes to the _changes table in the output database."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _changes (
            task        VARCHAR,
            view_name   VARCHAR,
            kind        VARCHAR,
            sql_before  VARCHAR,
            sql_after   VARCHAR,
            cols_before VARCHAR,
            cols_after  VARCHAR,
            rows_before INTEGER,
            rows_after  INTEGER
        )
    """)

    if not changes:
        return

    rows = []
    for c in changes:
        rows.append(
            (
                task_name,
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
