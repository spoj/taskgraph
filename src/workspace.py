"""Workspace orchestrator — resolve inputs, run task DAG, run exports.

A workspace is a single DuckDB database containing:
- Ingested input tables (from user-provided functions or data)
- Per-task output views (created by agents, materialized as tables after task completion)
- Metadata and trace tables

Tasks are scheduled greedily: each task starts as soon as all its
dependencies complete (not layer-by-layer). A failed task only blocks
its downstream dependents; unrelated branches continue.

After a task passes validation, its declared output views are
materialized as tables (frozen). The original SQL definitions are
preserved in the ``_view_definitions`` table for lineage queries.
Intermediate views (``{task}_*``) are left as views for debuggability.

Usage:

    workspace = Workspace(
        db_path="output.db",
        inputs={...},
        tasks=[task_prep, task_match],
        exports={"report.xlsx": write_report},
    )

    results = await workspace.run(client, model="openai/gpt-5.2")
"""

import asyncio
import json
import logging
import duckdb
import time
import sys
import platform
import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent_loop import AgentResult, DEFAULT_MAX_ITERATIONS
from .api import OpenRouterClient, DEFAULT_MODEL
from .diff import (
    snapshot_views,
    diff_snapshots,
    format_changes,
    persist_changes,
)
from .ingest import FileInput, ingest_file, ingest_table
from .agent import init_trace_table, run_task_agent, run_sql_only_task
from .task import (
    Task,
    resolve_dag,
    resolve_task_deps,
    validate_task_graph,
    validation_view_prefix,
    is_validation_view_for_task,
    validate_one_validation_view,
)
from .sql_utils import get_column_schema, split_sql_statements, extract_create_name

log = logging.getLogger(__name__)

# Type aliases
InputValue = Any  # Callable[[], TableData] | TableData | FileInput
ExportFn = Callable[[duckdb.DuckDBPyConnection, Path], None]


# --- Workspace metadata ---


def persist_workspace_meta(
    conn: duckdb.DuckDBPyConnection,
    model: str,
    tasks: list[Task],
    reasoning_effort: str | None = None,
    max_iterations: int | None = None,
    input_row_counts: dict[str, int] | None = None,
    spec_module: str | None = None,
) -> None:
    """Write workspace-level metadata to _workspace_meta.

    Contract: store enough information to make a workspace:
    - auditable (what was asked, what ran, when, with which runtime/model)
    - reproducible (spec identity)
    - debuggable (inputs overview + run parameters)

    Values are stored as strings. Most complex values are JSON.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _workspace_meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)
    conn.execute("DELETE FROM _workspace_meta")

    task_prompts = {t.name: t.prompt for t in tasks}

    created_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    input_tables = list(sorted(input_row_counts.keys())) if input_row_counts else []
    input_schemas: dict[str, list[dict[str, str]]] = {}
    if input_tables:
        for table in input_tables:
            rows_cols = get_column_schema(conn, table)
            input_schemas[table] = [{"name": r[0], "type": r[1]} for r in rows_cols]

    try:
        tg_version = importlib.metadata.version("taskgraph")
    except Exception:
        tg_version = "unknown"

    rows: list[tuple[str, str]] = [
        ("meta_version", "2"),
        ("created_at_utc", created_at_utc),
        ("taskgraph_version", tg_version),
        ("python_version", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("task_prompts", json.dumps(task_prompts, sort_keys=True)),
        ("llm_model", model),
    ]

    if reasoning_effort:
        rows.append(("llm_reasoning_effort", reasoning_effort))
    if max_iterations is not None:
        rows.append(("llm_max_iterations", str(max_iterations)))

    if input_row_counts:
        rows.append(("inputs_row_counts", json.dumps(input_row_counts, sort_keys=True)))
        rows.append(("inputs_schema", json.dumps(input_schemas, sort_keys=True)))

    run_context: dict[str, Any] = {"mode": "run"}
    rows.append(("run", json.dumps(run_context, sort_keys=True)))

    spec: dict[str, Any] = {}
    if spec_module:
        spec["module"] = spec_module
    if spec:
        rows.append(("spec", json.dumps(spec, sort_keys=True)))

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
    """Upsert additional workspace metadata rows.

    Unlike persist_workspace_meta(), this does not clear existing keys.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _workspace_meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)
    conn.executemany(
        """
        INSERT INTO _workspace_meta (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        rows,
    )


# --- View materialization ---


def _ensure_view_definitions_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the _view_definitions table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _view_definitions (
            task VARCHAR NOT NULL,
            view_name VARCHAR NOT NULL,
            sql VARCHAR NOT NULL,
            PRIMARY KEY (task, view_name)
        )
    """)


def _drop_views(conn: duckdb.DuckDBPyConnection, view_names: list[str]) -> None:
    """Drop a list of views (best-effort cleanup)."""
    for name in view_names:
        try:
            conn.execute(f'DROP VIEW IF EXISTS "{name}"')
        except duckdb.Error:
            pass


def materialize_views(
    conn: duckdb.DuckDBPyConnection,
    view_names: list[str],
    label: str,
) -> int:
    """Materialize a list of views as tables.

    Core materialization logic shared by task outputs and input validation
    views. For each view that exists:

    1. Saves the SQL definition to ``_view_definitions`` (keyed by *label*).
    2. Does a 3-step swap: CREATE TABLE from view, DROP VIEW, RENAME TABLE.

    Args:
        conn: DuckDB connection.
        view_names: View names to materialize. Duplicates are ignored.
            Missing views are silently skipped.
        label: Label stored in the ``task`` column of ``_view_definitions``
            (task name for task outputs, input table name for input validation).

    Returns the number of views materialized.
    """
    _ensure_view_definitions_table(conn)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in view_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    materialized = 0
    for view_name in unique_names:
        # Get the view's SQL definition before converting
        rows = conn.execute(
            "SELECT sql FROM duckdb_views() WHERE internal = false AND view_name = ?",
            [view_name],
        ).fetchall()
        if not rows:
            continue  # View doesn't exist (task may have failed or already materialized)

        view_sql = rows[0][0]

        # Preserve the SQL definition for lineage
        conn.execute(
            "INSERT INTO _view_definitions (task, view_name, sql) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (task, view_name) DO UPDATE SET sql = excluded.sql",
            [label, view_name, view_sql],
        )

        # Materialize: create table from view, drop view, rename table.
        # Drop any leftover tmp table from a previous crashed run, then
        # do the 3-step swap with cleanup on failure so we never leave
        # the catalog in a broken state (view gone + tmp not renamed).
        tmp_name = f"_materialize_tmp_{view_name}"
        conn.execute(f'DROP TABLE IF EXISTS "{tmp_name}"')
        conn.execute(f'CREATE TABLE "{tmp_name}" AS SELECT * FROM "{view_name}"')
        try:
            conn.execute(f'DROP VIEW "{view_name}"')
            conn.execute(f'ALTER TABLE "{tmp_name}" RENAME TO "{view_name}"')
        except duckdb.Error:
            # Swap failed — drop the tmp table so we don't leak it.
            # The original view may or may not still exist; either way
            # downstream can still query it by name.
            conn.execute(f'DROP TABLE IF EXISTS "{tmp_name}"')
            raise
        materialized += 1

    return materialized


def materialize_task_outputs(
    conn: duckdb.DuckDBPyConnection,
    task: Task,
) -> int:
    """Materialize a task's declared output views as tables.

    After a task completes and passes validation, its output views and
    validation views are converted to tables so downstream tasks read
    pre-computed data instead of re-evaluating the entire upstream view
    chain on every query.

    The original view SQL definitions are preserved in ``_view_definitions``
    for lineage and audit queries.

    What gets materialized:
    - Declared task outputs (``task.outputs``)
    - Validation views (``{task}__validation*``)

    What stays as views:
    - Intermediate ``{task}_*`` views (for debuggability)

    Returns the number of views materialized.
    """
    view_names = list(task.outputs) + task.validation_view_names()
    return materialize_views(conn, view_names, label=task.name)


@dataclass
class WorkspaceResult:
    """Aggregated results from running all tasks in a workspace."""

    success: bool  # True if ALL tasks passed validation
    task_results: dict[str, AgentResult]  # task_name -> AgentResult
    elapsed_s: float
    dag_layers: list[list[str]]  # For display: layer -> [task_names]
    export_errors: dict[str, str] = field(default_factory=dict)


@dataclass
class Workspace:
    """A multi-task workspace backed by a single DuckDB database.

    Args:
        db_path: Path for the output database (created fresh).
    inputs: Mapping of table_name -> file input, callable, or raw data.
        File inputs are parsed by the spec loader and ingested via DuckDB
        or the PDF extractor. Callables return DataFrame, list[dict], or
        dict[str, list]. Non-callables are treated as raw data directly.
        tasks: List of Task definitions forming a DAG.
        exports: Mapping of output_path -> fn(conn, path).
            Export functions run after all tasks pass. They receive the
            open database connection and the output file path. Paths
            are relative to CWD.
        input_columns: Optional column validation for input tables.
            Maps table_name -> list of required column names.
            Checked after ingestion, before tasks run.
        input_validate_sql: Optional SQL validation for input tables.
            Maps table_name -> SQL string that creates
            {input_name}__validation* views with status/message columns.
            Same contract as task validate_sql. Passing views are
            materialized as tables; failing views are dropped.
            Checked after input_columns, before tasks.
        spec_module: Module path used to load the spec.
    """

    db_path: Path | str
    inputs: dict[str, InputValue]
    tasks: list[Task]
    exports: dict[str, ExportFn] = field(default_factory=dict)
    input_columns: dict[str, list[str]] = field(default_factory=dict)
    input_validate_sql: dict[str, str] = field(default_factory=dict)
    spec_module: str | None = None

    def _validate_config(self) -> None:
        """Validate the workspace configuration before running."""
        errors = validate_task_graph(self.tasks, set(self.inputs.keys()))
        if errors:
            raise ValueError(
                "Task graph validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    async def _ingest_all(
        self, conn: duckdb.DuckDBPyConnection, client: Any | None
    ) -> dict[str, int]:
        """Resolve all inputs and ingest into the database.

        Auto-checks:
        - Callable errors are caught and re-raised with context.
        - Empty tables log a warning (not an error).

        Returns dict of table_name -> row_count.
        """
        row_counts: dict[str, int] = {}
        for table_name, value in self.inputs.items():
            if isinstance(value, FileInput):
                await ingest_file(conn, value, table_name, client=client)
            elif callable(value):
                try:
                    data = value()
                except Exception as e:
                    raise RuntimeError(
                        f"Input '{table_name}' callable failed: {e}"
                    ) from e
                ingest_table(conn, data, table_name)
            else:
                ingest_table(conn, value, table_name)
            row = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
            count = row[0] if row else 0
            row_counts[table_name] = count
            if count == 0:
                log.warning("  %s: 0 rows (empty table)", table_name)
            else:
                log.info("  %s: %d rows", table_name, count)
        return row_counts

    def _validate_inputs(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        """Validate input tables after ingestion.

        Checks run in order:
        1. input_columns: each declared table has all required columns.
        2. input_validate_sql: per-input CREATE VIEW statements producing
           {input_name}__validation* views with status/message columns.

        On success, validation views are materialized as tables (same
        pattern as task validation views) with SQL preserved in
        ``_view_definitions``. On failure, views are dropped for cleanup.

        Returns list of error messages (empty = pass).
        """
        errors: list[str] = []

        # 1. Column checks
        for table_name, required_cols in self.input_columns.items():
            try:
                actual_cols = {
                    row[0]
                    for row in conn.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = ?",
                        [table_name],
                    ).fetchall()
                }
            except duckdb.Error as e:
                errors.append(f"Schema check error for input '{table_name}': {e}")
                continue

            if not actual_cols:
                errors.append(f"Input table '{table_name}' not found.")
                continue

            missing = [c for c in required_cols if c not in actual_cols]
            if missing:
                errors.append(
                    f"Input '{table_name}' is missing required column(s): "
                    f"{', '.join(missing)}. "
                    f"Actual columns: {', '.join(sorted(actual_cols - {'_row_id'}))}"
                )

        if errors:
            return errors

        # 2. Per-input validation views (same contract as task validation)
        # Track views per input for materialization labeling
        all_created_views: list[str] = []  # flat list for cleanup on error
        per_input_views: dict[str, list[str]] = {}  # input_name -> view names

        for table_name, validate_sql in self.input_validate_sql.items():
            stmts = split_sql_statements(validate_sql)
            input_views: list[str] = []

            # Execute all statements (CREATE VIEW etc.)
            for sql in stmts:
                try:
                    conn.execute(sql)
                except duckdb.Error as e:
                    errors.append(f"Input validation SQL error for '{table_name}': {e}")
                    # Clean up any views created so far
                    _drop_views(conn, all_created_views)
                    return errors

            # Discover created validation views for this input
            prefix = validation_view_prefix(table_name)
            for stmt in stmts:
                name = extract_create_name(stmt)
                if name and is_validation_view_for_task(name, table_name):
                    input_views.append(name)
                    all_created_views.append(name)

            if not input_views:
                errors.append(
                    f"Input '{table_name}' validate_sql did not create any "
                    f"'{prefix}' views."
                )
                _drop_views(conn, all_created_views)
                return errors

            per_input_views[table_name] = input_views

            # Validate the views
            for view_name in input_views:
                view_errors, fatal = validate_one_validation_view(conn, view_name)
                if view_errors:
                    errors.extend(view_errors)
                if fatal:
                    _drop_views(conn, all_created_views)
                    return errors

            if errors:
                _drop_views(conn, all_created_views)
                return errors

        # All validation passed — materialize input validation views
        for table_name, view_names in per_input_views.items():
            n = materialize_views(conn, view_names, label=table_name)
            if n:
                log.debug(
                    "Materialized %d input validation view(s) for '%s'",
                    n,
                    table_name,
                )

        return errors

    @staticmethod
    async def _run_task_dag(
        tasks: list[Task],
        run_one: Callable[[Task], Any],
    ) -> tuple[dict[str, AgentResult], bool]:
        """Schedule tasks as soon as their dependencies are met.

        Args:
            tasks: All tasks in the DAG.
            run_one: Async callable (task) -> (task_name, AgentResult).

        Returns:
            (task_results dict, all_success bool)
        """
        task_by_name = {t.name: t for t in tasks}
        deps = resolve_task_deps(tasks)
        task_results: dict[str, AgentResult] = {}
        all_success = True
        done: set[str] = set()
        failed: set[str] = set()
        running: set[str] = set()
        pending = set(task_by_name.keys())
        active: set[asyncio.Task] = set()

        def launch_ready() -> None:
            newly_launched = []
            for name in list(pending):
                if name in running:
                    continue
                if deps[name] & failed:
                    continue
                if deps[name] <= done:
                    running.add(name)
                    t = asyncio.create_task(run_one(task_by_name[name]))
                    active.add(t)
                    newly_launched.append(name)
            for name in newly_launched:
                pending.discard(name)

        launch_ready()

        while active:
            finished, active = await asyncio.wait(
                active, return_when=asyncio.FIRST_COMPLETED
            )
            for fut in finished:
                exc = fut.exception()
                if exc is not None:
                    raise exc
                name, result = fut.result()
                task_results[name] = result
                done.add(name)
                running.discard(name)
                if not result.success:
                    all_success = False
                    failed.add(name)
                    log.warning("[%s] FAILED", name)

            launch_ready()

        blocked = pending - running
        if blocked:
            log.warning(
                "Skipped (blocked by failed deps): %s", ", ".join(sorted(blocked))
            )

        return task_results, all_success

    def _run_exports(self, conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
        """Run export functions. Returns dict of name -> error for failures."""
        errors: dict[str, str] = {}
        for output_path, export_fn in self.exports.items():
            try:
                export_fn(conn, Path(output_path))
                log.info("  %s: OK", output_path)
            except Exception as e:
                errors[output_path] = str(e)
                log.error("  %s: FAILED (%s)", output_path, e)
        return errors

    async def _execute_task(
        self,
        conn: duckdb.DuckDBPyConnection,
        task: Task,
        client: OpenRouterClient | None,
        model: str,
        max_iterations: int,
    ) -> AgentResult:
        """Execute a single task (SQL or LLM) and return its result.

        For LLM tasks, both structural and SQL validation run inside the
        agent loop — the agent gets feedback and self-corrects within its
        iteration budget.

        For SQL-only tasks, validation runs once after execution (no agent
        to retry).
        """
        mode = task.transform_mode()

        if mode == "sql":
            return await run_sql_only_task(conn=conn, task=task)
        elif mode == "prompt":
            if client is None:
                raise RuntimeError(f"Task '{task.name}' requires an OpenRouterClient")
            return await run_task_agent(
                conn=conn,
                task=task,
                client=client,
                model=model,
                max_iterations=max_iterations,
            )
        else:
            raise RuntimeError(f"Unknown task mode: {mode}")

    async def run(
        self,
        client: OpenRouterClient | None = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> WorkspaceResult:
        """Run the full workspace: ingest, resolve DAG, execute tasks, export.

        Each task starts as soon as all its dependencies complete
        (cooperative async — DuckDB access naturally serialized).
        If a task fails, only its downstream dependents are blocked.
        Export functions run after all tasks pass.

        Returns:
            WorkspaceResult with per-task results and overall status.
        """
        start_time = time.time()
        self._validate_config()

        layers = resolve_dag(self.tasks)
        dag_layer_names = [[t.name for t in layer] for layer in layers]

        # Workspace task display is now handled by the CLI for tree formatting.
        # This keeps the workspace focused on execution.

        # Create fresh database
        db_path = Path(self.db_path)
        if db_path.exists():
            db_path.unlink()
        conn = duckdb.connect(str(db_path))
        try:
            init_trace_table(conn)

            log.info("Ingesting %d input(s)...", len(self.inputs))
            input_row_counts = await self._ingest_all(conn, client=client)

            # Persist workspace metadata (after ingestion so we have row counts)
            persist_workspace_meta(
                conn,
                model,
                tasks=self.tasks,
                reasoning_effort=client.reasoning_effort if client else None,
                max_iterations=max_iterations,
                input_row_counts=input_row_counts,
                spec_module=self.spec_module,
            )

            # Validate inputs before running tasks
            if self.input_columns or self.input_validate_sql:
                log.info("Validating inputs...")
                input_errors = self._validate_inputs(conn)
                if input_errors:
                    elapsed_s = time.time() - start_time
                    log.error("Input validation failed:")
                    for e in input_errors:
                        log.error("  - %s", e)
                    raise ValueError(
                        "Input validation failed:\n"
                        + "\n".join(f"  - {e}" for e in input_errors)
                    )

            async def run_one(task: Task) -> tuple[str, AgentResult]:
                log.info(
                    "[%s] Starting (outputs: %s)",
                    task.name,
                    ", ".join(task.outputs),
                )
                before = snapshot_views(conn)
                result = await self._execute_task(
                    conn=conn,
                    task=task,
                    client=client,
                    model=model,
                    max_iterations=max_iterations,
                )
                after = snapshot_views(conn)
                changes = diff_snapshots(before, after)
                if changes:
                    persist_changes(conn, task.name, changes)
                    change_summary = format_changes(task.name, changes)
                    if change_summary:
                        log.info("%s", change_summary)
                # Materialize successful task outputs as tables so downstream
                # tasks read pre-computed data instead of re-evaluating view chains.
                if result.success:
                    n = materialize_task_outputs(conn, task)
                    if n:
                        log.debug("[%s] Materialized %d output(s)", task.name, n)
                return task.name, result

            task_results, all_success = await self._run_task_dag(self.tasks, run_one)

            # Run exports if all tasks passed
            export_errors: dict[str, str] = {}
            if all_success and self.exports:
                log.info("--- Exports (%d) ---", len(self.exports))
                export_errors = self._run_exports(conn)

            # Persist export results for later inspection
            if self.exports:
                results: dict[str, dict[str, Any]] = {}
                for name in sorted(self.exports.keys()):
                    if name in export_errors:
                        results[name] = {"ok": False, "error": export_errors[name]}
                    else:
                        results[name] = {"ok": bool(all_success), "error": None}
                exports_meta = {
                    "attempted": bool(all_success),
                    "results": results,
                }
                upsert_workspace_meta(
                    conn,
                    [("exports", json.dumps(exports_meta, sort_keys=True))],
                )

            elapsed_s = time.time() - start_time

            status = "ALL PASSED" if all_success else "SOME FAILED"
            log.info("--- Workspace complete: %s (%.1fs) ---", status, elapsed_s)
            for name, result in task_results.items():
                s = "PASS" if result.success else "FAIL"
                tokens = (
                    result.usage["prompt_tokens"] + result.usage["completion_tokens"]
                )
                log.info(
                    "  %s: %s (%d iters, %d tools, %d tokens)",
                    name,
                    s,
                    result.iterations,
                    result.tool_calls_count,
                    tokens,
                )

            return WorkspaceResult(
                success=all_success and not export_errors,
                task_results=task_results,
                elapsed_s=elapsed_s,
                dag_layers=dag_layer_names,
                export_errors=export_errors,
            )
        finally:
            conn.close()
