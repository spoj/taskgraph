"""Workspace orchestrator — resolve inputs, run task DAG, run exports.

A workspace is a single DuckDB database containing:
- Ingested input tables (from user-provided functions or data)
- Per-task output views (created by agents)
- Metadata and trace tables

Tasks are scheduled greedily: each task starts as soon as all its
dependencies complete (not layer-by-layer). A failed task only blocks
its downstream dependents; unrelated branches continue.

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

from .agent_loop import AgentResult
from .api import OpenRouterClient
from .diff import (
    snapshot_views,
    diff_snapshots,
    format_changes,
    persist_changes,
    ViewChange,
)
from .ingest import ingest_table
from .agent import run_task_agent, run_sql_only_task
from .task import Task, resolve_dag, resolve_task_deps, validate_task_graph

log = logging.getLogger(__name__)

# Type aliases
InputValue = Any  # Callable[[], TableData] | TableData
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

    task_intents = {t.name: t.intent for t in tasks}

    created_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    input_tables = list(sorted(input_row_counts.keys())) if input_row_counts else []
    input_schemas: dict[str, list[dict[str, str]]] = {}
    if input_tables:
        for table in input_tables:
            try:
                rows_cols = conn.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = ?
                    ORDER BY ordinal_position
                    """,
                    [table],
                ).fetchall()
                input_schemas[table] = [{"name": r[0], "type": r[1]} for r in rows_cols]
            except duckdb.Error:
                input_schemas[table] = []

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
        ("task_intents", json.dumps(task_intents, sort_keys=True)),
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
        inputs: Mapping of table_name -> callable or raw data.
            If callable, it is called at ingest time and must return
            a DataFrame, list[dict], or dict[str, list].
            If not callable, it is treated as raw data directly.
        tasks: List of Task definitions forming a DAG.
        exports: Mapping of output_path -> fn(conn, path).
            Export functions run after all tasks pass. They receive the
            open database connection and the output file path. Paths
            are relative to CWD.
        input_columns: Optional column validation for input tables.
            Maps table_name -> list of required column names.
            Checked after ingestion, before tasks run.
        input_validate_sql: Optional SQL validation for input tables.
            Maps table_name -> list of SQL queries that must each
            return zero rows. Checked after ingestion and input_columns,
            before tasks run.
        spec_module: Module path used to load the spec.
    """

    db_path: Path | str
    inputs: dict[str, InputValue]
    tasks: list[Task]
    exports: dict[str, ExportFn] = field(default_factory=dict)
    input_columns: dict[str, list[str]] = field(default_factory=dict)
    input_validate_sql: dict[str, list[str]] = field(default_factory=dict)
    spec_module: str | None = None

    def _validate_config(self) -> None:
        """Validate the workspace configuration before running."""
        errors = validate_task_graph(self.tasks, set(self.inputs.keys()))
        if errors:
            raise ValueError(
                "Task graph validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def _ingest_all(self, conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
        """Resolve all inputs and ingest into the database.

        Auto-checks:
        - Callable errors are caught and re-raised with context.
        - Empty tables log a warning (not an error).

        Returns dict of table_name -> row_count.
        """
        row_counts: dict[str, int] = {}
        for table_name, value in self.inputs.items():
            if callable(value):
                try:
                    data = value()
                except Exception as e:
                    raise RuntimeError(
                        f"Input '{table_name}' callable failed: {e}"
                    ) from e
            else:
                data = value
            ingest_table(conn, data, table_name)
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
        2. input_validate_sql: per-input SQL queries must return zero rows.

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

        # 2. Per-input SQL validation checks
        for table_name, sql_checks in self.input_validate_sql.items():
            for sql in sql_checks:
                try:
                    cursor = conn.execute(sql)
                    rows = cursor.fetchall()
                except duckdb.Error as e:
                    errors.append(
                        f"Input validation query error for '{table_name}': {e}"
                    )
                    return errors
                if rows:
                    cols = [d[0] for d in cursor.description]
                    for row in rows:
                        if len(cols) == 1:
                            errors.append(str(row[0]))
                        else:
                            errors.append(
                                ", ".join(f"{c}={v}" for c, v in zip(cols, row))
                            )
                    return errors

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
        client: OpenRouterClient,
        model: str,
        max_iterations: int,
    ) -> AgentResult:
        """Execute a single task (SQL or LLM) and return its result."""
        mode = task.run_mode()
        if mode == "sql_strict":
            return await run_sql_only_task(conn=conn, task=task)

        if mode == "sql":
            result = await run_sql_only_task(conn=conn, task=task)
            if result.success:
                if task.repair_on_warn:
                    warnings = task.validation_warnings(conn)
                    if warnings:
                        log.warning(
                            "[%s] Validation warnings; attempting LLM repair",
                            task.name,
                        )
                        issue = "Warnings:\n" + "\n".join(f"- {w}" for w in warnings)
                        return await run_task_agent(
                            conn=conn,
                            task=task,
                            client=client,
                            model=model,
                            max_iterations=max_iterations,
                            issue=issue,
                        )
                return result

            log.warning("[%s] SQL failed; attempting LLM repair", task.name)
            return await run_task_agent(
                conn=conn,
                task=task,
                client=client,
                model=model,
                max_iterations=max_iterations,
                issue=result.final_message,
            )

        raise RuntimeError(f"Unknown task mode: {mode}")

    async def run(
        self,
        client: OpenRouterClient,
        model: str = "openai/gpt-5.2",
        max_iterations: int = 200,
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

        log.info("Workspace: %d tasks, %d DAG layers", len(self.tasks), len(layers))
        for i, layer_names in enumerate(dag_layer_names):
            log.info("  Layer %d: %s", i, ", ".join(layer_names))

        # Create fresh database
        db_path = Path(self.db_path)
        if db_path.exists():
            db_path.unlink()
        conn = duckdb.connect(str(db_path))

        log.info("Ingesting %d input(s)...", len(self.inputs))
        input_row_counts = self._ingest_all(conn)

        # Persist workspace metadata (after ingestion so we have row counts)
        persist_workspace_meta(
            conn,
            model,
            tasks=self.tasks,
            reasoning_effort=client.reasoning_effort,
            max_iterations=max_iterations,
            input_row_counts=input_row_counts,
            spec_module=self.spec_module,
        )

        # Validate inputs before running tasks
        if self.input_columns or self.input_validate_sql:
            log.info("Validating inputs...")
            input_errors = self._validate_inputs(conn)
            if input_errors:
                conn.close()
                elapsed_s = time.time() - start_time
                log.error("Input validation failed:")
                for e in input_errors:
                    log.error("  - %s", e)
                raise ValueError(
                    "Input validation failed:\n"
                    + "\n".join(f"  - {e}" for e in input_errors)
                )

        # Accumulate per-task view changes for reporting
        task_changes: list[tuple[str, list[ViewChange]]] = []

        async def run_one(task: Task) -> tuple[str, AgentResult]:
            log.info("[%s] Starting", task.name)
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
                task_changes.append((task.name, changes))
                change_summary = format_changes(task.name, changes)
                if change_summary:
                    log.info("%s", change_summary)
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

        conn.close()
        elapsed_s = time.time() - start_time

        status = "ALL PASSED" if all_success else "SOME FAILED"
        log.info("--- Workspace complete: %s (%.1fs) ---", status, elapsed_s)
        for name, result in task_results.items():
            s = "PASS" if result.success else "FAIL"
            tokens = result.usage["prompt_tokens"] + result.usage["completion_tokens"]
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
