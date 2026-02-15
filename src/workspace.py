"""Workspace orchestrator — resolve inputs, run task DAG, run exports.

A workspace is a single DuckDB database containing:
- Ingested input tables (from user-provided functions or data)
- Per-task output views (created by agents)
- Metadata and trace tables
 - Optional spec source (for audit)

 The .db file stores resolved prompts, structural fingerprint, and all data.
 Spec source may be stored for audit, but reruns are driven by the
 spec module reference captured in workspace metadata.

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

    # Fresh run from spec
    results = await workspace.run(client, model="openai/gpt-5.2")

    # Rerun from a previous .db (re-ingest + re-validate)
    results = await workspace.rerun(
        source_db="previous.db", client=client, model="openai/gpt-5.2",
    )
"""

import asyncio
import json
import logging
import shutil
import duckdb
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent_loop import AgentResult
from .api import OpenRouterClient
from .ingest import ingest_table, get_schema_info_for_tables
from .agent import run_task_agent
from .task import Task, resolve_dag, resolve_task_deps, validate_task_graph

log = logging.getLogger(__name__)

# Type aliases
InputValue = Any  # Callable[[], TableData] | TableData
ExportFn = Callable[[duckdb.DuckDBPyConnection, Path], None]


# --- Workspace metadata ---


def _build_workspace_fingerprint(
    inputs: dict[str, Any],
    tasks: list[Task],
    input_columns: dict[str, list[str]],
) -> dict[str, Any]:
    """Build a fingerprint capturing the workspace spec's structural identity.

    This fingerprint is stored in the .db file so that a rerun can verify
    structural compatibility. Two specs are compatible when their
    fingerprints match.

    The fingerprint deliberately excludes:
    - Prompt text (can be refined between runs)
    - Export definitions (don't affect data)
    - validate_sql text (can be tightened)
    - Input data values (will be re-ingested)
    """
    return {
        "inputs": {
            name: {
                "columns": sorted(input_columns.get(name, [])),
            }
            for name in sorted(inputs.keys())
        },
        "tasks": [
            {
                "name": t.name,
                "inputs": sorted(t.inputs),
                "outputs": sorted(t.outputs),
                "output_columns": {
                    k: sorted(v) for k, v in sorted(t.output_columns.items())
                },
            }
            for t in sorted(tasks, key=lambda t: t.name)
        ],
    }


def persist_workspace_meta(
    conn: duckdb.DuckDBPyConnection,
    fingerprint: dict[str, Any],
    model: str,
    tasks: list[Task],
    input_row_counts: dict[str, int] | None = None,
    source_db: str | None = None,
    rerun_mode: str | None = None,
    spec_source: str | None = None,
    spec_module: str | None = None,
    spec_git_commit: str | None = None,
    spec_git_root: str | None = None,
    spec_git_dirty: bool | None = None,
) -> None:
    """Write workspace-level metadata to _workspace_meta table.

    Stores everything needed for auditability and spec-free reruns:
    - fingerprint: structural identity for rerun compatibility
    - prompts: resolved prompt text per task (handles Path-based prompts)
    - spec_source: raw Python source of the spec module (if available)
    - spec_module: module path used to load the spec
    - spec_git_commit: git commit hash for the spec repo
    - spec_git_root: git repo root for the spec module
    - spec_git_dirty: whether the spec repo had uncommitted changes
    - run context: model, timestamp
    - input_row_counts: row counts per input table
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _workspace_meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
    """)
    conn.execute("DELETE FROM _workspace_meta")

    # Resolved prompts per task — the only thing not recoverable from
    # spec_source (Path-based prompts reference files that may not exist
    # at rerun time).
    prompts = {t.name: t.prompt for t in tasks}

    rows: list[tuple[str, str]] = [
        ("fingerprint", json.dumps(fingerprint, sort_keys=True)),
        ("prompts", json.dumps(prompts)),
        ("model", model),
        ("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
    ]

    if input_row_counts:
        rows.append(("input_row_counts", json.dumps(input_row_counts, sort_keys=True)))

    if source_db:
        rows.append(("source_db", source_db))
    if rerun_mode:
        rows.append(("rerun_mode", rerun_mode))

    if spec_source:
        rows.append(("spec_source", spec_source))

    if spec_module:
        rows.append(("spec_module", spec_module))
    if spec_git_commit:
        rows.append(("spec_git_commit", spec_git_commit))
    if spec_git_root:
        rows.append(("spec_git_root", spec_git_root))
    if spec_git_dirty is not None:
        rows.append(("spec_git_dirty", "true" if spec_git_dirty else "false"))

    conn.executemany("INSERT INTO _workspace_meta (key, value) VALUES (?, ?)", rows)


def read_workspace_meta(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Read workspace metadata. Returns empty dict if table doesn't exist."""
    try:
        return dict(conn.execute("SELECT key, value FROM _workspace_meta").fetchall())
    except duckdb.Error:
        return {}


def check_fingerprint_compatibility(
    source_conn: duckdb.DuckDBPyConnection,
    current_fingerprint: dict[str, Any],
) -> list[str]:
    """Check if a source .db is structurally compatible with the current spec.

    Returns list of incompatibility messages (empty = compatible).
    """
    meta = read_workspace_meta(source_conn)
    if not meta:
        return ["Database has no _workspace_meta table (not a Taskgraph workspace)."]

    stored_fp_str = meta.get("fingerprint")
    if not stored_fp_str:
        return ["Database has no fingerprint in _workspace_meta."]

    stored_fp = json.loads(stored_fp_str)
    current_fp_str = json.dumps(current_fingerprint, sort_keys=True)

    if stored_fp_str == current_fp_str:
        return []

    # Produce specific diffs
    errors: list[str] = []

    # Compare inputs
    stored_inputs = set(stored_fp.get("inputs", {}).keys())
    current_inputs = set(current_fingerprint.get("inputs", {}).keys())
    for name in sorted(current_inputs - stored_inputs):
        errors.append(f"Input '{name}' not in source db.")
    for name in sorted(stored_inputs - current_inputs):
        errors.append(f"Source db has extra input '{name}' not in current spec.")

    # Compare tasks
    stored_tasks = {t["name"]: t for t in stored_fp.get("tasks", [])}
    current_tasks = {t["name"]: t for t in current_fingerprint.get("tasks", [])}
    for name in sorted(current_tasks.keys() | stored_tasks.keys()):
        if name not in stored_tasks:
            errors.append(f"Task '{name}' not in source db.")
        elif name not in current_tasks:
            errors.append(f"Source db has extra task '{name}' not in current spec.")
        else:
            st, ct = stored_tasks[name], current_tasks[name]
            if st["inputs"] != ct["inputs"]:
                errors.append(
                    f"Task '{name}' inputs differ: "
                    f"source={st['inputs']}, current={ct['inputs']}"
                )
            if st["outputs"] != ct["outputs"]:
                errors.append(
                    f"Task '{name}' outputs differ: "
                    f"source={st['outputs']}, current={ct['outputs']}"
                )
            if st["output_columns"] != ct["output_columns"]:
                errors.append(f"Task '{name}' output_columns differ.")

    return errors


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
        spec_source: Raw Python source of the spec module (optional).
        spec_module: Module path used to load the spec.
        spec_git_commit: Git commit hash for the spec repo.
        spec_git_root: Git repo root for the spec module.
    """

    db_path: Path | str
    inputs: dict[str, InputValue]
    tasks: list[Task]
    exports: dict[str, ExportFn] = field(default_factory=dict)
    input_columns: dict[str, list[str]] = field(default_factory=dict)
    input_validate_sql: dict[str, list[str]] = field(default_factory=dict)
    spec_source: str | None = None
    spec_module: str | None = None
    spec_git_commit: str | None = None
    spec_git_root: str | None = None
    spec_git_dirty: bool | None = None

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

        fingerprint = _build_workspace_fingerprint(
            self.inputs, self.tasks, self.input_columns
        )

        log.info("Ingesting %d input(s)...", len(self.inputs))
        input_row_counts = self._ingest_all(conn)

        # Persist workspace metadata (after ingestion so we have row counts)
        persist_workspace_meta(
            conn,
            fingerprint,
            model,
            tasks=self.tasks,
            input_row_counts=input_row_counts,
            spec_source=self.spec_source,
            spec_module=self.spec_module,
            spec_git_commit=self.spec_git_commit,
            spec_git_root=self.spec_git_root,
            spec_git_dirty=self.spec_git_dirty,
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

        async def run_one(task: Task) -> tuple[str, AgentResult]:
            log.info("[%s] Starting", task.name)
            schema_info = get_schema_info_for_tables(conn, task.inputs)
            result = await run_task_agent(
                conn=conn,
                task=task,
                schema_info=schema_info,
                client=client,
                model=model,
                max_iterations=max_iterations,
            )
            return task.name, result

        task_results, all_success = await self._run_task_dag(self.tasks, run_one)

        # Run exports if all tasks passed
        export_errors: dict[str, str] = {}
        if all_success and self.exports:
            log.info("--- Exports (%d) ---", len(self.exports))
            export_errors = self._run_exports(conn)

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

    @staticmethod
    def _get_task_views(
        conn: duckdb.DuckDBPyConnection, task: Task
    ) -> list[tuple[str, int]]:
        """Get existing views in a task's namespace with row counts.

        Returns list of (view_name, row_count) for views matching the
        task's declared outputs or namespace prefix.
        """
        all_views = [
            row[0]
            for row in conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE internal = false"
            ).fetchall()
        ]
        task_views = [
            v for v in all_views if v in task.outputs or v.startswith(f"{task.name}_")
        ]
        result = []
        for v in sorted(task_views):
            try:
                count = conn.execute(f'SELECT COUNT(*) FROM "{v}"').fetchone()[0]  # type: ignore[index]
            except duckdb.Error:
                count = -1  # View exists but is broken
            result.append((v, count))
        return result

    async def rerun(
        self,
        source_db: Path | str,
        client: OpenRouterClient,
        model: str = "openai/gpt-5.2",
        max_iterations: int = 200,
        mode: str = "validate",
        skip_ingest: bool = False,
    ) -> WorkspaceResult:
        """Rerun workspace using a previous .db as the starting point.

        Copies the source db (preserving existing views), optionally
        re-ingests fresh data, then validates and re-runs agents as needed.
        Each task starts as soon as all its dependencies complete.
        If a task fails, only its downstream dependents are blocked.

        Args:
            source_db: Path to a previous workspace .db file.
            mode: How to handle each task:
                "validate" — skip tasks whose views pass validation;
                    only invoke agents where validation fails.
                "review" — always invoke agents in review mode, even if
                    validation passes. Agents see existing views and are
                    prompted to inspect and adjust.
            skip_ingest: If True, use existing data tables from the source
                db instead of calling input callables for fresh data.

        Returns:
            WorkspaceResult with per-task results.
        """
        if mode not in ("validate", "review"):
            raise ValueError(f"Invalid mode: {mode!r} (use 'validate' or 'review')")

        start_time = time.time()
        self._validate_config()

        layers = resolve_dag(self.tasks)
        dag_layer_names = [[t.name for t in layer] for layer in layers]

        # Check compatibility
        source_db = Path(source_db)
        if not source_db.exists():
            raise ValueError(f"Source db not found: {source_db}")

        fingerprint = _build_workspace_fingerprint(
            self.inputs, self.tasks, self.input_columns
        )

        source_conn = duckdb.connect(str(source_db), read_only=True)
        compat_errors = check_fingerprint_compatibility(source_conn, fingerprint)
        source_conn.close()

        if compat_errors:
            raise ValueError(
                "Source db is not compatible with current spec:\n"
                + "\n".join(f"  - {e}" for e in compat_errors)
            )

        # Copy source to output path
        db_path = Path(self.db_path)
        if db_path.exists():
            db_path.unlink()
        shutil.copy2(source_db, db_path)
        conn = duckdb.connect(str(db_path))

        log.info("Source: %s", source_db)

        if skip_ingest:
            log.info("Skipping ingestion — using existing data")
            # Read row counts from existing tables
            input_row_counts: dict[str, int] = {}
            for table_name in self.inputs:
                try:
                    row = conn.execute(
                        f'SELECT COUNT(*) FROM "{table_name}"'
                    ).fetchone()
                    input_row_counts[table_name] = row[0] if row else 0
                except duckdb.Error:
                    input_row_counts[table_name] = 0
        else:
            log.info("Re-ingesting %d input(s)...", len(self.inputs))
            input_row_counts = self._ingest_all(conn)

        # Update workspace metadata for this run
        persist_workspace_meta(
            conn,
            fingerprint,
            model,
            tasks=self.tasks,
            input_row_counts=input_row_counts,
            source_db=str(source_db),
            rerun_mode=mode,
            spec_source=self.spec_source,
            spec_module=self.spec_module,
            spec_git_commit=self.spec_git_commit,
            spec_git_root=self.spec_git_root,
            spec_git_dirty=self.spec_git_dirty,
        )

        # Validate inputs (skip if we didn't re-ingest — data is already validated)
        if not skip_ingest and (self.input_columns or self.input_validate_sql):
            log.info("Validating inputs...")
            input_errors = self._validate_inputs(conn)
            if input_errors:
                conn.close()
                raise ValueError(
                    "Input validation failed:\n"
                    + "\n".join(f"  - {e}" for e in input_errors)
                )

        skipped = 0

        async def run_one(task: Task) -> tuple[str, AgentResult]:
            nonlocal skipped
            errors = task.validate(conn)

            # In validate mode, skip tasks that pass
            if mode == "validate" and not errors:
                log.info("[%s] Validation passed — skipping", task.name)
                skipped += 1
                return task.name, AgentResult(
                    success=True,
                    final_message="Existing views passed validation",
                    iterations=0,
                    messages=[],
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cache_read_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                    tool_calls_count=0,
                )

            # Run agent
            if errors:
                log.info(
                    "[%s] Validation failed (%d errors) — running agent",
                    task.name,
                    len(errors),
                )
            else:
                log.info("[%s] Review mode — running agent", task.name)

            existing = self._get_task_views(conn, task)
            schema_info = get_schema_info_for_tables(conn, task.inputs)
            result = await run_task_agent(
                conn=conn,
                task=task,
                schema_info=schema_info,
                client=client,
                model=model,
                max_iterations=max_iterations,
                existing_views=existing,
                validation_errors=errors if errors else None,
            )
            return task.name, result

        task_results, all_success = await self._run_task_dag(self.tasks, run_one)

        # Exports
        export_errors: dict[str, str] = {}
        if all_success and self.exports:
            log.info("--- Exports (%d) ---", len(self.exports))
            export_errors = self._run_exports(conn)

        conn.close()
        elapsed_s = time.time() - start_time

        status = "ALL PASSED" if all_success else "SOME FAILED"
        log.info(
            "--- Rerun complete: %s (%.1fs, %d/%d skipped) ---",
            status,
            elapsed_s,
            skipped,
            len(self.tasks),
        )
        for name, result in task_results.items():
            s = "PASS" if result.success else "FAIL"
            if result.iterations == 0:
                log.info("  %s: %s (skipped — valid)", name, s)
            else:
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
