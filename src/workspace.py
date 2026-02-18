"""Workspace orchestrator — resolve DAG of nodes, execute, run exports.

A workspace is a single DuckDB database containing:
- Ingested source tables (from user-provided functions, data, or files)
- Per-node output views (created by SQL or agents, materialized as
  tables after validation)
- Metadata and trace tables

Nodes are scheduled greedily: each node starts as soon as all its
dependencies complete (not layer-by-layer). A failed node only blocks
its downstream dependents; unrelated branches continue.

Node types:
- **source** — ingest data into table ``{name}``.
- **sql** — execute deterministic SQL creating ``{name}_*`` views.
- **prompt** — run LLM agent creating ``{name}_*`` views.

After execution, ALL nodes go through the same post-execution flow:
1. Validate outputs (``node.validate_outputs(conn)``)
2. Execute validate_sql if present (create validation views)
3. Check validation views (``node.validate_validation_views(conn)``)
4. Materialize views (sql/prompt: all ``{name}_*`` views; source: none,
   already a table — only validation views if they passed)

Usage:

    workspace = Workspace(
        nodes=[source_node, prep_node, match_node],
        exports={"report.xlsx": write_report},
        db_path="output.db",
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
from .agent import (
    init_trace_table,
    run_node_agent,
    run_sql_node,
    run_validate_sql,
)
from .task import (
    Node,
    resolve_dag,
    resolve_deps,
    validate_graph,
    is_validation_view,
)
from .sql_utils import get_column_schema

log = logging.getLogger(__name__)

# Type aliases
InputValue = Any  # Callable[[], TableData] | TableData | FileInput
ExportFn = Callable[[duckdb.DuckDBPyConnection, Path], None]


# --- Workspace metadata ---


def persist_workspace_meta(
    conn: duckdb.DuckDBPyConnection,
    model: str,
    nodes: list[Node],
    reasoning_effort: str | None = None,
    max_iterations: int | None = None,
    source_row_counts: dict[str, int] | None = None,
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

    # Extract prompts from prompt nodes
    node_prompts = {n.name: n.prompt for n in (nodes or []) if n.prompt}

    created_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    source_tables = list(sorted(source_row_counts.keys())) if source_row_counts else []
    source_schemas: dict[str, list[dict[str, str]]] = {}
    if source_tables:
        for table in source_tables:
            rows_cols = get_column_schema(conn, table)
            source_schemas[table] = [{"name": r[0], "type": r[1]} for r in rows_cols]

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
        ("node_prompts", json.dumps(node_prompts, sort_keys=True)),
        ("llm_model", model),
    ]

    if reasoning_effort:
        rows.append(("llm_reasoning_effort", reasoning_effort))
    if max_iterations is not None:
        rows.append(("llm_max_iterations", str(max_iterations)))

    if source_row_counts:
        rows.append(
            ("inputs_row_counts", json.dumps(source_row_counts, sort_keys=True))
        )
        rows.append(("inputs_schema", json.dumps(source_schemas, sort_keys=True)))

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
) -> int:
    """Materialize a list of views as tables.

    Core materialization logic shared by all node types. For each view
    that exists, does a 3-step swap: CREATE TABLE from view, DROP VIEW,
    RENAME TABLE.

    The original CREATE VIEW SQL is already recorded in ``_trace`` (from
    the exec that created it). The ``_view_definitions`` view on ``_trace``
    provides lineage queries.

    Args:
        conn: DuckDB connection.
        view_names: View names to materialize. Duplicates are ignored.
            Missing views are silently skipped.

    Returns the number of views materialized.
    """
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in view_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)

    materialized = 0
    for view_name in unique_names:
        # Check the view exists before attempting materialization
        rows = conn.execute(
            "SELECT 1 FROM duckdb_views() WHERE internal = false AND view_name = ?",
            [view_name],
        ).fetchall()
        if not rows:
            continue  # View doesn't exist (node may have failed or already materialized)

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


def materialize_node_outputs(
    conn: duckdb.DuckDBPyConnection,
    node: Node,
) -> int:
    """Materialize a node's output views as tables.

    After a node completes and passes validation, views are discovered
    from the DB catalog and materialized as tables so downstream nodes
    read pre-computed data instead of re-evaluating the entire upstream
    view chain on every query.

    What gets materialized depends on node type:

    - **Source nodes**: only validation views (the ingested table already
      exists).
    - **SQL/prompt nodes**: all ``{node.name}_*`` views (excluding
      validation views) plus validation views.

    Returns the number of views materialized.
    """
    prefix = f"{node.name}_"
    all_views = conn.execute(
        "SELECT view_name FROM duckdb_views() WHERE internal = false"
    ).fetchall()

    if node.is_source():
        # Source nodes: only validation views need materialization
        validation_views = [
            v[0] for v in all_views if is_validation_view(v[0], node.name)
        ]
        view_names = sorted(validation_views)
    else:
        # SQL/prompt nodes: all {name}_* views (excluding validation) + validation views
        node_views = [
            v[0]
            for v in all_views
            if v[0].startswith(prefix) and not is_validation_view(v[0], node.name)
        ]
        validation_views = [
            v[0] for v in all_views if is_validation_view(v[0], node.name)
        ]
        view_names = sorted(node_views) + sorted(validation_views)

    return materialize_views(conn, view_names)


@dataclass
class WorkspaceResult:
    """Aggregated results from running all nodes in a workspace."""

    success: bool  # True if ALL nodes passed validation
    node_results: dict[str, AgentResult]  # node_name -> AgentResult
    elapsed_s: float
    dag_layers: list[list[str]]  # For display: layer -> [node_names]
    export_errors: dict[str, str] = field(default_factory=dict)


@dataclass
class Workspace:
    """A multi-node workspace backed by a single DuckDB database.

    Args:
        nodes: List of Node definitions forming a DAG.  Each node is
            one of: source (data ingestion), sql (deterministic SQL),
            or prompt (LLM-driven).  Source nodes are scheduled through
            the same DAG as sql/prompt nodes.
        exports: Mapping of output_path -> fn(conn, path).
            Export functions run after all nodes pass. They receive the
            open database connection and the output file path. Paths
            are relative to CWD.
        db_path: Path for the output database (created fresh).
        spec_module: Module path used to load the spec.
    """

    db_path: Path | str
    nodes: list[Node] = field(default_factory=list)
    exports: dict[str, ExportFn] = field(default_factory=dict)
    spec_module: str | None = None

    def _validate_config(self) -> None:
        """Validate the workspace configuration before running."""
        errors = validate_graph(self.nodes)
        if errors:
            raise ValueError(
                "Graph validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Source node ingestion
    # ------------------------------------------------------------------

    async def _ingest_node(
        self, conn: duckdb.DuckDBPyConnection, node: Node, client: Any | None
    ) -> int:
        """Ingest a single source node into the database.

        Returns row count. Raises on failure.
        """
        value = node.source
        if isinstance(value, FileInput):
            await ingest_file(conn, value, node.name, client=client)
        elif callable(value):
            try:
                data = value()
            except Exception as e:
                raise RuntimeError(f"Source '{node.name}' callable failed: {e}") from e
            ingest_table(conn, data, node.name)
        else:
            ingest_table(conn, value, node.name)

        row = conn.execute(f'SELECT COUNT(*) FROM "{node.name}"').fetchone()
        count = row[0] if row else 0
        if count == 0:
            log.warning("  %s: 0 rows (empty table)", node.name)
        else:
            log.info("  %s: %d rows", node.name, count)
        return count

    # ------------------------------------------------------------------
    # Post-execution flow (unified for all node types)
    # ------------------------------------------------------------------

    def _post_execute(
        self, conn: duckdb.DuckDBPyConnection, node: Node
    ) -> tuple[bool, str]:
        """Unified post-execution: validate outputs, run validate_sql, check
        validation views, materialize.

        Returns (success, error_message).
        """
        # 1. Validate outputs (columns for source, output_columns for sql/prompt)
        errors = node.validate_outputs(conn)
        if errors:
            return False, "\n".join(f"- {e}" for e in errors)

        # 2. Run validate_sql if present
        if node.has_validation():
            val_errors = run_validate_sql(conn=conn, node=node)
            if val_errors:
                return False, "\n".join(f"- {e}" for e in val_errors)

            # 3. Check validation views
            errors = node.validate_validation_views(conn)
            if errors:
                return False, "\n".join(f"- {e}" for e in errors)

        # 4. Materialize
        self._materialize_node(conn, node)
        return True, ""

    def _materialize_node(self, conn: duckdb.DuckDBPyConnection, node: Node) -> int:
        """Materialize views for a node after successful validation.

        Delegates to :func:`materialize_node_outputs` which handles both
        source nodes (only validation views) and sql/prompt nodes (all
        ``{name}_*`` views + validation views).

        Returns the number of views materialized.
        """
        n = materialize_node_outputs(conn, node)
        if n:
            log.debug("[%s] Materialized %d view(s)", node.name, n)
        return n

    # ------------------------------------------------------------------
    # DAG execution
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_dag(
        nodes: list[Node],
        run_one: Callable[[Node], Any],
    ) -> tuple[dict[str, AgentResult], bool]:
        """Schedule nodes as soon as their dependencies are met.

        Args:
            nodes: All nodes in the DAG.
            run_one: Async callable (node) -> (node_name, AgentResult).

        Returns:
            (results dict, all_success bool)
        """
        node_by_name = {n.name: n for n in nodes}
        deps = resolve_deps(nodes)
        results: dict[str, AgentResult] = {}
        all_success = True
        done: set[str] = set()
        failed: set[str] = set()
        running: set[str] = set()
        pending = set(node_by_name.keys())
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
                    t = asyncio.create_task(run_one(node_by_name[name]))
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
                results[name] = result
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

        return results, all_success

    # ------------------------------------------------------------------
    # Node execution (dispatches by type)
    # ------------------------------------------------------------------

    async def _execute_source_node(
        self,
        conn: duckdb.DuckDBPyConnection,
        node: Node,
        client: Any | None,
    ) -> AgentResult:
        """Execute a source node: ingest data, return AgentResult."""
        start_time = time.time()
        try:
            row_count = await self._ingest_node(conn, node, client)
        except Exception as e:
            elapsed = time.time() - start_time
            return AgentResult(
                success=False,
                final_message=f"Source ingestion failed: {e}",
                iterations=0,
                messages=[],
                tool_calls_count=0,
            )

        # Post-execution: validate outputs + validate_sql + materialize
        success, err_msg = self._post_execute(conn, node)
        if not success:
            return AgentResult(
                success=False,
                final_message=err_msg,
                iterations=0,
                messages=[],
                tool_calls_count=0,
            )

        return AgentResult(
            success=True,
            final_message=f"OK ({row_count} rows)",
            iterations=0,
            messages=[],
            tool_calls_count=0,
        )

    async def _execute_sql_node(
        self,
        conn: duckdb.DuckDBPyConnection,
        node: Node,
    ) -> AgentResult:
        """Execute a SQL node and return its result.

        Validation (outputs + validate_sql) runs inside run_sql_node.
        Post-execution materialization is handled by the caller.
        """
        return await run_sql_node(conn=conn, node=node)

    async def _execute_prompt_node(
        self,
        conn: duckdb.DuckDBPyConnection,
        node: Node,
        client: OpenRouterClient,
        model: str,
        max_iterations: int,
    ) -> AgentResult:
        """Execute a prompt node (LLM agent) and return its result.

        Validation runs inside the agent loop.
        """
        return await run_node_agent(
            conn=conn,
            node=node,
            client=client,
            model=model,
            max_iterations=max_iterations,
        )

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        client: OpenRouterClient | None = None,
        model: str = DEFAULT_MODEL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> WorkspaceResult:
        """Run the full workspace: resolve DAG, execute all nodes, export.

        Each node starts as soon as all its dependencies complete
        (cooperative async — DuckDB access naturally serialized).
        If a node fails, only its downstream dependents are blocked.
        Export functions run after all nodes pass.

        Returns:
            WorkspaceResult with per-node results and overall status.
        """
        start_time = time.time()
        self._validate_config()

        layers = resolve_dag(self.nodes)
        dag_layer_names = [[n.name for n in layer] for layer in layers]

        # Create fresh database
        db_path = Path(self.db_path)
        if db_path.exists():
            db_path.unlink()
        conn = duckdb.connect(str(db_path))
        try:
            init_trace_table(conn)

            # Track row counts for source nodes (populated during execution)
            source_row_counts: dict[str, int] = {}

            async def run_one(node: Node) -> tuple[str, AgentResult]:
                log.info(
                    "[%s] Starting (%s)",
                    node.name,
                    node.node_type(),
                )
                before = snapshot_views(conn)

                if node.is_source():
                    result = await self._execute_source_node(conn, node, client)
                    # Track row count for metadata
                    if result.success:
                        try:
                            row = conn.execute(
                                f'SELECT COUNT(*) FROM "{node.name}"'
                            ).fetchone()
                            source_row_counts[node.name] = row[0] if row else 0
                        except duckdb.Error:
                            pass
                elif node.node_type() == "sql":
                    result = await self._execute_sql_node(conn, node)
                elif node.node_type() == "prompt":
                    if client is None:
                        raise RuntimeError(
                            f"Node '{node.name}' requires an OpenRouterClient"
                        )
                    result = await self._execute_prompt_node(
                        conn, node, client, model, max_iterations
                    )
                else:
                    raise RuntimeError(f"Unknown node type: {node.node_type()}")

                # Snapshot BEFORE materialization to capture view changes
                after = snapshot_views(conn)
                changes = diff_snapshots(before, after)

                # Materialize on success (sql/prompt nodes only — source
                # nodes handle materialization inside _execute_source_node
                # via _post_execute)
                if result.success and not node.is_source():
                    self._materialize_node(conn, node)

                if changes:
                    persist_changes(conn, node.name, changes)
                    change_summary = format_changes(node.name, changes)
                    if change_summary:
                        log.info("%s", change_summary)

                return node.name, result

            node_results, all_success = await self._run_dag(self.nodes, run_one)

            # Persist workspace metadata (after execution so we have row counts)
            persist_workspace_meta(
                conn,
                model,
                nodes=self.nodes,
                reasoning_effort=client.reasoning_effort if client else None,
                max_iterations=max_iterations,
                source_row_counts=source_row_counts if source_row_counts else None,
                spec_module=self.spec_module,
            )

            # Run exports if all nodes passed
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
            for name, result in node_results.items():
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
                node_results=node_results,
                elapsed_s=elapsed_s,
                dag_layers=dag_layer_names,
                export_errors=export_errors,
            )
        finally:
            conn.close()
