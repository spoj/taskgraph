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
2. Define validation views if configured
3. Check validation views (``node.validate_validation_views(conn)``)
4. Materialize all ``{name}_*`` views (source nodes simply have none)

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent_loop import AgentResult, DEFAULT_MAX_ITERATIONS
from .api import OpenRouterClient, DEFAULT_MODEL
from .catalog import count_rows, list_views, view_exists
from .infra import (
    init_infra,
    persist_node_meta,
    persist_workspace_meta,
    read_workspace_meta,
    upsert_workspace_meta,
)
from .ingest import FileInput, ingest_file, ingest_table
from .agent import (
    _simple_result,
    run_node_agent,
    run_sql_node,
    validate_node_complete,
)
from .task import (
    Node,
    resolve_dag,
    resolve_deps,
    validate_graph,
)

log = logging.getLogger(__name__)

# Type aliases
InputValue = Any  # Callable[[], TableData] | TableData | FileInput
ExportFn = Callable[[duckdb.DuckDBPyConnection, Path], None]


# --- View materialization ---


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
    unique_names = list(dict.fromkeys(view_names))

    materialized = 0
    for view_name in unique_names:
        # Check the view exists before attempting materialization
        if not view_exists(conn, view_name):
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

    After a node completes and passes validation, all ``{name}_*`` views
    (output views + validation views) are discovered from the catalog and
    materialized.  This is the same logic for ALL node types — source
    nodes simply won't have ``{name}_*`` output views (they produce a
    table, not views), so the unified code naturally handles them.

    Returns the number of views materialized.
    """
    prefix = f"{node.name}_"
    view_names = [v for v in list_views(conn) if v.startswith(prefix)]

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

        count = count_rows(conn, node.name)
        count = count if count is not None else 0
        if count == 0:
            log.warning("  %s: 0 rows (empty table)", node.name)
        else:
            log.info("  %s: %d rows", node.name, count)
        return count

    async def _execute_source_node(
        self,
        conn: duckdb.DuckDBPyConnection,
        node: Node,
        client: Any | None,
    ) -> tuple[AgentResult, int | None]:
        """Execute a source node: ingest data.

        Returns:
            (AgentResult, row_count or None on failure)
        """
        try:
            row_count = await self._ingest_node(conn, node, client)
        except Exception as e:
            return _simple_result(False, f"Source ingestion failed: {e}"), None
        return _simple_result(True, f"OK ({row_count} rows)"), row_count

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
            init_infra(conn)

            # Track row counts for source nodes (populated during execution)
            source_row_counts: dict[str, int] = {}

            async def run_one(node: Node) -> tuple[str, AgentResult]:
                log.info(
                    "[%s] Starting (%s)",
                    node.name,
                    node.node_type(),
                )
                node_start = time.time()

                # --- Phase 1: Execute (type-specific) ---
                if node.is_source():
                    result, row_count = await self._execute_source_node(
                        conn, node, client
                    )
                    if result.success and row_count is not None:
                        source_row_counts[node.name] = row_count
                elif node.node_type() == "sql":
                    result = await run_sql_node(conn=conn, node=node)
                elif node.node_type() == "prompt":
                    if client is None:
                        raise RuntimeError(
                            f"Node '{node.name}' requires an OpenRouterClient"
                        )
                    result = await run_node_agent(
                        conn=conn,
                        node=node,
                        client=client,
                        model=model,
                        max_iterations=max_iterations,
                    )
                else:
                    raise RuntimeError(f"Unknown node type: {node.node_type()}")

                # --- Phase 2: Unified post-execution (ALL node types) ---
                # Validate → materialize → persist metadata.
                # Prompt nodes run validation inside their agent loop, but
                # source and sql nodes need it here.  Running it uniformly
                # for all types is safe: for prompt nodes that already
                # passed, validate_node_complete is a cheap re-check.
                if result.success:
                    errors = validate_node_complete(conn, node)
                    if errors:
                        result = AgentResult(
                            success=False,
                            final_message="\n".join(f"- {e}" for e in errors),
                            iterations=result.iterations,
                            messages=result.messages,
                            tool_calls_count=result.tool_calls_count,
                            usage=result.usage,
                        )

                # Materialize on success (all node types)
                if result.success:
                    n = materialize_node_outputs(conn, node)
                    if n:
                        log.debug("[%s] Materialized %d view(s)", node.name, n)

                # Persist per-node metadata (all node types)
                elapsed_s = time.time() - node_start
                ntype = node.node_type()
                node_meta: dict[str, Any] = {
                    "node_type": ntype,
                    "depends_on": node.depends_on,
                    "iterations": result.iterations,
                    "tool_calls": result.tool_calls_count,
                    "elapsed_s": round(elapsed_s, 1),
                    "validation": "PASSED" if result.success else "FAILED",
                    "prompt_tokens": result.usage["prompt_tokens"],
                    "completion_tokens": result.usage["completion_tokens"],
                    "cache_read_tokens": result.usage["cache_read_tokens"],
                    "reasoning_tokens": result.usage["reasoning_tokens"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "model": model if ntype == "prompt" else ntype,
                }
                if ntype == "prompt":
                    node_meta["reasoning_effort"] = (
                        client.reasoning_effort if client else None
                    )
                if ntype in ("prompt", "sql"):
                    node_meta["output_columns"] = dict(node.output_columns)
                if not result.success:
                    node_meta["error"] = result.final_message
                persist_node_meta(conn, node.name, node_meta)

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
