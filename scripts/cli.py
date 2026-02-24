"""CLI entry point for Taskgraph workspace runner.

Usage:
    taskgraph init

    # Run the default spec (pyproject [tool.taskgraph].spec, else specs.main)
    taskgraph run

    # Run an explicit spec module
    taskgraph run --spec my_app.specs.main
"""

import asyncio
import json
import logging
import duckdb
import os
import sys
import click
import tomllib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv, find_dotenv

# Load .env by walking upward from the CWD.
# This matches common "app repo has a .env" workflows.
_dotenv_path = find_dotenv(usecwd=True)
DOTENV_PATH = Path(_dotenv_path) if _dotenv_path else None
if _dotenv_path:
    load_dotenv(_dotenv_path)

# Allow importing local spec modules (e.g. specs.main) from the CWD.
# Console-script entrypoints don't reliably include the working directory.
# CWD goes first so user's specs/ always wins over any same-named package
# in the installed taskgraph tree (e.g. editable installs).
_cwd = str(Path.cwd())
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

from src.api import (
    DEFAULT_MODEL,
    OPENROUTER_API_KEY_ENV,
    OpenRouterClient,
    has_openrouter_api_key,
)
from src.catalog import count_rows_display, list_tables, list_views
from src.ingest import FileInput
from src.agent_loop import DEFAULT_MAX_ITERATIONS
from src.infra import read_workspace_meta
from src.spec import load_spec_from_module, resolve_module_path
from src.workspace import Workspace
from src.task import (
    Node,
    resolve_dag,
    resolve_deps,
    validate_graph,
    MAX_INLINE_MESSAGES,
)

log = logging.getLogger(__name__)


def _meta_json(meta: dict[str, str], key: str) -> dict[str, Any]:
    """Parse a JSON blob from workspace meta."""
    raw = meta.get(key)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _describe_node(n: Node, include_validation: bool = False) -> str:
    """One-line description of a node for display.

    Produces a string like:
      ``[source] (data.csv) columns: id, amt``
      ``[sql] depends on: raw  -> prep_clean(id, amount)``
      ``[prompt] depends on: prep  -> match_results(id, score) validate=yes``

    Works identically for all node types.
    """
    parts: list[str] = [f"[{n.node_type()}]"]

    # Source description
    if n.is_source():
        src = n.source
        if callable(src) and not isinstance(src, FileInput):
            parts.append("(callable)")
        elif isinstance(src, FileInput):
            parts.append(f"({src.path})")
        elif isinstance(src, str):
            parts.append(f"({src})")
        elif isinstance(src, list):
            parts.append(f"({len(src)} rows)")
        elif isinstance(src, dict):
            parts.append("(dict)")

    # Columns (source) or output_columns (sql/prompt) — shown for all types
    if n.columns:
        parts.append(f"columns: {', '.join(n.columns)}")
    if n.depends_on:
        parts.append(f"depends on: {', '.join(n.depends_on)}")
    if n.output_columns:
        out_parts = []
        for o, cols in n.output_columns.items():
            out_parts.append(f"{o}({', '.join(cols)})" if cols else o)
        parts.append(" -> " + ", ".join(out_parts))
    if include_validation and n.has_validation():
        parts.append("validate=yes")

    return "  ".join(parts)


def _require_openrouter_api_key() -> None:
    if has_openrouter_api_key():
        return

    hint = (
        f"Set OPENROUTER_API_KEY in your environment or in {DOTENV_PATH}"
        if DOTENV_PATH
        else "Set OPENROUTER_API_KEY in your environment or in a .env in the current directory (or a parent directory)."
    )
    raise click.ClickException(f"{OPENROUTER_API_KEY_ENV} is required. {hint}")


@click.group()
def main():
    """Taskgraph — multi-node workspace runner."""


def _default_spec_module() -> str:
    """Return the implicit default spec module.

    Precedence:
    1) [tool.taskgraph].spec in pyproject.toml (if present)
    2) specs.main (repo-local default)
    """
    pyproject_path = Path.cwd() / "pyproject.toml"
    data: dict | None = None
    if pyproject_path.exists():
        try:
            data = tomllib.loads(pyproject_path.read_text())
        except OSError as e:
            raise click.ClickException(f"Failed to read {pyproject_path}: {e}")
        except tomllib.TOMLDecodeError as e:
            raise click.ClickException(f"Invalid TOML in {pyproject_path}: {e}")

        tool_spec = (
            data.get("tool", {}).get("taskgraph", {}).get("spec")
            if isinstance(data, dict)
            else None
        )
        if tool_spec:
            return str(tool_spec)

    # New implicit default: repo-local specs/main.py.
    if (Path.cwd() / "specs" / "main.py").exists():
        return "specs.main"

    raise click.ClickException(
        "No --spec provided and no default spec found. "
        "Run `taskgraph init` to create specs/main.py, or set [tool.taskgraph].spec, "
        "or pass --spec."
    )


def _resolve_spec_arg(spec: str) -> str:
    """Resolve a --spec argument that may be a file path or module path.

    If spec looks like a file path (contains / or \\, or ends with .py),
    convert to a dotted module path relative to CWD.
    Otherwise return as-is (already a module path).
    """
    if "/" not in spec and "\\" not in spec and not spec.endswith(".py"):
        return spec  # Already a module path

    # Normalize backslashes to forward slashes before Path parsing
    spec = spec.replace("\\", "/")
    path = Path(spec)
    if path.suffix == ".py":
        path = path.with_suffix("")

    # Strip leading './' and convert separators to dots
    parts = [p for p in path.parts if p != "."]
    return ".".join(parts)


def _slug_for_filename(text: str) -> str:
    """Convert an arbitrary string to a safe filename slug."""
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("._-")
    return slug or "spec"


def _default_output_db_path(spec_module: str, now: datetime | None = None) -> Path:
    """Generate a default output .db path from spec + timestamp."""
    if now is None:
        now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    spec_slug = _slug_for_filename(spec_module.replace(".", "-"))
    return Path("runs") / f"{spec_slug}_{ts}.db"


def _project_name_slug(directory: Path) -> str:
    """Derive a PEP 508 project name from a directory name."""
    raw = directory.name.lower()
    # Replace non-alphanumeric with hyphens, collapse runs, strip edges
    out: list[str] = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        else:
            if out and out[-1] != "-":
                out.append("-")
    slug = "".join(out).strip("-")
    return slug or "my-project"


@main.command()
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing files when initializing",
)
def init(force: bool):
    """Initialize a Taskgraph spec in the current project."""
    root = Path.cwd()
    pyproject_path = root / "pyproject.toml"
    gitignore_path = root / ".gitignore"
    env_path = root / ".env"

    created: list[str] = []
    skipped: list[str] = []

    # --- pyproject.toml ---
    if not pyproject_path.exists() or force:
        project_name = _project_name_slug(root)
        pyproject_path.write_text(
            f"""[project]
name = "{project_name}"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = ["taskgraph"]

[tool.taskgraph]
spec = "specs.main"
"""
        )
        created.append("pyproject.toml")
    else:
        skipped.append("pyproject.toml")

    # --- specs/ directory and main.py ---
    specs_dir = root / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    specs_init_py = specs_dir / "__init__.py"
    if not specs_init_py.exists() or force:
        specs_init_py.write_text("")

    spec_path = specs_dir / "main.py"
    if not spec_path.exists() or force:
        spec_path.write_text(
            """\
NODES = [
    {
        "name": "data",
        "source": [
            {"x": 1},
            {"x": 2},
            {"x": 3},
        ],
    },
    {
        "name": "double",
        "depends_on": ["data"],
        # sql: deterministic SQL, no LLM needed.
        # Use "prompt" for LLM-driven transforms.
        "sql": "CREATE OR REPLACE VIEW double_output AS SELECT x, x * 2 AS x2 FROM data",
        "output_columns": {"double_output": ["x", "x2"]},
    },
]
"""
        )
        created.append("specs/main.py")
    else:
        skipped.append("specs/main.py")

    # --- SPEC_GUIDE.md ---
    guide_dst = root / "SPEC_GUIDE.md"
    if not guide_dst.exists() or force:
        guide_src = Path(__file__).parent.parent / "SPEC_GUIDE.md"
        try:
            guide_dst.write_text(guide_src.read_text())
            created.append("SPEC_GUIDE.md")
        except OSError:
            skipped.append("SPEC_GUIDE.md (source not available)")
    else:
        skipped.append("SPEC_GUIDE.md")

    # --- .env ---
    if not env_path.exists() or force:
        env_path.write_text(
            """\
# Get your key at https://openrouter.ai/keys
# OPENROUTER_API_KEY=sk-or-...
"""
        )
        created.append(".env")
    else:
        skipped.append(".env")

    # --- .gitignore ---
    gitignore_entries = [".venv/", "__pycache__/", "*.db", ".env"]
    if not gitignore_path.exists() or force:
        gitignore_path.write_text("\n".join(gitignore_entries) + "\n")
        created.append(".gitignore")
    else:
        # Ensure .env is in existing .gitignore
        existing = gitignore_path.read_text()
        if ".env" not in existing:
            with open(gitignore_path, "a") as f:
                f.write(".env\n")
        skipped.append(".gitignore")

    # --- Output ---
    if created:
        for name in created:
            click.echo(f"  created  {name}")
    if skipped:
        for name in skipped:
            click.echo(f"  exists   {name}")

    click.echo("\nNext:")
    click.echo("  uv sync")
    click.echo("  tg run")


def _format_node_tree(nodes: list[Node]) -> list[str]:
    """Format node DAG as a list of tree lines.

    Shows all node types: [source], [sql], [prompt].
    """
    if not nodes:
        return ["  (no nodes)"]

    deps = resolve_deps(nodes)
    node_by_name = {n.name: n for n in nodes}
    # parent -> sorted children
    children: dict[str, list[str]] = {n.name: [] for n in nodes}
    for name, parents in deps.items():
        for p in parents:
            children[p].append(name)
    for p in children:
        children[p] = sorted(set(children[p]))

    roots = sorted([name for name, parents in deps.items() if not parents])

    lines: list[str] = []

    def _emit(name: str, prefix: str, is_last: bool, stack: set[str], seen: set[str]):
        n = node_by_name[name]
        branch = "`- " if is_last else "|- "
        lines.append(
            f"{prefix}{branch}{name}  {_describe_node(n, include_validation=True)}"
        )

        if name in stack:
            lines.append(f"{prefix}{'   ' if is_last else '|  '}[cycle]")
            return
        if name in seen:
            lines.append(f"{prefix}{'   ' if is_last else '|  '}[shared]")
            return

        seen.add(name)
        stack.add(name)
        kids = children.get(name, [])
        next_prefix = prefix + ("   " if is_last else "|  ")
        for i, child in enumerate(kids):
            _emit(
                child,
                prefix=next_prefix,
                is_last=(i == len(kids) - 1),
                stack=stack,
                seen=seen,
            )
        stack.remove(name)

    seen_all: set[str] = set()
    for i, r in enumerate(roots):
        _emit(
            r,
            prefix="  ",
            is_last=(i == len(roots) - 1),
            stack=set(),
            seen=seen_all,
        )
    return lines


@main.command()
@click.option(
    "--spec",
    "-s",
    default=None,
    help="Spec module path (default: [tool.taskgraph].spec from pyproject.toml)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=False,
    default=None,
    help="Output database path (default: runs/<spec>_<timestamp>.db)",
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    help=f"Model to use (default: {DEFAULT_MODEL})",
)
@click.option(
    "--max-iterations",
    default=DEFAULT_MAX_ITERATIONS,
    type=int,
    help=f"Maximum agent iterations per node (default: {DEFAULT_MAX_ITERATIONS})",
)
@click.option(
    "--max-concurrency",
    default=50,
    type=int,
    help="Maximum concurrent running nodes (0 = unlimited)",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress verbose output")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite output file without prompting",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(["low", "medium", "high"]),
    default=None,
    help="Reasoning effort level (default: low)",
)
def run(
    spec: str | None,
    output: Path | None,
    model: str,
    max_iterations: int,
    max_concurrency: int,
    quiet: bool,
    force: bool,
    reasoning_effort: str | None,
):
    """Run a Taskgraph workspace from a spec module."""
    # Configure logging
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    if spec is None:
        spec = _default_spec_module()
    else:
        spec = _resolve_spec_arg(spec)

    if output is None:
        output = _default_output_db_path(spec)
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output = Path(output)
        if output.suffix != ".db":
            output = output.with_suffix(".db")
            log.warning("Output path adjusted to %s (added .db suffix)", output)

    # Resolve and validate spec module
    try:
        resolve_module_path(spec)
    except ValueError as e:
        raise click.ClickException(str(e))

    # Load workspace spec
    try:
        nodes, exports = load_spec_from_module(spec)
    except Exception as e:
        raise click.ClickException(str(e))

    # Overwrite check
    if output.exists():
        if not force:
            click.confirm(
                f"{output} already exists and will be overwritten. Continue?",
                abort=True,
            )

    workspace = Workspace(
        db_path=output,
        nodes=nodes,
        exports=exports,
        spec_module=spec,
    )

    source_count = sum(1 for n in nodes if n.is_source())

    log.info("Spec: %s", spec)
    log.info("Output: %s", output)
    log.info("Model: %s", model)
    log.info("Max concurrency: %s", max_concurrency)
    log.info(
        "Nodes: %d (%d sources, %d transforms)",
        len(nodes),
        source_count,
        len(nodes) - source_count,
    )

    # Show node tree before running
    tree_lines = _format_node_tree(nodes)
    if tree_lines:
        log.info("\nDAG (tree):")
        for line in tree_lines:
            log.info(line)
        log.info("")

    needs_llm = any(n.node_type() == "prompt" for n in nodes)
    needs_client = needs_llm
    if needs_client:
        _require_openrouter_api_key()

    async def _run():
        if needs_client:
            async with OpenRouterClient(reasoning_effort=reasoning_effort) as client:
                return await workspace.run(
                    client=client,
                    model=model,
                    max_iterations=max_iterations,
                    max_concurrency=max_concurrency,
                )
        return await workspace.run(
            model=model,
            max_iterations=max_iterations,
            max_concurrency=max_concurrency,
        )

    result = asyncio.run(_run())

    log.info("Saved to: %s", output)
    log.info("Node metadata: SELECT node, meta_json FROM _node_meta")
    log.info("SQL trace: SELECT * FROM _trace")
    log.info(
        "Final report: SELECT value FROM _workspace_meta WHERE key = 'final_report'"
    )

    if not quiet:
        _report_run_summary(output, nodes)

    sys.exit(0 if result.success else 1)


def _report_run_summary(output: Path, nodes: list[Node]) -> None:
    try:
        conn = duckdb.connect(str(output), read_only=True)
    except duckdb.Error:
        return

    try:
        views = set(list_views(conn, exclude_prefixes=("_",)))
        tables = set(list_tables(conn, exclude_prefixes=("_",)))
        all_outputs = views | tables

        click.echo(
            f"\nOutputs: {len(all_outputs)} ({len(tables)} materialized, {len(views)} views)"
        )

        source_nodes = [n for n in nodes if n.is_source()]
        if source_nodes:
            click.echo("\n  Sources:")
            for node in source_nodes:
                if node.name in tables:
                    click.echo(
                        f"    {node.name}: {count_rows_display(conn, node.name)} rows"
                    )

        for node in nodes:
            if node.is_source():
                continue
            if node.output_columns:
                node_output_names = list(node.output_columns.keys())
            else:
                prefix = f"{node.name}_"
                node_output_names = sorted(
                    n for n in all_outputs if n.startswith(prefix)
                )
            if not node_output_names:
                continue
            click.echo(f"\n  {node.name}:")
            for output_name in node_output_names:
                if output_name in all_outputs:
                    click.echo(
                        f"    {output_name}: {count_rows_display(conn, output_name)} rows"
                    )
                else:
                    click.echo(f"    {output_name}: MISSING")

        # --- Warnings and errors (always shown, all node types) ---
        for node in nodes:
            warn_count, warn_msgs = node.validation_warnings(
                conn, limit=MAX_INLINE_MESSAGES
            )
            if warn_count:
                click.echo(f"\n  {node.name} warnings: {warn_count}")
                for msg in warn_msgs:
                    click.echo(f"    - {msg}")

            errors = node.validate_outputs(conn)
            if node.has_validation():
                errors.extend(node.validate_validation_views(conn))
            if errors:
                click.echo(f"\n  {node.name} errors: {len(errors)}")
                for msg in errors:
                    click.echo(f"    - {msg}")
    finally:
        conn.close()


@main.command()
@click.argument("target", required=False, type=click.Path(path_type=Path))
@click.option(
    "--spec",
    "-s",
    default=None,
    help="Spec module path (default: [tool.taskgraph].spec from pyproject.toml)",
)
def show(target: Path | None, spec: str | None):
    """Show a spec or workspace database.

    If TARGET is a .db file, shows workspace metadata and outputs.
    Otherwise, shows the spec structure.

    \b
    Example:
        taskgraph show output.db
        taskgraph show --spec my_app.specs.main
        taskgraph show
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if target is not None:
        if not target.suffix == ".db":
            raise click.ClickException(
                f"{target} is not a .db file. "
                f"Use --spec to show a spec module, or pass a .db workspace file."
            )
        if not target.exists():
            raise click.ClickException(f"{target} does not exist.")
        try:
            conn = duckdb.connect(str(target), read_only=True)
        except duckdb.Error as e:
            raise click.ClickException(
                f"Cannot open {target} as a DuckDB database: {e}"
            )
        meta = read_workspace_meta(conn)
        conn.close()
        if not meta:
            raise click.ClickException(
                f"{target} has no workspace metadata — not a Taskgraph workspace."
            )

        run = _meta_json(meta, "run")
        spec_m = _meta_json(meta, "spec")
        exports = _meta_json(meta, "exports")
        final_report = _meta_json(meta, "final_report")

        click.echo(f"Workspace: {target}\n")
        click.echo(f"Created: {meta.get('created_at_utc', '(unknown)')}")
        click.echo(f"Model: {meta.get('llm_model', '(unknown)')}")
        click.echo(f"Mode: {run.get('mode', '(unknown)')}")
        if spec_m.get("module"):
            click.echo(f"Spec: {spec_m.get('module')}")

        counts = _meta_json(meta, "inputs_row_counts")
        if counts:
            click.echo("\nInputs:")
            for name in sorted(counts.keys()):
                click.echo(f"  {name}: {counts[name]} rows")

        prompts = _meta_json(meta, "node_prompts")
        if prompts:
            click.echo(f"\nNodes ({len(prompts)}):")
            for name in sorted(prompts.keys()):
                click.echo(f"  {name}")

        if exports:
            attempted = exports.get("attempted")
            results = exports.get("results") or {}
            click.echo("\nExports:")
            click.echo(f"  attempted: {attempted}")
            for name in sorted(results.keys()):
                r = results[name]
                ok = r.get("ok")
                err = r.get("error")
                if ok:
                    click.echo(f"  {name}: OK")
                else:
                    click.echo(f"  {name}: FAILED ({err})")

        if final_report:
            status = final_report.get("status")
            click.echo("\nFinal Report:")
            click.echo(f"  status: {status}")
            if final_report.get("error"):
                click.echo(f"  error: {final_report.get('error')}")
            md = final_report.get("md") or ""
            if md.strip():
                click.echo("\n" + md.rstrip())
        return

    if spec is None:
        spec = _default_spec_module()
    else:
        spec = _resolve_spec_arg(spec)

    spec_module = spec
    try:
        resolve_module_path(spec_module)
    except ValueError as e:
        raise click.ClickException(str(e))

    try:
        nodes, exports = load_spec_from_module(spec_module)
    except Exception as e:
        raise click.ClickException(str(e))

    click.echo(f"Spec: {spec_module}\n")

    # --- Nodes by layer ---
    try:
        layers = resolve_dag(nodes)
    except ValueError as e:
        click.echo(f"\nDAG error: {e}")
        sys.exit(1)

    click.echo(f"Nodes ({len(nodes)}):")
    for layer_idx, layer in enumerate(layers):
        click.echo(f"  Layer {layer_idx}:")
        for n in layer:
            desc = _describe_node(n)
            click.echo(f"    {n.name:<20}{desc}")

    # --- DAG (tree) ---
    click.echo(f"\nDAG (tree, {len(nodes)} nodes):")
    for line in _format_node_tree(nodes):
        click.echo(line)
    click.echo()

    # --- Validation summary ---
    val_lines = []
    for n in nodes:
        parts = []
        if n.is_source() and n.columns:
            parts.append(f"{len(n.columns)} required columns")
        if n.output_columns:
            ns = len(n.output_columns)
            parts.append(f"{ns} output schema{'s' if ns > 1 else ''}")
        if n.has_validation():
            ns = len(n.validation_queries())
            parts.append(f"validate: {ns} quer{'y' if ns == 1 else 'ies'}")
        if parts:
            val_lines.append(f"  {n.name}: {', '.join(parts)}")

    if val_lines:
        click.echo("Validation:")
        for line in val_lines:
            click.echo(line)

    # --- Graph validation ---
    graph_errors = validate_graph(nodes)
    if graph_errors:
        click.echo("\nGraph errors:")
        for err in graph_errors:
            click.echo(f"  ! {err}")

    # --- Exports ---
    if exports:
        click.echo(f"\nExports ({len(exports)}):")
        for path in exports:
            click.echo(f"  {path}")


if __name__ == "__main__":
    main()
