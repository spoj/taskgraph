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
sys.path.insert(0, str(Path.cwd()))

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import OpenRouterClient
from src.spec import load_spec_from_module, resolve_module_path
from src.workspace import Workspace, read_workspace_meta
from src.task import (
    Task,
    resolve_dag,
    resolve_task_deps,
    validate_task_graph,
    validation_outputs,
)

log = logging.getLogger(__name__)


def _meta_json(meta: dict[str, str], key: str) -> dict[str, Any]:
    """Parse a JSON blob from workspace meta."""
    raw = meta.get(key)
    return json.loads(raw) if raw else {}


def _require_openrouter_api_key() -> None:
    if os.environ.get("OPENROUTER_API_KEY"):
        return

    hint = (
        f"Set OPENROUTER_API_KEY in your environment or in {DOTENV_PATH}"
        if DOTENV_PATH
        else "Set OPENROUTER_API_KEY in your environment or in a .env in the current directory (or a parent directory)."
    )
    raise click.ClickException(f"OPENROUTER_API_KEY is required. {hint}")


@click.group()
def main():
    """Taskgraph — multi-task workspace runner."""


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

    if not pyproject_path.exists():
        raise click.ClickException(
            "pyproject.toml not found. Start with `uv init --app --vcs git`, "
            "then `uv add taskgraph`, then run `taskgraph init` inside that project."
        )

    specs_dir = root / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)

    specs_init_py = specs_dir / "__init__.py"
    if not specs_init_py.exists() or force:
        specs_init_py.write_text("")

    spec_path = specs_dir / "main.py"
    if (not spec_path.exists()) or force:
        spec_path.write_text(
            """INPUTS = {
    "data": [
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ],
}

TASKS = [
    {
        "name": "double",
        "intent": "Create a view 'output' with columns x and x2 (x*2).",
        "sql": "CREATE VIEW output AS SELECT x, x * 2 AS x2 FROM data",
        "inputs": ["data"],
        "outputs": ["output"],
        "output_columns": {"output": ["x", "x2"]},
    }
]
"""
        )

    # Copy spec writer guide into the project root for convenience.
    guide_dst = root / "SPEC_GUIDE.md"
    if (not guide_dst.exists()) or force:
        guide_src = Path(__file__).parent.parent / "SPEC_GUIDE.md"
        try:
            guide_dst.write_text(guide_src.read_text())
        except OSError:
            # If the guide isn't available (e.g., installed without source docs),
            # silently skip.
            pass

    if not gitignore_path.exists():
        gitignore_path.write_text(
            """.venv/
__pycache__/
*.db
"""
        )
    elif not force:
        click.echo(".gitignore already exists; skipping.")

    click.echo("Initialized Taskgraph spec: specs/main.py")
    if guide_dst.exists():
        click.echo("Wrote spec guide: SPEC_GUIDE.md")
    click.echo("Default spec module (implicit): specs.main")
    click.echo("Next:")
    click.echo("  uv sync")
    click.echo("  taskgraph run")


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
    default="openai/gpt-5.2",
    help="Model to use (default: openai/gpt-5.2)",
)
@click.option(
    "--max-iterations",
    default=200,
    type=int,
    help="Maximum agent iterations per task (default: 200)",
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
    )

    if spec is None:
        spec = _default_spec_module()

    if output is None:
        output = _default_output_db_path(spec)
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output = Path(output)
        if output.suffix != ".db":
            output = output.with_suffix(".db")

    # Resolve and validate spec module
    try:
        resolve_module_path(spec)
    except ValueError as e:
        raise click.ClickException(str(e))

    # Load workspace spec
    try:
        loaded = load_spec_from_module(spec)
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
        inputs=loaded["inputs"],
        tasks=loaded["tasks"],
        exports=loaded["exports"],
        input_columns=loaded["input_columns"],
        input_validate_sql=loaded["input_validate_sql"],
        spec_module=spec,
    )

    log.info("Spec: %s", spec)
    log.info("Output: %s", output)
    log.info("Model: %s", model)
    log.info("Tasks: %d\n", len(loaded["tasks"]))

    needs_llm = any(
        getattr(t, "run_mode", lambda: "sql")() == "sql" for t in loaded["tasks"]
    )
    if needs_llm:
        _require_openrouter_api_key()

    async def _run():
        if needs_llm:
            async with OpenRouterClient(reasoning_effort=reasoning_effort) as client:
                return await workspace.run(
                    client=client,
                    model=model,
                    max_iterations=max_iterations,
                )

        # SQL-only workspace: don't require OPENROUTER_API_KEY and don't
        # create an http client.
        class _ClientStub:
            reasoning_effort: str | None

            def __init__(self, reasoning_effort: str | None):
                self.reasoning_effort = reasoning_effort

        client = _ClientStub(reasoning_effort=reasoning_effort)
        return await workspace.run(
            client=client,  # type: ignore[arg-type]
            model=model,
            max_iterations=max_iterations,
        )

    result = asyncio.run(_run())

    log.info("\nSaved to: %s", output)
    log.info("Task metadata: SELECT task, meta_json FROM _task_meta")
    log.info("SQL trace: SELECT * FROM _trace")

    if not quiet:
        _report_run_summary(output, loaded["tasks"])

    sys.exit(0 if result.success else 1)


def _report_run_summary(output: Path, tasks: list[Task]) -> None:
    try:
        conn = duckdb.connect(str(output), read_only=True)
    except duckdb.Error:
        return

    try:
        view_rows = conn.execute(
            "SELECT view_name FROM duckdb_views() WHERE internal = false"
        ).fetchall()
        views = {row[0] for row in view_rows}
        click.echo(f"\nViews: {len(views)}")

        for task in tasks:
            if not task.outputs:
                continue

            click.echo(f"\nTask {task.name}:")
            for output_name in task.outputs:
                if output_name in views:
                    try:
                        count = conn.execute(
                            f'SELECT COUNT(*) FROM "{output_name}"'
                        ).fetchone()
                        count_s = str(count[0]) if count else "0"
                        click.echo(f"  {output_name}: {count_s} rows")
                    except duckdb.Error:
                        click.echo(f"  {output_name}: error counting rows")
                else:
                    click.echo(f"  {output_name}: MISSING")

            warn_count, warn_msgs = _collect_warnings(conn, task, views)
            if warn_count:
                click.echo(f"  warnings: {warn_count}")
                for msg in warn_msgs:
                    click.echo(f"    - {msg}")

            errors = task.validate(conn)
            if errors:
                click.echo(f"  errors: {len(errors)}")
                for msg in errors:
                    click.echo(f"    - {msg}")
    finally:
        conn.close()


def _collect_warnings(
    conn: duckdb.DuckDBPyConnection, task: Task, views: set[str], limit: int = 20
) -> tuple[int, list[str]]:
    warnings: list[str] = []
    total = 0
    remaining = max(1, limit)

    for view_name in validation_outputs(task):
        if view_name not in views:
            continue

        try:
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM \"{view_name}\" WHERE lower(status) = 'warn'"
            ).fetchone()
            view_count = int(count_row[0]) if count_row else 0
        except duckdb.Error:
            continue

        if view_count <= 0:
            continue

        total += view_count
        if remaining <= 0:
            continue

        try:
            rows = conn.execute(
                f'SELECT message FROM "{view_name}" '
                f"WHERE lower(status) = 'warn' LIMIT {remaining}"
            ).fetchall()
        except duckdb.Error:
            continue

        if rows:
            warnings.extend([str(r[0]) for r in rows])
            remaining -= len(rows)

    return total, warnings


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
        conn = duckdb.connect(str(target), read_only=True)
        meta = read_workspace_meta(conn)
        conn.close()
        if not meta:
            raise click.ClickException(
                f"{target} has no workspace metadata — not a Taskgraph workspace."
            )

        run = _meta_json(meta, "run")
        spec_m = _meta_json(meta, "spec")
        exports = _meta_json(meta, "exports")

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

        contexts = _meta_json(meta, "task_intents")
        if contexts:
            click.echo(f"\nTasks ({len(contexts)}):")
            for name in sorted(contexts.keys()):
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
        return

    if spec is None:
        spec = _default_spec_module()

    spec_module = spec
    try:
        resolve_module_path(spec_module)
    except ValueError as e:
        raise click.ClickException(str(e))

    try:
        loaded = load_spec_from_module(spec_module)
    except Exception as e:
        raise click.ClickException(str(e))

    tasks = loaded["tasks"]
    inputs = loaded["inputs"]
    input_columns = loaded.get("input_columns", {})
    input_validate_sql = loaded.get("input_validate_sql", {})
    exports = loaded.get("exports", {})

    click.echo(f"Spec: {spec_module}\n")

    # --- Inputs ---
    click.echo(f"Inputs ({len(inputs)}):")
    name_width = max((len(n) for n in inputs), default=0)
    for name in inputs:
        cols = input_columns.get(name, [])
        sql_checks = input_validate_sql.get(name, [])
        parts = []
        if cols:
            parts.append(f"{len(cols)} cols: {', '.join(cols)}")
        if sql_checks:
            parts.append(
                f"{len(sql_checks)} SQL check{'s' if len(sql_checks) > 1 else ''}"
            )
        detail = "  ".join(parts) if parts else "(no validation)"
        click.echo(f"  {name:<{name_width}}  {detail}")

    # --- DAG (tree) ---
    # Use resolve_dag for full validation (cycles, duplicate outputs). Display
    # uses a dependency tree view instead of execution layers.
    try:
        resolve_dag(tasks)
        deps = resolve_task_deps(tasks)
    except ValueError as e:
        click.echo(f"\nDAG error: {e}")
        sys.exit(1)

    total_tasks = len(tasks)
    click.echo(f"\nDAG (tree, {total_tasks} tasks):\n")

    task_by_name = {t.name: t for t in tasks}
    # parent -> sorted children
    children: dict[str, list[str]] = {t.name: [] for t in tasks}
    for name, parents in deps.items():
        for p in parents:
            children[p].append(name)
    for p in children:
        children[p] = sorted(set(children[p]))

    roots = sorted([name for name, parents in deps.items() if not parents])

    # Collect all task outputs for showing which inputs are external vs task-produced
    all_task_outputs: set[str] = set()
    for t in tasks:
        all_task_outputs.update(t.outputs)

    def _fmt_task_details(t: Task) -> str:
        kind = t.run_mode()

        if t.inputs:
            inp_parts = []
            for inp in t.inputs:
                if inp in all_task_outputs:
                    inp_parts.append(inp)
                else:
                    inp_parts.append(f"{inp} (table)")
            inp_s = ", ".join(inp_parts)
        else:
            inp_s = "(none)"

        out_parts = []
        for o in t.outputs:
            cols = t.output_columns.get(o, [])
            out_parts.append(f"{o} ({len(cols)} cols)" if cols else o)

        return f"type={kind}  in={inp_s}  out={', '.join(out_parts) if out_parts else '(none)'}"

    def _emit_subtree(
        name: str, prefix: str, is_last: bool, stack: set[str], seen: set[str]
    ):
        t = task_by_name[name]
        branch = "`- " if is_last else "|- "
        click.echo(f"{prefix}{branch}{name}  {_fmt_task_details(t)}")

        # Avoid infinite recursion (shouldn't happen if resolve_dag passed) and
        # reduce noise for shared downstream tasks in DAGs.
        if name in stack:
            click.echo(f"{prefix}{'   ' if is_last else '|  '}[cycle]")
            return
        if name in seen:
            click.echo(f"{prefix}{'   ' if is_last else '|  '}[shared]")
            return

        seen.add(name)
        stack.add(name)

        kids = children.get(name, [])
        next_prefix = prefix + ("   " if is_last else "|  ")
        for i, child in enumerate(kids):
            _emit_subtree(
                child,
                prefix=next_prefix,
                is_last=(i == len(kids) - 1),
                stack=stack,
                seen=seen,
            )

        stack.remove(name)

    if not roots:
        click.echo("  (no tasks)")
    else:
        seen: set[str] = set()
        for i, r in enumerate(roots):
            _emit_subtree(
                r,
                prefix="  ",
                is_last=(i == len(roots) - 1),
                stack=set(),
                seen=seen,
            )
    click.echo()

    # --- Validation summary ---
    has_validation = False
    val_lines = []
    for t in tasks:
        parts = []
        if t.output_columns:
            n_schemas = len(t.output_columns)
            parts.append(f"{n_schemas} output schema{'s' if n_schemas > 1 else ''}")
        n_val = len(validation_outputs(t))
        if n_val:
            parts.append(f"{n_val} validation view{'s' if n_val > 1 else ''}")
        if parts:
            has_validation = True
            val_lines.append(f"  {t.name}: {', '.join(parts)}")

    if has_validation:
        click.echo("Validation:")
        for line in val_lines:
            click.echo(line)

    # --- Graph validation ---
    input_tables = set(inputs.keys())
    graph_errors = validate_task_graph(tasks, available_tables=input_tables)
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
