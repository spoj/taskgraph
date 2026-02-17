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

from src.api import OpenRouterClient, DEFAULT_MODEL
from src.agent_loop import DEFAULT_MAX_ITERATIONS
from src.spec import load_spec_from_module, resolve_module_path
from src.workspace import Workspace, read_workspace_meta
from src.task import (
    Task,
    resolve_dag,
    resolve_task_deps,
    validate_task_graph,
    MAX_INLINE_MESSAGES,
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
INPUTS = {
    "data": [
        {"x": 1},
        {"x": 2},
        {"x": 3},
    ],
}

TASKS = [
    {
        "name": "double",
        # sql: deterministic SQL, no LLM needed.
        # Use "prompt" for LLM-driven transforms.
        "sql": "CREATE VIEW output AS SELECT x, x * 2 AS x2 FROM data",
        "inputs": ["data"],
        "outputs": ["output"],
        "output_columns": {"output": ["x", "x2"]},
    }
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
            pass  # Source not available (installed without docs)
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


def _format_task_tree(tasks: list[Task]) -> list[str]:
    """Format task DAG as a list of tree lines."""
    if not tasks:
        return ["  (no tasks)"]

    deps = resolve_task_deps(tasks)
    task_by_name = {t.name: t for t in tasks}
    # parent -> sorted children
    children: dict[str, list[str]] = {t.name: [] for t in tasks}
    for name, parents in deps.items():
        for p in parents:
            children[p].append(name)
    for p in children:
        children[p] = sorted(set(children[p]))

    roots = sorted([name for name, parents in deps.items() if not parents])
    all_task_outputs: set[str] = set()
    for t in tasks:
        all_task_outputs.update(t.outputs)

    def _fmt_details(t: Task) -> str:
        kind = t.transform_mode()
        validation = "yes" if t.has_validation() else "no"
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

        return (
            f"type={kind}  validate={validation}  in={inp_s}  "
            f"out={', '.join(out_parts) if out_parts else '(none)'}"
        )

    lines: list[str] = []

    def _emit(name: str, prefix: str, is_last: bool, stack: set[str], seen: set[str]):
        t = task_by_name[name]
        branch = "`- " if is_last else "|- "
        lines.append(f"{prefix}{branch}{name}  {_fmt_details(t)}")

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
    help=f"Maximum agent iterations per task (default: {DEFAULT_MAX_ITERATIONS})",
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
    log.info("Tasks: %d", len(loaded["tasks"]))

    # Show task tree before running
    tree_lines = _format_task_tree(loaded["tasks"])
    if tree_lines:
        log.info("\nDAG (tree):")
        for line in tree_lines:
            log.info(line)
        log.info("")

    needs_llm = any(t.transform_mode() == "prompt" for t in loaded["tasks"])
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
        return await workspace.run(
            model=model,
            max_iterations=max_iterations,
        )

    result = asyncio.run(_run())

    log.info("Saved to: %s", output)
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

        # --- Change report from _changes table ---
        has_changes = False
        try:
            change_rows = conn.execute(
                "SELECT task, view_name, kind, sql_before, sql_after, "
                "cols_before, cols_after, rows_before, rows_after "
                "FROM _changes ORDER BY task, view_name"
            ).fetchall()
            has_changes = bool(change_rows)
        except duckdb.Error:
            change_rows = []

        if has_changes:
            # Changes are now reported real-time by the workspace
            pass
        else:
            # Fallback: no _changes table, show basic view listing
            click.echo(f"\nViews: {len(views)}")

            for task in tasks:
                if not task.outputs:
                    continue

                click.echo(f"\n  {task.name}:")
                for output_name in task.outputs:
                    if output_name in views:
                        try:
                            count = conn.execute(
                                f'SELECT COUNT(*) FROM "{output_name}"'
                            ).fetchone()
                            count_s = str(count[0]) if count else "0"
                            click.echo(f"    {output_name}: {count_s} rows")
                        except duckdb.Error:
                            click.echo(f"    {output_name}: error counting rows")
                    else:
                        click.echo(f"    {output_name}: MISSING")

        # --- Warnings and errors (always shown) ---
        for task in tasks:
            warn_count, warn_msgs = task.validation_warnings(
                conn, limit=MAX_INLINE_MESSAGES
            )
            if warn_count:
                click.echo(f"\n  {task.name} warnings: {warn_count}")
                for msg in warn_msgs:
                    click.echo(f"    - {msg}")

            errors = task.validate_transform(conn)
            if task.has_validation():
                errors.extend(task.validate_validation_views(conn))
            if errors:
                click.echo(f"\n  {task.name} errors: {len(errors)}")
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

        prompts = _meta_json(meta, "task_prompts")
        if prompts:
            click.echo(f"\nTasks ({len(prompts)}):")
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
    except ValueError as e:
        click.echo(f"\nDAG error: {e}")
        sys.exit(1)

    total_tasks = len(tasks)
    click.echo(f"\nDAG (tree, {total_tasks} tasks):")
    for line in _format_task_tree(tasks):
        click.echo(line)
    click.echo()

    # --- Validation summary ---
    has_validation = False
    val_lines = []
    for t in tasks:
        parts = []
        if t.output_columns:
            n_schemas = len(t.output_columns)
            parts.append(f"{n_schemas} output schema{'s' if n_schemas > 1 else ''}")
        n_val = len(t.validation_view_names())
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
