"""CLI entry point for Taskgraph workspace runner.

Usage:
    taskgraph init

    # Run the default spec (pyproject [tool.taskgraph].spec, else specs.main)
    taskgraph run -o output.db

    # Run an explicit spec module
    taskgraph run --spec my_app.specs.main -o output.db

    # Validate + fix only failing tasks from a prior run
    taskgraph rerun previous.db -o output.db
"""

import asyncio
import logging
import duckdb
import sys
import click
import tomllib
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load .env by walking upward from the CWD.
# This matches common "app repo has a .env" workflows.
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)

# Allow importing local spec modules (e.g. specs.main) from the CWD.
# Console-script entrypoints don't reliably include the working directory.
sys.path.insert(0, str(Path.cwd()))

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import OpenRouterClient
from src.spec import load_spec_from_module, resolve_module_path
from src.spec_repo import get_spec_repo_info
from src.workspace import Workspace, read_workspace_meta
from src.task import resolve_dag, validate_task_graph

log = logging.getLogger(__name__)


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
        "prompt": "Create a view 'output' with columns x and x2 (x*2).",
        "inputs": ["data"],
        "outputs": ["output"],
        "output_columns": {"output": ["x", "x2"]},
    }
]
"""
        )

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
    click.echo("Default spec module (implicit): specs.main")
    click.echo("Next:")
    click.echo("  uv sync")
    click.echo("  taskgraph run -o output.db")


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
    required=True,
    help="Output database path",
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
    output: Path,
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

    # Resolve and validate spec module
    try:
        spec_path = resolve_module_path(spec)
        repo_info = get_spec_repo_info(spec_path)
    except ValueError as e:
        raise click.ClickException(str(e))

    if repo_info.dirty:
        click.echo(
            f"WARNING: spec repo is dirty ({repo_info.root}); run is not strictly reproducible.",
            err=True,
        )

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
        spec_source=loaded.get("spec_source"),
        spec_module=spec,
        spec_git_commit=repo_info.commit,
        spec_git_root=str(repo_info.root),
        spec_git_dirty=repo_info.dirty,
    )

    log.info("Spec: %s", spec)
    log.info("Spec commit: %s", repo_info.commit)
    log.info("Output: %s", output)
    log.info("Model: %s", model)
    log.info("Tasks: %d\n", len(loaded["tasks"]))

    async def _run():
        async with OpenRouterClient(reasoning_effort=reasoning_effort) as client:
            return await workspace.run(
                client=client,
                model=model,
                max_iterations=max_iterations,
            )

    result = asyncio.run(_run())

    log.info("\nSaved to: %s", output)
    log.info("Task metadata: SELECT task, key, value FROM _task_meta")
    log.info("SQL trace: SELECT * FROM _trace")

    sys.exit(0 if result.success else 1)


@main.command()
@click.argument("db_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output database path",
)
@click.option(
    "--spec",
    "-s",
    default=None,
    help="Override spec module path (must be structurally compatible).",
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
@click.option(
    "--mode",
    type=click.Choice(["validate", "review"]),
    default="validate",
    help="'validate' skips tasks that pass; 'review' always invokes agents (default: validate).",
)
@click.option(
    "--reingest",
    is_flag=True,
    help="Re-run input callables for fresh data. Default when --spec is provided; "
    "without --spec, data is reused from the db automatically.",
)
def rerun(
    db_file: Path,
    output: Path,
    spec: str | None,
    model: str,
    max_iterations: int,
    quiet: bool,
    force: bool,
    reasoning_effort: str | None,
    mode: str,
    reingest: bool,
):
    """Rerun a workspace from a previous .db file.

    DB_FILE: A previous workspace .db file.

    \b
    By default, the recorded spec module is used and existing data is
    reused from the db. Use --spec to override with a new spec module
    (must be structurally compatible — same input names, task names,
    outputs), which also re-ingests fresh data from the new spec's
    input callables.

    \b
    Use --reingest to force re-ingestion even without --spec.

    \b
    Example:
        taskgraph rerun previous.db -o new.db
        taskgraph rerun previous.db -o new.db --mode review
        taskgraph rerun jan.db -o feb.db --spec my_app.specs.feb
        taskgraph rerun previous.db -o new.db --reingest
    """
    # Configure logging
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Determine ingestion behavior:
    # --spec provided -> reingest by default (new data paths)
    # no --spec       -> skip ingest by default (.db is self-contained)
    # --reingest      -> always reingest (explicit override)
    skip_ingest = not reingest and spec is None

    conn = duckdb.connect(str(db_file), read_only=True)
    meta = read_workspace_meta(conn)
    conn.close()

    if not meta:
        raise click.ClickException(
            f"{db_file} has no workspace metadata — not a Taskgraph workspace."
        )

    if spec:
        spec_module = spec
        log.info("Spec override: %s", spec_module)
    else:
        spec_module = meta.get("spec_module")
        if not spec_module:
            raise click.ClickException(
                f"{db_file} has no spec module reference. Use --spec to provide one."
            )

    try:
        spec_path = resolve_module_path(spec_module)
        repo_info = get_spec_repo_info(spec_path)
    except ValueError as e:
        raise click.ClickException(str(e))

    if repo_info.dirty:
        click.echo(
            f"WARNING: spec repo is dirty ({repo_info.root}); rerun is not strictly reproducible.",
            err=True,
        )

    if not spec:
        expected_commit = meta.get("spec_git_commit")
        if not expected_commit:
            raise click.ClickException(
                f"{db_file} has no recorded spec commit. Use --spec to provide one."
            )
        if repo_info.commit != expected_commit:
            raise click.ClickException(
                "Spec commit mismatch: expected "
                f"{expected_commit}, found {repo_info.commit}."
            )

    try:
        loaded = load_spec_from_module(spec_module)
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
        spec_source=loaded.get("spec_source"),
        spec_module=spec_module,
        spec_git_commit=repo_info.commit,
        spec_git_root=str(repo_info.root),
        spec_git_dirty=repo_info.dirty,
    )

    log.info("Rerun: %s", db_file)
    log.info("Output: %s", output)
    log.info("Model: %s", model)
    log.info("Mode: %s", mode)
    log.info("Spec: %s", spec_module)
    log.info("Spec commit: %s", repo_info.commit)
    log.info(
        "Ingestion: %s",
        "skipped (using existing data)" if skip_ingest else "fresh",
    )
    log.info("Tasks: %d\n", len(loaded["tasks"]))

    async def _run():
        async with OpenRouterClient(reasoning_effort=reasoning_effort) as client:
            return await workspace.rerun(
                source_db=db_file,
                client=client,
                model=model,
                max_iterations=max_iterations,
                mode=mode,
                skip_ingest=skip_ingest,
            )

    result = asyncio.run(_run())

    log.info("\nSaved to: %s", output)

    sys.exit(0 if result.success else 1)


@main.command("extract-spec")
@click.argument("db_file", type=click.Path(exists=True, path_type=Path))
@click.argument("out_file", type=click.Path(path_type=Path))
def extract_spec_cmd(db_file: Path, out_file: Path):
    """Extract the embedded spec source from a workspace .db file."""
    conn = duckdb.connect(str(db_file), read_only=True)
    meta = read_workspace_meta(conn)
    conn.close()

    if not meta:
        raise click.ClickException(
            f"{db_file} has no workspace metadata — not a Taskgraph workspace."
        )

    spec_source = meta.get("spec_source")
    if not spec_source:
        raise click.ClickException(f"{db_file} has no embedded spec source.")

    out_file.write_text(spec_source)
    click.echo(f"Extracted: {out_file}")


@main.command()
@click.option(
    "--spec",
    "-s",
    default=None,
    help="Spec module path (default: [tool.taskgraph].spec from pyproject.toml)",
)
def show(spec: str | None):
    """Show spec structure: inputs, DAG layers, task details.

    \b
    Example:
        taskgraph show --spec my_app.specs.main
        taskgraph show
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if spec is None:
        spec = _default_spec_module()

    spec_module = spec
    try:
        spec_path = resolve_module_path(spec_module)
        repo_info = get_spec_repo_info(spec_path)
    except ValueError as e:
        raise click.ClickException(str(e))

    if repo_info.dirty:
        click.echo(
            f"WARNING: spec repo is dirty ({repo_info.root}); output may not be reproducible.",
            err=True,
        )

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

    # --- DAG ---
    try:
        layers = resolve_dag(tasks)
    except ValueError as e:
        click.echo(f"\nDAG error: {e}")
        sys.exit(1)

    total_tasks = sum(len(layer) for layer in layers)
    click.echo(f"\nDAG ({len(layers)} layers, {total_tasks} tasks):\n")

    # Collect all task outputs for showing which inputs are external vs task-produced
    all_task_outputs = set()
    for t in tasks:
        all_task_outputs.update(t.outputs)

    for li, layer in enumerate(layers, 1):
        layer_names = " | ".join(t.name for t in layer)
        click.echo(f"  Layer {li} \u2500 {layer_names}")

        for t in layer:
            # If multiple tasks in layer, indent with task name header
            if len(layer) > 1:
                click.echo(f"    {t.name}")
                prefix = "      "
            else:
                prefix = "    "

            # Inputs — mark external (base table) vs task-produced
            if t.inputs:
                parts = []
                for inp in t.inputs:
                    if inp in all_task_outputs:
                        parts.append(inp)
                    else:
                        parts.append(f"{inp} (table)")
                click.echo(f"{prefix}in:  {', '.join(parts)}")
            else:
                click.echo(f"{prefix}in:  (none)")

            # Outputs — show column count if output_columns defined
            out_parts = []
            for o in t.outputs:
                cols = t.output_columns.get(o, [])
                if cols:
                    out_parts.append(f"{o} ({len(cols)} cols)")
                else:
                    out_parts.append(o)
            click.echo(f"{prefix}out: {', '.join(out_parts)}")

        click.echo()

    # --- Validation summary ---
    has_validation = False
    val_lines = []
    for t in tasks:
        parts = []
        if t.output_columns:
            n_schemas = len(t.output_columns)
            parts.append(f"{n_schemas} output schema{'s' if n_schemas > 1 else ''}")
        if t.validate_sql:
            n_sql = len(t.validate_sql)
            parts.append(f"{n_sql} SQL check{'s' if n_sql > 1 else ''}")
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
