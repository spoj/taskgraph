# Taskgraph

Taskgraph runs deterministic SQL and prompt-based transforms against your data and writes auditable DuckDB SQL views into a single workspace database.

You define a "workspace spec" (a Python module) that declares:
- `INPUTS`: how to load data into DuckDB tables
- `TASKS`: a DAG of tasks; each task runs via `sql` (deterministic SQL) or `prompt` (LLM transform) and produces one or more SQL views. Optional `validate_sql` runs after the transform to create validation views.
- `EXPORTS` (optional): functions that materialize reports (CSV/XLSX/etc.) from the finished workspace

The result is one portable `.db` file containing the raw inputs, all agent-created views, validation results, and an execution trace.

```
spec module -> ingest -> DAG -> tasks (concurrent) -> validate -> export -> output.db
```

## What Problem It Solves

- **Reproducible, reviewable work**: outputs are SQL view definitions stored in the database.
- **Strong guardrails**: agents can only `CREATE VIEW`/`CREATE MACRO` (no tables, no inserts) and are namespace-restricted per node.
- **Fast iteration**: views are late-binding; tweak upstream logic and downstream results update automatically.

## Quick Start

Prereqs: Python 3.13+ and [uv](https://docs.astral.sh/uv/). You also need an OpenRouter API key in `OPENROUTER_API_KEY`.

```bash
uv add taskgraph
uv sync

# Scaffold a default spec at specs/main.py (module: specs.main)
uv run taskgraph init

# Run it (output name is auto-generated if -o is omitted)
uv run taskgraph run

# Inspect results
duckdb output.db "SELECT * FROM duckdb_views() WHERE internal = false"
```

If you already have a spec module in your package:

```bash
uv run taskgraph run --spec my_app.specs.main
```

## Spec Discovery

When `--spec` is not provided, Taskgraph looks for:
- `[tool.taskgraph].spec` in `pyproject.toml`, otherwise
- `specs.main` (only if `specs/main.py` exists)

## Docs

- `SPEC_GUIDE.md` (spec writers): how to build spec packages, write good tasks, validation, and exports
