# Taskgraph

**DAG scheduler for LLM agents on DuckDB.**

Define a pipeline of SQL and LLM nodes as a Python module. Taskgraph runs them concurrently, enforces per-node namespace isolation, validates outputs with declarative SQL constraints, and writes everything — inputs, outputs, and full execution trace — into a single portable `.db` file.

You define a "workspace spec" (a Python module) that declares:
- `NODES`: a list of nodes — each either a **source** node (`source`) or a **transform** node (`sql` or `prompt`). Nodes form a DAG via `depends_on`.
- `EXPORTS` (optional): functions that export files (CSV/XLSX/etc.) from the finished workspace

The result is one portable `.db` file containing the raw inputs, materialized outputs, validation results, and an execution trace (with view definitions in `_view_definitions`).

```
spec module -> ingest -> DAG -> nodes (concurrent) -> validate -> materialize -> export -> output.db
```

## What Problem It Solves

- **Reproducible, reviewable work**: outputs are materialized tables; the SQL used to define them is recorded via `_trace` / `_view_definitions`.
- **Strong guardrails**: agents can only `CREATE VIEW`/`CREATE MACRO` (no tables, no inserts) and are namespace-restricted per node.
- **Fast iteration**: views are late-binding during execution; after each node passes validation, its `{name}_*` views are materialized into tables.

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
duckdb output.db "SELECT table_name FROM duckdb_tables() WHERE internal = false ORDER BY 1"
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

- `SPEC_GUIDE.md` (spec writers): how to build spec packages, write good prompts, validation, and exports
- `AGENTS.md` (contributors): architecture, DB schema, key functions, run times and costs
