# Taskgraph

Taskgraph runs LLM agents that execute data tasks by writing SQL views inside a single DuckDB database. You declare inputs, tasks, and validation in a Python spec module inside your app package. Taskgraph ingests the data, resolves a DAG of tasks, runs an LLM agent per task, validates the output, and saves everything in a single `.db` file.
CLI command: `taskgraph`.

## How it works

1. **Spec** -- a Python module declares `INPUTS` (data sources), `TASKS` (what the agents should do), and optional `EXPORTS` (files to produce).
2. **Ingest** -- data is loaded into a DuckDB database. Each table gets a `_row_id` primary key.
3. **DAG** -- tasks are topologically sorted by input/output edges. Independent tasks run concurrently.
4. **Agents** -- each task gets its own LLM agent that explores the data and creates SQL views. Agents are constrained to their declared namespace (no cross-task interference) and can only write views and macros (no tables, no inserts).
5. **Validate** -- when an agent finishes, its outputs are checked: view existence, column schema, and optional SQL assertions. Failures are fed back for self-correction.
6. **Export** -- optional export functions (CSV, XLSX, etc.) run after all tasks pass.
7. **Save** -- the `.db` file contains all data, views, SQL trace, metadata, and optional spec source. Reruns are driven by the recorded spec module + git commit.

```
spec module ──> ingest ──> DAG ──> agents (concurrent) ──> validate ──> export ──> output.db
```

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
    git clone <repo-url> && cd taskgraph
uv sync
```

For dev tooling (type checking, etc.):

```bash
uv sync --extra dev
```

Set your API key in `.env`:

```
OPENROUTER_API_KEY=sk-or-...
```

## Quick start

### Scaffold a spec (fast path)

Inside your app repo (must have `pyproject.toml`):

```bash
uv add taskgraph
uv sync
taskgraph init
taskgraph run -o output.db
```

This creates `specs/main.py` (import path: `specs.main`).

### Minimal spec

```python
# my_app/specs/main.py
INPUTS = {
    "employees": {
        "data": [
            {"id": 1, "name": "Alice", "department": "Engineering", "salary": 120000},
            {"id": 2, "name": "Bob", "department": "Engineering", "salary": 110000},
            {"id": 3, "name": "Carol", "department": "Sales", "salary": 95000},
            {"id": 4, "name": "Dave", "department": "Sales", "salary": 90000},
            {"id": 5, "name": "Eve", "department": "Engineering", "salary": 130000},
            {"id": 6, "name": "Frank", "department": "HR", "salary": 85000},
        ],
        "columns": ["id", "name", "department", "salary"],
    },
}

TASKS = [
    {
        "name": "summarize",
        "prompt": (
            "Summarize employees by department. Create a view 'department_summary' with:\n"
            "- department: department name\n"
            "- headcount: number of employees\n"
            "- total_salary: sum of salaries\n"
            "- avg_salary: average salary (rounded to nearest integer)\n"
            "Order by department name.\n"
        ),
        "inputs": ["employees"],
        "outputs": ["department_summary"],
        "output_columns": {
            "department_summary": ["department", "headcount", "total_salary", "avg_salary"],
        },
    },
]
```

Run it:

```bash
    taskgraph run --spec my_app.specs.main -o output.db
```

Query the results:

```bash
duckdb output.db "SELECT * FROM department_summary"
```

### Diamond DAG

Tasks can form arbitrary DAGs. Independent tasks run concurrently:

```python
#        prep
#       /    \
#   sales   costs
#       \    /
#       report

TASKS = [
    {"name": "prep",   "inputs": ["transactions"],                "outputs": ["prepared_sales", "prepared_costs"]},
    {"name": "sales",  "inputs": ["prepared_sales", "products"],  "outputs": ["sales_summary"]},
    {"name": "costs",  "inputs": ["prepared_costs"],              "outputs": ["costs_summary"]},
    {"name": "report", "inputs": ["sales_summary", "costs_summary"], "outputs": ["profit_report"]},
]
```

Here `sales` and `costs` run in parallel after `prep` completes. `report` waits for both.

## Spec format

A spec module defines three things:

### INPUTS

A dict mapping table names to data sources.

**Simple format** -- value is a callable or raw data:

```python
INPUTS = {
    "sales": load_sales,                              # callable returning DataFrame
    "rates": [{"ccy": "USD", "rate": 1.0}, ...],     # list[dict]
    "config": {"key": ["a", "b"], "val": [1, 2]},    # dict[str, list]
}
```

**Rich format** -- value is a dict with `"data"` key plus optional validation:

```python
INPUTS = {
    "invoices": {
        "data": load_invoices,
        "columns": ["id", "amount", "date"],                     # required columns
        "validate_sql": [
            "SELECT 'null amount' FROM invoices WHERE amount IS NULL",  # must return 0 rows
        ],
    },
}
```

Accepted data types: `polars.DataFrame`, `list[dict]`, `dict[str, list]`.
Allowed imports in spec modules: stdlib (`pathlib`, `csv`, `json`, etc.), `polars`, `openpyxl`.

### TASKS

A list of task dicts (or `Task` objects):

```python
TASKS = [
    {
        "name": "match",                                      # unique name
        "prompt": "Match invoices to payments by amount...",   # string
        "inputs": ["invoices", "payments"],                    # tables or views from other tasks
        "outputs": ["output"],                                 # views the agent must create
        "output_columns": {"output": ["left_id", "right_id"]},  # optional: required columns
        "validate_sql": [                                       # optional: assertions (0 rows = pass)
            "SELECT * FROM output WHERE left_id IS NULL AND right_id IS NULL",
        ],
    },
]
```

**Namespace enforcement**: each task can only create views and macros named as its declared `outputs` or prefixed with `{task_name}_`. This prevents cross-task interference.

### EXPORTS (optional)

A dict mapping filenames to export functions:

```python
def export_report(conn, path):
    import polars as pl
    df = pl.read_database("SELECT * FROM profit_report", conn)
    df.write_csv(path)

EXPORTS = {
    "report.csv": export_report,
}
```

Export functions receive the open DuckDB connection and a `Path` to write to. Export errors are captured per-file and don't crash the run.

## CLI reference

All commands can be run with `taskgraph <command>` or via the Justfile with `just <command>`.

### `taskgraph run`

Run a workspace spec from scratch.

```bash
taskgraph run --spec my_app.specs.main -o output.db
taskgraph run -o output.db -m anthropic/claude-sonnet-4 --reasoning-effort medium
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | required | Output `.db` file path |
| `-s, --spec` | `tool.taskgraph.spec` | Spec module path (if unset: `specs.main` when present) |
| `-m, --model` | `openai/gpt-5.2` | LLM model via OpenRouter |
| `--reasoning-effort` | `low` | `low`, `medium`, or `high` |
| `--max-iterations` | `200` | Max agent iterations per task |
| `-q, --quiet` | off | Suppress verbose output |
| `-f, --force` | off | Overwrite output file without prompting |

### `taskgraph rerun`

Rerun from a previous `.db` file. By default uses the recorded spec module + git commit and existing data. Only re-invokes agents for tasks whose validation fails.

```bash
# Rerun with recorded spec module
taskgraph rerun previous.db -o new.db

# Force re-ingestion of data from spec callables
taskgraph rerun previous.db -o new.db --reingest

# Override spec (month-to-month workflow)
taskgraph rerun jan.db -o feb.db --spec my_app.specs.feb

# Always invoke agents (even for passing tasks)
taskgraph rerun previous.db -o new.db --mode review
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | required | Output `.db` file path |
| `-s, --spec` | recorded module | Override spec module (must be structurally compatible) |
| `--mode` | `validate` | `validate`: skip passing tasks. `review`: always invoke agents |
| `--reingest` | off | Force re-ingestion from spec callables |
| `-m, --model` | same as original | LLM model |
| `--reasoning-effort` | `low` | Reasoning effort level |
| `--max-iterations` | `200` | Max agent iterations per task |
| `-q, --quiet` | off | Suppress verbose output |
| `-f, --force` | off | Overwrite output file without prompting |

### `taskgraph show`

Display a spec's structure: inputs, DAG layers, task details, validation, exports.

```bash
taskgraph show --spec my_app.specs.main
taskgraph show
```

### `taskgraph extract-spec`

Extract the embedded spec source from a `.db` file back to Python.

```bash
taskgraph extract-spec output.db extracted_spec.py
```

## Reruns and month-to-month workflow

Every `.db` file stores the data, all views, the resolved prompts, the spec module reference, and a structural fingerprint. This enables reproducible reruns and month-to-month workflows:

```bash
# January: fresh run
taskgraph run --spec my_app.specs.jan -o jan.db

# Update spec module + commit changes
git commit -am "feb spec"

# February: rerun with new spec, using Jan's views as starting point
taskgraph rerun jan.db -o feb.db --spec my_app.specs.feb
```

The rerun copies the previous `.db`, optionally re-ingests fresh data, validates existing views against the (possibly updated) spec, and only invokes agents for tasks that fail validation. If the data changed but the SQL is still valid, no LLM calls are needed.

**Structural compatibility**: the fingerprint covers input names/columns and task names/inputs/outputs/output_columns. Prompts and `validate_sql` are excluded and can evolve between runs.

## Validation

Validation runs automatically when an agent finishes and has three stages:

1. **View existence** -- all declared `outputs` must exist as views.
2. **Column schema** -- if `output_columns` is specified, those columns must be present in the view.
3. **SQL assertions** -- if `validate_sql` is specified, each query must return 0 rows. Each returned row is treated as an error message.

Validation failures are fed back to the agent, which gets another chance to fix the SQL. The loop continues until validation passes or the iteration limit is reached.

## Architecture

```
src/
  agent.py        # Task agent: system prompt, SQL execution, namespace enforcement
  agent_loop.py   # Generic async agent loop with concurrent tool execution
  api.py          # OpenRouter client with retry, cache control, reasoning effort
  ingest.py       # Data ingestion: DataFrame/list[dict]/dict -> DuckDB
   spec.py         # Spec loader: parse Python spec modules
  task.py         # Task dataclass, DAG resolution (Kahn's algorithm)
  workspace.py    # Workspace orchestrator: ingest, DAG, run tasks, rerun, exports

scripts/
  cli.py          # CLI entry point (click)
  inspect_xlsx.py # Excel file inspector

web/
  app.py          # FastAPI web interface with SSE streaming
  static/
    index.html    # Single-page dark terminal UI
```

### Design principles

- **Agents only write SQL views** -- no tables, no inserts. Views are auditable and late-binding (updating upstream data automatically propagates).
- **Single DuckDB file** -- all data, views, metadata, and trace in one portable file.
- **Namespace enforcement** -- uses DuckDB's `extract_statements` for statement classification and regex for name extraction. Each task is sandboxed to its declared outputs.
- **Greedy DAG scheduling** -- each task starts as soon as its specific dependencies complete, not when the entire layer finishes.
- **Failure isolation** -- a failed task only blocks its downstream dependents. Unrelated branches continue.

### Robustness

- **Token circuit breaker**: 20M token hard limit per agent.
- **Per-query timeout**: 30 seconds via `conn.interrupt()`. Connection stays usable after timeout.
- **Result size cap**: SELECT results exceeding 30k characters are rejected (nudges the agent to use `LIMIT`).
- **Retry with backoff**: API calls retry with exponential backoff and jitter, respecting `Retry-After` headers.

## Web interface

A FastAPI app with SSE-streamed output:

```bash
uv run uvicorn web.app:app
```

Features: built-in spec selection, data file drag-and-drop, model selection, live streaming console, download of `.db` and exported files.

## Development

```bash
# Install dependencies
uv sync

# Run tests (no LLM calls required)
just test

# Run a specific test
just test-k "test_diamond"

# Inspect an Excel file
just inspect-xlsx data.xlsx "Sheet1" "A1:D10"
```

### Justfile shortcuts

| Command | Description |
|---------|-------------|
| `just run <args>` | `taskgraph run` |
| `just rerun <args>` | `taskgraph rerun` |
| `just show <args>` | `taskgraph show` |
| `just extract-spec <args>` | `taskgraph extract-spec` |
| `just inspect-xlsx <file> [sheet] [range]` | Excel file inspector |
| `just test [args]` | `pytest tests/` |
| `just test-k <pattern>` | `pytest tests/ -k <pattern>` |
| `just sync` | `uv sync` |
| `just lock` | `uv lock` |

## Further reading

- [SPEC_GUIDE.md](SPEC_GUIDE.md) -- comprehensive guide for writing workspace specs
- [USE_CASES.md](USE_CASES.md) -- use cases, positioning, and design rationale
- Model eval logs and domain datasets live outside this repo
