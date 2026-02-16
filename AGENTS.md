# Agent Notes

## Model Configuration

**Default: `openai/gpt-5.2` with `reasoning_effort=low`**

OpenRouter pricing (per million tokens):
- Input: $1.75
- Output: $14.00
- Cache read: $0.175 (10x cheaper than input)

## Architecture

Single DuckDB database as shared workspace. Tasks form a DAG, each runs as an
independent agent writing namespace-enforced SQL views.

- `src/agent.py` — task agent: SQL execution, optional LLM repair, namespace enforcement
- `src/agent_loop.py` — generic async agent loop with concurrent tool execution
- `src/api.py` — OpenRouter client. Connection pooling, cache_control on last message, reasoning_effort
- `src/diff.py` — View catalog diffing: before/after snapshots of `duckdb_views().sql`, structured change reporting (created/modified/dropped), persistence to `_changes` table
- `src/task.py` — Task dataclass, DAG resolution (topo-sort via Kahn's algorithm), dependency graph, graph validation
- `src/workspace.py` — Workspace orchestrator: ingest inputs, resolve DAG, run tasks with greedy scheduling, per-task change tracking
- `src/ingest.py` — Ingestion: DataFrame/list[dict]/dict[str,list] -> DuckDB with _row_id PK
- `src/spec.py` — shared spec loader, spec module resolution
- `scripts/cli.py` — CLI entry point: `tg init`, `tg run`, `tg show`

## Workspace Spec Contract

A Python module defining:
```python
INPUTS     = {"table_name": callable_or_data, ...}  # callable returns DataFrame/list[dict]/dict[str,list]
TASKS      = [{"name": ..., "intent": ..., "sql": ..., "inputs": [...], "outputs": [...]}, ...]  # or sql_strict
EXPORTS    = {"report.xlsx": fn(conn, path), ...}     # optional export functions
```

INPUTS values can be **simple** (callable or raw data) or **rich** (dict with `"data"` key + optional validation):

```python
INPUTS = {
    "invoices": {
        "data": load_invoices,                                   # callable or raw data
        "columns": ["id", "amount", "date"],                     # optional: required columns
        "validate_sql": ["SELECT id FROM invoices WHERE amount IS NULL"],  # optional: must return 0 rows
    },
    "rates": load_rates,  # simple — no validation needed
}
```

At the `load_spec` boundary, if a value is a dict with a `"data"` key, `columns` and `validate_sql` are extracted per-input and passed to the Workspace as `input_columns: dict[str, list[str]]` and `input_validate_sql: dict[str, list[str]]`.

Task `sql` or `sql_strict` must be a **string**. `intent` is required for `sql` tasks.
`repair_on_warn` defaults to `true` — validation view warnings trigger LLM repair for `sql` tasks.

**Allowed libraries in spec modules**: stdlib (pathlib, csv, json, etc.), polars, openpyxl.
No other third-party imports. Spec modules should be pure data + ingestion logic.

## Key Design Decisions

- **Workspace = single .db** — all data, views, metadata, trace in one file (DuckDB format).
- **Agents only write SQL views and macros** — no tables, no inserts. Views auditable via `duckdb_views() WHERE internal = false`.
- **Namespace enforcement** — DuckDB `extract_statements` for statement type classification + regex for name extraction; each task can only CREATE/DROP views and macros with its declared outputs or `{name}_*` prefixed names
- **DAG is static** — declared upfront, deps resolved from output->input edges. Each task starts as soon as all its dependencies complete (greedy scheduling, not layer-by-layer). Layers still computed for display via `resolve_dag()`.
- **Failure isolation** — a failed task only blocks its downstream dependents, not the entire layer. Unrelated branches continue.
- **Result size cap** — SELECT results exceeding 30k chars (~20k tokens) are rejected with an error nudging the agent to use LIMIT. Configurable via `max_result_chars` on `execute_sql()`.
- **SQL trace** in _trace table with task column for per-task filtering
- **Per-task metadata** in _task_meta table (composite PK: task + key)
- **Display**: `.` per SQL tool call, newline per tool round
- **Concurrency**: asyncio cooperative — true parallelism only at LLM API call level, DuckDB access naturally serialized on single thread
- **DuckDB ingestion** uses native DataFrame scan (`CREATE TABLE AS SELECT ... FROM df`) — no row-by-row executemany
- **DuckDB views filter** — always use `internal = false` to exclude system catalog views

## Robustness Features

- **Token circuit breaker** (`agent_loop.py`): `DEFAULT_MAX_TOKENS = 20_000_000` (20M). Checked after each LLM call (prompt + completion combined). Returns `AgentResult(success=False)` if exceeded. Configurable via `max_tokens` param on `run_agent_loop()`.
- **Per-query timeout** (`agent.py`): `DEFAULT_QUERY_TIMEOUT_S = 30`. Uses `threading.Timer` + `conn.interrupt()`. Catches `duckdb.InterruptException`, connection stays usable. Configurable via `query_timeout_s` param on `execute_sql()`.
- **Output schema validation** (`task.py`): `output_columns: dict[str, list[str]]` on `Task`. Runs after view existence check, before validation view enforcement. Queries `information_schema.columns` and checks required columns are present.
- **Task validation views** (`task.py`): tasks can declare one or more outputs named `{task}__validation` / `{task}__validation_*` with columns `status`, `message`. Any row with status='fail' fails the task.
- **Input validation** (`workspace.py`): `input_columns: dict[str, list[str]]` and `input_validate_sql: dict[str, list[str]]` on `Workspace`. Runs after ingestion, before tasks. Column check short-circuits before SQL checks. Callable errors caught with context. Empty tables logged as warnings.

## Justfile

Shorthand commands via [just](https://github.com/casey/just). All `recon` commands go through `uv run`.

| Command | Expands to |
|---------|-----------|
| `just run <args>` | `uv run taskgraph run <args>` |
| `just show <args>` | `uv run taskgraph show <args>` |
| `just inspect-xlsx <file> [sheet] [range]` | `uv run python scripts/inspect_xlsx.py <file> [sheet] [range]` |
| `just test [args]` | `uv run pytest tests/ [args]` |
| `just test-k <pattern>` | `uv run pytest tests/ -k "<pattern>" -v` |
| `just sync` | `uv sync` |
| `just lock` | `uv lock` |

## CLI Commands

### `taskgraph init`
Initialize a Taskgraph spec in the current directory. Creates `pyproject.toml`, `specs/main.py`, `specs/__init__.py`, `SPEC_GUIDE.md`, `.env`, and `.gitignore`. Project name is derived from the directory name (slugified). Scaffold spec uses `sql_strict` so `tg run` works immediately without an API key.
```
tg init          # create scaffold (skip existing files)
tg init --force  # overwrite existing files
```

### `taskgraph run`
Run a workspace spec: ingest inputs, resolve DAG, execute tasks, run exports. The `--spec` argument accepts both module paths (`specs.main`) and file paths (`specs/main.py`, `./specs/main.py`); file paths are auto-resolved to module paths.
```
just run --spec tests.single_task -o output.db
just run --spec specs/main.py -o output.db          # file path auto-resolved
just run --spec tests.diamond_dag -o output.db -m anthropic/claude-sonnet-4 --reasoning-effort medium
```

After each task completes, view changes are reported inline:
- `+ view_name  N cols, N rows` — created views
- `~ view_name` with compact SQL diff — modified views
- `- view_name` — dropped views

Changes are also persisted to the `_changes` table in the output `.db` for later querying.

### `taskgraph show`
Visualize a spec's structure: inputs with column specs, DAG layers with concurrent task grouping, per-task inputs/outputs, validation summary, graph errors, exports. Accepts file paths like `run`.
```
just show --spec my_app.specs.main
just show --spec specs/main.py
```
