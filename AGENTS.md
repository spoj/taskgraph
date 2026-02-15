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

- `src/agent.py` — task agent: system prompt, SQL execution, namespace enforcement
- `src/agent_loop.py` — generic async agent loop with concurrent tool execution
- `src/api.py` — OpenRouter client. Connection pooling, cache_control on last message, reasoning_effort
- `src/task.py` — Task dataclass, DAG resolution (topo-sort via Kahn's algorithm), dependency graph, graph validation
- `src/workspace.py` — Workspace orchestrator: ingest inputs, resolve DAG, run tasks with greedy scheduling, rerun from previous .db
- `src/ingest.py` — Ingestion: DataFrame/list[dict]/dict[str,list] -> DuckDB with _row_id PK
- `src/spec.py` — shared spec loader (used by CLI and web), spec module resolution
- `scripts/cli.py` — CLI entry point: `taskgraph run`, `taskgraph rerun`, `taskgraph extract-spec`, `taskgraph show`

## Workspace Spec Contract

A Python module defining:
```python
INPUTS     = {"table_name": callable_or_data, ...}  # callable returns DataFrame/list[dict]/dict[str,list]
TASKS      = [{"name": ..., "prompt": ..., "inputs": [...], "outputs": [...]}, ...]
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

Task `prompt` must be a **string**.

**Allowed libraries in spec modules**: stdlib (pathlib, csv, json, etc.), polars, openpyxl.
No other third-party imports. Spec modules should be pure data + ingestion logic.

## Key Design Decisions

- **Workspace = single .db** — all data, views, metadata, trace in one file (DuckDB format)
- **Self-contained .db** — resolved prompts, structural fingerprint, and spec module reference are embedded in `_workspace_meta`. A db rerun requires the recorded spec module + git commit.
- **Agents only write SQL views and macros** — no tables, no inserts. Views auditable via `duckdb_views() WHERE internal = false`.
- **Namespace enforcement** — DuckDB `extract_statements` for statement type classification + regex for name extraction; each task can only CREATE/DROP views and macros with its declared outputs or `{name}_*` prefixed names
- **DAG is static** — declared upfront, deps resolved from output->input edges. Each task starts as soon as all its dependencies complete (greedy scheduling, not layer-by-layer). Layers still computed for display via `resolve_dag()`.
- **Failure isolation** — a failed task only blocks its downstream dependents, not the entire layer. Unrelated branches continue.
- **Result size cap** — SELECT results exceeding 30k chars (~20k tokens) are rejected with an error nudging the agent to use LIMIT. Configurable via `max_result_chars` on `execute_sql()`.
- **Reruns** — `taskgraph rerun prev.db -o new.db` copies a previous run's db, reuses existing data, validates (views auto-update via late-binding), and only invokes agents for tasks whose validation fails. Use `--spec` to override with a new spec module (re-ingests fresh data by default). Use `--reingest` to force re-ingestion without `--spec`.
- **Workspace fingerprint** — `_workspace_meta` table stores a structural fingerprint (input names/columns, task names/inputs/outputs/output_columns). Rerun compatibility is checked before re-use. Prompts and validate_sql are excluded (can evolve between runs).
- **SQL trace** in _trace table with task column for per-task filtering
- **Per-task metadata** in _task_meta table (composite PK: task + key)
- **Display**: `.` per SQL tool call, newline per tool round
- **Concurrency**: asyncio cooperative — true parallelism only at LLM API call level, DuckDB access naturally serialized on single thread
- **DuckDB ingestion** uses native DataFrame scan (`CREATE TABLE AS SELECT ... FROM df`) — no row-by-row executemany
- **DuckDB views filter** — always use `internal = false` to exclude system catalog views

## Self-Contained .db

Every `taskgraph run` embeds the resolved prompts and spec module reference in `_workspace_meta`:

| Key | Content |
|-----|---------|
| `fingerprint` | Structural identity (input names/columns, task names/inputs/outputs/output_columns) |
| `prompts` | `{task_name: prompt_text}` — frozen at run time |
| `spec_module` | Module path for the spec |
| `spec_git_commit` | Git commit hash for the spec repo |
| `spec_git_root` | Git repo root for the spec module |
| `spec_source` | Raw Python source of the spec module (optional) |
| `model` | Model used for this run |
| `timestamp` | ISO 8601 timestamp |
| `input_row_counts` | `{table_name: row_count}` (optional) |
| `source_db` | Path to source .db if this was a rerun (optional) |
| `rerun_mode` | `"validate"` or `"review"` if this was a rerun (optional) |

This enables the **rerun workflow**: a .db file is a complete, portable unit of work.

### Month-to-month workflow

```bash
# January: fresh run
taskgraph run --spec my_app.specs.jan -o jan.db

# Edit spec module and commit changes
git commit -am "feb spec"

# February: rerun with new spec, using Jan's views as starting point
taskgraph rerun jan.db -o feb.db --spec my_app.specs.feb
```

## Robustness Features

- **Token circuit breaker** (`agent_loop.py`): `DEFAULT_MAX_TOKENS = 20_000_000` (20M). Checked after each LLM call (prompt + completion combined). Returns `AgentResult(success=False)` if exceeded. Configurable via `max_tokens` param on `run_agent_loop()`.
- **Per-query timeout** (`agent.py`): `DEFAULT_QUERY_TIMEOUT_S = 30`. Uses `threading.Timer` + `conn.interrupt()`. Catches `duckdb.InterruptException`, connection stays usable. Configurable via `query_timeout_s` param on `execute_sql()`.
- **Output schema validation** (`task.py`): `output_columns: dict[str, list[str]]` on `Task`. Runs after view existence check, before `validate_sql`. Queries `information_schema.columns` and checks required columns are present. Error message includes actual columns for debugging.
- **Input validation** (`workspace.py`): `input_columns: dict[str, list[str]]` and `input_validate_sql: dict[str, list[str]]` on `Workspace`. Runs after ingestion, before tasks. Column check short-circuits before SQL checks. Callable errors caught with context. Empty tables logged as warnings.

## Justfile

Shorthand commands via [just](https://github.com/casey/just). All `recon` commands go through `uv run`.

| Command | Expands to |
|---------|-----------|
| `just run <args>` | `uv run taskgraph run <args>` |
| `just rerun <args>` | `uv run taskgraph rerun <args>` |
| `just show <args>` | `uv run taskgraph show <args>` |
| `just extract-spec <args>` | `uv run taskgraph extract-spec <args>` |
| `just inspect-xlsx <file> [sheet] [range]` | `uv run python scripts/inspect_xlsx.py <file> [sheet] [range]` |
| `just test [args]` | `uv run pytest tests/ [args]` |
| `just test-k <pattern>` | `uv run pytest tests/ -k "<pattern>" -v` |
| `just sync` | `uv sync` |
| `just lock` | `uv lock` |

## CLI Commands

### `taskgraph run`
Run a workspace spec: ingest inputs, resolve DAG, execute tasks, run exports.
```
just run --spec tests.single_task -o output.db
just run --spec tests.diamond_dag -o output.db -m anthropic/claude-sonnet-4 --reasoning-effort medium
```

### `taskgraph rerun`
Rerun a workspace from a previous .db file. By default uses the recorded spec module and existing data. Use `--spec` to override with a new spec module (re-ingests fresh data by default).
```
just rerun previous.db -o new.db                          # rerun with embedded spec + existing data
just rerun previous.db -o new.db --reingest                # force re-ingestion from spec callables
just rerun jan.db -o feb.db --spec my_app.specs.feb        # override spec (month-to-month)
just rerun previous.db -o new.db --mode review             # always invoke agents
```

### `taskgraph extract-spec`
Extract the embedded spec source from a workspace .db file.
```
just extract-spec output.db extracted_spec.py
```

### `taskgraph show`
Visualize a spec's structure: inputs with column specs, DAG layers with concurrent task grouping, per-task inputs/outputs, validation summary, graph errors, exports.
```
just show --spec my_app.specs.main
```
