# Agent Notes

## Model Configuration

**Default: `openai/gpt-5.2` with `reasoning_effort=low`**

OpenRouter pricing (per million tokens):
- Input: $1.75
- Output: $14.00
- Cache read: $0.175 (10x cheaper than input)

## Architecture

Single DuckDB database as shared workspace. Nodes form a DAG, each runs as an
independent agent writing namespace-enforced SQL views.

A **Node** is the single unit of work. Every node has a `name` and exactly one
execution mode — `source`, `sql`, or `prompt`:

- **source** — ingests data (callable, raw data, or file path) into table `{name}`.
- **sql** — executes deterministic SQL creating `{name}_*` views.
- **prompt** — runs an LLM agent creating `{name}_*` views.

There is no separate "inputs" concept — source nodes live in the same DAG,
share the same validation and materialization flow, and are scheduled by the
same greedy DAG executor as sql/prompt nodes.

### Source files

- `src/task.py` — `Node` dataclass, `_NO_SOURCE` sentinel, DAG resolution (topo-sort via Kahn's algorithm), dependency graph (`resolve_deps`), graph validation (`validate_graph`), output validation, validation view enforcement
- `src/agent.py` — node agent: prompt-based SQL transform, SQL execution with namespace enforcement, `persist_node_meta`, `run_node_agent`, `run_sql_node`, `run_validate_sql`, `validate_node_complete`
- `src/agent_loop.py` — generic async agent loop with concurrent tool execution
- `src/api.py` — OpenRouter client. Connection pooling, cache_control on last message (Anthropic only), reasoning_effort
- `src/diff.py` — View catalog diffing: before/after snapshots of `duckdb_views().sql`, structured change reporting (created/modified/dropped), persistence to `_changes` table, terminal formatting
- `src/namespace.py` — `Namespace` class: DDL name enforcement for nodes and validation. Factory methods `for_node()`, `for_validation()`, `for_source_validation()` encode naming conventions. Single `check_name()` entry point used by `is_sql_allowed()`.
- `src/sql_utils.py` — shared SQL utilities: parser connection, statement splitting, column schema queries, CREATE name extraction
- `src/workspace.py` — Workspace orchestrator: resolve DAG, run nodes with greedy scheduling, per-node change tracking, view materialization, unified post-execution flow
- `src/ingest.py` — Ingestion: DataFrame/list[dict]/dict[str,list] or file paths -> DuckDB with _row_id PK
- `src/spec.py` — spec loader: parses `NODES` list from Python modules, validates fields, resolves file paths
- `scripts/cli.py` — CLI entry point: `tg init`, `tg run`, `tg show`

## Workspace Spec Contract

A Python module defining a single `NODES` list (and optional `EXPORTS`):

```python
NODES = [
    # Source node — data ingestion
    {
        "name": "invoices",
        "source": "data/invoices.xlsx#Sheet1",   # callable, raw data, or file path
        "columns": ["id", "amount", "date"],      # optional: required columns
        "validate_sql": "CREATE OR REPLACE VIEW invoices__validation AS ...",  # optional
    },
    # Source node — callable
    {
        "name": "rates",
        "source": load_rates,  # callable returning DataFrame/list[dict]/dict[str,list]
    },
    # SQL node — deterministic transform
    {
        "name": "prep",
        "depends_on": ["invoices"],
        "sql": "CREATE OR REPLACE VIEW prep_clean AS SELECT * FROM invoices WHERE amount > 0",
        "output_columns": {"prep_clean": ["id", "amount"]},
    },
    # Prompt node — LLM-driven transform
    {
        "name": "match",
        "depends_on": ["prep", "rates"],
        "prompt": "Match invoices against rates...",
        "output_columns": {"match_results": ["id", "score"]},
        "validate_sql": "CREATE OR REPLACE VIEW match__validation AS ...",
    },
]

EXPORTS = {"report.xlsx": fn(conn, path), ...}  # optional export functions
```

### Node fields

All nodes: `name`, `depends_on`, `validate_sql`.
Source nodes: `source`, `columns`.
SQL/prompt nodes: `sql` OR `prompt`, `output_columns`.

- `name` — unique identifier. For source nodes, this becomes the table name; for sql/prompt nodes it is the namespace prefix (views must be `{name}_*`).
- `depends_on` — list of node names that must complete before this node runs.
- `source` — data source for ingestion: callable (returns DataFrame/list[dict]/dict[str,list]), raw data, or file path string.
- `sql` — deterministic SQL statements (string). Exactly one of `sql`/`prompt` required for transform nodes.
- `prompt` — LLM objective text (string).
- `columns` — (source nodes only) required column names checked after ingestion.
- `output_columns` — (sql/prompt nodes only) maps `view_name -> [required_columns]`. Keys define which views must exist; values define required columns per view. Keys must start with `{name}_`.
- `validate_sql` — SQL to create `{name}__validation*` views. Works for all node types.

**Allowed libraries in spec modules**: stdlib (pathlib, csv, json, etc.), polars, openpyxl.
No other third-party imports. Spec modules should be pure data + ingestion logic.

## Key Design Decisions

- **Workspace = single .db** — all data, views, metadata, trace in one file (DuckDB format).
- **Unified node model** — no separate "inputs" and "tasks". Source, SQL, and prompt nodes share the same DAG, validation, and materialization machinery. A single `NODES` list in specs replaces the old separate `INPUTS` dict + `TASKS` list.
- **Agents only write SQL views and macros** — no tables, no inserts. After node completion, views are materialized as tables. Original SQL is in `_trace`; the `_view_definitions` view (derived from `_trace`) provides lineage queries. Intermediate `{name}_*` views stay as views for debuggability.
- **Write namespace rule** — sql/prompt nodes can only create views named `{name}_*` (underscore required — bare `{name}` is NOT a valid view name). Source nodes create a table `{name}` via ingestion (not SQL).
- **Namespace enforcement** — `Namespace` class (`src/namespace.py`) owns DDL name rules. Factory methods: `for_node(node)` (`{name}_*` prefix, validation views forbidden), `for_validation(node)` (`{name}__validation*`), `for_source_validation(source_name)` (`{input}__validation*`). `is_sql_allowed()` uses DuckDB `extract_statements` for statement type classification + regex for name extraction, then delegates to `namespace.check_name()`. Name extraction failure (None) is blocked when namespace is set.
- **DAG is static** — declared upfront, deps resolved from `depends_on` edges. Each node starts as soon as all its dependencies complete (greedy scheduling, not layer-by-layer). Layers still computed for display via `resolve_dag()`.
- **Failure isolation** — a failed node only blocks its downstream dependents, not the entire layer. Unrelated branches continue.
- **Result size cap** — SELECT results exceeding 30k chars (~20k tokens) are rejected with an error nudging the agent to use LIMIT. Configurable via `max_result_chars` on `execute_sql()`.
- **`_NO_SOURCE` sentinel** — since `source` can be any value (callable, list, dict, string, even `None`), a sentinel object `_NO_SOURCE = object()` in `task.py` distinguishes "not provided" from valid data. `node.is_source()` checks `self.source is not _NO_SOURCE`.
- **Display**: `.` per SQL tool call at DEBUG level only
- **Concurrency**: asyncio cooperative — true parallelism only at LLM API call level, DuckDB access naturally serialized on single thread
- **DuckDB ingestion** uses native DataFrame scan (`CREATE TABLE AS SELECT ... FROM df`) — no row-by-row executemany
- **DuckDB views filter** — always use `internal = false` to exclude system catalog views
- **DuckDB tables filter** — `duckdb_tables() WHERE internal = false` for materialized outputs; `validate_node_complete()` checks both views and tables
- **Agent system prompt DuckDB dialect** — covers: QUALIFY, GROUP BY ALL, UNION BY NAME, SUMMARIZE, lists/lambdas, fuzzy matching, EXCLUDE/REPLACE, ASOF JOIN, PIVOT/UNPIVOT, COLUMNS() expressions, date functions (date_diff, date_trunc, strftime, make_date), string functions (regexp_extract, regexp_replace, string_split, string_agg, concat_ws), JSON arrow syntax (->>, json_extract_string, json_group_array)

## DB Schema

### Tables

| Table | PK | Columns | Purpose |
|-------|-----|---------|---------|
| `_node_meta` | `node` | `node VARCHAR`, `meta_json VARCHAR` | Per-node run metadata (model, iterations, tokens, elapsed, validation status) |
| `_trace` | `id` (sequence) | `id`, `timestamp`, `node VARCHAR`, `source VARCHAR`, `query`, `success`, `error`, `row_count`, `elapsed_ms` | SQL execution log |
| `_workspace_meta` | `key` | `key VARCHAR`, `value VARCHAR` | Workspace-level metadata (model, prompts, timing, spec info) |
| `_changes` | — | `node VARCHAR`, `view_name`, `kind`, `sql_before`, `sql_after`, `cols_before`, `cols_after`, `rows_before`, `rows_after` | View change history |

### Derived views

| View | Columns | Source |
|------|---------|--------|
| `_view_definitions` | `node VARCHAR`, `view_name`, `sql` | Derived from `_trace` — last successful CREATE VIEW per view name (DROP-aware) |

### Trace source values

| Value | Origin |
|-------|--------|
| `agent` | LLM agent tool calls (prompt nodes) |
| `sql_node` | Deterministic SQL node execution |
| `node_validation` | `validate_sql` execution |
| `input_validation` | (reserved — referenced in docstrings only) |

### Workspace metadata keys

| Key | Value |
|-----|-------|
| `meta_version` | `"2"` |
| `created_at_utc` | ISO timestamp |
| `taskgraph_version` | Package version |
| `python_version` | Python version |
| `platform` | Platform string |
| `node_prompts` | JSON: `{node_name: prompt_text}` for prompt nodes |
| `llm_model` | Model identifier |
| `llm_reasoning_effort` | Reasoning effort level (optional) |
| `llm_max_iterations` | Max iterations (optional) |
| `inputs_row_counts` | JSON: `{source_name: count}` |
| `inputs_schema` | JSON: `{source_name: [{name, type}]}` |
| `run` | JSON: `{"mode": "run"}` |
| `spec` | JSON: `{"module": "..."}` (optional) |
| `exports` | JSON: `{"attempted": bool, "results": {...}}` |

## Post-Execution Flow

All node types go through the same unified post-execution flow in `run_one()` (inside `Workspace.run()`):

1. **Execute** — type-specific: ingest (source), run SQL (sql), run agent (prompt).
2. **Validate** — `validate_node_complete(conn, node)`: validates outputs, runs validate_sql, checks validation views. Identical call for ALL node types.
3. **Snapshot** — `snapshot_views()` taken before materialization to capture view changes.
4. **Materialize** — `materialize_node_outputs(conn, node)`: discovers all `{name}_*` views and materializes them. Source nodes simply have no `{name}_*` output views, so the unified code naturally handles them.
5. **Persist metadata** — `persist_node_meta(conn, node_name, meta)`: writes per-node run metadata for ALL node types.

No per-type `_post_execute()` or `_materialize_node()` wrappers. The `run_sql_node()` and `run_node_agent()` functions ONLY execute their work — they do NOT validate or persist metadata.

## Robustness Features

- **Token circuit breaker** (`agent_loop.py`): `DEFAULT_MAX_TOKENS = 20_000_000` (20M). Checked after each LLM call (prompt + completion combined). Returns `AgentResult(success=False)` if exceeded. Configurable via `max_tokens` param on `run_agent_loop()`.
- **Per-query timeout** (`agent.py`): `DEFAULT_QUERY_TIMEOUT_S = 30`. Uses `threading.Timer` + `conn.interrupt()`. Catches `duckdb.InterruptException`, connection stays usable. Configurable via `query_timeout_s` param on `execute_sql()`.
- **Output schema validation** (`task.py`): `output_columns: dict[str, list[str]]` on `Node`. Runs after view existence check, before validation view enforcement. Queries `information_schema.columns` and checks required columns are present.
- **Node validation views** (`task.py`): nodes can declare `validate_sql` that creates `{name}__validation` / `{name}__validation_*` views with columns `status`, `message`. Any row with status='fail' fails the node.
- **Source column validation** (`task.py`): source nodes declare `columns: list[str]` — checked after ingestion against the actual table schema.

## Key Functions and Classes

### `src/task.py`

- `Node` — dataclass: `name`, `depends_on`, `source`, `sql`, `prompt`, `columns`, `output_columns`, `validate_sql`
- `Node.node_type()` — returns `"source"`, `"sql"`, or `"prompt"`
- `Node.is_source()` — `self.source is not _NO_SOURCE`
- `Node.validate_outputs(conn)` — unified: dispatches to `_validate_source_columns` or `_validate_output_columns`
- `Node.validate_validation_views(conn)` — check validation view schema and fail rows
- `resolve_deps(nodes)` — build dependency graph `{name: set_of_deps}`
- `resolve_dag(nodes)` — topo-sort into execution layers
- `validate_graph(nodes)` — structural validation (refs, cycles, namespace, cross-type field misuse)

### `src/agent.py`

- `run_node_agent(conn, node, client, model, max_iterations)` — run LLM agent for a prompt node
- `run_sql_node(conn, node)` — execute deterministic SQL node
- `run_validate_sql(conn, node)` — execute validate_sql with validation namespace
- `validate_node_complete(conn, node)` — full validation (outputs + validate_sql + validation views)
- `persist_node_meta(conn, node_name, meta)` — write to `_node_meta` table
- `execute_sql(conn, query, namespace, node_name, ...)` — SQL execution with namespace enforcement, timeout, size cap
- `is_sql_allowed(query, namespace, ddl_only)` — statement type + name checking

### `src/workspace.py`

- `Workspace` — dataclass: `db_path`, `nodes`, `exports`, `spec_module`
- `Workspace.run(client, model, max_iterations)` — full workspace execution
- `WorkspaceResult` — `success`, `node_results: dict[str, AgentResult]`, `elapsed_s`, `dag_layers`, `export_errors`
- `materialize_node_outputs(conn, node)` — discover and materialize `{name}_*` views
- `materialize_views(conn, view_names)` — core materialization (3-step swap)
- `persist_workspace_meta(conn, model, nodes, ...)` — write `_workspace_meta` table

### `src/namespace.py`

- `Namespace.for_node(node)` — `{name}_*` prefix, validation views forbidden
- `Namespace.for_validation(node)` — `{name}__validation*` prefix
- `Namespace.for_source_validation(source_name)` — `{input}__validation*` prefix (for source validation)

## CLI Commands

### `taskgraph init`
Initialize a Taskgraph spec in the current directory. Creates `pyproject.toml`, `specs/main.py`, `specs/__init__.py`, `SPEC_GUIDE.md`, `.env`, and `.gitignore`. Project name is derived from the directory name (slugified). Scaffold spec uses `sql` so `tg run` works immediately without an API key.
```
tg init          # create scaffold (skip existing files)
tg init --force  # overwrite existing files
```

### `taskgraph run`
Run a workspace spec: resolve DAG, execute all nodes, run exports. The `--spec` argument accepts both module paths (`specs.main`) and file paths (`specs/main.py`, `./specs/main.py`); file paths are auto-resolved to module paths.
```
just run --spec tests.single_task -o output.db
just run --spec specs/main.py -o output.db          # file path auto-resolved
just run --spec tests.diamond_dag -o output.db -m anthropic/claude-sonnet-4 --reasoning-effort medium
```

After each node completes, view changes are reported inline:
- `+ view_name  N cols, N rows` — created views
- `~ view_name` with compact SQL diff — modified views
- `- view_name` — dropped views

Changes are also persisted to the `_changes` table in the output `.db` for later querying.

### `taskgraph show`
Visualize a spec's structure: node DAG with type tags ([source], [sql], [prompt]), tree view, per-node details (dependencies, outputs, validation), graph errors, exports. Accepts file paths like `run`. When given a `.db` file, displays workspace metadata instead.
```
just show --spec my_app.specs.main
just show --spec specs/main.py
just show output.db
```
