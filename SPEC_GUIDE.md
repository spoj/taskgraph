# Workspace Spec Guide (For Spec Writers)

This guide targets people writing **workspace specs**: Python modules (usually shipped inside your own package) that call Taskgraph.

A spec defines:
- `NODES`: a list of nodes — each either an **input node** (has `source`) or a **task node** (has `sql` or `prompt`). Input nodes are ingested into DuckDB tables; task nodes form a DAG and produce SQL views.
- `EXPORTS` (optional): functions that export files from the final workspace

Specs are imported as modules (e.g. `my_app.specs.main`). File paths are accepted by the CLI and auto-resolved to module paths.

## Recommended Layout

For a new project, run `taskgraph init` to scaffold a `specs/` directory, `pyproject.toml`, and supporting files.

Taskgraph uses OpenRouter for LLM calls; set `OPENROUTER_API_KEY` in your environment or `.env` file. If your spec only uses `sql` tasks, no API key is required.

Two common patterns:

1) In an application package

```text
my_app/
  __init__.py
  specs/
    __init__.py
    main.py
    jan_2026.py
    feb_2026.py
pyproject.toml
```

Run with:

```bash
uv run taskgraph run --spec my_app.specs.main
```

2) In a repo-local `specs/` directory (what `taskgraph init` scaffolds)

```text
specs/
  main.py
pyproject.toml
```

Run with:

```bash
uv run taskgraph run
```

## Spec Discovery

If you don't pass `--spec`, Taskgraph uses:
- `[tool.taskgraph].spec` in `pyproject.toml`, otherwise
- `specs.main` (only if `specs/main.py` exists)

`pyproject.toml` example:

```toml
[tool.taskgraph]
spec = "my_app.specs.main"
```

## Minimal Example

```python
import polars as pl

def load_invoices() -> pl.DataFrame:
    return pl.read_csv("data/invoices.csv")

def load_payments() -> pl.DataFrame:
    return pl.read_csv("data/payments.csv")

NODES = [
    {"name": "invoices", "source": load_invoices},
    {"name": "payments", "source": load_payments},
    {
        "name": "match",
        "depends_on": ["invoices", "payments"],
        "prompt": (
            "Match invoices to payments. Create view 'match_results' with columns:\n"
            "- invoice_row_id: invoices._row_id\n"
            "- payment_row_id: payments._row_id\n"
            "- match_reason: brief explanation\n"
            "One invoice matches at most one payment; leave unmatched invoices out."
        ),
        "output_columns": {"match_results": ["invoice_row_id", "payment_row_id", "match_reason"]},
        "validate_sql": """
            CREATE OR REPLACE VIEW match__validation AS
            SELECT 'pass' AS status, 'ok' AS message
        """,
    },
]
```

Run it:

```bash
uv run taskgraph run --spec my_app.specs.main
```

---

## NODES

`NODES` is a list of node dicts. Each node has a `name` and is either an **input node** (has `source`) or a **task node** (has `sql` or `prompt`).

Node discrimination:
- Has `"source"` key → input node (ingested as a DuckDB table)
- Has `"sql"` or `"prompt"` → task node (produces SQL views)
- A node cannot have both `source` and `sql`/`prompt`.

---

## Input Nodes

An input node has `name` and `source`. Each becomes a DuckDB table before any tasks run.

### Simple input nodes

```python
NODES = [
    {"name": "sales", "source": load_sales},                           # callable -> DataFrame
    {"name": "rates", "source": [{"ccy": "USD", "rate": 1.0}, ...]},  # list[dict]
    {"name": "config", "source": {"key": ["a", "b"], "val": [1, 2]}}, # dict[str, list]
    {"name": "events", "source": "data/events.parquet"},               # file path (extension-based)
]
```

Accepted return types from callables:
- `pl.DataFrame` — used as-is
- `list[dict]` — array of structs, each dict is a row
- `dict[str, list]` — struct of arrays, each key is a column

### Input nodes with validation

Input nodes can include optional `columns` and `validate_sql` fields:

```python
NODES = [
    {
        "name": "invoices",
        "source": "data/invoices.xlsx#Sheet1",
        "columns": ["id", "amount", "date", "vendor"],
        "validate_sql": """
            CREATE OR REPLACE VIEW invoices__validation AS
            SELECT 'fail' AS status,
                   'null amount at id=' || CAST(id AS VARCHAR) AS message
            FROM invoices WHERE amount IS NULL
            UNION ALL
            SELECT 'pass' AS status, 'ok' AS message
            WHERE NOT EXISTS (SELECT 1 FROM invoices WHERE amount IS NULL)
        """,
    },
]
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `name` | `str` | Yes | Table name in DuckDB |
| `source` | callable, raw data, or file path | Yes | Data source |
| `columns` | `list[str]` | No | Required columns. Checked after ingestion, before tasks. Missing columns abort the run. |
| `validate_sql` | `str` | No | SQL that creates `{input_name}__validation*` views with `status` and `message` columns. Same contract as task `validate_sql`. On success, validation views are materialized as tables; on failure, validation views remain as views for debugging. |

Node type is determined by the presence of `source` vs `sql`/`prompt` keys.

### File paths

Strings ending in one of the supported extensions are treated as file inputs:

| Extension | Ingested via | Notes |
|-----------|--------------|-------|
| `.csv` | DuckDB `read_csv_auto` | Auto-detect schema |
| `.parquet` | DuckDB `read_parquet` | Native parquet reader |
| `.xlsx` / `.xls` | DuckDB `read_xlsx` | `header = false`; columns named `A1`, `B1`, etc. Use `#SheetName` to pick a sheet |
| `.pdf` | Gemini 3 Flash (OpenRouter) | Extracts tabular data into JSON (requires OPENROUTER_API_KEY) |

Excel sheet selection uses a fragment:

```python
"budget": "data/invoices.xlsx#Budget"
```

Relative file paths resolve from the spec file's directory.

### The `_row_id` column

Every ingested table gets a `_row_id INTEGER` column: a 1-based sequential row number. It serves as the primary key for row-level references in validation SQL and output views. The agent can query it but it is not shown in the schema display.

### Data loading guidelines

- **Allowed imports in spec modules**: `polars`, `openpyxl`, and Python stdlib (`pathlib`, `csv`, `json`, etc.). No other third-party libraries.
- Callables are invoked at ingest time, not import time. Exceptions are caught and reported with context.
- Empty tables (0 rows) produce a warning but do not abort the run.
- Polars handles type inference. If you need specific types, cast explicitly:

```python
def load_data():
    df = pl.read_csv("data.csv", infer_schema=False)  # all strings
    return df.with_columns(
        pl.col("amount").cast(pl.Float64, strict=False).fill_null(0.0)
    )
```

---

## Task Nodes

Task nodes are part of the same `NODES` list. Each task has exactly one transform mode:
- `sql`: deterministic SQL statements executed directly (views/macros only).
- `prompt`: LLM-driven transform. The agent writes SQL views/macros.

All views created by a task must be namespaced as `{name}_*` (underscore required — bare `{name}` is NOT a valid view name).

Validation is optional and deterministic via `validate_sql`, which runs after the transform to create `{task_name}__validation*` views. Validation views are not listed in `output_columns`.

Taskgraph treats `validate_sql` as a definition step: it runs once per node (after required outputs exist) to create `{name}__validation*` views, then re-evaluates those views on subsequent validation attempts.

If `validate_sql` must be retried due to an execution error, Taskgraph clears any existing `{name}__validation*` views first so plain `CREATE VIEW ...__validation AS ...` is safe (you do not need to remember `OR REPLACE`).

```python
NODES = [
    {"name": "invoices", "source": load_invoices},
    {"name": "payments", "source": load_payments},
    {
        "name": "match",
        "depends_on": ["invoices", "payments"],
        "prompt": "...",
        "output_columns": {
            "match_results": ["invoice_row_id", "payment_row_id", "match_reason"],
        },
        "validate_sql": """
            CREATE OR REPLACE VIEW match__validation AS
            SELECT 'pass' AS status, 'ok' AS message
        """,
    },
    {
        "name": "summary",
        "depends_on": ["match"],
        "sql": "CREATE OR REPLACE VIEW summary_report AS SELECT ...",
    },
]
```

### Task node fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier. Also the namespace prefix — all views must be named `{name}_*`. |
| `sql` | `str` | Exactly one of `sql` or `prompt` | Deterministic statements executed directly (views/macros only). Multiple statements are allowed in one string and will be split by DuckDB's parser. |
| `prompt` | `str` | Exactly one of `sql` or `prompt` | Objective text for LLM-driven transforms. |
| `validate_sql` | `str` | No | Deterministic SQL that creates `{name}__validation*` views. Runs after the transform. |
| `depends_on` | `list[str]` | No | Node names (input or task) that must complete before this task runs. Used for DAG scheduling. |
| `output_columns` | `dict[str, list[str]]` | No | Required views and their columns. Keys must start with `{name}_`. Checks names only, not types. Extra columns are fine. |

### Dependencies

- `depends_on` lists input node names and/or task node names.
- If a dependency is an input node, the input is ingested before the task runs.
- If a dependency is another task node, that task must complete first.
- Circular dependencies are detected and raise an error.

### Compensation over upstream rewrites

If a task discovers an issue with upstream data, it should **compensate in its own outputs** rather than attempting to rewrite upstream logic. Tasks cannot override another task's outputs due to namespace enforcement and duplicate-output checks. The intended pattern is:

- Add diagnostic or corrective intermediate views in the current task's namespace.
- Update the task's own output views to apply the correction.

This keeps provenance stable and avoids introducing extra variance across the DAG.

### What the agent sees

The agent receives:

1. A system prompt describing DuckDB SQL syntax, constraints, and workflow guidance.
2. A user message containing:
   - The task name
   - Your `prompt` text
   - Input tables with schemas (from `depends_on` references)
   - Required output views (with expected columns, if declared via `output_columns`)
   - `validate_sql` (if provided)
   - Naming rules: the agent can create views/macros named `{name}_*` (e.g., node `match` can create `match_step1`, `match_candidates`, etc.)
    
   If validation fails, the agent receives validation feedback and can retry within the iteration budget (default 200).

### What the agent can do

The agent has a single tool: `run_sql`. It can execute:

| Statement | Allowed | Namespace-restricted |
|-----------|---------|---------------------|
| `SELECT` | Yes | No — can read any table/view |
| `SUMMARIZE table_name` | Yes | No |
| `EXPLAIN query` | Yes | No |
| `CREATE [OR REPLACE] VIEW` | Yes | Yes — must be a declared output or `{name}_*` prefixed |
| `DROP VIEW` | Yes | Yes |
| `CREATE MACRO` | Yes | Yes |
| `DROP MACRO` | Yes | Yes |
| Everything else | **No** | — |

Deterministic `sql` tasks are more restrictive: they may only execute `CREATE/DROP VIEW` and `CREATE/DROP MACRO` statements (no standalone `SELECT`).
`prompt` tasks are the only ones that invoke the LLM.

The agent can call `run_sql` multiple times in parallel within a single turn.

### DuckDB features available to the agent

The agent is told about these DuckDB features in its system prompt:

- **Safe casting**: `TRY_CAST(expr AS type)` returns NULL instead of erroring. `TRY(expr)` wraps any expression.
- **Grouping**: `GROUP BY ALL` auto-groups by all non-aggregate columns.
- **Unions**: `UNION BY NAME` matches columns by name, not position.
- **Profiling**: `SUMMARIZE table_name` returns min, max, null count, unique count, avg for every column.
- **Fuzzy matching**: `jaro_winkler_similarity()`, `levenshtein()`, `jaccard()`, `damerau_levenshtein()`
- **Window filtering**: `QUALIFY row_number() OVER (...) = 1`
- **Selective columns**: `SELECT * EXCLUDE (col)`, `SELECT * REPLACE (expr AS col)`
- **Best-match selection**: `arg_min(val, ordering)`, `arg_max(val, ordering)`
- **Conditional aggregates**: `count(*) FILTER (WHERE cond)`
- **Nearest-match joins**: `ASOF JOIN`
- **List/array operations**: `list_agg()`, `unnest()`, `list_filter(lst, x -> cond)`, list comprehensions `[x * 2 FOR x IN list IF x > 0]`
- **Reusable logic**: `CREATE MACRO name(args) AS expr` (scalar) and `CREATE MACRO name(args) AS TABLE (SELECT ...)` (table-valued)

### Namespace and intermediate views

All views created by a task must be namespaced as `{name}_*` (underscore required). A task named `"match"` can create:
- `match_results`, `match_scored`, `match_candidates`, etc. — any view prefixed with `match_`
- `match_clean`, `match_normalize`, etc. — any macro prefixed with `match_`

If `output_columns` declares `{"match_results": [...]}`, then `match_results` is the required output view.

This allows the agent to build a chain of views:
```
match_legs → match_candidates → match_scored → match_pass1 → match_results
```

Views are late-binding in DuckDB — updating `match_legs` automatically propagates through the chain. The agent can iterate on upstream logic without recreating downstream views.

---

## Validation

Validation runs automatically after the transform. There are three checks, in order, short-circuiting on first failure:

### 1. Output view existence

Each view declared as a key in `output_columns` must exist. If not: `"Required output view 'X' was not created."`

### 2. Output column check

If `output_columns` is specified, each listed column must be present in the view. Extra columns are fine. Error includes actual columns for debugging:
```
View 'output' is missing required column(s): left_ids. Actual columns: category, count
```

### 3. Validation SQL

Task validation is expressed by `validate_sql`, which runs after the transform and creates one or more views named `{name}__validation` and/or `{name}__validation_*`.

Contract:
- The view must have columns `status` and `message`.
- `status` must be one of: `pass`, `warn`, `fail` (case-insensitive).
- If the view contains any row with `status='fail'`, the node fails.

Recommended pattern:

```sql
-- One row per issue (fail) with a human-readable message
CREATE OR REPLACE VIEW match__validation AS
SELECT 'fail' AS status, 'unmatched invoice _row_id=' || CAST(i._row_id AS VARCHAR) AS message
FROM invoices i
LEFT JOIN matches m ON m.invoice_row_id = i._row_id
WHERE m.invoice_row_id IS NULL

UNION ALL
SELECT 'pass' AS status, 'ok' AS message
WHERE NOT EXISTS (
    SELECT 1
    FROM invoices i
    LEFT JOIN matches m ON m.invoice_row_id = i._row_id
    WHERE m.invoice_row_id IS NULL
);
```

`warn` rows are informational and shown by the CLI; only `fail` rows block the node.
You can optionally add helper columns (not required by the harness), such as `evidence_view` to point reviewers to a drill-down view.

---

## EXPORTS (optional)

A dict mapping output file paths to export functions. Exports run only if all tasks pass validation.

```python
EXPORTS = {
    "report.xlsx": export_xlsx,
    "summary.csv": export_csv,
}
```

### Export function signature

```python
def export_fn(conn: duckdb.DuckDBPyConnection, path: Path) -> None:
```

- `conn` — open DuckDB connection to the workspace. All ingested tables and agent-created views are available. Run any SQL you want.
- `path` — output file path (constructed from the dict key).

### Getting data out of DuckDB

```python
def export_xlsx(conn, path):
    # Polars DataFrame
    df = conn.execute("SELECT * FROM output").pl()

    # Raw rows
    rows = conn.execute("SELECT * FROM output").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM output LIMIT 0").description]
```

### Excel export with openpyxl

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

def export_xlsx(conn, path):
    df = conn.execute("""
        SELECT category, detail_count, amount_diff
        FROM reconciliation ORDER BY ABS(amount_diff) DESC
    """).pl()

    wb = Workbook()
    ws = wb.active
    ws.title = "Reconciliation"

    # Headers
    for col_idx, col_name in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = Font(bold=True)

    # Data rows
    for row_idx, row in enumerate(df.iter_rows(), 2):
        for col_idx, value in enumerate(row, 1):
            ws.cell(row=row_idx, column=col_idx, value=value)

    wb.save(path)
```

### Error handling

Export function exceptions are caught — they don't crash the run. Errors are stored in the result and reported by the CLI. The `.db` is still saved even if exports fail.

---

## DAG: Multi-Node Specs

Nodes form a directed acyclic graph based on explicit `depends_on` edges. Nodes whose dependencies have all completed run concurrently (their LLM API calls overlap).

```python
NODES = [
    {"name": "raw_sales", "source": load_sales},
    {"name": "raw_costs", "source": load_costs},
    {
        "name": "clean_sales",
        "depends_on": ["raw_sales"],
        "sql": "CREATE OR REPLACE VIEW clean_sales_output AS SELECT ...",
    },
    {
        "name": "clean_costs",
        "depends_on": ["raw_costs"],
        "sql": "CREATE OR REPLACE VIEW clean_costs_output AS SELECT ...",
    },
    {
        "name": "reconcile",
        "depends_on": ["clean_sales", "clean_costs"],
        "sql": "CREATE OR REPLACE VIEW reconcile_output AS SELECT ...",
    },
]
```

This produces two layers:
- Layer 1: `clean_sales` and `clean_costs` (run concurrently — both only depend on input nodes)
- Layer 2: `reconcile` (depends on task nodes from layer 1)

### Dependency rules

- `depends_on` lists node names: input node names or task node names.
- Circular dependencies are detected and raise an error.

### Layer failure

A failed node only blocks its direct downstream dependents — nodes that consume its outputs. Unrelated nodes in later layers continue to run. Nodes in the same layer are not affected (they run concurrently and may succeed).

---

## Provenance and Lineage

The `.db` is a queryable audit trail. After a run, you can open it and trace how any output was produced:

### Inspect the database

After a run, node outputs and validation views are **materialized as tables** (frozen). Intermediate `{name}_*` views stay as views for debuggability. Every SQL statement (including input validation) is logged in `_trace`. The `_view_definitions` view (derived from `_trace`) provides lineage — the original CREATE VIEW SQL for each materialized output.

```bash
# Show all user-created tables (materialized outputs + inputs)
duckdb output.db "SELECT table_name FROM duckdb_tables() WHERE internal = false"

# Show intermediate views (still live)
duckdb output.db "SELECT view_name FROM duckdb_views() WHERE internal = false"

# Read a materialized output's original SQL definition
duckdb output.db "SELECT sql FROM _view_definitions WHERE view_name = 'output'"

# Query the output
duckdb output.db "SELECT * FROM output LIMIT 10"

# Check input data
duckdb output.db "SELECT * FROM invoices WHERE _row_id = 42"
```

### Lineage via `_view_definitions`

After materialization, output views become tables. Their original SQL is available via `_view_definitions` (a view derived from `_trace`). Views that were dropped are automatically excluded:

```sql
SELECT node, view_name, sql FROM _view_definitions ORDER BY node, view_name
```

Intermediate views (still live) reference materialized tables. The chain is the lineage:
```
intermediate_view → materialized_table → ... → input_table
```

### SQL execution trace

Every SQL query executed — by node agents, SQL-only nodes, node validation, and input validation — is logged in `_trace` with a `source` column indicating the origin (`agent`, `sql_node`, `node_validation`, `input_validation`):
```sql
SELECT node, source, query, success, error, row_count, elapsed_ms
FROM _trace
WHERE node = 'match'
ORDER BY id
```

### Node metadata

Per-node stats are in `_node_meta`:
```sql
SELECT node, meta_json FROM _node_meta ORDER BY node
```

`meta_json` includes fields like `model`, `reasoning_effort`, `iterations`, `tool_calls`, `elapsed_s`, `validation`, token counts, and a timestamp.

### Workspace metadata

Run-level metadata is in `_workspace_meta`:
```sql
SELECT key, value FROM _workspace_meta
```

Keys (v2):
- `meta_version`
- `created_at_utc`
- `taskgraph_version`
- `python_version`
- `platform`
- `node_prompts`
- `llm_model`, `llm_reasoning_effort`, `llm_max_iterations`
- `inputs_row_counts`, `inputs_schema`
- `run` (JSON: run context)
- `spec` (JSON: module)
- `exports` (JSON: export results, added after exports run)

### Adding provenance columns

To make outputs self-documenting, require provenance columns in `output_columns`:

```python
NODES = [
    {"name": "invoices", "source": load_invoices},
    {"name": "payments", "source": load_payments},
    {
        "name": "match",
        "depends_on": ["invoices", "payments"],
        "prompt": "... Include a match_reason column explaining why each pair was matched ...",
        "output_columns": {
            "match_output": ["left_ids", "right_ids", "match_reason", "match_score"],
        },
    },
]
```

The agent must produce these columns or validation fails. The prompt should explain what you expect in each column. This gives you per-row audit explanations in the output view itself.

---

## Writing Good Prompts

The `prompt` field is your objective text for LLM-driven transforms. Write it broad-to-specific: start with *why* the node exists, then *how* it works, then the *output contract*.

### Three layers

1. **Business context** (1-2 sentences): Why does this node exist? What business question does it answer? Who uses the output?
2. **Logic description** (2-4 sentences): The approach in plain English — matching strategy, edge cases, expected data shape.
3. **Output contract**: View names, required columns, constraints, tolerances.

### Write prompt from broad to specific

Include everything you would put in a code review checklist:
- output view names
- required columns (and semantics)
- uniqueness/completeness constraints
- tolerances for numeric comparisons
- how to break ties

Then enforce it with `output_columns` and `validate_sql` validation views.

### Weak vs strong prompt

```
WEAK:
"Match invoices to payments"

STRONG:
"Finance needs a monthly reconciliation of invoices against bank payments to
identify unpaid invoices and duplicate payments.

Match invoices to payments by amount (within 0.01 tolerance) and date (same day).
Use vendor name similarity (jaro_winkler > 0.8) as a tiebreaker when multiple
payments match the same amount and date. Some invoices will not match — these
should appear as unmatched rows.

OUTPUT VIEW: matches
  - invoice_row_id: INTEGER — invoices._row_id
  - payment_row_id: INTEGER — payments._row_id (NULL if unmatched)
  - match_reason: VARCHAR — why the pair was matched
  - match_score: DOUBLE — confidence (0-1)
Every invoice must appear exactly once."
```

### Explain data semantics

```
NOTE ON SIGNS: Detail amounts have OPPOSITE sign to summary amounts.
For a correct match: SUM(detail.Amount) + summary.Amount ≈ 0 for each category.
```

### Call out edge cases

```
- foreign_amount = 0 means there is no foreign leg for that row
- When domestic_currency = foreign_currency, treat as having only one leg
- Some rows will not match anything — these should appear as unmatched
```

### Request provenance when you need it

```
Include these columns in the output for audit purposes:
  - match_reason: VARCHAR explaining why the pair was matched
  - match_score: DOUBLE confidence score (0-1)
  - currency_used: VARCHAR the currency in which amounts were compared
  - amount_diff: DOUBLE the residual difference after matching
```

## Strong Validation and Fail-Fast Patterns

Stronger validation catches issues earlier and makes prompt-node retries more effective. Prefer failing fast with precise, actionable messages.

- Use `validate_sql` views that check completeness, totals, and uniqueness.
- Use `output_columns` to enforce schema contracts immediately.
- Keep validation errors concise and actionable.

### Warnings vs failures

- `fail` rows block the node.
- `warn` rows are informational and shown by the CLI.
- Prompt tasks can retry when validation fails, within the iteration budget (default 200).

Write `warn` thresholds at the quality level you want to monitor:

```sql
-- This validation view emits a warning when match rate drops below 90%
CREATE OR REPLACE VIEW match__validation AS
SELECT
  CASE WHEN pct < 0.90 THEN 'warn' ELSE 'pass' END AS status,
  format('Match rate {:.1f}%', pct * 100) AS message
FROM (
  SELECT count(*) FILTER (WHERE matched) * 1.0 / count(*) AS pct
  FROM matches
)
```

If you want the run to fail when quality drops, emit `fail` instead of `warn`.

---

## CLI Reference

### `taskgraph run`

```
taskgraph run --spec MODULE -o OUTPUT_DB [options]
taskgraph run -o OUTPUT_DB [options]
```

| Flag | Description |
|------|-------------|
| `-o, --output PATH` | Output `.db` file path (optional; default: `runs/<spec>_<timestamp>.db`) |
| `-s, --spec MODULE` | Spec module path (default: `[tool.taskgraph].spec` from `pyproject.toml`; if unset: `specs.main` when present) |
| `-m, --model MODEL` | LLM model (default: `openai/gpt-5.2`) |
| `--reasoning-effort low\|medium\|high` | Reasoning effort level |
| `--max-iterations N` | Max agent iterations per node (default: 200) |
| `-q, --quiet` | Suppress verbose output |
| `-f, --force` | Overwrite output file without prompting |

### `taskgraph show`

```
taskgraph show --spec MODULE
taskgraph show output.db
taskgraph show
```

When given a `.db` file, displays workspace metadata: creation time, model, inputs, tasks, and export results.
Otherwise, displays spec structure: inputs, DAG layers, per-node details, validation summary, exports.

## Appendix: Debugging a Workspace

When a spec isn't doing what you expect, treat `output.db` as the ground truth:

```bash
# What tables exist? (materialized outputs + inputs)
duckdb output.db "SELECT table_name FROM duckdb_tables() WHERE internal = false ORDER BY 1"

# What intermediate views exist?
duckdb output.db "SELECT view_name FROM duckdb_views() WHERE internal = false ORDER BY 1"

# What's the original SQL for a materialized output?
duckdb output.db "SELECT sql FROM _view_definitions WHERE view_name = 'matches'"

# What did the agent try?
duckdb output.db "SELECT id, node, source, success, row_count, elapsed_ms, query FROM _trace ORDER BY id"
```
