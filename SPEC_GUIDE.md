# Workspace Spec Guide (For Spec Writers)

This guide targets people writing **workspace specs**: Python modules (usually shipped inside your own package) that call Taskgraph.

A spec defines:
- `INPUTS`: how to ingest data into DuckDB tables
- `TASKS`: a DAG of tasks; each task runs via `sql` (deterministic SQL) or `prompt` (LLM transform) and produces one or more SQL views
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

INPUTS = {
    "invoices": load_invoices,
    "payments": load_payments,
}

TASKS = [
    {
        "name": "match",
        "prompt": (
            "Match invoices to payments. Create view 'matches' with columns:\n"
            "- invoice_row_id: invoices._row_id\n"
            "- payment_row_id: payments._row_id\n"
            "- match_reason: brief explanation\n"
            "One invoice matches at most one payment; leave unmatched invoices out."
        ),
        "inputs": ["invoices", "payments"],
        "outputs": ["matches"],
        "output_columns": {"matches": ["invoice_row_id", "payment_row_id", "match_reason"]},
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

## INPUTS

`INPUTS` is a dict mapping **table names** to data sources. Each key becomes a DuckDB table before any tasks run.

### Simple format

The value is a callable (called at ingest time), raw data, or a file path.

```python
INPUTS = {
    "sales": load_sales,                           # callable -> DataFrame
    "rates": [{"ccy": "USD", "rate": 1.0}, ...],  # list[dict]
    "config": {"key": ["a", "b"], "val": [1, 2]}, # dict[str, list]
    "events": "data/events.parquet",              # file path (extension-based)
}
```

Accepted return types from callables:
- `pl.DataFrame` — used as-is
- `list[dict]` — array of structs, each dict is a row
- `dict[str, list]` — struct of arrays, each key is a column

### Rich format

The value is a dict with a `"source"` key plus optional validation.

```python
INPUTS = {
    "invoices": {
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
}
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `source` | callable, raw data, or file path | Yes | Same as simple format, with file path support |
| `columns` | `list[str]` | No | Required columns. Checked after ingestion, before tasks. Missing columns abort the run. |
| `validate_sql` | `str` | No | SQL that creates `{input_name}__validation*` views with `status` and `message` columns. Same contract as task `validate_sql`. Passing views are materialized as tables; failing views are dropped. |

Detection: a dict value with a `"source"` key is treated as rich format. A dict without `"source"` is treated as raw `dict[str, list]` data.

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

## TASKS

A list of task definitions. Each task has exactly one transform mode:
- `sql`: deterministic SQL statements executed directly (views/macros only).
- `prompt`: LLM-driven transform. The agent writes SQL views/macros.

Validation is optional and deterministic via `validate_sql`, which runs after the transform to create `{task_name}__validation*` views. Validation views are not listed in `outputs`.

```python
TASKS = [
    {
        "name": "match",
        "prompt": "...",
        "inputs": ["invoices", "payments"],
        "outputs": ["matches"],
        "output_columns": {
            "matches": ["invoice_row_id", "payment_row_id", "match_reason"],
        },
        "validate_sql": """
            CREATE OR REPLACE VIEW match__validation AS
            SELECT 'pass' AS status, 'ok' AS message
        """,
    },
    {
        "name": "summary",
        "sql": "CREATE OR REPLACE VIEW match_summary AS SELECT ...",
        "inputs": ["matches"],
        "outputs": ["match_summary"],
    },
]
```

### Task fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier. Also the namespace prefix for intermediate views. |
| `sql` | `str` | Exactly one of `sql` or `prompt` | Deterministic statements executed directly (views/macros only). Multiple statements are allowed in one string and will be split by DuckDB's parser. |
| `prompt` | `str` | Exactly one of `sql` or `prompt` | Objective text for LLM-driven transforms. |
| `validate_sql` | `str` | No | Deterministic SQL that creates `{task}__validation*` views. Runs after the transform. |
| `inputs` | `list[str]` | Yes | Tables/views this task reads. Can be ingested tables or outputs of other tasks. |
| `outputs` | `list[str]` | Yes | Views the task must create. Validation checks these exist. |
| `output_columns` | `dict[str, list[str]]` | No | Required columns per output view. Checks names only, not types. Extra columns are fine. |

### Naming and dependencies

- A task may read any ingested table.
- A task may also read other tasks' outputs by listing those output view names in `inputs`.
- If an input name matches a prior task's output, Taskgraph wires the dependency automatically.
- Two tasks may not declare the same output view name.

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
   - Input tables with schemas
   - Required output view names (with expected columns, if declared)
   - `validate_sql` (if provided)
   - Naming rules: the agent can create views/macros named either as declared outputs or prefixed with `{task_name}_` (e.g., task `match` can create `match_step1`, `match_candidates`, etc.)
    
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

A task named `"match"` with `outputs: ["output"]` can create:
- `output` — the required output view
- `match_step1`, `match_candidates`, `match_scored`, etc. — any view prefixed with `match_`
- `match_clean`, `match_normalize`, etc. — any macro prefixed with `match_`

This allows the agent to build a chain of views:
```
match_legs → match_candidates → match_scored → match_pass1 → output
```

Views are late-binding in DuckDB — updating `match_legs` automatically propagates through the chain. The agent can iterate on upstream logic without recreating downstream views.

---

## Validation

Validation runs automatically after the transform. There are three checks, in order, short-circuiting on first failure:

### 1. Output view existence

Each view listed in `outputs` must exist. If not: `"Required output view 'X' was not created."`

### 2. Output column check

If `output_columns` is specified, each listed column must be present in the view. Extra columns are fine. Error includes actual columns for debugging:
```
View 'output' is missing required column(s): left_ids. Actual columns: category, count
```

### 3. Validation SQL

Task validation is expressed by `validate_sql`, which runs after the transform and creates one or more views named `{task_name}__validation` and/or `{task_name}__validation_*`.

Contract:
- The view must have columns `status` and `message`.
- `status` must be one of: `pass`, `warn`, `fail` (case-insensitive).
- If the view contains any row with `status='fail'`, the task fails.

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

`warn` rows are informational and shown by the CLI; only `fail` rows block the task.
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

## DAG: Multi-Task Specs

Tasks form a directed acyclic graph based on input/output overlap. Tasks in the same layer run concurrently (their LLM API calls overlap).

```python
INPUTS = {
    "raw_sales": load_sales,
    "raw_costs": load_costs,
}

TASKS = [
    {
        "name": "clean_sales",
        "sql": "CREATE OR REPLACE VIEW sales AS SELECT ...",
        "inputs": ["raw_sales"],
        "outputs": ["sales"],
    },
    {
        "name": "clean_costs",
        "sql": "CREATE OR REPLACE VIEW costs AS SELECT ...",
        "inputs": ["raw_costs"],
        "outputs": ["costs"],
    },
    {
        "name": "reconcile",
        "sql": "CREATE OR REPLACE VIEW output AS SELECT ...",
        "inputs": ["sales", "costs"],
        "outputs": ["output"],
    },
]
```

This produces two layers:
- Layer 1: `clean_sales` and `clean_costs` (run concurrently — both only depend on base tables)
- Layer 2: `reconcile` (depends on outputs from layer 1)

### Dependency rules

- If a task input name matches another task's output name, a dependency is created.
- Input names that don't match any task output are treated as ingested tables (no dependency).
- Circular dependencies are detected and raise an error.
- Two tasks cannot produce the same output name.

### Layer failure

A failed task only blocks its direct downstream dependents — tasks that consume its outputs. Unrelated tasks in later layers continue to run. Tasks in the same layer are not affected (they run concurrently and may succeed).

---

## Provenance and Lineage

The `.db` is a queryable audit trail. After a run, you can open it and trace how any output was produced:

### Inspect the database

After a run, task outputs and validation views are **materialized as tables** (frozen). Intermediate `{task}_*` views stay as views for debuggability. Every SQL statement (including input validation) is logged in `_trace`. The `_view_definitions` view (derived from `_trace`) provides lineage — the original CREATE VIEW SQL for each materialized output.

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
SELECT task, view_name, sql FROM _view_definitions ORDER BY task, view_name
```

Intermediate views (still live) reference materialized tables. The chain is the lineage:
```
intermediate_view → materialized_table → ... → input_table
```

### SQL execution trace

Every SQL query executed — by task agents, SQL-only tasks, task validation, and input validation — is logged in `_trace` with a `source` column indicating the origin (`agent`, `sql_task`, `task_validation`, `input_validation`):
```sql
SELECT task, source, query, success, error, row_count, elapsed_ms
FROM _trace
WHERE task = 'match'
ORDER BY id
```

### Task metadata

Per-task stats are in `_task_meta`:
```sql
SELECT task, meta_json FROM _task_meta ORDER BY task
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
- `task_prompts`
- `llm_model`, `llm_reasoning_effort`, `llm_max_iterations`
- `inputs_row_counts`, `inputs_schema`
- `run` (JSON: run context)
- `spec` (JSON: module)
- `exports` (JSON: export results, added after exports run)

### Adding provenance columns

To make outputs self-documenting, require provenance columns in `output_columns`:

```python
TASKS = [
    {
        "name": "match",
        "prompt": "... Include a match_reason column explaining why each pair was matched ...",
        "inputs": ["invoices", "payments"],
        "outputs": ["output"],
        "output_columns": {
            "output": ["left_ids", "right_ids", "match_reason", "match_score"],
        },
    },
]
```

The agent must produce these columns or validation fails. The prompt should explain what you expect in each column. This gives you per-row audit explanations in the output view itself.

---

## Writing Good Prompts

The `prompt` field is your objective text for LLM-driven transforms. Write it broad-to-specific: start with *why* the task exists, then *how* it works, then the *output contract*.

### Three layers

1. **Business context** (1-2 sentences): Why does this task exist? What business question does it answer? Who uses the output?
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

Stronger validation catches issues earlier and makes prompt-task retries more effective. Prefer failing fast with precise, actionable messages.

- Use `validate_sql` views that check completeness, totals, and uniqueness.
- Use `output_columns` to enforce schema contracts immediately.
- Keep validation errors concise and actionable.

### Warnings vs failures

- `fail` rows block the task.
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
| `--max-iterations N` | Max agent iterations per task (default: 200) |
| `-q, --quiet` | Suppress verbose output |
| `-f, --force` | Overwrite output file without prompting |

### `taskgraph show`

```
taskgraph show --spec MODULE
taskgraph show output.db
taskgraph show
```

When given a `.db` file, displays workspace metadata: creation time, model, inputs, tasks, and export results.
Otherwise, displays spec structure: inputs, DAG layers, per-task details, validation summary, exports.

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
duckdb output.db "SELECT id, task, source, success, row_count, elapsed_ms, query FROM _trace ORDER BY id"
```
