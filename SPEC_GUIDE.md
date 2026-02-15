# Workspace Spec Writer's Guide

A workspace spec is a Python module that defines what data to load, what to ask the LLM agent to do, how to validate the results, and optionally how to export them.

## Minimal Example

```python
import polars as pl

def load_invoices():
    return pl.read_csv("data/invoices.csv")

def load_payments():
    return pl.read_csv("data/payments.csv")

INPUTS = {
    "invoices": load_invoices,
    "payments": load_payments,
}

TASKS = [
    {
        "name": "match",
        "prompt": "Match invoices to payments by amount and date...",
        "inputs": ["invoices", "payments"],
        "outputs": ["output"],
    },
]
```

Run it:
```bash
taskgraph run --spec my_app.specs.main -o output.db
```

---

## INPUTS

A dict mapping table names to data sources. Each table is ingested into DuckDB before any tasks run.

### Simple format

The value is a callable (called at ingest time) or raw data.

```python
INPUTS = {
    "sales": load_sales,                           # callable -> DataFrame
    "rates": [{"ccy": "USD", "rate": 1.0}, ...],  # list[dict]
    "config": {"key": ["a", "b"], "val": [1, 2]}, # dict[str, list]
}
```

Accepted return types from callables:
- `pl.DataFrame` — used as-is
- `list[dict]` — array of structs, each dict is a row
- `dict[str, list]` — struct of arrays, each key is a column

### Rich format

The value is a dict with a `"data"` key plus optional validation.

```python
INPUTS = {
    "invoices": {
        "data": load_invoices,
        "columns": ["id", "amount", "date", "vendor"],
        "validate_sql": [
            "SELECT 'null amount at id=' || id FROM invoices WHERE amount IS NULL",
        ],
    },
}
```

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `data` | callable or raw data | Yes | Same as simple format |
| `columns` | `list[str]` | No | Required columns. Checked after ingestion, before tasks. Missing columns abort the run. |
| `validate_sql` | `list[str]` | No | SQL queries that must return 0 rows. Each returned row is an error. Runs after column check. |

Detection: a dict value with a `"data"` key is treated as rich format. A dict without `"data"` is treated as raw `dict[str, list]` data.

### The `_row_id` column

Every ingested table gets a `_row_id INTEGER` column: a 1-based sequential row number. It serves as the primary key for row-level references in validation SQL and output views. The agent can query it but it is not shown in the schema display.

### Data loading guidelines

- **Allowed imports**: `polars`, `openpyxl`, and Python stdlib (`pathlib`, `csv`, `json`, etc.). No other third-party libraries.
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

A list of task definitions. Each task becomes an independent LLM agent that writes SQL views against the database.

```python
TASKS = [
    {
        "name": "match",
        "prompt": "...",
        "inputs": ["invoices", "payments"],
        "outputs": ["output"],
        "output_columns": {"output": ["left_ids", "right_ids"]},
        "validate_sql": ["..."],
    },
]
```

### Task fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier. Also the namespace prefix for intermediate views. |
| `prompt` | `str` | Yes | Instructions for the agent. |
| `inputs` | `list[str]` | Yes | Tables and views this task can read. Can be ingested tables or outputs of other tasks. |
| `outputs` | `list[str]` | Yes | Views the agent must create. Validation checks these exist. |
| `output_columns` | `dict[str, list[str]]` | No | Required columns per output view. Checks names only, not types. Extra columns are fine. |
| `validate_sql` | `list[str]` | No | SQL queries that must return 0 rows after the agent finishes. See [Validation SQL](#validation-sql). |

### What the agent sees

The agent receives:

1. A system prompt describing DuckDB SQL syntax, constraints, and workflow guidance.
2. A user message containing:
   - The task name and your prompt text
   - Schema info for each input table: column names with types, row count, 3 sample rows (`_row_id` excluded from display)
   - Required output view names
   - Naming rules: the agent can create views/macros named either as declared outputs or prefixed with `{task_name}_` (e.g., task `match` can create `match_step1`, `match_candidates`, etc.)

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
- **List/array operations**: `list_value()`, `unnest()`, `list_sort()`, `list_filter(lst, x -> cond)`, list comprehensions `[x * 2 FOR x IN list IF x > 0]`
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

Validation runs automatically when the agent stops producing tool calls. There are three checks, in order, short-circuiting on first failure:

### 1. Output view existence

Each view listed in `outputs` must exist. If not: `"Required output view 'X' was not created."`

### 2. Output column check

If `output_columns` is specified, each listed column must be present in the view. Extra columns are fine. Error includes actual columns for debugging:
```
View 'output' is missing required column(s): left_ids. Actual columns: category, count
```

### 3. Validation SQL

Each query in `validate_sql` must return 0 rows. Queries run sequentially and short-circuit — the first query that returns rows stops validation.

**Returned rows become the error message.** Design your SELECT so each row is a human-readable diagnostic. The agent sees these errors and attempts to fix them.

Single-column result — each row's value is the error text:
```sql
SELECT 'invoice _row_id=' || CAST(i._row_id AS VARCHAR) || ' not matched'
FROM invoices i
WHERE i._row_id NOT IN (SELECT unnest(left_ids) FROM output WHERE left_ids IS NOT NULL)
```

Multi-column result — formatted as `col1=val1, col2=val2`:
```sql
SELECT category, ROUND(diff, 2) AS amount_diff
FROM reconciliation
WHERE ABS(diff) > 0.01
```

### Writing effective validation SQL

**Order queries by importance.** Validation short-circuits, so put the most critical check first. If the output view is structurally wrong, checking amount reconciliation is pointless.

**Make errors actionable.** Include enough context for the agent to fix the problem:
```sql
-- Bad: agent doesn't know which rows or why
SELECT COUNT(*) FROM output WHERE left_ids IS NULL

-- Good: agent can see which specific rows are unmatched
SELECT 'unmatched left _row_id=' || CAST(_row_id AS VARCHAR)
       || ' amount=' || CAST(amount AS VARCHAR)
FROM invoices
WHERE _row_id NOT IN (SELECT unnest(left_ids) FROM output WHERE left_ids IS NOT NULL)
```

**Use tolerances for numeric comparisons.** Floating-point arithmetic means exact equality rarely works:
```sql
SELECT category || ': diff=' || CAST(ROUND(detail_sum + summary_amt, 3) AS VARCHAR)
FROM reconciliation
WHERE ABS(detail_sum + summary_amt) > 0.01  -- tolerance
```

**Check completeness.** Common patterns:
```sql
-- Every input row appears exactly once
WITH used AS (
    SELECT unnest(left_ids) AS id FROM output WHERE left_ids IS NOT NULL
)
SELECT 'missing _row_id=' || CAST(d._row_id AS VARCHAR)
FROM detail d WHERE d._row_id NOT IN (SELECT id FROM used)

-- No duplicates
WITH used AS (
    SELECT unnest(left_ids) AS id FROM output WHERE left_ids IS NOT NULL
)
SELECT '_row_id=' || CAST(id AS VARCHAR) || ' appears ' || CAST(COUNT(*) AS VARCHAR) || ' times'
FROM used GROUP BY id HAVING COUNT(*) > 1
```

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
        "prompt": "Normalize sales data...",
        "inputs": ["raw_sales"],
        "outputs": ["sales"],
    },
    {
        "name": "clean_costs",
        "prompt": "Normalize cost data...",
        "inputs": ["raw_costs"],
        "outputs": ["costs"],
    },
    {
        "name": "reconcile",
        "prompt": "Match sales to costs...",
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

If any task in a layer fails, all downstream layers are skipped. Earlier tasks in the same layer are not affected (they run concurrently and may succeed).

---

## Reruns

The `.db` file stores the spec module reference, resolved prompts, data, views, and metadata. Reruns use the recorded spec module and git commit; Taskgraph will warn (but not fail) if your repo is dirty.

### Spec-free rerun (same data)

```bash
taskgraph rerun previous.db -o new.db
```

Uses the recorded spec module and existing data. Validates existing views — if they still pass, the agent is not invoked. Useful for verifying a previous run or forcing re-validation.

### Spec override rerun (new data)

```bash
taskgraph rerun jan.db -o feb.db --spec my_app.specs.feb
```

Uses a new spec module (must be structurally compatible — same input names/columns, task names/inputs/outputs). Re-ingests data from the new spec's loaders. Existing views are tested against the new data — the agent is only invoked for tasks that fail validation.

This is the month-to-month workflow:
1. `taskgraph run --spec my_app.specs.jan -o jan.db`
2. Edit your spec module: change data paths, tweak prompts
3. Commit changes in git
4. `taskgraph rerun jan.db -o feb.db --spec my_app.specs.feb`

### Rerun modes

- `--mode validate` (default): Skip tasks whose validation passes. Agent only runs for failures.
- `--mode review`: Always invoke the agent, even for passing tasks. Agent sees existing views and can revise them.

### Structural compatibility

The fingerprint includes: input names + declared columns, task names + inputs + outputs + output_columns. These must match between the source `.db` and the new spec. Things that **can** change freely: prompt text, `validate_sql`, export functions, data values.

### Force re-ingestion

```bash
taskgraph rerun previous.db -o new.db --reingest
```

Re-runs the spec's input callables even without `--spec`. Useful if the underlying data files changed but the spec hasn't.

---

## Provenance and Lineage

The `.db` is a queryable audit trail. After a run, you can open it and trace how any output was produced:

### Inspect the database

```bash
# Show all user-created views
duckdb output.db "SELECT view_name FROM duckdb_views() WHERE internal = false"

# Read a view's SQL definition
duckdb output.db "SELECT sql FROM duckdb_views() WHERE view_name = 'output'"

# Query the output
duckdb output.db "SELECT * FROM output LIMIT 10"

# Check input data
duckdb output.db "SELECT * FROM invoices WHERE _row_id = 42"
```

### View chain as lineage

Views reference other views. The chain is the lineage:
```
output → match_scored → match_candidates → match_legs → input_left, input_right
```

Because views are late-binding, you can modify input data and re-query to see the effect:
```sql
-- What happens if we change this invoice amount?
-- (read-only .db won't allow this, but in a rerun the agent would see updated data)
```

### SQL execution trace

Every SQL query the agent executed is logged in `_trace`:
```sql
SELECT task, query, success, error, row_count, elapsed_ms
FROM _trace
WHERE task = 'match'
ORDER BY id
```

### Task metadata

Per-task stats are in `_task_meta`:
```sql
SELECT task, key, value FROM _task_meta ORDER BY task, key
```

Keys: `model`, `iterations`, `tool_calls`, `elapsed_s`, `validation`, `prompt_tokens`, `completion_tokens`, `timestamp`.

### Workspace metadata

Run-level metadata is in `_workspace_meta`:
```sql
SELECT key, value FROM _workspace_meta
```

Keys: `fingerprint`, `prompts`, `model`, `timestamp`, `input_row_counts`, `spec_module`, `spec_git_commit`, `spec_git_root`, `spec_git_dirty`, `spec_source`.

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

## Prompt Writing Tips

The prompt is the most important part of the spec. The agent has access to the data schema and DuckDB features — your prompt should focus on the **domain logic**.

### Be specific about the matching criteria

```
BAD:  "Match invoices to payments"
GOOD: "Match invoices to payments by amount (within 0.01 tolerance) and date
       (same day). Use vendor name similarity (jaro_winkler > 0.8) as a
       tiebreaker when multiple payments match the same amount and date."
```

### Explain the data semantics

```
NOTE ON SIGNS: Detail amounts have OPPOSITE sign to summary amounts.
For a correct match: SUM(detail.Amount) + summary.Amount ≈ 0 for each category.
```

### Specify the output contract precisely

```
OUTPUT VIEW: output
Two columns:
  - left_ids:  INTEGER[] — list of _row_id values from invoices
  - right_ids: INTEGER[] — list of _row_id values from payments
Each row is one matched group. Unmatched invoices: right_ids is NULL.
Every row in both inputs must appear exactly once.
```

### Tell the agent about edge cases

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

---

## CLI Reference

### `taskgraph run`

```
taskgraph run --spec MODULE -o OUTPUT_DB [options]
taskgraph run -o OUTPUT_DB [options]
```

| Flag | Description |
|------|-------------|
| `-o, --output PATH` | Output `.db` file path (required) |
| `-s, --spec MODULE` | Spec module path (default: `[tool.taskgraph].spec` from `pyproject.toml`; if unset: `specs.main` when present) |
| `-m, --model MODEL` | LLM model (default: `openai/gpt-5.2`) |
| `--reasoning-effort low\|medium\|high` | Reasoning effort level |
| `--max-iterations N` | Max agent iterations per task (default: 200) |
| `-q, --quiet` | Suppress verbose output |
| `-f, --force` | Overwrite output file without prompting |

### `taskgraph rerun`

```
taskgraph rerun DB_FILE -o OUTPUT_DB [options]
```

| Flag | Description |
|------|-------------|
| `-o, --output PATH` | Output `.db` file path (required) |
| `-s, --spec MODULE` | Override spec module (must be structurally compatible) |
| `--mode validate\|review` | `validate`: skip passing tasks; `review`: always invoke agent |
| `--reingest` | Force re-ingestion of input data |
| `-m, --model MODEL` | LLM model |
| `--reasoning-effort low\|medium\|high` | Reasoning effort level |
| `--max-iterations N` | Max agent iterations per task |
| `-q, --quiet` | Suppress verbose output |
| `-f, --force` | Overwrite output file without prompting |

### `taskgraph show`

```
taskgraph show --spec MODULE
taskgraph show
```

Displays spec structure: inputs, DAG layers, per-task details, validation summary, exports.

### `taskgraph extract-spec`

```
taskgraph extract-spec DB_FILE OUT_FILE
```

Extracts embedded spec source from a `.db` file.
