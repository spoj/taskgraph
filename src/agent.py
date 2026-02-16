"""Task agent — executes a single task within a DuckDB workspace.

Each task runs as an independent agent that can only CREATE/DROP views
within its declared outputs or namespace prefix. SELECTs can read anything.
"""

import json
import logging
import re
import duckdb
import threading
import time
import datetime
import decimal
import uuid
from typing import Any, cast
from .agent_loop import run_agent_loop, AgentResult
from .api import OpenRouterClient, create_model_callable
from .task import Task, validation_outputs

log = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Handle DuckDB-native types that json.dumps can't serialize."""
    if isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return str(obj)
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Default per-query timeout in seconds. Prevents runaway queries
# (e.g., accidental cartesian joins) from blocking the event loop.
DEFAULT_QUERY_TIMEOUT_S = 30
# Maximum characters in a serialized tool result. ~20k tokens ≈ 30k chars.
MAX_RESULT_CHARS = 30_000


# --- System prompt ---

SYSTEM_PROMPT = """You are a SQL repair agent. A task's SQL failed and you must fix it.
You work in a DuckDB database. You can read anything, but can only write views and macros
within the task's namespace (specified per task).

CONSTRAINTS:
- Allowed statements: SELECT, SUMMARIZE, CREATE/DROP VIEW, CREATE/DROP MACRO
- No tables, no inserts, no updates, no deletes
- Use CREATE OR REPLACE VIEW to fix outputs
- CREATE MACRO name(args) AS expr — reusable scalar expression
- CREATE MACRO name(args) AS TABLE (SELECT ...) — reusable table function
- Batch independent run_sql calls in parallel to minimize rounds

VALIDATION:
- Runs automatically when you stop (no tool calls in your response).
- If validation fails, you receive the error and can keep fixing.
- Some tasks require validation views (named '{task}__validation' or '{task}__validation_*').
  These must have columns: status VARCHAR ('pass'|'warn'|'fail'), message VARCHAR.
  Any 'fail' row fails the task. No fail rows = pass.

CATALOG INTROSPECTION:
  SELECT view_name, sql FROM duckdb_views() WHERE internal = false
  SELECT table_name FROM duckdb_tables()
  SELECT function_name, macro_definition FROM duckdb_functions() WHERE function_type = 'macro'

DUCKDB DIALECT REFERENCE:
- Identifiers: double-quote ("col"), not backticks. Quote reserved words
  (left, right, match, group, order, etc.). Always use AS for column aliases.
- Types: VARCHAR (not TEXT), DOUBLE (not REAL), BOOLEAN (not INTEGER 0/1)
- TRY_CAST(expr AS type) returns NULL on failure. TRY(expr) wraps any expression.
- QUALIFY — filter window results: SELECT * FROM t QUALIFY row_number() OVER (...) = 1
- GROUP BY ALL — auto-groups by all non-aggregate columns
- UNION BY NAME — union by column name, not position
- SUMMARIZE table_name — instant profiling: min, max, nulls, uniques, avg per column
- Lists: [1,2,3], list_agg(), unnest(), list_filter(lst, x -> cond),
  list_transform(lst, x -> expr), [x * 2 FOR x IN list IF x > 0]
- Fuzzy matching: jaro_winkler_similarity(), levenshtein(), jaccard()
- SELECT * EXCLUDE (col), SELECT * REPLACE (expr AS col)
- arg_min(val, ordering), arg_max(val, ordering)
- count(*) FILTER (WHERE cond), greatest(), least()
- ASOF JOIN — nearest-match on ordered column
- Views are late-binding (store SQL text, not data). CREATE OR REPLACE propagates instantly.
"""


# --- Tool definitions ---


def _create_sql_tool() -> dict[str, Any]:
    """Define the SQL tool for the agent."""
    return {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a SQL query against the DuckDB database. Allowed: SELECT, SUMMARIZE, CREATE/DROP VIEW, CREATE/DROP MACRO. No tables, inserts, updates, or deletes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute (SELECT or CREATE/DROP VIEW only)",
                    }
                },
                "required": ["query"],
            },
        },
    }


# --- SQL validation ---

# Regex to extract the object name from CREATE/DROP statements.
# Handles: CREATE [OR REPLACE] [TEMP] VIEW|MACRO "name" ...
#          DROP VIEW|MACRO [IF EXISTS] "name" ...
_CREATE_NAME_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?"
    r"(?:VIEW|MACRO)\s+"
    r'"?(\w+)"?',
    re.IGNORECASE,
)
_DROP_NAME_RE = re.compile(
    r"DROP\s+(?:MACRO\s+TABLE|VIEW|MACRO)\s+"
    r"(?:IF\s+EXISTS\s+)?"
    r'"?(\w+)"?',
    re.IGNORECASE,
)

# Allowed sub-kinds within CREATE/DROP (matched against the SQL text)
_CREATE_KIND_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?"
    r"(VIEW|MACRO)\b",
    re.IGNORECASE,
)
_DROP_KIND_RE = re.compile(
    r"DROP\s+(MACRO\s+TABLE|VIEW|MACRO)\b",
    re.IGNORECASE,
)


def _extract_name(query: str, pattern: re.Pattern[str]) -> str | None:
    """Extract the object name from a SQL statement using regex."""
    m = pattern.search(query)
    return m.group(1) if m else None


def _check_name_allowed(
    kind_label: str,
    action: str,
    name: str | None,
    allowed_views: set[str] | None,
    namespace: str | None,
) -> tuple[bool, str]:
    """Check if a view/macro name is allowed by namespace rules."""
    if name and not _is_view_name_allowed(name, allowed_views, namespace):
        allowed_str = _format_allowed_names(allowed_views, namespace)
        return (
            False,
            f"Cannot {action} {kind_label} '{name}'. Allowed names: {allowed_str}",
        )
    return True, ""


# Connection used solely for extract_statements (lightweight, never writes)
_parser_conn: duckdb.DuckDBPyConnection | None = None


def _get_parser_conn() -> duckdb.DuckDBPyConnection:
    """Lazy singleton in-memory connection for SQL parsing."""
    global _parser_conn
    if _parser_conn is None:
        _parser_conn = duckdb.connect(":memory:")
    return _parser_conn


def is_sql_allowed(
    query: str,
    allowed_views: set[str] | None = None,
    namespace: str | None = None,
) -> tuple[bool, str]:
    """Check if a SQL query is allowed.

    Uses DuckDB's own parser (extract_statements) for statement type
    classification, then regex for name extraction on CREATE/DROP.

    Args:
        query: SQL query string.
        allowed_views: If set, only these view/macro names can be created/dropped
            (in addition to namespace-prefixed names).
        namespace: If set, names starting with "{namespace}_" are also allowed.
    """
    try:
        statements = _get_parser_conn().extract_statements(query)
    except duckdb.ParserException as e:
        return False, f"SQL parse error: {e}"
    except Exception as e:
        return False, f"SQL parse error: {e}"

    if not statements:
        return False, "Empty query"

    if len(statements) > 1:
        return False, "Only one statement allowed at a time"

    stmt = statements[0]
    st = stmt.type

    # DuckDB's Python typing for StatementType is incomplete in some versions;
    # cast to Any so pyright can type-check the rest of this module.
    StatementType = cast(Any, duckdb.StatementType)

    # SELECT — always allowed (covers UNION, INTERSECT, EXCEPT, SUMMARIZE, DESCRIBE)
    if st == StatementType.SELECT:
        return True, ""

    # EXPLAIN — allowed (read-only introspection)
    if st == StatementType.EXPLAIN:
        return True, ""

    # CREATE — only VIEW and MACRO
    if st == StatementType.CREATE:
        sql = stmt.query
        kind_match = _CREATE_KIND_RE.search(sql)
        if kind_match:
            kind = kind_match.group(1).upper()
            label = "view" if kind == "VIEW" else "macro"
            if allowed_views is not None or namespace is not None:
                name = _extract_name(sql, _CREATE_NAME_RE)
                ok, err = _check_name_allowed(
                    label, "create", name, allowed_views, namespace
                )
                if not ok:
                    return False, err
            return True, ""
        return (
            False,
            "Only CREATE VIEW and CREATE MACRO are permitted.",
        )

    # DROP — only VIEW and MACRO
    if st == StatementType.DROP:
        sql = stmt.query
        kind_match = _DROP_KIND_RE.search(sql)
        if kind_match:
            kind = kind_match.group(1).upper()
            label = "view" if kind == "VIEW" else "macro"
            if allowed_views is not None or namespace is not None:
                name = _extract_name(sql, _DROP_NAME_RE)
                ok, err = _check_name_allowed(
                    label, "drop", name, allowed_views, namespace
                )
                if not ok:
                    return False, err
            return True, ""
        return (
            False,
            "Only DROP VIEW and DROP MACRO are permitted.",
        )

    return (
        False,
        f"{st.name} is not allowed. Only SELECT, SUMMARIZE, CREATE/DROP VIEW, and CREATE/DROP MACRO are permitted.",
    )


def _is_view_name_allowed(
    view_name: str,
    allowed_views: set[str] | None,
    namespace: str | None,
) -> bool:
    """Check if a view name is allowed given constraints."""
    if allowed_views and view_name in allowed_views:
        return True
    if namespace and view_name.startswith(f"{namespace}_"):
        return True
    return False


def _format_allowed_names(allowed_views: set[str] | None, namespace: str | None) -> str:
    """Format allowed view names for error messages."""
    parts = []
    if allowed_views:
        parts.append(", ".join(sorted(allowed_views)))
    if namespace:
        parts.append(f"{namespace}_* (prefix)")
    return " | ".join(parts) if parts else "(none)"


# --- Metadata persistence ---


def persist_task_meta(
    conn: duckdb.DuckDBPyConnection, task_name: str, meta: dict[str, Any]
) -> None:
    """Persist per-task run metadata in _task_meta.

    _task_meta is a simple per-task JSON blob. It is overwritten per run.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _task_meta (
            task VARCHAR PRIMARY KEY,
            meta_json VARCHAR NOT NULL
        )
    """)

    conn.execute("DELETE FROM _task_meta WHERE task = ?", [task_name])
    conn.execute(
        "INSERT INTO _task_meta (task, meta_json) VALUES (?, ?)",
        [task_name, json.dumps(meta, sort_keys=True)],
    )


# --- Trace ---


def init_trace_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the _trace table if it doesn't exist."""
    conn.execute("CREATE SEQUENCE IF NOT EXISTS _trace_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _trace (
            id INTEGER DEFAULT nextval('_trace_seq'),
            timestamp TIMESTAMP DEFAULT current_timestamp,
            task VARCHAR,
            query VARCHAR NOT NULL,
            success BOOLEAN NOT NULL,
            error VARCHAR,
            row_count INTEGER,
            elapsed_ms DOUBLE
        )
    """)


def log_trace(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    success: bool,
    error: str | None = None,
    row_count: int | None = None,
    elapsed_ms: float | None = None,
    task_name: str | None = None,
) -> None:
    """Log a SQL query execution to the _trace table."""
    conn.execute(
        """
        INSERT INTO _trace (task, query, success, error, row_count, elapsed_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [task_name, query, success, error, row_count, elapsed_ms],
    )


# --- SQL execution ---


def execute_sql(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    allowed_views: set[str] | None = None,
    namespace: str | None = None,
    task_name: str | None = None,
    query_timeout_s: int = DEFAULT_QUERY_TIMEOUT_S,
    max_result_chars: int = MAX_RESULT_CHARS,
) -> dict[str, Any]:
    """Execute SQL query and return results.

    Results exceeding *max_result_chars* when serialized are rejected
    with an error asking the caller to add LIMIT or narrow the query.

    Args:
        query_timeout_s: Maximum seconds a query can run before being
            interrupted. Uses conn.interrupt() via a background timer thread.
            Set to 0 to disable.
        max_result_chars: Maximum characters in the JSON-serialized result.
            Set to 0 to disable.
    """
    start_time = time.perf_counter()

    allowed, error_msg = is_sql_allowed(query, allowed_views, namespace)
    if not allowed:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        log_trace(
            conn,
            query,
            success=False,
            error=error_msg,
            elapsed_ms=elapsed_ms,
            task_name=task_name,
        )
        return {"success": False, "error": error_msg}

    # Set up interrupt timer for long-running queries
    timer: threading.Timer | None = None
    timed_out = False

    if query_timeout_s > 0:

        def _interrupt():
            nonlocal timed_out
            timed_out = True
            conn.interrupt()

        timer = threading.Timer(query_timeout_s, _interrupt)
        timer.start()

    try:
        cursor = conn.execute(query)

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            log_trace(
                conn,
                query,
                success=True,
                row_count=len(rows),
                elapsed_ms=elapsed_ms,
                task_name=task_name,
            )
            result = {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            }
            # Check serialized size — reject oversized results so the
            # model learns to use LIMIT or narrow its query.
            if max_result_chars > 0 and len(rows) > 0:
                serialized = json.dumps(result, default=_json_default)
                if len(serialized) > max_result_chars:
                    return {
                        "success": False,
                        "error": (
                            f"Result too large ({len(rows)} rows, "
                            f"{len(serialized):,} chars). "
                            "Add LIMIT or narrow your query."
                        ),
                    }
            return result
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            log_trace(
                conn,
                query,
                success=True,
                row_count=0,
                elapsed_ms=elapsed_ms,
                task_name=task_name,
            )
            return {"success": True, "message": "OK"}
    except duckdb.InterruptException:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        error_msg = f"Query timed out after {query_timeout_s}s"
        log.warning("[%s] %s", task_name or "?", error_msg)
        log_trace(
            conn,
            query,
            success=False,
            error=error_msg,
            elapsed_ms=elapsed_ms,
            task_name=task_name,
        )
        return {"success": False, "error": error_msg}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        error_str = str(e)
        # If we timed out but got a different exception type, label it clearly
        if timed_out:
            error_str = f"Query timed out after {query_timeout_s}s ({error_str})"
        log_trace(
            conn,
            query,
            success=False,
            error=error_str,
            elapsed_ms=elapsed_ms,
            task_name=task_name,
        )
        return {"success": False, "error": error_str}
    finally:
        if timer is not None:
            timer.cancel()


def build_sql_repair_prompt(task: Task, issue: str) -> str:
    """Build a prompt to repair a failed SQL task using the LLM."""
    statements = task.sql_statements()
    sql_block_lines = []
    for i, stmt in enumerate(statements, start=1):
        sql_block_lines.append(f"-- statement {i}")
        sql_block_lines.append(stmt)
        sql_block_lines.append("")
    sql_block = "\n".join(sql_block_lines).rstrip()

    issue_text = _summarize_validation_issue(issue)

    has_column_rules = bool(task.output_columns)
    required_outputs: list[str] = []
    for view_name in task.outputs:
        cols = task.output_columns.get(view_name, [])
        if cols:
            required_outputs.append(f"- {view_name}: {', '.join(cols)}")
        elif has_column_rules:
            required_outputs.append(f"- {view_name}: (no required columns)")
        else:
            required_outputs.append(f"- {view_name}")
    if not required_outputs:
        required_outputs.append("(none)")

    val_outputs = validation_outputs(task)
    validation_lines: list[str] = []
    if val_outputs:
        validation_lines.append("VALIDATION VIEWS (must create):")
        for v in val_outputs:
            validation_lines.append(f"  - {v}")
        validation_lines.append("")

    parts = [
        f"TASK: {task.name}",
        "",
        "ISSUE:",
        issue_text or "(no issue details)",
        "",
        "REQUIRED OUTPUTS:",
        *required_outputs,
        "",
        "INTENT:",
        task.intent or "(no intent provided)",
        "",
        "ORIGINAL SQL:",
        "```sql",
        sql_block or "-- (no statements provided)",
        "```",
        "",
        *validation_lines,
        f"ALLOWED VIEWS: {', '.join(task.outputs)} or {task.name}_*",
    ]
    return "\n".join(parts)


def _summarize_validation_issue(issue: str) -> str:
    """Summarize validation warnings/failures to view references."""
    issue_text = (issue or "").strip()
    if not issue_text:
        return ""

    warning_views = re.findall(r"Warnings via '([^']+)'", issue_text)
    if issue_text.startswith("Warnings:") and warning_views:
        seen: set[str] = set()
        ordered = [v for v in warning_views if not (v in seen or seen.add(v))]
        targets = ", ".join(f"`{v}`" for v in ordered)
        return f"Validation warnings in {targets}; refer to view for details."

    error_views = re.findall(r"Fail rows in '([^']+)'", issue_text)
    if error_views:
        seen: set[str] = set()
        ordered = [v for v in error_views if not (v in seen or seen.add(v))]
        targets = ", ".join(f"`{v}`" for v in ordered)
        return f"Validation errors in {targets}; refer to view for details."

    return issue_text


# --- Agent entry point ---


async def run_task_agent(
    conn: duckdb.DuckDBPyConnection,
    task: Task,
    client: OpenRouterClient,
    model: str = "openai/gpt-5.2",
    max_iterations: int = 200,
    issue: str = "",
) -> AgentResult:
    """Run the agent for a single Task within a shared workspace database.

    The agent is namespace-restricted: it can only CREATE/DROP views in
    its declared outputs or with its name as prefix. It can SELECT from
    any table/view.

    Args:
        conn: DuckDB connection (shared workspace database).
        task: Task specification.
        issue: Failure detail or warning summary for repair context.
        client: OpenRouterClient (must be in async context).
        model: Model identifier.
        max_iterations: Maximum agent iterations.
    Returns:
        AgentResult with success status, final message, and usage stats.
    """
    init_trace_table(conn)

    start_time = time.time()

    # Namespace enforcement
    allowed_views = set(task.outputs)
    namespace = task.name

    # Build messages
    user_message = build_sql_repair_prompt(task, issue)

    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Build tool list
    tools = [_create_sql_tool()]

    call_model = create_model_callable(client, model, tools)

    # Tool executor with namespace enforcement
    async def tool_executor(name: str, args: dict[str, Any]) -> str:
        if name == "run_sql":
            query = args.get("query", "")
            result = execute_sql(
                conn,
                query,
                allowed_views=allowed_views,
                namespace=namespace,
                task_name=task.name,
            )
            log.debug(".")
            return json.dumps(result, default=_json_default)

        return json.dumps({"success": False, "error": f"Unknown tool: {name}"})

    # Validation: check declared outputs exist, schemas, and validation view
    def validation_fn() -> tuple[bool, str]:
        errors = task.validate(conn)
        if errors:
            return False, "\n".join(f"- {e}" for e in errors)
        return True, ""

    # Iteration callback
    def on_iteration(
        iteration: int,
        assistant_msg: dict[str, Any] | None,
        tool_results: list[dict[str, Any]] | None,
    ) -> None:
        if not tool_results:
            log.debug("(done)")

    log.info("[%s] Starting (outputs: %s)", task.name, ", ".join(task.outputs))

    # Run the agent loop
    result = await run_agent_loop(
        call_model=call_model,
        tool_executor=tool_executor,
        initial_messages=initial_messages,
        validation_fn=validation_fn,
        max_iterations=max_iterations,
        on_iteration=on_iteration,
    )

    elapsed_s = time.time() - start_time

    # Persist per-task metadata
    persist_task_meta(
        conn,
        task.name,
        {
            "model": model,
            "reasoning_effort": client.reasoning_effort,
            "outputs": task.outputs,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls_count,
            "elapsed_s": round(elapsed_s, 1),
            "validation": "PASSED" if result.success else "FAILED",
            "prompt_tokens": result.usage["prompt_tokens"],
            "completion_tokens": result.usage["completion_tokens"],
            "cache_read_tokens": result.usage["cache_read_tokens"],
            "reasoning_tokens": result.usage["reasoning_tokens"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    status = "PASSED" if result.success else "FAILED"
    log.info("[%s] Validation: %s (%.1fs)", task.name, status, elapsed_s)

    return result


# --- SQL-only task executor ---


def _is_sql_only_statement_allowed(query: str) -> tuple[bool, str]:
    """Restrict SQL-only tasks to CREATE/DROP VIEW|MACRO statements.

    SQL-only tasks are intended to be fully deterministic and side-effect free
    beyond defining views/macros. We explicitly disallow SELECT/EXPLAIN even
    though the agent tool allows them.
    """
    try:
        statements = _get_parser_conn().extract_statements(query)
    except duckdb.ParserException as e:
        return False, f"SQL parse error: {e}"
    except Exception as e:
        return False, f"SQL parse error: {e}"

    if not statements:
        return False, "Empty query"
    if len(statements) > 1:
        return False, "Only one statement allowed at a time"

    stmt = statements[0]
    st = stmt.type
    StatementType = cast(Any, duckdb.StatementType)

    if st not in {StatementType.CREATE, StatementType.DROP}:
        return False, "SQL-only tasks only allow CREATE/DROP VIEW|MACRO statements"
    return True, ""


async def run_sql_only_task(
    conn: duckdb.DuckDBPyConnection,
    task: Task,
) -> AgentResult:
    """Execute a deterministic SQL-only task.

    The task provides a list of SQL statements (task.sql or task.sql_strict).
    Each statement is executed with the same namespace enforcement as agent tasks.
    """
    init_trace_table(conn)

    start_time = time.time()

    statements = task.sql_statements()
    if not statements:
        return AgentResult(
            success=False,
            final_message="SQL-only task has no SQL statements",
            iterations=0,
            messages=[],
            tool_calls_count=0,
        )

    allowed_views = set(task.outputs)
    namespace = task.name

    mode_label = "sql_strict" if task.run_mode() == "sql_strict" else "sql"

    for q in statements:
        ok, err = _is_sql_only_statement_allowed(q)
        if not ok:
            persist_task_meta(
                conn,
                task.name,
                {
                    "model": mode_label,
                    "outputs": task.outputs,
                    "iterations": 0,
                    "tool_calls": 0,
                    "elapsed_s": round(time.time() - start_time, 1),
                    "validation": "FAILED",
                    "error": err,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            return AgentResult(
                success=False,
                final_message=err,
                iterations=0,
                messages=[],
                tool_calls_count=0,
            )

        result = execute_sql(
            conn,
            q,
            allowed_views=allowed_views,
            namespace=namespace,
            task_name=task.name,
        )
        if not result.get("success", False):
            err_msg = str(result.get("error") or "SQL execution failed")
            persist_task_meta(
                conn,
                task.name,
                {
                    "model": mode_label,
                    "outputs": task.outputs,
                    "iterations": 0,
                    "tool_calls": 0,
                    "elapsed_s": round(time.time() - start_time, 1),
                    "validation": "FAILED",
                    "error": err_msg,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
            )
            return AgentResult(
                success=False,
                final_message=err_msg,
                iterations=0,
                messages=[],
                tool_calls_count=0,
            )

    errors = task.validate(conn)
    elapsed_s = time.time() - start_time
    if errors:
        msg = "\n".join(f"- {e}" for e in errors)
        persist_task_meta(
            conn,
            task.name,
            {
                "model": mode_label,
                "outputs": task.outputs,
                "iterations": 0,
                "tool_calls": 0,
                "elapsed_s": round(elapsed_s, 1),
                "validation": "FAILED",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        return AgentResult(
            success=False,
            final_message=msg,
            iterations=0,
            messages=[],
            tool_calls_count=0,
        )

    persist_task_meta(
        conn,
        task.name,
        {
            "model": mode_label,
            "outputs": task.outputs,
            "iterations": 0,
            "tool_calls": 0,
            "elapsed_s": round(elapsed_s, 1),
            "validation": "PASSED",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    return AgentResult(
        success=True,
        final_message="OK",
        iterations=0,
        messages=[],
        tool_calls_count=0,
    )
