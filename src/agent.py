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
from typing import Any, Callable, cast
from .agent_loop import run_agent_loop, AgentResult
from .api import OpenRouterClient, create_model_callable
from .task import Task, is_validation_view_for_task

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

SYSTEM_PROMPT = """You are a SQL transform agent working in a DuckDB database.
You build task output views based on the task prompt and required outputs.
You can read anything but can only write views/macros within the task's allowed names.

RULES:
- Allowed: SELECT, SUMMARIZE, EXPLAIN, CREATE/DROP VIEW, CREATE/DROP MACRO. Nothing else.
- Use CREATE OR REPLACE VIEW for outputs; namespace-prefix intermediates (task_name_*)
- Do not create validation views (task_name__validation*); the system creates those separately
- Batch independent run_sql calls in parallel to minimize rounds
- When done, reply with a short message (no tool calls) to trigger validation
- If validation fails, you get feedback — fix and reply again

MACROS:
- CREATE MACRO name(args) AS expr — reusable scalar
- CREATE MACRO name(args) AS TABLE (SELECT ...) — reusable table function

CATALOG:
  SELECT view_name, sql FROM duckdb_views() WHERE internal = false
  SELECT table_name FROM duckdb_tables()

DUCKDB DIALECT:
- Identifiers: double-quote ("col"), not backticks. Quote reserved words. Always use AS for aliases.
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
            "description": "Execute a SQL query against the DuckDB database. Allowed: SELECT, SUMMARIZE, EXPLAIN, CREATE/DROP VIEW, CREATE/DROP MACRO. No tables, inserts, updates, or deletes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute (SELECT, SUMMARIZE, EXPLAIN, CREATE/DROP VIEW, CREATE/DROP MACRO)",
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
    forbidden_names: Callable[[str], bool] | None = None,
    ddl_only: bool = False,
) -> tuple[bool, str]:
    """Check if a SQL query is allowed.

    Uses DuckDB's own parser (extract_statements) for statement type
    classification, then regex for name extraction on CREATE/DROP.

    Args:
        query: SQL query string.
        allowed_views: If set, only these view/macro names can be created/dropped
            (in addition to namespace-prefixed names).
        namespace: If set, names starting with "{namespace}_" are also allowed.
        forbidden_names: Optional function that returns True for disallowed names.
        ddl_only: If True, only allow CREATE/DROP VIEW|MACRO statements.
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
        if ddl_only:
            return False, "SQL-only tasks only allow CREATE/DROP VIEW|MACRO statements"
        return True, ""

    # EXPLAIN — allowed (read-only introspection)
    if st == StatementType.EXPLAIN:
        if ddl_only:
            return False, "SQL-only tasks only allow CREATE/DROP VIEW|MACRO statements"
        return True, ""

    # CREATE — only VIEW and MACRO
    if st == StatementType.CREATE:
        sql = stmt.query
        kind_match = _CREATE_KIND_RE.search(sql)
        if kind_match:
            kind = kind_match.group(1).upper()
            label = "view" if kind == "VIEW" else "macro"
            name = _extract_name(sql, _CREATE_NAME_RE)
            if name and forbidden_names and forbidden_names(name):
                return (
                    False,
                    (
                        f"Cannot create/drop validation view '{name}' during transform. "
                        "Validation views are created in validate_sql."
                    ),
                )
            if allowed_views is not None or namespace is not None:
                ok, err = _check_name_allowed(
                    label, "create", name, allowed_views, namespace
                )
                if not ok:
                    return False, err
            return True, ""
        return False, "Only CREATE VIEW and CREATE MACRO are permitted."

    # DROP — only VIEW and MACRO
    if st == StatementType.DROP:
        sql = stmt.query
        kind_match = _DROP_KIND_RE.search(sql)
        if kind_match:
            kind = kind_match.group(1).upper()
            label = "view" if kind == "VIEW" else "macro"
            name = _extract_name(sql, _DROP_NAME_RE)
            if name and forbidden_names and forbidden_names(name):
                return (
                    False,
                    (
                        f"Cannot create/drop validation view '{name}' during transform. "
                        "Validation views are created in validate_sql."
                    ),
                )
            if allowed_views is not None or namespace is not None:
                ok, err = _check_name_allowed(
                    label, "drop", name, allowed_views, namespace
                )
                if not ok:
                    return False, err
            return True, ""
        return False, "Only DROP VIEW and DROP MACRO are permitted."

    return False, f"{st.name} is not allowed."


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
    forbidden_names: Callable[[str], bool] | None = None,
    ddl_only: bool = False,
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

    allowed, error_msg = is_sql_allowed(
        query,
        allowed_views=allowed_views,
        namespace=namespace,
        forbidden_names=forbidden_names,
        ddl_only=ddl_only,
    )
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


def _format_input_schema_lines(
    conn: duckdb.DuckDBPyConnection, inputs: list[str]
) -> list[str]:
    if not inputs:
        return ["(none)"]

    lines: list[str] = []
    for table_name in inputs:
        try:
            rows = conn.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = ?
                ORDER BY ordinal_position
                """,
                [table_name],
            ).fetchall()
        except duckdb.Error:
            rows = []

        if rows:
            cols = ", ".join(f"{name} {dtype}" for name, dtype in rows)
            lines.append(f"- {table_name} ({cols})")
        else:
            lines.append(f"- {table_name} (schema unavailable)")

    return lines


def build_transform_prompt(
    task: Task,
    input_lines: list[str] | None = None,
) -> str:
    """Build a prompt for the LLM to produce task outputs."""
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

    parts = [
        f"TASK: {task.name}",
        "",
        "PROMPT:",
        task.prompt or "(no prompt provided)",
        "",
        "INPUTS:",
        *(
            input_lines
            if input_lines is not None
            else [", ".join(task.inputs) if task.inputs else "(none)"]
        ),
        "",
        "REQUIRED OUTPUTS:",
        *required_outputs,
    ]

    if task.has_validation():
        parts.extend(
            [
                "",
                f"VALIDATION SQL (runs after transform; creates {task.name}__validation* views):",
                task.validate_sql,
            ]
        )

    parts.extend(
        [
            "",
            f"ALLOWED VIEWS: {', '.join(task.outputs)} or {task.name}_*",
        ]
    )

    return "\n".join(parts)


# --- Agent entry point ---


async def run_task_agent(
    conn: duckdb.DuckDBPyConnection,
    task: Task,
    client: OpenRouterClient,
    model: str = "openai/gpt-5.2",
    max_iterations: int = 200,
) -> AgentResult:
    """Run the agent for a single Task within a shared workspace database.

    The agent is namespace-restricted: it can only CREATE/DROP views in
    its declared outputs or with its name as prefix. It can SELECT from
    any table/view.

    Validation (structural + SQL validation views) runs inside the agent
    loop. If validation fails, the agent gets feedback and continues
    within the same iteration budget.

    Args:
        conn: DuckDB connection (shared workspace database).
        task: Task specification.
        client: OpenRouterClient (must be in async context).
        model: Model identifier.
        max_iterations: Maximum agent iterations.
    Returns:
        AgentResult with success status, final message, and usage stats.
    """
    start_time = time.time()

    # Namespace enforcement
    allowed_views = set(task.outputs)
    namespace = task.name

    # Build messages
    input_lines = _format_input_schema_lines(conn, task.inputs)
    user_message = build_transform_prompt(task, input_lines)
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
            forbidden_names = lambda view_name: is_validation_view_for_task(
                view_name, task.name
            )
            result = execute_sql(
                conn,
                query,
                allowed_views=allowed_views,
                namespace=namespace,
                task_name=task.name,
                forbidden_names=forbidden_names,
            )
            log.debug(".")
            return json.dumps(result, default=_json_default)

        return json.dumps({"success": False, "error": f"Unknown tool: {name}"})

    # Validation: structural checks + SQL validation views
    def validation_fn() -> tuple[bool, str]:
        errors = validate_task_complete(conn, task)
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


async def run_sql_only_task(
    conn: duckdb.DuckDBPyConnection,
    task: Task,
) -> AgentResult:
    """Execute a deterministic SQL-only task.

    The task provides a list of SQL statements (task.sql).
    Each statement is executed with the same namespace enforcement as agent tasks.
    """
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

    mode_label = "sql"
    forbidden_names = lambda view_name: is_validation_view_for_task(
        view_name, task.name
    )
    execution_error: str | None = None

    for q in statements:
        result = execute_sql(
            conn,
            q,
            allowed_views=allowed_views,
            namespace=namespace,
            task_name=task.name,
            forbidden_names=forbidden_names,
            ddl_only=True,
        )
        if not result.get("success", False):
            execution_error = str(result.get("error") or "SQL execution failed")
            break

    validation_errors: list[str] = []
    if execution_error is None:
        validation_errors = validate_task_complete(conn, task)

    elapsed_s = time.time() - start_time
    validation_status = "FAILED" if execution_error or validation_errors else "PASSED"
    meta: dict[str, Any] = {
        "model": mode_label,
        "outputs": task.outputs,
        "iterations": 0,
        "tool_calls": 0,
        "elapsed_s": round(elapsed_s, 1),
        "validation": validation_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if execution_error:
        meta["error"] = execution_error

    persist_task_meta(conn, task.name, meta)

    if execution_error or validation_errors:
        msg = execution_error or "\n".join(f"- {e}" for e in validation_errors)
        return AgentResult(
            success=False,
            final_message=msg,
            iterations=0,
            messages=[],
            tool_calls_count=0,
        )

    return AgentResult(
        success=True,
        final_message="OK",
        iterations=0,
        messages=[],
        tool_calls_count=0,
    )


def run_validate_sql(
    conn: duckdb.DuckDBPyConnection,
    task: Task,
) -> list[str]:
    """Execute validate_sql statements with validation namespace restrictions.

    Returns a list of error messages (empty = success).
    """
    statements = task.validate_sql_statements()
    if not statements:
        return []

    allowed_views = set(task.validation_view_names())
    namespace = f"{task.name}__validation"

    for q in statements:
        result = execute_sql(
            conn,
            q,
            allowed_views=allowed_views,
            namespace=namespace,
            task_name=task.name,
            ddl_only=True,
        )
        if not result.get("success", False):
            return [str(result.get("error") or "validate_sql execution failed")]

    return []


def validate_task_complete(
    conn: duckdb.DuckDBPyConnection,
    task: Task,
) -> list[str]:
    """Run full task validation and return error messages (empty = pass)."""
    errors = task.validate_transform(conn)
    if errors:
        return errors
    if task.has_validation():
        errors = run_validate_sql(conn=conn, task=task)
        if errors:
            return errors
        errors = task.validate_validation_views(conn)
        if errors:
            return errors
    return []
