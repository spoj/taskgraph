"""Node agent — executes a single node within a DuckDB workspace.

Each node runs as an independent agent that can only CREATE/DROP views
within its namespace prefix ({name}_*). SELECTs can read anything.
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
from .agent_loop import run_agent_loop, AgentResult, DEFAULT_MAX_ITERATIONS
from .api import OpenRouterClient, create_model_callable, DEFAULT_MODEL
from .namespace import Namespace
from .task import Node
from .sql_utils import (
    get_parser_conn,
    get_column_schema,
    extract_create_name,
)

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
You build node output views based on the node prompt and required outputs.
You can read anything but can only write views/macros within the node's allowed names.

RULES:
- Allowed: SELECT, SUMMARIZE, EXPLAIN, CREATE/DROP VIEW, CREATE/DROP MACRO. Nothing else.
- Use CREATE OR REPLACE VIEW for outputs; namespace-prefix intermediates ({name}_*)
- Do not create validation views ({name}__validation*); the system creates those separately
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
- PIVOT: PIVOT t ON col USING sum(val) — wide format. UNPIVOT t ON col1, col2 INTO NAME k VALUE v
- COLUMNS(*) — apply expression to all columns: SELECT MIN(COLUMNS(*)) FROM t
  COLUMNS(c -> c LIKE '%amt%') — lambda filter on column names
- Dates: date_diff('day', a, b), date_trunc('month', d), strftime(d, '%Y-%m'), make_date(y,m,d)
  current_date, interval '3 days', d + INTERVAL 1 MONTH
- Strings: regexp_extract(s, pattern, group), regexp_replace(s, pat, repl),
  string_split(s, delim), string_agg(col, ', '), concat_ws('-', a, b, c)
- JSON: col->>'key' (extract as text), col->'key' (extract as JSON),
  json_extract_string(col, '$.path'), json_group_array(col), json_group_object(k, v)
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


def _extract_drop_name(query: str) -> str | None:
    """Extract the object name from a DROP VIEW/MACRO statement."""
    m = _DROP_NAME_RE.search(query)
    return m.group(1) if m else None


def is_sql_allowed(
    query: str,
    namespace: Namespace | None = None,
    ddl_only: bool = False,
) -> tuple[bool, str]:
    """Check if a SQL query is allowed.

    Uses DuckDB's own parser (extract_statements) for statement type
    classification, then regex for name extraction on CREATE/DROP.

    Args:
        query: SQL query string.
        namespace: If set, name enforcement is delegated to
            ``namespace.check_name()``.  If None, any name is allowed.
        ddl_only: If True, only allow CREATE/DROP VIEW|MACRO statements.
    """
    try:
        statements = get_parser_conn().extract_statements(query)
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
            return False, "SQL-only nodes only allow CREATE/DROP VIEW|MACRO statements"
        return True, ""

    # EXPLAIN — allowed (read-only introspection)
    if st == StatementType.EXPLAIN:
        if ddl_only:
            return False, "SQL-only nodes only allow CREATE/DROP VIEW|MACRO statements"
        return True, ""

    # CREATE — only VIEW and MACRO
    if st == StatementType.CREATE:
        sql = stmt.query
        kind_match = _CREATE_KIND_RE.search(sql)
        if kind_match:
            kind = kind_match.group(1).upper()
            label = "view" if kind == "VIEW" else "macro"
            name = extract_create_name(sql)
            if namespace is not None:
                ok, err = namespace.check_name(name, label, "create")
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
            name = _extract_drop_name(sql)
            if namespace is not None:
                ok, err = namespace.check_name(name, label, "drop")
                if not ok:
                    return False, err
            return True, ""
        return False, "Only DROP VIEW and DROP MACRO are permitted."

    return False, f"{st.name} is not allowed."


# --- Metadata persistence ---


def persist_node_meta(
    conn: duckdb.DuckDBPyConnection, node_name: str, meta: dict[str, Any]
) -> None:
    """Persist per-node run metadata in _node_meta.

    _node_meta is a simple per-node JSON blob. It is overwritten per run.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _node_meta (
            node VARCHAR PRIMARY KEY,
            meta_json VARCHAR NOT NULL
        )
    """)

    conn.execute("DELETE FROM _node_meta WHERE node = ?", [node_name])
    conn.execute(
        "INSERT INTO _node_meta (node, meta_json) VALUES (?, ?)",
        [node_name, json.dumps(meta, sort_keys=True)],
    )


# --- Trace ---


def init_trace_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the _trace table and derived views.

    Also creates ``_view_definitions`` as a view on ``_trace`` — it
    extracts the last successful ``CREATE VIEW`` statement per view name,
    giving the same ``(node, view_name, sql)`` interface that the old
    ``_view_definitions`` table provided, but derived entirely from the
    trace log.
    """
    conn.execute("CREATE SEQUENCE IF NOT EXISTS _trace_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _trace (
            id INTEGER DEFAULT nextval('_trace_seq'),
            timestamp TIMESTAMP DEFAULT current_timestamp,
            node VARCHAR,
            source VARCHAR,
            query VARCHAR NOT NULL,
            success BOOLEAN NOT NULL,
            error VARCHAR,
            row_count INTEGER,
            elapsed_ms DOUBLE
        )
    """)
    # Add source column to existing databases that lack it.
    try:
        conn.execute("ALTER TABLE _trace ADD COLUMN source VARCHAR")
    except duckdb.Error:
        pass  # column already exists
    conn.execute(r"""
        CREATE OR REPLACE VIEW _view_definitions AS
        WITH actions AS (
            SELECT id, node, query,
                   regexp_extract(
                       query,
                       '(?i)VIEW\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?"?(\w+)"?',
                       1
                   ) AS view_name,
                   CASE WHEN regexp_matches(query, '(?i)^\s*DROP\s')
                        THEN 'drop' ELSE 'create' END AS action
            FROM _trace
            WHERE success = true
              AND (regexp_matches(query, '(?i)^\s*CREATE\s+(OR\s+REPLACE\s+)?VIEW\s')
                   OR regexp_matches(query, '(?i)^\s*DROP\s+VIEW\s'))
        ),
        latest AS (
            SELECT *,
                   row_number() OVER (
                       PARTITION BY view_name ORDER BY id DESC
                   ) AS rn
            FROM actions
            WHERE view_name IS NOT NULL
        )
        SELECT node, view_name, query AS sql
        FROM latest
        WHERE rn = 1 AND action = 'create'
    """)


def log_trace(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    success: bool,
    error: str | None = None,
    row_count: int | None = None,
    elapsed_ms: float | None = None,
    node_name: str | None = None,
    source: str | None = None,
) -> None:
    """Log a SQL query execution to the _trace table.

    Args:
        source: Origin of the query — ``'agent'``, ``'sql_node'``,
            ``'node_validation'``, or ``'input_validation'``.
    """
    conn.execute(
        """
        INSERT INTO _trace (node, source, query, success, error, row_count, elapsed_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [node_name, source, query, success, error, row_count, elapsed_ms],
    )


# --- SQL execution ---


def execute_sql(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    namespace: Namespace | None = None,
    node_name: str | None = None,
    query_timeout_s: int = DEFAULT_QUERY_TIMEOUT_S,
    max_result_chars: int = MAX_RESULT_CHARS,
    ddl_only: bool = False,
    source: str | None = None,
) -> dict[str, Any]:
    """Execute SQL query and return results.

    Results exceeding *max_result_chars* when serialized are rejected
    with an error asking the caller to add LIMIT or narrow the query.

    Args:
        namespace: If set, DDL name enforcement is applied via
            ``namespace.check_name()``.  If None, no name restrictions.
        query_timeout_s: Maximum seconds a query can run before being
            interrupted. Uses conn.interrupt() via a background timer thread.
            Set to 0 to disable.
        max_result_chars: Maximum characters in the JSON-serialized result.
            Set to 0 to disable.
        source: Origin of the query (passed through to ``log_trace``).
    """
    start_time = time.perf_counter()

    allowed, error_msg = is_sql_allowed(
        query,
        namespace=namespace,
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
            node_name=node_name,
            source=source,
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

    success = False
    error_str: str | None = None
    row_count: int | None = None
    result: dict[str, Any] = {}

    try:
        cursor = conn.execute(query)

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            success = True
            row_count = len(rows)
            result = {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": row_count,
            }
            # Check serialized size — reject oversized results so the
            # model learns to use LIMIT or narrow its query.
            if max_result_chars > 0 and len(rows) > 0:
                serialized = json.dumps(result, default=_json_default)
                if len(serialized) > max_result_chars:
                    result = {
                        "success": False,
                        "error": (
                            f"Result too large ({len(rows)} rows, "
                            f"{len(serialized):,} chars). "
                            "Add LIMIT or narrow your query."
                        ),
                    }
        else:
            success = True
            row_count = 0
            result = {"success": True, "message": "OK"}
    except duckdb.InterruptException:
        error_str = f"Query timed out after {query_timeout_s}s"
        log.warning("[%s] %s", node_name or "?", error_str)
        result = {"success": False, "error": error_str}
    except Exception as e:
        error_str = str(e)
        # If we timed out but got a different exception type, label it clearly
        if timed_out:
            error_str = f"Query timed out after {query_timeout_s}s ({error_str})"
        result = {"success": False, "error": error_str}
    finally:
        if timer is not None:
            timer.cancel()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    log_trace(
        conn,
        query,
        success=success,
        error=error_str,
        row_count=row_count,
        elapsed_ms=elapsed_ms,
        node_name=node_name,
        source=source,
    )
    return result


def _discover_available_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return all non-internal table names from the database."""
    rows = conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE internal = false ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def _format_table_schema_lines(
    conn: duckdb.DuckDBPyConnection, tables: list[str] | None = None
) -> list[str]:
    """Format schema lines for the given tables (or all available tables).

    If *tables* is None, discovers all non-internal tables from the DB.
    """
    if tables is None:
        tables = _discover_available_tables(conn)
    if not tables:
        return ["(none)"]

    lines: list[str] = []
    for table_name in tables:
        rows = get_column_schema(conn, table_name)
        if rows:
            cols = ", ".join(f"{name} {dtype}" for name, dtype in rows)
            lines.append(f"- {table_name} ({cols})")
        else:
            lines.append(f"- {table_name} (schema unavailable)")

    return lines


def build_transform_prompt(
    node: Node,
    input_lines: list[str] | None = None,
) -> str:
    """Build a prompt for the LLM to produce node outputs.

    Args:
        node: Node specification.
        input_lines: Pre-formatted table schema lines. If None, a
            placeholder is shown (caller should pass discovered tables).
    """
    # Required outputs from output_columns keys
    required_outputs: list[str] = []
    if node.output_columns:
        for view_name, cols in node.output_columns.items():
            if cols:
                required_outputs.append(f"- {view_name}: {', '.join(cols)}")
            else:
                required_outputs.append(f"- {view_name}: (no required columns)")
    else:
        required_outputs.append(
            f"(none declared — create at least one view named {node.name}_*)"
        )

    parts = [
        f"NODE: {node.name}",
        "",
        "PROMPT:",
        node.prompt or "(no prompt provided)",
        "",
        "AVAILABLE TABLES:",
        *(
            input_lines
            if input_lines is not None
            else [
                "(discover via: SELECT table_name FROM duckdb_tables() WHERE internal = false)"
            ]
        ),
        "",
        "REQUIRED OUTPUTS:",
        *required_outputs,
    ]

    if node.has_validation():
        parts.extend(
            [
                "",
                f"VALIDATION SQL (runs after transform; creates {node.name}__validation* views):",
                node.validate_sql,
            ]
        )

    parts.extend(
        [
            "",
            f"ALLOWED VIEWS: {node.name}_* (all views must start with {node.name}_)",
        ]
    )

    return "\n".join(parts)


# --- Agent entry point ---


async def run_node_agent(
    conn: duckdb.DuckDBPyConnection,
    node: Node,
    client: OpenRouterClient,
    model: str = DEFAULT_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentResult:
    """Run the agent for a single Node within a shared workspace database.

    The agent is namespace-restricted: it can only CREATE/DROP views
    with its name as prefix ({name}_*). It can SELECT from any table/view.

    Validation (structural + SQL validation views) runs inside the agent
    loop. If validation fails, the agent gets feedback and continues
    within the same iteration budget.

    Args:
        conn: DuckDB connection (shared workspace database).
        node: Node specification.
        client: OpenRouterClient (must be in async context).
        model: Model identifier.
        max_iterations: Maximum agent iterations.
    Returns:
        AgentResult with success status, final message, and usage stats.
    """
    start_time = time.time()
    ns = Namespace.for_node(node)

    # Build messages
    input_lines = _format_table_schema_lines(conn)
    user_message = build_transform_prompt(node, input_lines)
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
                conn, query, namespace=ns, node_name=node.name, source="agent"
            )
            log.debug(".")
            return json.dumps(result, default=_json_default)

        return json.dumps({"success": False, "error": f"Unknown tool: {name}"})

    # Validation: structural checks + SQL validation views
    def validation_fn() -> tuple[bool, str]:
        errors = validate_node_complete(conn, node)
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

    # Persist per-node metadata
    persist_node_meta(
        conn,
        node.name,
        {
            "model": model,
            "reasoning_effort": client.reasoning_effort,
            "depends_on": node.depends_on,
            "output_columns": {k: v for k, v in node.output_columns.items()},
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
    log.info("[%s] Validation: %s (%.1fs)", node.name, status, elapsed_s)

    return result


# --- SQL-only node executor ---


async def run_sql_node(
    conn: duckdb.DuckDBPyConnection,
    node: Node,
) -> AgentResult:
    """Execute a deterministic SQL-only node.

    The node provides a list of SQL statements (node.sql).
    Each statement is executed with the same namespace enforcement as agent nodes.
    """
    start_time = time.time()

    statements = node.sql_statements()
    if not statements:
        return AgentResult(
            success=False,
            final_message="SQL-only node has no SQL statements",
            iterations=0,
            messages=[],
            tool_calls_count=0,
        )

    ns = Namespace.for_node(node)
    execution_error: str | None = None

    for q in statements:
        result = execute_sql(
            conn, q, namespace=ns, node_name=node.name, ddl_only=True, source="sql_node"
        )
        if not result.get("success", False):
            execution_error = str(result.get("error") or "SQL execution failed")
            break

    validation_errors: list[str] = []
    if execution_error is None:
        validation_errors = validate_node_complete(conn, node)

    elapsed_s = time.time() - start_time
    validation_status = "FAILED" if execution_error or validation_errors else "PASSED"
    meta: dict[str, Any] = {
        "model": "sql",
        "depends_on": node.depends_on,
        "output_columns": {k: v for k, v in node.output_columns.items()},
        "iterations": 0,
        "tool_calls": 0,
        "elapsed_s": round(elapsed_s, 1),
        "validation": validation_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if execution_error:
        meta["error"] = execution_error

    persist_node_meta(conn, node.name, meta)

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
    node: Node,
) -> list[str]:
    """Execute validate_sql statements with validation namespace restrictions.

    Returns a list of error messages (empty = success).
    """
    statements = node.validate_sql_statements()
    if not statements:
        return []

    vns = Namespace.for_validation(node)

    for q in statements:
        result = execute_sql(
            conn,
            q,
            namespace=vns,
            node_name=node.name,
            ddl_only=True,
            source="node_validation",
        )
        if not result.get("success", False):
            return [str(result.get("error") or "validate_sql execution failed")]

    return []


def validate_node_complete(
    conn: duckdb.DuckDBPyConnection,
    node: Node,
) -> list[str]:
    """Run full node validation and return error messages (empty = pass)."""
    errors = node.validate_outputs(conn)
    if errors:
        return errors
    if node.has_validation():
        errors = run_validate_sql(conn=conn, node=node)
        if errors:
            return errors
        errors = node.validate_validation_views(conn)
        if errors:
            return errors
    return []
