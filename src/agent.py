"""Node agent — executes a single node within a DuckDB workspace.

Each node runs as an independent agent that can only CREATE/DROP views
within its namespace prefix ({name}_*). SELECTs can read anything.
"""

import json
import logging
import duckdb
import threading
import time
import datetime
import decimal
import uuid
from typing import Any, cast
from .agent_loop import run_agent_loop, AgentResult, DEFAULT_MAX_ITERATIONS
from .api import OpenRouterClient, create_model_callable, DEFAULT_MODEL
from .catalog import list_tables, list_views
from .infra import log_trace, persist_node_meta, ensure_trace as init_trace_table
from .namespace import Namespace
from .task import Node, validation_view_prefix
from .sql_utils import (
    get_column_schema,
    SqlParseError,
    extract_ddl_target,
    parse_one_statement,
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
- When done, reply with a completion note (no tool calls) to trigger validation.
  Include: created/updated view names, key decisions, and any assumptions/warnings.
  If the node prompt asks for a full report, comply.
- If validation fails, you get feedback — fix and reply again

MACROS:
- CREATE MACRO name(args) AS expr — reusable scalar
- CREATE MACRO name(args) AS TABLE (SELECT ...) — reusable table function

CATALOG:
  SELECT view_name, sql FROM duckdb_views() WHERE internal = false
  SELECT table_name FROM duckdb_tables() WHERE internal = false

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
- Literal tables: SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, val)
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
        parsed = parse_one_statement(query)
    except SqlParseError as e:
        return False, str(e)

    st = parsed.stmt_type

    # DuckDB's Python typing for StatementType is incomplete in some versions;
    # cast to Any so pyright can type-check the rest of this module.
    StatementType = cast(Any, duckdb.StatementType)

    # Read-only statements (SELECT, EXPLAIN, SUMMARIZE, DESCRIBE)
    if st in (StatementType.SELECT, StatementType.EXPLAIN):
        if ddl_only:
            return False, "SQL-only nodes only allow CREATE/DROP VIEW|MACRO statements"
        return True, ""

    # CREATE/DROP — only VIEW and MACRO, with namespace enforcement
    if st in (StatementType.CREATE, StatementType.DROP):
        sql = parsed.sql
        ddl = extract_ddl_target(sql)
        if ddl is None:
            verb = "CREATE" if st == StatementType.CREATE else "DROP"
            return False, f"Only {verb} VIEW and {verb} MACRO are permitted."
        label = ddl.kind
        action = ddl.action
        name = ddl.name
        if namespace is not None:
            ok, err = namespace.check_name(name, label, action)
            if not ok:
                return False, err
        return True, ""

    st_name = getattr(st, "name", str(st))
    return False, f"{st_name} is not allowed."


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
                    success = False
                    error_str = (
                        f"The return set serialized to more than the length limit of {max_result_chars:,} chars. "
                        "Try again with more targeted queries, e.g., using a LIMIT clause or requesting only specific columns."
                    )
                    result = {
                        "success": False,
                        "error": error_str,
                    }
        else:
            success = True
            row_count = 0
            result = {"success": True, "message": "OK"}
    except duckdb.InterruptException:
        error_str = f"Query execution exceeded the {query_timeout_s} second timeout limit and was aborted. Please reformulate your approach with faster, optimized queries."
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
    return list_tables(conn, exclude_prefixes=("_",))


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
    """Build a prompt for the LLM to produce node outputs."""
    if node.output_columns:
        required_outputs = [
            f"- {vn}: {', '.join(cols)}" if cols else f"- {vn}: (no required columns)"
            for vn, cols in node.output_columns.items()
        ]
    else:
        required_outputs = [
            f"(none declared — create at least one view named {node.name}_*)"
        ]

    parts = [
        f"NODE: {node.name}",
        "",
        "PROMPT:",
        node.prompt or "(no prompt provided)",
        "",
        "AVAILABLE TABLES:",
        *(
            input_lines
            or [
                "(discover via: SELECT table_name FROM duckdb_tables() WHERE internal = false)"
            ]
        ),
        "",
        "REQUIRED OUTPUTS:",
        *required_outputs,
    ]

    if node.has_validation():
        parts.append("")
        parts.append(
            f"VALIDATION (each query is wrapped into {node.name}__validation_<check_name> views):"
        )
        for check_name, query in sorted(node.validation_queries().items()):
            view_name = f"{validation_view_prefix(node.name)}_{check_name}"
            parts.extend([f"- {view_name}:", query])

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

    status = "PASSED" if result.success else "FAILED"
    elapsed_s = time.time() - start_time
    log.info("[%s] Validation: %s (%.1fs)", node.name, status, elapsed_s)

    # Persist the agent's final assistant text as a trace event.
    # This is valuable commentary for later inspection and is also used
    # by the harness final report flow.
    try:
        log_trace(
            conn,
            "ASSISTANT_FINAL",
            success=result.success,
            kind="assistant_final",
            content=result.final_message,
            elapsed_ms=elapsed_s * 1000,
            node_name=node.name,
            source="assistant",
        )
    except Exception:
        # Never fail a node due to reporting/trace persistence issues.
        pass

    return result


# --- SQL-only node executor ---


def _simple_result(success: bool, message: str) -> AgentResult:
    """Create an AgentResult for non-agent nodes (source/sql)."""
    return AgentResult(
        success=success,
        final_message=message,
        iterations=0,
        messages=[],
        tool_calls_count=0,
    )


async def run_sql_node(
    conn: duckdb.DuckDBPyConnection,
    node: Node,
) -> AgentResult:
    """Execute a deterministic SQL-only node.

    Executes all SQL statements with namespace enforcement. Does NOT
    run validation — the caller (``run_one``) handles the unified
    post-execution flow (validate + materialize) for all node types.
    """
    statements = node.sql_statements()
    if not statements:
        return _simple_result(False, "SQL-only node has no SQL statements")

    ns = Namespace.for_node(node)

    for q in statements:
        result = execute_sql(
            conn, q, namespace=ns, node_name=node.name, ddl_only=True, source="sql_node"
        )
        if not result.get("success", False):
            return _simple_result(
                False, str(result.get("error") or "SQL execution failed")
            )

    return _simple_result(True, "OK")


def run_validate_sql(
    conn: duckdb.DuckDBPyConnection,
    node: Node,
) -> list[str]:
    """Define validation views for a node.

    Validation is specified as a mapping of ``check_name -> query`` on the node.
    Each entry becomes a view named:
      - ``{name}__validation_{check_name}``

    Returns a list of error messages (empty = success).
    """
    validate = node.validation_queries()
    if not validate:
        return []

    vns = Namespace.for_validation(node)

    tables = set(list_tables(conn, exclude_prefixes=()))

    for check_name, query in sorted(validate.items()):
        view_name = f"{validation_view_prefix(node.name)}_{check_name}"

        # If a validation view was already materialized into a table, don't
        # attempt to re-create it as a view.
        if view_name in tables:
            continue

        q = query.rstrip(";").strip()
        ddl = f'CREATE OR REPLACE VIEW "{view_name}" AS\n{q}'
        result = execute_sql(
            conn,
            ddl,
            namespace=vns,
            node_name=node.name,
            ddl_only=True,
            source="node_validation",
        )
        if not result.get("success", False):
            err = str(result.get("error") or "Validation view definition failed")
            return [f"{view_name}: {err}"]

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
        # Define validation views once; subsequent validations just re-evaluate.
        base = validation_view_prefix(node.name)
        expected = [f"{base}_{k}" for k in node.validation_queries().keys()]
        if expected:
            existing = set(list_views(conn, exclude_prefixes=())) | set(
                list_tables(conn, exclude_prefixes=())
            )
            if not all(v in existing for v in expected):
                errors = run_validate_sql(conn=conn, node=node)
                if errors:
                    return errors
        errors = node.validate_validation_views(conn)
        if errors:
            return errors
    return []
