"""Unit tests for Taskgraph system internals.

Tests the internal machinery in isolation (no LLM calls needed):
- Task validation: output existence, output_columns, validation views
- Namespace enforcement: allowed/blocked CREATE VIEW, SELECT passthrough
- DAG resolution: topo-sort layers, cycle detection, duplicate outputs
- Token circuit breaker: budget exceeded stops agent
- Query timeout: conn.interrupt() kills long queries
- Ingestion: DataFrame, list[dict], dict[str,list], callable, _row_id
- Input validation: column checks, SQL checks, per-input validation
"""

import asyncio
import importlib
import json
import sys
import uuid
from pathlib import Path

import duckdb
import polars as pl
import pytest
from click.testing import CliRunner

from src.agent import (
    execute_sql,
    is_sql_allowed,
    init_trace_table,
    _build_task_user_message,
    _json_default,
    DEFAULT_QUERY_TIMEOUT_S,
    MAX_RESULT_CHARS,
)
from src.agent_loop import run_agent_loop, AgentResult, DEFAULT_MAX_TOKENS
from src.ingest import coerce_to_dataframe, ingest_table
from src.task import Task, resolve_dag, validate_task_graph
from src.workspace import Workspace, persist_workspace_meta, read_workspace_meta
from src.agent import run_sql_only_task


def test_cli_run_sql_strict_does_not_require_openrouter_api_key(tmp_path, monkeypatch):
    """sql_strict specs should run without OPENROUTER_API_KEY.

    This is a regression test for the CLI requiring the key even when no
    task would invoke the LLM.
    """
    from scripts.cli import main

    spec_source = """\
INPUTS = {}

TASKS = [
    {
        "name": "t",
        "inputs": [],
        "outputs": ["v"],
        "sql_strict": ["CREATE VIEW v AS SELECT 1 AS x"],
    }
]
"""

    spec_module = _write_spec_module(tmp_path, spec_source)
    out_db = tmp_path / "out.db"

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--spec", spec_module, "-o", str(out_db)],
        env={"OPENROUTER_API_KEY": ""},
    )
    assert result.exit_code == 0, result.output
    assert out_db.exists()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn():
    """In-memory DuckDB connection for each test."""
    c = duckdb.connect(":memory:")
    init_trace_table(c)
    yield c
    c.close()


def _make_task(**kwargs) -> Task:
    """Helper to create a Task with defaults."""
    defaults = {"name": "t", "prompt": "test", "inputs": [], "outputs": []}
    defaults.update(kwargs)
    return Task(**defaults)


def _write_spec_module(tmp_path: Path, source: str) -> str:
    """Create a temporary spec module and return its module path."""
    module_name = f"spec_{uuid.uuid4().hex}"
    module_dir = tmp_path / module_name
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text(source)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    return module_name


# ===========================================================================
# 1. Task Validation
# ===========================================================================


class TestTaskValidation:
    """Tests for Task.validate() — output existence, output_columns, validation views."""

    def test_missing_output_view(self, conn):
        """validate() returns error when a declared output view doesn't exist."""
        task = _make_task(outputs=["my_view"])
        errors = task.validate(conn)
        assert len(errors) == 1
        assert "my_view" in errors[0]
        assert "not created" in errors[0].lower()

    def test_existing_output_view_passes(self, conn):
        """validate() passes when all declared output views exist."""
        conn.execute("CREATE VIEW my_view AS SELECT 1 AS x")
        task = _make_task(outputs=["my_view"])
        errors = task.validate(conn)
        assert errors == []

    def test_multiple_missing_views(self, conn):
        """validate() reports all missing views."""
        conn.execute("CREATE VIEW v1 AS SELECT 1 AS x")
        task = _make_task(outputs=["v1", "v2", "v3"])
        errors = task.validate(conn)
        assert len(errors) == 2  # v2 and v3 missing
        assert any("v2" in e for e in errors)
        assert any("v3" in e for e in errors)

    def test_output_columns_pass(self, conn):
        """output_columns validation passes when view has required columns."""
        conn.execute(
            "CREATE VIEW report AS SELECT 1 AS product, 2.0 AS total, 0.5 AS margin"
        )
        task = _make_task(
            outputs=["report"],
            output_columns={"report": ["product", "total", "margin"]},
        )
        errors = task.validate(conn)
        assert errors == []

    def test_output_columns_missing_column(self, conn):
        """output_columns validation fails when view is missing a required column."""
        conn.execute("CREATE VIEW report AS SELECT 1 AS product, 2.0 AS total")
        task = _make_task(
            outputs=["report"],
            output_columns={"report": ["product", "total", "margin"]},
        )
        errors = task.validate(conn)
        assert len(errors) == 1
        assert "margin" in errors[0]
        assert "product" in errors[0]  # actual columns listed in error

    def test_output_columns_multiple_missing(self, conn):
        """output_columns validation reports all missing columns at once."""
        conn.execute("CREATE VIEW report AS SELECT 1 AS id")
        task = _make_task(
            outputs=["report"],
            output_columns={"report": ["id", "amount", "category"]},
        )
        errors = task.validate(conn)
        assert len(errors) == 1
        assert "amount" in errors[0]
        assert "category" in errors[0]

    def test_output_columns_extra_columns_ok(self, conn):
        """Extra columns beyond required are fine."""
        conn.execute("CREATE VIEW report AS SELECT 1 AS a, 2 AS b, 3 AS c, 4 AS d")
        task = _make_task(
            outputs=["report"],
            output_columns={"report": ["a", "c"]},
        )
        errors = task.validate(conn)
        assert errors == []

    def test_output_columns_skips_missing_view(self, conn):
        """output_columns check skips views that don't exist (caught in step 1)."""
        task = _make_task(
            outputs=["missing_view"],
            output_columns={"missing_view": ["col1"]},
        )
        errors = task.validate(conn)
        # Should get "not created" error from step 1, not a column error
        assert len(errors) == 1
        assert "not created" in errors[0].lower()

    def test_validation_view_pass(self, conn):
        """Validation view passes when it contains no fail rows."""
        conn.execute(
            "CREATE VIEW t__validation AS SELECT 'pass' AS status, 'ok' AS message"
        )
        task = _make_task(name="t", outputs=["t__validation"])
        errors = task.validate(conn)
        assert errors == []

    def test_validation_view_fail(self, conn):
        """Validation view fails when it contains any fail row."""
        conn.execute(
            "CREATE VIEW t__validation AS SELECT 'fail' AS status, 'bad' AS message"
        )
        task = _make_task(name="t", outputs=["t__validation"])
        errors = task.validate(conn)
        assert errors
        assert "bad" in "\n".join(errors)

    def test_multiple_validation_views_enforced(self, conn):
        conn.execute(
            "CREATE VIEW t__validation_a AS SELECT 'pass' AS status, 'ok' AS message"
        )
        conn.execute(
            "CREATE VIEW t__validation_b AS SELECT 'fail' AS status, 'nope' AS message"
        )
        task = _make_task(name="t", outputs=["t__validation_a", "t__validation_b"])
        errors = task.validate(conn)
        assert errors
        # Should mention the failing message
        assert "nope" in "\n".join(errors)

    def test_validation_order_view_before_columns(self, conn):
        """Step 1 (view existence) runs before step 2 (column check)."""
        task = _make_task(
            outputs=["missing"],
            output_columns={"missing": ["col1"]},
        )
        errors = task.validate(conn)
        assert len(errors) == 1
        assert "not created" in errors[0].lower()


# ==========================================================================
# 1b. SQL-only and validation-only tasks
# ==========================================================================


class TestSqlOnlyTasks:
    def test_sql_only_task_executes_and_validates(self, conn):
        task = _make_task(
            name="sql_task",
            inputs=[],
            outputs=["out_view", "sql_task__validation"],
            sql=[
                "CREATE VIEW out_view AS SELECT 1 AS x",
                "CREATE VIEW sql_task__validation AS SELECT 'pass' AS status, 'ok' AS message",
            ],
        )

        result = asyncio.run(run_sql_only_task(conn=conn, task=task))
        assert result.success is True

    def test_sql_only_task_disallows_select(self, conn):
        task = _make_task(
            name="sql_task",
            outputs=["out_view"],
            sql=["SELECT 1"],
        )
        result = asyncio.run(run_sql_only_task(conn=conn, task=task))
        assert result.success is False
        assert "only allow" in result.final_message.lower()

    def test_sql_only_task_requires_sql(self, conn):
        task = _make_task(name="sql_task", outputs=["sql_task__validation"], sql=[])
        result = asyncio.run(run_sql_only_task(conn=conn, task=task))
        assert result.success is False

    def test_validation_view_is_enforced(self, conn):
        # Any view that matches validation naming conventions should be enforced
        conn.execute(
            "CREATE VIEW mytask__validation AS "
            "SELECT 'fail' AS status, 'bad things' AS message"
        )
        task = _make_task(name="mytask", outputs=["mytask__validation"])
        errors = task.validate(conn)
        assert errors
        assert "bad things" in "\n".join(errors)

    def test_validation_order_columns_before_validation_view(self, conn):
        """Column check runs before validation view enforcement."""
        conn.execute("CREATE VIEW v AS SELECT 1 AS wrong_col")
        task = _make_task(
            outputs=["v"],
            output_columns={"v": ["expected_col"]},
        )
        errors = task.validate(conn)
        assert len(errors) == 1
        assert "expected_col" in errors[0]


# ===========================================================================
# 2. Namespace Enforcement
# ===========================================================================


class TestNamespaceEnforcement:
    """Tests for is_sql_allowed() and execute_sql() namespace logic."""

    def test_select_always_allowed(self):
        """SELECT queries are always allowed regardless of namespace."""
        ok, err = is_sql_allowed("SELECT 1", allowed_views={"x"}, namespace="t")
        assert ok
        assert err == ""

    def test_select_from_any_table(self):
        """SELECT FROM any table is allowed."""
        ok, err = is_sql_allowed(
            "SELECT * FROM other_task_output", allowed_views={"x"}, namespace="t"
        )
        assert ok

    def test_union_allowed(self):
        """UNION queries are allowed (they're SELECT variants)."""
        ok, err = is_sql_allowed("SELECT 1 UNION ALL SELECT 2")
        assert ok

    def test_create_view_declared_output(self):
        """CREATE VIEW for a declared output is allowed."""
        ok, err = is_sql_allowed(
            "CREATE VIEW output AS SELECT 1",
            allowed_views={"output", "summary"},
            namespace="task1",
        )
        assert ok

    def test_create_view_namespaced_prefix(self):
        """CREATE VIEW with task name prefix is allowed."""
        ok, err = is_sql_allowed(
            "CREATE VIEW task1_intermediate AS SELECT 1",
            allowed_views={"output"},
            namespace="task1",
        )
        assert ok

    def test_create_view_blocked(self):
        """CREATE VIEW outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "CREATE VIEW other_output AS SELECT 1",
            allowed_views={"output"},
            namespace="task1",
        )
        assert not ok
        assert "other_output" in err

    def test_drop_view_declared_output(self):
        """DROP VIEW for a declared output is allowed."""
        ok, err = is_sql_allowed(
            "DROP VIEW IF EXISTS output",
            allowed_views={"output"},
            namespace="task1",
        )
        assert ok

    def test_drop_view_namespaced(self):
        """DROP VIEW with task name prefix is allowed."""
        ok, err = is_sql_allowed(
            "DROP VIEW task1_temp",
            allowed_views={"output"},
            namespace="task1",
        )
        assert ok

    def test_drop_view_blocked(self):
        """DROP VIEW outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "DROP VIEW other_task_view",
            allowed_views={"output"},
            namespace="task1",
        )
        assert not ok
        assert "other_task_view" in err

    def test_create_table_blocked(self):
        """CREATE TABLE is always blocked."""
        ok, err = is_sql_allowed("CREATE TABLE t (id INT)")
        assert not ok
        assert "VIEW" in err  # error message explains what IS allowed

    def test_insert_blocked(self):
        """INSERT is always blocked."""
        ok, err = is_sql_allowed("INSERT INTO t VALUES (1)")
        assert not ok

    def test_update_blocked(self):
        """UPDATE is always blocked."""
        ok, err = is_sql_allowed("UPDATE t SET x = 1")
        assert not ok

    def test_delete_blocked(self):
        """DELETE is always blocked."""
        ok, err = is_sql_allowed("DELETE FROM t")
        assert not ok

    def test_drop_table_blocked(self):
        """DROP TABLE is always blocked."""
        ok, err = is_sql_allowed("DROP TABLE t")
        assert not ok
        assert "VIEW" in err  # error message explains what IS allowed

    def test_multiple_statements_blocked(self):
        """Multiple statements in one query are blocked."""
        ok, err = is_sql_allowed("SELECT 1; SELECT 2")
        assert not ok
        assert "one statement" in err.lower()

    def test_empty_query_blocked(self):
        """Empty query is blocked."""
        ok, err = is_sql_allowed("")
        assert not ok

    def test_create_temp_view_allowed(self):
        """CREATE TEMP VIEW is allowed (it's a view variant)."""
        ok, err = is_sql_allowed("CREATE TEMP VIEW tmp AS SELECT 1")
        assert ok

    def test_no_namespace_no_restriction(self):
        """Without namespace, any view name is rejected (neither in allowed nor prefixed)."""
        # When both are None, no check is done — CREATE VIEW is always allowed
        ok, err = is_sql_allowed("CREATE VIEW anything AS SELECT 1")
        assert ok

    def test_create_scalar_macro_namespaced(self):
        """CREATE MACRO with namespace prefix is allowed."""
        ok, err = is_sql_allowed(
            "CREATE MACRO task1_clean(s) AS lower(trim(s))",
            allowed_views={"output"},
            namespace="task1",
        )
        assert ok

    def test_create_scalar_macro_blocked(self):
        """CREATE MACRO outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "CREATE MACRO other_clean(s) AS lower(trim(s))",
            allowed_views={"output"},
            namespace="task1",
        )
        assert not ok
        assert "other_clean" in err

    def test_create_table_macro_namespaced(self):
        """CREATE MACRO ... AS TABLE with namespace prefix is allowed."""
        ok, err = is_sql_allowed(
            "CREATE MACRO task1_best(t) AS TABLE SELECT * FROM t",
            allowed_views={"output"},
            namespace="task1",
        )
        assert ok

    def test_create_table_macro_blocked(self):
        """CREATE MACRO ... AS TABLE outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "CREATE MACRO other_best(t) AS TABLE SELECT * FROM t",
            allowed_views={"output"},
            namespace="task1",
        )
        assert not ok
        assert "other_best" in err

    def test_drop_macro_namespaced(self):
        """DROP MACRO with namespace prefix is allowed."""
        ok, err = is_sql_allowed(
            "DROP MACRO task1_clean",
            allowed_views={"output"},
            namespace="task1",
        )
        assert ok

    def test_drop_macro_blocked(self):
        """DROP MACRO outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "DROP MACRO other_clean",
            allowed_views={"output"},
            namespace="task1",
        )
        assert not ok
        assert "other_clean" in err

    def test_drop_macro_if_exists(self):
        """DROP MACRO IF EXISTS is allowed within namespace."""
        ok, err = is_sql_allowed(
            "DROP MACRO IF EXISTS task1_helper",
            allowed_views={"output"},
            namespace="task1",
        )
        assert ok

    def test_create_macro_no_namespace(self):
        """CREATE MACRO without namespace constraints is allowed."""
        ok, err = is_sql_allowed("CREATE MACRO my_func(x) AS x + 1")
        assert ok

    def test_summarize_allowed(self):
        """SUMMARIZE is allowed (read-only data profiling)."""
        ok, err = is_sql_allowed("SUMMARIZE my_table")
        assert ok

    def test_explain_allowed(self):
        """EXPLAIN is allowed (read-only introspection)."""
        ok, err = is_sql_allowed("EXPLAIN SELECT 1")
        assert ok

    def test_execute_sql_namespace_blocks(self, conn):
        """execute_sql() returns error for namespace-violating queries."""
        result = execute_sql(
            conn,
            "CREATE VIEW forbidden AS SELECT 1",
            allowed_views={"output"},
            namespace="task1",
            query_timeout_s=0,
        )
        assert not result["success"]
        assert "forbidden" in result["error"]

    def test_execute_sql_namespace_allows(self, conn):
        """execute_sql() allows queries within namespace."""
        result = execute_sql(
            conn,
            "CREATE VIEW output AS SELECT 1 AS x",
            allowed_views={"output"},
            namespace="task1",
            query_timeout_s=0,
        )
        assert result["success"]

    def test_execute_sql_select_returns_data(self, conn):
        """execute_sql() returns columns and rows for SELECT."""
        conn.execute("CREATE TABLE t (id INT, name VARCHAR)")
        conn.execute("INSERT INTO t VALUES (1, 'alice'), (2, 'bob')")
        result = execute_sql(conn, "SELECT * FROM t ORDER BY id", query_timeout_s=0)
        assert result["success"]
        assert result["columns"] == ["id", "name"]
        assert result["row_count"] == 2
        assert result["rows"] == [(1, "alice"), (2, "bob")]

    def test_execute_sql_logs_trace(self, conn):
        """execute_sql() logs to _trace table."""
        execute_sql(conn, "SELECT 42", task_name="test_task", query_timeout_s=0)
        rows = conn.execute(
            "SELECT task, success FROM _trace WHERE task = 'test_task'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == ("test_task", True)

    def test_execute_sql_logs_error_trace(self, conn):
        """execute_sql() logs failed queries to _trace."""
        execute_sql(
            conn,
            "CREATE TABLE bad (id INT)",
            task_name="test_task",
            query_timeout_s=0,
        )
        rows = conn.execute(
            "SELECT task, success, error FROM _trace WHERE task = 'test_task'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] is False  # success = False
        assert rows[0][2] is not None  # error message present


# ===========================================================================
# 3. DAG Resolution
# ===========================================================================


class TestDAGResolution:
    """Tests for resolve_dag() and validate_task_graph()."""

    def test_single_task_one_layer(self):
        """Single task produces one layer."""
        tasks = [_make_task(name="a", outputs=["v1"])]
        layers = resolve_dag(tasks)
        assert len(layers) == 1
        assert [t.name for t in layers[0]] == ["a"]

    def test_linear_chain(self):
        """A -> B -> C produces 3 layers."""
        tasks = [
            _make_task(name="a", outputs=["v1"]),
            _make_task(name="b", inputs=["v1"], outputs=["v2"]),
            _make_task(name="c", inputs=["v2"], outputs=["v3"]),
        ]
        layers = resolve_dag(tasks)
        assert len(layers) == 3
        assert [t.name for t in layers[0]] == ["a"]
        assert [t.name for t in layers[1]] == ["b"]
        assert [t.name for t in layers[2]] == ["c"]

    def test_diamond_dag(self):
        """Diamond: prep -> (sales, costs) -> report produces 3 layers."""
        tasks = [
            _make_task(name="prep", outputs=["prepared"]),
            _make_task(name="sales", inputs=["prepared"], outputs=["sales_out"]),
            _make_task(name="costs", inputs=["prepared"], outputs=["costs_out"]),
            _make_task(
                name="report", inputs=["sales_out", "costs_out"], outputs=["report_out"]
            ),
        ]
        layers = resolve_dag(tasks)
        assert len(layers) == 3
        assert [t.name for t in layers[0]] == ["prep"]
        assert sorted(t.name for t in layers[1]) == ["costs", "sales"]
        assert [t.name for t in layers[2]] == ["report"]

    def test_parallel_independent_tasks(self):
        """Independent tasks land in the same layer."""
        tasks = [
            _make_task(name="a", outputs=["v1"]),
            _make_task(name="b", outputs=["v2"]),
            _make_task(name="c", outputs=["v3"]),
        ]
        layers = resolve_dag(tasks)
        assert len(layers) == 1
        assert sorted(t.name for t in layers[0]) == ["a", "b", "c"]

    def test_cycle_detection(self):
        """Cycle raises ValueError."""
        tasks = [
            _make_task(name="a", inputs=["v2"], outputs=["v1"]),
            _make_task(name="b", inputs=["v1"], outputs=["v2"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            resolve_dag(tasks)

    def test_duplicate_output_raises(self):
        """Two tasks producing the same output raises ValueError."""
        tasks = [
            _make_task(name="a", outputs=["shared"]),
            _make_task(name="b", outputs=["shared"]),
        ]
        with pytest.raises(ValueError, match="shared"):
            resolve_dag(tasks)

    def test_external_inputs_ignored(self):
        """Inputs not produced by any task are treated as external (pre-ingested)."""
        tasks = [
            _make_task(name="a", inputs=["external_table"], outputs=["v1"]),
        ]
        layers = resolve_dag(tasks)
        assert len(layers) == 1

    def test_validate_task_graph_valid(self):
        """Valid graph with all inputs satisfied."""
        tasks = [
            _make_task(name="a", inputs=["raw_data"], outputs=["v1"]),
            _make_task(name="b", inputs=["v1"], outputs=["v2"]),
        ]
        errors = validate_task_graph(tasks, available_tables={"raw_data"})
        assert errors == []

    def test_validate_task_graph_missing_input(self):
        """Reports missing inputs that are neither tables nor task outputs."""
        tasks = [
            _make_task(name="a", inputs=["nonexistent"], outputs=["v1"]),
        ]
        errors = validate_task_graph(tasks, available_tables=set())
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_validate_task_graph_available_table(self):
        """Inputs available as ingested tables are fine."""
        tasks = [
            _make_task(name="a", inputs=["employees", "departments"], outputs=["v1"]),
        ]
        errors = validate_task_graph(
            tasks, available_tables={"employees", "departments"}
        )
        assert errors == []

    def test_complex_dag_ordering(self):
        """Complex DAG: 5 tasks with mixed dependencies produce correct layers."""
        tasks = [
            _make_task(name="ingest", outputs=["raw"]),
            _make_task(name="clean", inputs=["raw"], outputs=["cleaned"]),
            _make_task(name="enrich", inputs=["raw"], outputs=["enriched"]),
            _make_task(
                name="merge", inputs=["cleaned", "enriched"], outputs=["merged"]
            ),
            _make_task(name="report", inputs=["merged"], outputs=["final"]),
        ]
        layers = resolve_dag(tasks)
        assert len(layers) == 4
        assert [t.name for t in layers[0]] == ["ingest"]
        assert sorted(t.name for t in layers[1]) == ["clean", "enrich"]
        assert [t.name for t in layers[2]] == ["merge"]
        assert [t.name for t in layers[3]] == ["report"]


# ===========================================================================
# 4. Token Circuit Breaker
# ===========================================================================


class TestTokenCircuitBreaker:
    """Tests for the token budget enforcement in run_agent_loop()."""

    def test_default_max_tokens(self):
        """Default max tokens is 20M."""
        assert DEFAULT_MAX_TOKENS == 20_000_000

    def test_budget_exceeded_stops_agent(self):
        """Agent stops with success=False when token budget is exceeded."""
        call_count = 0

        async def mock_call_model(messages):
            nonlocal call_count
            call_count += 1
            return (
                {"role": "assistant", "content": "done"},
                {"prompt_tokens": 600_000, "completion_tokens": 500_000},
            )

        async def mock_tool_executor(name, args):
            return json.dumps({"result": "ok"})

        def mock_validation():
            return False, "not valid yet"  # Always fail to keep loop going

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                validation_fn=mock_validation,
                max_tokens=1_000_000,  # 1M budget
                max_iterations=100,
            )
        )
        assert not result.success
        assert "token limit" in result.final_message.lower()
        # With 1.1M tokens per call and 1M budget, should stop after 1 call
        assert call_count == 1

    def test_budget_not_exceeded_continues(self):
        """Agent continues normally when under budget."""
        call_count = 0

        async def mock_call_model(messages):
            nonlocal call_count
            call_count += 1
            # First call: use tools. Second call: stop.
            if call_count == 1:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ],
                    },
                    {"prompt_tokens": 100, "completion_tokens": 50},
                )
            return (
                {"role": "assistant", "content": "done"},
                {"prompt_tokens": 100, "completion_tokens": 50},
            )

        async def mock_tool_executor(name, args):
            return "ok"

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                max_tokens=1_000_000,
                max_iterations=10,
            )
        )
        assert result.success
        assert call_count == 2

    def test_budget_disabled_when_zero(self):
        """Setting max_tokens=0 disables the token budget check."""
        call_count = 0

        async def mock_call_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"tc{call_count}",
                                "function": {"name": "test", "arguments": "{}"},
                            }
                        ],
                    },
                    {"prompt_tokens": 10_000_000, "completion_tokens": 10_000_000},
                )
            return (
                {"role": "assistant", "content": "done"},
                {"prompt_tokens": 10_000_000, "completion_tokens": 10_000_000},
            )

        async def mock_tool_executor(name, args):
            return "ok"

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                max_tokens=0,  # disabled
                max_iterations=10,
            )
        )
        assert result.success  # Completes despite enormous token usage
        assert result.usage["prompt_tokens"] == 30_000_000

    def test_usage_accumulation(self):
        """Token usage is correctly accumulated across iterations."""
        call_count = 0

        async def mock_call_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"tc{call_count}",
                                "function": {"name": "t", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "prompt_tokens_details": {"cached_tokens": 200},
                        "completion_tokens_details": {"reasoning_tokens": 100},
                    },
                )
            return (
                {"role": "assistant", "content": "done"},
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "prompt_tokens_details": {"cached_tokens": 200},
                    "completion_tokens_details": {"reasoning_tokens": 100},
                },
            )

        async def mock_tool_executor(name, args):
            return "ok"

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                max_tokens=0,
                max_iterations=10,
            )
        )
        assert result.usage["prompt_tokens"] == 4000
        assert result.usage["completion_tokens"] == 2000
        assert result.usage["cache_read_tokens"] == 800
        assert result.usage["reasoning_tokens"] == 400

    def test_max_iterations_exceeded(self):
        """Agent stops with success=False when max iterations exceeded."""

        async def mock_call_model(messages):
            return (
                {"role": "assistant", "content": "thinking..."},
                {"prompt_tokens": 100, "completion_tokens": 50},
            )

        async def mock_tool_executor(name, args):
            return "ok"

        def mock_validation():
            return False, "still not valid"

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_call_model,
                tool_executor=mock_tool_executor,
                initial_messages=[{"role": "user", "content": "test"}],
                validation_fn=mock_validation,
                max_iterations=3,
                max_tokens=0,
            )
        )
        assert not result.success
        assert result.iterations == 3
        assert "max iterations" in result.final_message.lower()


# ===========================================================================
# 5. Query Timeout
# ===========================================================================


class TestQueryTimeout:
    """Tests for per-query timeout via conn.interrupt()."""

    def test_default_timeout(self):
        """Default timeout is 30s."""
        assert DEFAULT_QUERY_TIMEOUT_S == 30

    def test_fast_query_completes(self, conn):
        """A fast query completes normally within timeout."""
        result = execute_sql(conn, "SELECT 42 AS answer", query_timeout_s=5)
        assert result["success"]
        assert result["rows"] == [(42,)]

    def test_slow_query_interrupted(self, conn):
        """A slow query is interrupted by the timeout timer."""
        # Cross join on large ranges — reliably slow and interruptible
        result = execute_sql(
            conn,
            "SELECT COUNT(*) FROM range(100000000) a, range(100000) b",
            query_timeout_s=1,  # 1 second timeout
        )
        assert not result["success"]
        assert "timed out" in result["error"].lower()

    def test_timeout_zero_disables(self, conn):
        """query_timeout_s=0 disables the timeout."""
        result = execute_sql(conn, "SELECT 1", query_timeout_s=0)
        assert result["success"]

    def test_connection_usable_after_timeout(self, conn):
        """Connection remains usable after a timeout."""
        # First: timeout via cross join
        result1 = execute_sql(
            conn,
            "SELECT COUNT(*) FROM range(100000000) a, range(100000) b",
            query_timeout_s=1,
        )
        assert not result1["success"]

        # Second: normal query should still work
        result2 = execute_sql(conn, "SELECT 1 AS ok", query_timeout_s=5)
        assert result2["success"]
        assert result2["rows"] == [(1,)]

    def test_timeout_logs_to_trace(self, conn):
        """Timed out queries are logged in _trace."""
        execute_sql(
            conn,
            "SELECT COUNT(*) FROM range(100000000) a, range(100000) b",
            task_name="timeout_test",
            query_timeout_s=1,
        )
        rows = conn.execute(
            "SELECT success, error FROM _trace WHERE task = 'timeout_test'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] is False
        assert "timed out" in rows[0][1].lower()


# ===========================================================================
# 6. Ingestion
# ===========================================================================


class TestIngestion:
    """Tests for data ingestion: coerce_to_dataframe, ingest_table."""

    def test_coerce_dataframe(self):
        """DataFrame passes through unchanged."""
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = coerce_to_dataframe(df)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["a", "b"]
        assert len(result) == 2

    def test_coerce_list_of_dicts(self):
        """list[dict] is converted to DataFrame."""
        data = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        result = coerce_to_dataframe(data)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["x", "y"]
        assert len(result) == 2

    def test_coerce_dict_of_lists(self):
        """dict[str, list] is converted to DataFrame."""
        data = {"col1": [10, 20, 30], "col2": ["a", "b", "c"]}
        result = coerce_to_dataframe(data)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["col1", "col2"]
        assert len(result) == 3

    def test_coerce_unsupported_type(self):
        """Unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported"):
            coerce_to_dataframe("not a table")
        with pytest.raises(TypeError, match="Unsupported"):
            coerce_to_dataframe(42)

    def test_ingest_list_of_dicts(self, conn):
        """Ingest list[dict] creates table with _row_id."""
        data = [{"name": "alice", "age": 30}, {"name": "bob", "age": 25}]
        ingest_table(conn, data, "people")

        rows = conn.execute(
            "SELECT _row_id, name, age FROM people ORDER BY _row_id"
        ).fetchall()
        assert rows == [(1, "alice", 30), (2, "bob", 25)]

    def test_ingest_dataframe(self, conn):
        """Ingest DataFrame creates table with _row_id."""
        df = pl.DataFrame({"product": ["A", "B"], "price": [10.0, 20.0]})
        ingest_table(conn, df, "products")

        rows = conn.execute(
            "SELECT _row_id, product, price FROM products ORDER BY _row_id"
        ).fetchall()
        assert rows == [(1, "A", 10.0), (2, "B", 20.0)]

    def test_ingest_dict_of_lists(self, conn):
        """Ingest dict[str, list] creates table with _row_id."""
        data = {"city": ["NYC", "LA", "CHI"], "pop": [8, 4, 3]}
        ingest_table(conn, data, "cities")

        count = conn.execute("SELECT COUNT(*) FROM cities").fetchone()[0]
        assert count == 3

        # Check _row_id exists and is sequential
        ids = conn.execute("SELECT _row_id FROM cities ORDER BY _row_id").fetchall()
        assert ids == [(1,), (2,), (3,)]

    def test_row_id_is_integer(self, conn):
        """_row_id column is INTEGER type."""
        ingest_table(conn, [{"x": 1}], "t")
        cols = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 't' AND column_name = '_row_id'"
        ).fetchall()
        assert len(cols) == 1
        assert cols[0][1] == "INTEGER"

    def test_ingest_overwrites_existing(self, conn):
        """Ingesting to an existing table name drops and recreates it."""
        ingest_table(conn, [{"v": 1}], "t")
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1

        ingest_table(conn, [{"v": 10}, {"v": 20}], "t")
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 2

    def test_ingest_preserves_types(self, conn):
        """Ingested data preserves column types."""
        data = [
            {"int_col": 42, "float_col": 3.14, "str_col": "hello", "bool_col": True},
        ]
        ingest_table(conn, data, "typed")

        row = conn.execute(
            "SELECT int_col, float_col, str_col, bool_col FROM typed"
        ).fetchone()
        assert row[0] == 42
        assert abs(row[1] - 3.14) < 0.001
        assert row[2] == "hello"
        assert row[3] is True

    def test_ingest_empty_list(self, conn):
        """Ingesting empty list creates table with zero rows."""
        df = pl.DataFrame({"x": pl.Series([], dtype=pl.Int64)})
        ingest_table(conn, df, "empty_t")
        count = conn.execute("SELECT COUNT(*) FROM empty_t").fetchone()[0]
        assert count == 0


# ===========================================================================
# 7. Input Validation
# ===========================================================================


class TestInputValidation:
    """Tests for Workspace._validate_inputs() and _ingest_all() auto-checks."""

    def _make_workspace(self, **kwargs) -> Workspace:
        """Helper to create a Workspace with defaults."""
        defaults = {
            "db_path": ":memory:",
            "inputs": {},
            "tasks": [],
        }
        defaults.update(kwargs)
        return Workspace(**defaults)

    def test_input_columns_pass(self, conn):
        """Input column validation passes when all required columns exist."""
        ingest_table(conn, [{"name": "alice", "age": 30}], "people")
        ws = self._make_workspace(
            input_columns={"people": ["name", "age"]},
        )
        errors = ws._validate_inputs(conn)
        assert errors == []

    def test_input_columns_missing(self, conn):
        """Input column validation fails when required columns are missing."""
        ingest_table(conn, [{"name": "alice"}], "people")
        ws = self._make_workspace(
            input_columns={"people": ["name", "age", "email"]},
        )
        errors = ws._validate_inputs(conn)
        assert len(errors) == 1
        assert "age" in errors[0]
        assert "email" in errors[0]
        assert "name" in errors[0]  # actual columns listed

    def test_input_columns_table_not_found(self, conn):
        """Input column validation reports missing table."""
        ws = self._make_workspace(
            input_columns={"nonexistent": ["col1"]},
        )
        errors = ws._validate_inputs(conn)
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_input_columns_extra_columns_ok(self, conn):
        """Extra columns beyond required are fine."""
        ingest_table(conn, [{"a": 1, "b": 2, "c": 3}], "t")
        ws = self._make_workspace(
            input_columns={"t": ["a"]},
        )
        errors = ws._validate_inputs(conn)
        assert errors == []

    def test_input_columns_excludes_row_id(self, conn):
        """_row_id is excluded from the actual columns listed in error messages."""
        ingest_table(conn, [{"x": 1}], "t")
        ws = self._make_workspace(
            input_columns={"t": ["missing_col"]},
        )
        errors = ws._validate_inputs(conn)
        assert len(errors) == 1
        assert "_row_id" not in errors[0]

    def test_input_columns_short_circuits_before_sql(self, conn):
        """Column errors short-circuit before SQL validation runs."""
        ingest_table(conn, [{"x": 1}], "t")
        ws = self._make_workspace(
            input_columns={"t": ["missing"]},
            input_validate_sql={"t": ["SELECT 'should not run'"]},
        )
        errors = ws._validate_inputs(conn)
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_input_validate_sql_pass(self, conn):
        """SQL validation passes when queries return zero rows."""
        ingest_table(conn, [{"id": 1, "val": 10}, {"id": 2, "val": 20}], "data")
        ws = self._make_workspace(
            input_validate_sql={"data": ["SELECT id FROM data WHERE val < 0"]},
        )
        errors = ws._validate_inputs(conn)
        assert errors == []

    def test_input_validate_sql_fail(self, conn):
        """SQL validation fails when query returns rows."""
        ingest_table(conn, [{"id": 1, "val": -5}], "data")
        ws = self._make_workspace(
            input_validate_sql={
                "data": [
                    "SELECT 'negative value for id=' || CAST(id AS VARCHAR) "
                    "FROM data WHERE val < 0"
                ]
            },
        )
        errors = ws._validate_inputs(conn)
        assert len(errors) == 1
        assert "negative value for id=1" in errors[0]

    def test_input_validate_sql_error_handling(self, conn):
        """SQL validation catches query errors gracefully."""
        ws = self._make_workspace(
            input_validate_sql={"bad": ["SELECT * FROM nonexistent_table"]},
        )
        errors = ws._validate_inputs(conn)
        assert len(errors) == 1
        assert "error" in errors[0].lower()

    def test_input_validate_sql_multicolumn(self, conn):
        """Multi-column SQL results are formatted as col=val pairs."""
        ingest_table(conn, [{"id": 1, "status": "bad"}], "items")
        ws = self._make_workspace(
            input_validate_sql={
                "items": ["SELECT id, status FROM items WHERE status = 'bad'"]
            },
        )
        errors = ws._validate_inputs(conn)
        assert len(errors) == 1
        assert "id=1" in errors[0]
        assert "status=bad" in errors[0]

    def test_input_validate_sql_short_circuits(self, conn):
        """SQL validation stops at first failing query."""
        ingest_table(conn, [{"id": 1}], "t")
        ws = self._make_workspace(
            input_validate_sql={
                "t": [
                    "SELECT 'error1' WHERE 1=1",
                    "SELECT 'error2' WHERE 1=1",
                ]
            },
        )
        errors = ws._validate_inputs(conn)
        assert errors == ["error1"]

    def test_no_validation_returns_empty(self, conn):
        """No input_columns or input_validate_sql returns no errors."""
        ws = self._make_workspace()
        errors = ws._validate_inputs(conn)
        assert errors == []

    def test_multiple_tables_column_check(self, conn):
        """Column validation works across multiple tables."""
        ingest_table(conn, [{"a": 1, "b": 2}], "t1")
        ingest_table(conn, [{"x": 1}], "t2")
        ws = self._make_workspace(
            input_columns={
                "t1": ["a", "b"],
                "t2": ["x", "y"],  # y is missing
            },
        )
        errors = ws._validate_inputs(conn)
        assert len(errors) == 1
        assert "t2" in errors[0]
        assert "y" in errors[0]

    def test_ingest_callable_error_handling(self, conn):
        """_ingest_all catches callable errors with context."""

        def bad_loader():
            raise FileNotFoundError("data.csv not found")

        ws = self._make_workspace(
            inputs={"broken": bad_loader},
        )
        with pytest.raises(RuntimeError, match="Input 'broken' callable failed"):
            ws._ingest_all(conn)


# ===========================================================================
# 8. Prompt Resolution
# ===========================================================================


class TestPromptResolution:
    """Tests for prompt handling: prompts must be strings."""

    def test_string_passthrough(self, tmp_path):
        """Plain string prompt is accepted in spec loading."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": [{"x": 1}]}\n'
            'TASKS = [{"name": "a", "prompt": "do the thing", '
            '"inputs": ["t"], "outputs": ["out"]}]\n',
        )
        result = load_spec_from_module(module_path)
        assert result["tasks"][0].prompt == "do the thing"

    def test_non_string_prompt_raises(self, tmp_path):
        """Non-string prompt raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": [{"x": 1}]}\n'
            'TASKS = [{"name": "a", "prompt": 123, '
            '"inputs": ["t"], "outputs": ["out"]}]\n',
        )
        with pytest.raises(ValueError, match="prompt must be a string"):
            load_spec_from_module(module_path)

    def test_list_prompt_raises(self, tmp_path):
        """List prompt raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": [{"x": 1}]}\n'
            'TASKS = [{"name": "a", "prompt": ["first", "second"], '
            '"inputs": ["t"], "outputs": ["out"]}]\n',
        )
        with pytest.raises(ValueError, match="prompt must be a string"):
            load_spec_from_module(module_path)


# ===========================================================================
# 9. API Cache Control
# ===========================================================================


class TestAddCacheControl:
    """Tests for add_cache_control: message mutation for Anthropic caching."""

    def test_string_content_converted_to_list(self):
        """String content is converted to a list with cache_control."""
        from src.api import add_cache_control

        messages = [{"role": "user", "content": "hello"}]
        result = add_cache_control(messages)

        assert isinstance(result[0]["content"], list)
        block = result[0]["content"][0]
        assert block["type"] == "text"
        assert block["text"] == "hello"
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_list_content_gets_cache_on_last_text(self):
        """List content gets cache_control on the last text block."""
        from src.api import add_cache_control

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        result = add_cache_control(messages)

        # First text block should NOT have cache_control
        assert "cache_control" not in result[0]["content"][0]
        # Last text block should have cache_control
        assert result[0]["content"][1]["cache_control"] == {"type": "ephemeral"}

    def test_only_last_message_with_content_modified(self):
        """Only the last message with content gets cache_control."""
        from src.api import add_cache_control

        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user msg"},
        ]
        result = add_cache_control(messages)

        # First message should be untouched (still a string)
        assert isinstance(result[0]["content"], str)
        # Last message should be modified
        assert isinstance(result[1]["content"], list)

    def test_none_content_skipped(self):
        """Messages with None content are skipped."""
        from src.api import add_cache_control

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None},
        ]
        result = add_cache_control(messages)

        # The assistant message (None content) is skipped;
        # the user message should be modified
        assert isinstance(result[0]["content"], list)
        assert result[1]["content"] is None

    def test_does_not_mutate_original(self):
        """Original messages list is not mutated."""
        from src.api import add_cache_control

        original = [{"role": "user", "content": "hello"}]
        result = add_cache_control(original)

        # Original should still have string content
        assert isinstance(original[0]["content"], str)
        assert isinstance(result[0]["content"], list)

    def test_empty_messages(self):
        """Empty message list returns empty list."""
        from src.api import add_cache_control

        assert add_cache_control([]) == []

    def test_tool_result_message_with_content(self):
        """Tool result messages with content can get cache_control."""
        from src.api import add_cache_control

        messages = [
            {"role": "user", "content": "query"},
            {"role": "tool", "tool_call_id": "123", "content": "result data"},
        ]
        result = add_cache_control(messages)

        # The tool message is last with content, so it gets modified
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["cache_control"] == {"type": "ephemeral"}


class TestIsAnthropicModel:
    """Tests for _is_anthropic_model."""

    def test_anthropic_model(self):
        from src.api import _is_anthropic_model

        assert _is_anthropic_model("anthropic/claude-opus-4.5") is True

    def test_non_anthropic_model(self):
        from src.api import _is_anthropic_model

        assert _is_anthropic_model("openai/gpt-5.2") is False

    def test_case_insensitive(self):
        from src.api import _is_anthropic_model

        assert _is_anthropic_model("Anthropic/Claude-Opus-4.5") is True


# ===========================================================================
# 11. Spec Loading (module)
# ===========================================================================


class TestLoadSpec:
    """Tests for load_spec_from_module: end-to-end spec module loading."""

    def test_simple_spec(self, tmp_path):
        """Loads a minimal spec with INPUTS and TASKS."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"data": [{"x": 1}]}\n'
            'TASKS = [{"name": "t1", "prompt": "do it", '
            '"inputs": ["data"], "outputs": ["out"]}]\n',
        )

        result = load_spec_from_module(module_path)
        assert "data" in result["inputs"]
        assert len(result["tasks"]) == 1
        assert result["tasks"][0].name == "t1"
        assert result["tasks"][0].prompt == "do it"

    def test_rich_inputs_extracted(self, tmp_path):
        """Rich INPUTS with columns and validate_sql are parsed."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "INPUTS = {\n"
            '    "tbl": {\n'
            '        "data": [{"a": 1, "b": 2}],\n'
            '        "columns": ["a", "b"],\n'
            '        "validate_sql": ["SELECT 1 FROM tbl WHERE a IS NULL"],\n'
            "    }\n"
            "}\n"
            'TASKS = [{"name": "t", "prompt": "p", "inputs": ["tbl"], "outputs": ["o"]}]\n',
        )

        result = load_spec_from_module(module_path)
        assert result["input_columns"] == {"tbl": ["a", "b"]}
        assert result["input_validate_sql"] == {
            "tbl": ["SELECT 1 FROM tbl WHERE a IS NULL"]
        }
        # The actual data is the list, not the dict
        assert result["inputs"]["tbl"] == [{"a": 1, "b": 2}]

    def test_simple_input_no_validation(self, tmp_path):
        """Simple INPUTS (no dict with 'data') have no validation metadata."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": [{"x": 1}]}\n'
            'TASKS = [{"name": "t", "prompt": "p", "inputs": ["t"], "outputs": ["o"]}]\n',
        )

        result = load_spec_from_module(module_path)
        assert result["input_columns"] == {}
        assert result["input_validate_sql"] == {}

    def test_missing_inputs_raises(self, tmp_path):
        """Spec without INPUTS raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(tmp_path, "TASKS = []\n")

        with pytest.raises(ValueError, match="must define INPUTS"):
            load_spec_from_module(module_path)

    def test_missing_tasks_raises(self, tmp_path):
        """Spec without TASKS raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(tmp_path, 'INPUTS = {"t": []}\n')

        with pytest.raises(ValueError, match="must define TASKS"):
            load_spec_from_module(module_path)

    def test_exports_optional(self, tmp_path):
        """Spec without EXPORTS returns empty dict."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": []}\n'
            'TASKS = [{"name": "t", "prompt": "p", "inputs": ["t"], "outputs": ["o"]}]\n',
        )

        result = load_spec_from_module(module_path)
        assert result["exports"] == {}

    def test_exports_loaded(self, tmp_path):
        """Spec with EXPORTS includes them in the result."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": []}\n'
            'TASKS = [{"name": "t", "prompt": "p", "inputs": ["t"], "outputs": ["o"]}]\n'
            "def my_export(conn, path): pass\n"
            'EXPORTS = {"out.csv": my_export}\n',
        )

        result = load_spec_from_module(module_path)
        assert "out.csv" in result["exports"]
        assert callable(result["exports"]["out.csv"])

    def test_path_prompt_rejected(self, tmp_path):
        """Path prompts are rejected (prompts must be strings)."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "from pathlib import Path\n"
            'INPUTS = {"t": []}\n'
            'TASKS = [{"name": "t", "prompt": Path("recipe.sql"), '
            '"inputs": ["t"], "outputs": ["out"]}]\n',
        )

        with pytest.raises(ValueError, match="prompt must be a string"):
            load_spec_from_module(module_path)

    def test_list_prompt_rejected(self, tmp_path):
        """List prompts are rejected (prompts must be strings)."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": []}\n'
            'TASKS = [{"name": "t", "prompt": ["Do the thing."], '
            '"inputs": ["t"], "outputs": ["out"]}]\n',
        )

        with pytest.raises(ValueError, match="prompt must be a string"):
            load_spec_from_module(module_path)

    def test_task_objects_accepted(self, tmp_path):
        """Tasks can be Task objects directly (not just dicts)."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "from src.task import Task\n"
            'INPUTS = {"t": []}\n'
            'TASKS = [Task(name="t", prompt="p", inputs=["t"], outputs=["o"])]\n',
        )

        result = load_spec_from_module(module_path)
        assert result["tasks"][0].name == "t"

    def test_invalid_task_type_raises(self, tmp_path):
        """Non-dict, non-Task task raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": []}\nTASKS = ["not a valid task"]\n',
        )

        with pytest.raises(ValueError, match="must be a dict or Task"):
            load_spec_from_module(module_path)

    def test_callable_input_preserved(self, tmp_path):
        """Callable INPUTS values are preserved (not called at load time)."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'def load_data(): return [{"x": 1}]\n'
            'INPUTS = {"t": load_data}\n'
            'TASKS = [{"name": "t", "prompt": "p", "inputs": ["t"], "outputs": ["o"]}]\n',
        )

        result = load_spec_from_module(module_path)
        assert callable(result["inputs"]["t"])

    def test_existing_spec_files_load(self):
        """The existing test spec files load without error."""
        from src.spec import load_spec_from_module

        # diamond_dag.py
        result = load_spec_from_module("tests.diamond_dag")
        assert len(result["tasks"]) == 4
        assert result["input_columns"]["transactions"] == [
            "id",
            "date",
            "type",
            "product",
            "amount",
            "region",
        ]

        # validation_view_demo.py
        result = load_spec_from_module("tests.validation_view_demo")
        assert len(result["tasks"]) == 1
        assert "expenses" in result["inputs"]


# ===========================================================================
# 12. Agent Loop — Concurrent Tools & Error Handling
# ===========================================================================


class TestAgentLoopConcurrentTools:
    """Tests for run_agent_loop: concurrent tool execution."""

    def test_multiple_tools_executed_concurrently(self):
        """Multiple tool calls in one round are all executed."""
        call_count = 0

        async def mock_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return two tool calls
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {
                                "name": "tool_a",
                                "arguments": '{"x": 1}',
                            },
                        },
                        {
                            "id": "tc2",
                            "function": {
                                "name": "tool_b",
                                "arguments": '{"y": 2}',
                            },
                        },
                    ],
                }, {"prompt_tokens": 100, "completion_tokens": 50}
            else:
                # Second call: done
                return {
                    "role": "assistant",
                    "content": "done",
                }, {"prompt_tokens": 100, "completion_tokens": 50}

        tool_calls_received = []

        async def mock_executor(name, args):
            tool_calls_received.append((name, args))
            return json.dumps({"result": f"ok from {name}"})

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_model,
                tool_executor=mock_executor,
                initial_messages=[{"role": "user", "content": "go"}],
            )
        )

        assert result.success is True
        assert result.tool_calls_count == 2
        assert len(tool_calls_received) == 2
        names = {tc[0] for tc in tool_calls_received}
        assert names == {"tool_a", "tool_b"}

    def test_tool_results_added_to_messages(self):
        """Tool results are appended as tool messages with correct IDs."""
        call_count = 0

        async def mock_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_abc",
                            "function": {
                                "name": "run_sql",
                                "arguments": '{"query": "SELECT 1"}',
                            },
                        },
                    ],
                }, {"prompt_tokens": 10, "completion_tokens": 5}
            else:
                # Verify tool result was in messages
                tool_msgs = [m for m in messages if m.get("role") == "tool"]
                assert len(tool_msgs) == 1
                assert tool_msgs[0]["tool_call_id"] == "tc_abc"
                return {
                    "role": "assistant",
                    "content": "done",
                }, {"prompt_tokens": 10, "completion_tokens": 5}

        async def mock_executor(name, args):
            return '{"success": true}'

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_model,
                tool_executor=mock_executor,
                initial_messages=[{"role": "user", "content": "go"}],
            )
        )
        assert result.success is True


class TestAgentLoopJSONDecodeError:
    """Tests for agent_loop: malformed tool call arguments."""

    def test_malformed_json_args_fallback_to_empty(self):
        """Malformed JSON in tool arguments falls back to empty dict."""
        call_count = 0
        received_args = []

        async def mock_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {
                                "name": "run_sql",
                                "arguments": "NOT VALID JSON {{{",
                            },
                        },
                    ],
                }, {"prompt_tokens": 10, "completion_tokens": 5}
            else:
                return {
                    "role": "assistant",
                    "content": "done",
                }, {"prompt_tokens": 10, "completion_tokens": 5}

        async def mock_executor(name, args):
            received_args.append(args)
            return '{"ok": true}'

        result = asyncio.run(
            run_agent_loop(
                call_model=mock_model,
                tool_executor=mock_executor,
                initial_messages=[{"role": "user", "content": "go"}],
            )
        )

        assert result.success is True
        assert received_args[0] == {}


class TestAgentLoopOnIteration:
    """Tests for agent_loop: on_iteration callback."""

    def test_on_iteration_called_with_tool_results(self):
        """on_iteration receives assistant message and tool results."""
        call_count = 0
        callbacks = []

        async def mock_model(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {"name": "t", "arguments": "{}"},
                        },
                    ],
                }, {"prompt_tokens": 10, "completion_tokens": 5}
            else:
                return {
                    "role": "assistant",
                    "content": "done",
                }, {"prompt_tokens": 10, "completion_tokens": 5}

        async def mock_executor(name, args):
            return '{"ok": true}'

        def on_iter(iteration, assistant_msg, tool_results):
            callbacks.append((iteration, assistant_msg is not None, tool_results))

        asyncio.run(
            run_agent_loop(
                call_model=mock_model,
                tool_executor=mock_executor,
                initial_messages=[{"role": "user", "content": "go"}],
                on_iteration=on_iter,
            )
        )

        assert len(callbacks) == 2
        # First iteration: has tool results
        assert callbacks[0][0] == 1
        assert callbacks[0][1] is True  # assistant_msg not None
        assert callbacks[0][2] is not None  # tool_results present
        # Second iteration: no tool results (agent done)
        assert callbacks[1][0] == 2
        assert callbacks[1][2] is None


# ===========================================================================
# 13. Schema Info (get_schema_info_for_tables)
# ===========================================================================


class TestGetSchemaInfo:
    """Tests for get_schema_info_for_tables: schema formatting for LLM prompts."""

    def test_normal_table(self, conn):
        """Formats schema and sample data for a normal table."""
        from src.ingest import get_schema_info_for_tables

        ingest_table(
            conn, [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}], "people"
        )
        info = get_schema_info_for_tables(conn, ["people"])

        assert "Table: people" in info
        assert "name" in info
        assert "age" in info
        assert "Row count: 2" in info
        assert "_row_id" not in info

    def test_missing_table(self, conn):
        """Missing table shows (not found)."""
        from src.ingest import get_schema_info_for_tables

        info = get_schema_info_for_tables(conn, ["nonexistent"])
        assert "nonexistent" in info
        assert "(not found)" in info

    def test_empty_table(self, conn):
        """Empty table shows row count 0."""
        from src.ingest import get_schema_info_for_tables

        # Use dict-of-lists with empty lists to get a table with columns but no rows
        ingest_table(conn, {"val": []}, "empty")
        info = get_schema_info_for_tables(conn, ["empty"])
        assert "Row count: 0" in info

    def test_multiple_tables(self, conn):
        """Multiple tables all appear in output."""
        from src.ingest import get_schema_info_for_tables

        ingest_table(conn, [{"x": 1}], "t1")
        ingest_table(conn, [{"y": 2}], "t2")
        info = get_schema_info_for_tables(conn, ["t1", "t2"])

        assert "Table: t1" in info
        assert "Table: t2" in info

    def test_excludes_row_id_from_columns(self, conn):
        """_row_id column is excluded from the displayed columns."""
        from src.ingest import get_schema_info_for_tables

        ingest_table(conn, [{"val": 42}], "t")
        info = get_schema_info_for_tables(conn, ["t"])

        assert "val" in info
        assert "_row_id" not in info


# ===========================================================================
# 14. Persist Task Meta
# ===========================================================================


class TestPersistTaskMeta:
    """Tests for persist_task_meta: per-task metadata persistence."""

    def test_basic_write_and_read(self, conn):
        """Writes metadata and reads it back."""
        from src.agent import persist_task_meta

        persist_task_meta(conn, "task1", {"model": "gpt-5", "iterations": 3})

        row = conn.execute(
            "SELECT meta_json FROM _task_meta WHERE task = 'task1'"
        ).fetchone()
        assert row is not None
        meta = json.loads(row[0])
        assert meta["model"] == "gpt-5"
        assert meta["iterations"] == 3

    def test_overwrites_previous(self, conn):
        """Writing meta for the same task overwrites previous values."""
        from src.agent import persist_task_meta

        persist_task_meta(conn, "task1", {"v": 1})
        persist_task_meta(conn, "task1", {"v": 2, "extra": "new"})

        row = conn.execute(
            "SELECT meta_json FROM _task_meta WHERE task = 'task1'"
        ).fetchone()
        assert row is not None
        meta = json.loads(row[0])
        assert meta["v"] == 2
        assert meta["extra"] == "new"
        assert set(meta.keys()) == {"v", "extra"}

    def test_multiple_tasks_isolated(self, conn):
        """Different tasks have independent metadata."""
        from src.agent import persist_task_meta

        persist_task_meta(conn, "a", {"x": 1})
        persist_task_meta(conn, "b", {"x": 2})

        a_row = conn.execute(
            "SELECT meta_json FROM _task_meta WHERE task = 'a'"
        ).fetchone()
        b_row = conn.execute(
            "SELECT meta_json FROM _task_meta WHERE task = 'b'"
        ).fetchone()
        assert a_row is not None
        assert b_row is not None
        assert json.loads(a_row[0])["x"] == 1
        assert json.loads(b_row[0])["x"] == 2

    def test_json_serialization(self, conn):
        """Complex values are JSON-serialized."""
        from src.agent import persist_task_meta

        persist_task_meta(conn, "t", {"list_val": [1, 2, 3], "dict_val": {"a": "b"}})

        row = conn.execute(
            "SELECT meta_json FROM _task_meta WHERE task = 't'"
        ).fetchone()
        assert row is not None
        meta = json.loads(row[0])
        assert meta["dict_val"] == {"a": "b"}
        assert meta["list_val"] == [1, 2, 3]

    def test_creates_table_if_not_exists(self):
        """Works on a fresh connection without pre-existing table."""
        from src.agent import persist_task_meta

        fresh_conn = duckdb.connect(":memory:")
        persist_task_meta(fresh_conn, "t", {"k": "v"})

        rows = fresh_conn.execute("SELECT * FROM _task_meta").fetchall()
        assert len(rows) == 1
        fresh_conn.close()


# ===========================================================================
# 15. Workspace Exports
# ===========================================================================


class TestWorkspaceExports:
    """Tests for Workspace._run_exports: export error handling."""

    def test_successful_export(self, conn):
        """Successful export returns empty error dict."""
        called_with = {}

        def export_fn(c, path):
            called_with["conn"] = c
            called_with["path"] = path

        ws = Workspace(
            db_path=":memory:",
            inputs={},
            tasks=[],
            exports={"report.csv": export_fn},
        )
        errors = ws._run_exports(conn)

        assert errors == {}
        assert called_with["path"] == Path("report.csv")

    def test_export_error_captured(self, conn):
        """Exception in export function is captured in error dict."""

        def bad_export(c, path):
            raise RuntimeError("disk full")

        ws = Workspace(
            db_path=":memory:",
            inputs={},
            tasks=[],
            exports={"report.csv": bad_export},
        )
        errors = ws._run_exports(conn)

        assert "report.csv" in errors
        assert "disk full" in errors["report.csv"]

    def test_multiple_exports_partial_failure(self, conn):
        """One failing export doesn't prevent others from running."""

        results = []

        def good_export(c, path):
            results.append(str(path))

        def bad_export(c, path):
            raise ValueError("oops")

        ws = Workspace(
            db_path=":memory:",
            inputs={},
            tasks=[],
            exports={
                "good.csv": good_export,
                "bad.csv": bad_export,
            },
        )
        errors = ws._run_exports(conn)

        assert "good.csv" not in errors
        assert "bad.csv" in errors
        assert "good.csv" in results


class TestWorkspaceValidateConfig:
    """Tests for Workspace._validate_config: config validation wrapper."""

    def test_valid_config_passes(self):
        """Valid task graph passes without error."""
        ws = Workspace(
            db_path=":memory:",
            inputs={"data": []},
            tasks=[_make_task(name="t", inputs=["data"], outputs=["out"])],
        )
        ws._validate_config()  # Should not raise

    def test_invalid_config_raises(self):
        """Invalid task graph raises ValueError."""
        ws = Workspace(
            db_path=":memory:",
            inputs={},
            tasks=[_make_task(name="t", inputs=["missing"], outputs=["out"])],
        )
        with pytest.raises(ValueError, match="Task graph validation failed"):
            ws._validate_config()


# ===========================================================================
# JSON serialization for DuckDB types
# ===========================================================================


class TestJsonDefault:
    """Tests for _json_default: DuckDB-native type serialization."""

    def test_date(self):
        import datetime

        assert _json_default(datetime.date(2024, 1, 15)) == "2024-01-15"

    def test_datetime(self):
        import datetime

        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        assert _json_default(dt) == "2024-01-15T10:30:00"

    def test_time(self):
        import datetime

        assert _json_default(datetime.time(10, 30)) == "10:30:00"

    def test_timedelta(self):
        import datetime

        assert _json_default(datetime.timedelta(days=3, hours=4)) == "3 days, 4:00:00"

    def test_decimal(self):
        from decimal import Decimal

        assert _json_default(Decimal("123.456")) == 123.456

    def test_uuid(self):
        import uuid

        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert _json_default(u) == "12345678-1234-5678-1234-567812345678"

    def test_bytes(self):
        assert _json_default(b"\x48\x45\x4c\x4c\x4f") == "48454c4c4f"

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())

    def test_nested_in_json_dumps(self):
        """Verify _json_default works via json.dumps for nested structures."""
        import datetime
        from decimal import Decimal

        data = {
            "dates": [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)],
            "amount": Decimal("99.99"),
        }
        result = json.loads(json.dumps(data, default=_json_default))
        assert result["dates"] == ["2024-01-01", "2024-02-01"]
        assert result["amount"] == 99.99


# ===========================================================================
# Result size cap
# ===========================================================================


class TestResultSizeCap:
    """Tests for execute_sql result size limiting."""

    def test_small_result_passes(self, conn):
        """Results under the cap are returned normally."""
        result = execute_sql(conn, "SELECT 1 AS x", max_result_chars=1000)
        assert result["success"] is True
        assert result["rows"] == [(1,)]

    def test_large_result_rejected(self, conn):
        """Results exceeding the cap return an error."""
        conn.execute(
            "CREATE TABLE big AS SELECT i, repeat('x', 100) AS txt FROM range(500) t(i)"
        )
        result = execute_sql(conn, "SELECT * FROM big", max_result_chars=1000)
        assert result["success"] is False
        assert "too large" in result["error"].lower()
        assert "500 rows" in result["error"]

    def test_limit_zero_disables(self, conn):
        """max_result_chars=0 disables the check."""
        conn.execute(
            "CREATE TABLE big AS SELECT i, repeat('x', 100) AS txt FROM range(500) t(i)"
        )
        result = execute_sql(conn, "SELECT * FROM big", max_result_chars=0)
        assert result["success"] is True
        assert result["row_count"] == 500

    def test_create_view_not_affected(self, conn):
        """CREATE VIEW (no result rows) is never affected by size cap."""
        result = execute_sql(
            conn,
            "CREATE VIEW v AS SELECT 1",
            allowed_views={"v"},
            namespace="test",
            max_result_chars=10,  # Extremely small
        )
        assert result["success"] is True

    def test_default_cap_is_30k(self):
        """Verify the default constant."""
        assert MAX_RESULT_CHARS == 30_000


class TestWorkspaceMeta:
    """Tests for persist_workspace_meta / read_workspace_meta."""

    def test_roundtrip(self, conn):
        """Write and read workspace metadata."""
        tasks = [_make_task(name="t", prompt="test prompt", outputs=["out"])]
        persist_workspace_meta(
            conn,
            model="gpt-5",
            tasks=tasks,
            input_row_counts={"data": 100},
        )
        meta = read_workspace_meta(conn)
        assert meta["llm_model"] == "gpt-5"
        assert meta["meta_version"] == "2"
        assert "created_at_utc" in meta
        assert "task_prompts" in meta
        assert "run" in meta

        # Check prompts stored correctly
        prompts = json.loads(meta["task_prompts"])
        assert prompts["t"] == "test prompt"

        # Check input row counts
        counts = json.loads(meta["inputs_row_counts"])
        assert counts["data"] == 100

    def test_rerun_fields(self, conn):
        """Source db is stored when provided."""
        persist_workspace_meta(
            conn,
            model="m",
            tasks=[],
            source_db="/tmp/prev.db",
        )
        meta = read_workspace_meta(conn)
        run = json.loads(meta["run"])
        assert run["mode"] == "rerun"
        assert run["source_db"] == "/tmp/prev.db"

    def test_no_rerun_fields_when_fresh(self, conn):
        """Rerun fields are absent for fresh runs."""
        persist_workspace_meta(
            conn,
            model="m",
            tasks=[],
        )
        meta = read_workspace_meta(conn)
        run = json.loads(meta["run"])
        assert run["mode"] == "run"
        assert "source_db" not in run

    def test_overwrites_on_rewrite(self, conn):
        """Second call replaces all metadata."""
        persist_workspace_meta(
            conn,
            model="m1",
            tasks=[],
        )
        persist_workspace_meta(
            conn,
            model="m2",
            tasks=[],
        )
        meta = read_workspace_meta(conn)
        assert meta["llm_model"] == "m2"

    def test_read_missing_table(self):
        """read_workspace_meta returns empty dict if table doesn't exist."""
        c = duckdb.connect(":memory:")
        assert read_workspace_meta(c) == {}
        c.close()

    def test_full_prompts_stored(self, conn):
        """Full prompts are stored without truncation."""
        long_prompt = "x" * 1000
        tasks = [_make_task(name="t", prompt=long_prompt, outputs=["out"])]
        persist_workspace_meta(
            conn,
            model="m",
            tasks=tasks,
        )
        meta = read_workspace_meta(conn)
        prompts = json.loads(meta["task_prompts"])
        assert prompts["t"] == long_prompt
        assert len(prompts["t"]) == 1000


# Spec source embedding removed.
# (Fingerprint compatibility checks removed — rerun is intentionally simple.)
# ===========================================================================
# Get task views helper
# ===========================================================================


class TestGetTaskViews:
    """Tests for Workspace._get_task_views."""

    def test_finds_output_views(self, conn):
        """Finds views matching task outputs."""
        conn.execute("CREATE VIEW output AS SELECT 1 AS x")
        task = _make_task(name="match", outputs=["output"])
        views = Workspace._get_task_views(conn, task)
        assert ("output", 1) in views

    def test_finds_namespace_prefix_views(self, conn):
        """Finds views matching task namespace prefix."""
        conn.execute("CREATE VIEW match_step1 AS SELECT 1 AS x")
        conn.execute("CREATE VIEW match_step2 AS SELECT 1 AS x UNION ALL SELECT 2")
        task = _make_task(name="match", outputs=["output"])
        views = Workspace._get_task_views(conn, task)
        names = [v[0] for v in views]
        assert "match_step1" in names
        assert "match_step2" in names
        # match_step2 has 2 rows
        assert ("match_step2", 2) in views

    def test_ignores_other_task_views(self, conn):
        """Does not include views from other task namespaces."""
        conn.execute("CREATE VIEW match_step1 AS SELECT 1")
        conn.execute("CREATE VIEW other_step1 AS SELECT 1")
        task = _make_task(name="match", outputs=["output"])
        views = Workspace._get_task_views(conn, task)
        names = [v[0] for v in views]
        assert "other_step1" not in names

    def test_broken_view_returns_negative(self, conn):
        """Views that error on query get count=-1."""
        conn.execute("CREATE TABLE tmp AS SELECT 1 AS x")
        conn.execute("CREATE VIEW match_v AS SELECT * FROM tmp")
        conn.execute("DROP TABLE tmp")
        task = _make_task(name="match", outputs=[])
        views = Workspace._get_task_views(conn, task)
        assert ("match_v", -1) in views


# ===========================================================================
# Agent review prompt
# ===========================================================================


class TestAgentReviewPrompt:
    """Tests for _build_task_user_message with review context."""

    def test_basic_prompt_no_review(self):
        """Without review context, prompt has standard structure."""
        task = _make_task(name="match", inputs=["data"], outputs=["output"])
        msg = _build_task_user_message(task, "Table: data\n  Columns: x (INT)")
        assert "TASK: match" in msg
        assert "REQUIRED OUTPUTS: output" in msg
        assert "EXISTING VIEWS" not in msg
        assert "VALIDATION FAILURES" not in msg

    def test_review_with_existing_views(self):
        """Review context includes existing views."""
        task = _make_task(name="match", outputs=["output"])
        msg = _build_task_user_message(
            task,
            "Table: data",
            existing_views=[("output", 271), ("match_step1", 100)],
        )
        assert "EXISTING VIEWS FROM PREVIOUS RUN" in msg
        assert "output: 271 rows" in msg
        assert "match_step1: 100 rows" in msg
        assert "do not rebuild from scratch" in msg.lower()

    def test_review_with_validation_errors(self):
        """Review context includes validation errors when present."""
        task = _make_task(name="match", outputs=["output"])
        msg = _build_task_user_message(
            task,
            "Table: data",
            existing_views=[("output", 271)],
            validation_errors=["View 'output' missing column 'status'"],
        )
        assert "VALIDATION FAILURES" in msg
        assert "missing column 'status'" in msg
        assert "fix the validation errors" in msg.lower()

    def test_review_without_errors(self):
        """Review mode without errors shows views but no error section."""
        task = _make_task(name="match", outputs=["output"])
        msg = _build_task_user_message(
            task,
            "Table: data",
            existing_views=[("output", 100)],
            validation_errors=None,
        )
        assert "EXISTING VIEWS" in msg
        assert "VALIDATION FAILURES" not in msg


# ===========================================================================
# Workspace reruns (integration-style, no LLM)
# ===========================================================================


class TestRerun:
    """Tests for Workspace.rerun — rerun lifecycle without LLM."""

    def _create_source_db(self, tmp_path):
        """Create a source .db with views and metadata."""
        db_path = tmp_path / "source.db"
        conn = duckdb.connect(str(db_path))
        init_trace_table(conn)

        # Ingest data
        ingest_table(conn, [{"x": 1}, {"x": 2}], "data")

        # Create views as if an agent ran
        conn.execute("CREATE VIEW output AS SELECT x, x * 2 AS y FROM data")

        # Persist workspace metadata
        tasks = [
            _make_task(
                name="t",
                inputs=["data"],
                outputs=["output"],
                output_columns={"output": ["x", "y"]},
            )
        ]
        persist_workspace_meta(
            conn,
            model="test",
            tasks=tasks,
            input_row_counts={"data": 2},
        )

        conn.close()
        return db_path, tasks

    def test_rerun_reuses_source_views(self, tmp_path, monkeypatch):
        """Rerun copies db and can succeed without LLM changes."""
        source_path, tasks = self._create_source_db(tmp_path)
        output_path = tmp_path / "output.db"

        ws = Workspace(
            db_path=str(output_path),
            inputs={"data": [{"x": 1}, {"x": 2}]},  # same data
            tasks=tasks,
        )

        async def _dummy_run_task_agent(**kwargs):
            return AgentResult(
                success=True,
                final_message="dummy",
                iterations=1,
                messages=[],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                },
                tool_calls_count=0,
            )

        monkeypatch.setattr("src.workspace.run_task_agent", _dummy_run_task_agent)

        async def _run():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws.rerun(
                    source_db=source_path,
                    client=client,
                )

        result = asyncio.run(_run())
        assert result.success is True

        # Verify the output is correct
        conn = duckdb.connect(str(output_path), read_only=True)
        rows = conn.execute("SELECT * FROM output ORDER BY x").fetchall()
        assert rows == [(1, 2), (2, 4)]

        # Verify workspace metadata was updated
        meta = read_workspace_meta(conn)
        run = json.loads(meta["run"])
        assert run["source_db"] == str(source_path)
        conn.close()

    def test_missing_source_raises(self, tmp_path):
        """Non-existent source db raises ValueError."""
        ws = Workspace(
            db_path=str(tmp_path / "out.db"),
            inputs={"data": []},
            tasks=[_make_task(name="t", inputs=["data"], outputs=["out"])],
        )

        async def _run():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws.rerun(
                    source_db=tmp_path / "nonexistent.db",
                    client=client,
                )

        with pytest.raises(ValueError, match="not found"):
            asyncio.run(_run())

    def test_data_refresh_propagates(self, tmp_path, monkeypatch):
        """Re-ingested data propagates through views."""
        source_path, tasks = self._create_source_db(tmp_path)
        output_path = tmp_path / "output.db"

        # Different data: 3 rows instead of 2
        ws = Workspace(
            db_path=str(output_path),
            inputs={"data": [{"x": 10}, {"x": 20}, {"x": 30}]},
            tasks=tasks,
        )

        async def _dummy_run_task_agent(**kwargs):
            return AgentResult(
                success=True,
                final_message="dummy",
                iterations=1,
                messages=[],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                },
                tool_calls_count=0,
            )

        monkeypatch.setattr("src.workspace.run_task_agent", _dummy_run_task_agent)

        async def _run():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws.rerun(
                    source_db=source_path,
                    client=client,
                    reingest=True,
                )

        result = asyncio.run(_run())
        assert result.success is True

        # Verify the view reflects new data
        conn = duckdb.connect(str(output_path), read_only=True)
        rows = conn.execute("SELECT * FROM output ORDER BY x").fetchall()
        assert rows == [(10, 20), (20, 40), (30, 60)]
        assert len(rows) == 3  # Was 2 in source
        conn.close()

    def test_reuse_data_ignores_new_inputs(self, tmp_path, monkeypatch):
        """Default rerun reuses existing data from the source db."""
        source_path, tasks = self._create_source_db(tmp_path)
        output_path = tmp_path / "output.db"

        ws = Workspace(
            db_path=str(output_path),
            inputs={"data": [{"x": 999}]},  # different data — should be ignored
            tasks=tasks,
        )

        async def _dummy_run_task_agent(**kwargs):
            return AgentResult(
                success=True,
                final_message="dummy",
                iterations=1,
                messages=[],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                },
                tool_calls_count=0,
            )

        monkeypatch.setattr("src.workspace.run_task_agent", _dummy_run_task_agent)

        async def _run():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws.rerun(
                    source_db=source_path,
                    client=client,
                )

        result = asyncio.run(_run())
        assert result.success is True

        # Data should be from source (1,2), not from new inputs (999)
        conn = duckdb.connect(str(output_path), read_only=True)
        rows = conn.execute("SELECT * FROM output ORDER BY x").fetchall()
        assert rows == [(1, 2), (2, 4)]
        conn.close()


# Spec source embedding removed.
# ===========================================================================
# End-to-end rerun workflow
# ===========================================================================


class TestRerunE2E:
    """End-to-end tests for the rerun lifecycle.

    Simulates the full Jan-to-Feb workflow without LLM calls by manually
    creating views (as an agent would), then verifying that rerun correctly
    handles spec overrides, data refresh, prompt changes, and metadata chains.
    """

    # -- helpers --

    JAN_SPEC = """\
import json

def load_sales():
    return [
        {"id": 1, "month": "jan", "amount": 100, "region": "east"},
        {"id": 2, "month": "jan", "amount": 200, "region": "west"},
        {"id": 3, "month": "jan", "amount": 150, "region": "east"},
    ]

INPUTS = {
    "sales": {
        "data": load_sales,
        "columns": ["id", "month", "amount", "region"],
    },
    "targets": [
        {"region": "east", "target": 200},
        {"region": "west", "target": 250},
    ],
}

TASKS = [
    {
        "name": "summarize",
        "prompt": "Summarize sales by region.",
        "inputs": ["sales"],
        "outputs": ["summary"],
        "output_columns": {"summary": ["region", "total"]},
    },
    {
        "name": "compare",
        "prompt": "Compare summary to targets. Flag regions below target.",
        "inputs": ["summary", "targets"],
        "outputs": ["comparison"],
        "output_columns": {"comparison": ["region", "total", "target", "gap"]},
    },
]

def export_report(conn, path):
    import csv
    rows = conn.execute("SELECT * FROM comparison ORDER BY region").fetchall()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "total", "target", "gap"])
        w.writerows(rows)

EXPORTS = {"report.csv": export_report}
"""

    FEB_SPEC = """\
import json

def load_sales():
    return [
        {"id": 4, "month": "feb", "amount": 300, "region": "east"},
        {"id": 5, "month": "feb", "amount": 100, "region": "west"},
        {"id": 6, "month": "feb", "amount": 250, "region": "east"},
        {"id": 7, "month": "feb", "amount": 50,  "region": "west"},
    ]

INPUTS = {
    "sales": {
        "data": load_sales,
        "columns": ["id", "month", "amount", "region"],
    },
    "targets": [
        {"region": "east", "target": 200},
        {"region": "west", "target": 250},
    ],
}

TASKS = [
    {
        "name": "summarize",
        "prompt": "Summarize sales by region. Include row count per region.",
        "inputs": ["sales"],
        "outputs": ["summary"],
        "output_columns": {"summary": ["region", "total"]},
    },
    {
        "name": "compare",
        "prompt": "Compare summary to targets. Flag regions below target. Add pct_of_target column.",
        "inputs": ["summary", "targets"],
        "outputs": ["comparison"],
        "output_columns": {"comparison": ["region", "total", "target", "gap"]},
    },
]

def export_report(conn, path):
    import csv
    rows = conn.execute("SELECT * FROM comparison ORDER BY region").fetchall()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "total", "target", "gap"])
        w.writerows(rows)

EXPORTS = {"report.csv": export_report}
"""

    def _simulate_jan_run(self, tmp_path):
        """Create jan.db as if a full 'taskgraph run' completed.

        Writes the spec file, creates the database with ingested data,
        agent-created views, and full metadata — mimicking what a real
        run produces.
        """
        from src.spec import load_spec_from_module

        spec_module = _write_spec_module(tmp_path, self.JAN_SPEC)

        spec = load_spec_from_module(spec_module)
        db_path = tmp_path / "jan.db"
        conn = duckdb.connect(str(db_path))
        init_trace_table(conn)

        # Ingest inputs (as workspace.run would)
        for name, value in spec["inputs"].items():
            data = value() if callable(value) else value
            ingest_table(conn, data, name)

        # Simulate agent work: create views
        conn.execute("""
            CREATE VIEW summary AS
            SELECT region, SUM(amount) AS total
            FROM sales
            GROUP BY region
        """)
        conn.execute("""
            CREATE VIEW comparison AS
            SELECT
                s.region,
                s.total,
                t.target,
                s.total - t.target AS gap
            FROM summary s
            JOIN targets t ON s.region = t.region
        """)

        persist_workspace_meta(
            conn,
            model="test-model",
            tasks=spec["tasks"],
            input_row_counts={"sales": 3, "targets": 2},
            spec_module=spec_module,
        )

        conn.close()
        return db_path, spec_module

    # -- tests --

    def test_jan_db_is_self_contained(self, tmp_path):
        """jan.db stores prompts and spec module reference."""
        db_path, spec_module = self._simulate_jan_run(tmp_path)

        conn = duckdb.connect(str(db_path), read_only=True)
        meta = read_workspace_meta(conn)

        spec = json.loads(meta["spec"])
        assert spec["module"] == spec_module

        # Prompts stored per task
        prompts = json.loads(meta["task_prompts"])
        assert "summarize" in prompts
        assert "compare" in prompts
        assert "Summarize sales by region" in prompts["summarize"]

        # Data is present
        sales = conn.execute("SELECT COUNT(*) FROM sales").fetchone()
        assert sales[0] == 3  # type: ignore[index]

        # Views work
        rows = conn.execute("SELECT * FROM comparison ORDER BY region").fetchall()
        assert len(rows) == 2
        # east: 100+150=250, target=200, gap=50
        assert rows[0] == ("east", 250, 200, 50)
        # west: 200, target=250, gap=-50
        assert rows[1] == ("west", 200, 250, -50)

        conn.close()

    def test_jan_to_feb_with_spec_override(self, tmp_path, monkeypatch):
        """Full month-to-month workflow: jan.db + feb_spec -> feb.db.

        Feb spec has different data (4 rows vs 3) and tweaked prompts.
        Views auto-update via late-binding (DuckDB recalculates from new tables).
        """
        from src.spec import load_spec_from_module

        db_path, _ = self._simulate_jan_run(tmp_path)

        # Write and load Feb spec
        feb_module = _write_spec_module(tmp_path, self.FEB_SPEC)
        feb_spec = load_spec_from_module(feb_module)

        output_path = tmp_path / "feb.db"
        ws = Workspace(
            db_path=str(output_path),
            inputs=feb_spec["inputs"],
            tasks=feb_spec["tasks"],
            exports=feb_spec["exports"],
            input_columns=feb_spec["input_columns"],
            input_validate_sql=feb_spec["input_validate_sql"],
            spec_module=feb_module,
        )

        async def _dummy_run_task_agent(**kwargs):
            return AgentResult(
                success=True,
                final_message="dummy",
                iterations=1,
                messages=[],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                },
                tool_calls_count=0,
            )

        monkeypatch.setattr("src.workspace.run_task_agent", _dummy_run_task_agent)

        async def _run():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws.rerun(
                    source_db=db_path,
                    client=client,
                    reingest=True,
                )

        result = asyncio.run(_run())
        assert result.success is True

        # Verify Feb data propagated through the views
        conn = duckdb.connect(str(output_path), read_only=True)

        # Sales should be Feb data (4 rows)
        sales_count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()
        assert sales_count[0] == 4  # type: ignore[index]

        # Summary: east=300+250=550, west=100+50=150
        summary = dict(conn.execute("SELECT region, total FROM summary").fetchall())
        assert summary["east"] == 550
        assert summary["west"] == 150

        # Comparison: east 550 vs 200 = +350, west 150 vs 250 = -100
        rows = conn.execute("SELECT * FROM comparison ORDER BY region").fetchall()
        assert rows[0] == ("east", 550, 200, 350)
        assert rows[1] == ("west", 150, 250, -100)

        # Metadata should reflect Feb run
        meta = read_workspace_meta(conn)
        run = json.loads(meta["run"])
        assert run["source_db"] == str(db_path)
        # Feb prompts stored (not Jan's)
        prompts = json.loads(meta["task_prompts"])
        assert "Include row count" in prompts["summarize"]
        assert "pct_of_target" in prompts["compare"]
        spec_meta = json.loads(meta["spec"])
        assert "module" in spec_meta

        conn.close()

    def test_rerun_chain_preserves_metadata_lineage(self, tmp_path, monkeypatch):
        """Chained reruns: jan -> feb -> mar preserve source_db chain."""
        from src.spec import load_spec_from_module

        jan_path, _ = self._simulate_jan_run(tmp_path)

        # -- Feb rerun from Jan --
        feb_path = tmp_path / "feb.db"
        conn = duckdb.connect(str(jan_path), read_only=True)
        meta = read_workspace_meta(conn)
        conn.close()

        spec_meta = json.loads(meta["spec"])
        spec = load_spec_from_module(spec_meta["module"])

        ws = Workspace(
            db_path=str(feb_path),
            inputs=spec["inputs"],
            tasks=spec["tasks"],
            spec_module=spec_meta["module"],
        )

        async def _dummy_run_task_agent(**kwargs):
            return AgentResult(
                success=True,
                final_message="dummy",
                iterations=1,
                messages=[],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                },
                tool_calls_count=0,
            )

        monkeypatch.setattr("src.workspace.run_task_agent", _dummy_run_task_agent)

        async def _run_feb():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws.rerun(
                    source_db=jan_path,
                    client=client,
                )

        asyncio.run(_run_feb())

        # -- Mar rerun from Feb --
        mar_path = tmp_path / "mar.db"
        conn = duckdb.connect(str(feb_path), read_only=True)
        meta_feb = read_workspace_meta(conn)
        conn.close()

        spec_meta_feb = json.loads(meta_feb["spec"])
        spec_feb = load_spec_from_module(spec_meta_feb["module"])

        ws2 = Workspace(
            db_path=str(mar_path),
            inputs=spec_feb["inputs"],
            tasks=spec_feb["tasks"],
            spec_module=spec_meta_feb["module"],
        )

        async def _run_mar():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws2.rerun(
                    source_db=feb_path,
                    client=client,
                )

        asyncio.run(_run_mar())

        # Verify lineage
        conn = duckdb.connect(str(feb_path), read_only=True)
        feb_meta = read_workspace_meta(conn)
        feb_run = json.loads(feb_meta["run"])
        assert feb_run["source_db"] == str(jan_path)
        conn.close()

        conn = duckdb.connect(str(mar_path), read_only=True)
        mar_meta = read_workspace_meta(conn)
        mar_run = json.loads(mar_meta["run"])
        assert mar_run["source_db"] == str(feb_path)
        conn.close()

    def test_export_runs_after_rerun(self, tmp_path, monkeypatch):
        """Exports from the spec execute correctly after rerun."""
        from src.spec import load_spec_from_module

        db_path, _ = self._simulate_jan_run(tmp_path)

        feb_module = _write_spec_module(tmp_path, self.FEB_SPEC)
        feb_spec = load_spec_from_module(feb_module)

        output_path = tmp_path / "feb_export.db"
        ws = Workspace(
            db_path=str(output_path),
            inputs=feb_spec["inputs"],
            tasks=feb_spec["tasks"],
            exports=feb_spec["exports"],
            input_columns=feb_spec["input_columns"],
            input_validate_sql=feb_spec["input_validate_sql"],
        )

        async def _dummy_run_task_agent(**kwargs):
            return AgentResult(
                success=True,
                final_message="dummy",
                iterations=1,
                messages=[],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                },
                tool_calls_count=0,
            )

        monkeypatch.setattr("src.workspace.run_task_agent", _dummy_run_task_agent)

        async def _run():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws.rerun(
                    source_db=db_path,
                    client=client,
                    reingest=True,
                )

        result = asyncio.run(_run())
        assert result.success is True

        # Now run exports manually (as CLI rerun does)
        report_path = tmp_path / "report.csv"
        conn = duckdb.connect(str(output_path), read_only=True)
        feb_spec["exports"]["report.csv"](conn, report_path)
        conn.close()

        # Verify CSV content matches Feb data
        import csv

        with open(report_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        east = next(r for r in rows if r["region"] == "east")
        west = next(r for r in rows if r["region"] == "west")
        assert int(east["total"]) == 550
        assert int(east["gap"]) == 350
        assert int(west["total"]) == 150
        assert int(west["gap"]) == -100

    def test_prompt_evolution_across_reruns(self, tmp_path, monkeypatch):
        """Prompts can evolve between reruns while keeping same structure."""
        from src.spec import load_spec_from_module

        db_path, _ = self._simulate_jan_run(tmp_path)

        # Jan prompts
        conn = duckdb.connect(str(db_path), read_only=True)
        jan_meta = read_workspace_meta(conn)
        conn.close()
        jan_prompts = json.loads(jan_meta["task_prompts"])
        assert jan_prompts["summarize"] == "Summarize sales by region."

        # Feb spec has different prompts
        feb_module = _write_spec_module(tmp_path, self.FEB_SPEC)
        feb_spec = load_spec_from_module(feb_module)

        output_path = tmp_path / "feb_prompts.db"
        ws = Workspace(
            db_path=str(output_path),
            inputs=feb_spec["inputs"],
            tasks=feb_spec["tasks"],
        )

        async def _dummy_run_task_agent(**kwargs):
            return AgentResult(
                success=True,
                final_message="dummy",
                iterations=1,
                messages=[],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_tokens": 0,
                    "reasoning_tokens": 0,
                },
                tool_calls_count=0,
            )

        monkeypatch.setattr("src.workspace.run_task_agent", _dummy_run_task_agent)

        async def _run():
            from src.api import OpenRouterClient

            async with OpenRouterClient() as client:
                return await ws.rerun(
                    source_db=db_path,
                    client=client,
                    reingest=True,
                )

        asyncio.run(_run())

        # Feb db stores Feb's prompts, not Jan's
        conn = duckdb.connect(str(output_path), read_only=True)
        feb_meta = read_workspace_meta(conn)
        conn.close()
        feb_prompts = json.loads(feb_meta["task_prompts"])
        assert "Include row count" in feb_prompts["summarize"]
        assert "pct_of_target" in feb_prompts["compare"]
        # And they're different from Jan
        assert feb_prompts["summarize"] != jan_prompts["summarize"]


def test_default_output_db_path_is_stable():
    from datetime import datetime, timezone

    from scripts.cli import _default_output_db_path

    p = _default_output_db_path(
        "my_app.specs.main", now=datetime(2026, 2, 16, 12, 34, 56, tzinfo=timezone.utc)
    )
    assert str(p) == "runs/my_app-specs-main_20260216_123456.db"
