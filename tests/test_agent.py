import json

import duckdb
import pytest

from tests.conftest import _make_task
from src.agent import (
    build_transform_prompt,
    execute_sql,
    is_sql_allowed,
    _json_default,
    DEFAULT_QUERY_TIMEOUT_S,
    MAX_RESULT_CHARS,
)
from src.namespace import Namespace


def _ns(allowed: set[str], prefix: str) -> Namespace:
    """Shorthand for creating a Namespace in tests."""
    return Namespace(frozenset(allowed), prefix)


class TestNamespaceEnforcement:
    """Tests for is_sql_allowed() and execute_sql() namespace logic."""

    def test_select_always_allowed(self):
        """SELECT queries are always allowed regardless of namespace."""
        ok, err = is_sql_allowed("SELECT 1", namespace=_ns({"x"}, "t"))
        assert ok
        assert err == ""

    def test_select_from_any_table(self):
        """SELECT FROM any table is allowed."""
        ok, err = is_sql_allowed(
            "SELECT * FROM other_task_output", namespace=_ns({"x"}, "t")
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
            namespace=_ns({"output", "summary"}, "task1"),
        )
        assert ok

    def test_create_view_namespaced_prefix(self):
        """CREATE VIEW with task name prefix is allowed."""
        ok, err = is_sql_allowed(
            "CREATE VIEW task1_intermediate AS SELECT 1",
            namespace=_ns({"output"}, "task1"),
        )
        assert ok

    def test_create_view_blocked(self):
        """CREATE VIEW outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "CREATE VIEW other_output AS SELECT 1",
            namespace=_ns({"output"}, "task1"),
        )
        assert not ok
        assert "other_output" in err

    def test_drop_view_declared_output(self):
        """DROP VIEW for a declared output is allowed."""
        ok, err = is_sql_allowed(
            "DROP VIEW IF EXISTS output",
            namespace=_ns({"output"}, "task1"),
        )
        assert ok

    def test_drop_view_namespaced(self):
        """DROP VIEW with task name prefix is allowed."""
        ok, err = is_sql_allowed(
            "DROP VIEW task1_temp",
            namespace=_ns({"output"}, "task1"),
        )
        assert ok

    def test_drop_view_blocked(self):
        """DROP VIEW outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "DROP VIEW other_task_view",
            namespace=_ns({"output"}, "task1"),
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
        # When namespace is None, no check is done — CREATE VIEW is always allowed
        ok, err = is_sql_allowed("CREATE VIEW anything AS SELECT 1")
        assert ok

    def test_create_scalar_macro_namespaced(self):
        """CREATE MACRO with namespace prefix is allowed."""
        ok, err = is_sql_allowed(
            "CREATE MACRO task1_clean(s) AS lower(trim(s))",
            namespace=_ns({"output"}, "task1"),
        )
        assert ok

    def test_create_scalar_macro_blocked(self):
        """CREATE MACRO outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "CREATE MACRO other_clean(s) AS lower(trim(s))",
            namespace=_ns({"output"}, "task1"),
        )
        assert not ok
        assert "other_clean" in err

    def test_create_table_macro_namespaced(self):
        """CREATE MACRO ... AS TABLE with namespace prefix is allowed."""
        ok, err = is_sql_allowed(
            "CREATE MACRO task1_best(t) AS TABLE SELECT * FROM t",
            namespace=_ns({"output"}, "task1"),
        )
        assert ok

    def test_create_table_macro_blocked(self):
        """CREATE MACRO ... AS TABLE outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "CREATE MACRO other_best(t) AS TABLE SELECT * FROM t",
            namespace=_ns({"output"}, "task1"),
        )
        assert not ok
        assert "other_best" in err

    def test_drop_macro_namespaced(self):
        """DROP MACRO with namespace prefix is allowed."""
        ok, err = is_sql_allowed(
            "DROP MACRO task1_clean",
            namespace=_ns({"output"}, "task1"),
        )
        assert ok

    def test_drop_macro_blocked(self):
        """DROP MACRO outside namespace is blocked."""
        ok, err = is_sql_allowed(
            "DROP MACRO other_clean",
            namespace=_ns({"output"}, "task1"),
        )
        assert not ok
        assert "other_clean" in err

    def test_drop_macro_if_exists(self):
        """DROP MACRO IF EXISTS is allowed within namespace."""
        ok, err = is_sql_allowed(
            "DROP MACRO IF EXISTS task1_helper",
            namespace=_ns({"output"}, "task1"),
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
            namespace=_ns({"output"}, "task1"),
            query_timeout_s=0,
        )
        assert not result["success"]
        assert "forbidden" in result["error"]

    def test_execute_sql_namespace_allows(self, conn):
        """execute_sql() allows queries within namespace."""
        result = execute_sql(
            conn,
            "CREATE VIEW output AS SELECT 1 AS x",
            namespace=_ns({"output"}, "task1"),
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

    def test_execute_sql_logs_source(self, conn):
        """execute_sql() records the source column in _trace."""
        execute_sql(conn, "SELECT 1", task_name="t", query_timeout_s=0, source="agent")
        rows = conn.execute("SELECT source FROM _trace WHERE task = 't'").fetchall()
        assert rows == [("agent",)]

    def test_execute_sql_source_none_by_default(self, conn):
        """source defaults to NULL when not provided."""
        execute_sql(conn, "SELECT 1", task_name="t", query_timeout_s=0)
        rows = conn.execute("SELECT source FROM _trace WHERE task = 't'").fetchall()
        assert rows == [(None,)]

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
            namespace=_ns({"v"}, "test"),
            max_result_chars=10,  # Extremely small
        )
        assert result["success"] is True

    def test_default_cap_is_30k(self):
        """Verify the default constant."""
        assert MAX_RESULT_CHARS == 30_000


class TestTransformPrompt:
    """Tests for build_transform_prompt — the user message sent to the transform agent."""

    def test_basic_structure(self):
        """Transform prompt has TASK, PROMPT, INPUTS, REQUIRED OUTPUTS, ALLOWED VIEWS."""
        task = _make_task(
            name="match",
            inputs=["data"],
            outputs=["output"],
            prompt="Match records into an output view.",
        )
        msg = build_transform_prompt(task)
        assert "TASK: match" in msg
        assert "PROMPT:" in msg
        assert "Match records" in msg
        assert "INPUTS:" in msg
        assert "data" in msg
        assert "REQUIRED OUTPUTS:" in msg
        assert "- output" in msg
        assert "ALLOWED VIEWS: output or match_*" in msg

    def test_output_columns_shown(self):
        """Required columns are listed next to their output views."""
        task = _make_task(
            name="t",
            outputs=["result"],
            output_columns={"result": ["id", "score"]},
            prompt="Create result view",
        )
        msg = build_transform_prompt(task)
        assert "- result: id, score" in msg


class TestNamespace:
    """Tests for the Namespace class — factory methods, check_name, edge cases."""

    def test_for_task_allows_declared_outputs(self):
        """for_task() allows creating declared output views."""
        task = _make_task(name="match", outputs=["result", "summary"])
        ns = Namespace.for_task(task)
        assert ns.is_name_allowed("result")
        assert ns.is_name_allowed("summary")

    def test_for_task_allows_prefixed_intermediates(self):
        """for_task() allows creating {task_name}_* intermediates."""
        task = _make_task(name="match", outputs=["result"])
        ns = Namespace.for_task(task)
        assert ns.is_name_allowed("match_temp")
        assert ns.is_name_allowed("match_staging")

    def test_for_task_blocks_validation_views(self):
        """for_task() blocks validation views via forbidden check."""
        task = _make_task(name="match", outputs=["result"])
        ns = Namespace.for_task(task)
        ok, err = ns.check_name("match__validation", "view", "create")
        assert not ok
        assert "Validation views" in err

    def test_for_task_blocks_validation_prefixed(self):
        """for_task() blocks validation-prefixed views."""
        task = _make_task(name="match", outputs=["result"])
        ns = Namespace.for_task(task)
        ok, err = ns.check_name("match__validation_extra", "view", "create")
        assert not ok
        assert "Validation views" in err

    def test_for_task_blocks_unrelated_names(self):
        """for_task() blocks names outside outputs and prefix."""
        task = _make_task(name="match", outputs=["result"])
        ns = Namespace.for_task(task)
        ok, err = ns.check_name("other_view", "view", "create")
        assert not ok
        assert "other_view" in err

    def test_for_validation_allows_base_name(self):
        """for_validation() allows {task}__validation."""
        task = _make_task(name="match", outputs=["result"])
        ns = Namespace.for_validation(task)
        assert ns.is_name_allowed("match__validation")

    def test_for_validation_allows_prefixed(self):
        """for_validation() allows {task}__validation_* names."""
        task = _make_task(name="match", outputs=["result"])
        ns = Namespace.for_validation(task)
        assert ns.is_name_allowed("match__validation_extra")
        assert ns.is_name_allowed("match__validation_nulls")

    def test_for_validation_blocks_task_outputs(self):
        """for_validation() blocks task outputs."""
        task = _make_task(name="match", outputs=["result"])
        ns = Namespace.for_validation(task)
        assert not ns.is_name_allowed("result")

    def test_for_input_allows_base_name(self):
        """for_input() allows {input}__validation."""
        ns = Namespace.for_input("invoices")
        assert ns.is_name_allowed("invoices__validation")

    def test_for_input_allows_prefixed(self):
        """for_input() allows {input}__validation_* names."""
        ns = Namespace.for_input("invoices")
        assert ns.is_name_allowed("invoices__validation_amounts")

    def test_for_input_blocks_unrelated(self):
        """for_input() blocks names outside its validation prefix."""
        ns = Namespace.for_input("invoices")
        assert not ns.is_name_allowed("invoices")
        assert not ns.is_name_allowed("other__validation")

    def test_check_name_none_blocked(self):
        """check_name(None) is always blocked (tightened GAP 7)."""
        ns = Namespace.for_input("invoices")
        ok, err = ns.check_name(None, "view", "create")
        assert not ok
        assert "Could not extract" in err

    def test_check_name_returns_ok_for_valid(self):
        """check_name returns (True, '') for valid names."""
        ns = Namespace.for_input("invoices")
        ok, err = ns.check_name("invoices__validation", "view", "create")
        assert ok
        assert err == ""

    def test_format_allowed(self):
        """format_allowed() includes both explicit names and prefix."""
        ns = Namespace(frozenset({"a", "b"}), "task1")
        formatted = ns.format_allowed()
        assert "a" in formatted
        assert "b" in formatted
        assert "task1_*" in formatted

    def test_repr(self):
        """__repr__ includes allowed_names and prefix."""
        ns = Namespace(frozenset({"out"}), "t")
        r = repr(ns)
        assert "Namespace" in r
        assert "out" in r
        assert "t" in r

    def test_eq(self):
        """Two Namespaces with same names/prefix/msg are equal."""
        a = Namespace(frozenset({"x"}), "p")
        b = Namespace(frozenset({"x"}), "p")
        assert a == b

    def test_neq_different_names(self):
        """Namespaces with different allowed_names are not equal."""
        a = Namespace(frozenset({"x"}), "p")
        b = Namespace(frozenset({"y"}), "p")
        assert a != b

    def test_neq_different_type(self):
        """Namespace compared to non-Namespace returns NotImplemented."""
        ns = Namespace(frozenset({"x"}), "p")
        assert ns != "not a namespace"
