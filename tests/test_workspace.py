import asyncio
import json
from pathlib import Path

import duckdb
import pytest

from tests.conftest import _make_node
from src.agent import run_sql_node, log_trace
from src.workspace import Workspace, persist_workspace_meta, read_workspace_meta
from src.workspace import materialize_node_outputs


class TestSqlOnlyNodes:
    def test_sql_only_task_executes_and_validates(self, conn):
        node = _make_node(
            name="sql_task",
            sql="CREATE VIEW sql_task_out AS SELECT 1 AS x",
        )

        result = asyncio.run(run_sql_node(conn=conn, node=node))
        assert result.success is True

    def test_sql_only_task_disallows_select(self, conn):
        node = _make_node(
            name="sql_task",
            sql="SELECT 1",
        )
        result = asyncio.run(run_sql_node(conn=conn, node=node))
        assert result.success is False
        assert "only allow" in result.final_message.lower()

    def test_sql_only_task_requires_sql(self, conn):
        node = _make_node(name="sql_task", sql="")
        result = asyncio.run(run_sql_node(conn=conn, node=node))
        assert result.success is False

    def test_validation_view_is_enforced(self, conn):
        # Validation views are enforced when validate_sql declares them
        conn.execute(
            "CREATE VIEW mytask__validation AS "
            "SELECT 'fail' AS status, 'bad things' AS message"
        )
        node = _make_node(
            name="mytask",
            validate_sql=(
                "CREATE VIEW mytask__validation AS "
                "SELECT 'fail' AS status, 'bad things' AS message"
            ),
        )
        errors = node.validate_validation_views(conn)
        assert errors
        assert "bad things" in "\n".join(errors)

    def test_validation_order_columns_before_validation_view(self, conn):
        """Column check runs before validation view enforcement."""
        conn.execute("CREATE VIEW t_v AS SELECT 1 AS wrong_col")
        node = _make_node(
            output_columns={"t_v": ["expected_col"]},
        )
        errors = node.validate_outputs(conn)
        assert len(errors) == 1
        assert "expected_col" in errors[0]


class TestTwoPhaseValidation:
    def test_validate_sql_runs_after_transform(self, conn):
        node = _make_node(
            name="t",
            sql="CREATE VIEW t_out AS SELECT 1 AS x",
            validate_sql=(
                "CREATE VIEW t__validation AS SELECT 'fail' AS status, 'bad' AS message"
            ),
        )

        ws = Workspace(db_path=":memory:")

        result = asyncio.run(
            ws._execute_sql_node(
                conn=conn,
                node=node,
            )
        )

        assert result.success is False
        assert "bad" in result.final_message

        views = {
            row[0]
            for row in conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE internal = false"
            ).fetchall()
        }
        assert "t__validation" in views


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
            nodes=[
                _make_node(name="data", source=[], sql=""),
                _make_node(
                    name="t", depends_on=["data"], sql="CREATE VIEW t_out AS SELECT 1"
                ),
            ],
        )
        ws._validate_config()  # Should not raise

    def test_invalid_config_raises(self):
        """Invalid task graph raises ValueError."""
        ws = Workspace(
            db_path=":memory:",
            nodes=[_make_node(name="t", depends_on=["missing"])],
        )
        with pytest.raises(ValueError, match="Graph validation failed"):
            ws._validate_config()


class TestWorkspaceMeta:
    """Tests for persist_workspace_meta / read_workspace_meta."""

    def test_roundtrip(self, conn):
        """Write and read workspace metadata."""
        nodes = [_make_node(name="t", prompt="test context")]
        persist_workspace_meta(
            conn,
            model="gpt-5",
            nodes=nodes,
            source_row_counts={"data": 100},
        )
        meta = read_workspace_meta(conn)
        assert meta["llm_model"] == "gpt-5"
        assert meta["meta_version"] == "2"
        assert "created_at_utc" in meta
        assert "node_prompts" in meta
        assert "run" in meta

        # Check prompts stored correctly
        contexts = json.loads(meta["node_prompts"])
        assert contexts["t"] == "test context"

        # Check input row counts
        counts = json.loads(meta["inputs_row_counts"])
        assert counts["data"] == 100

    def test_run_mode_is_run(self, conn):
        """Run mode is always 'run'."""
        persist_workspace_meta(
            conn,
            model="m",
            nodes=[],
        )
        meta = read_workspace_meta(conn)
        run = json.loads(meta["run"])
        assert run["mode"] == "run"

    def test_overwrites_on_rewrite(self, conn):
        """Second call replaces all metadata."""
        persist_workspace_meta(
            conn,
            model="m1",
            nodes=[],
        )
        persist_workspace_meta(
            conn,
            model="m2",
            nodes=[],
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
        long_context = "x" * 1000
        nodes = [_make_node(name="t", prompt=long_context, sql="")]
        persist_workspace_meta(
            conn,
            model="m",
            nodes=nodes,
        )
        meta = read_workspace_meta(conn)
        contexts = json.loads(meta["node_prompts"])
        assert contexts["t"] == long_context
        assert len(contexts["t"]) == 1000


class TestMaterializeNodeOutputs:
    """Tests for materialize_node_outputs: view-to-table conversion."""

    def test_materializes_output_view_as_table(self, conn):
        """Output view is converted to a table with identical data."""
        conn.execute("CREATE VIEW t_out AS SELECT 1 AS x, 'hello' AS y")
        node = _make_node(name="t")

        n = materialize_node_outputs(conn, node)

        assert n == 1
        # View should be gone
        views = {
            r[0]
            for r in conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE internal = false"
            ).fetchall()
        }
        assert "t_out" not in views
        # Table should exist with same data
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM duckdb_tables() WHERE internal = false"
            ).fetchall()
        }
        assert "t_out" in tables
        row = conn.execute("SELECT x, y FROM t_out").fetchone()
        assert row == (1, "hello")

    def test_preserves_sql_in_view_definitions(self, conn):
        """Original view SQL is visible in _view_definitions (derived from _trace)."""
        sql = "CREATE VIEW t_out AS SELECT 42 AS val"
        conn.execute(sql)
        log_trace(conn, sql, success=True, node_name="t")
        node = _make_node(name="t")

        materialize_node_outputs(conn, node)

        rows = conn.execute(
            "SELECT node, view_name, sql FROM _view_definitions"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "t"
        assert rows[0][1] == "t_out"
        assert "42" in rows[0][2]

    def test_multiple_outputs(self, conn):
        """All declared outputs are materialized."""
        sql_a = "CREATE VIEW t_a AS SELECT 1 AS x"
        sql_b = "CREATE VIEW t_b AS SELECT 2 AS x"
        conn.execute(sql_a)
        conn.execute(sql_b)
        log_trace(conn, sql_a, success=True, node_name="t")
        log_trace(conn, sql_b, success=True, node_name="t")
        node = _make_node(name="t")

        n = materialize_node_outputs(conn, node)

        assert n == 2
        assert conn.execute("SELECT x FROM t_a").fetchone() == (1,)
        assert conn.execute("SELECT x FROM t_b").fetchone() == (2,)
        defs = conn.execute(
            "SELECT view_name FROM _view_definitions ORDER BY view_name"
        ).fetchall()
        assert [r[0] for r in defs] == ["t_a", "t_b"]

    def test_intermediate_views_materialized(self, conn):
        """All namespace-prefixed views are materialized (no distinction between intermediate and output)."""
        conn.execute("CREATE VIEW t_step1 AS SELECT 1 AS x")
        conn.execute("CREATE VIEW t_out AS SELECT x FROM t_step1")
        node = _make_node(name="t")

        n = materialize_node_outputs(conn, node)

        # Both views should be materialized
        assert n == 2
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM duckdb_tables() WHERE internal = false"
            ).fetchall()
        }
        assert "t_step1" in tables
        assert "t_out" in tables

    def test_materialized_data_matches_original(self, conn):
        """Multi-row view data is preserved exactly after materialization."""
        conn.execute(
            "CREATE VIEW t_out AS "
            "SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(id, name)"
        )
        node = _make_node(name="t")

        materialize_node_outputs(conn, node)

        rows = conn.execute("SELECT id, name FROM t_out ORDER BY id").fetchall()
        assert rows == [(1, "a"), (2, "b"), (3, "c")]

    def test_downstream_view_reads_materialized_table(self, conn):
        """A downstream view referencing a materialized output still works."""
        conn.execute("CREATE VIEW upstream_out AS SELECT 10 AS val")
        node = _make_node(name="upstream")
        materialize_node_outputs(conn, node)

        # Now create a downstream view that reads the (now-table) output
        conn.execute(
            "CREATE VIEW downstream_step AS SELECT val * 2 AS doubled FROM upstream_out"
        )
        row = conn.execute("SELECT doubled FROM downstream_step").fetchone()
        assert row == (20,)

    def test_validate_outputs_works_after_materialization(self, conn):
        """validate_outputs accepts materialized tables as valid outputs."""
        conn.execute("CREATE VIEW t_out AS SELECT 1 AS x, 2 AS y")
        node = _make_node(
            name="t",
            output_columns={"t_out": ["x", "y"]},
        )

        # Validate before materialization (view)
        assert node.validate_outputs(conn) == []

        # Materialize
        materialize_node_outputs(conn, node)

        # Validate after materialization (table)
        assert node.validate_outputs(conn) == []

    def test_idempotent_on_already_materialized(self, conn):
        """Calling materialize twice skips already-materialized outputs."""
        conn.execute("CREATE VIEW t_out AS SELECT 1 AS x")
        node = _make_node(name="t")

        n1 = materialize_node_outputs(conn, node)
        assert n1 == 1

        # Second call: view is gone, so it's skipped
        n2 = materialize_node_outputs(conn, node)
        assert n2 == 0

        # Data still intact
        assert conn.execute("SELECT x FROM t_out").fetchone() == (1,)

    def test_materializes_validation_views(self, conn):
        """Validation views are materialized alongside output views."""
        conn.execute("CREATE VIEW t_out AS SELECT 1 AS x")
        conn.execute(
            "CREATE VIEW t__validation AS "
            "SELECT 'pass' AS status, 'all good' AS message"
        )
        node = _make_node(
            name="t",
            validate_sql=(
                "CREATE VIEW t__validation AS "
                "SELECT 'pass' AS status, 'all good' AS message"
            ),
        )

        n = materialize_node_outputs(conn, node)

        assert n == 2  # output + validation view
        # Both should be tables now
        views = {
            r[0]
            for r in conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE internal = false"
            ).fetchall()
        }
        assert "t_out" not in views
        assert "t__validation" not in views
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM duckdb_tables() WHERE internal = false"
            ).fetchall()
        }
        assert "t_out" in tables
        assert "t__validation" in tables
        # Data intact
        row = conn.execute("SELECT status, message FROM t__validation").fetchone()
        assert row == ("pass", "all good")

    def test_materializes_multiple_validation_views(self, conn):
        """Multiple validation views (e.g. t__validation, t__validation_extra) are materialized."""
        conn.execute("CREATE VIEW t_out AS SELECT 1 AS x")
        conn.execute(
            "CREATE VIEW t__validation AS SELECT 'pass' AS status, 'ok' AS message"
        )
        conn.execute(
            "CREATE VIEW t__validation_extra AS "
            "SELECT 'warn' AS status, 'heads up' AS message"
        )
        node = _make_node(
            name="t",
            validate_sql=(
                "CREATE VIEW t__validation AS "
                "SELECT 'pass' AS status, 'ok' AS message; "
                "CREATE VIEW t__validation_extra AS "
                "SELECT 'warn' AS status, 'heads up' AS message"
            ),
        )

        n = materialize_node_outputs(conn, node)

        assert n == 3  # t_out + t__validation + t__validation_extra
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM duckdb_tables() WHERE internal = false"
            ).fetchall()
        }
        assert {"t_out", "t__validation", "t__validation_extra"} <= tables

    def test_validation_warnings_works_after_materialization(self, conn):
        """validation_warnings() reads from materialized validation tables."""
        conn.execute(
            "CREATE VIEW t__validation AS "
            "SELECT 'warn' AS status, 'watch out' AS message"
        )
        node = _make_node(
            name="t",
            validate_sql=(
                "CREATE VIEW t__validation AS "
                "SELECT 'warn' AS status, 'watch out' AS message"
            ),
        )

        # Materialize the validation view
        materialize_node_outputs(conn, node)

        # validation_warnings should still work (reads from table now)
        total, warnings = node.validation_warnings(conn)
        assert total == 1
        assert len(warnings) == 1
        assert "watch out" in warnings[0]

    def test_validation_view_sql_preserved(self, conn):
        """Validation view SQL is visible in _view_definitions (derived from _trace)."""
        sql = "CREATE VIEW t__validation AS SELECT 'pass' AS status, 'ok' AS message"
        conn.execute(sql)
        log_trace(conn, sql, success=True, node_name="t")
        node = _make_node(
            name="t",
            validate_sql=sql,
        )

        materialize_node_outputs(conn, node)

        rows = conn.execute(
            "SELECT view_name, sql FROM _view_definitions WHERE node = 't'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "t__validation"
        assert "pass" in rows[0][1]

    def test_view_definitions_excludes_dropped_views(self, conn):
        """_view_definitions omits views whose last trace action is DROP."""
        create_sql = "CREATE VIEW t_dropped_v AS SELECT 1 AS x"
        drop_sql = "DROP VIEW t_dropped_v"
        keep_sql = "CREATE VIEW t_kept_v AS SELECT 2 AS x"
        conn.execute(create_sql)
        log_trace(conn, create_sql, success=True, node_name="t")
        conn.execute(drop_sql)
        log_trace(conn, drop_sql, success=True, node_name="t")
        conn.execute(keep_sql)
        log_trace(conn, keep_sql, success=True, node_name="t")

        names = [
            r[0]
            for r in conn.execute(
                "SELECT view_name FROM _view_definitions ORDER BY view_name"
            ).fetchall()
        ]
        assert "t_dropped_v" not in names
        assert "t_kept_v" in names

    def test_leftover_tmp_table_cleaned_up(self, conn):
        """A leftover _materialize_tmp_ table from a crashed run doesn't block."""
        conn.execute("CREATE TABLE _materialize_tmp_t_out AS SELECT 999 AS stale")
        conn.execute("CREATE VIEW t_out AS SELECT 1 AS x")
        node = _make_node(name="t")

        n = materialize_node_outputs(conn, node)

        assert n == 1
        # Stale tmp table replaced, final table has correct data
        assert conn.execute("SELECT x FROM t_out").fetchone() == (1,)
        # No leftover tmp table
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM duckdb_tables() WHERE internal = false"
            ).fetchall()
        }
        assert "_materialize_tmp_t_out" not in tables
