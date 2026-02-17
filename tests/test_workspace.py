import asyncio
import json
from pathlib import Path

import duckdb
import pytest

from tests.conftest import _make_task
from src.agent import run_sql_only_task
from src.ingest import ingest_table
from src.workspace import Workspace, persist_workspace_meta, read_workspace_meta


class TestSqlOnlyTasks:
    def test_sql_only_task_executes_and_validates(self, conn):
        task = _make_task(
            name="sql_task",
            inputs=[],
            outputs=["out_view"],
            sql="CREATE VIEW out_view AS SELECT 1 AS x",
        )

        result = asyncio.run(run_sql_only_task(conn=conn, task=task))
        assert result.success is True

    def test_sql_only_task_disallows_select(self, conn):
        task = _make_task(
            name="sql_task",
            outputs=["out_view"],
            sql="SELECT 1",
        )
        result = asyncio.run(run_sql_only_task(conn=conn, task=task))
        assert result.success is False
        assert "only allow" in result.final_message.lower()

    def test_sql_only_task_requires_sql(self, conn):
        task = _make_task(name="sql_task", outputs=["sql_task__validation"], sql="")
        result = asyncio.run(run_sql_only_task(conn=conn, task=task))
        assert result.success is False

    def test_validation_view_is_enforced(self, conn):
        # Validation views are enforced when validate_sql declares them
        conn.execute(
            "CREATE VIEW mytask__validation AS "
            "SELECT 'fail' AS status, 'bad things' AS message"
        )
        task = _make_task(
            name="mytask",
            validate_sql=(
                "CREATE VIEW mytask__validation AS "
                "SELECT 'fail' AS status, 'bad things' AS message"
            ),
        )
        errors = task.validate_validation_views(conn)
        assert errors
        assert "bad things" in "\n".join(errors)

    def test_validation_order_columns_before_validation_view(self, conn):
        """Column check runs before validation view enforcement."""
        conn.execute("CREATE VIEW v AS SELECT 1 AS wrong_col")
        task = _make_task(
            outputs=["v"],
            output_columns={"v": ["expected_col"]},
        )
        errors = task.validate_transform(conn)
        assert len(errors) == 1
        assert "expected_col" in errors[0]


class TestTwoPhaseValidation:
    def test_validate_sql_runs_after_transform(self, conn):
        task = _make_task(
            name="t",
            outputs=["out"],
            sql="CREATE VIEW out AS SELECT 1 AS x",
            validate_sql=(
                "CREATE VIEW t__validation AS SELECT 'fail' AS status, 'bad' AS message"
            ),
        )

        ws = Workspace(db_path=":memory:", inputs={}, tasks=[])
        client = type("StubClient", (), {"reasoning_effort": None})()

        result = asyncio.run(
            ws._execute_task(
                conn=conn,
                task=task,
                client=client,
                model="test",
                max_iterations=1,
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


class TestWorkspaceMeta:
    """Tests for persist_workspace_meta / read_workspace_meta."""

    def test_roundtrip(self, conn):
        """Write and read workspace metadata."""
        tasks = [_make_task(name="t", prompt="test context", outputs=["out"])]
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
        contexts = json.loads(meta["task_prompts"])
        assert contexts["t"] == "test context"

        # Check input row counts
        counts = json.loads(meta["inputs_row_counts"])
        assert counts["data"] == 100

    def test_run_mode_is_run(self, conn):
        """Run mode is always 'run'."""
        persist_workspace_meta(
            conn,
            model="m",
            tasks=[],
        )
        meta = read_workspace_meta(conn)
        run = json.loads(meta["run"])
        assert run["mode"] == "run"

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
        long_context = "x" * 1000
        tasks = [_make_task(name="t", prompt=long_context, outputs=["out"])]
        persist_workspace_meta(
            conn,
            model="m",
            tasks=tasks,
        )
        meta = read_workspace_meta(conn)
        contexts = json.loads(meta["task_prompts"])
        assert contexts["t"] == long_context
        assert len(contexts["t"]) == 1000
