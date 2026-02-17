import duckdb
from click.testing import CliRunner

from tests.conftest import _write_spec_module


class TestSnapshotViews:
    """Test snapshot_views captures view SQL, columns, and row counts."""

    def test_snapshot_empty_db(self, conn):
        from src.diff import snapshot_views

        snap = snapshot_views(conn)
        assert snap == {}

    def test_snapshot_captures_view_sql(self, conn):
        from src.diff import snapshot_views

        conn.execute("CREATE TABLE t (x INT, y VARCHAR)")
        conn.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
        conn.execute("CREATE VIEW v AS SELECT x, y FROM t WHERE x > 0")

        snap = snapshot_views(conn)
        assert "v" in snap
        assert "SELECT" in snap["v"].sql
        assert snap["v"].row_count == 2
        col_names = [c[0] for c in snap["v"].columns]
        assert "x" in col_names
        assert "y" in col_names

    def test_snapshot_multiple_views(self, conn):
        from src.diff import snapshot_views

        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (1), (2), (3)")
        conn.execute("CREATE VIEW v1 AS SELECT x FROM t")
        conn.execute("CREATE VIEW v2 AS SELECT x * 2 AS x2 FROM t")

        snap = snapshot_views(conn)
        assert len(snap) == 2
        assert snap["v1"].row_count == 3
        assert snap["v2"].row_count == 3


class TestDiffSnapshots:
    """Test diff_snapshots detects created, dropped, and modified views."""

    def test_created_view(self):
        from src.diff import ViewSnapshot, diff_snapshots

        before: dict[str, ViewSnapshot] = {}
        after = {
            "v": ViewSnapshot(
                name="v",
                sql="CREATE VIEW v AS SELECT 1 AS x;",
                columns=[("x", "INTEGER")],
                row_count=1,
            )
        }
        changes = diff_snapshots(before, after)
        assert len(changes) == 1
        assert changes[0].kind == "created"
        assert changes[0].view_name == "v"
        assert changes[0].sql_before is None
        assert changes[0].sql_after is not None

    def test_dropped_view(self):
        from src.diff import ViewSnapshot, diff_snapshots

        before = {
            "v": ViewSnapshot(
                name="v",
                sql="CREATE VIEW v AS SELECT 1 AS x;",
                columns=[("x", "INTEGER")],
                row_count=1,
            )
        }
        after: dict[str, ViewSnapshot] = {}
        changes = diff_snapshots(before, after)
        assert len(changes) == 1
        assert changes[0].kind == "dropped"
        assert changes[0].sql_before is not None
        assert changes[0].sql_after is None

    def test_modified_view(self):
        from src.diff import ViewSnapshot, diff_snapshots

        before = {
            "v": ViewSnapshot(
                name="v",
                sql="CREATE VIEW v AS SELECT x FROM t;",
                columns=[("x", "INTEGER")],
                row_count=3,
            )
        }
        after = {
            "v": ViewSnapshot(
                name="v",
                sql="CREATE VIEW v AS SELECT x, x * 2 AS x2 FROM t;",
                columns=[("x", "INTEGER"), ("x2", "INTEGER")],
                row_count=3,
            )
        }
        changes = diff_snapshots(before, after)
        assert len(changes) == 1
        assert changes[0].kind == "modified"
        assert changes[0].sql_before != changes[0].sql_after

    def test_unchanged_view_omitted(self):
        from src.diff import ViewSnapshot, diff_snapshots

        snap = ViewSnapshot(
            name="v",
            sql="CREATE VIEW v AS SELECT 1 AS x;",
            columns=[("x", "INTEGER")],
            row_count=1,
        )
        changes = diff_snapshots({"v": snap}, {"v": snap})
        assert len(changes) == 0

    def test_whitespace_only_change_ignored(self):
        from src.diff import ViewSnapshot, diff_snapshots

        before = {
            "v": ViewSnapshot(
                name="v",
                sql="CREATE VIEW v AS SELECT  x  FROM t;",
                columns=[("x", "INTEGER")],
                row_count=1,
            )
        }
        after = {
            "v": ViewSnapshot(
                name="v",
                sql="CREATE VIEW v AS SELECT x FROM t;",
                columns=[("x", "INTEGER")],
                row_count=1,
            )
        }
        changes = diff_snapshots(before, after)
        assert len(changes) == 0

    def test_mixed_changes(self):
        from src.diff import ViewSnapshot, diff_snapshots

        before = {
            "keep": ViewSnapshot("keep", "SELECT 1;", [("x", "INT")], 1),
            "drop_me": ViewSnapshot("drop_me", "SELECT 2;", [("y", "INT")], 1),
            "modify_me": ViewSnapshot("modify_me", "SELECT 3;", [("z", "INT")], 1),
        }
        after = {
            "keep": ViewSnapshot("keep", "SELECT 1;", [("x", "INT")], 1),
            "modify_me": ViewSnapshot("modify_me", "SELECT 33;", [("z", "INT")], 1),
            "new_one": ViewSnapshot("new_one", "SELECT 4;", [("w", "INT")], 1),
        }
        changes = diff_snapshots(before, after)
        kinds = {c.view_name: c.kind for c in changes}
        assert kinds == {
            "drop_me": "dropped",
            "modify_me": "modified",
            "new_one": "created",
        }


class TestFormatChanges:
    """Test terminal output formatting of changes."""

    def test_format_created_view(self):
        from src.diff import ViewChange, format_changes

        changes = [
            ViewChange(
                view_name="output",
                kind="created",
                sql_before=None,
                sql_after="CREATE VIEW output AS SELECT 1;",
                cols_before=None,
                cols_after=[("x", "INTEGER"), ("y", "VARCHAR")],
                rows_before=None,
                rows_after=5,
            )
        ]
        result = format_changes("prep", changes)
        assert "prep:" in result
        assert "+ output" in result
        assert "2 cols" in result
        assert "5 rows" in result

    def test_format_dropped_view(self):
        from src.diff import ViewChange, format_changes

        changes = [
            ViewChange(
                view_name="old_view",
                kind="dropped",
                sql_before="CREATE VIEW old_view AS SELECT 1;",
                sql_after=None,
                cols_before=[("x", "INTEGER")],
                cols_after=None,
                rows_before=3,
                rows_after=None,
            )
        ]
        result = format_changes("cleanup", changes)
        assert "- old_view" in result

    def test_format_modified_view_with_col_change(self):
        from src.diff import ViewChange, format_changes

        changes = [
            ViewChange(
                view_name="report",
                kind="modified",
                sql_before="CREATE VIEW report AS SELECT x FROM t;",
                sql_after="CREATE VIEW report AS SELECT x, y FROM t;",
                cols_before=[("x", "INTEGER")],
                cols_after=[("x", "INTEGER"), ("y", "VARCHAR")],
                rows_before=10,
                rows_after=12,
            )
        ]
        result = format_changes("analyze", changes)
        assert "~ report" in result
        assert "+y" in result
        assert "10 rows -> 12 rows" in result

    def test_format_empty_changes(self):
        from src.diff import format_changes

        assert format_changes("task", []) == ""


class TestPersistChanges:
    """Test persisting changes to the _changes table."""

    def test_persist_and_read_back(self, conn):
        from src.diff import ViewChange, persist_changes

        changes = [
            ViewChange(
                view_name="v1",
                kind="created",
                sql_before=None,
                sql_after="CREATE VIEW v1 AS SELECT 1;",
                cols_before=None,
                cols_after=[("x", "INTEGER")],
                rows_before=None,
                rows_after=1,
            ),
            ViewChange(
                view_name="v2",
                kind="modified",
                sql_before="CREATE VIEW v2 AS SELECT 1;",
                sql_after="CREATE VIEW v2 AS SELECT 2;",
                cols_before=[("x", "INTEGER")],
                cols_after=[("x", "INTEGER")],
                rows_before=1,
                rows_after=1,
            ),
        ]
        persist_changes(conn, "my_task", changes)

        rows = conn.execute(
            "SELECT task, view_name, kind FROM _changes ORDER BY view_name"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == ("my_task", "v1", "created")
        assert rows[1] == ("my_task", "v2", "modified")

    def test_persist_empty_changes(self, conn):
        from src.diff import persist_changes

        persist_changes(conn, "my_task", [])
        # Table should be created but empty
        rows = conn.execute("SELECT COUNT(*) FROM _changes").fetchone()
        assert rows[0] == 0

    def test_persist_multiple_tasks(self, conn):
        from src.diff import ViewChange, persist_changes

        c1 = ViewChange("v1", "created", None, "SQL1", None, [("x", "INT")], None, 1)
        c2 = ViewChange("v2", "created", None, "SQL2", None, [("y", "INT")], None, 2)
        persist_changes(conn, "task_a", [c1])
        persist_changes(conn, "task_b", [c2])

        rows = conn.execute(
            "SELECT task, view_name FROM _changes ORDER BY task"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == ("task_a", "v1")
        assert rows[1] == ("task_b", "v2")


class TestSnapshotDiffIntegration:
    """Integration test: snapshot -> modify -> snapshot -> diff."""

    def test_end_to_end(self, conn):
        from src.diff import snapshot_views, diff_snapshots

        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (1), (2), (3)")
        conn.execute("CREATE VIEW existing AS SELECT x FROM t")

        before = snapshot_views(conn)
        assert "existing" in before

        # Create a new view and modify an existing one
        conn.execute("CREATE OR REPLACE VIEW existing AS SELECT x, x * 2 AS x2 FROM t")
        conn.execute("CREATE VIEW new_view AS SELECT x FROM t WHERE x > 1")

        after = snapshot_views(conn)
        changes = diff_snapshots(before, after)

        kinds = {c.view_name: c.kind for c in changes}
        assert kinds == {"existing": "modified", "new_view": "created"}

        # Verify the modified view has the right SQL diff content
        existing_change = [c for c in changes if c.view_name == "existing"][0]
        assert "x2" not in (existing_change.sql_before or "")
        assert "x2" in (existing_change.sql_after or "")


class TestWorkspaceChangesIntegration:
    """Test that workspace.run() persists changes to _changes table."""

    def test_changes_persisted_after_run(self, tmp_path):
        """A SQL-only workspace run should persist view changes."""
        from scripts.cli import main

        spec_source = """\
INPUTS = {"t": [{"x": 1}, {"x": 2}]}

TASKS = [
    {
        "name": "double",
        "sql": "CREATE VIEW output AS SELECT x, x * 2 AS x2 FROM t",
        "inputs": ["t"],
        "outputs": ["output"],
    }
]
"""
        spec_module = _write_spec_module(tmp_path, spec_source)
        out_db = tmp_path / "out.db"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "--spec", spec_module, "-o", str(out_db), "-q"],
            env={"OPENROUTER_API_KEY": ""},
        )
        assert result.exit_code == 0, result.output

        conn = duckdb.connect(str(out_db), read_only=True)
        try:
            rows = conn.execute("SELECT task, view_name, kind FROM _changes").fetchall()
            assert len(rows) == 1
            assert rows[0] == ("double", "output", "created")
        finally:
            conn.close()

    def test_changes_report_in_output(self, tmp_path):
        """Non-quiet run should show change report with + for created views."""
        from scripts.cli import main

        spec_source = """\
INPUTS = {"t": [{"x": 1}, {"x": 2}]}

TASKS = [
    {
        "name": "double",
        "sql": "CREATE VIEW output AS SELECT x, x * 2 AS x2 FROM t",
        "inputs": ["t"],
        "outputs": ["output"],
    }
]
"""
        spec_module = _write_spec_module(tmp_path, spec_source)
        out_db = tmp_path / "out.db"

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "--spec", spec_module, "-o", str(out_db)],
            env={"OPENROUTER_API_KEY": ""},
        )
        assert result.exit_code == 0, result.output
        assert "+ output" in result.output
