import duckdb
import pytest
from click.testing import CliRunner

from tests.conftest import _write_spec_module


def test_cli_run_sql_does_not_require_openrouter_api_key(tmp_path, monkeypatch):
    """SQL-only specs should run without OPENROUTER_API_KEY.

    This is a regression test for the CLI requiring the key even when no
    node would invoke the LLM.
    """
    from scripts.cli import main

    spec_source = """\
NODES = [
    {
        "name": "t",
        "sql": "CREATE OR REPLACE VIEW t_v AS SELECT 1 AS x",
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


class TestLoadSpec:
    """Tests for load_spec_from_module: end-to-end spec module loading."""

    def test_simple_spec(self, tmp_path):
        """Loads a minimal spec with NODES."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "NODES = [\n"
            '    {"name": "data", "source": [{"x": 1}]},\n'
            '    {"name": "t1", "depends_on": ["data"], '
            '"sql": "CREATE OR REPLACE VIEW t1_out AS SELECT 1 AS x"},\n'
            "]\n",
        )

        nodes, exports = load_spec_from_module(module_path)
        source_nodes = [n for n in nodes if n.is_source()]
        transform_nodes = [n for n in nodes if not n.is_source()]
        assert len(source_nodes) == 1
        assert source_nodes[0].name == "data"
        assert len(transform_nodes) == 1
        assert transform_nodes[0].name == "t1"

    def test_source_node_with_columns_and_validate_sql(self, tmp_path):
        """Source nodes with columns and validate_sql are parsed."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "NODES = [\n"
            "    {\n"
            '        "name": "tbl",\n'
            '        "source": [{"a": 1, "b": 2}],\n'
            '        "columns": ["a", "b"],\n'
            "        \"validate_sql\": \"CREATE OR REPLACE VIEW tbl__validation AS SELECT CASE WHEN COUNT(*) > 0 THEN 'fail' ELSE 'pass' END AS status, 'null a' AS message FROM tbl WHERE a IS NULL\",\n"
            "    },\n"
            '    {"name": "t", "depends_on": ["tbl"], '
            '"sql": "CREATE OR REPLACE VIEW t_o AS SELECT 1 AS x"},\n'
            "]\n",
        )

        nodes, exports = load_spec_from_module(module_path)
        tbl_node = [n for n in nodes if n.name == "tbl"][0]
        assert tbl_node.is_source()
        assert tbl_node.columns == ["a", "b"]
        assert "tbl__validation" in tbl_node.validate_sql
        assert tbl_node.source == [{"a": 1, "b": 2}]

    def test_file_input_parsed_relative_to_spec(self, tmp_path):
        """File source strings resolve relative to the spec directory."""
        from src.ingest import FileInput
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "NODES = [\n"
            '    {"name": "sales", "source": "data/sales.csv"},\n'
            '    {"name": "t", "depends_on": ["sales"], '
            '"sql": "CREATE OR REPLACE VIEW t_o AS SELECT 1 AS x"},\n'
            "]\n",
        )

        module_dir = tmp_path / module_path
        data_dir = module_dir / "data"
        data_dir.mkdir()
        (data_dir / "sales.csv").write_text("id,val\n1,a\n")

        nodes, exports = load_spec_from_module(module_path)
        sales_node = [n for n in nodes if n.name == "sales"][0]
        file_input = sales_node.source
        assert isinstance(file_input, FileInput)
        assert file_input.format == "csv"
        assert file_input.path == (data_dir / "sales.csv").resolve()

    def test_simple_source_no_validation(self, tmp_path):
        """Simple source nodes (no validate_sql) have no validation metadata."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [\n    {"name": "t", "source": [{"x": 1}]},\n]\n',
        )

        nodes, exports = load_spec_from_module(module_path)
        assert len(nodes) == 1
        assert nodes[0].is_source()
        assert not nodes[0].has_validation()
        assert nodes[0].columns == []

    def test_missing_nodes_raises(self, tmp_path):
        """Spec without NODES raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(tmp_path, "X = 1\n")

        with pytest.raises(ValueError, match="must define NODES"):
            load_spec_from_module(module_path)

    def test_exports_optional(self, tmp_path):
        """Spec without EXPORTS returns empty dict."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [\n    {"name": "t", "source": []},\n]\n',
        )

        nodes, exports = load_spec_from_module(module_path)
        assert exports == {}

    def test_exports_loaded(self, tmp_path):
        """Spec with EXPORTS includes them in the result."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "NODES = [\n"
            '    {"name": "t", "source": []},\n'
            "]\n"
            "def my_export(conn, path): pass\n"
            'EXPORTS = {"out.csv": my_export}\n',
        )

        nodes, exports = load_spec_from_module(module_path)
        assert "out.csv" in exports
        assert callable(exports["out.csv"])

    def test_invalid_node_type_raises(self, tmp_path):
        """Non-dict node raises ValueError with index."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = ["not a valid node"]\n',
        )

        with pytest.raises(ValueError, match=r"NODES\[0\] must be a dict"):
            load_spec_from_module(module_path)

    def test_empty_node_name_raises(self, tmp_path):
        """Node with empty name raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [{"name": "", "source": []}]\n',
        )

        with pytest.raises(ValueError, match="missing required 'name'"):
            load_spec_from_module(module_path)

    def test_missing_node_name_raises(self, tmp_path):
        """Node without name key raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [{"source": []}]\n',
        )

        with pytest.raises(ValueError, match=r"NODES\[0\] is missing required 'name'"):
            load_spec_from_module(module_path)

    def test_duplicate_node_name_raises(self, tmp_path):
        """Duplicate node names raise ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "NODES = [\n"
            '  {"name": "dup", "source": [{"x": 1}]},\n'
            '  {"name": "dup", "sql": "CREATE OR REPLACE VIEW dup_a AS SELECT 1 AS x"},\n'
            "]\n",
        )

        with pytest.raises(ValueError, match="Duplicate node name.*dup"):
            load_spec_from_module(module_path)

    def test_unknown_node_fields_raises(self, tmp_path):
        """Unknown node dict keys raise ValueError listing valid fields."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [{"name": "t", "sql": "CREATE OR REPLACE VIEW t_o AS SELECT 1 AS x", '
            '"description": "nope"}]\n',
        )

        with pytest.raises(ValueError, match="unknown field.*description"):
            load_spec_from_module(module_path)

    def test_node_with_both_source_and_sql_raises(self, tmp_path):
        """Node with both source and sql raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [{"name": "t", "source": [{"x": 1}], '
            '"sql": "CREATE OR REPLACE VIEW t_o AS SELECT 1 AS x"}]\n',
        )

        with pytest.raises(ValueError, match="multiple modes"):
            load_spec_from_module(module_path)

    def test_node_missing_mode_raises(self, tmp_path):
        """Node with none of source/sql/prompt raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [{"name": "t"}]\n',
        )

        with pytest.raises(
            ValueError, match="must have exactly one of.*source.*sql.*prompt"
        ):
            load_spec_from_module(module_path)

    def test_output_columns_as_list_raises(self, tmp_path):
        """Node output_columns as list (not dict) raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [{"name": "t", '
            '"sql": "CREATE OR REPLACE VIEW t_o AS SELECT 1 AS x", '
            '"output_columns": ["x"]}]\n',
        )

        with pytest.raises(ValueError, match="output_columns must be a dict"):
            load_spec_from_module(module_path)

    def test_import_error_preserves_context(self, tmp_path):
        """Module import errors are wrapped with context."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "import nonexistent_module_xyz\nNODES = []\n",
        )

        with pytest.raises(ValueError, match="Cannot import spec module"):
            load_spec_from_module(module_path)

    def test_callable_source_preserved(self, tmp_path):
        """Callable source values are preserved (not called at load time)."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'def load_data(): return [{"x": 1}]\n'
            "NODES = [\n"
            '    {"name": "t", "source": load_data},\n'
            "]\n",
        )

        nodes, exports = load_spec_from_module(module_path)
        source_node = [n for n in nodes if n.name == "t"][0]
        assert callable(source_node.source)

    def test_existing_spec_files_load(self):
        """The existing test spec files load without error."""
        from src.spec import load_spec_from_module

        # diamond_dag.py
        nodes, exports = load_spec_from_module("tests.diamond_dag")
        source_nodes = [n for n in nodes if n.is_source()]
        transform_nodes = [n for n in nodes if not n.is_source()]
        assert len(transform_nodes) == 4
        transactions_node = [n for n in source_nodes if n.name == "transactions"][0]
        assert transactions_node.columns == [
            "id",
            "date",
            "type",
            "product",
            "amount",
            "region",
        ]

        # validation_view_demo.py
        nodes, exports = load_spec_from_module("tests.validation_view_demo")
        transform_nodes = [n for n in nodes if not n.is_source()]
        assert len(transform_nodes) == 1
        source_names = {n.name for n in nodes if n.is_source()}
        assert "expenses" in source_names


def test_default_output_db_path_is_stable():
    from datetime import datetime, timezone

    from scripts.cli import _default_output_db_path

    p = _default_output_db_path(
        "my_app.specs.main", now=datetime(2026, 2, 16, 12, 34, 56, tzinfo=timezone.utc)
    )
    assert str(p) == "runs/my_app-specs-main_20260216_123456.db"


# _resolve_spec_arg tests


class TestResolveSpecArg:
    """Test file-path-to-module-path resolution."""

    def test_module_path_passthrough(self):
        from scripts.cli import _resolve_spec_arg

        assert _resolve_spec_arg("specs.main") == "specs.main"
        assert _resolve_spec_arg("tests.single_task") == "tests.single_task"

    def test_file_path_with_py_extension(self):
        from scripts.cli import _resolve_spec_arg

        assert _resolve_spec_arg("specs/main.py") == "specs.main"
        assert _resolve_spec_arg("tests/diamond_dag.py") == "tests.diamond_dag"

    def test_file_path_without_extension(self):
        from scripts.cli import _resolve_spec_arg

        assert _resolve_spec_arg("specs/main") == "specs.main"

    def test_dotslash_prefix(self):
        from scripts.cli import _resolve_spec_arg

        assert _resolve_spec_arg("./specs/main.py") == "specs.main"

    def test_deep_path(self):
        from scripts.cli import _resolve_spec_arg

        assert _resolve_spec_arg("a/b/c.py") == "a.b.c"

    def test_backslash_path(self):
        from scripts.cli import _resolve_spec_arg

        assert _resolve_spec_arg("specs\\main.py") == "specs.main"


# tg init tests


class TestInit:
    """Test the init command creates all expected files."""

    def test_init_creates_all_files(self, tmp_path, monkeypatch):
        from scripts.cli import main

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output

        assert (tmp_path / "pyproject.toml").exists()
        assert (tmp_path / "specs" / "main.py").exists()
        assert (tmp_path / "specs" / "__init__.py").exists()
        assert (tmp_path / ".gitignore").exists()
        assert (tmp_path / ".env").exists()

        # pyproject.toml content
        content = (tmp_path / "pyproject.toml").read_text()
        assert "[project]" in content
        assert "[tool.taskgraph]" in content
        assert 'spec = "specs.main"' in content

        # .env content
        env_content = (tmp_path / ".env").read_text()
        assert "OPENROUTER_API_KEY" in env_content

        # .gitignore includes .env
        gi = (tmp_path / ".gitignore").read_text()
        assert ".env" in gi

        # Output messages
        assert "created" in result.output
        assert "tg run" in result.output

    def test_init_skips_existing_files(self, tmp_path, monkeypatch):
        from scripts.cli import main

        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'existing'\n")

        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output

        # Should not overwrite existing pyproject.toml
        content = (tmp_path / "pyproject.toml").read_text()
        assert "existing" in content
        assert "exists" in result.output

    def test_init_force_overwrites(self, tmp_path, monkeypatch):
        from scripts.cli import main

        monkeypatch.chdir(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'old'\n")

        runner = CliRunner()
        result = runner.invoke(main, ["init", "--force"])
        assert result.exit_code == 0, result.output

        content = (tmp_path / "pyproject.toml").read_text()
        assert "old" not in content
        assert "[tool.taskgraph]" in content

    def test_init_project_name_from_dir(self, tmp_path, monkeypatch):
        from scripts.cli import main

        project_dir = tmp_path / "My Cool Project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output

        content = (project_dir / "pyproject.toml").read_text()
        assert 'name = "my-cool-project"' in content

    def test_init_scaffold_is_sql(self, tmp_path, monkeypatch):
        """Scaffold spec uses sql so tg run works without an API key."""
        from scripts.cli import main

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output

        spec_content = (tmp_path / "specs" / "main.py").read_text()
        assert '"sql":' in spec_content
        assert "sql_strict" not in spec_content

    def test_init_appends_env_to_existing_gitignore(self, tmp_path, monkeypatch):
        from scripts.cli import main

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".gitignore").write_text("*.pyc\n")

        runner = CliRunner()
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output

        gi = (tmp_path / ".gitignore").read_text()
        assert ".env" in gi
        assert "*.pyc" in gi  # Original content preserved


class TestShow:
    """Tests for the tg show command."""

    def test_show_nondb_file_rejects(self, tmp_path, monkeypatch):
        """tg show with a non-.db file gives a clear error."""
        from scripts.cli import main

        monkeypatch.chdir(tmp_path)
        (tmp_path / "data.txt").write_text("hello")

        runner = CliRunner()
        result = runner.invoke(main, ["show", str(tmp_path / "data.txt")])
        assert result.exit_code != 0
        assert "not a .db file" in result.output

    def test_show_missing_db_rejects(self, tmp_path, monkeypatch):
        """tg show with a nonexistent .db file gives a clear error."""
        from scripts.cli import main

        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["show", str(tmp_path / "missing.db")])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_show_spec(self, tmp_path, monkeypatch):
        """tg show --spec displays spec structure."""
        from scripts.cli import main

        spec_source = """\
NODES = [
    {"name": "data", "source": [{"x": 1}]},
    {
        "name": "t",
        "depends_on": ["data"],
        "sql": "CREATE OR REPLACE VIEW t_o AS SELECT 1 AS x",
    },
]
"""
        spec_module = _write_spec_module(tmp_path, spec_source)

        runner = CliRunner()
        result = runner.invoke(main, ["show", "--spec", spec_module])
        assert result.exit_code == 0, result.output
        assert "Nodes (" in result.output
        assert "DAG" in result.output
