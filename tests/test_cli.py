import duckdb
import pytest
from click.testing import CliRunner

from tests.conftest import _write_spec_module


def test_cli_run_sql_does_not_require_openrouter_api_key(tmp_path, monkeypatch):
    """SQL-only specs should run without OPENROUTER_API_KEY.

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
        "sql": "CREATE VIEW v AS SELECT 1 AS x",
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
        """Loads a minimal spec with INPUTS and TASKS."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"data": [{"x": 1}]}\n'
            'TASKS = [{"name": "t1", '
            '"sql": "CREATE VIEW out AS SELECT 1 AS x", '
            '"inputs": ["data"], "outputs": ["out"]}]\n',
        )

        result = load_spec_from_module(module_path)
        assert "data" in result["inputs"]
        assert len(result["tasks"]) == 1
        assert result["tasks"][0].name == "t1"

    def test_rich_inputs_extracted(self, tmp_path):
        """Rich INPUTS with columns and validate_sql are parsed."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "INPUTS = {\n"
            '    "tbl": {\n'
            '        "source": [{"a": 1, "b": 2}],\n'
            '        "columns": ["a", "b"],\n'
            '        "validate_sql": "SELECT 1 FROM tbl WHERE a IS NULL",\n'
            "    }\n"
            "}\n"
            'TASKS = [{"name": "t", '
            '"sql": "CREATE VIEW o AS SELECT 1 AS x", '
            '"inputs": ["tbl"], "outputs": ["o"]}]\n',
        )

        result = load_spec_from_module(module_path)
        assert result["input_columns"] == {"tbl": ["a", "b"]}
        assert result["input_validate_sql"] == {
            "tbl": "SELECT 1 FROM tbl WHERE a IS NULL"
        }
        # The actual data is the list, not the dict
        assert result["inputs"]["tbl"] == [{"a": 1, "b": 2}]

    def test_file_input_parsed_relative_to_spec(self, tmp_path):
        """File input strings resolve relative to the spec directory."""
        from src.ingest import FileInput
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"sales": "data/sales.csv"}\n'
            'TASKS = [{"name": "t", '
            '"sql": "CREATE VIEW o AS SELECT 1 AS x", '
            '"inputs": ["sales"], "outputs": ["o"]}]\n',
        )

        module_dir = tmp_path / module_path
        data_dir = module_dir / "data"
        data_dir.mkdir()
        (data_dir / "sales.csv").write_text("id,val\n1,a\n")

        result = load_spec_from_module(module_path)
        file_input = result["inputs"]["sales"]
        assert isinstance(file_input, FileInput)
        assert file_input.format == "csv"
        assert file_input.path == (data_dir / "sales.csv").resolve()

    def test_simple_input_no_validation(self, tmp_path):
        """Simple INPUTS (no dict with 'source') have no validation metadata."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": [{"x": 1}]}\n'
            'TASKS = [{"name": "t", '
            '"sql": "CREATE VIEW o AS SELECT 1 AS x", '
            '"inputs": ["t"], "outputs": ["o"]}]\n',
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
            'TASKS = [{"name": "t", '
            '"sql": "CREATE VIEW o AS SELECT 1 AS x", '
            '"inputs": ["t"], "outputs": ["o"]}]\n',
        )

        result = load_spec_from_module(module_path)
        assert result["exports"] == {}

    def test_exports_loaded(self, tmp_path):
        """Spec with EXPORTS includes them in the result."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": []}\n'
            'TASKS = [{"name": "t", '
            '"sql": "CREATE VIEW o AS SELECT 1 AS x", '
            '"inputs": ["t"], "outputs": ["o"]}]\n'
            "def my_export(conn, path): pass\n"
            'EXPORTS = {"out.csv": my_export}\n',
        )

        result = load_spec_from_module(module_path)
        assert "out.csv" in result["exports"]
        assert callable(result["exports"]["out.csv"])

    def test_task_objects_accepted(self, tmp_path):
        """Tasks can be Task objects directly (not just dicts)."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "from src.task import Task\n"
            'INPUTS = {"t": []}\n'
            'TASKS = [Task(name="t", '
            'sql="CREATE VIEW o AS SELECT 1 AS x", '
            'inputs=["t"], outputs=["o"])]\n',
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
            'TASKS = [{"name": "t", '
            '"sql": "CREATE VIEW o AS SELECT 1 AS x", '
            '"inputs": ["t"], "outputs": ["o"]}]\n',
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
