import pytest

from tests.conftest import _make_task, _write_spec_module
from src.task import resolve_dag, validate_task_graph


class TestTaskValidation:
    """Tests for Task.validate_transform/validate_validation_views."""

    def test_missing_output_view(self, conn):
        """validate_transform returns error when a declared output view doesn't exist."""
        task = _make_task(outputs=["my_view"])
        errors = task.validate_transform(conn)
        assert len(errors) == 1
        assert "my_view" in errors[0]
        assert "not created" in errors[0].lower()

    def test_existing_output_view_passes(self, conn):
        """validate_transform passes when all declared output views exist."""
        conn.execute("CREATE VIEW my_view AS SELECT 1 AS x")
        task = _make_task(outputs=["my_view"])
        errors = task.validate_transform(conn)
        assert errors == []

    def test_multiple_missing_views(self, conn):
        """validate_transform reports all missing views."""
        conn.execute("CREATE VIEW v1 AS SELECT 1 AS x")
        task = _make_task(outputs=["v1", "v2", "v3"])
        errors = task.validate_transform(conn)
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
        errors = task.validate_transform(conn)
        assert errors == []

    def test_output_columns_missing_column(self, conn):
        """output_columns validation fails when view is missing a required column."""
        conn.execute("CREATE VIEW report AS SELECT 1 AS product, 2.0 AS total")
        task = _make_task(
            outputs=["report"],
            output_columns={"report": ["product", "total", "margin"]},
        )
        errors = task.validate_transform(conn)
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
        errors = task.validate_transform(conn)
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
        errors = task.validate_transform(conn)
        assert errors == []

    def test_output_columns_skips_missing_view(self, conn):
        """output_columns check skips views that don't exist (caught in step 1)."""
        task = _make_task(
            outputs=["missing_view"],
            output_columns={"missing_view": ["col1"]},
        )
        errors = task.validate_transform(conn)
        # Should get "not created" error from step 1, not a column error
        assert len(errors) == 1
        assert "not created" in errors[0].lower()

    def test_validation_view_pass(self, conn):
        """Validation view passes when it contains no fail rows."""
        conn.execute(
            "CREATE VIEW t__validation AS SELECT 'pass' AS status, 'ok' AS message"
        )
        task = _make_task(
            name="t",
            validate_sql="CREATE VIEW t__validation AS SELECT 'pass' AS status, 'ok' AS message",
        )
        errors = task.validate_validation_views(conn)
        assert errors == []

    def test_validation_view_fail(self, conn):
        """Validation view fails when it contains any fail row."""
        conn.execute(
            "CREATE VIEW t__validation AS SELECT 'fail' AS status, 'bad' AS message"
        )
        task = _make_task(
            name="t",
            validate_sql="CREATE VIEW t__validation AS SELECT 'fail' AS status, 'bad' AS message",
        )
        errors = task.validate_validation_views(conn)
        assert errors
        assert "bad" in "\n".join(errors)

    def test_multiple_validation_views_enforced(self, conn):
        conn.execute(
            "CREATE VIEW t__validation_a AS SELECT 'pass' AS status, 'ok' AS message"
        )
        conn.execute(
            "CREATE VIEW t__validation_b AS SELECT 'fail' AS status, 'nope' AS message"
        )
        task = _make_task(
            name="t",
            validate_sql=(
                "CREATE VIEW t__validation_a AS SELECT 'pass' AS status, 'ok' AS message;"
                "CREATE VIEW t__validation_b AS SELECT 'fail' AS status, 'nope' AS message"
            ),
        )
        errors = task.validate_validation_views(conn)
        assert errors
        # Should mention the failing message
        assert "nope" in "\n".join(errors)

    def test_validation_order_view_before_columns(self, conn):
        """Step 1 (view existence) runs before step 2 (column check)."""
        task = _make_task(
            outputs=["missing"],
            output_columns={"missing": ["col1"]},
        )
        errors = task.validate_transform(conn)
        assert len(errors) == 1
        assert "not created" in errors[0].lower()

    def test_validation_view_names(self):
        task = _make_task(
            name="t",
            validate_sql=(
                "CREATE VIEW t__validation AS SELECT 'pass' AS status, 'ok' AS message;"
                "CREATE VIEW t__validation_extra AS SELECT 'pass' AS status, 'ok' AS message;"
                "CREATE VIEW other_view AS SELECT 1 AS x"
            ),
        )
        assert task.validation_view_names() == ["t__validation", "t__validation_extra"]

    def test_transform_mode(self):
        prompt_task = _make_task()
        assert prompt_task.transform_mode() == "prompt"

        sql_task = _make_task(sql="CREATE VIEW out AS SELECT 1 AS x")
        assert sql_task.transform_mode() == "sql"


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


class TestPromptResolution:
    """Tests for prompt handling in spec loading."""

    def test_string_prompt_passthrough(self, tmp_path):
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

    def test_empty_prompt_raises(self, tmp_path):
        """Prompt tasks must provide non-empty prompt."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": [{"x": 1}]}\n'
            'TASKS = [{"name": "a", "prompt": " ", '
            '"inputs": ["t"], "outputs": ["out"]}]\n',
        )
        with pytest.raises(ValueError, match="prompt must not be empty"):
            load_spec_from_module(module_path)

    def test_missing_transform_raises(self, tmp_path):
        """Tasks must specify exactly one of sql or prompt."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'INPUTS = {"t": [{"x": 1}]}\n'
            'TASKS = [{"name": "a", "inputs": ["t"], "outputs": ["out"]}]\n',
        )
        with pytest.raises(
            ValueError, match="must specify exactly one of 'sql' or 'prompt'"
        ):
            load_spec_from_module(module_path)
