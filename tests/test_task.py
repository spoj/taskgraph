import pytest

from tests.conftest import _make_node, _write_spec_module
from src.task import discover_validation_objects, resolve_dag, validate_graph


class TestValidateOutputs:
    """Tests for Node.validate_outputs/validate_validation_views."""

    def test_missing_output_view(self, conn):
        """validate_outputs returns error when a declared output view doesn't exist."""
        node = _make_node(name="t", output_columns={"t_missing": []})
        errors = node.validate_outputs(conn)
        assert len(errors) == 1
        assert "t_missing" in errors[0]
        assert "not created" in errors[0].lower()

    def test_existing_output_view_passes(self, conn):
        """validate_outputs passes when all declared output views exist."""
        conn.execute("CREATE VIEW t_view AS SELECT 1 AS x")
        node = _make_node(name="t", output_columns={"t_view": []})
        errors = node.validate_outputs(conn)
        assert errors == []

    def test_multiple_missing_views(self, conn):
        """validate_outputs reports all missing views."""
        conn.execute("CREATE VIEW t_v1 AS SELECT 1 AS x")
        node = _make_node(name="t", output_columns={"t_v1": [], "t_v2": [], "t_v3": []})
        errors = node.validate_outputs(conn)
        assert len(errors) == 2  # t_v2 and t_v3 missing
        assert any("t_v2" in e for e in errors)
        assert any("t_v3" in e for e in errors)

    def test_output_columns_pass(self, conn):
        """output_columns validation passes when view has required columns."""
        conn.execute(
            "CREATE VIEW t_report AS SELECT 1 AS product, 2.0 AS total, 0.5 AS margin"
        )
        node = _make_node(
            name="t",
            output_columns={"t_report": ["product", "total", "margin"]},
        )
        errors = node.validate_outputs(conn)
        assert errors == []

    def test_output_columns_missing_column(self, conn):
        """output_columns validation fails when view is missing a required column."""
        conn.execute("CREATE VIEW t_report AS SELECT 1 AS product, 2.0 AS total")
        node = _make_node(
            name="t",
            output_columns={"t_report": ["product", "total", "margin"]},
        )
        errors = node.validate_outputs(conn)
        assert len(errors) == 1
        assert "margin" in errors[0]
        assert "product" in errors[0]  # actual columns listed in error

    def test_output_columns_multiple_missing(self, conn):
        """output_columns validation reports all missing columns at once."""
        conn.execute("CREATE VIEW t_report AS SELECT 1 AS id")
        node = _make_node(
            name="t",
            output_columns={"t_report": ["id", "amount", "category"]},
        )
        errors = node.validate_outputs(conn)
        assert len(errors) == 1
        assert "amount" in errors[0]
        assert "category" in errors[0]

    def test_output_columns_extra_columns_ok(self, conn):
        """Extra columns beyond required are fine."""
        conn.execute("CREATE VIEW t_report AS SELECT 1 AS a, 2 AS b, 3 AS c, 4 AS d")
        node = _make_node(
            name="t",
            output_columns={"t_report": ["a", "c"]},
        )
        errors = node.validate_outputs(conn)
        assert errors == []

    def test_output_columns_case_insensitive(self, conn):
        """Column check is case-insensitive (DuckDB normalizes names)."""
        conn.execute("CREATE VIEW t_report AS SELECT 1 AS Product, 2.0 AS TOTAL")
        node = _make_node(
            name="t",
            output_columns={"t_report": ["product", "total"]},
        )
        errors = node.validate_outputs(conn)
        assert errors == []

    def test_output_columns_reports_all_views(self, conn):
        """Column check reports errors for all views, not just the first."""
        conn.execute("CREATE VIEW t_v1 AS SELECT 1 AS a")
        conn.execute("CREATE VIEW t_v2 AS SELECT 1 AS x")
        node = _make_node(
            name="t",
            output_columns={"t_v1": ["a", "b"], "t_v2": ["x", "y"]},
        )
        errors = node.validate_outputs(conn)
        assert len(errors) == 2
        assert any("t_v1" in e and "b" in e for e in errors)
        assert any("t_v2" in e and "y" in e for e in errors)

    def test_output_columns_skips_missing_view(self, conn):
        """output_columns check skips views that don't exist (caught in step 1)."""
        node = _make_node(
            name="t",
            output_columns={"t_missing": ["col1"]},
        )
        errors = node.validate_outputs(conn)
        # Should get "not created" error from step 1, not a column error
        assert len(errors) == 1
        assert "not created" in errors[0].lower()

    def test_validation_view_pass(self, conn):
        """Validation view passes when it contains no fail rows."""
        conn.execute(
            "CREATE VIEW t__validation_main AS SELECT 'pass' AS status, 'ok' AS message"
        )
        node = _make_node(
            name="t",
            validate={"main": "SELECT 'pass' AS status, 'ok' AS message"},
        )
        errors = node.validate_validation_views(conn)
        assert errors == []

    def test_validation_view_fail(self, conn):
        """Validation view fails when it contains any fail row."""
        conn.execute(
            "CREATE VIEW t__validation_main AS SELECT 'fail' AS status, 'bad' AS message"
        )
        node = _make_node(
            name="t",
            validate={"main": "SELECT 'fail' AS status, 'bad' AS message"},
        )
        errors = node.validate_validation_views(conn)
        assert errors
        assert "bad" in "\n".join(errors)

    def test_multiple_validation_views_enforced(self, conn):
        conn.execute(
            "CREATE VIEW t__validation_a AS SELECT 'pass' AS status, 'ok' AS message"
        )
        conn.execute(
            "CREATE VIEW t__validation_b AS SELECT 'fail' AS status, 'nope' AS message"
        )
        node = _make_node(
            name="t",
            validate={
                "a": "SELECT 'pass' AS status, 'ok' AS message",
                "b": "SELECT 'fail' AS status, 'nope' AS message",
            },
        )
        errors = node.validate_validation_views(conn)
        assert errors
        # Should mention the failing message
        assert "nope" in "\n".join(errors)

    def test_validation_order_view_before_columns(self, conn):
        """Step 1 (view existence) runs before step 2 (column check)."""
        node = _make_node(
            name="t",
            output_columns={"t_missing": ["col1"]},
        )
        errors = node.validate_outputs(conn)
        assert len(errors) == 1
        assert "not created" in errors[0].lower()

    def test_discover_validation_objects(self, conn):
        conn.execute(
            "CREATE VIEW t__validation_main AS SELECT 'pass' AS status, 'ok' AS message"
        )
        conn.execute(
            "CREATE VIEW t__validation_extra AS SELECT 'pass' AS status, 'ok' AS message"
        )
        conn.execute("CREATE VIEW other_view AS SELECT 1 AS x")
        assert discover_validation_objects(conn, "t") == [
            "t__validation_extra",
            "t__validation_main",
        ]

    def test_validate_graph_rejects_double_underscore_in_name(self):
        node = _make_node(name="bad__name")
        errors = validate_graph([node])
        assert errors
        assert any("bad__name" in e and "__" in e for e in errors)

    def test_node_type(self):
        prompt_node = _make_node(prompt="do the thing", sql="")
        assert prompt_node.node_type() == "prompt"

        sql_node = _make_node(sql="CREATE VIEW t_out AS SELECT 1 AS x")
        assert sql_node.node_type() == "sql"

        source_node = _make_node(name="s", source=[{"a": 1}], sql="")
        assert source_node.node_type() == "source"


class TestDAGResolution:
    """Tests for resolve_dag() and validate_graph()."""

    def test_single_node_one_layer(self):
        """Single node produces one layer."""
        nodes = [_make_node(name="a")]
        layers = resolve_dag(nodes)
        assert len(layers) == 1
        assert [n.name for n in layers[0]] == ["a"]

    def test_linear_chain(self):
        """A -> B -> C produces 3 layers."""
        nodes = [
            _make_node(name="a"),
            _make_node(name="b", depends_on=["a"]),
            _make_node(name="c", depends_on=["b"]),
        ]
        layers = resolve_dag(nodes)
        assert len(layers) == 3
        assert [n.name for n in layers[0]] == ["a"]
        assert [n.name for n in layers[1]] == ["b"]
        assert [n.name for n in layers[2]] == ["c"]

    def test_diamond_dag(self):
        """Diamond: prep -> (sales, costs) -> report produces 3 layers."""
        nodes = [
            _make_node(name="prep"),
            _make_node(name="sales", depends_on=["prep"]),
            _make_node(name="costs", depends_on=["prep"]),
            _make_node(name="report", depends_on=["sales", "costs"]),
        ]
        layers = resolve_dag(nodes)
        assert len(layers) == 3
        assert [n.name for n in layers[0]] == ["prep"]
        assert sorted(n.name for n in layers[1]) == ["costs", "sales"]
        assert [n.name for n in layers[2]] == ["report"]

    def test_parallel_independent_nodes(self):
        """Independent nodes land in the same layer."""
        nodes = [
            _make_node(name="a"),
            _make_node(name="b"),
            _make_node(name="c"),
        ]
        layers = resolve_dag(nodes)
        assert len(layers) == 1
        assert sorted(n.name for n in layers[0]) == ["a", "b", "c"]

    def test_cycle_detection(self):
        """Cycle raises ValueError."""
        nodes = [
            _make_node(name="a", depends_on=["b"]),
            _make_node(name="b", depends_on=["a"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            resolve_dag(nodes)

    def test_validate_graph_valid(self):
        """Valid graph with all dependencies satisfied."""
        raw_node = _make_node(name="raw_data", source=[{"id": 1}], sql="")
        proc_node = _make_node(name="proc", depends_on=["raw_data"])
        errors = validate_graph([raw_node, proc_node])
        assert errors == []

    def test_validate_graph_missing_dep(self):
        """Reports missing dependencies that are not known node names."""
        node = _make_node(name="t", depends_on=["nonexistent"])
        errors = validate_graph([node])
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_validate_graph_all_deps_present(self):
        """Dependencies available as source nodes are fine."""
        emp_node = _make_node(name="employees", source=[{"id": 1}], sql="")
        dep_node = _make_node(name="departments", source=[{"id": 1}], sql="")
        proc_node = _make_node(name="proc", depends_on=["employees", "departments"])
        errors = validate_graph([emp_node, dep_node, proc_node])
        assert errors == []

    def test_complex_dag_ordering(self):
        """Complex DAG: 5 nodes with mixed dependencies produce correct layers."""
        nodes = [
            _make_node(name="ingest"),
            _make_node(name="clean", depends_on=["ingest"]),
            _make_node(name="enrich", depends_on=["ingest"]),
            _make_node(name="merge", depends_on=["clean", "enrich"]),
            _make_node(name="report", depends_on=["merge"]),
        ]
        layers = resolve_dag(nodes)
        assert len(layers) == 4
        assert [n.name for n in layers[0]] == ["ingest"]
        assert sorted(n.name for n in layers[1]) == ["clean", "enrich"]
        assert [n.name for n in layers[2]] == ["merge"]
        assert [n.name for n in layers[3]] == ["report"]


class TestPromptResolution:
    """Tests for prompt handling in spec loading."""

    def test_string_prompt_passthrough(self, tmp_path):
        """Plain string prompt is accepted in spec loading."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "NODES = [\n"
            '    {"name": "t", "source": [{"x": 1}]},\n'
            '    {"name": "a", "depends_on": ["t"], "prompt": "do the thing",\n'
            '     "output_columns": {"a_out": ["x"]}},\n'
            "]\n",
        )
        nodes, exports = load_spec_from_module(module_path)
        prompt_nodes = [n for n in nodes if n.prompt]
        assert len(prompt_nodes) == 1
        assert prompt_nodes[0].prompt == "do the thing"

    def test_non_string_prompt_raises(self, tmp_path):
        """Non-string prompt raises ValueError."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "NODES = [\n"
            '    {"name": "t", "source": [{"x": 1}]},\n'
            '    {"name": "a", "depends_on": ["t"], "prompt": 123,\n'
            '     "output_columns": {"a_out": ["x"]}},\n'
            "]\n",
        )
        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            load_spec_from_module(module_path)

    def test_empty_prompt_raises(self, tmp_path):
        """Prompt nodes must provide non-empty prompt."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            "NODES = [\n"
            '    {"name": "t", "source": [{"x": 1}]},\n'
            '    {"name": "a", "depends_on": ["t"], "prompt": " ",\n'
            '     "output_columns": {"a_out": ["x"]}},\n'
            "]\n",
        )
        with pytest.raises(ValueError, match="prompt must be a non-empty string"):
            load_spec_from_module(module_path)

    def test_missing_transform_raises(self, tmp_path):
        """Nodes must specify exactly one of source, sql, or prompt."""
        from src.spec import load_spec_from_module

        module_path = _write_spec_module(
            tmp_path,
            'NODES = [\n    {"name": "a", "depends_on": []},\n]\n',
        )
        with pytest.raises(
            ValueError, match="must have exactly one of: source, sql, prompt"
        ):
            load_spec_from_module(module_path)
