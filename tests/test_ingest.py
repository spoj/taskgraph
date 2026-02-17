import pytest
import polars as pl

from tests.conftest import _make_task
from src.ingest import coerce_to_dataframe, ingest_table
from src.workspace import Workspace


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
