"""Tests for src/sql_utils.py — SQL parsing, DDL extraction, column queries."""

import pytest

from src.sql_utils import (
    DdlTarget,
    SqlParseError,
    extract_create_name,
    extract_ddl_target,
    get_column_names,
    get_column_schema,
    parse_one_statement,
    split_sql_statements,
    strip_leading_sql_comments,
)


class TestSplitSqlStatements:
    def test_single_statement(self):
        assert split_sql_statements("SELECT 1") == ["SELECT 1"]

    def test_multiple_statements(self):
        stmts = split_sql_statements("SELECT 1; SELECT 2")
        assert len(stmts) == 2

    def test_empty_string(self):
        assert split_sql_statements("") == []

    def test_none(self):
        assert split_sql_statements(None) == []

    def test_whitespace_only(self):
        assert split_sql_statements("   \n\t  ") == []

    def test_unparseable_returns_as_single(self):
        """Garbage SQL falls back to returning input as single statement."""
        result = split_sql_statements("THIS IS NOT SQL AT ALL }{}{")
        assert len(result) == 1


class TestParseOneStatement:
    def test_valid_select(self):
        parsed = parse_one_statement("SELECT 1")
        assert "SELECT" in parsed.sql.upper()

    def test_empty_raises(self):
        with pytest.raises(SqlParseError, match="Empty"):
            parse_one_statement("")

    def test_none_raises(self):
        with pytest.raises(SqlParseError, match="Empty"):
            parse_one_statement(None)

    def test_multiple_statements_raises(self):
        with pytest.raises(SqlParseError, match="one statement"):
            parse_one_statement("SELECT 1; SELECT 2")

    def test_preserves_sql_text(self):
        parsed = parse_one_statement("CREATE VIEW v AS SELECT 42")
        assert "42" in parsed.sql


class TestStripLeadingSqlComments:
    def test_no_comments(self):
        assert strip_leading_sql_comments("SELECT 1") == "SELECT 1"

    def test_line_comment(self):
        result = strip_leading_sql_comments("-- comment\nSELECT 1")
        assert result == "SELECT 1"

    def test_block_comment(self):
        result = strip_leading_sql_comments("/* comment */\nSELECT 1")
        assert result == "SELECT 1"

    def test_multiple_comments(self):
        sql = "-- first\n/* second */\n-- third\nSELECT 1"
        assert strip_leading_sql_comments(sql) == "SELECT 1"

    def test_comment_only_line(self):
        assert strip_leading_sql_comments("-- nothing here") == ""

    def test_unterminated_block_comment(self):
        assert strip_leading_sql_comments("/* never closed") == ""

    def test_empty(self):
        assert strip_leading_sql_comments("") == ""

    def test_none(self):
        assert strip_leading_sql_comments(None) == ""

    def test_leading_whitespace(self):
        result = strip_leading_sql_comments("  \n  SELECT 1")
        assert result == "SELECT 1"


class TestExtractDdlTarget:
    """Tests for extract_ddl_target — the regex-based DDL name extractor."""

    # --- CREATE VIEW ---
    def test_create_view(self):
        t = extract_ddl_target("CREATE VIEW my_view AS SELECT 1")
        assert t == DdlTarget(action="create", kind="view", name="my_view")

    def test_create_or_replace_view(self):
        t = extract_ddl_target("CREATE OR REPLACE VIEW v AS SELECT 1")
        assert t == DdlTarget(action="create", kind="view", name="v")

    def test_create_temp_view(self):
        t = extract_ddl_target("CREATE TEMP VIEW tmp AS SELECT 1")
        assert t == DdlTarget(action="create", kind="view", name="tmp")

    def test_create_view_if_not_exists(self):
        t = extract_ddl_target("CREATE VIEW IF NOT EXISTS v AS SELECT 1")
        assert t == DdlTarget(action="create", kind="view", name="v")

    def test_create_view_quoted(self):
        t = extract_ddl_target('CREATE VIEW "my_view" AS SELECT 1')
        assert t is not None
        assert t.name == "my_view"

    # --- DROP VIEW ---
    def test_drop_view(self):
        t = extract_ddl_target("DROP VIEW my_view")
        assert t == DdlTarget(action="drop", kind="view", name="my_view")

    def test_drop_view_if_exists(self):
        t = extract_ddl_target("DROP VIEW IF EXISTS v")
        assert t == DdlTarget(action="drop", kind="view", name="v")

    # --- CREATE MACRO ---
    def test_create_macro(self):
        t = extract_ddl_target("CREATE MACRO my_fn(x) AS x + 1")
        assert t == DdlTarget(action="create", kind="macro", name="my_fn")

    def test_create_or_replace_macro(self):
        t = extract_ddl_target("CREATE OR REPLACE MACRO m(x) AS x")
        assert t == DdlTarget(action="create", kind="macro", name="m")

    # --- DROP MACRO ---
    def test_drop_macro(self):
        t = extract_ddl_target("DROP MACRO my_fn")
        assert t == DdlTarget(action="drop", kind="macro", name="my_fn")

    def test_drop_macro_table(self):
        t = extract_ddl_target("DROP MACRO TABLE my_fn")
        assert t == DdlTarget(action="drop", kind="macro", name="my_fn")

    # --- Non-DDL ---
    def test_select_returns_none(self):
        assert extract_ddl_target("SELECT 1") is None

    def test_create_table_returns_none(self):
        """CREATE TABLE is not a VIEW or MACRO — should return None."""
        assert extract_ddl_target("CREATE TABLE t (id INT)") is None

    def test_insert_returns_none(self):
        assert extract_ddl_target("INSERT INTO t VALUES (1)") is None

    # --- With leading comments ---
    def test_with_leading_line_comment(self):
        t = extract_ddl_target("-- note\nCREATE VIEW v AS SELECT 1")
        assert t is not None
        assert t.name == "v"

    def test_with_leading_block_comment(self):
        t = extract_ddl_target("/* note */\nCREATE VIEW v AS SELECT 1")
        assert t is not None
        assert t.name == "v"


class TestExtractCreateName:
    """Tests for the backward-compatible extract_create_name wrapper."""

    def test_create_view(self):
        assert extract_create_name("CREATE VIEW foo AS SELECT 1") == "foo"

    def test_drop_returns_none(self):
        assert extract_create_name("DROP VIEW foo") is None

    def test_select_returns_none(self):
        assert extract_create_name("SELECT 1") is None


class TestGetColumnNames:
    def test_returns_column_set(self, conn):
        conn.execute("CREATE TABLE t (id INT, name VARCHAR, score DOUBLE)")
        cols = get_column_names(conn, "t")
        assert cols == {"id", "name", "score"}

    def test_view_columns(self, conn):
        conn.execute("CREATE VIEW v AS SELECT 1 AS a, 2 AS b")
        cols = get_column_names(conn, "v")
        assert cols == {"a", "b"}

    def test_nonexistent_table(self, conn):
        cols = get_column_names(conn, "nonexistent")
        assert cols == set()


class TestGetColumnSchema:
    def test_returns_ordered_columns_with_types(self, conn):
        conn.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")
        schema = get_column_schema(conn, "t")
        assert len(schema) == 2
        assert schema[0][0] == "id"
        assert schema[1][0] == "name"
        # Types should be present
        assert all(isinstance(s[1], str) for s in schema)

    def test_nonexistent_table(self, conn):
        assert get_column_schema(conn, "nonexistent") == []
