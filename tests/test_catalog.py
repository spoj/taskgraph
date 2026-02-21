"""Tests for src/catalog.py — DuckDB catalog helpers."""

from src.catalog import (
    count_rows,
    count_rows_display,
    list_tables,
    list_views,
    list_views_with_sql,
    quote_ident,
    view_exists,
)


class TestQuoteIdent:
    def test_simple_name(self):
        assert quote_ident("foo") == '"foo"'

    def test_name_with_quotes(self):
        assert quote_ident('my"table') == '"my""table"'

    def test_empty_string(self):
        assert quote_ident("") == '""'


class TestListViews:
    def test_empty_db(self, conn):
        # infra creates _view_definitions, so filter it out
        views = list_views(conn, exclude_prefixes=("_",))
        assert views == []

    def test_returns_user_views(self, conn):
        conn.execute("CREATE VIEW alpha AS SELECT 1")
        conn.execute("CREATE VIEW beta AS SELECT 2")
        views = list_views(conn, exclude_prefixes=("_",))
        assert views == ["alpha", "beta"]  # sorted

    def test_exclude_prefixes(self, conn):
        conn.execute("CREATE VIEW keep_me AS SELECT 1")
        conn.execute("CREATE VIEW tmp_discard AS SELECT 2")
        views = list_views(conn, exclude_prefixes=("tmp_", "_"))
        assert views == ["keep_me"]


class TestListViewsWithSql:
    def test_returns_name_and_sql(self, conn):
        conn.execute("CREATE VIEW v AS SELECT 42 AS x")
        rows = list_views_with_sql(conn, exclude_prefixes=("_",))
        assert len(rows) == 1
        assert rows[0][0] == "v"
        assert "42" in rows[0][1]


class TestViewExists:
    def test_exists(self, conn):
        conn.execute("CREATE VIEW v AS SELECT 1")
        assert view_exists(conn, "v") is True

    def test_not_exists(self, conn):
        assert view_exists(conn, "nonexistent") is False


class TestListTables:
    def test_returns_user_tables(self, conn):
        conn.execute("CREATE TABLE t1 (id INT)")
        conn.execute("CREATE TABLE t2 (id INT)")
        tables = list_tables(conn, exclude_prefixes=("_",))
        assert tables == ["t1", "t2"]

    def test_exclude_prefixes(self, conn):
        conn.execute("CREATE TABLE keep (id INT)")
        conn.execute("CREATE TABLE _internal (id INT)")
        tables = list_tables(conn, exclude_prefixes=("_",))
        assert tables == ["keep"]


class TestCountRows:
    def test_counts_table(self, conn):
        conn.execute("CREATE TABLE t AS SELECT * FROM range(5) r(x)")
        assert count_rows(conn, "t") == 5

    def test_counts_view(self, conn):
        conn.execute("CREATE VIEW v AS SELECT 1 UNION ALL SELECT 2")
        assert count_rows(conn, "v") == 2

    def test_nonexistent_returns_none(self, conn):
        assert count_rows(conn, "nonexistent") is None

    def test_empty_table(self, conn):
        conn.execute("CREATE TABLE t (id INT)")
        assert count_rows(conn, "t") == 0


class TestCountRowsDisplay:
    def test_normal(self, conn):
        conn.execute("CREATE TABLE t AS SELECT 1 AS x")
        assert count_rows_display(conn, "t") == "1"

    def test_error(self, conn):
        assert count_rows_display(conn, "nonexistent") == "error"
