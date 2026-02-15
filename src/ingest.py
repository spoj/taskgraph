"""Ingestion module — load tabular data into DuckDB.

Accepts Polars DataFrames, list[dict] (array of structs), or dict[str, list]
(struct of arrays). All are coerced to DataFrame before writing.
"""

import duckdb
from typing import Any

import polars as pl


# Type alias for data the user can return from input functions
TableData = Any  # pl.DataFrame | list[dict] | dict[str, list]


def coerce_to_dataframe(data: TableData) -> pl.DataFrame:
    """Convert supported tabular formats to a Polars DataFrame.

    Accepted formats:
    - pl.DataFrame: returned as-is
    - list[dict]: array of structs, e.g. [{"a": 1, "b": 2}, ...]
    - dict[str, list]: struct of arrays, e.g. {"a": [1, 2], "b": [3, 4]}

    Raises TypeError for unsupported formats.
    """
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, list):
        return pl.DataFrame(data)
    if isinstance(data, dict):
        return pl.DataFrame(data)
    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. "
        f"Expected DataFrame, list[dict], or dict[str, list]."
    )


def _write_table(
    conn: duckdb.DuckDBPyConnection, df: pl.DataFrame, table_name: str
) -> None:
    """Write a DataFrame to DuckDB with _row_id INTEGER primary key.

    Uses DuckDB's native DataFrame scan for bulk ingestion.
    """
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(
        f'CREATE TABLE "{table_name}" AS '
        f'SELECT CAST(row_number() OVER () AS INTEGER) AS "_row_id", * FROM df'
    )


def ingest_table(
    conn: duckdb.DuckDBPyConnection, data: TableData, table_name: str
) -> None:
    """Ingest tabular data into the database as a named table.

    Accepts DataFrame, list[dict], or dict[str, list].
    Creates a table with _row_id INTEGER primary key.
    """
    df = coerce_to_dataframe(data)
    _write_table(conn, df, table_name)


def get_schema_info_for_tables(
    conn: duckdb.DuckDBPyConnection, table_names: list[str]
) -> str:
    """Get schema information for a list of tables/views.

    Returns a formatted string describing schemas and sample data,
    suitable for inclusion in the LLM prompt.
    """
    parts = []

    for name in table_names:
        cols = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [name],
        ).fetchall()

        if not cols:
            parts.append(f"Table: {name}  (not found)")
            parts.append("")
            continue

        # Exclude internal _row_id from displayed columns
        data_cols = [c for c in cols if c[0] != "_row_id"]
        col_info = ", ".join(f"{c[0]} ({c[1]})" for c in data_cols)
        col_names = [c[0] for c in data_cols]

        row = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()
        count = row[0] if row else 0

        col_list = ", ".join(f'"{c}"' for c in col_names)
        sample = conn.execute(f'SELECT {col_list} FROM "{name}" LIMIT 3').fetchall()

        parts.append(f"Table: {name}")
        parts.append(f"  Columns: {col_info}")
        parts.append(f"  Row count: {count}")
        parts.append(f"  Sample rows ({', '.join(col_names)}): {sample}")
        parts.append("")

    return "\n".join(parts)
