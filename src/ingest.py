"""Ingestion module — load tabular data into DuckDB.

Accepts Polars DataFrames, list[dict] (array of structs), or dict[str, list]
(struct of arrays). All are coerced to DataFrame before writing.

Also supports file-based inputs (csv, parquet, excel).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl


# Type alias for data the user can return from input functions
TableData = Any  # pl.DataFrame | list[dict] | dict[str, list]


SUPPORTED_FILE_EXTENSIONS = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
}

@dataclass(frozen=True)
class FileInput:
    path: Path
    format: str
    sheet: str | None = None


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


def is_supported_file_string(value: str) -> bool:
    """Return True if value looks like a supported file path."""
    raw = value.rsplit("#", 1)[0]
    suffix = Path(raw).suffix.lower()
    return suffix in SUPPORTED_FILE_EXTENSIONS


def _normalize_path(path: Path, base_dir: Path | None = None) -> Path:
    """Resolve path against base_dir (or cwd) and expand user/symlinks."""
    if not path.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        path = base_dir / path
    return path.expanduser().resolve()


def parse_file_string(value: str, base_dir: Path | None = None) -> FileInput:
    """Parse a file input string into a FileInput.

    Supports Excel sheet fragments: "file.xlsx#Sheet1".
    """
    if not value or not value.strip():
        raise ValueError("File path must be a non-empty string")

    raw = value.strip()
    sheet: str | None = None
    if "#" in raw:
        path_part, sheet_part = raw.rsplit("#", 1)
        if not path_part:
            raise ValueError("File path must precede '#'")
        if not sheet_part:
            raise ValueError("Excel sheet name must follow '#' fragment")
        raw = path_part
        sheet = sheet_part

    path = _normalize_path(Path(raw), base_dir=base_dir)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")
    fmt = SUPPORTED_FILE_EXTENSIONS[suffix]
    if sheet and fmt != "excel":
        raise ValueError("Sheet fragments are only supported for Excel files")
    return FileInput(path=path, format=fmt, sheet=sheet)


def parse_file_path(path: Path, base_dir: Path | None = None) -> FileInput:
    """Parse a Path into a FileInput (no sheet support)."""
    normalized = _normalize_path(path, base_dir=base_dir)
    suffix = normalized.suffix.lower()
    if suffix not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")
    fmt = SUPPORTED_FILE_EXTENSIONS[suffix]
    return FileInput(path=normalized, format=fmt)


def _write_table(
    conn: duckdb.DuckDBPyConnection, df: pl.DataFrame, table_name: str
) -> None:
    """Write a DataFrame to DuckDB with _row_id INTEGER primary key.

    Uses DuckDB's native DataFrame scan for bulk ingestion.
    """
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.register("_df", df)
    try:
        conn.execute(
            f'CREATE TABLE "{table_name}" AS '
            'SELECT CAST(row_number() OVER () AS INTEGER) AS "_row_id", * '
            "FROM _df"
        )
    finally:
        conn.unregister("_df")


def _write_table_from_query(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    query: str,
    params: list[Any] | None = None,
) -> None:
    """Write query results to a DuckDB table with a given name.

    If params is provided, executes a parameterized query.
    """
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    if params is None:
        conn.execute(f'CREATE TABLE "{table_name}" AS {query}')
    else:
        conn.execute(f'CREATE TABLE "{table_name}" AS {query}', params)


def ingest_table(
    conn: duckdb.DuckDBPyConnection, data: TableData, table_name: str
) -> None:
    """Ingest tabular data into the database as a named table.

    Accepts DataFrame, list[dict], or dict[str, list].
    Creates a table with _row_id INTEGER primary key.
    """
    df = coerce_to_dataframe(data)
    _write_table(conn, df, table_name)


def _ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _ingest_single_file(
    conn: duckdb.DuckDBPyConnection, path: Path, table_name: str, reader_fn: str
) -> None:
    """Ingest a single-file format (csv or parquet) using a DuckDB reader function."""
    _ensure_file_exists(path)
    query = (
        "SELECT CAST(row_number() OVER () AS INTEGER) AS _row_id, * "
        f"FROM {reader_fn}(?)"
    )
    _write_table_from_query(conn, table_name, query, [str(path)])


# Public aliases for direct use / tests
def ingest_csv(conn: duckdb.DuckDBPyConnection, path: Path, table_name: str) -> None:
    _ingest_single_file(conn, path, table_name, "read_csv_auto")


def ingest_parquet(
    conn: duckdb.DuckDBPyConnection, path: Path, table_name: str
) -> None:
    _ingest_single_file(conn, path, table_name, "read_parquet")


def ingest_excel(
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    table_name: str,
    sheet: str | None = None,
) -> None:
    _ensure_file_exists(path)
    try:
        conn.execute("LOAD excel")
    except duckdb.Error as e:
        raise RuntimeError(f"Failed to load DuckDB excel extension: {e}") from e

    if sheet:
        query = (
            "SELECT CAST(row_number() OVER () AS INTEGER) AS _row_id, * "
            "FROM read_xlsx(?, sheet = ?, header = false)"
        )
        params = [str(path), sheet]
    else:
        query = (
            "SELECT CAST(row_number() OVER () AS INTEGER) AS _row_id, * "
            "FROM read_xlsx(?, header = false)"
        )
        params = [str(path)]
    _write_table_from_query(conn, table_name, query, params)


def ingest_file(
    conn: duckdb.DuckDBPyConnection,
    file_input: FileInput,
    table_name: str,
) -> None:
    fmt = file_input.format
    if fmt in ("csv", "parquet"):
        reader = "read_csv_auto" if fmt == "csv" else "read_parquet"
        _ingest_single_file(conn, file_input.path, table_name, reader)
    elif fmt == "excel":
        ingest_excel(conn, file_input.path, table_name, sheet=file_input.sheet)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")

