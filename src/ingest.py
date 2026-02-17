"""Ingestion module — load tabular data into DuckDB.

Accepts Polars DataFrames, list[dict] (array of structs), or dict[str, list]
(struct of arrays). All are coerced to DataFrame before writing.

Also supports file-based inputs (csv, parquet, excel, pdf).
"""

from __future__ import annotations

import base64
import json
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
    ".pdf": "pdf",
}

PDF_EXTRACTION_PROMPT = (
    "Extract all tabular data from this PDF document. Return a JSON object with "
    'a single key "rows" containing an array of objects where each object is one '
    "row and keys are column headers. Use consistent column names."
)

PDF_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}

GEMINI_PDF_MODEL = "google/gemini-3-flash-preview"


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


def ingest_csv(conn: duckdb.DuckDBPyConnection, path: Path, table_name: str) -> None:
    _ensure_file_exists(path)
    query = (
        "SELECT CAST(row_number() OVER () AS INTEGER) AS _row_id, * "
        "FROM read_csv_auto(?)"
    )
    _write_table_from_query(conn, table_name, query, [str(path)])


def ingest_parquet(
    conn: duckdb.DuckDBPyConnection, path: Path, table_name: str
) -> None:
    _ensure_file_exists(path)
    query = (
        "SELECT CAST(row_number() OVER () AS INTEGER) AS _row_id, * "
        "FROM read_parquet(?)"
    )
    _write_table_from_query(conn, table_name, query, [str(path)])


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
            "FROM read_xlsx(?, sheet = ?, header = true)"
        )
        params = [str(path), sheet]
    else:
        query = (
            "SELECT CAST(row_number() OVER () AS INTEGER) AS _row_id, * "
            "FROM read_xlsx(?, header = true)"
        )
        params = [str(path)]
    _write_table_from_query(conn, table_name, query, params)


def _extract_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _parse_pdf_json(text: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[:200].replace("\n", " ")
        raise RuntimeError(f"PDF extraction returned invalid JSON: {snippet}") from e
    # Unwrap {"rows": [...]} envelope if present
    if isinstance(parsed, dict):
        rows = parsed.get("rows")
        if rows is None:
            raise RuntimeError(
                "PDF extraction returned object without 'rows' key; "
                f"got keys: {list(parsed.keys())}"
            )
        parsed = rows
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("PDF extraction returned empty or non-list JSON")
    for item in parsed:
        if not isinstance(item, dict):
            raise RuntimeError("PDF extraction JSON must be a list of objects")
    return parsed


async def ingest_pdf(
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    table_name: str,
    client: Any,
    model: str = GEMINI_PDF_MODEL,
) -> None:
    if client is None:
        raise RuntimeError(
            "PDF ingestion requires an OpenRouterClient (OPENROUTER_API_KEY)"
        )
    _ensure_file_exists(path)
    data = path.read_bytes()
    if not data:
        raise RuntimeError(f"PDF file is empty: {path}")
    encoded = base64.b64encode(data).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PDF_EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:application/pdf;base64,{encoded}"},
                },
            ],
        }
    ]

    response = await client.chat(
        model=model,
        messages=messages,
        response_format=PDF_RESPONSE_FORMAT,
    )
    message = response.get("message", {})
    text = _extract_message_text(message.get("content"))
    if not text.strip():
        raise RuntimeError("PDF extraction returned empty response")
    rows = _parse_pdf_json(text)
    ingest_table(conn, rows, table_name)


async def ingest_file(
    conn: duckdb.DuckDBPyConnection,
    file_input: FileInput,
    table_name: str,
    client: Any | None = None,
) -> None:
    if file_input.format == "csv":
        ingest_csv(conn, file_input.path, table_name)
        return
    if file_input.format == "parquet":
        ingest_parquet(conn, file_input.path, table_name)
        return
    if file_input.format == "excel":
        ingest_excel(conn, file_input.path, table_name, sheet=file_input.sheet)
        return
    if file_input.format == "pdf":
        await ingest_pdf(conn, file_input.path, table_name, client=client)
        return
    raise ValueError(f"Unsupported file format: {file_input.format}")
