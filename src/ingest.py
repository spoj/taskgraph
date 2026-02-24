"""Ingestion module — load tabular data into DuckDB.

Accepts Polars DataFrames, list[dict] (array of structs), or dict[str, list]
(struct of arrays). All are coerced to DataFrame before writing.

Also supports file-based inputs (csv, parquet, excel, pdf).
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from pypdf import PdfReader, PdfWriter


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


@dataclass(frozen=True)
class LLMSource:
    """Multi-file LLM extraction source.

    Created via the ``llm()`` factory function::

        llm("data/a.pdf", "data/b.pdf")
        llm("data/a.pdf", prompt="Extract line items with amounts")

    ``files`` stores raw path strings before spec resolution and
    :class:`FileInput` objects after resolution by ``spec.py``.
    """

    files: tuple  # tuple[str, ...] pre-resolution, tuple[FileInput, ...] post
    prompt: str = PDF_EXTRACTION_PROMPT


def llm(*files: str, prompt: str = PDF_EXTRACTION_PROMPT) -> LLMSource:
    """Create a multi-file LLM extraction source.

    All positional arguments are file paths.  Use ``prompt`` to override
    the default extraction prompt.

    Examples::

        llm("data/a.pdf", "data/b.pdf")
        llm("data/report.pdf", prompt="Extract all invoices as rows")
    """
    if not files:
        raise ValueError("llm() requires at least one file path")
    for f in files:
        if not isinstance(f, str):
            raise TypeError(
                f"llm() file arguments must be strings, got {type(f).__name__}"
            )
    return LLMSource(files=files, prompt=prompt)


PDF_PAGES_EXTRACTION_PROMPT = (
    "Extract all tabular data from this page. Return a JSON object "
    'with a single key "rows" containing an array of objects where each object is '
    "one row and keys are column headers. Use consistent column names. "
    "Include every row visible on this page."
)


@dataclass(frozen=True)
class LLMPagesSource:
    """Per-page LLM extraction with 2/3 majority vote.

    Each page of each PDF is sent to the LLM independently, with two
    passes per page.  If the two passes agree (after normalizing
    whitespace via Polars DataFrame comparison), the result is accepted.
    If they disagree, a third pass breaks the tie (2/3 majority).
    If all three differ, the page errors.

    Created via the ``llm_pages()`` factory::

        llm_pages("data/jan.pdf", "data/feb.pdf")
        llm_pages("data/stmt.pdf", prompt="Extract transactions with date, description, amount")
    """

    files: tuple  # tuple[str, ...] pre-resolution, tuple[FileInput, ...] post
    prompt: str = PDF_PAGES_EXTRACTION_PROMPT


def llm_pages(*files: str, prompt: str = PDF_PAGES_EXTRACTION_PROMPT) -> LLMPagesSource:
    """Create a per-page LLM extraction source with majority-vote verification.

    All positional arguments are file paths.  Use ``prompt`` to override
    the default extraction prompt.

    Examples::

        llm_pages("data/jan_stmt.pdf", "data/feb_stmt.pdf")
        llm_pages("data/stmt.pdf", prompt="Extract all transactions as rows")
    """
    if not files:
        raise ValueError("llm_pages() requires at least one file path")
    for f in files:
        if not isinstance(f, str):
            raise TypeError(
                f"llm_pages() file arguments must be strings, got {type(f).__name__}"
            )
    return LLMPagesSource(files=files, prompt=prompt)


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


async def _extract_pdf_rows(
    path: Path,
    client: Any,
    prompt: str = PDF_EXTRACTION_PROMPT,
    model: str = GEMINI_PDF_MODEL,
) -> list[dict[str, Any]]:
    """Extract rows from a single PDF via LLM. Returns list[dict]."""
    _ensure_file_exists(path)
    data = path.read_bytes()
    if not data:
        raise RuntimeError(f"PDF file is empty: {path}")
    encoded = base64.b64encode(data).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
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
        raise RuntimeError(f"PDF extraction returned empty response for {path}")
    return _parse_pdf_json(text)


async def ingest_pdf(
    conn: duckdb.DuckDBPyConnection,
    path: Path,
    table_name: str,
    client: Any,
    prompt: str = PDF_EXTRACTION_PROMPT,
    model: str = GEMINI_PDF_MODEL,
) -> None:
    if client is None:
        raise RuntimeError(
            "PDF ingestion requires an OpenRouterClient (OPENROUTER_API_KEY)"
        )
    rows = await _extract_pdf_rows(path, client, prompt=prompt, model=model)
    ingest_table(conn, rows, table_name)


async def ingest_llm_source(
    conn: duckdb.DuckDBPyConnection,
    source: LLMSource,
    table_name: str,
    client: Any,
    model: str = GEMINI_PDF_MODEL,
) -> None:
    """Ingest one or more files via LLM extraction into a single table.

    Extracts rows from each file sequentially, concatenates all rows,
    and writes them as a single table.
    """
    if client is None:
        raise RuntimeError(
            "LLM source ingestion requires an OpenRouterClient (OPENROUTER_API_KEY)"
        )
    all_rows: list[dict[str, Any]] = []
    for file_input in source.files:
        if not isinstance(file_input, FileInput):
            raise RuntimeError(
                f"LLMSource file not resolved: {file_input!r} "
                "(expected FileInput after spec resolution)"
            )
        rows = await _extract_pdf_rows(
            file_input.path, client, prompt=source.prompt, model=model
        )
        all_rows.extend(rows)
    if not all_rows:
        raise RuntimeError("LLM source extraction returned no rows from any file")
    ingest_table(conn, all_rows, table_name)


# ---------------------------------------------------------------------------
# Per-page PDF extraction with majority vote
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def _split_pdf_pages(path: Path) -> list[bytes]:
    """Split a PDF file into individual single-page PDFs.

    Returns a list of bytes, one per page, each a valid standalone PDF.
    """
    _ensure_file_exists(path)
    raw = path.read_bytes()
    if not raw:
        raise RuntimeError(f"PDF file is empty: {path}")
    reader = PdfReader(io.BytesIO(raw))
    pages: list[bytes] = []
    for page in reader.pages:
        writer = PdfWriter()
        writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        pages.append(buf.getvalue())
    if not pages:
        raise RuntimeError(f"PDF has no pages: {path}")
    return pages


async def _extract_page_rows(
    client: Any,
    page_pdf: bytes,
    prompt: str,
    page_header: str,
    model: str = GEMINI_PDF_MODEL,
) -> list[dict[str, Any]]:
    """Send a single page PDF to the LLM and return extracted rows."""
    encoded = base64.b64encode(page_pdf).decode("ascii")
    full_prompt = f"{page_header}\n\n{prompt}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_prompt},
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
        return []
    return _parse_pdf_json(text)


def _normalize_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert rows to a Polars DataFrame with whitespace-stripped strings.

    Used for equality comparison between extraction passes.
    Returns an empty DataFrame (0 cols) for empty input.
    """
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    # Strip whitespace from all string columns
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).str.strip_chars().alias(col))
    return df


def _rows_match(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    """Return True if two row lists produce equal DataFrames after normalization."""
    df_a = _normalize_rows(a)
    df_b = _normalize_rows(b)
    if df_a.shape != df_b.shape:
        return False
    if df_a.is_empty() and df_b.is_empty():
        return True
    return df_a.equals(df_b)


async def _extract_page_with_majority(
    client: Any,
    page_pdf: bytes,
    prompt: str,
    page_header: str,
    model: str = GEMINI_PDF_MODEL,
) -> list[dict[str, Any]]:
    """Extract rows from a single page with 2/3 majority vote.

    Pass 1 and 2 run unconditionally.  If they agree, return immediately.
    Otherwise pass 3 runs as tiebreaker.  If pass 3 matches either
    pass 1 or pass 2, that result wins.  If all three disagree, raises
    RuntimeError.
    """
    pass1 = await _extract_page_rows(client, page_pdf, prompt, page_header, model)
    pass2 = await _extract_page_rows(client, page_pdf, prompt, page_header, model)

    if _rows_match(pass1, pass2):
        return pass1

    log.info("  %s: passes 1 and 2 disagree, running tiebreaker pass 3", page_header)
    pass3 = await _extract_page_rows(client, page_pdf, prompt, page_header, model)

    if _rows_match(pass1, pass3):
        return pass1
    if _rows_match(pass2, pass3):
        return pass2

    raise RuntimeError(
        f"All 3 extraction passes produced different results for {page_header}. "
        f"Pass 1: {len(pass1)} rows, Pass 2: {len(pass2)} rows, "
        f"Pass 3: {len(pass3)} rows."
    )


async def ingest_llm_pages(
    conn: duckdb.DuckDBPyConnection,
    source: LLMPagesSource,
    table_name: str,
    client: Any,
    model: str = GEMINI_PDF_MODEL,
) -> None:
    """Ingest PDFs page-by-page with majority-vote verification.

    For each page of each PDF:
    1. Extract the page as a standalone single-page PDF.
    2. Run two LLM extraction passes.
    3. If results match (normalized), accept.  Otherwise run a third
       pass and take the 2/3 majority.
    4. Concatenate all accepted page results into a single table.
    """
    if client is None:
        raise RuntimeError(
            "LLM pages ingestion requires an OpenRouterClient (OPENROUTER_API_KEY)"
        )

    all_rows: list[dict[str, Any]] = []

    for file_input in source.files:
        if not isinstance(file_input, FileInput):
            raise RuntimeError(
                f"LLMPagesSource file not resolved: {file_input!r} "
                "(expected FileInput after spec resolution)"
            )
        filename = file_input.path.name
        pages = _split_pdf_pages(file_input.path)
        total_pages = len(pages)
        log.info("  %s: %d page(s)", filename, total_pages)

        for page_idx, page_pdf in enumerate(pages):
            page_num = page_idx + 1
            page_header = (
                f"=== page {page_num} of {total_pages} from file {filename} ==="
            )
            rows = await _extract_page_with_majority(
                client, page_pdf, source.prompt, page_header, model
            )
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("LLM pages extraction returned no rows from any page")
    ingest_table(conn, all_rows, table_name)


async def ingest_file(
    conn: duckdb.DuckDBPyConnection,
    file_input: FileInput,
    table_name: str,
    client: Any | None = None,
) -> None:
    fmt = file_input.format
    if fmt in ("csv", "parquet"):
        reader = "read_csv_auto" if fmt == "csv" else "read_parquet"
        _ingest_single_file(conn, file_input.path, table_name, reader)
    elif fmt == "excel":
        ingest_excel(conn, file_input.path, table_name, sheet=file_input.sheet)
    elif fmt == "pdf":
        await ingest_pdf(conn, file_input.path, table_name, client=client)
    else:
        raise ValueError(f"Unsupported file format: {fmt}")
