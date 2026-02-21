import json
from datetime import datetime
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "dataset.json"


def _load_data():
    return json.loads(_DATA_PATH.read_text())


def load_bank():
    data = _load_data()["bank_transactions"]
    for row in data:
        row["date"] = datetime.strptime(row["date"], "%Y-%m-%d").date()
    return data


def load_gl():
    data = _load_data()["gl_entries"]
    for row in data:
        row["date"] = datetime.strptime(row["date"], "%Y-%m-%d").date()
    return data


PROMPT = """You are a bank reconciliation expert.
Write SQL to match `bank_txns` to `gl_entries`.

Create the view `match_residual_all_matched` with columns: bank_id, gl_id.

Rules for matching:
- Same amount and dates within 5 days.
- Use simple fuzzy matching on descriptions (e.g. UPPER(bank.description) LIKE '%' || UPPER(gl.description) || '%') where possible.
- Try to find batch deposits (up to 3 gl entries) by self-joining gl_entries.
- Avoid duplicates: a bank or GL entry should only be matched once.

IMPORTANT: You must produce ONLY standard SQL. Use duckdb dialect.
"""

NODES = [
    {"name": "bank_txns", "source": load_bank},
    {"name": "gl_entries", "source": load_gl},
    {
        "name": "match_residual",
        "prompt": PROMPT,
        "depends_on": ["bank_txns", "gl_entries"],
        "output_columns": {"match_residual_all_matched": ["bank_id", "gl_id"]},
    },
    {
        "name": "offsetting",
        "prompt": "Create view `offsetting_pairs` with columns `original_id`, `reversal_id` for pairs of gl_entries with exact opposite amounts and same vendor/close dates.",
        "depends_on": ["gl_entries"],
        "output_columns": {"offsetting_pairs": ["original_id", "reversal_id"]},
    },
]
