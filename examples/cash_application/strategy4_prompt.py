"""Strategy 4: Taskgraph Pure Prompt (LLM does all matching).

Architecture: Source nodes ingest data, then a single prompt node does all
matching logic via SQL tool calls. A second prompt node identifies unmatched
payments and unapplied invoices.

DAG:
    invoices, payments, remittance_lines
                    |
              match_all            (prompt: full matching)
                    |
              report               (prompt: unmatched/unapplied classification)

Design principles:
  - Minimal SQL preprocessing — LLM decides all matching strategy
  - LLM has full freedom to use any DuckDB SQL features
  - Tests whether an LLM can replicate expert domain logic from scratch
"""

import json
from datetime import datetime
from pathlib import Path

# ── Data Loaders ──────────────────────────────────────────────────────────────

_DATA_PATH = Path(__file__).parent / "problems" / "n100_seed42.json"


def _load_data():
    data = json.loads(_DATA_PATH.read_text())
    return data


def load_invoices():
    data = _load_data()
    for row in data["invoices"]:
        row["invoice_date"] = datetime.strptime(row["invoice_date"], "%Y-%m-%d").date()
        row["due_date"] = datetime.strptime(row["due_date"], "%Y-%m-%d").date()
    return data["invoices"]


def load_payments():
    data = _load_data()
    for row in data["payments"]:
        row["payment_date"] = datetime.strptime(row["payment_date"], "%Y-%m-%d").date()
    return data["payments"]


def load_remittance():
    data = _load_data()
    return data["remittance_lines"]


# ── Prompt ────────────────────────────────────────────────────────────────────

MATCH_PROMPT = """\
You are a cash application specialist. Your task is to match incoming customer
payments to outstanding invoices using SQL queries against a DuckDB database.

TABLES AVAILABLE:
  invoices:          invoice_id, customer_id, customer_name, invoice_date, due_date, amount, description
  payments:          payment_id, payment_date, payer_name, amount, method, reference_info
  remittance_lines:  remittance_id, payment_id, invoice_ref, amount, memo

TASK: Create a view called `match_all_applications` with columns:
  payment_id VARCHAR, invoice_id VARCHAR, applied_amount DOUBLE

This view should contain all payment-to-invoice matches you can identify.

MATCHING SCENARIOS TO HANDLE:
1. SIMPLE 1:1 — Payment exactly matches one invoice. Remittance has correct ref.
2. MULTI-INVOICE — One payment covers 2-4 invoices (sum of invoice amounts = payment).
   Remittance may list multiple invoice refs.
3. PARTIAL PAYMENT — Payment is 25-80% of invoice amount. Customer pays in installments.
4. DISCOUNT TAKEN — Payment is 97-99.5% of invoice (early payment discount).
5. SHORT-PAY — Payment is 85-97% of invoice (deduction for dispute/damage/etc).
6. NO REMITTANCE — Payment has no remittance lines; match by amount + customer name.
7. CROSS-REFERENCE — Remittance has garbled/wrong invoice ref (typos, PO numbers).
   Extract numeric part and fuzzy-match.
8. OVERPAYMENT — Payment exceeds invoice amount by 2-15%. Applied amount = invoice amount.
9. CREDIT MEMO — Negative invoice offsets positive invoice. Payment = net amount.
   Look for pairs of positive + negative invoices from same customer.
10. DUPLICATE PAYMENT — Same invoice paid twice. Only first payment applies.

HINTS:
- Customer names on payments (payer_name) may differ from invoice customer_name.
  Use jaro_winkler_similarity() > 0.80 or starts_with() for fuzzy matching.
  Normalize names: UPPER, strip legal suffixes (INC, LLC, LTD, CORP, CO, etc).
- Invoice refs in remittance may be garbled: "INV-00042" might appear as
  "Invoice #42", "INV-00024" (transposed), or "PO-12345" (unrelated).
  Extract the numeric part with regexp_extract and compare.
- For multi-invoice payments, group remittance lines by payment_id and try
  to match each line's amount to an invoice from the same customer.
- applied_amount rules:
  * Simple/no-remittance: applied_amount = invoice amount
  * Partial/discount/short-pay: applied_amount = payment amount (the lesser)
  * Overpayment: applied_amount = invoice amount (not the overpayment)
  * Credit memo: applied_amount = invoice amount (positive for invoice, negative for credit)
  * Multi-invoice: applied_amount per invoice = that invoice's amount
    (unless partial on last invoice, then = remainder)

RULES:
- Each invoice should be matched at most once.
- A payment can match multiple invoices (multi-invoice scenario).
- Be conservative: false positives are worse than missed matches.
- Start with high-confidence matches, then work your way to fuzzy/uncertain ones.

DuckDB FEATURES:
- jaro_winkler_similarity(s1, s2) for fuzzy string matching
- regexp_extract(str, pattern, group) for regex extraction
- starts_with(str, prefix) for prefix matching
- QUALIFY for window function filtering
- GROUP BY ALL for convenience
"""

REPORT_PROMPT = """\
Create views for unmatched payments and unapplied invoices based on the matching
results in `match_all_applications`.

Create these views:
1. `report_unmatched_payments` with column: payment_id
   — Payments that don't appear in match_all_applications.

2. `report_unapplied_invoices` with column: invoice_id
   — Invoices that don't appear in match_all_applications.

3. `report_results` with columns: payment_id, invoice_id, applied_amount
   — Copy of match_all_applications (for scoring compatibility).

Use simple LEFT JOIN / NOT IN patterns. These are straightforward set operations.
"""


# ── NODES ─────────────────────────────────────────────────────────────────────

NODES = [
    {
        "name": "invoices",
        "source": load_invoices,
        "columns": [
            "invoice_id",
            "customer_id",
            "customer_name",
            "invoice_date",
            "due_date",
            "amount",
            "description",
        ],
    },
    {
        "name": "payments",
        "source": load_payments,
        "columns": [
            "payment_id",
            "payment_date",
            "payer_name",
            "amount",
            "method",
            "reference_info",
        ],
    },
    {
        "name": "remittance_lines",
        "source": load_remittance,
        "columns": ["remittance_id", "payment_id", "invoice_ref", "amount", "memo"],
    },
    {
        "name": "match_all",
        "prompt": MATCH_PROMPT,
        "depends_on": ["invoices", "payments", "remittance_lines"],
        "output_columns": {
            "match_all_applications": [
                "payment_id",
                "invoice_id",
                "applied_amount",
            ],
        },
    },
    {
        "name": "report",
        "prompt": REPORT_PROMPT,
        "depends_on": ["invoices", "payments", "match_all"],
        "output_columns": {
            "report_results": ["payment_id", "invoice_id", "applied_amount"],
            "report_unmatched_payments": ["payment_id"],
            "report_unapplied_invoices": ["invoice_id"],
        },
    },
]
