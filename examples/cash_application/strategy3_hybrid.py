"""Strategy 3: Taskgraph Hybrid (SQL preprocessing + LLM residual matching).

Architecture: SQL handles structurally unambiguous matches (exact refs, exact
amounts, ref+partial); LLM handles everything requiring judgment (discounts,
short-pays, credit memos, multi-invoice grouping, fuzzy entity resolution).

DAG:
    invoices, payments, remittance_lines
                    |
                features           (sql: name normalization, ref extraction)
                    |
              match_exact          (sql: exact ref + exact amount, ref+partial, amount+customer)
                   |
              match_residual       (prompt: discounts, short-pays, credit memos, multi-invoice, fuzzy)
                   |
                report             (sql: consolidate results + unmatched/unapplied)

Design principles:
  - SQL nodes handle only structurally unambiguous matches (exact ref lookups,
    exact amount equality, ref match with partial amount)
  - LLM handles all pattern-based judgment calls: discount detection, short-pay,
    overpayment, credit memos, multi-invoice grouping, partial payments, fallback
  - No hardcoded thresholds calibrated to specific data generators
  - No hardcoded customer name lists — entity resolution via normalization + similarity
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


# ── Node 1: Features (sql) ───────────────────────────────────────────────────

FEATURES_SQL = """\
-- Normalize invoice data for matching
CREATE VIEW features_inv AS
SELECT
    invoice_id,
    customer_id,
    customer_name,
    invoice_date,
    due_date,
    amount,
    description,
    -- Normalized customer name (uppercase, strip legal suffixes, collapse spaces)
    trim(regexp_replace(
        regexp_replace(
            regexp_replace(upper(customer_name),
                '(^|\\s)(INCORPORATED|INC\\.?|LLC|LTD\\.?|CORP\\.?|CO\\.?|COMPANY|GROUP|GRP|ASSOCIATES|ASSOC|LP|LLP|PLC)(\\s|$)',
                ' ', 'g'),
            '[^A-Z0-9 ]', ' ', 'g'),
        '\\s+', ' ', 'g'
    )) AS customer_norm,
    -- Extract numeric part from invoice_id (e.g., INV-00042 -> 42)
    TRY_CAST(regexp_extract(invoice_id, 'INV-(\\d+)', 1) AS INTEGER) AS inv_number
FROM invoices;

-- Normalize payment data for matching
CREATE VIEW features_pmt AS
SELECT
    payment_id,
    payment_date,
    payer_name,
    amount,
    method,
    reference_info,
    -- Normalized payer name
    trim(regexp_replace(
        regexp_replace(
            regexp_replace(upper(payer_name),
                '(^|\\s)(INCORPORATED|INC\\.?|LLC|LTD\\.?|CORP\\.?|CO\\.?|COMPANY|GROUP|GRP|ASSOCIATES|ASSOC|LP|LLP|PLC)(\\s|$)',
                ' ', 'g'),
            '[^A-Z0-9 ]', ' ', 'g'),
        '\\s+', ' ', 'g'
    )) AS payer_norm
FROM payments;

-- Normalize remittance data
CREATE VIEW features_rem AS
SELECT
    remittance_id,
    payment_id,
    invoice_ref,
    amount,
    memo,
    -- Try to extract invoice number from garbled refs
    TRY_CAST(regexp_extract(invoice_ref, '(?:INV[- ]?|Invoice\\s*#?\\s*)(\\d+)', 1) AS INTEGER) AS ref_inv_number,
    -- Direct match attempt
    CASE
        WHEN invoice_ref LIKE 'INV-%' THEN invoice_ref
        ELSE NULL
    END AS ref_direct,
    -- Partial payment hint from memo
    CASE
        WHEN lower(memo) LIKE '%partial%'
          OR lower(memo) LIKE '%installment%'
          OR lower(memo) LIKE '%progress%'
        THEN true ELSE false
    END AS has_partial_hint
FROM remittance_lines;
"""


# ── Node 2: Exact Matches (sql) ──────────────────────────────────────────────

MATCH_EXACT_SQL = """\
-- Round 1: Exact remittance ref + exact amount
-- A payment may legitimately match multiple invoices (multi-invoice payment with
-- multiple remittance lines), so we do NOT enforce 1:1 on payments here.
-- But each invoice can only be matched once — take the first payment by date.
CREATE VIEW match_exact_r1 AS
WITH candidates AS (
    SELECT
        r.payment_id,
        i.invoice_id,
        i.amount AS inv_amount,
        ROW_NUMBER() OVER (PARTITION BY i.invoice_id ORDER BY r.payment_id) AS inv_rnk
    FROM features_rem r
    JOIN features_inv i
      ON r.ref_direct = i.invoice_id
     AND ABS(r.amount - i.amount) < 0.01
)
SELECT payment_id, invoice_id, inv_amount AS applied_amount,
       'exact_ref' AS match_type
FROM candidates
WHERE inv_rnk = 1;

-- Round 2: Remittance ref number match (handles garbled prefixes like "Invoice #42")
-- NOTE: Do NOT exclude payments already in R1 — a multi-invoice payment may have
-- some lines matched in R1 and others here. Only exclude already-matched invoices.
CREATE VIEW match_exact_r2 AS
WITH already_matched_inv AS (
    SELECT invoice_id FROM match_exact_r1
),
candidates AS (
    SELECT
        r.payment_id,
        i.invoice_id,
        i.amount AS inv_amount,
        ABS(r.ref_inv_number - i.inv_number) AS num_diff
    FROM features_rem r
    JOIN features_inv i
      ON r.ref_inv_number IS NOT NULL
     AND i.inv_number IS NOT NULL
     AND ABS(r.ref_inv_number - i.inv_number) <= 1  -- allow off-by-one for transpositions
     AND ABS(r.amount - i.amount) < 0.01
    WHERE i.invoice_id NOT IN (SELECT invoice_id FROM already_matched_inv)
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY num_diff, payment_id) AS inv_rnk
    FROM candidates
)
SELECT payment_id, invoice_id, inv_amount AS applied_amount,
       'fuzzy_ref' AS match_type
FROM ranked WHERE inv_rnk = 1;

-- Round 3: Ref match + partial amount (ref is correct but amount is a fraction
-- of the invoice — e.g., partial payments, installments).
-- Only exact ref matches (ref_direct or exact inv_number — NO off-by-one) to stay safe.
-- Amount ratio 0.20–1.005 covers partial through exact amounts.
CREATE VIEW match_exact_r3 AS
WITH already_matched_inv AS (
    SELECT invoice_id FROM match_exact_r1
    UNION ALL SELECT invoice_id FROM match_exact_r2
),
already_matched_pairs AS (
    SELECT payment_id, invoice_id FROM match_exact_r1
    UNION ALL SELECT payment_id, invoice_id FROM match_exact_r2
),
candidates AS (
    SELECT
        r.payment_id,
        i.invoice_id,
        r.amount AS applied_amount,  -- use remittance amount (the actual partial)
        i.amount AS inv_amount,
        r.amount / i.amount AS ratio
    FROM features_rem r
    JOIN features_inv i
      ON (r.ref_direct = i.invoice_id
          OR (r.ref_inv_number IS NOT NULL AND i.inv_number IS NOT NULL
              AND r.ref_inv_number = i.inv_number))
     AND i.amount > 0
     AND r.amount / i.amount BETWEEN 0.20 AND 1.005
    WHERE i.invoice_id NOT IN (SELECT invoice_id FROM already_matched_inv)
      -- Don't re-create pairs that already exist in R1/R2
      AND NOT EXISTS (
          SELECT 1 FROM already_matched_pairs ap
          WHERE ap.payment_id = r.payment_id AND ap.invoice_id = i.invoice_id
      )
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY ratio DESC, payment_id) AS inv_rnk
    FROM candidates
)
SELECT payment_id, invoice_id, applied_amount,
       'ref_partial' AS match_type
FROM ranked WHERE inv_rnk = 1;

-- Round 4: Exact amount + exact customer name (for no-remittance payments)
CREATE VIEW match_exact_r4 AS
WITH already_matched_inv AS (
    SELECT invoice_id FROM match_exact_r1
    UNION ALL SELECT invoice_id FROM match_exact_r2
    UNION ALL SELECT invoice_id FROM match_exact_r3
),
already_matched_pmt AS (
    SELECT DISTINCT payment_id FROM match_exact_r1
    UNION ALL SELECT DISTINCT payment_id FROM match_exact_r2
    UNION ALL SELECT DISTINCT payment_id FROM match_exact_r3
),
candidates AS (
    SELECT
        p.payment_id,
        i.invoice_id,
        i.amount AS applied_amount,
        ABS(date_diff('day', p.payment_date, i.invoice_date)) AS date_gap,
        jaro_winkler_similarity(p.payer_norm, i.customer_norm) AS name_sim
    FROM features_pmt p
    JOIN features_inv i
      ON ABS(p.amount - i.amount) < 0.01
     AND (
         p.payer_norm = i.customer_norm
         OR jaro_winkler_similarity(p.payer_norm, i.customer_norm) > 0.85
         OR starts_with(i.customer_norm, p.payer_norm)
         OR starts_with(p.payer_norm, i.customer_norm)
     )
    WHERE i.invoice_id NOT IN (SELECT invoice_id FROM already_matched_inv)
      AND p.payment_id NOT IN (SELECT payment_id FROM already_matched_pmt)
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY payment_id ORDER BY name_sim DESC, date_gap ASC) AS p_rnk,
        ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY name_sim DESC, date_gap ASC) AS i_rnk
    FROM candidates
)
SELECT payment_id, invoice_id, applied_amount,
       'exact_amount_customer' AS match_type
FROM ranked
WHERE p_rnk = 1 AND i_rnk = 1;

-- Combined exact matches
CREATE VIEW match_exact_all AS
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_exact_r1
UNION ALL
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_exact_r2
UNION ALL
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_exact_r3
UNION ALL
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_exact_r4;

-- Remaining pools after exact matching
CREATE VIEW match_exact_remaining_pmt AS
SELECT p.* FROM features_pmt p
LEFT JOIN (SELECT DISTINCT payment_id FROM match_exact_all) m ON p.payment_id = m.payment_id
WHERE m.payment_id IS NULL;

CREATE VIEW match_exact_remaining_inv AS
SELECT i.* FROM features_inv i
LEFT JOIN (SELECT DISTINCT invoice_id FROM match_exact_all) m ON i.invoice_id = m.invoice_id
WHERE m.invoice_id IS NULL;

CREATE VIEW match_exact_remaining_rem AS
SELECT r.* FROM features_rem r
WHERE r.payment_id IN (SELECT payment_id FROM match_exact_remaining_pmt);
"""

MATCH_EXACT_VALIDATE = """\
SELECT 'fail' AS status,
       'invoice ' || invoice_id || ' matched ' || cnt || 'x in exact' AS message
FROM (SELECT invoice_id, COUNT(*) AS cnt FROM match_exact_all GROUP BY invoice_id)
WHERE cnt > 1;
"""


# ── Node 3: Residual LLM Matching (prompt) ───────────────────────────────────

MATCH_RESIDUAL_PROMPT = """\
Resolve remaining unmatched payments and invoices. Prior SQL nodes have matched
only structurally unambiguous cases:
1) Exact remittance reference + exact amount
2) Fuzzy remittance ref number (off-by-one) + exact amount
3) Exact ref match + partial amount (ref is correct, amount is a fraction of invoice)
4) Exact amount + customer name match (no remittance, Jaro-Winkler similarity)

Everything else — discounts, short-pays, overpayments, credit memos, partial
payments, multi-invoice grouping, and ambiguous cases — is up to you.

AVAILABLE VIEWS:
  match_exact_remaining_pmt  — fully unmatched payments (columns: payment_id, payment_date, payer_name, amount, method, reference_info, payer_norm)
  match_exact_remaining_inv  — unmatched invoices (columns: invoice_id, customer_id, customer_name, invoice_date, due_date, amount, description, customer_norm, inv_number)
  match_exact_remaining_rem  — remittance lines for unmatched payments (columns: remittance_id, payment_id, invoice_ref, amount, memo, ref_inv_number, ref_direct, has_partial_hint)
  features_rem               — ALL remittance lines (columns: remittance_id, payment_id, invoice_ref, amount, memo, ref_inv_number, ref_direct, has_partial_hint)
  features_pmt               — ALL payments with normalized names (columns: payment_id, payment_date, payer_name, amount, method, reference_info, payer_norm)
  match_exact_all            — already matched pairs (columns: payment_id, invoice_id, applied_amount, match_type)

Start by querying match_exact_remaining_pmt and match_exact_remaining_inv to
understand the size and shape of the residual pool. Then query
match_exact_remaining_rem for remittance details on unmatched payments.

Also check for PARTIALLY MATCHED PAYMENTS: payments that have some matches in
match_exact_all but whose total matched amount does not account for the full
payment. Query features_pmt and match_exact_all to find these, then look at
their remaining remittance lines for unmatched invoices.

MATCHING PATTERNS (apply in order of confidence):

1) DISCOUNT TAKEN: Payment is slightly less than invoice amount, typically 1-5%
   less. Same customer. The payer deducted an early-payment discount.
   applied_amount = payment amount (the discounted amount actually paid).

2) SHORT-PAY / DEDUCTION: Payment is noticeably less than invoice amount (e.g.,
   5-15% less), same customer. May have a memo referencing a dispute, damage,
   or quality issue. applied_amount = payment amount.

3) OVERPAYMENT: Payment exceeds invoice amount by a small percentage, same
   customer. applied_amount = invoice amount (the excess is unapplied).

4) CREDIT MEMO: A customer has a negative-amount invoice (credit note) and a
   positive invoice from the same customer. The payment equals the net of the
   two (positive + negative). Match both invoices to the payment.

5) PARTIAL PAYMENT: Payment covers only a portion of the invoice (e.g., 25-80%).
   Same customer. Remittance memo may mention "partial", "installment", or
   "progress". applied_amount = payment amount.

6) MULTI-INVOICE PAYMENT: A single payment covers multiple invoices from the
   same customer. The payment amount equals the sum of 2+ invoices. Use
   remittance lines to identify which invoices are covered. Each invoice gets
   its own row with applied_amount = invoice amount.

7) PARTIALLY MATCHED MULTI-INVOICE PAYMENTS: A payment already has some matches
   in match_exact_all but still has unaccounted amount. Find the remaining
   unmatched remittance lines and match them to remaining invoices.

8) FALLBACK — EXACT AMOUNT, DIFFERENT CUSTOMER: Payment amount exactly matches
   an invoice amount but the customer names don't match (cross-reference,
   subsidiary paying for parent, etc.). Use date proximity as a tiebreaker.
   Be conservative — only match when the pool is small and the amount is unique.

9) DUPLICATE PAYMENTS: If two payments from the same customer have the same
   amount and one is already matched, the second may be a duplicate. Leave the
   duplicate unmatched (do NOT match it to the same invoice).

RULES:
- Each invoice can only be matched once across ALL matching stages.
- A payment can match multiple invoices (multi-invoice payment) but each
  payment-invoice pair must be unique.
- The applied_amount should be the actual amount applied (which may differ from
  the invoice amount for partial payments, discounts, etc.).
- Be conservative: a false positive is worse than a missed match.
- Use jaro_winkler_similarity() > 0.80 for customer name matching, or
  starts_with() for prefix matching.

OUTPUT: Create a view called match_residual_results with columns:
  payment_id, invoice_id, applied_amount, match_type, note

This should contain ONLY your new matches (not the ones from match_exact_all —
those will be merged in the report node).

DuckDB NOTES:
- Use jaro_winkler_similarity() for fuzzy string matching
- Use starts_with() for prefix matching
- Keep view chains shallow (max 2 levels). Use CTEs instead.
"""

MATCH_RESIDUAL_VALIDATE = """\
-- Check no invoice matched in residual is already matched in exact
SELECT 'fail' AS status,
       'invoice ' || r.invoice_id || ' already in exact matches' AS message
FROM match_residual_results r
JOIN match_exact_all e ON r.invoice_id = e.invoice_id
UNION ALL
SELECT 'fail' AS status,
       'invoice ' || invoice_id || ' matched ' || cnt || 'x in residual' AS message
FROM (SELECT invoice_id, COUNT(*) AS cnt FROM match_residual_results GROUP BY invoice_id)
WHERE cnt > 1;
"""


# ── Node 4: Report (sql) ─────────────────────────────────────────────────────

REPORT_SQL = """\
-- Consolidated results from all matching stages
CREATE VIEW report_results AS
SELECT payment_id, invoice_id, applied_amount FROM match_exact_all
UNION ALL
SELECT payment_id, invoice_id, applied_amount FROM match_residual_results;

-- Unmatched payments (not matched in any stage)
CREATE VIEW report_unmatched_payments AS
SELECT p.payment_id
FROM payments p
LEFT JOIN report_results r ON p.payment_id = r.payment_id
WHERE r.payment_id IS NULL;

-- Unapplied invoices (not matched in any stage)
CREATE VIEW report_unapplied_invoices AS
SELECT i.invoice_id
FROM invoices i
LEFT JOIN report_results r ON i.invoice_id = r.invoice_id
WHERE r.invoice_id IS NULL;

-- Summary
CREATE VIEW report_summary AS
SELECT
    (SELECT COUNT(*) FROM invoices) AS total_invoices,
    (SELECT COUNT(*) FROM payments) AS total_payments,
    (SELECT COUNT(DISTINCT invoice_id) FROM report_results) AS matched_invoices,
    (SELECT COUNT(DISTINCT payment_id) FROM report_results) AS matched_payments,
    (SELECT COUNT(*) FROM report_unmatched_payments) AS unmatched_payments,
    (SELECT COUNT(*) FROM report_unapplied_invoices) AS unapplied_invoices;
"""

REPORT_VALIDATE = """\
-- Every payment must be either matched or unmatched (not both, not neither)
SELECT 'fail' AS status,
       'payment ' || p.payment_id || ' in both matched and unmatched' AS message
FROM report_results p
JOIN report_unmatched_payments u ON p.payment_id = u.payment_id
UNION ALL
SELECT 'fail' AS status,
       'invoice ' || i.invoice_id || ' in both matched and unapplied' AS message
FROM report_results i
JOIN report_unapplied_invoices u ON i.invoice_id = u.invoice_id;
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
        "name": "features",
        "sql": FEATURES_SQL,
        "depends_on": ["invoices", "payments", "remittance_lines"],
    },
    {
        "name": "match_exact",
        "sql": MATCH_EXACT_SQL,
        "validate": {"main": MATCH_EXACT_VALIDATE},
        "depends_on": ["features"],
        "output_columns": {
            "match_exact_all": ["payment_id", "invoice_id", "applied_amount"],
            "match_exact_remaining_pmt": ["payment_id"],
            "match_exact_remaining_inv": ["invoice_id"],
            "match_exact_remaining_rem": ["remittance_id", "payment_id"],
        },
    },
    {
        "name": "match_residual",
        "prompt": MATCH_RESIDUAL_PROMPT,
        "validate": {"main": MATCH_RESIDUAL_VALIDATE},
        "depends_on": ["match_exact", "features"],
        "output_columns": {
            "match_residual_results": [
                "payment_id",
                "invoice_id",
                "applied_amount",
            ],
        },
    },
    {
        "name": "report",
        "sql": REPORT_SQL,
        "validate": {"main": REPORT_VALIDATE},
        "depends_on": [
            "invoices",
            "payments",
            "match_exact",
            "match_residual",
        ],
        "output_columns": {
            "report_results": ["payment_id", "invoice_id", "applied_amount"],
            "report_unmatched_payments": ["payment_id"],
            "report_unapplied_invoices": ["invoice_id"],
        },
    },
]
