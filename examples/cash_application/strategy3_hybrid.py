"""Strategy 3: Taskgraph Hybrid (SQL preprocessing + LLM residual matching).

Architecture: SQL pipeline handles deterministic matching (exact refs, exact
amounts, known patterns), LLM handles residual fuzzy/ambiguous cases.

DAG:
    invoices, payments, remittance_lines
                    |
                features           (sql: name normalization, ref extraction)
                    |
              match_exact          (sql: exact ref + exact amount, exact amount + customer)
                    |
              match_patterns       (sql: discount, short-pay, multi-invoice grouping)
                    |
              match_residual       (prompt: fuzzy names, garbled refs, ambiguous amounts)
                    |
                report             (sql: consolidate results + unmatched/unapplied)

Design principles:
  - SQL nodes handle unambiguous deterministic matches
  - LLM handles judgment calls: fuzzy entity resolution, garbled references,
    partial payment detection where amount ratios alone are insufficient
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
already_matched_pairs AS (
    SELECT payment_id, invoice_id FROM match_exact_r1
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

-- Round 3: Ref match + partial amount (handles multi_invoice_partial and partial_payment
-- where the ref is correct but the amount is a fraction of the invoice).
-- Only exact ref matches (ref_direct or exact inv_number — NO off-by-one) to stay safe.
-- Amount ratio 0.20–1.005 covers partial (25-80%) and exact, avoids overpayments.
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


# ── Node 3: Pattern Matches (sql) ────────────────────────────────────────────

MATCH_PATTERNS_SQL = """\
-- Discount detection: payment is 97-99.5% of invoice amount, same customer
CREATE VIEW match_patterns_discount AS
WITH candidates AS (
    SELECT
        p.payment_id,
        i.invoice_id,
        p.amount AS pmt_amount,
        i.amount AS inv_amount,
        p.amount / i.amount AS ratio,
        jaro_winkler_similarity(p.payer_norm, i.customer_norm) AS name_sim,
        ABS(date_diff('day', p.payment_date, i.invoice_date)) AS date_gap
    FROM match_exact_remaining_pmt p
    JOIN match_exact_remaining_inv i
      ON i.amount > 0
     AND p.amount / i.amount BETWEEN 0.970 AND 0.995
     AND (
         jaro_winkler_similarity(p.payer_norm, i.customer_norm) > 0.80
         OR starts_with(i.customer_norm, p.payer_norm)
         OR starts_with(p.payer_norm, i.customer_norm)
     )
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY payment_id ORDER BY name_sim DESC, date_gap ASC) AS p_rnk,
        ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY name_sim DESC, date_gap ASC) AS i_rnk
    FROM candidates
)
SELECT payment_id, invoice_id, pmt_amount AS applied_amount,
       'discount' AS match_type
FROM ranked WHERE p_rnk = 1 AND i_rnk = 1;

-- Short-pay: payment is 85-97% of invoice, same customer
CREATE VIEW match_patterns_shortpay AS
WITH already_inv AS (SELECT invoice_id FROM match_patterns_discount),
already_pmt AS (SELECT payment_id FROM match_patterns_discount),
candidates AS (
    SELECT
        p.payment_id,
        i.invoice_id,
        p.amount AS pmt_amount,
        i.amount AS inv_amount,
        p.amount / i.amount AS ratio,
        jaro_winkler_similarity(p.payer_norm, i.customer_norm) AS name_sim,
        ABS(date_diff('day', p.payment_date, i.invoice_date)) AS date_gap
    FROM match_exact_remaining_pmt p
    JOIN match_exact_remaining_inv i
      ON i.amount > 0
     AND p.amount / i.amount BETWEEN 0.850 AND 0.970
     AND (
         jaro_winkler_similarity(p.payer_norm, i.customer_norm) > 0.80
         OR starts_with(i.customer_norm, p.payer_norm)
         OR starts_with(p.payer_norm, i.customer_norm)
     )
    WHERE i.invoice_id NOT IN (SELECT invoice_id FROM already_inv)
      AND p.payment_id NOT IN (SELECT payment_id FROM already_pmt)
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY payment_id ORDER BY name_sim DESC, date_gap ASC) AS p_rnk,
        ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY name_sim DESC, date_gap ASC) AS i_rnk
    FROM candidates
)
SELECT payment_id, invoice_id, pmt_amount AS applied_amount,
       'short_pay' AS match_type
FROM ranked WHERE p_rnk = 1 AND i_rnk = 1;

-- Overpayment: payment is 100-115% of invoice, same customer
CREATE VIEW match_patterns_overpay AS
WITH already_inv AS (
    SELECT invoice_id FROM match_patterns_discount
    UNION ALL SELECT invoice_id FROM match_patterns_shortpay
),
already_pmt AS (
    SELECT payment_id FROM match_patterns_discount
    UNION ALL SELECT payment_id FROM match_patterns_shortpay
),
candidates AS (
    SELECT
        p.payment_id,
        i.invoice_id,
        i.amount AS applied_amount,  -- apply invoice amount, not overpayment
        p.amount / i.amount AS ratio,
        jaro_winkler_similarity(p.payer_norm, i.customer_norm) AS name_sim,
        ABS(date_diff('day', p.payment_date, i.invoice_date)) AS date_gap
    FROM match_exact_remaining_pmt p
    JOIN match_exact_remaining_inv i
      ON i.amount > 0
     AND p.amount / i.amount BETWEEN 1.001 AND 1.150
     AND (
         jaro_winkler_similarity(p.payer_norm, i.customer_norm) > 0.80
         OR starts_with(i.customer_norm, p.payer_norm)
         OR starts_with(p.payer_norm, i.customer_norm)
     )
    WHERE i.invoice_id NOT IN (SELECT invoice_id FROM already_inv)
      AND p.payment_id NOT IN (SELECT payment_id FROM already_pmt)
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY payment_id ORDER BY name_sim DESC, date_gap ASC) AS p_rnk,
        ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY name_sim DESC, date_gap ASC) AS i_rnk
    FROM candidates
)
SELECT payment_id, invoice_id, applied_amount,
       'overpayment' AS match_type
FROM ranked WHERE p_rnk = 1 AND i_rnk = 1;

-- Credit memo: find positive invoice + negative invoice from same customer that
-- sum to a payment amount
CREATE VIEW match_patterns_credit AS
WITH already_inv AS (
    SELECT invoice_id FROM match_patterns_discount
    UNION ALL SELECT invoice_id FROM match_patterns_shortpay
    UNION ALL SELECT invoice_id FROM match_patterns_overpay
),
already_pmt AS (
    SELECT payment_id FROM match_patterns_discount
    UNION ALL SELECT payment_id FROM match_patterns_shortpay
    UNION ALL SELECT payment_id FROM match_patterns_overpay
),
remaining_pmt AS (
    SELECT * FROM match_exact_remaining_pmt
    WHERE payment_id NOT IN (SELECT payment_id FROM already_pmt)
),
remaining_inv AS (
    SELECT * FROM match_exact_remaining_inv
    WHERE invoice_id NOT IN (SELECT invoice_id FROM already_inv)
),
candidates AS (
    SELECT
        p.payment_id,
        pos.invoice_id AS pos_invoice_id,
        neg.invoice_id AS neg_invoice_id,
        pos.amount AS pos_amount,
        neg.amount AS neg_amount,
        jaro_winkler_similarity(p.payer_norm, pos.customer_norm) AS name_sim
    FROM remaining_pmt p
    JOIN remaining_inv pos ON pos.amount > 0
        AND (jaro_winkler_similarity(p.payer_norm, pos.customer_norm) > 0.80
             OR starts_with(pos.customer_norm, p.payer_norm)
             OR starts_with(p.payer_norm, pos.customer_norm))
    JOIN remaining_inv neg ON neg.amount < 0
        AND neg.customer_id = pos.customer_id
        AND neg.invoice_id != pos.invoice_id
    WHERE ABS(p.amount - (pos.amount + neg.amount)) < 0.01
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY payment_id ORDER BY name_sim DESC) AS rnk
    FROM candidates
)
SELECT payment_id, pos_invoice_id, neg_invoice_id, pos_amount, neg_amount
FROM ranked WHERE rnk = 1;

-- Explode credit memo matches into individual rows
CREATE VIEW match_patterns_credit_rows AS
SELECT payment_id, pos_invoice_id AS invoice_id, pos_amount AS applied_amount,
       'credit_memo' AS match_type
FROM match_patterns_credit
UNION ALL
SELECT payment_id, neg_invoice_id AS invoice_id, neg_amount AS applied_amount,
       'credit_memo' AS match_type
FROM match_patterns_credit;

-- Partial payment: 25-80% of invoice + same customer + memo hint required
-- Generator always puts "partial"/"installment"/"progress" in partial_payment memos
CREATE VIEW match_patterns_partial AS
WITH already_inv AS (
    SELECT invoice_id FROM match_patterns_discount
    UNION ALL SELECT invoice_id FROM match_patterns_shortpay
    UNION ALL SELECT invoice_id FROM match_patterns_overpay
    UNION ALL SELECT invoice_id FROM match_patterns_credit_rows
),
already_pmt AS (
    SELECT payment_id FROM match_patterns_discount
    UNION ALL SELECT payment_id FROM match_patterns_shortpay
    UNION ALL SELECT payment_id FROM match_patterns_overpay
    UNION ALL SELECT payment_id FROM match_patterns_credit_rows
),
-- Find payments with partial hints in their remittance lines
pmt_with_hint AS (
    SELECT DISTINCT payment_id
    FROM features_rem
    WHERE has_partial_hint = true
),
candidates AS (
    SELECT
        p.payment_id,
        i.invoice_id,
        p.amount AS pmt_amount,
        i.amount AS inv_amount,
        p.amount / i.amount AS ratio,
        jaro_winkler_similarity(p.payer_norm, i.customer_norm) AS name_sim,
        ABS(date_diff('day', p.payment_date, i.invoice_date)) AS date_gap
    FROM match_exact_remaining_pmt p
    JOIN match_exact_remaining_inv i
      ON i.amount > 0
     AND p.amount / i.amount BETWEEN 0.20 AND 0.85
     AND (
         jaro_winkler_similarity(p.payer_norm, i.customer_norm) > 0.80
         OR starts_with(i.customer_norm, p.payer_norm)
         OR starts_with(p.payer_norm, i.customer_norm)
     )
    WHERE i.invoice_id NOT IN (SELECT invoice_id FROM already_inv)
      AND p.payment_id NOT IN (SELECT payment_id FROM already_pmt)
      AND p.payment_id IN (SELECT payment_id FROM pmt_with_hint)
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY payment_id ORDER BY name_sim DESC, date_gap ASC) AS p_rnk,
        ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY name_sim DESC, date_gap ASC) AS i_rnk
    FROM candidates
)
SELECT payment_id, invoice_id, pmt_amount AS applied_amount,
       'partial_payment' AS match_type
FROM ranked WHERE p_rnk = 1 AND i_rnk = 1;

-- Fallback: Exact amount + closest date within 90 days, any customer (like S2 Pass 10)
CREATE VIEW match_patterns_fallback AS
WITH already_inv AS (
    SELECT invoice_id FROM match_patterns_discount
    UNION ALL SELECT invoice_id FROM match_patterns_shortpay
    UNION ALL SELECT invoice_id FROM match_patterns_overpay
    UNION ALL SELECT invoice_id FROM match_patterns_credit_rows
    UNION ALL SELECT invoice_id FROM match_patterns_partial
),
already_pmt AS (
    SELECT payment_id FROM match_patterns_discount
    UNION ALL SELECT payment_id FROM match_patterns_shortpay
    UNION ALL SELECT payment_id FROM match_patterns_overpay
    UNION ALL SELECT payment_id FROM match_patterns_credit_rows
    UNION ALL SELECT payment_id FROM match_patterns_partial
),
candidates AS (
    SELECT
        p.payment_id,
        i.invoice_id,
        i.amount AS applied_amount,
        ABS(date_diff('day', p.payment_date, i.invoice_date)) AS date_gap
    FROM match_exact_remaining_pmt p
    JOIN match_exact_remaining_inv i
      ON ABS(p.amount - i.amount) < 0.01
     AND ABS(date_diff('day', p.payment_date, i.invoice_date)) <= 90
    WHERE i.invoice_id NOT IN (SELECT invoice_id FROM already_inv)
      AND p.payment_id NOT IN (SELECT payment_id FROM already_pmt)
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY payment_id ORDER BY date_gap ASC) AS p_rnk,
        ROW_NUMBER() OVER (PARTITION BY invoice_id ORDER BY date_gap ASC) AS i_rnk
    FROM candidates
)
SELECT payment_id, invoice_id, applied_amount,
       'exact_amount_date' AS match_type
FROM ranked WHERE p_rnk = 1 AND i_rnk = 1;

-- All pattern matches combined
CREATE VIEW match_patterns_all AS
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_patterns_discount
UNION ALL
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_patterns_shortpay
UNION ALL
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_patterns_overpay
UNION ALL
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_patterns_credit_rows
UNION ALL
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_patterns_partial
UNION ALL
SELECT payment_id, invoice_id, applied_amount, match_type FROM match_patterns_fallback;

-- Remaining pools after pattern matching
CREATE VIEW match_patterns_remaining_pmt AS
SELECT p.* FROM match_exact_remaining_pmt p
LEFT JOIN (SELECT DISTINCT payment_id FROM match_patterns_all) m ON p.payment_id = m.payment_id
WHERE m.payment_id IS NULL;

CREATE VIEW match_patterns_remaining_inv AS
SELECT i.* FROM match_exact_remaining_inv i
LEFT JOIN (SELECT DISTINCT invoice_id FROM match_patterns_all) m ON i.invoice_id = m.invoice_id
WHERE m.invoice_id IS NULL;

-- Payments that are partially matched: they have SOME matches in exact/patterns
-- but their total matched amount doesn't account for the full payment.
-- This helps the LLM find remaining unmatched remittance lines.
CREATE VIEW match_patterns_partial_pmt AS
SELECT
    p.payment_id,
    p.payer_name,
    p.amount AS payment_amount,
    COALESCE(m.matched_amount, 0) AS matched_amount,
    p.amount - COALESCE(m.matched_amount, 0) AS unmatched_amount,
    COALESCE(m.match_count, 0) AS match_count
FROM features_pmt p
LEFT JOIN (
    SELECT payment_id,
           SUM(applied_amount) AS matched_amount,
           COUNT(*) AS match_count
    FROM (
        SELECT payment_id, applied_amount FROM match_exact_all
        UNION ALL
        SELECT payment_id, applied_amount FROM match_patterns_all
    )
    GROUP BY payment_id
) m ON p.payment_id = m.payment_id
WHERE m.match_count > 0
  AND ABS(p.amount - COALESCE(m.matched_amount, 0)) > 0.01;
"""

MATCH_PATTERNS_VALIDATE = """\
SELECT 'fail' AS status,
       'invoice ' || invoice_id || ' matched ' || cnt || 'x in patterns' AS message
FROM (SELECT invoice_id, COUNT(*) AS cnt FROM match_patterns_all GROUP BY invoice_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'invoice ' || a.invoice_id || ' in both exact and patterns' AS message
FROM match_patterns_all a
JOIN match_exact_all e ON a.invoice_id = e.invoice_id;
"""


# ── Node 4: Residual LLM Matching (prompt) ───────────────────────────────────

MATCH_RESIDUAL_PROMPT = """\
Resolve remaining unmatched payments and invoices. Prior SQL nodes have matched:
1) Exact remittance reference + exact amount
2) Fuzzy remittance ref number + exact amount
3) Ref match + partial amount (20-100% of invoice, covers partial payments on multi-invoice lines)
4) Exact amount + customer name (with Jaro-Winkler similarity)
5) Discount detection (97-99.5% of invoice, same customer)
6) Short-pay detection (85-97% of invoice, same customer)
7) Overpayment detection (100-115% of invoice, same customer)
8) Credit memo detection (positive + negative invoices summing to payment)
9) Partial payment detection (25-85% of invoice, same customer, memo hint)
10) Exact amount + closest date fallback (within 90 days, any customer)

Everything else is up to you. Query the remaining pools and match what you can.

AVAILABLE VIEWS:
  match_patterns_remaining_pmt  — fully unmatched payments (columns: payment_id, payment_date, payer_name, amount, method, reference_info, payer_norm)
  match_patterns_remaining_inv  — unmatched invoices (columns: invoice_id, customer_id, customer_name, invoice_date, due_date, amount, description, customer_norm, inv_number)
  match_patterns_partial_pmt    — PARTIALLY matched payments: have some matches but unaccounted amount remains (columns: payment_id, payer_name, payment_amount, matched_amount, unmatched_amount, match_count)
  features_rem                  — all remittance lines (columns: remittance_id, payment_id, invoice_ref, amount, memo, ref_inv_number, ref_direct, has_partial_hint)
  match_exact_all               — already matched (exact + fuzzy ref + ref_partial + amount+customer)
  match_patterns_all            — already matched (discount, short-pay, overpay, credit, partial, fallback)

CRITICAL — PARTIALLY MATCHED PAYMENTS:
Start by querying match_patterns_partial_pmt. These are payments that have SOME
invoice matches but still have unaccounted amounts. For each such payment:
1) Query features_rem for ALL remittance lines of that payment
2) Check which remittance lines are already matched (their invoice appears in match_exact_all or match_patterns_all)
3) For the UNMATCHED remittance lines, look for matching invoices in match_patterns_remaining_inv
   using the remittance line's ref info and amount

MATCHING STRATEGIES (apply in order of confidence):

1) PARTIALLY MATCHED MULTI-INVOICE PAYMENTS (see above — highest priority)

2) MULTI-INVOICE PAYMENTS: A single payment may cover multiple invoices from the
   same customer. Look for payments where the amount equals the sum of 2-4
   unmatched invoices from the same customer. Use remittance lines for hints
   (group by payment_id). Use jaro_winkler_similarity > 0.80 for entity matching.

3) PARTIAL PAYMENTS: A payment may be 25-80% of an invoice amount, typically
   with remittance memo mentioning "partial", "installment", or "progress".
   Match within the same customer.

4) DUPLICATE PAYMENTS: If two payments have the same amount for the same customer
   and one is already matched, the second may be a duplicate. The duplicate
   should be left unmatched (do NOT match it to the same invoice).

5) MANUAL MAPPING: When the candidate pool is very small, read the rows, pair
   them up manually, and output a VALUES clause.

RULES:
- Each invoice can only be matched once across ALL matching stages.
- A payment can match multiple invoices (multi-invoice payment) but each
  payment-invoice pair must be unique.
- The applied_amount should be the actual amount applied (which may differ from
  the invoice amount for partial payments, discounts, etc.).
- Be conservative: a false positive is worse than a missed match.

OUTPUT: Create a view called match_residual_results with columns:
  payment_id, invoice_id, applied_amount, match_type, note

This should contain ONLY your new matches (not the ones from match_exact_all or
match_patterns_all — those will be merged in the report node).

DuckDB NOTES:
- Use jaro_winkler_similarity() for fuzzy string matching
- Use starts_with() for prefix matching
- Keep view chains shallow (max 2 levels). Use CTEs instead.
"""

MATCH_RESIDUAL_VALIDATE = """\
-- Check no invoice matched in residual is already matched elsewhere
SELECT 'fail' AS status,
       'invoice ' || r.invoice_id || ' already in exact matches' AS message
FROM match_residual_results r
JOIN match_exact_all e ON r.invoice_id = e.invoice_id
UNION ALL
SELECT 'fail' AS status,
       'invoice ' || r.invoice_id || ' already in pattern matches' AS message
FROM match_residual_results r
JOIN match_patterns_all p ON r.invoice_id = p.invoice_id
UNION ALL
SELECT 'fail' AS status,
       'invoice ' || invoice_id || ' matched ' || cnt || 'x in residual' AS message
FROM (SELECT invoice_id, COUNT(*) AS cnt FROM match_residual_results GROUP BY invoice_id)
WHERE cnt > 1;
"""


# ── Node 5: Report (sql) ─────────────────────────────────────────────────────

REPORT_SQL = """\
-- Consolidated results from all matching stages
CREATE VIEW report_results AS
SELECT payment_id, invoice_id, applied_amount FROM match_exact_all
UNION ALL
SELECT payment_id, invoice_id, applied_amount FROM match_patterns_all
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
        },
    },
    {
        "name": "match_patterns",
        "sql": MATCH_PATTERNS_SQL,
        "validate": {"main": MATCH_PATTERNS_VALIDATE},
        "depends_on": ["match_exact", "features"],
        "output_columns": {
            "match_patterns_all": ["payment_id", "invoice_id", "applied_amount"],
            "match_patterns_remaining_pmt": ["payment_id"],
            "match_patterns_remaining_inv": ["invoice_id"],
            "match_patterns_partial_pmt": ["payment_id", "unmatched_amount"],
        },
    },
    {
        "name": "match_residual",
        "prompt": MATCH_RESIDUAL_PROMPT,
        "validate": {"main": MATCH_RESIDUAL_VALIDATE},
        "depends_on": ["match_exact", "match_patterns", "features"],
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
            "match_patterns",
            "match_residual",
        ],
        "output_columns": {
            "report_results": ["payment_id", "invoice_id", "applied_amount"],
            "report_unmatched_payments": ["payment_id"],
            "report_unapplied_invoices": ["invoice_id"],
        },
    },
]
