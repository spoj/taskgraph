"""Bank Reconciliation V3 — LLM-Heavy (minimal SQL, maximum reasoning)

Minimizes deterministic SQL. Only normalize is sql. A single large
LLM task handles ALL matching. The LLM explores the data via run_sql.

DAG:
    bank_txns, gl_entries
            |
        normalize       (sql)
            |
          match          (prompt + LLM: all matching)
            |
          report         (sql)

Usage:
    tg run --spec specs/bank_rec_v3_llm.py -o bank_rec_v3.db
"""

from examples.bank_rec_problem import BANK_TRANSACTIONS, GL_ENTRIES

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

INPUTS = {
    "bank_txns": {
        "data": BANK_TRANSACTIONS,
        "columns": ["id", "date", "description", "amount"],
    },
    "gl_entries": {
        "data": GL_ENTRIES,
        "columns": ["id", "date", "description", "amount", "ref", "entry_type"],
    },
}


# ---------------------------------------------------------------------------
# Task 1 — Normalize (sql)
# ---------------------------------------------------------------------------

NORMALIZE_SQL = """\
CREATE VIEW bank_norm AS
SELECT
    id,
    date,
    description,
    amount,
    UPPER(regexp_replace(description, '[^A-Za-z0-9 ]', ' ', 'g')) AS desc_clean,
    TRY_CAST(regexp_extract(description, '#(\\d+)', 1) AS INTEGER) AS check_no
FROM bank_txns;

CREATE VIEW gl_norm AS
SELECT
    id,
    date,
    description,
    amount,
    ref,
    entry_type,
    UPPER(regexp_replace(description, '[^A-Za-z0-9 ]', ' ', 'g')) AS desc_clean,
    COALESCE(
        TRY_CAST(regexp_extract(ref, 'CHK-(\\d+)', 1) AS INTEGER),
        TRY_CAST(regexp_extract(description, '#(\\d+)', 1) AS INTEGER)
    ) AS check_no
FROM gl_entries;
"""


# ---------------------------------------------------------------------------
# Task 2 — Match Everything (sql + LLM)
# ---------------------------------------------------------------------------

MATCH_INTENT = """\
Reconcile 43 bank transactions against 49 GL entries for January 2025.
Create all_matched and offsetting_pairs views.

MATCH CATEGORIES (match_type values in all_matched):

1. EXACT 1:1 ('exact_1to1'): Same amount, dates within 5 days.
   - Check numbers first: bank check_no = GL check_no.
   - Then amount + date proximity + description similarity.
   - Use jaro_winkler_similarity() — but bank descriptions are cryptic/truncated
     (e.g. 'AMZN MKTP US*RT4K29ZQ1'), so scores are typically 0.3–0.6, NOT 0.8+.
   - Multiple pairs share the same amount on close dates. Use ROW_NUMBER with
     PARTITION BY on both sides for greedy 1:1 assignment (two rounds).

2. BATCH DEPOSITS ('batch'): One bank deposit = sum of multiple GL credits.
   - Bank description contains 'DEPOSIT', positive amount.
   - Find subsets of unmatched positive GL entries summing exactly to bank amount.
   - One row per GL component with gl_amount = GL entry's amount.

3. AMOUNT MISMATCH ('amount_mismatch'): Amounts differ by < $100 (e.g. wire fees).
   - Same sign, within 5-day window.

OFFSETTING GL PAIRS:
- Unmatched GL entries that cancel each other (amount_a + amount_b = 0).
- offsetting_pairs: original_id, reversal_id, original_desc, reversal_desc,
  original_amount, reversal_amount, original_date, reversal_date, original_ref, reversal_ref

REMAINING: Some bank items (fees) and GL items (outstanding checks) stay unmatched — OK.

OUTPUT: all_matched (bank_id, gl_id, bank_amount, gl_amount, match_type, note)
        offsetting_pairs (schema above)
"""

MATCH_VALIDATE_SQL = """\
CREATE VIEW match__validation AS
WITH bank_agg AS (
    SELECT bank_id, COUNT(*) AS cnt,
           COUNT(*) FILTER (WHERE match_type != 'batch') AS non_batch,
           SUM(gl_amount) AS gl_sum, MAX(bank_amount) AS bank_amt
    FROM all_matched GROUP BY bank_id
)
SELECT 'fail' AS status,
       'bank_id ' || bank_id || ' has invalid multi-match' AS message
FROM bank_agg
WHERE cnt > 1 AND (non_batch > 0 OR ABS(gl_sum - bank_amt) > 0.01)
UNION ALL
SELECT 'fail' AS status,
       'gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM all_matched GROUP BY gl_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'Only ' || (SELECT COUNT(*) FROM all_matched) || ' matches, expected >= 35' AS message
WHERE (SELECT COUNT(*) FROM all_matched) < 35
UNION ALL
SELECT 'warn' AS status,
       'Unmatched bank: ' || b.id || ' $' || b.amount || ' ' || b.description AS message
FROM bank_norm b LEFT JOIN all_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL
UNION ALL
SELECT 'warn' AS status,
       'Unmatched GL: ' || g.id || ' $' || g.amount || ' ' || g.description AS message
FROM gl_norm g
LEFT JOIN all_matched m ON g.id = m.gl_id
LEFT JOIN (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
) oi ON g.id = oi.id
WHERE m.gl_id IS NULL AND oi.id IS NULL;
"""


# ---------------------------------------------------------------------------
# Task 3 — Report (sql)
# ---------------------------------------------------------------------------

REPORT_SQL = """\
CREATE VIEW report_matched AS
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note
FROM all_matched ORDER BY bank_id;

CREATE VIEW report_unmatched_bank AS
SELECT b.id, b.date, b.description, b.amount
FROM bank_norm b LEFT JOIN all_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL ORDER BY b.date;

CREATE VIEW report_unmatched_gl AS
WITH offsetting_ids AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
)
SELECT g.id, g.date, g.description, g.amount, g.ref, g.entry_type
FROM gl_norm g
LEFT JOIN all_matched m ON g.id = m.gl_id
LEFT JOIN offsetting_ids oi ON g.id = oi.id
WHERE m.gl_id IS NULL AND oi.id IS NULL ORDER BY g.date;

CREATE VIEW report_summary AS
SELECT
    (SELECT COUNT(*) FROM bank_norm) AS bank_count,
    (SELECT COUNT(*) FROM gl_norm) AS gl_count,
    (SELECT COUNT(*) FROM all_matched WHERE match_type = 'exact_1to1') AS exact_matches,
    (SELECT COUNT(*) FROM all_matched WHERE match_type = 'batch') AS batch_gl_rows,
    (SELECT COUNT(*) FROM all_matched WHERE match_type = 'amount_mismatch') AS mismatch_matches,
    (SELECT COUNT(*) FROM report_unmatched_bank) AS unmatched_bank,
    (SELECT COUNT(*) FROM report_unmatched_gl) AS unmatched_gl,
    (SELECT COUNT(*) FROM offsetting_pairs) AS offsetting_pairs;
"""

REPORT_VALIDATE_SQL = """\
CREATE VIEW report__validation AS
SELECT 'fail' AS status,
       'Bank count mismatch: ' || bc || ' != ' || mc || ' + ' || ub AS message
FROM (
    SELECT (SELECT COUNT(*) FROM bank_norm) AS bc,
           (SELECT COUNT(DISTINCT bank_id) FROM all_matched) AS mc,
           (SELECT COUNT(*) FROM report_unmatched_bank) AS ub
) WHERE bc != mc + ub
UNION ALL
SELECT 'fail' AS status,
       'GL count mismatch: ' || gc || ' != ' || mc || ' + ' || ug || ' + ' || oc AS message
FROM (
    SELECT (SELECT COUNT(*) FROM gl_norm) AS gc,
           (SELECT COUNT(DISTINCT gl_id) FROM all_matched) AS mc,
           (SELECT COUNT(*) FROM report_unmatched_gl) AS ug,
           (SELECT COUNT(*) * 2 FROM offsetting_pairs) AS oc
) WHERE gc != mc + ug + oc;
"""


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS = [
    {
        "name": "normalize",
        "sql": NORMALIZE_SQL,
        "inputs": ["bank_txns", "gl_entries"],
        "outputs": ["bank_norm", "gl_norm"],
    },
    {
        "name": "match",
        "prompt": MATCH_INTENT,
        "validate_sql": MATCH_VALIDATE_SQL,
        "inputs": ["bank_norm", "gl_norm"],
        "outputs": [
            "all_matched",
            "offsetting_pairs",
        ],
        "output_columns": {
            "all_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
            "offsetting_pairs": [
                "original_id",
                "reversal_id",
                "original_amount",
                "reversal_amount",
            ],
        },
    },
    {
        "name": "report",
        "sql": REPORT_SQL,
        "validate_sql": REPORT_VALIDATE_SQL,
        "inputs": [
            "bank_norm",
            "gl_norm",
            "all_matched",
            "offsetting_pairs",
        ],
        "outputs": [
            "report_matched",
            "report_unmatched_bank",
            "report_unmatched_gl",
            "report_summary",
        ],
    },
]
