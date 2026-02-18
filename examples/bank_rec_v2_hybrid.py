"""Bank Reconciliation V2 — Hybrid: sql base + LLM for hard cases

Deterministic SQL handles easy/medium matches (exact amount + date window +
check numbers). LLM picks up the leftovers: batch deposits, amount mismatches.

DAG:
    bank_txns, gl_entries
            |
        normalize           (sql)
            |
      match_confident       (sql: two-round greedy 1:1)
            |
        offsetting          (sql: self-canceling GL pairs)
            |
       match_hard           (sql + LLM: batch deposits, tolerance)
            |
          report            (sql)

Usage:
    tg run --spec specs/bank_rec_v2_hybrid.py -o bank_rec_v2.db
"""

from examples.bank_rec_problem import BANK_TRANSACTIONS, GL_ENTRIES

# ---------------------------------------------------------------------------
# Source data (referenced by source nodes in NODES below)
# ---------------------------------------------------------------------------

# (Source nodes are defined in NODES below)


# ---------------------------------------------------------------------------
# Task 1 — Normalize (sql)
# ---------------------------------------------------------------------------

NORMALIZE_SQL = """\
CREATE VIEW normalize_bank AS
SELECT
    id, date, description, amount,
    UPPER(regexp_replace(description, '[^A-Za-z0-9 ]', ' ', 'g')) AS desc_clean,
    TRY_CAST(regexp_extract(description, '#(\\d+)', 1) AS INTEGER) AS check_no
FROM bank_txns;

CREATE VIEW normalize_gl AS
SELECT
    id, date, description, amount, ref, entry_type,
    UPPER(regexp_replace(description, '[^A-Za-z0-9 ]', ' ', 'g')) AS desc_clean,
    COALESCE(
        TRY_CAST(regexp_extract(ref, 'CHK-(\\d+)', 1) AS INTEGER),
        TRY_CAST(regexp_extract(description, '#(\\d+)', 1) AS INTEGER)
    ) AS check_no
FROM gl_entries;
"""


# ---------------------------------------------------------------------------
# Task 2 — Confident 1:1 matching (sql)
# ---------------------------------------------------------------------------

MATCH_CONFIDENT_SQL = """\
CREATE VIEW match_confident_matched AS
WITH
candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS bank_amount, g.amount AS gl_amount,
        b.date AS bank_date, g.date AS gl_date,
        b.description AS bank_desc, g.description AS gl_desc,
        ABS(date_diff('day', b.date, g.date)) AS date_gap,
        jaro_winkler_similarity(b.desc_clean, g.desc_clean) AS desc_score,
        CASE WHEN b.check_no IS NOT NULL
              AND b.check_no = g.check_no THEN 1 ELSE 0 END AS check_match
    FROM normalize_bank b
    JOIN normalize_gl g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 5
),
r1_ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY bank_id
            ORDER BY check_match DESC, date_gap ASC, desc_score DESC) AS b_rank,
        ROW_NUMBER() OVER (PARTITION BY gl_id
            ORDER BY check_match DESC, date_gap ASC, desc_score DESC) AS g_rank
    FROM candidates
),
round1 AS (
    SELECT bank_id, gl_id, bank_amount, gl_amount, bank_date, gl_date,
           bank_desc, gl_desc, date_gap, desc_score, check_match
    FROM r1_ranked WHERE b_rank = 1 AND g_rank = 1
),
r2_pool AS (
    SELECT c.* FROM candidates c
    WHERE c.bank_id NOT IN (SELECT bank_id FROM round1)
      AND c.gl_id   NOT IN (SELECT gl_id   FROM round1)
),
r2_ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY bank_id
            ORDER BY check_match DESC, date_gap ASC, desc_score DESC) AS b_rank,
        ROW_NUMBER() OVER (PARTITION BY gl_id
            ORDER BY check_match DESC, date_gap ASC, desc_score DESC) AS g_rank
    FROM r2_pool
),
round2 AS (
    SELECT bank_id, gl_id, bank_amount, gl_amount, bank_date, gl_date,
           bank_desc, gl_desc, date_gap, desc_score, check_match
    FROM r2_ranked WHERE b_rank = 1 AND g_rank = 1
)
SELECT *, 'exact_1to1' AS match_type, '' AS note FROM round1
UNION ALL
SELECT *, 'exact_1to1' AS match_type, '' AS note FROM round2;

CREATE VIEW match_confident_unmatched_bank AS
SELECT b.* FROM normalize_bank b
LEFT JOIN match_confident_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL;

CREATE VIEW match_confident_unmatched_gl AS
SELECT g.* FROM normalize_gl g
LEFT JOIN match_confident_matched m ON g.id = m.gl_id
WHERE m.gl_id IS NULL;
"""

MATCH_CONFIDENT_VALIDATE_SQL = """\
CREATE VIEW match_confident__validation AS
SELECT 'fail' AS status,
       'bank_id ' || bank_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT bank_id, COUNT(*) AS cnt FROM match_confident_matched GROUP BY bank_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM match_confident_matched GROUP BY gl_id)
WHERE cnt > 1;
"""


# ---------------------------------------------------------------------------
# Task 3 — Offsetting GL pairs (sql)
# ---------------------------------------------------------------------------

OFFSETTING_SQL = """\
CREATE VIEW offsetting_pairs AS
WITH unmatched_gl AS (
    SELECT g.* FROM normalize_gl g
    LEFT JOIN match_confident_matched m ON g.id = m.gl_id
    WHERE m.gl_id IS NULL
)
SELECT
    a.id AS original_id, b.id AS reversal_id,
    a.description AS original_desc, b.description AS reversal_desc,
    a.amount AS original_amount, b.amount AS reversal_amount,
    a.date AS original_date, b.date AS reversal_date,
    a.ref AS original_ref, b.ref AS reversal_ref
FROM unmatched_gl a
JOIN unmatched_gl b
  ON a.amount + b.amount = 0
 AND a.amount < 0
 AND a.id < b.id
 AND ABS(date_diff('day', a.date, b.date)) <= 10
 AND jaro_winkler_similarity(a.desc_clean, b.desc_clean) > 0.4;
"""


# ---------------------------------------------------------------------------
# Task 4 — Hard matching (sql + LLM)
# ---------------------------------------------------------------------------

MATCH_HARD_INTENT = """\
Resolve remaining unmatched bank and GL items after exact matching.

STRATEGIES:
1. BATCH DEPOSITS: One bank deposit (description contains 'DEPOSIT') = sum of
   multiple GL credits. Find subsets of unmatched positive GL entries that sum
   exactly to the bank amount. Try 2-5 element combinations.
2. AMOUNT TOLERANCE: Bank and GL differ by a small fee (< $100), e.g. wire fee
   netted by accountant. Same sign, within 5-day window.

Bank descriptions are cryptic/truncated (e.g. 'AMZN MKTP US*RT4K29ZQ1').
jaro_winkler scores between bank and GL are typically 0.3-0.6, not 0.8+.

OUTPUT: match_hard_all_matched view with columns:
  bank_id, gl_id, bank_amount, gl_amount, match_type, note
Must include ALL match_confident_matched rows plus any new batch/tolerance matches.
Batch deposits: one row per GL component, match_type='batch'.
"""

MATCH_HARD_VALIDATE_SQL = """\
CREATE VIEW match_hard__validation AS
WITH bank_agg AS (
    SELECT bank_id, COUNT(*) AS cnt,
           COUNT(*) FILTER (WHERE match_type != 'batch') AS non_batch,
           SUM(gl_amount) AS gl_sum, MAX(bank_amount) AS bank_amt
    FROM match_hard_all_matched GROUP BY bank_id
)
SELECT 'fail' AS status,
       'bank_id ' || bank_id || ' has invalid multi-match' AS message
FROM bank_agg
WHERE cnt > 1 AND (non_batch > 0 OR ABS(gl_sum - bank_amt) > 0.01)
UNION ALL
SELECT 'fail' AS status,
       'gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM match_hard_all_matched GROUP BY gl_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'Confident match ' || c.bank_id || '->' || c.gl_id || ' missing' AS message
FROM match_confident_matched c
LEFT JOIN match_hard_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL
UNION ALL
SELECT 'warn' AS status,
       'Unmatched bank: ' || b.id || ' $' || b.amount || ' ' || b.description AS message
FROM normalize_bank b LEFT JOIN match_hard_all_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL
UNION ALL
SELECT 'warn' AS status,
       'Unmatched GL: ' || g.id || ' $' || g.amount || ' ' || g.description AS message
FROM match_hard_remaining_gl g LEFT JOIN match_hard_all_matched m ON g.id = m.gl_id
WHERE m.gl_id IS NULL;
"""


# ---------------------------------------------------------------------------
# Task 5 — Report (sql)
# ---------------------------------------------------------------------------

REPORT_SQL = """\
CREATE VIEW report_matched AS
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note
FROM match_hard_all_matched ORDER BY bank_id;

CREATE VIEW report_unmatched_bank AS
SELECT b.id, b.date, b.description, b.amount
FROM normalize_bank b LEFT JOIN match_hard_all_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL ORDER BY b.date;

CREATE VIEW report_unmatched_gl AS
WITH offsetting_ids AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
)
SELECT g.id, g.date, g.description, g.amount, g.ref, g.entry_type
FROM normalize_gl g
LEFT JOIN match_hard_all_matched m ON g.id = m.gl_id
LEFT JOIN offsetting_ids oi ON g.id = oi.id
WHERE m.gl_id IS NULL AND oi.id IS NULL ORDER BY g.date;

CREATE VIEW report_summary AS
SELECT
    (SELECT COUNT(*) FROM normalize_bank) AS bank_count,
    (SELECT COUNT(*) FROM normalize_gl) AS gl_count,
    (SELECT COUNT(*) FROM match_hard_all_matched WHERE match_type = 'exact_1to1') AS exact_matches,
    (SELECT COUNT(*) FROM match_hard_all_matched WHERE match_type = 'batch') AS batch_gl_rows,
    (SELECT COUNT(*) FROM match_hard_all_matched WHERE match_type = 'amount_mismatch') AS mismatch_matches,
    (SELECT COUNT(*) FROM report_unmatched_bank) AS unmatched_bank,
    (SELECT COUNT(*) FROM report_unmatched_gl) AS unmatched_gl,
    (SELECT COUNT(*) FROM offsetting_pairs) AS offsetting_pairs;
"""

REPORT_VALIDATE_SQL = """\
CREATE VIEW report__validation AS
SELECT 'fail' AS status,
       'Bank count mismatch: ' || bc || ' != ' || mc || ' + ' || ub AS message
FROM (
    SELECT (SELECT COUNT(*) FROM normalize_bank) AS bc,
           (SELECT COUNT(DISTINCT bank_id) FROM match_hard_all_matched) AS mc,
           (SELECT COUNT(*) FROM report_unmatched_bank) AS ub
) WHERE bc != mc + ub
UNION ALL
SELECT 'fail' AS status,
       'GL count mismatch: ' || gc || ' != ' || mc || ' + ' || ug || ' + ' || oc AS message
FROM (
    SELECT (SELECT COUNT(*) FROM normalize_gl) AS gc,
           (SELECT COUNT(DISTINCT gl_id) FROM match_hard_all_matched) AS mc,
           (SELECT COUNT(*) FROM report_unmatched_gl) AS ug,
           (SELECT COUNT(*) * 2 FROM offsetting_pairs) AS oc
) WHERE gc != mc + ug + oc;
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

NODES = [
    # Source nodes
    {
        "name": "bank_txns",
        "source": BANK_TRANSACTIONS,
        "columns": ["id", "date", "description", "amount"],
    },
    {
        "name": "gl_entries",
        "source": GL_ENTRIES,
        "columns": ["id", "date", "description", "amount", "ref", "entry_type"],
    },
    # Transform nodes
    {
        "name": "normalize",
        "sql": NORMALIZE_SQL,
        "depends_on": ["bank_txns", "gl_entries"],
    },
    {
        "name": "match_confident",
        "sql": MATCH_CONFIDENT_SQL,
        "validate_sql": MATCH_CONFIDENT_VALIDATE_SQL,
        "depends_on": ["normalize"],
        "output_columns": {
            "match_confident_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
        },
    },
    {
        "name": "offsetting",
        "sql": OFFSETTING_SQL,
        "depends_on": ["normalize", "match_confident"],
    },
    {
        "name": "match_hard",
        "prompt": MATCH_HARD_INTENT,
        "validate_sql": MATCH_HARD_VALIDATE_SQL,
        "depends_on": ["normalize", "match_confident", "offsetting"],
        "output_columns": {
            "match_hard_all_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
        },
    },
    {
        "name": "report",
        "sql": REPORT_SQL,
        "validate_sql": REPORT_VALIDATE_SQL,
        "depends_on": ["normalize", "match_hard", "offsetting"],
    },
]
