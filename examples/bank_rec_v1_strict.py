"""Bank Reconciliation V1 — Pure Deterministic SQL (sql only)

All tasks use sql: no LLM involvement. Shows the ceiling of what
purely deterministic SQL can achieve on the hard bank rec problem.

Expected results (~35-37 of 43 bank items matched):
  - Easy 1:1 matches via amount + date window: ~16
  - Medium 1:1 with same-amount disambiguation by date/description: ~14
  - Hard check-number matches: ~5
  - Offsetting GL pairs detected: 1 (G20/G21)
  - Batch deposits: WILL MISS (no subset-sum solver in pure SQL)
  - Amount mismatch B28 ($75K vs $74,975): WILL MISS (no fuzzy amount join)
  - Unmatched bank (no GL): 3 (B13, B41, B42)
  - Unmatched GL (no bank): 2 (G48, G49)

DAG:
    bank_txns, gl_entries
            |
        normalize       (clean descriptions, extract check numbers)
            |
          match         (two-round greedy: amount + date + check# + description)
            |
        offsetting      (find self-canceling GL pairs among unmatched)
            |
          report        (reconciliation summary views)

Usage:
    tg run --spec specs/bank_rec_v1_strict.py -o bank_rec_v1.db
    tg show --spec specs/bank_rec_v1_strict.py
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
# Task 1 — Normalize
# ---------------------------------------------------------------------------
# Clean descriptions, extract check numbers from both bank and GL.
# Bank: CHECK #4518 -> check_no = 4518
# GL ref: CHK-4518 -> check_no = 4518 (COALESCE prefers ref-based extraction)

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
# Task 2 — Match (two-round greedy 1:1 assignment)
# ---------------------------------------------------------------------------
# Round 1: All candidate pairs (same amount, date within 5 days).
#          Score by check_match > date_proximity > description_similarity.
#          Keep mutual best matches (both sides rank each other #1).
# Round 2: Re-match leftovers from round 1 conflicts using same scoring.
#
# The 5-day window handles the prior-period check B03 (Dec 30 GL -> Jan 3 bank = 4 days).

MATCH_SQL = """\
CREATE VIEW all_matched AS
WITH
-- All valid candidate pairs: same amount, date within 5 days
candidates AS (
    SELECT
        b.id              AS bank_id,
        g.id              AS gl_id,
        b.amount          AS bank_amount,
        g.amount          AS gl_amount,
        b.date            AS bank_date,
        g.date            AS gl_date,
        b.description     AS bank_desc,
        g.description     AS gl_desc,
        ABS(date_diff('day', b.date, g.date))               AS date_gap,
        jaro_winkler_similarity(b.desc_clean, g.desc_clean)  AS desc_score,
        CASE WHEN b.check_no IS NOT NULL
              AND b.check_no = g.check_no THEN 1 ELSE 0
        END AS check_match
    FROM bank_norm b
    JOIN gl_norm g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 5
),

-- Round 1: rank each side, keep mutual best
r1_ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY bank_id
            ORDER BY check_match DESC, date_gap ASC, desc_score DESC
        ) AS b_rank,
        ROW_NUMBER() OVER (
            PARTITION BY gl_id
            ORDER BY check_match DESC, date_gap ASC, desc_score DESC
        ) AS g_rank
    FROM candidates
),
round1 AS (
    SELECT bank_id, gl_id, bank_amount, gl_amount, bank_date, gl_date,
           bank_desc, gl_desc, date_gap, desc_score, check_match,
           'round1' AS match_round
    FROM r1_ranked
    WHERE b_rank = 1 AND g_rank = 1
),

-- Round 2: re-match leftovers
r2_pool AS (
    SELECT c.*
    FROM candidates c
    WHERE c.bank_id NOT IN (SELECT bank_id FROM round1)
      AND c.gl_id   NOT IN (SELECT gl_id   FROM round1)
),
r2_ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY bank_id
            ORDER BY check_match DESC, date_gap ASC, desc_score DESC
        ) AS b_rank,
        ROW_NUMBER() OVER (
            PARTITION BY gl_id
            ORDER BY check_match DESC, date_gap ASC, desc_score DESC
        ) AS g_rank
    FROM r2_pool
),
round2 AS (
    SELECT bank_id, gl_id, bank_amount, gl_amount, bank_date, gl_date,
           bank_desc, gl_desc, date_gap, desc_score, check_match,
           'round2' AS match_round
    FROM r2_ranked
    WHERE b_rank = 1 AND g_rank = 1
)

SELECT *, 'exact_1to1' AS match_type, '' AS note
FROM round1
UNION ALL
SELECT *, 'exact_1to1' AS match_type, '' AS note
FROM round2;
"""

MATCH_VALIDATE_SQL = """\
CREATE VIEW match__validation AS
SELECT 'fail' AS status,
       'bank_id ' || bank_id || ' matched ' || cnt || ' times' AS message
FROM (SELECT bank_id, COUNT(*) AS cnt FROM all_matched GROUP BY bank_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'gl_id ' || gl_id || ' matched ' || cnt || ' times' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM all_matched GROUP BY gl_id)
WHERE cnt > 1;
"""


# ---------------------------------------------------------------------------
# Task 3 — Detect offsetting GL pairs
# ---------------------------------------------------------------------------
# Among unmatched GL entries, find pairs that cancel each other:
# amount_a + amount_b = 0, within 10-day window, with description similarity.

OFFSETTING_SQL = """\
CREATE VIEW offsetting_pairs AS
WITH unmatched_gl AS (
    SELECT g.*
    FROM gl_norm g
    LEFT JOIN all_matched m ON g.id = m.gl_id
    WHERE m.gl_id IS NULL
)
SELECT
    a.id              AS original_id,
    b.id              AS reversal_id,
    a.description     AS original_desc,
    b.description     AS reversal_desc,
    a.amount          AS original_amount,
    b.amount          AS reversal_amount,
    a.date            AS original_date,
    b.date            AS reversal_date,
    a.ref             AS original_ref,
    b.ref             AS reversal_ref
FROM unmatched_gl a
JOIN unmatched_gl b
  ON a.amount + b.amount = 0
 AND a.amount < 0                                     -- debit is "original"
 AND a.id < b.id                                      -- deduplicate
 AND ABS(date_diff('day', a.date, b.date)) <= 10
 AND jaro_winkler_similarity(a.desc_clean, b.desc_clean) > 0.4;
"""


# ---------------------------------------------------------------------------
# Task 4 — Report
# ---------------------------------------------------------------------------
# Final reconciliation views: matched, unmatched bank, unmatched GL, summary.

REPORT_SQL = """\
CREATE VIEW report_matched AS
SELECT
    m.bank_id,
    m.gl_id,
    m.bank_date,
    m.gl_date,
    m.bank_amount,
    m.gl_amount,
    m.bank_desc,
    m.gl_desc,
    m.date_gap,
    ROUND(m.desc_score, 3) AS desc_score,
    m.check_match,
    m.match_round,
    m.match_type,
    m.note,
    CASE
        WHEN m.check_match = 1          THEN 'check_number'
        WHEN m.date_gap = 0             THEN 'same_date'
        WHEN m.date_gap <= 2            THEN 'close_date'
        ELSE 'date_window'
    END AS match_confidence
FROM all_matched m
ORDER BY m.bank_date, m.bank_id;

CREATE VIEW report_unmatched_bank AS
SELECT
    b.id, b.date, b.description, b.amount
FROM bank_norm b
LEFT JOIN all_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL
ORDER BY b.date;

CREATE VIEW report_unmatched_gl AS
WITH offsetting_ids AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL
    SELECT reversal_id FROM offsetting_pairs
)
SELECT
    g.id, g.date, g.description, g.amount, g.ref, g.entry_type
FROM gl_norm g
LEFT JOIN all_matched m ON g.id = m.gl_id
LEFT JOIN offsetting_ids oi ON g.id = oi.id
WHERE m.gl_id IS NULL
  AND oi.id IS NULL
ORDER BY g.date;

CREATE VIEW report_summary AS
SELECT
    (SELECT COUNT(*) FROM bank_norm)              AS bank_count,
    (SELECT COUNT(*) FROM gl_norm)                AS gl_count,
    (SELECT COUNT(*) FROM all_matched)            AS matched_pairs,
    (SELECT COUNT(*) FROM report_unmatched_bank)  AS unmatched_bank,
    (SELECT COALESCE(SUM(amount), 0)
       FROM report_unmatched_bank)                AS unmatched_bank_total,
    (SELECT COUNT(*) FROM report_unmatched_gl)    AS unmatched_gl,
    (SELECT COALESCE(SUM(amount), 0)
       FROM report_unmatched_gl)                  AS unmatched_gl_total,
    (SELECT COUNT(*) FROM offsetting_pairs)       AS offsetting_pairs;
"""

REPORT_VALIDATE_SQL = """\
CREATE VIEW report__validation AS
SELECT 'fail' AS status,
       'Bank count mismatch: ' || bc || ' total != ' || mc || ' matched + ' || ub || ' unmatched'
       AS message
FROM (
    SELECT
        (SELECT COUNT(*) FROM bank_norm)              AS bc,
        (SELECT COUNT(*) FROM all_matched)            AS mc,
        (SELECT COUNT(*) FROM report_unmatched_bank)  AS ub
)
WHERE bc != mc + ub
UNION ALL
SELECT 'fail' AS status,
       'GL count mismatch: ' || gc || ' total != ' || mc || ' matched + ' || ug
       || ' unmatched + ' || oc || ' offsetting' AS message
FROM (
    SELECT
        (SELECT COUNT(*) FROM gl_norm)                AS gc,
        (SELECT COUNT(*) FROM all_matched)            AS mc,
        (SELECT COUNT(*) FROM report_unmatched_gl)    AS ug,
        (SELECT COUNT(*) * 2 FROM offsetting_pairs)   AS oc
)
WHERE gc != mc + ug + oc;
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
        "sql": MATCH_SQL,
        "validate_sql": MATCH_VALIDATE_SQL,
        "inputs": ["bank_norm", "gl_norm"],
        "outputs": ["all_matched"],
        "output_columns": {
            "all_matched": [
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
        "inputs": ["gl_norm", "all_matched"],
        "outputs": ["offsetting_pairs"],
        "output_columns": {
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


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def export_recon_report(conn, path):
    """Write reconciliation results to a multi-sheet Excel workbook."""
    from openpyxl import Workbook

    wb = Workbook()
    sheets = [
        ("Matched", "SELECT * FROM report_matched ORDER BY bank_date, bank_id"),
        ("Unmatched Bank", "SELECT * FROM report_unmatched_bank ORDER BY date"),
        ("Unmatched GL", "SELECT * FROM report_unmatched_gl ORDER BY date"),
        ("Offsetting", "SELECT * FROM offsetting_pairs ORDER BY original_date"),
        ("Summary", "SELECT * FROM report_summary"),
    ]

    for i, (name, query) in enumerate(sheets):
        ws = wb.active if i == 0 else wb.create_sheet()
        ws.title = name
        result = conn.execute(query)
        ws.append([col[0] for col in result.description])
        for row in result.fetchall():
            ws.append(list(row))

    wb.save(str(path))


EXPORTS = {"bank_rec_v1_report.xlsx": export_recon_report}
