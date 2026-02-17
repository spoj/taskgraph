"""Bank reconciliation: match bank statement to general ledger.

Demonstrates:
  - Partial description matching (Jaro-Winkler similarity + check-number extraction)
  - Offsetting/correcting entries in GL (original + reversal cancel to zero)
  - Date tolerance matching (0-3 day window)
  - Two-round greedy 1:1 assignment to handle ambiguous same-amount pairs
  - Validation views for data integrity checks

Data: January 2025 cash account — 22 bank transactions, 23 GL entries.

Expected results:
  19 matched pairs, 3 unmatched bank (fees/interest),
  2 unmatched GL (outstanding items), 1 offsetting pair (Initech correction).

DAG:
    bank_txns, gl_entries
            |
        normalize
            |
          match         (amount + date window + description scoring)
            |
        offsetting      (detect self-canceling GL reversal pairs)
            |
          report        (final reconciliation views + summary)

Usage:
    tg run --spec examples/bank_rec.py -o bank_rec.db
    tg show --spec examples/bank_rec.py
"""

from datetime import date


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

bank_txns = [
    # --- wires ---
    {
        "date": date(2025, 1, 2),
        "description": "WIRE OUT REF#98765 ACME CORP",
        "amount": -15_000.00,
        "ref": "W98765",
    },
    {
        "date": date(2025, 1, 6),
        "description": "WIRE IN REF#11223 GLOBEX INC",
        "amount": 28_750.00,
        "ref": "W11223",
    },
    {
        "date": date(2025, 1, 8),
        "description": "WIRE OUT REF#98770 SMITH AND CO",
        "amount": -8_400.00,
        "ref": "W98770",
    },
    {
        "date": date(2025, 1, 13),
        "description": "WIRE IN REF#11230 WAYNE ENT",
        "amount": 52_000.00,
        "ref": "W11230",
    },
    {
        "date": date(2025, 1, 16),
        "description": "WIRE OUT REF#98780 INITECH",
        "amount": -6_200.00,
        "ref": "W98780",
    },
    {
        "date": date(2025, 1, 21),
        "description": "WIRE IN REF#11240 STARK IND",
        "amount": 34_500.00,
        "ref": "W11240",
    },
    {
        "date": date(2025, 1, 22),
        "description": "WIRE OUT REF#98785 ACME CORP",
        "amount": -15_000.00,
        "ref": "W98785",
    },
    {
        "date": date(2025, 1, 27),
        "description": "WIRE OUT REF#98790 OCEANIC AIR",
        "amount": -11_350.00,
        "ref": "W98790",
    },
    # --- ACH ---
    {
        "date": date(2025, 1, 3),
        "description": "ACH DEBIT ADP PAYROLL",
        "amount": -45_230.50,
        "ref": "ACH001",
    },
    {
        "date": date(2025, 1, 7),
        "description": "ACH CREDIT INSURANCE REFUND",
        "amount": 1_200.00,
        "ref": "ACH002",
    },
    {
        "date": date(2025, 1, 15),
        "description": "ACH DEBIT ADP PAYROLL",
        "amount": -45_230.50,
        "ref": "ACH003",
    },
    {
        "date": date(2025, 1, 17),
        "description": "ACH DEBIT OFFICE LEASE Q1",
        "amount": -18_000.00,
        "ref": "ACH004",
    },
    {
        "date": date(2025, 1, 20),
        "description": "ACH DEBIT STAPLES SUPPLIES",
        "amount": -1_825.00,
        "ref": "ACH005",
    },
    {
        "date": date(2025, 1, 23),
        "description": "ACH CREDIT TAX REFUND",
        "amount": 4_750.00,
        "ref": "ACH006",
    },
    # --- checks ---
    {
        "date": date(2025, 1, 3),
        "description": "CHECK #1042",
        "amount": -2_500.00,
        "ref": "CHK1042",
    },
    {
        "date": date(2025, 1, 10),
        "description": "CHECK #1043",
        "amount": -3_150.00,
        "ref": "CHK1043",
    },
    {
        "date": date(2025, 1, 10),
        "description": "CHECK #1044",
        "amount": -975.00,
        "ref": "CHK1044",
    },
    {
        "date": date(2025, 1, 20),
        "description": "CHECK #1045",
        "amount": -1_825.00,
        "ref": "CHK1045",
    },
    {
        "date": date(2025, 1, 28),
        "description": "CHECK #1046",
        "amount": -620.00,
        "ref": "CHK1046",
    },
    # --- fees / interest (no GL match) ---
    {
        "date": date(2025, 1, 14),
        "description": "MONTHLY SERVICE FEE",
        "amount": -45.00,
        "ref": "FEE001",
    },
    {
        "date": date(2025, 1, 29),
        "description": "INTEREST EARNED",
        "amount": 125.30,
        "ref": "INT001",
    },
    {
        "date": date(2025, 1, 31),
        "description": "WIRE TRANSFER FEE",
        "amount": -25.00,
        "ref": "FEE002",
    },
]

gl_entries = [
    # --- normal entries (match bank 1:1 with 0-3 day lag) ---
    {
        "date": date(2025, 1, 2),
        "description": "Pmt to Acme Corporation - Inv 2024-098",
        "amount": -15_000.00,
        "ref": "AP-1001",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 2),
        "description": "Payroll - Jan 1-15 via ADP",
        "amount": -45_230.50,
        "ref": "PR-0101",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 2),
        "description": "Check #1042 Office Depot supplies",
        "amount": -2_500.00,
        "ref": "AP-1002",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 5),
        "description": "Rcvd from Globex Inc - Inv 2024-087",
        "amount": 28_750.00,
        "ref": "AR-2001",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 6),
        "description": "Insurance premium refund - Progressive",
        "amount": 1_200.00,
        "ref": "JE-3001",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 7),
        "description": "Pmt to Smith & Co Consulting - Dec svcs",
        "amount": -8_400.00,
        "ref": "AP-1003",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 9),
        "description": "Check #1043 Staples toner order",
        "amount": -3_150.00,
        "ref": "AP-1004",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 9),
        "description": "Check #1044 FedEx shipping charges",
        "amount": -975.00,
        "ref": "AP-1005",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 10),
        "description": "Rcvd from Wayne Enterprises - Project Alpha",
        "amount": 52_000.00,
        "ref": "AR-2002",
        "entry_type": "normal",
    },
    # --- correcting entries: Initech (original wrong, reversal, then correct) ---
    {
        "date": date(2025, 1, 14),
        "description": "Pmt to Initech LLC - Inv 2024-112",
        "amount": -6_800.00,
        "ref": "AP-1006",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 14),
        "description": "REVERSE: Pmt to Initech LLC - wrong amount",
        "amount": 6_800.00,
        "ref": "AP-1006R",
        "entry_type": "reversal",
    },
    {
        "date": date(2025, 1, 15),
        "description": "Pmt to Initech LLC - Inv 2024-112 corrected",
        "amount": -6_200.00,
        "ref": "AP-1006C",
        "entry_type": "correction",
    },
    # --- more normal entries ---
    {
        "date": date(2025, 1, 15),
        "description": "Payroll - Jan 16-31 via ADP",
        "amount": -45_230.50,
        "ref": "PR-0115",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 16),
        "description": "Office lease payment Q1 2025",
        "amount": -18_000.00,
        "ref": "AP-1007",
        "entry_type": "normal",
    },
    # --- two GL entries at same amount (-1825) to test description disambiguation ---
    {
        "date": date(2025, 1, 19),
        "description": "Check #1045 Amazon Web Services",
        "amount": -1_825.00,
        "ref": "AP-1008",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 18),
        "description": "Staples office supplies reorder",
        "amount": -1_825.00,
        "ref": "AP-1009",
        "entry_type": "normal",
    },
    # --- more normal entries ---
    {
        "date": date(2025, 1, 20),
        "description": "Rcvd from Stark Industries - Inv 2025-003",
        "amount": 34_500.00,
        "ref": "AR-2003",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 21),
        "description": "Pmt to Acme Corporation - Inv 2025-001",
        "amount": -15_000.00,
        "ref": "AP-1010",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 22),
        "description": "State tax refund received",
        "amount": 4_750.00,
        "ref": "JE-3002",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 27),
        "description": "Pmt to Oceanic Airlines - Corp travel Jan",
        "amount": -11_350.00,
        "ref": "AP-1011",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 27),
        "description": "Check #1046 Uber Business charges",
        "amount": -620.00,
        "ref": "AP-1012",
        "entry_type": "normal",
    },
    # --- outstanding: posted in GL but not yet cleared on bank ---
    {
        "date": date(2025, 1, 30),
        "description": "Check #1047 Grainger industrial supplies",
        "amount": -2_100.00,
        "ref": "AP-1013",
        "entry_type": "normal",
    },
    {
        "date": date(2025, 1, 31),
        "description": "Pmt to Dunder Mifflin - paper supplies",
        "amount": -450.00,
        "ref": "AP-1014",
        "entry_type": "normal",
    },
]

INPUTS = {
    "bank_txns": {
        "data": bank_txns,
        "columns": ["date", "description", "amount", "ref"],
    },
    "gl_entries": {
        "data": gl_entries,
        "columns": ["date", "description", "amount", "ref", "entry_type"],
    },
}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

# Task 1 — Normalize descriptions for matching.
# Strips punctuation, uppercases, extracts check numbers.

NORMALIZE_SQL = """\
CREATE VIEW bank_norm AS
SELECT
    _row_id,
    date,
    description,
    amount,
    ref,
    UPPER(regexp_replace(description, '[^A-Za-z0-9 ]', ' ', 'g')) AS desc_clean,
    TRY_CAST(regexp_extract(description, '#(\\d+)', 1) AS INTEGER) AS check_no
FROM bank_txns;

CREATE VIEW gl_norm AS
SELECT
    _row_id,
    date,
    description,
    amount,
    ref,
    entry_type,
    UPPER(regexp_replace(description, '[^A-Za-z0-9 ]', ' ', 'g')) AS desc_clean,
    TRY_CAST(regexp_extract(description, '#(\\d+)', 1) AS INTEGER) AS check_no
FROM gl_entries;
"""

# Task 2 — Match bank to GL: exact amount + 0-3 day window.
# Uses two-round greedy assignment:
#   Round 1: both sides agree on best partner (mutual best match).
#   Round 2: re-match any leftovers from round 1 conflicts.
# Scoring priority: check-number match > date proximity > description similarity.

MATCH_SQL = """\
CREATE VIEW matched AS
WITH
-- All valid candidate pairs: same amount, date within 3 days
candidates AS (
    SELECT
        b._row_id       AS bank_id,
        g._row_id       AS gl_id,
        b.date          AS bank_date,
        g.date          AS gl_date,
        b.amount,
        b.description   AS bank_desc,
        g.description   AS gl_desc,
        ABS(date_diff('day', b.date, g.date))            AS date_diff,
        jaro_winkler_similarity(b.desc_clean, g.desc_clean) AS desc_score,
        CASE WHEN b.check_no IS NOT NULL
              AND b.check_no = g.check_no THEN 1 ELSE 0
        END AS check_match
    FROM bank_norm b
    JOIN gl_norm g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 3
),

-- Round 1: rank each side, keep mutual best
r1 AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY bank_id
            ORDER BY check_match DESC, date_diff, desc_score DESC
        ) AS b_rank,
        ROW_NUMBER() OVER (
            PARTITION BY gl_id
            ORDER BY check_match DESC, date_diff, desc_score DESC
        ) AS g_rank
    FROM candidates
),
round1 AS (
    SELECT bank_id, gl_id, bank_date, gl_date, amount,
           bank_desc, gl_desc, date_diff, desc_score, check_match
    FROM r1
    WHERE b_rank = 1 AND g_rank = 1
),

-- Round 2: re-match leftovers
r2_pool AS (
    SELECT c.*
    FROM candidates c
    WHERE c.bank_id NOT IN (SELECT bank_id FROM round1)
      AND c.gl_id   NOT IN (SELECT gl_id   FROM round1)
),
r2 AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY bank_id
            ORDER BY check_match DESC, date_diff, desc_score DESC
        ) AS b_rank,
        ROW_NUMBER() OVER (
            PARTITION BY gl_id
            ORDER BY check_match DESC, date_diff, desc_score DESC
        ) AS g_rank
    FROM r2_pool
),
round2 AS (
    SELECT bank_id, gl_id, bank_date, gl_date, amount,
           bank_desc, gl_desc, date_diff, desc_score, check_match
    FROM r2
    WHERE b_rank = 1 AND g_rank = 1
)

SELECT * FROM round1
UNION ALL
SELECT * FROM round2;
"""

MATCH_VALIDATE_SQL = """\
CREATE VIEW match__validation AS
SELECT 'fail' AS status,
       'bank txn ' || bank_id || ' matched ' || cnt || ' times' AS message
FROM (SELECT bank_id, COUNT(*) AS cnt FROM matched GROUP BY bank_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'GL entry ' || gl_id || ' matched ' || cnt || ' times' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM matched GROUP BY gl_id)
WHERE cnt > 1;
"""

# Task 3 — Identify offsetting GL entries.
# Finds unmatched GL entries that cancel each other (amount + amount = 0)
# within a 5-day window and with some description similarity.

OFFSETTING_SQL = """\
CREATE VIEW offsetting_pairs AS
WITH unmatched AS (
    SELECT g.*
    FROM gl_norm g
    LEFT JOIN matched m ON g._row_id = m.gl_id
    WHERE m.gl_id IS NULL
)
SELECT
    a._row_id     AS original_id,
    b._row_id     AS reversal_id,
    a.description AS original_desc,
    b.description AS reversal_desc,
    a.amount      AS original_amount,
    b.amount      AS reversal_amount,
    a.date        AS original_date,
    b.date        AS reversal_date,
    a.ref         AS original_ref,
    b.ref         AS reversal_ref
FROM unmatched a
JOIN unmatched b
  ON a.amount + b.amount = 0
 AND a.amount < 0                                   -- debit = "original"
 AND a._row_id < b._row_id                          -- deduplicate
 AND ABS(date_diff('day', a.date, b.date)) <= 5
 AND jaro_winkler_similarity(a.desc_clean, b.desc_clean) > 0.5;
"""

# Task 4 — Final reconciliation report.
# Produces matched pairs, unmatched bank/GL, and summary statistics.

REPORT_SQL = """\
CREATE VIEW report_matched AS
SELECT
    m.bank_id,
    m.gl_id,
    m.bank_date,
    m.gl_date,
    m.amount,
    m.bank_desc,
    m.gl_desc,
    m.date_diff,
    ROUND(m.desc_score, 3) AS desc_score,
    CASE
        WHEN m.check_match = 1          THEN 'check_number'
        WHEN m.date_diff = 0            THEN 'same_date'
        ELSE 'date_proximity'
    END AS match_confidence
FROM matched m
ORDER BY m.bank_date;

CREATE VIEW report_unmatched_bank AS
SELECT
    b._row_id AS bank_id, b.date, b.description, b.amount, b.ref
FROM bank_norm b
LEFT JOIN matched m ON b._row_id = m.bank_id
WHERE m.bank_id IS NULL
ORDER BY b.date;

CREATE VIEW report_unmatched_gl AS
WITH offsetting_ids AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL
    SELECT reversal_id FROM offsetting_pairs
)
SELECT
    g._row_id AS gl_id, g.date, g.description, g.amount, g.ref, g.entry_type
FROM gl_norm g
LEFT JOIN matched m ON g._row_id = m.gl_id
LEFT JOIN offsetting_ids oi ON g._row_id = oi.id
WHERE m.gl_id IS NULL
  AND oi.id IS NULL
ORDER BY g.date;

CREATE VIEW report_summary AS
SELECT
    (SELECT COUNT(*) FROM bank_norm)            AS bank_entries,
    (SELECT COUNT(*) FROM gl_norm)              AS gl_entries,
    (SELECT COUNT(*) FROM matched)              AS matched_pairs,
    (SELECT COUNT(*) FROM report_unmatched_bank) AS unmatched_bank,
    (SELECT ROUND(COALESCE(SUM(amount), 0), 2)
       FROM report_unmatched_bank)              AS unmatched_bank_net,
    (SELECT COUNT(*) FROM report_unmatched_gl)  AS unmatched_gl,
    (SELECT ROUND(COALESCE(SUM(amount), 0), 2)
       FROM report_unmatched_gl)                AS unmatched_gl_net,
    (SELECT COUNT(*) FROM offsetting_pairs)     AS offsetting_pairs,
    (SELECT ROUND(COALESCE(SUM(ABS(original_amount)), 0), 2)
       FROM offsetting_pairs)                   AS offsetting_gross;
"""

REPORT_VALIDATE_SQL = """\
CREATE VIEW report__validation AS
SELECT 'fail' AS status,
       'Bank count mismatch: ' || bt || ' != ' || mc || ' matched + ' || ub || ' unmatched'
       AS message
FROM (
    SELECT
        (SELECT COUNT(*) FROM bank_norm)             AS bt,
        (SELECT COUNT(*) FROM matched)               AS mc,
        (SELECT COUNT(*) FROM report_unmatched_bank) AS ub
)
WHERE bt != mc + ub
UNION ALL
SELECT 'fail' AS status,
       'GL count mismatch: ' || gt || ' != ' || mc || ' matched + ' || ug
       || ' unmatched + ' || oc || ' offsetting' AS message
FROM (
    SELECT
        (SELECT COUNT(*) FROM gl_norm)              AS gt,
        (SELECT COUNT(*) FROM matched)              AS mc,
        (SELECT COUNT(*) FROM report_unmatched_gl)  AS ug,
        (SELECT COUNT(*) * 2 FROM offsetting_pairs) AS oc
)
WHERE gt != mc + ug + oc;
"""


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
        "outputs": ["matched"],
        "output_columns": {
            "matched": [
                "bank_id",
                "gl_id",
                "amount",
                "bank_date",
                "gl_date",
                "desc_score",
            ],
        },
    },
    {
        "name": "offsetting",
        "sql": OFFSETTING_SQL,
        "inputs": ["gl_norm", "matched"],
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
        "inputs": ["bank_norm", "gl_norm", "matched", "offsetting_pairs"],
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
        ("Matched", "SELECT * FROM report_matched ORDER BY bank_date"),
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


EXPORTS = {"bank_rec_report.xlsx": export_recon_report}
