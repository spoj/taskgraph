"""Bank Reconciliation V3 — Redesigned: SQL-heavy pipeline with focused LLM

Architecture: push as much as possible into deterministic SQL nodes.
The LLM prompt node handles only residual ambiguous cases.

DAG:
    bank_txns, gl_entries
            |
        features           (sql: entity extraction, check numbers, clean text)
            |
        match_certain      (sql: check# match + entity-based 1:1)
            |
        offsetting         (sql: self-canceling GL pairs)
            |
        batch_match        (sql: entity-grouped batch deposits)
            |
        tolerance          (sql: entity-prefix + small amount diff)
            |
        amount_match       (sql: exact amount + date for cryptic descriptions)
            |
       match_residual      (prompt: multi-deposit subsets, ambiguous remaining)
            |
          report           (sql)

Key design decisions:
  - Entity extraction in SQL, not in the LLM prompt
  - 1:1 matching uses entity prefix match, not Jaro-Winkler
  - Batch deposits are pure SQL: group GL by entity, sum, match to bank
  - LLM only sees pre-structured residual candidates
"""

from examples.bank_rec_problem import BANK_TRANSACTIONS, GL_ENTRIES

# ---------------------------------------------------------------------------
# Node 1 — Features  (sql)
# ---------------------------------------------------------------------------

FEATURES_SQL = """\
CREATE VIEW features_bank AS
WITH raw AS (
    SELECT *, upper(description) AS udesc
    FROM bank_txns
),
step1 AS (
    SELECT *,
        -- Cleaned description (letters/digits/spaces only)
        regexp_replace(udesc, '[^A-Z0-9 ]', ' ', 'g') AS desc_clean,
        -- Check number (from "CHECK #1234" or "CHECK 1234")
        TRY_CAST(regexp_extract(udesc, 'CHECK\\s*#?\\s*(\\d+)', 1) AS INTEGER) AS check_no,
        -- Raw entity: strip ACH/WIRE prefix and trailing wire-ref numbers
        trim(regexp_replace(
            regexp_replace(
                regexp_replace(udesc,
                    '^(ACH (CREDIT|DEBIT|CR|DR)|WIRE TRF (IN|OUT))\\s+', ''),
                '\\d{6,}\\s*', ' '),
            '\\s+', ' ', 'g'
        )) AS entity_raw
    FROM raw
)
SELECT
    id, date, description, amount, desc_clean, check_no, entity_raw,
    -- Normalized entity: strip legal suffixes, then strip trailing 1-3 char fragments
    -- (caused by bank description truncation cutting legal suffixes mid-word)
    -- Use (^|\\s)..(\\s|$) to avoid stripping "CO" from "CONSULTING" etc
    trim(regexp_replace(
        regexp_replace(
            regexp_replace(
                regexp_replace(entity_raw,
                    '(^|\\s)(INCORPORATED|INC|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|LP|LLP|PLC)(\\s|$)', ' ', 'g'),
                '\\s+[A-Z]{1,3}$', ''),
            '\\s+', ' ', 'g'),
        '\\s+$', ''
    )) AS entity_norm
FROM step1;

CREATE VIEW features_gl AS
WITH raw AS (
    SELECT *,
        -- Entity = text before first ' - ' separator
        CASE WHEN strpos(description, ' - ') > 0
             THEN trim(substr(description, 1, strpos(description, ' - ') - 1))
             ELSE description END AS entity_full
    FROM gl_entries
)
SELECT
    id, date, description, amount, ref, entry_type,
    regexp_replace(upper(description), '[^A-Z0-9 ]', ' ', 'g') AS desc_clean,
    -- Check number from ref
    TRY_CAST(regexp_extract(ref, 'CHK-(\\d+)', 1) AS INTEGER) AS check_no,
    upper(entity_full) AS entity_raw,
    -- Normalized: strip legal suffixes (no trailing fragment issue for GL — full names)
    -- Use (^|\\s)..(\\s|$) to avoid stripping "CO" from "CONSULTING" etc
    trim(regexp_replace(
        regexp_replace(
            upper(entity_full),
            '(^|\\s)(INCORPORATED|INC|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|LP|LLP|PLC)(\\s|$)', ' ', 'g'),
        '\\s+', ' ', 'g'
    )) AS entity_norm
FROM raw;
"""


# ---------------------------------------------------------------------------
# Node 2 — Certain 1:1 matches  (sql)
# ---------------------------------------------------------------------------

MATCH_CERTAIN_SQL = """\
-- Round 1: Check-number matches (highest confidence)
CREATE VIEW match_certain_r1 AS
SELECT
    b.id AS bank_id, g.id AS gl_id,
    b.amount AS bank_amount, g.amount AS gl_amount,
    'check_number' AS match_type,
    'Check #' || b.check_no AS note
FROM features_bank b
JOIN features_gl g
  ON b.check_no = g.check_no
 AND b.amount = g.amount
WHERE b.check_no IS NOT NULL AND g.check_no IS NOT NULL;

-- Round 2: Entity-based exact-amount matches
-- Uses prefix matching: bank entity (possibly truncated) must be a prefix of GL entity
-- Only match when the bank item has exactly ONE best candidate to avoid swaps
CREATE VIEW match_certain_r2 AS
WITH remaining_bank AS (
    SELECT b.* FROM features_bank b
    LEFT JOIN match_certain_r1 m ON b.id = m.bank_id
    WHERE m.bank_id IS NULL
),
remaining_gl AS (
    SELECT g.* FROM features_gl g
    LEFT JOIN match_certain_r1 m ON g.id = m.gl_id
    WHERE m.gl_id IS NULL
),
candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS bank_amount, g.amount AS gl_amount,
        b.date AS bank_date, g.date AS gl_date,
        b.entity_norm AS bank_entity, g.entity_norm AS gl_entity,
        ABS(date_diff('day', b.date, g.date)) AS date_gap,
        -- Entity match quality: length of matching prefix relative to bank entity
        length(b.entity_norm) AS bank_ent_len
    FROM remaining_bank b
    JOIN remaining_gl g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 10
     -- Entity prefix match: bank entity is prefix of GL entity (handles truncation)
     AND starts_with(g.entity_norm, b.entity_norm)
     AND length(b.entity_norm) >= 4  -- avoid trivially short matches
),
-- For each bank item, count how many GL candidates match
bank_counts AS (
    SELECT bank_id, COUNT(*) AS n_candidates
    FROM candidates GROUP BY bank_id
),
-- For each GL item, count how many bank candidates match
gl_counts AS (
    SELECT gl_id, COUNT(*) AS n_candidates
    FROM candidates GROUP BY gl_id
),
-- Rank: prefer closest date, then longest entity match
ranked AS (
    SELECT c.*,
        ROW_NUMBER() OVER (PARTITION BY c.bank_id ORDER BY c.date_gap, c.bank_ent_len DESC) AS b_rank,
        ROW_NUMBER() OVER (PARTITION BY c.gl_id   ORDER BY c.date_gap, c.bank_ent_len DESC) AS g_rank,
        bc.n_candidates AS bank_n,
        gc.n_candidates AS gl_n
    FROM candidates c
    JOIN bank_counts bc ON c.bank_id = bc.bank_id
    JOIN gl_counts gc ON c.gl_id = gc.gl_id
)
SELECT
    bank_id, gl_id, bank_amount, gl_amount,
    'exact_1to1' AS match_type,
    '' AS note
FROM ranked
WHERE b_rank = 1 AND g_rank = 1
  -- Only match when unambiguous: either 1 candidate, or top choice is mutual best
  AND (bank_n = 1 OR gl_n = 1);

-- Combined certain matches
CREATE VIEW match_certain_matched AS
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM match_certain_r1
UNION ALL
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM match_certain_r2;

-- Remaining pools
CREATE VIEW match_certain_unmatched_bank AS
SELECT b.* FROM features_bank b
LEFT JOIN match_certain_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL;

CREATE VIEW match_certain_unmatched_gl AS
SELECT g.* FROM features_gl g
LEFT JOIN match_certain_matched m ON g.id = m.gl_id
WHERE m.gl_id IS NULL;
"""

MATCH_CERTAIN_VALIDATE = """\
SELECT 'fail' AS status,
       'bank_id ' || bank_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT bank_id, COUNT(*) AS cnt FROM match_certain_matched GROUP BY bank_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM match_certain_matched GROUP BY gl_id)
WHERE cnt > 1;
"""


# ---------------------------------------------------------------------------
# Node 3 — Offsetting GL pairs  (sql)
# ---------------------------------------------------------------------------

OFFSETTING_SQL = """\
CREATE VIEW offsetting_pairs AS
WITH unmatched_gl AS (
    SELECT g.* FROM features_gl g
    LEFT JOIN match_certain_matched m ON g.id = m.gl_id
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
# Node 4 — Batch deposits  (sql)
# ---------------------------------------------------------------------------

BATCH_MATCH_SQL = """\
-- Batch deposit matching: group GL entries by entity, match sums to bank deposits
CREATE VIEW batch_match_matched AS
WITH
-- Exclude offsetting GL from the pool
offset_gl AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
),
-- Unmatched positive bank items (deposits)
bank_pool AS (
    SELECT b.*
    FROM match_certain_unmatched_bank b
    WHERE b.amount > 0
),
-- Unmatched positive GL items (credits), excluding offsets
gl_pool AS (
    SELECT g.*
    FROM match_certain_unmatched_gl g
    LEFT JOIN offset_gl o ON g.id = o.id
    WHERE o.id IS NULL
      AND g.amount > 0
      AND g.entry_type = 'normal'
),
-- Group GL credits by normalized entity
gl_groups AS (
    SELECT
        entity_norm,
        COUNT(*) AS gl_cnt,
        SUM(amount) AS gl_sum,
        MIN(date) AS gl_min_date,
        MAX(date) AS gl_max_date
    FROM gl_pool
    GROUP BY entity_norm
    HAVING COUNT(*) BETWEEN 2 AND 5
),
-- Match bank deposits to GL groups via entity prefix + amount
-- Bank entity (possibly truncated) is prefix of GL group entity
matched_groups AS (
    SELECT
        b.id AS bank_id,
        b.amount AS bank_amount,
        b.date AS bank_date,
        g.entity_norm AS gl_entity,
        g.gl_cnt,
        g.gl_sum
    FROM bank_pool b
    JOIN gl_groups g
      ON starts_with(g.entity_norm, b.entity_norm)
     AND ABS(g.gl_sum - b.amount) < 0.01
     AND b.date BETWEEN g.gl_min_date - INTERVAL 7 DAY
                     AND g.gl_max_date + INTERVAL 7 DAY
    -- Each bank item matches at most one group
    QUALIFY ROW_NUMBER() OVER (PARTITION BY b.id ORDER BY g.gl_cnt) = 1
)
-- Explode: one row per GL entry in matched groups
SELECT
    mg.bank_id,
    gp.id AS gl_id,
    mg.bank_amount,
    gp.amount AS gl_amount,
    'batch' AS match_type,
    'Batch: ' || mg.gl_cnt || ' items, entity=' || mg.gl_entity AS note
FROM matched_groups mg
JOIN gl_pool gp ON gp.entity_norm = mg.gl_entity;

-- Remaining unmatched after batch
CREATE VIEW batch_match_remaining_bank AS
SELECT b.* FROM match_certain_unmatched_bank b
LEFT JOIN batch_match_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL;

CREATE VIEW batch_match_remaining_gl AS
WITH offset_gl AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
)
SELECT g.* FROM match_certain_unmatched_gl g
LEFT JOIN batch_match_matched m ON g.id = m.gl_id
LEFT JOIN offset_gl o ON g.id = o.id
WHERE m.gl_id IS NULL AND o.id IS NULL;
"""

BATCH_MATCH_VALIDATE = """\
SELECT 'fail' AS status,
       'batch bank_id ' || bank_id || ' gl_sum mismatch: ' ||
       ROUND(gl_sum, 2) || ' vs ' || bank_amt AS message
FROM (
    SELECT bank_id, SUM(gl_amount) AS gl_sum, MAX(bank_amount) AS bank_amt
    FROM batch_match_matched GROUP BY bank_id
)
WHERE ABS(gl_sum - bank_amt) > 0.01
UNION ALL
SELECT 'fail' AS status,
       'batch gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM batch_match_matched GROUP BY gl_id)
WHERE cnt > 1;
"""


# ---------------------------------------------------------------------------
# Node 4b — Tolerance matching  (sql)
# ---------------------------------------------------------------------------

TOLERANCE_SQL = """\
-- Tolerance matches: same entity, small amount difference (wire fees, processing fees)
CREATE VIEW tolerance_matched AS
WITH
-- Exclude offsetting GL from the pool
offset_gl AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
),
-- Bank items unmatched after certain + batch
bank_pool AS (
    SELECT b.*
    FROM match_certain_unmatched_bank b
    LEFT JOIN batch_match_matched bm ON b.id = bm.bank_id
    WHERE bm.bank_id IS NULL
),
-- GL items unmatched after certain + batch + offsets
gl_pool AS (
    SELECT g.*
    FROM match_certain_unmatched_gl g
    LEFT JOIN batch_match_matched bm ON g.id = bm.gl_id
    LEFT JOIN offset_gl o ON g.id = o.id
    WHERE bm.gl_id IS NULL AND o.id IS NULL
),
-- Find entity-prefix candidates with small amount difference
candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS bank_amount, g.amount AS gl_amount,
        ABS(b.amount - g.amount) AS amt_diff,
        ABS(date_diff('day', b.date, g.date)) AS date_gap,
        b.entity_norm AS bank_entity, g.entity_norm AS gl_entity
    FROM bank_pool b
    JOIN gl_pool g
      ON ABS(b.amount - g.amount) BETWEEN 0.01 AND 100
     AND (b.amount > 0) = (g.amount > 0)  -- same sign
     AND ABS(date_diff('day', b.date, g.date)) <= 10
     AND starts_with(g.entity_norm, b.entity_norm)
     AND length(b.entity_norm) >= 4
),
-- Greedy: smallest diff first, each item used once
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY bank_id ORDER BY amt_diff, date_gap) AS b_rank,
        ROW_NUMBER() OVER (PARTITION BY gl_id ORDER BY amt_diff, date_gap) AS g_rank
    FROM candidates
)
SELECT
    bank_id, gl_id, bank_amount, gl_amount,
    'tolerance' AS match_type,
    'Fee diff: $' || ROUND(amt_diff, 2) AS note
FROM ranked
WHERE b_rank = 1 AND g_rank = 1;

-- Remaining unmatched after tolerance
CREATE VIEW tolerance_remaining_bank AS
SELECT b.* FROM batch_match_remaining_bank b
LEFT JOIN tolerance_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL;

CREATE VIEW tolerance_remaining_gl AS
SELECT g.* FROM batch_match_remaining_gl g
LEFT JOIN tolerance_matched m ON g.id = m.gl_id
WHERE m.gl_id IS NULL;
"""

TOLERANCE_VALIDATE = """\
SELECT 'fail' AS status,
       'tolerance bank_id ' || bank_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT bank_id, COUNT(*) AS cnt FROM tolerance_matched GROUP BY bank_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'tolerance gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM tolerance_matched GROUP BY gl_id)
WHERE cnt > 1;
"""


# ---------------------------------------------------------------------------
# Node 4c — Amount-based matching for cryptic descriptions  (sql)
# ---------------------------------------------------------------------------

AMOUNT_MATCH_SQL = """\
-- Match bank items with cryptic descriptions (entity extraction failed) to GL
-- by exact amount + date proximity. This catches DEPOSIT, AMZN, GOOG, airline
-- codes, etc. where entity prefix matching is impossible.
--
-- Strategy: find all (bank, GL) pairs with exact amount match within 10 days,
-- then greedily assign using Hungarian-style mutual-best matching.
-- Exclude bank fees which have no GL counterpart.

CREATE VIEW amount_match_matched AS
WITH
-- Exclude bank-generated fees/income (no GL counterpart)
bank_pool AS (
    SELECT b.*
    FROM tolerance_remaining_bank b
    WHERE b.entity_norm NOT IN (
        'SERVICE CHARGE', 'MONTHLY MAINTENANCE', 'ANALYSIS CHARGE',
        'WIRE TRF FEE', 'FOREIGN TXN', 'INTEREST PAYMENT'
    )
    AND b.description NOT ILIKE '%SERVICE CHARGE%'
    AND b.description NOT ILIKE '%MAINTENANCE FEE%'
    AND b.description NOT ILIKE '%INTEREST PAYMENT%'
    AND b.description NOT ILIKE '%FOREIGN TXN FEE%'
    AND b.description NOT ILIKE '%ANALYSIS CHARGE%'
    AND b.description NOT ILIKE '%WIRE TRF FEE%'
),
gl_pool AS (
    SELECT * FROM tolerance_remaining_gl
),
-- All exact-amount candidates within date window
candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS bank_amount, g.amount AS gl_amount,
        ABS(date_diff('day', b.date, g.date)) AS date_gap,
        b.date AS bank_date, g.date AS gl_date,
        b.description AS bank_desc, g.description AS gl_desc
    FROM bank_pool b
    JOIN gl_pool g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 10
),
-- Count candidates per bank item and per GL item
bank_counts AS (
    SELECT bank_id, COUNT(*) AS n FROM candidates GROUP BY bank_id
),
gl_counts AS (
    SELECT gl_id, COUNT(*) AS n FROM candidates GROUP BY gl_id
),
-- Rank by date proximity (closest first)
ranked AS (
    SELECT c.*,
        ROW_NUMBER() OVER (PARTITION BY c.bank_id ORDER BY c.date_gap, c.gl_id) AS b_rank,
        ROW_NUMBER() OVER (PARTITION BY c.gl_id ORDER BY c.date_gap, c.bank_id) AS g_rank,
        bc.n AS bank_n,
        gc.n AS gl_n
    FROM candidates c
    JOIN bank_counts bc ON c.bank_id = bc.bank_id
    JOIN gl_counts gc ON c.gl_id = gc.gl_id
)
-- Match when: unique candidate (1:1), OR mutual-best (both rank #1)
SELECT
    bank_id, gl_id, bank_amount, gl_amount,
    'amount_match' AS match_type,
    'Amount+date match (gap=' || date_gap || 'd)' AS note
FROM ranked
WHERE b_rank = 1 AND g_rank = 1
  AND (bank_n = 1 OR gl_n = 1 OR (b_rank = 1 AND g_rank = 1));

-- Remaining unmatched after amount matching
CREATE VIEW amount_match_remaining_bank AS
SELECT b.* FROM tolerance_remaining_bank b
LEFT JOIN amount_match_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL;

CREATE VIEW amount_match_remaining_gl AS
SELECT g.* FROM tolerance_remaining_gl g
LEFT JOIN amount_match_matched m ON g.id = m.gl_id
WHERE m.gl_id IS NULL;
"""

AMOUNT_MATCH_VALIDATE = """\
SELECT 'fail' AS status,
       'amount_match bank_id ' || bank_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT bank_id, COUNT(*) AS cnt FROM amount_match_matched GROUP BY bank_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'amount_match gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM amount_match_matched GROUP BY gl_id)
WHERE cnt > 1;
"""


# ---------------------------------------------------------------------------
# Node 5 — Residual matching  (prompt)
# ---------------------------------------------------------------------------

MATCH_RESIDUAL_INTENT = """\
Resolve remaining unmatched bank and GL items. Prior SQL nodes have already
handled: exact check-number matches, entity-based 1:1 matches, offsetting
GL pairs, entity-based batch deposits, entity-based tolerance matches,
AND amount-based 1:1 matches for cryptic bank descriptions.

What remains are the HARDEST cases that SQL couldn't resolve:
- Multi-deposit batches (multiple bank deposits from same vendor, need subset-sum)
- Ambiguous multi-candidate amount matches (multiple bank items compete for
  same GL items at the same amount)
- Tolerance matches where entity extraction failed

AVAILABLE VIEWS (query these to understand what's left):
  amount_match_remaining_bank — unmatched bank items (after all SQL matching)
  amount_match_remaining_gl   — unmatched GL items (after all SQL matching)
  match_certain_matched       — already-matched (check# + entity 1:1)
  batch_match_matched         — already-matched batch deposits
  tolerance_matched           — already-matched tolerance pairs
  amount_match_matched        — already-matched by amount+date

TASKS (in order of priority):

1) MULTI-DEPOSIT BATCH MATCHING.
   Look for unmatched bank items with clear entity names (ACH CREDIT ...,
   WIRE TRF IN ...) and positive amounts. For each, find unmatched GL entries
   with matching entity_norm (using starts_with for prefix matching).
   If the full set of GL entries for that entity doesn't sum to the bank amount,
   try finding a SUBSET of 2-5 GL entries that does sum to the bank amount.
   IMPORTANT: there may be MULTIPLE bank deposits from the same entity — each
   needs a different subset of GL entries. Match them greedily: largest bank
   amount first, assign GL entries, then next bank amount with remaining GL.

2) AMBIGUOUS AMOUNT MATCHES.
   The amount_match SQL node handled cases where a bank item has exactly ONE
   GL candidate at the same amount. What's left are cases where MULTIPLE bank
   items share the same amount with MULTIPLE GL items (e.g., two bank items
   at -$1450 and two GL items at -$1450). Use description hints, date proximity,
   and cross-elimination to assign. Be conservative — only match when confident.

3) REMAINING TOLERANCE MATCHES.
   Any pairs with ABS(bank_amount - gl_amount) < 100 that SQL missed due
   to entity extraction failure. Match by amount proximity + date.

IMPORTANT — DO NOT MATCH these bank items (they are bank-generated fees):
  - SERVICE CHARGE, MONTHLY MAINTENANCE FEE, ANALYSIS CHARGE, WIRE TRF FEE,
    FOREIGN TXN FEE, INTEREST PAYMENT — no corresponding GL entry exists.

ASSEMBLE the final view combining ALL matches from all sources:

OUTPUT: match_residual_all_matched view with columns:
  bank_id, gl_id, bank_amount, gl_amount, match_type, note
Must include ALL match_certain_matched rows, ALL batch_match_matched rows,
ALL tolerance_matched rows, ALL amount_match_matched rows,
plus any new matches you find.
One row per GL entry. Batch rows: match_type='batch'.

DuckDB NOTES:
- Views are late-binding. Keep view chains shallow (max 2 levels). Use CTEs.
- Do NOT build deep view-of-view stacks.
"""

MATCH_RESIDUAL_VALIDATE = """\
WITH bank_agg AS (
    SELECT bank_id, COUNT(*) AS cnt,
           COUNT(*) FILTER (WHERE match_type != 'batch') AS non_batch,
           SUM(gl_amount) AS gl_sum, MAX(bank_amount) AS bank_amt
    FROM match_residual_all_matched GROUP BY bank_id
)
SELECT 'fail' AS status,
       'bank_id ' || bank_id || ' has invalid multi-match' AS message
FROM bank_agg
WHERE cnt > 1 AND (non_batch > 0 OR ABS(gl_sum - bank_amt) > 0.01)
UNION ALL
SELECT 'fail' AS status,
       'gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM match_residual_all_matched GROUP BY gl_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'Certain match ' || c.bank_id || '->' || c.gl_id || ' missing' AS message
FROM match_certain_matched c
LEFT JOIN match_residual_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL
UNION ALL
SELECT 'fail' AS status,
       'Batch match ' || c.bank_id || '->' || c.gl_id || ' missing' AS message
FROM batch_match_matched c
LEFT JOIN match_residual_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL
UNION ALL
SELECT 'fail' AS status,
       'Tolerance match ' || c.bank_id || '->' || c.gl_id || ' missing' AS message
FROM tolerance_matched c
LEFT JOIN match_residual_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL
UNION ALL
SELECT 'fail' AS status,
       'Amount match ' || c.bank_id || '->' || c.gl_id || ' missing' AS message
FROM amount_match_matched c
LEFT JOIN match_residual_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL
UNION ALL
SELECT 'warn' AS status,
       'Unmatched bank: ' || b.id || ' $' || b.amount || ' ' || b.description AS message
FROM amount_match_remaining_bank b
LEFT JOIN match_residual_all_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL
UNION ALL
SELECT 'warn' AS status,
       'Unmatched GL: ' || g.id || ' $' || g.amount || ' ' || g.description AS message
FROM amount_match_remaining_gl g
LEFT JOIN match_residual_all_matched m ON g.id = m.gl_id
WHERE m.gl_id IS NULL;
"""


# ---------------------------------------------------------------------------
# Node 6 — Report  (sql)
# ---------------------------------------------------------------------------

REPORT_SQL = """\
CREATE VIEW report_matched AS
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note
FROM match_residual_all_matched ORDER BY bank_id;

CREATE VIEW report_unmatched_bank AS
SELECT b.id, b.date, b.description, b.amount
FROM features_bank b LEFT JOIN match_residual_all_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL ORDER BY b.date;

CREATE VIEW report_unmatched_gl AS
WITH offsetting_ids AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
)
SELECT g.id, g.date, g.description, g.amount, g.ref, g.entry_type
FROM features_gl g
LEFT JOIN match_residual_all_matched m ON g.id = m.gl_id
LEFT JOIN offsetting_ids oi ON g.id = oi.id
WHERE m.gl_id IS NULL AND oi.id IS NULL ORDER BY g.date;

CREATE VIEW report_summary AS
SELECT
    (SELECT COUNT(*) FROM features_bank) AS bank_count,
    (SELECT COUNT(*) FROM features_gl) AS gl_count,
    (SELECT COUNT(DISTINCT bank_id) FROM match_residual_all_matched) AS matched_bank,
    (SELECT COUNT(DISTINCT gl_id) FROM match_residual_all_matched) AS matched_gl,
    (SELECT COUNT(*) FROM match_residual_all_matched WHERE match_type IN ('check_number','exact_1to1')) AS exact_matches,
    (SELECT COUNT(*) FROM match_residual_all_matched WHERE match_type = 'batch') AS batch_gl_rows,
    (SELECT COUNT(*) FROM match_residual_all_matched WHERE match_type = 'tolerance') AS tolerance_matches,
    (SELECT COUNT(*) FROM report_unmatched_bank) AS unmatched_bank,
    (SELECT COUNT(*) FROM report_unmatched_gl) AS unmatched_gl,
    (SELECT COUNT(*) FROM offsetting_pairs) AS offsetting_pairs;
"""

REPORT_VALIDATE = """\
SELECT 'fail' AS status,
       'Bank count mismatch: ' || bc || ' != ' || mc || ' + ' || ub AS message
FROM (
    SELECT (SELECT COUNT(*) FROM features_bank) AS bc,
           (SELECT COUNT(DISTINCT bank_id) FROM match_residual_all_matched) AS mc,
           (SELECT COUNT(*) FROM report_unmatched_bank) AS ub
) WHERE bc != mc + ub
UNION ALL
SELECT 'fail' AS status,
       'GL count mismatch: ' || gc || ' != ' || mc || ' + ' || ug || ' + ' || oc AS message
FROM (
    SELECT (SELECT COUNT(*) FROM features_gl) AS gc,
           (SELECT COUNT(DISTINCT gl_id) FROM match_residual_all_matched) AS mc,
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
    # Feature extraction
    {
        "name": "features",
        "sql": FEATURES_SQL,
        "depends_on": ["bank_txns", "gl_entries"],
    },
    # Conservative 1:1 matching
    {
        "name": "match_certain",
        "sql": MATCH_CERTAIN_SQL,
        "validate": {"main": MATCH_CERTAIN_VALIDATE},
        "depends_on": ["features"],
        "output_columns": {
            "match_certain_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
        },
    },
    # Offsetting GL pairs
    {
        "name": "offsetting",
        "sql": OFFSETTING_SQL,
        "depends_on": ["features", "match_certain"],
    },
    # Batch deposit matching (SQL)
    {
        "name": "batch_match",
        "sql": BATCH_MATCH_SQL,
        "validate": {"main": BATCH_MATCH_VALIDATE},
        "depends_on": ["features", "match_certain", "offsetting"],
    },
    # Tolerance matching (SQL — wire fees, small amount diffs)
    {
        "name": "tolerance",
        "sql": TOLERANCE_SQL,
        "validate": {"main": TOLERANCE_VALIDATE},
        "depends_on": ["features", "match_certain", "offsetting", "batch_match"],
    },
    # Amount-based matching for cryptic descriptions (SQL)
    {
        "name": "amount_match",
        "sql": AMOUNT_MATCH_SQL,
        "validate": {"main": AMOUNT_MATCH_VALIDATE},
        "depends_on": [
            "features",
            "match_certain",
            "offsetting",
            "batch_match",
            "tolerance",
        ],
    },
    # Residual matching (prompt — cryptic descriptions only)
    {
        "name": "match_residual",
        "prompt": MATCH_RESIDUAL_INTENT,
        "validate": {"main": MATCH_RESIDUAL_VALIDATE},
        "depends_on": [
            "features",
            "match_certain",
            "offsetting",
            "batch_match",
            "tolerance",
            "amount_match",
        ],
        "output_columns": {
            "match_residual_all_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
        },
    },
    # Report
    {
        "name": "report",
        "sql": REPORT_SQL,
        "validate": {"main": REPORT_VALIDATE},
        "depends_on": ["features", "match_residual", "offsetting"],
    },
]
