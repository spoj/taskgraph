"""Bank Reconciliation V5 — All-SQL deterministic pipeline (no LLM)

Replaces the LLM prompt node from v4 with additional SQL nodes.
All matching logic is deterministic SQL — zero API cost, ~1s execution.

DAG (10 nodes: 2 source + 8 SQL):
    bank_txns, gl_entries
            |
        features           (sql: entity extraction, check numbers) [from v4]
            |
        match_certain      (sql: check# + entity-based 1:1) [from v4]
            |
        deposit_nsf        (sql: generic DEPOSIT/NSF pairing via GL entity) [new]
            |
        offsetting         (sql: self-canceling GL pairs) [modified]
            |
        batch_match        (sql: entity-grouped batch deposits - full group) [modified]
            |
        batch_subset       (sql: subset-sum batch deposits via self-joins 2-5) [new]
            |
        match_remaining    (sql: unique amount + entity tolerance ≤$100) [new]
            |
        report             (sql: final views) [modified]

Matching passes (mirrors solve_bank_rec.py):
  1. Check number + exact amount (match_certain)
  2. Entity-prefix + exact amount 1:1 (match_certain)
  3. Generic DEPOSIT/NSF pairing via GL entity (deposit_nsf)
  4. Offsetting GL pairs — entity-confirmed (offsetting)
  5. Batch deposits — full entity group (batch_match)
  6. Batch deposits — subset-sum via self-joins (batch_subset)
  7. Unique exact amount + date proximity (match_remaining)
  8. Entity-prefix + tolerance ≤$100 (match_remaining)
"""

from examples.bank_rec_problem import BANK_TRANSACTIONS, GL_ENTRIES

# ---------------------------------------------------------------------------
# Node 1 — Features  (sql)  [reused from v4]
# ---------------------------------------------------------------------------

FEATURES_SQL = """\
CREATE VIEW features_bank AS
WITH raw AS (
    SELECT *, upper(description) AS udesc
    FROM bank_txns
)
SELECT
    id, date, description, amount,
    -- Cleaned description (letters/digits/spaces only)
    regexp_replace(udesc, '[^A-Z0-9 ]', ' ', 'g') AS desc_clean,
    -- Check number (from "CHECK #1234" or "CHECK 1234")
    TRY_CAST(regexp_extract(udesc, 'CHECK\\s*#?\\s*(\\d+)', 1) AS INTEGER) AS check_no,
    -- Entity: strip payment-method prefixes, trailing reference numbers, legal suffixes,
    -- then strip trailing 1-3 char fragments (from bank truncation of legal suffixes)
    trim(regexp_replace(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(udesc,
                        '^(ACH (CREDIT|DEBIT|CR|DR)|WIRE (TRF )?(IN|OUT))\\s+', ''),
                    '\\d{6,}\\s*', ' '),
                '(^|\\s)(INCORPORATED|INC|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|LP|LLP|PLC)(\\s|$)', ' ', 'g'),
            '\\s+[A-Z]{1,3}$', ''),
        '\\s+', ' ', 'g'
    )) AS entity_norm
FROM raw;

CREATE VIEW features_gl AS
SELECT
    id, date, description, amount, ref, entry_type,
    regexp_replace(upper(description), '[^A-Z0-9 ]', ' ', 'g') AS desc_clean,
    -- Check number from ref (common patterns: CHK-1234, CHECK-1234, #1234)
    TRY_CAST(regexp_extract(upper(ref), '(?:CHK|CHECK)[-#]?(\\d+)', 1) AS INTEGER) AS check_no,
    -- Entity: first segment before common separators, strip legal suffixes
    trim(regexp_replace(
        regexp_replace(
            upper(CASE
                WHEN strpos(description, ' - ') > 0
                     THEN trim(substr(description, 1, strpos(description, ' - ') - 1))
                WHEN strpos(description, ' / ') > 0
                     THEN trim(substr(description, 1, strpos(description, ' / ') - 1))
                ELSE description
            END),
            '(^|\\s)(INCORPORATED|INC|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|LP|LLP|PLC)(\\s|$)', ' ', 'g'),
        '\\s+', ' ', 'g'
    )) AS entity_norm
FROM gl_entries;
"""


# ---------------------------------------------------------------------------
# Node 2 — Certain 1:1 matches  (sql)  [reused from v4]
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

-- Round 2: Entity-prefix + exact-amount matches
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
        ABS(date_diff('day', b.date, g.date)) AS date_gap,
        length(b.entity_norm) AS bank_ent_len
    FROM remaining_bank b
    JOIN remaining_gl g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 10
     AND starts_with(g.entity_norm, b.entity_norm)
     AND length(b.entity_norm) >= 4
),
bank_counts AS (
    SELECT bank_id, COUNT(*) AS n_candidates
    FROM candidates GROUP BY bank_id
),
gl_counts AS (
    SELECT gl_id, COUNT(*) AS n_candidates
    FROM candidates GROUP BY gl_id
),
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
# Node 3 — Deposit/NSF pairing  (sql)  [NEW]
# ---------------------------------------------------------------------------
# Generic bank "DEPOSIT" and "RETURNED ITEM - NSF" have no entity info.
# GL payment (+X) and NSF return (-X) entries share the same entity.
# 4-way join: bank_dep(+X) + bank_nsf(-X) + gl_pos(+X, E) + gl_neg(-X, E)

DEPOSIT_NSF_SQL = """\
CREATE VIEW deposit_nsf_matched AS
WITH
remaining_bank AS (
    SELECT b.* FROM features_bank b
    LEFT JOIN match_certain_matched m ON b.id = m.bank_id
    WHERE m.bank_id IS NULL
),
remaining_gl AS (
    SELECT g.* FROM features_gl g
    LEFT JOIN match_certain_matched m ON g.id = m.gl_id
    WHERE m.gl_id IS NULL
),
-- Generic bank deposits (description is exactly "DEPOSIT")
bank_dep AS (
    SELECT * FROM remaining_bank
    WHERE upper(description) = 'DEPOSIT' AND amount > 0
),
-- Generic bank NSF returns (description contains "RETURNED ITEM")
bank_nsf AS (
    SELECT * FROM remaining_bank
    WHERE upper(description) LIKE '%RETURNED ITEM%' AND amount < 0
),
-- GL positives and negatives
gl_pos AS (SELECT * FROM remaining_gl WHERE amount > 0),
gl_neg AS (SELECT * FROM remaining_gl WHERE amount < 0),
-- 4-way join: match by amount equality + GL entity linkage + date proximity
quadruples AS (
    SELECT
        bd.id AS dep_bank_id, bd.amount AS dep_amount, bd.date AS dep_date,
        bn.id AS nsf_bank_id, bn.amount AS nsf_amount, bn.date AS nsf_date,
        gp.id AS gl_pos_id, gp.amount AS gl_pos_amount,
        gn.id AS gl_neg_id, gn.amount AS gl_neg_amount,
        gp.entity_norm AS gl_entity
    FROM bank_dep bd
    JOIN bank_nsf bn ON bd.amount = -bn.amount
    JOIN gl_pos gp ON gp.amount = bd.amount
        AND ABS(date_diff('day', bd.date, gp.date)) <= 10
    JOIN gl_neg gn ON gn.amount = bn.amount
        AND gn.entity_norm = gp.entity_norm
        AND ABS(date_diff('day', bn.date, gn.date)) <= 10
),
-- Greedy dedup: each bank/GL item at most once (cascading row_number)
step1 AS (
    SELECT * FROM quadruples
    QUALIFY ROW_NUMBER() OVER (PARTITION BY dep_bank_id ORDER BY gl_entity) = 1
),
step2 AS (
    SELECT * FROM step1
    QUALIFY ROW_NUMBER() OVER (PARTITION BY nsf_bank_id ORDER BY dep_bank_id) = 1
),
step3 AS (
    SELECT * FROM step2
    QUALIFY ROW_NUMBER() OVER (PARTITION BY gl_pos_id ORDER BY dep_bank_id) = 1
),
final_quads AS (
    SELECT * FROM step3
    QUALIFY ROW_NUMBER() OVER (PARTITION BY gl_neg_id ORDER BY dep_bank_id) = 1
)
-- Flatten: deposit match + NSF match (two rows per quadruple)
SELECT dep_bank_id AS bank_id, gl_pos_id AS gl_id,
       dep_amount AS bank_amount, gl_pos_amount AS gl_amount,
       'deposit_nsf' AS match_type,
       'Deposit+NSF pair, entity=' || gl_entity AS note
FROM final_quads
UNION ALL
SELECT nsf_bank_id AS bank_id, gl_neg_id AS gl_id,
       nsf_amount AS bank_amount, gl_neg_amount AS gl_amount,
       'deposit_nsf' AS match_type,
       'Deposit+NSF pair, entity=' || gl_entity AS note
FROM final_quads;
"""


# ---------------------------------------------------------------------------
# Node 4 — Offsetting GL pairs  (sql)  [modified: exclude deposit_nsf]
# ---------------------------------------------------------------------------

OFFSETTING_SQL = """\
CREATE VIEW offsetting_pairs AS
WITH unmatched_gl AS (
    SELECT g.* FROM features_gl g
    LEFT JOIN match_certain_matched m ON g.id = m.gl_id
    LEFT JOIN deposit_nsf_matched dn ON g.id = dn.gl_id
    WHERE m.gl_id IS NULL AND dn.gl_id IS NULL
),
-- For VOID entries, extract the entity name from the description
-- Patterns: "VOID - Check #NNN to ENTITY (reason)" or "VOID - Payment to ENTITY (reason)"
void_entities AS (
    SELECT id,
        trim(regexp_replace(
            upper(regexp_extract(description, '(?i)VOID\\s*-\\s*.*?\\s+to\\s+(.+?)\\s*\\(', 1)),
            '(^|\\s)(INCORPORATED|INC|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|LP|LLP|PLC)(\\s|$)', ' ', 'g'
        )) AS void_entity_norm
    FROM unmatched_gl
    WHERE upper(description) LIKE '%VOID%'
),
candidates AS (
    SELECT
        a.id AS original_id, b.id AS reversal_id,
        a.description AS original_desc, b.description AS reversal_desc,
        a.amount AS original_amount, b.amount AS reversal_amount,
        a.date AS original_date, b.date AS reversal_date,
        a.ref AS original_ref, b.ref AS reversal_ref,
        a.entity_norm AS a_entity, b.entity_norm AS b_entity,
        -- Entity confirmation: same entity, or VOID entity matches
        CASE
            WHEN a.entity_norm = b.entity_norm THEN true
            WHEN ve_b.void_entity_norm IS NOT NULL
                 AND starts_with(ve_b.void_entity_norm, a.entity_norm)
                 AND length(a.entity_norm) >= 4 THEN true
            WHEN ve_a.void_entity_norm IS NOT NULL
                 AND starts_with(ve_a.void_entity_norm, b.entity_norm)
                 AND length(b.entity_norm) >= 4 THEN true
            ELSE false
        END AS entity_confirmed
    FROM unmatched_gl a
    JOIN unmatched_gl b
      ON a.amount + b.amount = 0
     AND a.amount < 0
     AND a.id < b.id
     AND ABS(date_diff('day', a.date, b.date)) <= 10
    LEFT JOIN void_entities ve_a ON a.id = ve_a.id
    LEFT JOIN void_entities ve_b ON b.id = ve_b.id
)
SELECT original_id, reversal_id, original_desc, reversal_desc,
       original_amount, reversal_amount, original_date, reversal_date,
       original_ref, reversal_ref
FROM candidates
WHERE entity_confirmed
  AND jaro_winkler_similarity(
        regexp_replace(upper(original_desc), '[^A-Z0-9 ]', ' ', 'g'),
        regexp_replace(upper(reversal_desc), '[^A-Z0-9 ]', ' ', 'g')
      ) > 0.4;
"""


# ---------------------------------------------------------------------------
# Node 5 — Batch deposits — full group  (sql)  [modified: exclude deposit_nsf]
# ---------------------------------------------------------------------------

BATCH_MATCH_SQL = """\
-- Batch deposit matching: group GL entries by entity, match sums to bank deposits.
-- Pattern: one customer pays multiple invoices in a single wire/ACH transfer.
CREATE VIEW batch_match_matched AS
WITH
-- Exclude offsetting GL from the pool
offset_gl AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
),
-- Unmatched positive bank items (deposits), excluding deposit_nsf
bank_pool AS (
    SELECT b.*
    FROM match_certain_unmatched_bank b
    LEFT JOIN deposit_nsf_matched dn ON b.id = dn.bank_id
    WHERE dn.bank_id IS NULL AND b.amount > 0
),
-- Unmatched positive GL items, excluding deposit_nsf + offsets
gl_pool AS (
    SELECT g.*
    FROM match_certain_unmatched_gl g
    LEFT JOIN deposit_nsf_matched dn ON g.id = dn.gl_id
    LEFT JOIN offset_gl o ON g.id = o.id
    WHERE dn.gl_id IS NULL AND o.id IS NULL AND g.amount > 0
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
    HAVING COUNT(*) >= 2
),
-- Match bank deposits to GL groups via entity prefix + exact amount sum
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
     AND length(b.entity_norm) >= 4
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

-- Remaining unmatched after batch (includes all amounts, not just positive)
CREATE VIEW batch_match_remaining_bank AS
SELECT b.* FROM match_certain_unmatched_bank b
LEFT JOIN deposit_nsf_matched dn ON b.id = dn.bank_id
LEFT JOIN batch_match_matched m ON b.id = m.bank_id
WHERE dn.bank_id IS NULL AND m.bank_id IS NULL;

CREATE VIEW batch_match_remaining_gl AS
WITH offset_gl AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
)
SELECT g.* FROM match_certain_unmatched_gl g
LEFT JOIN deposit_nsf_matched dn ON g.id = dn.gl_id
LEFT JOIN batch_match_matched m ON g.id = m.gl_id
LEFT JOIN offset_gl o ON g.id = o.id
WHERE dn.gl_id IS NULL AND m.gl_id IS NULL AND o.id IS NULL;
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
# Node 6 — Batch deposits — subset-sum  (sql)  [NEW]
# ---------------------------------------------------------------------------
# For bank deposits still unmatched after full-group batch, try subsets of
# GL entries from the same entity that sum to the bank amount.
# Explicit self-joins for sizes 2-5 (max batch size is 5 for HARD).

BATCH_SUBSET_SQL = """\
CREATE VIEW batch_subset_matched AS
WITH
bank_pool AS (
    SELECT * FROM batch_match_remaining_bank WHERE amount > 0
),
gl_pool AS (
    SELECT * FROM batch_match_remaining_gl WHERE amount > 0
),
-- Size 2 subsets
size2 AS (
    SELECT
        b.id AS bank_id, b.amount AS bank_amount,
        g1.id AS g1_id, g2.id AS g2_id,
        CAST(NULL AS VARCHAR) AS g3_id, CAST(NULL AS VARCHAR) AS g4_id,
        CAST(NULL AS VARCHAR) AS g5_id,
        2 AS batch_size,
        g1.entity_norm AS gl_entity
    FROM bank_pool b
    JOIN gl_pool g1 ON starts_with(g1.entity_norm, b.entity_norm)
        AND length(b.entity_norm) >= 4
    JOIN gl_pool g2 ON starts_with(g2.entity_norm, b.entity_norm)
        AND g2.id > g1.id
    WHERE ABS(g1.amount + g2.amount - b.amount) < 0.01
      AND b.date BETWEEN LEAST(g1.date, g2.date) - INTERVAL 7 DAY
                     AND GREATEST(g1.date, g2.date) + INTERVAL 7 DAY
),
-- Size 3 subsets
size3 AS (
    SELECT
        b.id AS bank_id, b.amount AS bank_amount,
        g1.id AS g1_id, g2.id AS g2_id, g3.id AS g3_id,
        CAST(NULL AS VARCHAR) AS g4_id, CAST(NULL AS VARCHAR) AS g5_id,
        3 AS batch_size,
        g1.entity_norm AS gl_entity
    FROM bank_pool b
    JOIN gl_pool g1 ON starts_with(g1.entity_norm, b.entity_norm)
        AND length(b.entity_norm) >= 4
    JOIN gl_pool g2 ON starts_with(g2.entity_norm, b.entity_norm)
        AND g2.id > g1.id
    JOIN gl_pool g3 ON starts_with(g3.entity_norm, b.entity_norm)
        AND g3.id > g2.id
    WHERE ABS(g1.amount + g2.amount + g3.amount - b.amount) < 0.01
      AND b.date BETWEEN LEAST(g1.date, g2.date, g3.date) - INTERVAL 7 DAY
                     AND GREATEST(g1.date, g2.date, g3.date) + INTERVAL 7 DAY
),
-- Size 4 subsets
size4 AS (
    SELECT
        b.id AS bank_id, b.amount AS bank_amount,
        g1.id AS g1_id, g2.id AS g2_id, g3.id AS g3_id, g4.id AS g4_id,
        CAST(NULL AS VARCHAR) AS g5_id,
        4 AS batch_size,
        g1.entity_norm AS gl_entity
    FROM bank_pool b
    JOIN gl_pool g1 ON starts_with(g1.entity_norm, b.entity_norm)
        AND length(b.entity_norm) >= 4
    JOIN gl_pool g2 ON starts_with(g2.entity_norm, b.entity_norm)
        AND g2.id > g1.id
    JOIN gl_pool g3 ON starts_with(g3.entity_norm, b.entity_norm)
        AND g3.id > g2.id
    JOIN gl_pool g4 ON starts_with(g4.entity_norm, b.entity_norm)
        AND g4.id > g3.id
    WHERE ABS(g1.amount + g2.amount + g3.amount + g4.amount - b.amount) < 0.01
      AND b.date BETWEEN LEAST(g1.date, g2.date, g3.date, g4.date) - INTERVAL 7 DAY
                     AND GREATEST(g1.date, g2.date, g3.date, g4.date) + INTERVAL 7 DAY
),
-- Size 5 subsets
size5 AS (
    SELECT
        b.id AS bank_id, b.amount AS bank_amount,
        g1.id AS g1_id, g2.id AS g2_id, g3.id AS g3_id, g4.id AS g4_id,
        g5.id AS g5_id,
        5 AS batch_size,
        g1.entity_norm AS gl_entity
    FROM bank_pool b
    JOIN gl_pool g1 ON starts_with(g1.entity_norm, b.entity_norm)
        AND length(b.entity_norm) >= 4
    JOIN gl_pool g2 ON starts_with(g2.entity_norm, b.entity_norm)
        AND g2.id > g1.id
    JOIN gl_pool g3 ON starts_with(g3.entity_norm, b.entity_norm)
        AND g3.id > g2.id
    JOIN gl_pool g4 ON starts_with(g4.entity_norm, b.entity_norm)
        AND g4.id > g3.id
    JOIN gl_pool g5 ON starts_with(g5.entity_norm, b.entity_norm)
        AND g5.id > g4.id
    WHERE ABS(g1.amount + g2.amount + g3.amount + g4.amount + g5.amount - b.amount) < 0.01
      AND b.date BETWEEN LEAST(g1.date, g2.date, g3.date, g4.date, g5.date) - INTERVAL 7 DAY
                     AND GREATEST(g1.date, g2.date, g3.date, g4.date, g5.date) + INTERVAL 7 DAY
),
-- Union all sizes, pick smallest subset per bank_id
all_subsets AS (
    SELECT * FROM size2
    UNION ALL SELECT * FROM size3
    UNION ALL SELECT * FROM size4
    UNION ALL SELECT * FROM size5
),
winners AS (
    SELECT * FROM all_subsets
    QUALIFY ROW_NUMBER() OVER (PARTITION BY bank_id ORDER BY batch_size, g1_id) = 1
),
-- Flatten: one row per GL item in winning subsets
flattened AS (
    SELECT bank_id, bank_amount, g1_id AS gl_id, gl_entity, batch_size FROM winners
    UNION ALL
    SELECT bank_id, bank_amount, g2_id, gl_entity, batch_size FROM winners
    UNION ALL
    SELECT bank_id, bank_amount, g3_id, gl_entity, batch_size FROM winners WHERE g3_id IS NOT NULL
    UNION ALL
    SELECT bank_id, bank_amount, g4_id, gl_entity, batch_size FROM winners WHERE g4_id IS NOT NULL
    UNION ALL
    SELECT bank_id, bank_amount, g5_id, gl_entity, batch_size FROM winners WHERE g5_id IS NOT NULL
)
-- Join back to get GL amounts
SELECT
    f.bank_id, f.gl_id, f.bank_amount,
    gp.amount AS gl_amount,
    'batch' AS match_type,
    'Subset: ' || f.batch_size || ' items, entity=' || f.gl_entity AS note
FROM flattened f
JOIN gl_pool gp ON gp.id = f.gl_id;
"""

BATCH_SUBSET_VALIDATE = """\
SELECT 'fail' AS status,
       'subset bank_id ' || bank_id || ' gl_sum mismatch: ' ||
       ROUND(gl_sum, 2) || ' vs ' || bank_amt AS message
FROM (
    SELECT bank_id, SUM(gl_amount) AS gl_sum, MAX(bank_amount) AS bank_amt
    FROM batch_subset_matched GROUP BY bank_id
)
WHERE ABS(gl_sum - bank_amt) > 0.01
UNION ALL
SELECT 'fail' AS status,
       'subset gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM batch_subset_matched GROUP BY gl_id)
WHERE cnt > 1;
"""


# ---------------------------------------------------------------------------
# Node 7 — Remaining matches  (sql)  [NEW — replaces LLM]
# ---------------------------------------------------------------------------
# Pass 7: Unique exact amount + date proximity (both sides unique)
# Pass 8: Entity-prefix + tolerance ≤$100

MATCH_REMAINING_SQL = """\
CREATE VIEW match_remaining_all_matched AS
WITH
-- Accumulate all prior matches
prior_matches AS (
    SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note
    FROM match_certain_matched
    UNION ALL
    SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note
    FROM deposit_nsf_matched
    UNION ALL
    SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note
    FROM batch_match_matched
    UNION ALL
    SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note
    FROM batch_subset_matched
),
-- Offsetting GL IDs (excluded from matching pools)
offset_gl AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
),
-- Remaining pools after all prior matching
remaining_bank AS (
    SELECT b.* FROM features_bank b
    LEFT JOIN prior_matches m ON b.id = m.bank_id
    WHERE m.bank_id IS NULL
),
remaining_gl AS (
    SELECT g.* FROM features_gl g
    LEFT JOIN prior_matches m ON g.id = m.gl_id
    LEFT JOIN offset_gl o ON g.id = o.id
    WHERE m.gl_id IS NULL AND o.id IS NULL
),

-- ---------------------------------------------------------------
-- Pass 7: Unique amount + date proximity
-- Only match when amount is unique in BOTH pools (no ambiguity)
-- ---------------------------------------------------------------
bank_amt_counts AS (
    SELECT amount, COUNT(*) AS cnt FROM remaining_bank GROUP BY amount
),
gl_amt_counts AS (
    SELECT amount, COUNT(*) AS cnt FROM remaining_gl GROUP BY amount
),
unique_candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS bank_amount, g.amount AS gl_amount,
        ABS(date_diff('day', b.date, g.date)) AS date_gap
    FROM remaining_bank b
    JOIN remaining_gl g ON b.amount = g.amount
        AND ABS(date_diff('day', b.date, g.date)) <= 10
    JOIN bank_amt_counts bc ON bc.amount = b.amount AND bc.cnt = 1
    JOIN gl_amt_counts gc ON gc.amount = g.amount AND gc.cnt = 1
),
unique_matched AS (
    SELECT bank_id, gl_id, bank_amount, gl_amount,
           'exact_amount_1to1' AS match_type,
           'Unique amount + date proximity' AS note
    FROM unique_candidates
    QUALIFY ROW_NUMBER() OVER (PARTITION BY bank_id ORDER BY date_gap) = 1
),

-- ---------------------------------------------------------------
-- Pass 8: Entity-prefix + tolerance ≤$100
-- For wire transfers where bank amount slightly differs from GL
-- ---------------------------------------------------------------
remaining_bank2 AS (
    SELECT b.* FROM remaining_bank b
    LEFT JOIN unique_matched um ON b.id = um.bank_id
    WHERE um.bank_id IS NULL
),
remaining_gl2 AS (
    SELECT g.* FROM remaining_gl g
    LEFT JOIN unique_matched um ON g.id = um.gl_id
    WHERE um.gl_id IS NULL
),
tol_candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS bank_amount, g.amount AS gl_amount,
        ABS(b.amount - g.amount) AS amt_diff,
        ABS(date_diff('day', b.date, g.date)) AS date_gap
    FROM remaining_bank2 b
    JOIN remaining_gl2 g
      ON (b.amount > 0) = (g.amount > 0)
     AND ABS(b.amount - g.amount) BETWEEN 0.01 AND 100
     AND ABS(date_diff('day', b.date, g.date)) <= 10
     AND starts_with(g.entity_norm, b.entity_norm)
     AND length(b.entity_norm) >= 4
),
tol_bank_cnt AS (
    SELECT bank_id, COUNT(*) AS cnt FROM tol_candidates GROUP BY bank_id
),
tol_gl_cnt AS (
    SELECT gl_id, COUNT(*) AS cnt FROM tol_candidates GROUP BY gl_id
),
tol_ranked AS (
    SELECT c.*,
        ROW_NUMBER() OVER (PARTITION BY c.bank_id ORDER BY c.amt_diff, c.date_gap) AS b_rank,
        ROW_NUMBER() OVER (PARTITION BY c.gl_id ORDER BY c.amt_diff, c.date_gap) AS g_rank,
        bc.cnt AS bank_n, gc.cnt AS gl_n
    FROM tol_candidates c
    JOIN tol_bank_cnt bc ON c.bank_id = bc.bank_id
    JOIN tol_gl_cnt gc ON c.gl_id = gc.gl_id
),
tol_matched AS (
    SELECT bank_id, gl_id, bank_amount, gl_amount,
           'tolerance_entity' AS match_type,
           'Entity prefix + amount diff $' || ROUND(amt_diff, 2) AS note
    FROM tol_ranked
    WHERE b_rank = 1 AND g_rank = 1
      AND (bank_n = 1 OR gl_n = 1)
)
-- Final: accumulate all matches
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM prior_matches
UNION ALL
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM unique_matched
UNION ALL
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM tol_matched;
"""

MATCH_REMAINING_VALIDATE = """\
WITH bank_agg AS (
    SELECT bank_id, COUNT(*) AS cnt,
           COUNT(*) FILTER (WHERE match_type != 'batch') AS non_batch,
           SUM(gl_amount) AS gl_sum, MAX(bank_amount) AS bank_amt
    FROM match_remaining_all_matched GROUP BY bank_id
)
SELECT 'fail' AS status,
       'bank_id ' || bank_id || ' has invalid multi-match' AS message
FROM bank_agg
WHERE cnt > 1 AND (non_batch > 0 OR ABS(gl_sum - bank_amt) > 0.01)
UNION ALL
SELECT 'fail' AS status,
       'gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM match_remaining_all_matched GROUP BY gl_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'Certain match ' || c.bank_id || '->' || c.gl_id || ' missing' AS message
FROM match_certain_matched c
LEFT JOIN match_remaining_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL
UNION ALL
SELECT 'fail' AS status,
       'Batch match ' || c.bank_id || '->' || c.gl_id || ' missing' AS message
FROM batch_match_matched c
LEFT JOIN match_remaining_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL;
"""


# ---------------------------------------------------------------------------
# Node 8 — Report  (sql)  [modified: reference match_remaining]
# ---------------------------------------------------------------------------

REPORT_SQL = """\
CREATE VIEW report_matched AS
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note
FROM match_remaining_all_matched ORDER BY bank_id;

CREATE VIEW report_unmatched_bank AS
SELECT b.id, b.date, b.description, b.amount
FROM features_bank b LEFT JOIN match_remaining_all_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL ORDER BY b.date;

CREATE VIEW report_unmatched_gl AS
WITH offsetting_ids AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
)
SELECT g.id, g.date, g.description, g.amount, g.ref, g.entry_type
FROM features_gl g
LEFT JOIN match_remaining_all_matched m ON g.id = m.gl_id
LEFT JOIN offsetting_ids oi ON g.id = oi.id
WHERE m.gl_id IS NULL AND oi.id IS NULL ORDER BY g.date;

CREATE VIEW report_summary AS
SELECT
    (SELECT COUNT(*) FROM features_bank) AS bank_count,
    (SELECT COUNT(*) FROM features_gl) AS gl_count,
    (SELECT COUNT(DISTINCT bank_id) FROM match_remaining_all_matched) AS matched_bank,
    (SELECT COUNT(DISTINCT gl_id) FROM match_remaining_all_matched) AS matched_gl,
    (SELECT COUNT(*) FROM report_unmatched_bank) AS unmatched_bank,
    (SELECT COUNT(*) FROM report_unmatched_gl) AS unmatched_gl,
    (SELECT COUNT(*) FROM offsetting_pairs) AS offsetting_pairs;
"""

REPORT_VALIDATE = """\
SELECT 'fail' AS status,
       'Bank count mismatch: ' || bc || ' != ' || mc || ' + ' || ub AS message
FROM (
    SELECT (SELECT COUNT(*) FROM features_bank) AS bc,
           (SELECT COUNT(DISTINCT bank_id) FROM match_remaining_all_matched) AS mc,
           (SELECT COUNT(*) FROM report_unmatched_bank) AS ub
) WHERE bc != mc + ub
UNION ALL
SELECT 'fail' AS status,
       'GL count mismatch: ' || gc || ' != ' || mc || ' + ' || ug || ' + ' || oc AS message
FROM (
    SELECT (SELECT COUNT(*) FROM features_gl) AS gc,
           (SELECT COUNT(DISTINCT gl_id) FROM match_remaining_all_matched) AS mc,
           (SELECT COUNT(*) FROM report_unmatched_gl) AS ug,
           (SELECT COUNT(*) * 2 FROM offsetting_pairs) AS oc
) WHERE gc != mc + ug + oc;
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

NODES = [
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
    {
        "name": "features",
        "sql": FEATURES_SQL,
        "depends_on": ["bank_txns", "gl_entries"],
    },
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
    {
        "name": "deposit_nsf",
        "sql": DEPOSIT_NSF_SQL,
        "depends_on": ["features", "match_certain"],
        "output_columns": {
            "deposit_nsf_matched": [
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
        "depends_on": ["features", "match_certain", "deposit_nsf"],
    },
    {
        "name": "batch_match",
        "sql": BATCH_MATCH_SQL,
        "validate": {"main": BATCH_MATCH_VALIDATE},
        "depends_on": ["features", "match_certain", "offsetting", "deposit_nsf"],
    },
    {
        "name": "batch_subset",
        "sql": BATCH_SUBSET_SQL,
        "validate": {"main": BATCH_SUBSET_VALIDATE},
        "depends_on": ["batch_match"],
    },
    {
        "name": "match_remaining",
        "sql": MATCH_REMAINING_SQL,
        "validate": {"main": MATCH_REMAINING_VALIDATE},
        "depends_on": [
            "features",
            "match_certain",
            "deposit_nsf",
            "offsetting",
            "batch_match",
            "batch_subset",
        ],
        "output_columns": {
            "match_remaining_all_matched": [
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
        "validate": {"main": REPORT_VALIDATE},
        "depends_on": ["features", "match_remaining", "offsetting"],
    },
]
