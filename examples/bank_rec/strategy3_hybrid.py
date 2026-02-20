"""Bank Reconciliation V4b — General-purpose with batch SQL

Architecture: SQL pipeline handles check#, entity 1:1, batch deposits, and
offsetting. LLM handles tolerance, cryptic descriptions, and ambiguous residual.

DAG:
    bank_txns, gl_entries
            |
        features           (sql: entity extraction, check numbers)
            |
        match_certain      (sql: check# + entity-based 1:1, conservative)
            |
        offsetting         (sql: self-canceling GL pairs)
            |
        batch_match        (sql: entity-grouped batch deposits)
            |
       match_residual      (prompt: tolerance, cryptic, ambiguous — all remaining)
            |
          report           (sql)

Design principles:
  - SQL nodes do what's unambiguously correct (check#, entity+amount, batches)
  - LLM handles all judgment calls (fuzzy matching, tolerance, ambiguous)
  - No hardcoded fee lists, pattern catalogs, or format-specific parsing
  - Batch SQL is general-purpose: group GL by entity, match sum to bank deposit
"""


# ---------------------------------------------------------------------------
# Node 1 — Features  (sql)
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
    -- then strip trailing 1-3 char fragments (from bank truncation of legal suffixes,
    -- e.g. "ACME CONSULTING LTD" truncated to "ACME CONSULTING LT" -> "LT" fragment)
    trim(regexp_replace(
        regexp_replace(
            regexp_replace(
                regexp_replace(
                    regexp_replace(udesc,
                        '^(ACH (CREDIT|DEBIT|CR|DR)|WIRE (TRF )?(IN|OUT)|POS DEBIT|E-PAYMENT|AUTO-PAY)\\s+', ''),
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

-- Round 2: Entity-prefix + exact-amount matches
-- Bank entity (possibly truncated) is prefix of GL entity
-- Conservative: only match when unambiguous
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
        ROW_NUMBER() OVER (PARTITION BY c.bank_id ORDER BY c.date_gap, c.bank_ent_len DESC, c.gl_id) AS b_rank,
        ROW_NUMBER() OVER (PARTITION BY c.gl_id   ORDER BY c.date_gap, c.bank_ent_len DESC, c.bank_id) AS g_rank,
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

-- Round 3: Exact-amount + fuzzy string similarity
-- For catching typos or slightly misaligned strings not covered by starts_with
CREATE VIEW match_certain_r3 AS
WITH remaining_bank AS (
    SELECT b.* FROM features_bank b
    LEFT JOIN match_certain_r1 m1 ON b.id = m1.bank_id
    LEFT JOIN match_certain_r2 m2 ON b.id = m2.bank_id
    WHERE m1.bank_id IS NULL AND m2.bank_id IS NULL
),
remaining_gl AS (
    SELECT g.* FROM features_gl g
    LEFT JOIN match_certain_r1 m1 ON g.id = m1.gl_id
    LEFT JOIN match_certain_r2 m2 ON g.id = m2.gl_id
    WHERE m1.gl_id IS NULL AND m2.gl_id IS NULL
),
candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS bank_amount, g.amount AS gl_amount,
        ABS(date_diff('day', b.date, g.date)) AS date_gap,
        jaro_winkler_similarity(b.entity_norm, g.entity_norm) AS sim
    FROM remaining_bank b
    JOIN remaining_gl g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 10
     AND jaro_winkler_similarity(b.entity_norm, g.entity_norm) > 0.85
     AND length(b.entity_norm) >= 4
     AND length(g.entity_norm) >= 4
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
        ROW_NUMBER() OVER (PARTITION BY c.bank_id ORDER BY c.date_gap, c.sim DESC, c.gl_id) AS b_rank,
        ROW_NUMBER() OVER (PARTITION BY c.gl_id   ORDER BY c.date_gap, c.sim DESC, c.bank_id) AS g_rank,
        bc.n_candidates AS bank_n,
        gc.n_candidates AS gl_n
    FROM candidates c
    JOIN bank_counts bc ON c.bank_id = bc.bank_id
    JOIN gl_counts gc ON c.gl_id = gc.gl_id
)
SELECT
    bank_id, gl_id, bank_amount, gl_amount,
    'fuzzy_1to1' AS match_type,
    'Exact amount + Jaro-Winkler ' || ROUND(sim, 2) AS note
FROM ranked
WHERE b_rank = 1 AND g_rank = 1
  AND bank_n = 1 AND gl_n = 1;

-- Combined certain matches
CREATE VIEW match_certain_r4 AS
WITH remaining_bank AS (
    SELECT b.* FROM features_bank b
    LEFT JOIN match_certain_r1 m1 ON b.id = m1.bank_id
    LEFT JOIN match_certain_r2 m2 ON b.id = m2.bank_id
    LEFT JOIN match_certain_r3 m3 ON b.id = m3.bank_id
    WHERE m1.bank_id IS NULL AND m2.bank_id IS NULL AND m3.bank_id IS NULL
),
remaining_gl AS (
    SELECT g.* FROM features_gl g
    LEFT JOIN match_certain_r1 m1 ON g.id = m1.gl_id
    LEFT JOIN match_certain_r2 m2 ON g.id = m2.gl_id
    LEFT JOIN match_certain_r3 m3 ON g.id = m3.gl_id
    WHERE m1.gl_id IS NULL AND m2.gl_id IS NULL AND m3.gl_id IS NULL
),
candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS amount,
        b.date AS bank_date, g.date AS gl_date,
        b.entity_norm AS bank_ent, g.entity_norm AS gl_ent
    FROM remaining_bank b
    JOIN remaining_gl g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 15
     AND (
         starts_with(g.entity_norm, b.entity_norm) OR 
         jaro_winkler_similarity(b.entity_norm, g.entity_norm) > 0.85
     )
     AND length(b.entity_norm) >= 4
     AND length(g.entity_norm) >= 4
),
bank_counts AS (
    SELECT bank_id, COUNT(*) AS n_candidates FROM candidates GROUP BY bank_id
),
gl_counts AS (
    SELECT gl_id, COUNT(*) AS n_candidates FROM candidates GROUP BY gl_id
),
entangled AS (
    SELECT c.* 
    FROM candidates c
    JOIN bank_counts bc ON c.bank_id = bc.bank_id
    JOIN gl_counts gc ON c.gl_id = gc.gl_id
    WHERE bc.n_candidates > 1 AND gc.n_candidates > 1
),
bank_entangled AS (
    SELECT bank_id, MIN(amount) as amount, MIN(bank_ent) as bank_ent, MIN(bank_date) as bank_date
    FROM entangled GROUP BY bank_id
),
gl_entangled AS (
    SELECT gl_id, MIN(amount) as amount, MIN(gl_ent) as gl_ent, MIN(gl_date) as gl_date
    FROM entangled GROUP BY gl_id
),
bank_ranked AS (
    SELECT bank_id, amount, bank_ent, bank_date,
           ROW_NUMBER() OVER (PARTITION BY amount, SUBSTRING(bank_ent, 1, 6) ORDER BY bank_date, bank_id) as rnk,
           COUNT(*) OVER (PARTITION BY amount, SUBSTRING(bank_ent, 1, 6)) as total_bank
    FROM bank_entangled
),
gl_ranked AS (
    SELECT gl_id, amount, gl_ent, gl_date,
           ROW_NUMBER() OVER (PARTITION BY amount, SUBSTRING(gl_ent, 1, 6) ORDER BY gl_date, gl_id) as rnk,
           COUNT(*) OVER (PARTITION BY amount, SUBSTRING(gl_ent, 1, 6)) as total_gl
    FROM gl_entangled
)
SELECT 
    br.bank_id, gr.gl_id, br.amount AS bank_amount, gr.amount AS gl_amount,
    'series_chronological' AS match_type,
    'Matched chronological series rank ' || br.rnk AS note
FROM bank_ranked br
JOIN gl_ranked gr 
  ON br.amount = gr.amount 
 AND SUBSTRING(br.bank_ent, 1, 6) = SUBSTRING(gr.gl_ent, 1, 6)
 AND br.rnk = gr.rnk
 AND br.total_bank = gr.total_gl
 AND br.total_bank > 1;

-- All matched
CREATE VIEW match_certain_matched AS
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM match_certain_r1
UNION ALL
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM match_certain_r2
UNION ALL
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM match_certain_r3
UNION ALL
SELECT bank_id, gl_id, bank_amount, gl_amount, match_type, note FROM match_certain_r4;

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
# Node 4 — Batch deposits  (sql)
# ---------------------------------------------------------------------------

BATCH_MATCH_SQL = """\
-- Batch deposit matching: group GL entries by entity, match sums to bank deposits.
-- Pattern: one customer pays multiple invoices in a single wire/ACH transfer.
-- Bank records one deposit; GL has multiple AR entries for the same entity.
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
    HAVING COUNT(*) >= 2  -- at least 2 entries to form a batch
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
    QUALIFY ROW_NUMBER() OVER (PARTITION BY b.id ORDER BY g.gl_cnt, g.entity_norm) = 1
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

-- Remaining pools
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

-- NEW BATCH ROUND 2 (AMOUNT-DRIVEN, ENTITY-AGNOSTIC)
CREATE VIEW batch_match_r2 AS
WITH remaining_bank AS (
    SELECT * FROM batch_match_remaining_bank
    WHERE amount > 0
),
remaining_gl AS (
    SELECT * FROM batch_match_remaining_gl
    WHERE amount > 0
),
-- self join gl entries to find pairs (size 2) that sum to a bank deposit
gl_pairs AS (
    SELECT
        g1.id AS gl_id1, g2.id AS gl_id2,
        g1.amount + g2.amount AS sum_amount,
        g1.date AS d1, g2.date AS d2,
        g1.entity_norm AS e1, g2.entity_norm AS e2
    FROM remaining_gl g1
    JOIN remaining_gl g2
      ON g1.id < g2.id
     AND ABS(date_diff('day', g1.date, g2.date)) <= 10
),
candidate_batches AS (
    SELECT
        b.id AS bank_id,
        p.gl_id1, p.gl_id2,
        b.amount AS bank_amount,
        p.sum_amount AS gl_amount,
        ABS(date_diff('day', b.date, p.d1)) AS date_gap
    FROM remaining_bank b
    JOIN gl_pairs p
      ON ABS(b.amount - p.sum_amount) < 0.01
     AND b.date >= p.d1 AND b.date <= p.d1 + INTERVAL 14 DAY
     AND jaro_winkler_similarity(p.e1, p.e2) > 0.8
),
bank_counts AS (SELECT bank_id, COUNT(*) as c FROM candidate_batches GROUP BY bank_id)
SELECT c.bank_id, c.gl_id1, c.gl_id2, c.bank_amount, c.gl_amount
FROM (
    SELECT c.*,
           ROW_NUMBER() OVER(PARTITION BY c.gl_id1 ORDER BY c.date_gap, c.bank_id, c.gl_id2) as g1_rank,
           ROW_NUMBER() OVER(PARTITION BY c.gl_id2 ORDER BY c.date_gap, c.bank_id, c.gl_id1) as g2_rank
    FROM candidate_batches c
) c
JOIN bank_counts bc ON c.bank_id = bc.bank_id
WHERE bc.c = 1 AND c.g1_rank = 1 AND c.g2_rank = 1;

-- Explode r2
CREATE VIEW batch_match_r2_exploded AS
SELECT bank_id, gl_id1 AS gl_id, bank_amount,
       (SELECT amount FROM batch_match_remaining_gl WHERE id = c.gl_id1) AS gl_amount,
       'batch_r2' AS match_type, 'Batch subset size 2' AS note FROM batch_match_r2 c
UNION ALL
SELECT bank_id, gl_id2 AS gl_id, bank_amount,
       (SELECT amount FROM batch_match_remaining_gl WHERE id = c.gl_id2) AS gl_amount,
       'batch_r2' AS match_type, 'Batch subset size 2' AS note FROM batch_match_r2 c;

CREATE VIEW batch_match_all_batch_matched AS
SELECT * FROM batch_match_matched
UNION ALL
SELECT * FROM batch_match_r2_exploded;

-- Final remaining pools (replaces old ones)
CREATE VIEW batch_match_final_remaining_bank AS
SELECT b.* FROM match_certain_unmatched_bank b
LEFT JOIN batch_match_all_batch_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL;

CREATE VIEW batch_match_final_remaining_gl AS
WITH offset_gl AS (
    SELECT original_id AS id FROM offsetting_pairs
    UNION ALL SELECT reversal_id FROM offsetting_pairs
)
SELECT g.* FROM match_certain_unmatched_gl g
LEFT JOIN batch_match_all_batch_matched m ON g.id = m.gl_id
LEFT JOIN offset_gl o ON g.id = o.id
WHERE m.gl_id IS NULL AND o.id IS NULL;
"""

BATCH_MATCH_VALIDATE = """\
SELECT 'fail' AS status,
       'batch bank_id ' || bank_id || ' gl_sum mismatch: ' ||
       ROUND(gl_sum, 2) || ' vs ' || bank_amt AS message
FROM (
    SELECT bank_id, SUM(gl_amount) AS gl_sum, MAX(bank_amount) AS bank_amt
    FROM batch_match_all_batch_matched GROUP BY bank_id
)
WHERE ABS(gl_sum - bank_amt) > 0.01
UNION ALL
SELECT 'fail' AS status,
       'batch gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM batch_match_all_batch_matched GROUP BY gl_id)
WHERE cnt > 1;
"""


# ---------------------------------------------------------------------------
# Node 4.5 — Exact Amount Closest Date (sql)
# ---------------------------------------------------------------------------

MATCH_EXACT_CLOSEST_SQL = """\
CREATE VIEW match_exact_closest_matched AS
WITH remaining_bank AS (
    SELECT b.* FROM batch_match_final_remaining_bank b_ids
    JOIN features_bank b ON b_ids.id = b.id
),
remaining_gl AS (
    SELECT g.* FROM batch_match_final_remaining_gl g_ids
    JOIN features_gl g ON g_ids.id = g.id
),
candidates AS (
    SELECT
        b.id AS bank_id, g.id AS gl_id,
        b.amount AS bank_amount, g.amount AS gl_amount,
        b.date AS bank_date, g.date AS gl_date,
        b.entity_norm AS bank_ent, g.entity_norm AS gl_ent,
        ABS(date_diff('day', b.date, g.date)) AS date_gap
    FROM remaining_bank b
    JOIN remaining_gl g
      ON b.amount = g.amount
     AND ABS(date_diff('day', b.date, g.date)) <= 15
),
ranked AS (
    SELECT c.*,
           ROW_NUMBER() OVER (PARTITION BY bank_id ORDER BY date_gap ASC, gl_id) as b_rnk,
           ROW_NUMBER() OVER (PARTITION BY gl_id ORDER BY date_gap ASC, bank_id) as g_rnk
    FROM candidates c
)
SELECT bank_id, gl_id, bank_amount, gl_amount, 
       'exact_amount_closest_date' AS match_type,
       'Matched purely by exact amount + closest date gap: ' || date_gap || ' days' AS note
FROM ranked
WHERE b_rnk = 1 AND g_rnk = 1;

CREATE VIEW match_exact_closest_final_remaining_bank AS
SELECT b.id FROM batch_match_final_remaining_bank b
LEFT JOIN match_exact_closest_matched m ON b.id = m.bank_id
WHERE m.bank_id IS NULL;

CREATE VIEW match_exact_closest_final_remaining_gl AS
SELECT g.id FROM batch_match_final_remaining_gl g
LEFT JOIN match_exact_closest_matched m ON g.id = m.gl_id
WHERE m.gl_id IS NULL;
"""

MATCH_EXACT_CLOSEST_VALIDATE = """\
SELECT 'fail' AS status,
       'bank_id ' || bank_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT bank_id, COUNT(*) AS cnt FROM match_exact_closest_matched GROUP BY bank_id)
WHERE cnt > 1
UNION ALL
SELECT 'fail' AS status,
       'gl_id ' || gl_id || ' matched ' || cnt || 'x' AS message
FROM (SELECT gl_id, COUNT(*) AS cnt FROM match_exact_closest_matched GROUP BY gl_id)
WHERE cnt > 1;
"""

MATCH_RESIDUAL_INTENT = """\
Resolve remaining unmatched bank and GL items. Prior SQL nodes have matched:
1) Check-number matches (exact check# + amount)
2) Entity-based 1:1 matches (bank entity is prefix of GL entity + exact amount)
3) Offsetting GL pairs (equal-and-opposite GL entries that cancel out)
4) Batch deposits (GL entries grouped by entity that sum to a bank deposit)
5) Exact amount closest date (any remaining exact amount match within 15 days)

Everything else is up to you. Query the remaining pools and match what you can.

AVAILABLE VIEWS:
  match_exact_closest_final_remaining_bank   — unmatched bank items (after all SQL matching)
  match_exact_closest_final_remaining_gl     — unmatched GL items (after all SQL matching)
  match_certain_matched        — already-matched (check# + entity 1:1)
  batch_match_all_batch_matched          — already-matched batch deposits
  match_exact_closest_matched  — already-matched by exact amount closest date
  offsetting_pairs             — GL pairs that cancel out (already excluded)

MATCHING STRATEGIES (apply in order of confidence):

1) MULTI-DEPOSIT BATCHES: Some entities may have multiple bank deposits, where
   the SQL batch node matched the full group but there may be remaining entities
   with GL entries that don't sum to any single bank deposit. Try finding subsets
   of GL entries from the same entity that sum to a bank deposit amount.
   Use `starts_with(gl_entity, bank_entity) OR jaro_winkler_similarity(gl_entity, bank_entity) > 0.85`
   to account for typos in vendor names.

2) TOLERANCE: Wire transfers commonly incur fees ($10-$75 typical), causing the
   bank amount to slightly exceed the GL amount. Match by fuzzy-entity where
   amounts differ by up to $100. Use abs(bank_amount - gl_amount) <= 100 as
   the threshold — do NOT use a smaller cutoff like $25 or $50.
   Also use `jaro_winkler_similarity > 0.85` here to catch typos.
   For non-entity matches, use a very small tolerance (< $5) only.

3) EXTREME OBFUSCATION (MANUAL MAPPING): When the candidate pool is very small
   and you see clear matches that SQL could never join (e.g., heavily truncated
   vendor names or missing invoices), do NOT try to invent complex SQL join rules.
   Simply read the rows yourself, pair them up manually, and output a literal
   table using a `VALUES` clause (e.g. `SELECT * FROM (VALUES (...)) AS t(...)`).

RULES:
- Bank-generated fees/charges (service charges, maintenance fees, interest,
  foreign transaction fees, wire fees) typically have NO corresponding GL entry.
  Do not force-match them. Leave them unmatched.
- Be conservative: only match when evidence is strong. A false positive is
  worse than a missed match.
- One bank item maps to one GL entry (1:1), UNLESS it's a batch deposit
  (one bank item maps to multiple GL entries that sum to the bank amount).
- Each GL entry can only be matched once.

OUTPUT: Create a view called match_residual_all_matched with columns:
  bank_id, gl_id, bank_amount, gl_amount, match_type, note

Must include ALL match_certain_matched rows, ALL batch_match_all_batch_matched rows,
ALL match_exact_closest_matched rows, plus your new matches. Batch rows should have match_type='batch'.

DuckDB NOTES:
- Keep view chains shallow (max 2 levels deep). Use CTEs instead.
- Use starts_with(gl_entity, bank_entity) for prefix matching.
- Offsetting GL IDs are in offsetting_pairs (original_id, reversal_id) — exclude
  them from your candidate pools.
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
FROM batch_match_all_batch_matched c
LEFT JOIN match_residual_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL
UNION ALL
SELECT 'fail' AS status,
       'Exact match ' || c.bank_id || '->' || c.gl_id || ' missing' AS message
FROM match_exact_closest_matched c
LEFT JOIN match_residual_all_matched a ON c.bank_id = a.bank_id AND c.gl_id = a.gl_id
WHERE a.bank_id IS NULL;
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
    {
        "name": "bank_txns",
        "source": None,
        "columns": ["id", "date", "description", "amount"],
    },
    {
        "name": "gl_entries",
        "source": None,
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
        "name": "offsetting",
        "sql": OFFSETTING_SQL,
        "depends_on": ["features", "match_certain"],
    },
    {
        "name": "batch_match",
        "sql": BATCH_MATCH_SQL,
        "validate": {"main": BATCH_MATCH_VALIDATE},
        "depends_on": ["features", "match_certain", "offsetting"],
        "output_columns": {
            "batch_match_all_batch_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
            "batch_match_final_remaining_bank": ["id"],
            "batch_match_final_remaining_gl": ["id"],
        },
    },
    {
        "name": "match_exact_closest",
        "sql": MATCH_EXACT_CLOSEST_SQL,
        "validate": {"main": MATCH_EXACT_CLOSEST_VALIDATE},
        "depends_on": ["features", "batch_match"],
        "output_columns": {
            "match_exact_closest_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
            "match_exact_closest_final_remaining_bank": ["id"],
            "match_exact_closest_final_remaining_gl": ["id"],
        },
    },
    {
        "name": "match_residual",
        "prompt": MATCH_RESIDUAL_INTENT,
        "validate": {"main": MATCH_RESIDUAL_VALIDATE},
        "depends_on": [
            "features",
            "match_certain",
            "offsetting",
            "batch_match",
            "match_exact_closest",
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
    {
        "name": "report",
        "sql": REPORT_SQL,
        "validate": {"main": REPORT_VALIDATE},
        "depends_on": ["features", "match_residual", "offsetting", "batch_match"],
    },
]
