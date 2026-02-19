# Bank Reconciliation Spec — Lessons Learned

Notes from iterating v1→v4b of a taskgraph spec that matches bank transactions
to general ledger entries using a DAG of SQL and LLM (prompt) nodes.

## Architecture evolution

### v1–v2: Monolithic LLM
Single prompt node tried to match everything. Poor recall on batch deposits and
tolerance matches because the LLM had to reason over ~1000 items at once.

### v3: Heavy SQL pipeline (10 nodes)
Moved matching logic into SQL: `check_match`, `entity_match`, `batch_match`,
`amount_match`, `tolerance`, `offsetting`. LLM only handled leftovers.

Achieved 99.8% F1 on seed=42 but contained generator-specific fine-tuning:
- Trailing 1-3 char fragment stripping (generator truncation artifact)
- `entry_type = 'normal'` filter (generator-specific field)
- `HAVING COUNT(*) BETWEEN 2 AND 5` (generator batch sizes)
- Hardcoded fee list (SERVICE CHARGE, MAINTENANCE FEE, etc.)
- Prompt mentioning generator patterns (AMZN MKTP, GOOG *, DEPOSIT)

### v4: Simplified (7 nodes)
Dropped 3 SQL nodes (`batch_match`, `tolerance`, `amount_match`). Let the LLM
handle all matching beyond check-number and entity+amount 1:1.

100% precision but recall collapsed to 80.7% on a different seed (seed=99) —
batch recall dropped to 17.5%. The LLM couldn't reliably handle batch matching
at scale in a single prompt.

### v4b: Balanced (8 nodes, current)
Re-added a **general-purpose** batch SQL node and fixed two bugs. Three changes:

1. Entity truncation fix (general-purpose)
2. General-purpose batch SQL node (no generator-specific params)
3. Offsetting entity confirmation (prevents false positive GL pairs)

Results: 100% precision, 98.5–100% recall across 3 seeds and 3 difficulty levels.

## Key lessons

### 1. SQL handles structure, LLM handles judgment

The best split is: SQL nodes do what's **unambiguously correct** (check numbers,
entity+exact amount, same-entity batch sums, offsetting pairs). The LLM handles
everything that requires **judgment** (fuzzy entity matching, tolerance thresholds,
cryptic descriptions, ambiguous cases).

Trying to push too much into the LLM (v4) fails at scale — 200+ remaining items
overwhelm the context. Trying to push too much into SQL (v3) creates brittle
rules that overfit to the data.

### 2. Precision is non-negotiable

In reconciliation, false positives get booked incorrectly while false negatives
stay in a review queue. We targeted 100% precision first, then optimized recall.
All v4b runs maintain 100% precision across seeds and difficulties.

### 3. Don't fine-tune SQL to generator quirks

The v3 spec hit 99.8% F1 but was fragile — tuned to specific generator patterns
(fee lists, batch size ranges, entry_type fields). When we audited for
generator-specific patterns, we found 7 distinct fine-tuning points.

General rules work nearly as well and are robust across seeds:
- `starts_with(gl_entity, bank_entity)` instead of exact match
- `>= 2` for batch groups instead of `BETWEEN 2 AND 5`
- No hardcoded fee lists — let the LLM judge bank fees
- No `entry_type` filters

### 4. LLM prompt specificity matters for thresholds

The tolerance matching prompt said "less than $100" but on one seed the LLM chose
a $25 threshold, missing all wire fee mismatches in the $25–$50 range (25 of 47
mismatches). Making it explicit — "Use `abs(...) <= 100` as the threshold — do NOT
use a smaller cutoff like $25 or $50" — fixed the issue completely (0 mismatches
missed on all seeds).

**Lesson**: When a specific numeric threshold matters, state the exact value AND
explicitly prohibit the common conservative alternative. LLMs tend toward
conservatism; counteract this with direct instructions.

### 5. Bank description truncation breaks entity matching

Bank descriptions are typically truncated to ~25 characters. This cuts legal
suffixes mid-word: "CORNERSTONE DEVELOPMENT LTD" → "CORNERSTONE DEVELOPMENT L".

Fix: strip trailing 1–3 character fragments (`\s+[A-Z]{1,3}$`) from the bank
entity. This is general-purpose — any bank system with fixed-width description
fields produces this pattern.

### 6. Batch deposits need SQL, not just LLM

Batch deposits (one bank item = multiple GL entries from same vendor) need SQL
because:
- Grouping + summing is exact, deterministic work
- The LLM struggles with subset-sum reasoning over many candidates
- SQL catches the easy cases (full entity group matches), leaving only hard
  subset cases for the LLM

Without the SQL batch node, batch recall was 17.5%. With it: 92–100%.

### 7. The subset-sum problem is the hardest remaining challenge

When an entity has more GL entries than any single bank deposit covers, the SQL
batch node can't match (it groups ALL entries for the entity). The LLM should
handle subsets but sometimes doesn't.

Example: "OAKWOOD ENGINEERING" has 5 GL entries totaling $251,510 but the batch
deposit is $241,950 (4 of 5 entries). The 5th entry belongs to a different match.

This accounts for most remaining FN (4–17 per run on HARD).

### 8. Some cases are fundamentally unsolvable

"Duplicate amount pairs" — two bank items with identical amounts and no entity
signal, matched to two GL items also with no distinguishing signal. Example: two
airline charges both at -$1,450 (DELTA AIR / UNITED) matched to two GL entries
both described as "Business travel" with different vendor names.

No algorithm can solve these without external data. They account for 2–4 FN per
run and should be accepted as a ceiling.

### 9. Offsetting pairs need entity confirmation

GL offsetting pairs (equal-and-opposite entries that cancel) had false positives
when matched on amount alone. Two problems:
- "Self-correcting" entries that should be 1:1 bank-matched, not offset
- Cross-entity coincidental amount matches

Fix: require the entity in both entries to match (for void entries, extract
entity from the "VOID - ... to ENTITY (...)" description pattern). This
eliminated all 5 false positive offsetting pairs.

### 10. DuckDB-specific gotchas

- **Views are late-binding**: deeply nested view chains timeout at 30s. Use CTEs
  within a single view instead.
- **RE2 regex**: `\b` word boundary doesn't work. Use `(^|\s)..(\s|$)` instead.
- **Legal suffix stripping**: A global regex replacing "CO" also strips from
  "CONSULTING" → "NSULTING". Must use word-boundary patterns.
- `starts_with()` for prefix matching, `jaro_winkler_similarity()` for fuzzy
  matching, `QUALIFY` for window function filtering.

## Benchmark results (all n=1000, model=gpt-5.2)

| Version | Difficulty | Seed | F1 | Precision | Recall | Batch | Mismatch | FP | FN | Time | Cost |
|---------|-----------|------|-----|-----------|--------|-------|----------|----|----|------|------|
| v4b | easy | 42 | 100% | 100% | 100% | 100% | 100% | 0 | 0 | 255s | $0.39 |
| v4b | medium | 42 | 100% | 100% | 100% | 100% | 100% | 0 | 0 | 351s | $0.61 |
| v4b | hard | 42 | 99.8% | 100% | 99.6% | 100% | 100% | 0 | 4 | 334s | $0.36 |
| v4b | hard | 99 | 99.7% | 100% | 99.5% | 98.4% | 100% | 0 | 6 | 222s | $0.33 |
| v4b | hard | 123 | 99.2% | 100% | 98.5% | 92.9% | 100% | 0 | 17 | 231s | $0.32 |

Earlier versions for comparison:

| Version | Difficulty | Seed | F1 | Precision | Recall | Notes |
|---------|-----------|------|-----|-----------|--------|-------|
| v3d | hard | 42 | 99.8% | 99.8% | 99.8% | Generator-specific tuning |
| v4 | hard | 42 | 97.0% | 100% | 94.2% | No batch SQL |
| v4 | hard | 99 | 89.3% | 100% | 80.7% | Batch recall collapsed to 17.5% |

## Cost breakdown

At OpenRouter gpt-5.2 pricing ($1.75/M input, $0.175/M cached, $14/M output):
- Average cost per 1000-transaction run: **$0.40**
- ~80% of tokens are cached (system prompt reuse across iterations)
- The prompt node (match_residual) uses 5–15 LLM iterations
- Total wall-clock time: 3.5–6 minutes

## Remaining FN analysis (HARD difficulty)

| Category | Count per run | Solvable? |
|----------|--------------|-----------|
| Batch subset-sum | 4–15 | Possibly with combinatorial SQL or better LLM prompting |
| Ambiguous dup-amount pairs | 2–4 | No — no textual signal exists |
| Isolated 1:1 misses | 0–2 | LLM non-determinism; varies by run |
