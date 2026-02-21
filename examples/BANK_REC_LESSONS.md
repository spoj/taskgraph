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

### v4: Simplified (8 nodes)
Dropped 3 SQL nodes (`batch_match`, `tolerance`, `amount_match`). Let the LLM
handle all matching beyond check-number and entity+amount 1:1.

100% precision but recall collapsed to 80.7% on a different seed (seed=99) —
batch recall dropped to 17.5%. The LLM couldn't reliably handle batch matching
at scale in a single prompt.

### v4b: Balanced (9 nodes, current)
Re-added a **general-purpose** batch SQL node, added an exact-amount-closest-date
SQL node, and fixed two bugs. Key changes:

1. Entity truncation fix (general-purpose)
2. General-purpose batch SQL node (no generator-specific params)
3. Exact-amount + closest-date SQL node (fallback for remaining exact matches)
4. Offsetting entity confirmation (prevents false positive GL pairs)

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

## Benchmark results

### Multi-model comparison (v4b, n=1000, hard, seed=123)

All runs use the same strategy3_hybrid spec and dataset. The only variable is
the model driving the `match_residual` prompt node.

| Model | F1 | Precision | Recall | 1:1 | Batch | Mismatch | FP | FN | Iters | Time | Tokens | Est. Cost |
|-------|-----|-----------|--------|-----|-------|----------|----|----|-------|------|--------|-----------|
| claude-opus-4.6 | **99.7%** | 99.8% | 99.6% | 99.6% | **100%** | **95.7%** | 2 | 5 | 14 | 1440s | 1.32M | ~$25 |
| gpt-5.2 | 94.7% | 99.8% | 90.2% | 99.6% | 61.3% | 70.2% | 2 | 111 | 8 | 100s | 74K | ~$0.25 |
| gemini-3-flash | 91.2% | 99.7% | 84.0% | 99.8% | 32.1% | 70.2% | 3 | 181 | 18 | 114s | 231K | ~$0.15 |

All models achieve ~100% precision and identical results on 1:1 matches,
offsetting pairs, and unmatched identification — these are handled by
deterministic SQL nodes. The entire F1 gap is in the LLM-dependent categories:
batch deposits and amount mismatches.

### GPT-5.2 across seeds and difficulties

| Difficulty | Seed | F1 | Precision | Recall | Batch | Mismatch | FP | FN | Time | Cost |
|-----------|------|-----|-----------|--------|-------|----------|----|----|------|------|
| easy | 42 | 100% | 100% | 100% | 100% | 100% | 0 | 0 | 255s | $0.39 |
| medium | 42 | 100% | 100% | 100% | 100% | 100% | 0 | 0 | 351s | $0.61 |
| hard | 42 | 99.8% | 100% | 99.6% | 100% | 100% | 0 | 4 | 334s | $0.36 |
| hard | 99 | 99.7% | 100% | 99.5% | 98.4% | 100% | 0 | 6 | 222s | $0.33 |
| hard | 123 | 94.7% | 99.8% | 90.2% | 61.3% | 70.2% | 2 | 111 | 100s | $0.25 |

Note: seeds 42 and 99 were benchmarked before a code change to `_bank_desc()`
in commit `89b6d1a` that shifted the generator's RNG stream. Those results may
not be directly comparable to seed=123 results. The generator is deterministic
within a fixed code version but not across code changes — see "Scoring
infrastructure" below.

### Earlier versions for comparison

| Version | Difficulty | Seed | F1 | Precision | Recall | Notes |
|---------|-----------|------|-----|-----------|--------|-------|
| v3d | hard | 42 | 99.8% | 99.8% | 99.8% | Generator-specific tuning |
| v4 | hard | 42 | 97.0% | 100% | 94.2% | No batch SQL |
| v4 | hard | 99 | 89.3% | 100% | 80.7% | Batch recall collapsed to 17.5% |

## How Opus achieved 99.7%

The `match_residual` node is a prompt node — the LLM agent writes SQL queries
against the remaining unmatched transactions. Examining the Opus run's SQL trace
reveals a three-phase approach:

**Phase 1: General SQL discovery.** Opus wrote fuzzy-matching queries using
`jaro_winkler_similarity`, `starts_with` prefix matching, and amount tolerance
joins to find 1:1 candidates. It also wrote a batch-matching query that groups
GL entries by entity and checks if sums match bank amounts. These are the same
general techniques GPT-5.2 uses.

**Phase 2: Manual verification.** For the batch candidates surfaced by general
queries, Opus wrote verification queries with specific IDs to confirm the sums:
`SELECT SUM(amount) FROM gl_entries WHERE id IN ('G0043','G0044',...)`. It did
this systematically for all ~25 candidate batch groups across two queries.

**Phase 3: Hardcoded VALUES.** The final `match_residual_all_matched` view
contains a `manual_pairs` CTE with ~90 hardcoded `(bank_id, gl_id, match_type)`
tuples in a VALUES clause, UNIONed with the prior deterministic matches.

GPT-5.2 uses the same general approach but with fewer iterations (8 vs 14) and
far fewer tokens (74K vs 1.32M). The difference is thoroughness — Opus
systematically verified and included cross-entity batch matches (where the bank
entity and GL entity differ due to typos/truncation), while GPT-5.2 missed 94
of 243 batch GL entries.

The key insight: **batch deposit matching is where model quality matters most**.
All models handle 1:1 matches equally well via SQL. Batch matching requires the
LLM to identify that 3-5 GL entries with slightly different entity names sum to
a single bank deposit — a task that rewards systematic exploration over speed.

## Cost breakdown

At OpenRouter pricing:

| Model | Input $/M | Output $/M | Cache discount | Typical run cost |
|-------|-----------|------------|----------------|-----------------|
| gpt-5.2 | $1.75 | $14.00 | 10x | **$0.25–0.60** |
| gemini-3-flash | $0.50 | $3.00 | — | **$0.10–0.20** |
| claude-opus-4.6 | $15.00 | $75.00 | 10x | **$5–25** |

Opus cost is 50–100x GPT-5.2 for a 5% F1 improvement. Whether that tradeoff
is worth it depends on the use case — for a reconciliation where each false
negative requires manual review of a $100K+ transaction, $25 is trivial. For
routine processing, GPT-5.2 at $0.25 is the clear choice.

## Scoring infrastructure

`score.py` accepts a `--dataset` flag pointing to `dataset.json`, which contains
both the input data and the ground truth. This is the correct way to score:

```bash
uv run python score.py output.db --dataset dataset.json
```

The legacy path (no `--dataset` flag) regenerates ground truth by calling
`generate()` with the same seed. This is **unreliable** because the generator
is not deterministic across code changes — any modification that alters the
number of RNG draws (even in an unrelated code path) shifts the entire random
stream, producing different data for the same seed. The `--dataset` flag
bypasses this entirely by loading the frozen ground truth.

## Remaining FN analysis (HARD difficulty)

| Category | Count per run | Solvable? |
|----------|--------------|-----------|
| Batch subset-sum | 4–15 (GPT-5.2), 0 (Opus) | Yes — more LLM iterations or a stronger model resolves most |
| Ambiguous dup-amount pairs | 2–4 | No — no textual signal exists |
| Amount mismatch (fee/rounding) | 0–14 | Model-dependent; Opus gets 95.7%, GPT-5.2 gets 70.2% |
| Isolated 1:1 misses | 0–2 | LLM non-determinism; varies by run |
