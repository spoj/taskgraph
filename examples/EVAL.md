# Strategy Evaluation: Bank Reconciliation & Cash Application

Comparative evaluation of four matching strategies across two financial
reconciliation benchmarks. All LLM-dependent results use `gpt-5.2` at
`reasoning_effort=low` unless otherwise noted.

---

## 1. Bank Reconciliation

### Problem Statement

Bank reconciliation matches **bank statement transactions** against **general
ledger (GL) entries** to verify they represent the same economic events. The two
sides record the same transactions but with different descriptions, dates,
amounts, and groupings.

The solver must produce:

- **Matched pairs** `(bank_id, gl_id)` — which bank items correspond to which GL entries
- **Batch matches** — one bank deposit covering multiple GL entries that sum to the bank amount
- **Offsetting GL pairs** — GL entries that cancel each other (void + original) with no bank counterpart
- **Unmatched bank** — bank fees/interest with no GL entry
- **Unmatched GL** — outstanding checks/deposits in transit

### Dataset

Generated with `n=1000, seed=123, difficulty=hard`. Contains 1,004 bank
transactions and 1,341 GL entries.

| Category | Count |
|----------|-------|
| 1:1 matches | 840 |
| Batch deposits (70 deposits, 243 GL entries) | 70 |
| Amount mismatches (fee/transposition) | 47 |
| Offsetting GL pairs | 82 |
| Unmatched bank (fees) | 47 |
| Unmatched GL (outstanding) | 47 |

### Generator Design

The generator produces 10 structural event types, each modeling a real-world
complication, plus 8 difficulty modifiers applied to simple matches.

#### Structural Event Types

| Event Type | Hard Rate | What It Models |
|-----------|-----------|----------------|
| Simple 1:1 | 42% | Clean match — but subject to modifier flags below |
| Batch deposit | 6% | One bank deposit = sum of 2-5 GL entries from same vendor |
| Void/reissue | 4% | Original check voided, new check issued. GL has 3 entries (original + void + reissue) |
| Self-correcting | 4% | Wrong payment, reversal, correct payment — 3 bank entries matched to 3 GL entries |
| NSF return | 3% | Deposit + returned-item pair (bounced check) |
| Amount mismatch | 4% | Wire fee netted ($10-$50) or digit transposition error |
| Duplicate vendor payment | 4% | Same vendor, same amount, different check numbers, days apart |
| Unmatched bank | 4% | Service charges, wire fees, maintenance fees — no GL counterpart |
| Unmatched GL | 4% | Outstanding checks / deposits in transit |
| Offsetting GL | 3% | GL void pair that nets to zero, no bank counterpart |

#### Difficulty Modifiers (applied to simple 1:1 events)

| Modifier | Hard Rate | Example |
|----------|-----------|---------|
| Date shift | 40% | Check clears 3-14 days after GL posting; ACH clears 1-4 days later. Weekend-aware. |
| Cryptic description | 25% | Bank shows `AMZN MKTP US*A1B2C3` instead of "Amazon Office Supplies". Also: `ADP TOFCPYRL`, `MSFT *E5PLAN`, `PG&E WEBPMT`. |
| Truncated description | 30% | `CORNERSTONE DEVELOPMENT LTD` becomes `CORNERSTONE DEVELOP` (12-20 chars, sometimes with spaces stripped) |
| Duplicate amounts | 22% | Two unrelated transactions with identical amounts but different vendors, 5-15 days apart |
| Same amount + date | 8% | Two transactions with identical amounts AND dates. Uses confusable vendor names from same root (e.g., "Johnson & Associates" vs "K. Johnson Consulting Group") |
| Weekly recurring | 12% | 4 weekly payments to same vendor, 60% fixed amount / 40% with +/-5% variance |
| Prior period | 5% | GL dated last 5 days of prior month, bank clears in first 5 days of current month |
| Typos | 15% | Character deletion or adjacent transposition in GL vendor name |

At hard difficulty, a simple 1:1 match can have multiple modifiers stacked —
e.g., a date-shifted, truncated, typo-containing entry with a duplicate amount
from another vendor.

### Strategies

**S1: Naive Deterministic.** Loops over bank entries, finds GL entries with exact
amount match within 3 calendar days, picks closest date. No description matching
at all. No handling of batches, offsets, mismatches, or fees.

**S2: Tuned Deterministic.** Multi-pass Python solver (10 passes, highest
confidence first): check number + exact amount, entity-prefix + exact amount,
chronological series matching (weekly recurring), NSF pairing, offsetting GL
pairs (entity-confirmed), batch deposits (full entity group sum, then
brute-force subset-sum up to size 8), exact amount + unique amount + date
proximity, entity-prefix + tolerance, fallback exact amount + closest date.
Implements Jaro-Winkler similarity, edit distance, legal suffix stripping.

**S3: Hybrid (SQL + LLM).** 9-node DAG. Six SQL nodes handle deterministic
matching: check number, entity-prefix + exact amount, Jaro-Winkler fuzzy match,
chronological series, offsetting pairs (entity-confirmed), batch deposits
(full-group sum + pairwise subset-sum), exact-amount closest-date fallback. One
LLM prompt node (`match_residual`) handles the remaining ambiguous cases: amount
tolerance matching, cryptic description resolution, cross-entity batch subsets,
manual judgment calls. Final SQL node consolidates results.

**S3a: SQL-Only Ablation.** Identical to S3 but with the `match_residual` LLM
node stripped out. Measures how much the LLM contributes beyond deterministic
SQL.

**S4: Pure Prompt.** Two LLM prompt nodes, no SQL preprocessing. One node writes
SQL to match bank/GL entries using fuzzy string matching and amount tolerance.
Second node identifies offsetting GL pairs. The LLM decides all matching
strategy from a brief natural-language description.

### Results

| Strategy | F1 | Precision | Recall | 1:1 Recall | Batch Recall | Mismatch Recall | Offset Recall |
|----------|-----|-----------|--------|------------|--------------|-----------------|---------------|
| S1: Naive | 76.0% | 95.3% | 63.2% | 85.0% | 0.0% | 0.0% | 0.0% |
| S2: Tuned | 90.3% | 99.6% | 82.7% | 99.5% | 26.7% | 70.2% | 100% |
| S3a: SQL-only | 88.4% | 99.9% | 79.3% | 99.5% | 24.7% | 0.0% | 100% |
| S3: Hybrid | **97.0%** | 99.8% | 94.3% | 99.6% | **80.7%** | **70.2%** | 100% |
| S4: Pure Prompt | ~79%\* | — | — | — | <10%\* | — | — |

\* S4 result is approximate from an earlier run; no gpt-5.2 low artifact was
available for re-scoring.

All strategies achieve 100% recall on unmatched bank and unmatched GL
identification (except S1, which doesn't attempt it).

### Analysis

**1:1 matches are solved by deterministic logic.** S2, S3a, and S3 all achieve
99.5%+ recall on 1:1 pairs — the SQL nodes handle these entirely. The 0.5% gap
(3-4 misses) comes from same-amount+same-date pairs with confusable vendor
names, where no algorithm has enough signal.

**Batch deposits are the primary differentiator.** This is where the strategies
diverge most:

| Strategy | Batch GL Entries Matched | Batch Recall |
|----------|------------------------|--------------|
| S1 | 0 / 243 | 0.0% |
| S3a | 60 / 243 | 24.7% |
| S2 | 65 / 243 | 26.7% |
| S3 | 196 / 243 | 80.7% |

S3a and S2 perform comparably on batches — both handle full entity-group
matches (where all GL entries for a vendor sum to the bank deposit) but fail on
subset cases (where only some of a vendor's GL entries belong to the deposit).
The LLM in S3 recovers an additional 136 batch matches by reasoning about
which GL subsets sum to the bank amount, including cross-entity matches where
the bank description is truncated or cryptic.

**Amount mismatches require LLM judgment.** S2 and S3 both achieve 70.2%
mismatch recall (33/47), while S3a gets 0%. S2 handles mismatches via explicit
tolerance rules ($10-$100 range); S3's SQL nodes have no tolerance logic, so the
LLM handles all 33 recovered mismatches. The remaining 14 mismatches are digit
transpositions where the bank amount differs from the GL amount with no entity
signal to confirm the match.

**S3a vs S2: SQL nodes alone underperform tuned Python.** Despite being built
from the same SQL building blocks, S3a (F1=88.4%) underperforms S2 (F1=90.3%).
The difference is that S2 implements amount tolerance matching and additional
heuristics (NSF pairing, chronological series) that the S3 SQL nodes deliberately
leave to the LLM. The SQL nodes in S3 are designed to be conservative and
precise, not exhaustive.

**S4 struggles with combinatorial logic.** The pure-prompt approach can write
basic fuzzy-matching SQL, recovering most 1:1 matches. But batch deposits
require exploring combinations of GL entries that sum to a bank amount — a task
the LLM handles poorly without SQL infrastructure to narrow the candidate set.

**Precision is uniformly high.** All strategies except S1 maintain 99.6%+
precision. The 2 false positives in S3 come from ambiguous duplicate-amount
pairs where the solver picks the wrong GL entry.

---

## 2. Cash Application

### Problem Statement

Cash application matches **incoming customer payments** to **outstanding
invoices** using remittance advice data. Unlike bank reconciliation (which is
a 1:1 matching problem with some batch cases), cash application involves:

- **Multi-invoice payments** — one payment covers 2-4 invoices
- **Partial payments** — customer pays only a fraction of the invoice
- **Discounts and deductions** — early payment discounts, short-pays for disputes
- **Entity resolution** — payer name on payment differs from customer name on invoice
- **Reference garbling** — remittance invoice references contain typos, PO numbers, or wrong formats

The solver must produce:

- **Applications** `(payment_id, invoice_id, applied_amount)` — which payment applies to which invoice
- **Unmatched payments** — payments with no valid invoice match
- **Unapplied invoices** — invoices with no payment received

### Dataset

Generated with `n=100, seed=42, difficulty=hard`. Contains 101 invoices, 75
payments, and 92 remittance lines.

| Category | Count |
|----------|-------|
| Ground truth applications | 97 |
| Unmatched payments | 5 |
| Unapplied invoices | 4 |

### Generator Design

The generator produces 12 event types, each modeling a distinct accounts
receivable scenario, plus cross-cutting messiness modifiers.

#### Event Types

| Event Type | Hard Rate | What It Models |
|-----------|-----------|----------------|
| Simple 1:1 | 25% | One payment for one invoice, with remittance. Baseline case. |
| Multi-invoice | 15% | One payment covers 2-4 invoices from same customer. Payment = sum of invoice amounts. 40% chance last invoice is only partially paid. |
| Partial payment | 12% | Payment covers only 25-80% of invoice. Customer pays in installments. |
| No remittance | 10% | Payment with zero remittance lines. Must match by amount + payer name. |
| Discount taken | 8% | Customer deducts 1-3% early payment discount. Memo may mention "discount". |
| Short-pay deduction | 7% | Customer deducts 3-15% for disputes/damage. Memo has deduction reason (12 possible: damaged goods, shipping shortage, quality defect, pricing dispute, etc.). |
| Cross-reference | 5% | Remittance has garbled invoice ref. 5 garble types: digit transposition, wrong digit, prefix change ("Invoice #42"), leading zero drop, PO number substitution. |
| Unmatched payment | 4% | Payment with no corresponding invoice. May have fake remittance ref. |
| Unapplied invoice | 4% | Invoice with no payment. |
| Credit memo | 4% | Negative invoice (credit note) offsets positive invoice. Payment = net of both. |
| Duplicate payment | 3% | Same invoice paid twice. First payment applies; second is unmatched. |
| Overpayment | 3% | Customer pays 2-15% more than invoice. Applied amount = invoice amount. |

#### Messiness Modifiers (at hard difficulty)

| Modifier | Rate | Example |
|----------|------|---------|
| Name variation | 40% | "Pacific Coast Supply Co" appears as "Pac Coast Supply", "PACIFIC COAST SUPPLY CO", "PCS Co", or "Pacific Coast Wholesale" |
| Typo in ref | 20% | `INV-00042` appears as `INV-00024` (transposed), `Invoice #42` (prefix change), `INV-0042` (leading zero dropped) |
| Duplicate amounts | 15% | Reuses invoice amounts from previous invoices, creating ambiguous amount matches |
| Multi-invoice partial | 40% | In multi-invoice payments, last invoice is only partially paid |
| Date spread | 90 days | Invoice dates spread across 90 days |

Each of the 30 generated companies has 4 name variations (abbreviation,
all-caps, acronym, subsidiary name), ensuring entity resolution is non-trivial.

### Strategies

**S1: Naive Deterministic.** Two passes only: (1) exact remittance reference
lookup + exact amount match, (2) exact amount + exact customer name match
(case-insensitive). No fuzzy matching, no partial payment handling, no discount
detection, no multi-invoice grouping.

**S2: Tuned Deterministic.** Multi-pass Python solver (10 matching passes):
exact ref + exact amount, fuzzy ref (edit distance) + amount tolerance,
multi-invoice grouping from remittance, exact amount + exact customer, exact
amount + fuzzy customer (Jaro-Winkler >= 0.80), discount detection (97-99.5%
ratio), short-pay deduction (85-97% ratio), overpayment (100-115% ratio), credit
memo (positive + negative invoice netting), partial payment (25-85% ratio with
hint), fallback exact amount + closest date. Implements Jaro-Winkler similarity,
edit distance, name normalization.

**S3: Hybrid (SQL + LLM).** 7-node DAG. `features` (SQL: name normalization,
ref extraction), `match_exact` (SQL: 4 rounds — exact ref + exact amount,
fuzzy ref number + exact amount, exact ref + partial amount, exact amount +
customer name with Jaro-Winkler > 0.85), `match_residual` (LLM prompt: handles
discounts, short-pays, overpayments, credit memos, multi-invoice grouping,
duplicate detection, fallback matches), `report` (SQL: consolidate results).

**S4: Pure Prompt.** Two LLM prompt nodes, no SQL preprocessing. `match_all`
receives a detailed description of all 10 matching scenarios with DuckDB hints
and writes SQL for full matching. `report` classifies unmatched payments and
unapplied invoices.

### Results

| Strategy | F1 | Precision | Recall | Amount Accuracy |
|----------|-----|-----------|--------|-----------------|
| S1: Naive | 74.8% | 100% | 59.8% | 100% |
| S2: Tuned | **97.9%** | 100% | 95.9% | 100% |
| S3: Hybrid | **98.4%** | 100% | 96.9% | 100% |
| S4: Pure Prompt | 85.9% | 100% | 75.3% | 100% |

All strategies achieve 100% precision and 100% amount accuracy (every matched
pair has the correct applied_amount within $0.01). Unmatched payment and
unapplied invoice recall is 100% across all four strategies.

#### Per-Type Recall Breakdown

| Match Type | Count | S1 | S2 | S3 | S4 |
|-----------|-------|-----|-----|-----|-----|
| simple_1to1 | 19 | **100%** | **100%** | **100%** | 89.5% |
| multi_invoice | 32 | 84.4% | 93.8% | **100%** | 84.4% |
| multi_invoice_partial | 4 | 0% | **100%** | **100%** | **100%** |
| partial_payment | 9 | 0% | **100%** | **100%** | 77.8% |
| no_remittance | 8 | 25.0% | **100%** | 87.5% | 25.0% |
| discount_taken | 6 | 0% | 83.3% | 83.3% | 33.3% |
| short_pay_deduction | 5 | 0% | 80.0% | **100%** | 80.0% |
| cross_reference | 4 | 50.0% | **100%** | **100%** | 0% |
| duplicate_payment | 2 | **100%** | **100%** | **100%** | **100%** |
| overpayment | 2 | 0% | **100%** | 50.0% | **100%** |
| credit_memo | 6 | **100%** | **100%** | **100%** | **100%** |

### Analysis

**S2 and S3 are near-parity.** Unlike bank reconciliation — where the LLM adds
a large lift over deterministic code — in cash application the tuned
deterministic solver nearly matches the hybrid. S2 achieves F1=97.9% vs S3's
98.4%, a gap of just 0.5%. This reflects the structural difference between the
two problems: cash application has richer structured signals (remittance lines
with explicit invoice references) that deterministic code can exploit effectively.

**S3's advantage is in multi-invoice and short-pay.** The per-type breakdown
shows S3 achieves 100% on multi_invoice (vs S2's 93.8%) and 100% on
short_pay_deduction (vs S2's 80.0%). The LLM's ability to reason about
deduction memos ("damaged goods", "shipping shortage") and group remittance
lines into multi-invoice matches exceeds what regex-based rules can do.

**S3's weakness: overpayment and no-remittance.** S3 scores 50% on overpayment
(1/2) and 87.5% on no_remittance (7/8), while S2 achieves 100% on both. These
are cases where the LLM prompt's approach to residual matching doesn't
systematically check for overpayment ratios the way S2's explicit pass does.

**S4 struggles with cross-reference and no-remittance.** The pure-prompt
approach scores 0% on cross_reference (garbled invoice refs) and 25% on
no_remittance. Without SQL preprocessing to normalize names and extract numeric
parts from garbled refs, the LLM's self-authored SQL doesn't implement
sufficiently robust fuzzy matching. S4 also misses 2 of 19 simple_1to1 matches,
suggesting its SQL logic has edge cases even on easy matches.

**Discount detection is hard for everyone.** All strategies miss at least 1 of
6 discounts. The 1-3% range overlaps with rounding, making it difficult to
distinguish a discount from a minor amount mismatch without explicit remittance
memo signals.

**S1's multi-invoice success is misleading.** S1 scores 84.4% on multi_invoice
despite having no multi-invoice logic. This happens because its exact ref +
exact amount pass matches individual remittance lines whose amount equals the
invoice amount exactly — effectively matching the individual components of
multi-invoice payments one by one, as long as the ref is clean and the amount
per invoice is exact.

---

## 3. Cross-Benchmark Observations

### The hybrid approach (S3) wins in both benchmarks, but by different margins

| Benchmark | S2 F1 | S3 F1 | S3 Lift | Why |
|-----------|-------|-------|---------|-----|
| Bank rec | 90.3% | 97.0% | +6.7pp | LLM handles batch subset-sum, cryptic descriptions, amount tolerance |
| Cash app | 97.9% | 98.4% | +0.5pp | LLM handles residual multi-invoice grouping, deduction memos |

The lift is proportional to how much **ambiguous judgment** the problem requires.
Bank reconciliation has severe description mangling (truncation, cryptic codes,
typos) and combinatorial batch matching — areas where deterministic rules break
down. Cash application has richer structured data (remittance lines, explicit
invoice refs) that gives deterministic code more to work with.

### SQL handles structure, LLM handles judgment

Across both benchmarks, the pattern is consistent:

- **Exact reference matching, amount equality, entity-prefix matching** — handled
  entirely by SQL nodes with 99.5%+ accuracy
- **Fuzzy entity resolution, amount tolerance, combinatorial grouping, memo
  interpretation** — handled by the LLM residual node

The S3a ablation (bank rec, SQL-only) quantifies this split: SQL alone reaches
88.4% F1; the LLM adds 8.6 percentage points by handling the cases SQL can't.

### Precision is uniformly high across all approaches

Every strategy (except bank rec S1) maintains 99.6%+ precision. In both
benchmarks, false positives are rare — strategies err toward missing matches
rather than making wrong ones. This is the correct bias for financial
reconciliation, where a false positive gets booked incorrectly while a false
negative stays in a review queue.

### The pure-prompt approach (S4) underperforms the naive baseline on some sub-tasks

S4 scores below S1 on bank rec overall and below S2 on cash app. The LLM can
write basic matching SQL from scratch, but:

- It doesn't implement robust fuzzy matching (missing cross-references entirely in cash app)
- It can't handle combinatorial search (batch deposits in bank rec)
- Its self-authored SQL sometimes has edge cases that miss even simple matches

S4 demonstrates that LLM capability is best leveraged when paired with
structured SQL preprocessing rather than used as a replacement for it.

### Problem structure determines the ceiling for deterministic approaches

| Problem | Deterministic Ceiling (S2) | Hybrid (S3) | Gap Source |
|---------|---------------------------|-------------|------------|
| Bank rec | 90.3% | 97.0% | Batch subset-sum, cryptic descriptions, tolerance |
| Cash app | 97.9% | 98.4% | Multi-invoice edge cases, deduction memo parsing |

Cash application's higher deterministic ceiling (97.9% vs 90.3%) reflects its
richer structured signals — each payment has remittance lines with invoice
references, amounts, and memos that directly map to invoices. Bank reconciliation
has no such structured link; the two sides share only amounts, dates, and
free-text descriptions.

---

## Appendix: Run Configuration

| Benchmark | Dataset | Records | LLM Strategies Model | Reasoning |
|-----------|---------|---------|---------------------|-----------|
| Bank rec | `dataset.json` (n=1000, seed=123, hard) | 1,004 bank + 1,341 GL | openai/gpt-5.2 | low |
| Cash app | `problems/n100_seed42.json` (n=100, seed=42, hard) | 101 inv + 75 pmt + 92 rem | openai/gpt-5.2 | low |

S1 and S2 are pure Python (zero LLM cost, sub-second execution). S3 and S4 are
taskgraph specs.

**Missing artifacts:** Bank rec S4 has no gpt-5.2 low run artifact; the ~79%
figure is approximate from an earlier run documented in `bank_rec/README.md`.
