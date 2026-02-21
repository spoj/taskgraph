# Examples

This directory contains examples and experiments exploring how to use TaskGraph.

## Basic Examples (`basics/`)

Simple TaskGraph specs demonstrating core concepts:
- `single_task/` — Minimal spec: one source node, one SQL node.
- `linear_chain/` — Linear chain of dependent nodes.
- `diamond_dag/` — Diamond-shaped DAG with convergent dependencies.
- `llm_task/` — Single LLM prompt node classifying data.
- `validation_demo/` — Validation views and output schema checks.

## Bank Reconciliation Benchmark (`bank_rec/`)

A complex, heuristic-heavy matching problem traditionally solved by fine-tuned Python scripts. Showcases how combining deterministic SQL nodes with LLM prompt nodes can achieve high accuracy while remaining highly generalized.

- `generator.py` — Synthetic dataset generator with tunable difficulty.
- `generate_dataset.py` — CLI wrapper that generates `dataset.json`.
- `score.py` — Scoring library (F1, precision, recall, per-type breakdown).
- `strategy1_generic.py` — Naive deterministic Python solver (amount+date only).
- `strategy2_tuned.py` / `strategy2a_tuned_detailed.py` — Hand-tuned Python solvers with regexes, subset-sum, entity normalization.
- `strategy3_hybrid.py` — Hybrid pipeline: SQL handles deterministic matches, LLM handles fuzzy residual.
- `strategy3a_sql_only.py` — Ablation: hybrid pipeline with the LLM node stripped out (SQL-only).
- `strategy4_pure_prompt.py` — Pure LLM approach: prompt node writes all matching SQL from scratch.

See `bank_rec/README.md` for strategy descriptions and `BANK_REC_LESSONS.md` for detailed multi-model benchmark results, architectural evolution, and cost analysis.

## Cash Application Benchmark (`cash_application/`)

Accounts receivable cash application: match incoming payments to open invoices using remittance data. Covers partial payments, multi-invoice payments, credit memos, discounts, overpayments, and entity name garbling.

- `generator.py` — Synthetic problem generator with 12 event types and 3 difficulty levels.
- `generate_problems.py` — CLI to produce problem sets at various sizes.
- `score.py` — Scoring library (pair F1, amount accuracy, per-type breakdown).
- `strategy1_generic.py` — Naive greedy exact matcher.
- `strategy2_tuned.py` — Multi-pass deterministic solver with fuzzy matching.
- `strategy3_hybrid.py` — Hybrid pipeline: SQL preprocessing + LLM residual matching.
- `strategy4_prompt.py` — Pure LLM approach.

## Lessons Learned (`BANK_REC_LESSONS.md`)

Narrative log of architectural evolution from v1 (monolithic LLM) through v4b (balanced hybrid), with benchmark results and key design decisions.
