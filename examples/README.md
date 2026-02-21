# Examples

This directory contains examples and experiments exploring how to use TaskGraph.

## Basic Examples (`basics/`)

Simple TaskGraph specs demonstrating core concepts:
- `single_task/` — Minimal single-node spec.
- `linear_chain/` — Linear chain of dependent nodes.
- `diamond_dag/` — Diamond-shaped DAG with convergent dependencies.
- `validation_demo/` — Validation views and output schema checks.

## Bank Reconciliation Benchmark (`bank_rec/`)

A complex, heuristic-heavy matching problem traditionally solved by fine-tuned Python scripts. Showcases how combining deterministic SQL nodes with LLM prompt nodes can achieve high accuracy while remaining highly generalized.

- `generator.py` — Synthetic bank reconciliation dataset generator with tunable difficulty (typos, cryptic descriptions, offsetting entries, transpositions, date lags).
- `generate_dataset.py` — CLI wrapper that generates `dataset.json` for benchmarking.
- `strategy3_hybrid.py` — The definitive hybrid pipeline: SQL handles deterministic matches (check numbers, entity+amount, batches, offsetting), LLM handles fuzzy residual.
- `strategy1_generic.py` — Naive deterministic Python solver (amount+date only).
- `strategy2_tuned.py` / `strategy2a_tuned_detailed.py` — Hand-tuned Python solvers with regexes, subset-sum, entity normalization.
- `strategy3a_sql_only.py` — Ablation: hybrid pipeline with the LLM node stripped out (SQL-only).
- `strategy4_pure_prompt.py` — Pure LLM approach: prompt node writes all matching SQL from scratch.
- `score.py` — Scoring library for evaluating solver output against ground truth.
- `score_sql_only.py` — CLI tool for scoring a `.db` workspace file directly.
- `run_all.sh` — Generates dataset and runs all strategies sequentially.
- `LEARNING_LOG.md` — Iterative discoveries and prompt engineering lessons.

See `bank_rec/README.md` for strategy descriptions and `BANK_REC_LESSONS.md` for detailed multi-model benchmark results, architectural evolution, and cost analysis.

## Lessons Learned (`BANK_REC_LESSONS.md`)

Narrative log of architectural evolution from v1 (monolithic LLM) through v4b (balanced hybrid), with benchmark results and key design decisions.

## Archive (`archive/`)

Historical iterations (v2, v3, v4, v5) preserved for reference. These files have stale import paths and are not runnable — they document the evolution of the approach.
