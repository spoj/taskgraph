# Bank Reconciliation Examples

This directory contains code and experiments exploring how to perform Bank Reconciliation using TaskGraph. Bank Rec is a complex, heuristic-heavy matching problem traditionally solved by fine-tuned Python scripts. We use it here to showcase how combining deterministic SQL nodes with non-deterministic LLM `prompt` nodes can achieve high accuracy while remaining highly generalized.

## Core Files

- `bank_rec_generator.py`: An extremely robust synthetic bank reconciliation problem generator. Generates datasets of arbitrary size with tunable difficulties (injecting typo noise, cryptic descriptions, offsetting entries, transpositions, and date clearing lags).
- `bank_rec_v4.py`: The definitive "Hybrid" pipeline. It uses pure SQL to filter out all obvious deterministic matches, then delegates the remaining messy/ambiguous transactions to a `prompt` node for final fuzzy matching.
- `solve_bank_rec.py`: A highly-tuned deterministic Python solver for comparison. Uses heavy regexes, subsets sums, and text normalization.
- `score_bank_rec.py`: CLI tool for scoring the output of a solver/pipeline against the Ground Truth produced by the generator. 
- `BANK_REC_LESSONS.md`: Contains a narrative log of our previous learnings trying to solve this task iteratively with TaskGraph (v2, v3, v4).

## Evaluation Benchmark

Check out the `eval_bank_rec/` directory for a fully packaged, reproducible benchmark suite that compares 5 different reconciliation strategies (from generic python, to hand-tuned python, to LLM-SQL hybrid, to pure-prompt SQL). It includes `run_all.sh` to generate the testbed and execute all strategies automatically.

## Archive

The `archive/` directory contains older iterations and experiments.
- `bank_rec_problem.py`: The original, static ~100-record problem definition file.
- `bank_rec_v2_*.py / bank_rec_v3_*.py`: Our early attempts at building the pipeline before we discovered the ideal Hybrid split in v4.
- `bank_rec_v5.py`: An attempt to completely replace the LLM with 8 additional SQL nodes (which proved that SQL alone misses too many edge cases).
