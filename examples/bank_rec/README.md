# Bank Reconciliation Evaluation Benchmarks

This directory contains a suite of experiments to evaluate different approaches to solving the Bank Reconciliation problem using `taskgraph`. 

We evaluate 5 distinct strategies, ranging from rigid Python heuristics to pure LLM zero-shot prompts.

## The Problem
We use `generate_dataset.py` to generate a `dataset.json` of 1,000 transactions at `hard` difficulty.
This dataset includes:
- Typographical errors in vendor names.
- Severe truncations of Bank descriptions.
- Complex cryptic descriptions (e.g. `AMZN MKTP US`).
- Transposition errors (e.g. $145.23 vs $154.23).
- Small wire fees (amount mismatches).
- Offsetting GL entries (same day void/reversals).
- Batch deposits / Payments.
- Multi-day weekend-aware clearing lags.

## How to Run

Execute the `run_all.sh` script to generate a fresh dataset and run all strategies sequentially. Be warned that the TaskGraph strategies will use your default LLM and may take a few minutes each.

```bash
bash run_all.sh
```

## The Strategies

### 1. Strategy 1: Naive Deterministic Python (`strategy1_generic.py`)
- **Approach**: A simple python script that loops over all entries, matching exactly on amount and looking for dates within 3 days. It ignores descriptions completely.
- **Result**: F1 ~74%. It perfectly catches clean 1:1 matches but completely hallucinates on rows with identical amounts, and natively fails on all batch deposits, fee mismatches, and offsetting pairs.

### 2. Strategy 2: Fine-Tuned Python Heuristics (`strategy2_tuned.py`)
- **Approach**: A highly complex, hand-tuned Python script (`solve_bank_rec.py`). Uses regex for check numbers, normalizes entity names by stripping suffixes, isolates offset pairs, and uses brute-force subset sum for batching.
- **Result**: F1 ~84%. Extremely fast (milliseconds). It perfectly handles the offsetting pairs and standard batching. However, it is fundamentally brittle. When faced with the severe truncation and typo modifications introduced in the `hard` dataset, the entity normalization heuristics break down, causing it to miss over a hundred valid pairs.

### 3. Strategy 3: Hybrid TaskGraph (`strategy3_hybrid.py`)
- **Approach**: A TaskGraph pipeline that uses strict, highly accurate SQL nodes to solve the deterministic parts of the problem (exact matches, check numbers, batching). It then funnels the ambiguous leftovers (typos, mismatches, transpositions) into a single LLM prompt node to resolve using zero-shot fuzzy reasoning.
- **Result (GPT-5.2 low)**: F1 **97.0%** (Precision 99.8%, Recall 94.3%). 193s, $0.26. With early validation feedback. Batch recall 80.7%.
- **Result (GPT-5.2 high)**: F1 **98.9%** (Precision 99.8%, Recall 98.1%). 692s, $0.91. Batch recall 97.9%.
- **Result (Claude Opus 4.6)**: F1 **99.7%** (Precision 99.8%, Recall 99.6%). 1440s, $3.66. Opus systematically explores cross-entity batch matches, achieving 100% batch recall and 95.7% mismatch recall.
- **Result (Gemini Flash)**: F1 **91.2%** (Precision 99.7%, Recall 84.0%). 114s, $0.15. Cheapest and fastest but misses ~68% of batch deposits.

### 4. Strategy 4: Pure Prompt TaskGraph (`strategy4_pure_prompt.py`)
- **Approach**: An entirely LLM-driven approach. The LLM is provided the schemas and asked to write raw DuckDB SQL to perform the reconciliation autonomously, guided by high-level plain English rules.
- **Result**: F1 ~79%. Surprisingly competent. The LLM successfully crafted SQL queries using `UPPER` and `LIKE` string matching that recovered the vast majority of 1:1 matches. However, it struggled with the combinatorial logic needed to discover multi-row batch deposits (recovering <10% of them). 

### 5. Ablation Study: SQL Only (`strategy3a_sql_only.py`)
- **Approach**: To measure exactly how much the LLM contributes to the Hybrid approach, this strategy runs the Hybrid TaskGraph but simply strips out the LLM node (`match_residual`).
- **Result**: F1 ~60%. Without the LLM, the SQL nodes missed over 600 valid matches, entirely failing on fee mismatches, transpositions, and mangled vendor names. In the Hybrid pipeline, the LLM autonomously recovers ~300+ of these missed matches, demonstrating exactly why deterministic heuristics alone are insufficient for real-world reconciliation.
