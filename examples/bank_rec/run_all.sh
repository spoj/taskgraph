#!/bin/bash
set -e

echo "=========================================================="
echo "Generating Dataset (1000 records, HARD difficulty)"
echo "=========================================================="
uv run python generate_dataset.py --n 1000 --seed 123 --difficulty hard --out dataset.json

echo ""
echo "=========================================================="
echo "Strategy 1: Naive Deterministic Python"
echo "=========================================================="
uv run python strategy1_generic.py --dataset dataset.json

echo ""
echo "=========================================================="
echo "Strategy 2: Fine-Tuned Python Heuristics"
echo "=========================================================="
uv run python strategy2_tuned.py --dataset dataset.json

echo ""
echo "=========================================================="
echo "Strategy 3: Hybrid TaskGraph (SQL Fuzzy + LLM)"
echo "=========================================================="
rm -f runs/strategy3_hybrid.db
uv run python ../../scripts/cli.py run --spec run_strategy3_hybrid.py -o runs/strategy3_hybrid.db
uv run python score.py runs/strategy3_hybrid.db --dataset dataset.json

echo ""
echo "=========================================================="
echo "Strategy 3a: Ablation (SQL Only, No LLM)"
echo "=========================================================="
rm -f runs/strategy3a_sql_only.db
uv run python ../../scripts/cli.py run --spec strategy3a_sql_only.py -o runs/strategy3a_sql_only.db
uv run python score_sql_only.py runs/strategy3a_sql_only.db

echo ""
echo "=========================================================="
echo "Strategy 4: Pure Prompt TaskGraph"
echo "=========================================================="
rm -f runs/strategy4_pure_prompt.db
uv run python ../../scripts/cli.py run --spec strategy4_pure_prompt.py -o runs/strategy4_pure_prompt.db
uv run python score.py runs/strategy4_pure_prompt.db --dataset dataset.json
