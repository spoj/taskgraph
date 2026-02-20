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
echo "Strategy 3: Hybrid TaskGraph (SQL + LLM)"
echo "=========================================================="
rm -f hybrid.db
uv run python ../../scripts/cli.py run --spec strategy3_hybrid.py -o hybrid.db
# Evaluate using custom query script or the generic scorer
# Wait, score_bank_rec.py expects the GT to be generated via arguments. We will just use the standard scorer but feed it the seed:
uv run python ../score_bank_rec.py hybrid.db --n 1000 --seed 123 --difficulty hard

echo ""
echo "=========================================================="
echo "Strategy 4: Pure Prompt TaskGraph"
echo "=========================================================="
rm -f pure_prompt.db
uv run python ../../scripts/cli.py run --spec strategy4_pure_prompt.py -o pure_prompt.db
uv run python ../score_bank_rec.py pure_prompt.db --n 1000 --seed 123 --difficulty hard

echo ""
echo "=========================================================="
echo "Strategy 5: Ablation (SQL Only, No LLM)"
echo "=========================================================="
rm -f sql_only.db
uv run python ../../scripts/cli.py run --spec strategy5_sql_only.py -o sql_only.db
uv run python score_sql_only.py
