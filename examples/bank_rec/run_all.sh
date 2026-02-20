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
echo "Strategy 3: Hybrid TaskGraph v4 (Older baseline)"
echo "=========================================================="
rm -f runs/hybrid_v4.db
uv run python ../../scripts/cli.py run --spec run_hybrid_v4.py -o runs/hybrid_v4.db
uv run python score.py runs/hybrid_v4.db --n 1000 --seed 123 --difficulty hard

echo ""
echo "=========================================================="
echo "Strategy 4: Hybrid TaskGraph (Current SOTA)"
echo "=========================================================="
rm -f runs/hybrid.db
uv run python ../../scripts/cli.py run --spec run_hybrid.py -o runs/hybrid.db
uv run python score.py runs/hybrid.db --n 1000 --seed 123 --difficulty hard

echo ""
echo "=========================================================="
echo "Strategy 5: Pure Prompt TaskGraph"
echo "=========================================================="
rm -f runs/pure_prompt.db
uv run python ../../scripts/cli.py run --spec strategy5_pure_prompt.py -o runs/pure_prompt.db
uv run python score.py runs/pure_prompt.db --n 1000 --seed 123 --difficulty hard

echo ""
echo "=========================================================="
echo "Strategy 6: Ablation (SQL Only, No LLM)"
echo "=========================================================="
rm -f runs/sql_only.db
uv run python ../../scripts/cli.py run --spec strategy6_sql_only.py -o runs/sql_only.db
uv run python score_sql_only.py runs/sql_only.db
