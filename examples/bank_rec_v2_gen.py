"""Bank Reconciliation V2 — Generated Problems

Same solver pipeline as bank_rec_v2_hybrid.py but fed with synthetic data
from bank_rec_generator.py.  Ground truth is NOT loaded into the database
to avoid contaminating the LLM agent.  Use score_bank_rec.py after the run
to evaluate accuracy.

Configure via environment variables:
    BANK_REC_N          Target bank transaction count  (default: 200)
    BANK_REC_DIFFICULTY  easy | medium | hard           (default: hard)
    BANK_REC_SEED       Random seed                    (default: 42)

Usage:
    BANK_REC_N=200  tg run --spec examples/bank_rec_v2_gen.py -o gen200.db
    BANK_REC_N=1000 tg run --spec examples/bank_rec_v2_gen.py -o gen1000.db

Scoring (after run completes):
    uv run python examples/score_bank_rec.py gen200.db --n 200 --seed 42
"""

import os

from examples.bank_rec_generator import generate, verify
from examples.bank_rec_v2_hybrid import (
    MATCH_CONFIDENT_SQL,
    MATCH_CONFIDENT_VALIDATE,
    MATCH_HARD_INTENT,
    MATCH_HARD_VALIDATE,
    NORMALIZE_SQL,
    OFFSETTING_SQL,
    REPORT_SQL,
    REPORT_VALIDATE,
)

# ---------------------------------------------------------------------------
# Generate problem
# ---------------------------------------------------------------------------

N_BANK = int(os.environ.get("BANK_REC_N", "200"))
DIFFICULTY = os.environ.get("BANK_REC_DIFFICULTY", "hard")
SEED = int(os.environ.get("BANK_REC_SEED", "42"))

_result = generate(n_bank=N_BANK, seed=SEED, difficulty=DIFFICULTY)
_errors = verify(_result)
if _errors:
    raise ValueError(f"Generator verification failed: {_errors[:5]}")

BANK_TRANSACTIONS = _result["bank_transactions"]
GL_ENTRIES = _result["gl_entries"]
_summary = _result["ground_truth"]["summary"]

print(
    f"[bank_rec_gen] n_bank={_summary['bank_transactions']}  "
    f"n_gl={_summary['gl_entries']}  difficulty={DIFFICULTY}  seed={SEED}"
)
print(
    f"[bank_rec_gen] 1:1={_summary['matches_1to1']}  "
    f"batch={_summary['matches_many_to_one']}  "
    f"mismatch={_summary['matches_amount_mismatch']}  "
    f"offset={_summary['offsetting_gl_pairs']}  "
    f"ub={_summary['unmatched_bank']}  ug={_summary['unmatched_gl']}"
)

# ---------------------------------------------------------------------------
# Nodes — same solver pipeline as bank_rec_v2_hybrid.py
# ---------------------------------------------------------------------------

NODES = [
    # ---- Source: transaction data ----
    {
        "name": "bank_txns",
        "source": BANK_TRANSACTIONS,
        "columns": ["id", "date", "description", "amount"],
    },
    {
        "name": "gl_entries",
        "source": GL_ENTRIES,
        "columns": ["id", "date", "description", "amount", "ref", "entry_type"],
    },
    # ---- Solver pipeline (identical to bank_rec_v2_hybrid.py) ----
    {
        "name": "normalize",
        "sql": NORMALIZE_SQL,
        "depends_on": ["bank_txns", "gl_entries"],
    },
    {
        "name": "match_confident",
        "sql": MATCH_CONFIDENT_SQL,
        "validate": {"main": MATCH_CONFIDENT_VALIDATE},
        "depends_on": ["normalize"],
        "output_columns": {
            "match_confident_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
        },
    },
    {
        "name": "offsetting",
        "sql": OFFSETTING_SQL,
        "depends_on": ["normalize", "match_confident"],
    },
    {
        "name": "match_hard",
        "prompt": MATCH_HARD_INTENT,
        "validate": {"main": MATCH_HARD_VALIDATE},
        "depends_on": ["normalize", "match_confident", "offsetting"],
        "output_columns": {
            "match_hard_all_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
        },
    },
    {
        "name": "report",
        "sql": REPORT_SQL,
        "validate": {"main": REPORT_VALIDATE},
        "depends_on": ["normalize", "match_hard", "offsetting"],
    },
]
