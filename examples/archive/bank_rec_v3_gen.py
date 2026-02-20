"""Bank Reconciliation V3 — Generated Problems

Same as bank_rec_v2_gen.py but uses the V3 redesigned spec.

Configure via environment variables:
    BANK_REC_N          Target bank transaction count  (default: 200)
    BANK_REC_DIFFICULTY  easy | medium | hard           (default: hard)
    BANK_REC_SEED       Random seed                    (default: 42)

Usage:
    BANK_REC_N=200  tg run --spec examples/bank_rec_v3_gen.py -o gen200.db
    BANK_REC_N=1000 tg run --spec examples/bank_rec_v3_gen.py -o gen1000.db

Scoring (after run completes):
    uv run python examples/score_bank_rec.py gen200.db --n 200 --seed 42
"""

import os

from examples.bank_rec_generator import generate, verify
from examples.bank_rec_v3_hybrid import (
    AMOUNT_MATCH_SQL,
    AMOUNT_MATCH_VALIDATE,
    BATCH_MATCH_SQL,
    BATCH_MATCH_VALIDATE,
    FEATURES_SQL,
    MATCH_CERTAIN_SQL,
    MATCH_CERTAIN_VALIDATE,
    MATCH_RESIDUAL_INTENT,
    MATCH_RESIDUAL_VALIDATE,
    OFFSETTING_SQL,
    REPORT_SQL,
    REPORT_VALIDATE,
    TOLERANCE_SQL,
    TOLERANCE_VALIDATE,
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
# Nodes
# ---------------------------------------------------------------------------

NODES = [
    # Sources
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
    # Feature extraction
    {
        "name": "features",
        "sql": FEATURES_SQL,
        "depends_on": ["bank_txns", "gl_entries"],
    },
    # Conservative 1:1 matching
    {
        "name": "match_certain",
        "sql": MATCH_CERTAIN_SQL,
        "validate": {"main": MATCH_CERTAIN_VALIDATE},
        "depends_on": ["features"],
        "output_columns": {
            "match_certain_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
        },
    },
    # Offsetting GL pairs
    {
        "name": "offsetting",
        "sql": OFFSETTING_SQL,
        "depends_on": ["features", "match_certain"],
    },
    # Batch deposit matching (SQL)
    {
        "name": "batch_match",
        "sql": BATCH_MATCH_SQL,
        "validate": {"main": BATCH_MATCH_VALIDATE},
        "depends_on": ["features", "match_certain", "offsetting"],
    },
    # Tolerance matching (SQL — wire fees)
    {
        "name": "tolerance",
        "sql": TOLERANCE_SQL,
        "validate": {"main": TOLERANCE_VALIDATE},
        "depends_on": ["features", "match_certain", "offsetting", "batch_match"],
    },
    # Amount-based matching for cryptic descriptions (SQL)
    {
        "name": "amount_match",
        "sql": AMOUNT_MATCH_SQL,
        "validate": {"main": AMOUNT_MATCH_VALIDATE},
        "depends_on": [
            "features",
            "match_certain",
            "offsetting",
            "batch_match",
            "tolerance",
        ],
    },
    # Residual matching (prompt — cryptic descriptions only)
    {
        "name": "match_residual",
        "prompt": MATCH_RESIDUAL_INTENT,
        "validate": {"main": MATCH_RESIDUAL_VALIDATE},
        "depends_on": [
            "features",
            "match_certain",
            "offsetting",
            "batch_match",
            "tolerance",
            "amount_match",
        ],
        "output_columns": {
            "match_residual_all_matched": [
                "bank_id",
                "gl_id",
                "bank_amount",
                "gl_amount",
                "match_type",
                "note",
            ],
        },
    },
    # Report
    {
        "name": "report",
        "sql": REPORT_SQL,
        "validate": {"main": REPORT_VALIDATE},
        "depends_on": ["features", "match_residual", "offsetting"],
    },
]
