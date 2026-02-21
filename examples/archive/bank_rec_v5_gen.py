# ARCHIVE: Historical snapshot — imports reference old paths and will not run. See examples/bank_rec/ for current code.
"""Bank Reconciliation V5 — Generated Problems (all-SQL, no LLM)

All-SQL deterministic pipeline. No API key needed.

Configure via environment variables:
    BANK_REC_N          Target bank transaction count  (default: 200)
    BANK_REC_DIFFICULTY  easy | medium | hard           (default: hard)
    BANK_REC_SEED       Random seed                    (default: 42)

Usage:
    BANK_REC_N=1000 BANK_REC_DIFFICULTY=hard BANK_REC_SEED=42 \
        uv run tg run --spec examples/bank_rec_v5_gen.py -o gen1000_v5.db

Scoring:
    uv run python examples/score_bank_rec.py gen1000_v5.db --n 1000 --seed 42 --difficulty hard
"""

import os

from examples.bank_rec_generator import generate, verify
from examples.bank_rec_v5 import (
    BATCH_MATCH_SQL,
    BATCH_MATCH_VALIDATE,
    BATCH_SUBSET_SQL,
    BATCH_SUBSET_VALIDATE,
    DEPOSIT_NSF_SQL,
    FEATURES_SQL,
    MATCH_CERTAIN_SQL,
    MATCH_CERTAIN_VALIDATE,
    MATCH_REMAINING_SQL,
    MATCH_REMAINING_VALIDATE,
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
    f"[bank_rec_v5_gen] n_bank={_summary['bank_transactions']}  "
    f"n_gl={_summary['gl_entries']}  difficulty={DIFFICULTY}  seed={SEED}"
)
print(
    f"[bank_rec_v5_gen] 1:1={_summary['matches_1to1']}  "
    f"batch={_summary['matches_many_to_one']}  "
    f"mismatch={_summary['matches_amount_mismatch']}  "
    f"offset={_summary['offsetting_gl_pairs']}  "
    f"ub={_summary['unmatched_bank']}  ug={_summary['unmatched_gl']}"
)

# ---------------------------------------------------------------------------
# Nodes (10: 2 source + 8 SQL, 0 prompt)
# ---------------------------------------------------------------------------

NODES = [
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
    {
        "name": "features",
        "sql": FEATURES_SQL,
        "depends_on": ["bank_txns", "gl_entries"],
    },
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
    {
        "name": "deposit_nsf",
        "sql": DEPOSIT_NSF_SQL,
        "depends_on": ["features", "match_certain"],
        "output_columns": {
            "deposit_nsf_matched": [
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
        "depends_on": ["features", "match_certain", "deposit_nsf"],
    },
    {
        "name": "batch_match",
        "sql": BATCH_MATCH_SQL,
        "validate": {"main": BATCH_MATCH_VALIDATE},
        "depends_on": ["features", "match_certain", "offsetting", "deposit_nsf"],
    },
    {
        "name": "batch_subset",
        "sql": BATCH_SUBSET_SQL,
        "validate": {"main": BATCH_SUBSET_VALIDATE},
        "depends_on": ["batch_match"],
    },
    {
        "name": "match_remaining",
        "sql": MATCH_REMAINING_SQL,
        "validate": {"main": MATCH_REMAINING_VALIDATE},
        "depends_on": [
            "features",
            "match_certain",
            "deposit_nsf",
            "offsetting",
            "batch_match",
            "batch_subset",
        ],
        "output_columns": {
            "match_remaining_all_matched": [
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
        "depends_on": ["features", "match_remaining", "offsetting"],
    },
]
