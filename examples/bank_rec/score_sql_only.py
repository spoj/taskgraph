"""Score SQL-only strategy output against ground truth from dataset.json.

Usage:
    uv run python examples/bank_rec/score_sql_only.py runs/sql_only.db
"""

import json
import sys

import duckdb

from examples.bank_rec.score import flatten_truth, print_report, score


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "runs/sql_only.db"

    with open("dataset.json") as f:
        gt = json.load(f)["ground_truth"]

    conn = duckdb.connect(db_path, read_only=True)
    try:
        solver_pairs: set[tuple[str, str]] = set()
        try:
            rows = conn.execute("SELECT bank_id, gl_id FROM report_matched").fetchall()
            solver_pairs = {(r[0], r[1]) for r in rows}
        except duckdb.CatalogException:
            print("WARNING: report_matched view not found in DB")

        solver_offsetting: set[tuple[str, str]] = set()
        try:
            rows = conn.execute(
                "SELECT original_id, reversal_id FROM offsetting_pairs"
            ).fetchall()
            solver_offsetting = {(min(r[0], r[1]), max(r[0], r[1])) for r in rows}
        except duckdb.CatalogException:
            print("WARNING: offsetting_pairs view not found in DB")

        solver_unmatched_bank: set[str] = set()
        try:
            rows = conn.execute("SELECT id FROM report_unmatched_bank").fetchall()
            solver_unmatched_bank = {r[0] for r in rows}
        except duckdb.CatalogException:
            pass

        solver_unmatched_gl: set[str] = set()
        try:
            rows = conn.execute("SELECT id FROM report_unmatched_gl").fetchall()
            solver_unmatched_gl = {r[0] for r in rows}
        except duckdb.CatalogException:
            pass
    finally:
        conn.close()

    solver = {
        "matched_pairs": solver_pairs,
        "offsetting": solver_offsetting,
        "unmatched_bank": solver_unmatched_bank,
        "unmatched_gl": solver_unmatched_gl,
    }

    truth = flatten_truth(gt)
    scores = score(truth, solver)
    print_report(scores, solver)


if __name__ == "__main__":
    main()
