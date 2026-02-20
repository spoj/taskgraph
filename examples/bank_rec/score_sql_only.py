import duckdb
from examples.bank_rec.score import score, flatten_truth, print_report
import json

def main():
    with open("dataset.json") as f:
        gt = json.load(f)["ground_truth"]

    import sys; db_path = sys.argv[1] if len(sys.argv) > 1 else "runs/sql_only.db"; conn = duckdb.connect(db_path, read_only=True)
    rows = conn.execute("SELECT bank_id, gl_id FROM report_matched").fetchall()
    solver_pairs = {(r[0], r[1]) for r in rows}

    rows = conn.execute("SELECT original_id, reversal_id FROM offsetting_pairs").fetchall()
    solver_offsetting = {(min(r[0], r[1]), max(r[0], r[1])) for r in rows}

    solver = {
        "matched_pairs": solver_pairs,
        "offsetting": solver_offsetting,
        "unmatched_bank": set(r[0] for r in conn.execute("SELECT id FROM report_unmatched_bank").fetchall()),
        "unmatched_gl": set(r[0] for r in conn.execute("SELECT id FROM report_unmatched_gl").fetchall()),
    }

    truth = flatten_truth(gt)
    scores = score(truth, solver)
    print_report(scores, solver, gt["summary"])

if __name__ == "__main__":
    main()
