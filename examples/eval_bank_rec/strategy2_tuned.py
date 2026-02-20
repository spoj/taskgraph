import json
import argparse
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from examples.score_bank_rec import score, print_report, flatten_truth
from examples.solve_bank_rec import solve, Match

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)

    # Convert date strings to date objects
    bank = data["bank_transactions"]
    for row in bank:
        row["date"] = datetime.strptime(row["date"], "%Y-%m-%d").date()
    
    gl = data["gl_entries"]
    for row in gl:
        row["date"] = datetime.strptime(row["date"], "%Y-%m-%d").date()

    # Reuse the hardcoded tuned heuristic solver
    solver = solve(bank, gl)
    
    gt = data["ground_truth"]
    s = gt["summary"]
    truth = flatten_truth(gt)
    scores = score(truth, solver)
    print_report(scores, solver, s)

if __name__ == "__main__":
    main()
