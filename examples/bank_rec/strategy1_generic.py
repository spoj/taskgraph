import json
import argparse
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from examples.bank_rec.score import score, print_report, flatten_truth

@dataclass
class Match:
    bank_id: str
    gl_id: str
    match_type: str = "generic"

def solve(bank_txns: list[dict], gl_entries: list[dict]) -> dict:
    bank_unmatched = {t["id"]: t for t in bank_txns}
    gl_unmatched = {t["id"]: t for t in gl_entries}

    matches = []

    for bid, b in list(bank_unmatched.items()):
        candidates = []
        for gid, g in list(gl_unmatched.items()):
            if abs(b["amount"] - g["amount"]) < 0.01:
                # generic date logic
                diff = abs((b["date_obj"] - g["date_obj"]).days)
                if diff <= 3: # 3 days max since it can clear on Monday
                    candidates.append((diff, gid, g))
        
        # Take the closest
        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_gid = candidates[0][1]
            matches.append(Match(bid, best_gid))
            del bank_unmatched[bid]
            del gl_unmatched[best_gid]
            
    # Format output
    return {
        "matched_pairs": {(m.bank_id, m.gl_id) for m in matches},
        "offsetting": set(),
        "unmatched_bank": set(bank_unmatched.keys()),
        "unmatched_gl": set(gl_unmatched.keys()),
        "matches": matches,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        data = json.load(f)

    # Hydrate date objects
    for row in data["bank_transactions"]:
        row["date_obj"] = datetime.strptime(row["date"], "%Y-%m-%d").date()
    for row in data["gl_entries"]:
        row["date_obj"] = datetime.strptime(row["date"], "%Y-%m-%d").date()

    solver = solve(data["bank_transactions"], data["gl_entries"])
    
    gt = data["ground_truth"]
    s = gt["summary"]
    truth = flatten_truth(gt)
    scores = score(truth, solver)
    print_report(scores, solver, s)

if __name__ == "__main__":
    main()
