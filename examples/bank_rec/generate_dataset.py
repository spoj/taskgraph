import json
import argparse
from examples.bank_rec.generator import generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--out", default="dataset.json")
    args = parser.parse_args()

    res = generate(n_bank=args.n, seed=args.seed, difficulty=args.difficulty)

    # Format dates as strings for JSON serialization
    dataset = {
        "bank_transactions": [
            {"id": t["id"], "date": str(t["date"]), "description": t["description"], "amount": t["amount"]}
            for t in res["bank_transactions"]
        ],
        "gl_entries": [
            {
                "id": t["id"], 
                "date": str(t["date"]), 
                "description": t["description"], 
                "amount": t["amount"], 
                "ref": t["ref"], 
                "entry_type": t.get("entry_type", "")
            }
            for t in res["gl_entries"]
        ],
        "ground_truth": res["ground_truth"]
    }

    with open(args.out, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset generated: {args.out}")

if __name__ == "__main__":
    main()
