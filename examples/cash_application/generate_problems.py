"""Generate cash application problem sets at standard sizes.

Usage:
    python -m examples.cash_application.generate_problems [--seed SEED] [--difficulty DIFFICULTY] [--out-dir DIR]

Generates JSON problem sets for sizes: 10, 30, 100, 300, 1000.
Default seed: 42, default difficulty: hard.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from examples.cash_application.generator import generate, verify

SIZES = [10, 30, 100, 300, 1000]


def _json_serializer(obj: object) -> str:
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cash application problem sets"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="hard",
        help="Difficulty level (default: hard)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: examples/cash_application/problems)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "problems"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in SIZES:
        result = generate(n, seed=args.seed, difficulty=args.difficulty)
        errors = verify(result)
        if errors:
            print(f"FAIL n={n}: {len(errors)} verification errors")
            for e in errors[:5]:
                print(f"  - {e}")
            continue

        out_path = out_dir / f"n{n}_seed{args.seed}.json"
        out_path.write_text(
            json.dumps(result, indent=2, default=_json_serializer),
        )
        m = result["metadata"]
        print(
            f"n={n:>4d}  invoices={m['n_invoices']:>4d}  "
            f"payments={m['n_payments']:>4d}  "
            f"remittance={m['n_remittance_lines']:>4d}  "
            f"applications={m['n_applications']:>4d}  "
            f"-> {out_path}"
        )

    print(f"\nDone. Files written to {out_dir}/")


if __name__ == "__main__":
    main()
