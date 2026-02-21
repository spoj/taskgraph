"""Scoring framework for cash application strategies.

Evaluates solver output against ground truth. Supports two modes:
1. Direct dict input (for Python strategies that return results)
2. DuckDB database file (for taskgraph strategies)

Metrics:
- Pair Match F1: precision/recall/F1 on (payment_id, invoice_id) pairs
- Amount Accuracy: fraction of matched pairs with correct applied_amount
- Per-type breakdown: F1 by match_type (simple, multi, partial, discount, etc.)
- Unmatched payment recall: correctly identified unmatched payments
- Unapplied invoice recall: correctly identified unapplied invoices

Usage:
    # Score a taskgraph output DB against a problem set:
    python -m examples.cash_application.score output.db --dataset problems/n100_seed42.json

    # Or import and call directly:
    from examples.cash_application.score import score, print_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Ground Truth Flattening ───────────────────────────────────────────────────


def flatten_truth(gt: dict) -> dict:
    """Flatten ground truth into scoring-friendly sets.

    Returns:
        {
            "pairs": set of (payment_id, invoice_id),
            "pair_amounts": dict of (payment_id, invoice_id) -> applied_amount,
            "pairs_by_type": dict of match_type -> set of (payment_id, invoice_id),
            "unmatched_payments": set of payment_id,
            "unapplied_invoices": set of invoice_id,
        }
    """
    pairs = set()
    pair_amounts: dict[tuple[str, str], float] = {}
    pairs_by_type: dict[str, set[tuple[str, str]]] = {}

    for app in gt["applications"]:
        key = (app["payment_id"], app["invoice_id"])
        pairs.add(key)
        pair_amounts[key] = app["applied_amount"]
        mt = app["match_type"]
        pairs_by_type.setdefault(mt, set()).add(key)

    return {
        "pairs": pairs,
        "pair_amounts": pair_amounts,
        "pairs_by_type": pairs_by_type,
        "unmatched_payments": set(gt["unmatched_payments"]),
        "unapplied_invoices": set(gt["unapplied_invoices"]),
    }


# ── Read Solver Output from DuckDB ───────────────────────────────────────────


def read_solver_output(db_path: str) -> dict:
    """Read solver output from a taskgraph workspace .db file.

    Expected views/tables:
    - match_results or report_results: payment_id, invoice_id, applied_amount
    - report_unmatched_payments: payment_id (optional)
    - report_unapplied_invoices: invoice_id (optional)
    - _node_meta: node metadata (optional)
    """
    import duckdb

    conn = duckdb.connect(db_path, read_only=True)

    # Auto-detect the main results view
    results_view = None
    candidates = [
        "match_results",
        "report_results",
        "match_all_applications",
        "report_all_applications",
        "match_residual_results",
        "match_residual_all_applications",
    ]
    existing = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'",
        ).fetchall()
    }

    for name in candidates:
        if name in existing:
            results_view = name
            break

    if results_view is None:
        conn.close()
        print(f"ERROR: No results view found in {db_path}")
        print(f"  Looked for: {candidates}")
        print(
            f"  Found tables/views: {sorted(existing - {t for t in existing if t.startswith('_')})}"
        )
        sys.exit(1)

    # Read main results
    rows = conn.execute(
        f"SELECT payment_id, invoice_id, applied_amount FROM {results_view}",
    ).fetchall()

    pairs = set()
    pair_amounts: dict[tuple[str, str], float] = {}
    for pmt_id, inv_id, amt in rows:
        key = (str(pmt_id), str(inv_id))
        pairs.add(key)
        pair_amounts[key] = float(amt) if amt is not None else 0.0

    # Read unmatched payments
    unmatched_payments: set[str] = set()
    for name in ["report_unmatched_payments", "match_unmatched_payments"]:
        if name in existing:
            rows = conn.execute(f"SELECT payment_id FROM {name}").fetchall()
            unmatched_payments = {str(r[0]) for r in rows}
            break

    # Read unapplied invoices
    unapplied_invoices: set[str] = set()
    for name in ["report_unapplied_invoices", "match_unapplied_invoices"]:
        if name in existing:
            rows = conn.execute(f"SELECT invoice_id FROM {name}").fetchall()
            unapplied_invoices = {str(r[0]) for r in rows}
            break

    # Read node metadata
    node_meta: dict[str, dict] = {}
    if "_node_meta" in existing:
        rows = conn.execute("SELECT node, meta_json FROM _node_meta").fetchall()
        for node, meta_json in rows:
            try:
                node_meta[node] = json.loads(meta_json)
            except (json.JSONDecodeError, TypeError):
                pass

    conn.close()

    return {
        "pairs": pairs,
        "pair_amounts": pair_amounts,
        "unmatched_payments": unmatched_payments,
        "unapplied_invoices": unapplied_invoices,
        "node_meta": node_meta,
        "results_view": results_view,
    }


# ── Scoring ───────────────────────────────────────────────────────────────────


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score(truth: dict, solver: dict) -> dict:
    """Compute scoring metrics.

    Args:
        truth: output of flatten_truth()
        solver: dict with "pairs", "pair_amounts", optionally "unmatched_payments", "unapplied_invoices"

    Returns:
        dict of scoring metrics
    """
    t_pairs = truth["pairs"]
    s_pairs = solver["pairs"]

    correct_pairs = t_pairs & s_pairs
    false_positives = s_pairs - t_pairs
    false_negatives = t_pairs - s_pairs

    precision = len(correct_pairs) / len(s_pairs) if s_pairs else 0.0
    recall = len(correct_pairs) / len(t_pairs) if t_pairs else 0.0
    f1 = _f1(precision, recall)

    # Amount accuracy (for correctly matched pairs)
    amount_correct = 0
    amount_total = len(correct_pairs)
    amount_errors: list[dict] = []
    for pair in correct_pairs:
        t_amt = truth["pair_amounts"].get(pair, 0)
        s_amt = solver["pair_amounts"].get(pair, 0)
        if abs(t_amt - s_amt) < 0.01:
            amount_correct += 1
        else:
            amount_errors.append(
                {
                    "pair": pair,
                    "expected": t_amt,
                    "actual": s_amt,
                    "diff": round(s_amt - t_amt, 2),
                }
            )

    amount_accuracy = amount_correct / amount_total if amount_total > 0 else 0.0

    # Per-type breakdown
    by_type: dict[str, dict] = {}
    for match_type, type_pairs in truth.get("pairs_by_type", {}).items():
        type_correct = type_pairs & s_pairs
        type_fn = type_pairs - s_pairs
        type_p = len(type_correct) / len(s_pairs) if s_pairs else 0.0
        type_r = len(type_correct) / len(type_pairs) if type_pairs else 0.0
        by_type[match_type] = {
            "total": len(type_pairs),
            "matched": len(type_correct),
            "missed": len(type_fn),
            "recall": type_r,
        }

    # Unmatched payment recall
    t_unmatched = truth.get("unmatched_payments", set())
    s_unmatched = solver.get("unmatched_payments", set())
    unmatched_correct = t_unmatched & s_unmatched
    unmatched_recall = len(unmatched_correct) / len(t_unmatched) if t_unmatched else 1.0

    # Unapplied invoice recall
    t_unapplied = truth.get("unapplied_invoices", set())
    s_unapplied = solver.get("unapplied_invoices", set())
    unapplied_correct = t_unapplied & s_unapplied
    unapplied_recall = len(unapplied_correct) / len(t_unapplied) if t_unapplied else 1.0

    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "correct_pairs": len(correct_pairs),
        "solver_pairs": len(s_pairs),
        "truth_pairs": len(t_pairs),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "amount_accuracy": amount_accuracy,
        "amount_correct": amount_correct,
        "amount_total": amount_total,
        "amount_errors": sorted(
            amount_errors, key=lambda x: abs(x["diff"]), reverse=True
        )[:20],
        "by_type": by_type,
        "unmatched_payment_recall": unmatched_recall,
        "unmatched_payments_found": len(unmatched_correct),
        "unmatched_payments_total": len(t_unmatched),
        "unapplied_invoice_recall": unapplied_recall,
        "unapplied_invoices_found": len(unapplied_correct),
        "unapplied_invoices_total": len(t_unapplied),
    }


# ── Report Printing ───────────────────────────────────────────────────────────


def print_report(scores: dict, solver: dict | None = None) -> None:
    """Print a formatted scoring report."""
    print("\n" + "=" * 70)
    print("CASH APPLICATION SCORING REPORT")
    print("=" * 70)

    # Pair matching
    print(f"\n{'Pair Matching':30s}")
    print(
        f"  {'Precision':20s} {scores['pair_precision']:6.1%}  "
        f"({scores['correct_pairs']}/{scores['solver_pairs']})"
    )
    print(
        f"  {'Recall':20s} {scores['pair_recall']:6.1%}  "
        f"({scores['correct_pairs']}/{scores['truth_pairs']})"
    )
    print(f"  {'F1':20s} {scores['pair_f1']:6.1%}")

    # Amount accuracy
    print(f"\n{'Amount Accuracy':30s}")
    print(
        f"  {'Exact (within $0.01)':20s} {scores['amount_accuracy']:6.1%}  "
        f"({scores['amount_correct']}/{scores['amount_total']})"
    )

    # Per-type breakdown
    if scores["by_type"]:
        print(f"\n{'By Match Type':30s} {'Total':>6s} {'Found':>6s} {'Recall':>8s}")
        print("  " + "-" * 50)
        for mt, stats in sorted(scores["by_type"].items()):
            print(
                f"  {mt:28s} {stats['total']:6d} {stats['matched']:6d} "
                f"{stats['recall']:7.1%}"
            )

    # Unmatched / unapplied
    print(f"\n{'Unmatched Payments':30s}")
    print(
        f"  {'Recall':20s} {scores['unmatched_payment_recall']:6.1%}  "
        f"({scores['unmatched_payments_found']}/{scores['unmatched_payments_total']})"
    )

    print(f"\n{'Unapplied Invoices':30s}")
    print(
        f"  {'Recall':20s} {scores['unapplied_invoice_recall']:6.1%}  "
        f"({scores['unapplied_invoices_found']}/{scores['unapplied_invoices_total']})"
    )

    # False positives / negatives
    if scores["false_positives"]:
        n = len(scores["false_positives"])
        print(f"\nFalse Positives ({n} total, showing up to 10):")
        for fp in scores["false_positives"][:10]:
            print(f"  {fp[0]} -> {fp[1]}")

    if scores["false_negatives"]:
        n = len(scores["false_negatives"])
        print(f"\nFalse Negatives ({n} total, showing up to 10):")
        for fn_ in scores["false_negatives"][:10]:
            print(f"  {fn_[0]} -> {fn_[1]}")

    # Amount errors
    if scores["amount_errors"]:
        n = len(scores["amount_errors"])
        print(f"\nAmount Mismatches ({n} shown, largest first):")
        for err in scores["amount_errors"][:10]:
            p, i = err["pair"]
            print(
                f"  {p} -> {i}: expected ${err['expected']:.2f}, "
                f"got ${err['actual']:.2f} (diff ${err['diff']:+.2f})"
            )

    # Node metadata (for taskgraph strategies)
    if solver and solver.get("node_meta"):
        print(f"\n{'Node Metadata':30s}")
        for node, meta in sorted(solver["node_meta"].items()):
            elapsed = meta.get("elapsed_s", "?")
            iters = meta.get("iterations", "?")
            tokens = meta.get("total_tokens", "?")
            status = meta.get("validation_status", "?")
            print(
                f"  {node:25s} elapsed={elapsed}s  iters={iters}  "
                f"tokens={tokens}  status={status}"
            )

    print("\n" + "=" * 70)
    print(f"OVERALL F1: {scores['pair_f1']:.1%}")
    print("=" * 70)


# ── Ground Truth Loading ──────────────────────────────────────────────────────


def load_ground_truth(dataset_path: str) -> dict:
    """Load ground truth from a dataset JSON file."""
    data = json.loads(Path(dataset_path).read_text())
    return flatten_truth(data["ground_truth"])


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Score cash application solver output")
    parser.add_argument("db_path", help="Path to solver output .db file")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset JSON file with ground truth",
    )
    args = parser.parse_args()

    truth = load_ground_truth(args.dataset)
    solver = read_solver_output(args.db_path)
    scores_result = score(truth, solver)
    print_report(scores_result, solver)

    sys.exit(0 if scores_result["pair_f1"] > 0.50 else 1)


if __name__ == "__main__":
    main()
