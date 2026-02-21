"""Strategy 1: Generic Deterministic Python Solver.

Simplest baseline approach — greedy matching using only:
1. Exact remittance reference lookup
2. Exact amount + customer name matching

No fuzzy matching, no partial payment handling, no discount detection,
no multi-invoice grouping. This establishes the performance floor.

Usage:
    python -m examples.cash_application.strategy1_generic [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def solve(
    invoices: list[dict],
    payments: list[dict],
    remittance_lines: list[dict],
) -> dict:
    """Greedy exact-match solver.

    Pass 1: Match by remittance invoice_ref (exact) + exact amount.
    Pass 2: Match remaining by exact amount + customer name.
    """

    # Build lookup structures
    inv_by_id: dict[str, dict] = {inv["invoice_id"]: inv for inv in invoices}
    inv_by_cust_amt: dict[tuple[str, float], list[dict]] = {}
    for inv in invoices:
        key = (inv["customer_id"], inv["amount"])
        inv_by_cust_amt.setdefault(key, []).append(inv)

    # Build remittance index: payment_id -> list of remittance lines
    rem_by_pmt: dict[str, list[dict]] = {}
    for rem in remittance_lines:
        rem_by_pmt.setdefault(rem["payment_id"], []).append(rem)

    # Build customer name -> customer_id mapping
    cust_name_to_id: dict[str, str] = {}
    for inv in invoices:
        cust_name_to_id[inv["customer_name"].lower()] = inv["customer_id"]

    applications: list[dict] = []
    matched_inv_ids: set[str] = set()
    matched_pmt_ids: set[str] = set()

    # ── Pass 1: Exact remittance reference match ──────────────────────────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        rems = rem_by_pmt.get(pmt["payment_id"], [])
        if not rems:
            continue

        pmt_matched = False
        for rem in rems:
            inv_ref = rem["invoice_ref"]
            if inv_ref in inv_by_id and inv_ref not in matched_inv_ids:
                inv = inv_by_id[inv_ref]
                # Require exact amount match (remittance amount == invoice amount)
                if abs(rem["amount"] - inv["amount"]) < 0.01:
                    applications.append(
                        {
                            "payment_id": pmt["payment_id"],
                            "invoice_id": inv["invoice_id"],
                            "applied_amount": inv["amount"],
                        }
                    )
                    matched_inv_ids.add(inv["invoice_id"])
                    pmt_matched = True

        if pmt_matched:
            matched_pmt_ids.add(pmt["payment_id"])

    # ── Pass 2: Exact amount + customer name match ────────────────────────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        payer = pmt["payer_name"].lower()
        cust_id = cust_name_to_id.get(payer)
        if not cust_id:
            continue

        key = (cust_id, pmt["amount"])
        candidates = inv_by_cust_amt.get(key, [])

        for inv in candidates:
            if inv["invoice_id"] not in matched_inv_ids:
                applications.append(
                    {
                        "payment_id": pmt["payment_id"],
                        "invoice_id": inv["invoice_id"],
                        "applied_amount": inv["amount"],
                    }
                )
                matched_inv_ids.add(inv["invoice_id"])
                matched_pmt_ids.add(pmt["payment_id"])
                break

    # Identify unmatched
    unmatched_payments = [
        pmt["payment_id"]
        for pmt in payments
        if pmt["payment_id"] not in matched_pmt_ids
    ]
    unapplied_invoices = [
        inv["invoice_id"]
        for inv in invoices
        if inv["invoice_id"] not in matched_inv_ids
    ]

    return {
        "applications": applications,
        "unmatched_payments": unmatched_payments,
        "unapplied_invoices": unapplied_invoices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy 1: Generic greedy matcher")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "problems" / "n100_seed42.json"),
        help="Path to dataset JSON",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text())

    # Convert date strings back to date objects
    for inv in data["invoices"]:
        inv["invoice_date"] = date.fromisoformat(inv["invoice_date"])
        inv["due_date"] = date.fromisoformat(inv["due_date"])
    for pmt in data["payments"]:
        pmt["payment_date"] = date.fromisoformat(pmt["payment_date"])

    result = solve(data["invoices"], data["payments"], data["remittance_lines"])

    # Score
    from examples.cash_application.score import flatten_truth, score, print_report

    truth = flatten_truth(data["ground_truth"])
    solver_output = {
        "pairs": {(a["payment_id"], a["invoice_id"]) for a in result["applications"]},
        "pair_amounts": {
            (a["payment_id"], a["invoice_id"]): a["applied_amount"]
            for a in result["applications"]
        },
        "unmatched_payments": set(result["unmatched_payments"]),
        "unapplied_invoices": set(result["unapplied_invoices"]),
    }

    scores = score(truth, solver_output)
    print_report(scores)
    sys.exit(0 if scores["pair_f1"] > 0.50 else 1)


if __name__ == "__main__":
    main()
