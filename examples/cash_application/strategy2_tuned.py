"""Strategy 2: Fine-Tuned Deterministic Python Solver.

Multi-pass matching with fuzzy logic, partial payment detection,
discount tolerance, multi-invoice grouping, and entity resolution.
Designed to handle all the messiness modeled by the generator.

Matching passes (in order):
  1. Exact remittance reference + exact amount
  2. Fuzzy remittance reference (edit distance) + amount tolerance
  3. Multi-invoice grouping: group remittance lines by payment, match each
  4. Exact amount + customer name (direct match)
  5. Exact amount + fuzzy customer name (Jaro-Winkler)
  6. Discount detection (1-3% tolerance + customer match)
  7. Short-pay / deduction detection (3-15% tolerance + customer match)
  8. Partial payment detection (25-80% of invoice + customer match)
  9. No-remittance: amount + closest date within customer
  10. Fallback: exact amount + closest date (any customer)

Usage:
    python -m examples.cash_application.strategy2_tuned [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path


# ── String Similarity ─────────────────────────────────────────────────────────


def _jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity (0-1). Pure Python implementation."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
    ) / 3

    # Winkler prefix bonus
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


def _normalize_name(name: str) -> str:
    """Normalize a company name for matching."""
    s = name.upper().strip()
    # Remove common suffixes
    for suffix in [
        " INC",
        " INC.",
        " LLC",
        " LTD",
        " LTD.",
        " CORP",
        " CORP.",
        " CO",
        " CO.",
        " GROUP",
        " GRP",
        " ASSOCIATES",
        " ASSOC",
    ]:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    # Remove punctuation
    s = re.sub(r"[^A-Z0-9\s]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_inv_number(ref: str) -> str | None:
    """Extract numeric part from an invoice reference."""
    # Try INV-NNNNN, Invoice #NNNNN, INV NNNNN patterns
    m = re.search(r"(?:INV[- ]?|Invoice\s*#?\s*)(\d+)", ref, re.IGNORECASE)
    if m:
        return m.group(1).lstrip("0") or "0"
    return None


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ── Solver ────────────────────────────────────────────────────────────────────


def solve(
    invoices: list[dict],
    payments: list[dict],
    remittance_lines: list[dict],
) -> dict:
    """Multi-pass fine-tuned solver."""

    # ── Build indexes ─────────────────────────────────────────────────────
    inv_by_id: dict[str, dict] = {inv["invoice_id"]: inv for inv in invoices}
    inv_by_num: dict[str, list[str]] = {}
    for inv in invoices:
        num = inv["invoice_id"].split("-")[1].lstrip("0") or "0"
        inv_by_num.setdefault(num, []).append(inv["invoice_id"])

    # Customer name normalization index
    cust_id_by_name_norm: dict[str, str] = {}
    cust_names_by_id: dict[str, str] = {}
    for inv in invoices:
        norm = _normalize_name(inv["customer_name"])
        cust_id_by_name_norm[norm] = inv["customer_id"]
        cust_names_by_id[inv["customer_id"]] = inv["customer_name"]

    # Invoices by customer
    inv_by_cust: dict[str, list[dict]] = {}
    for inv in invoices:
        inv_by_cust.setdefault(inv["customer_id"], []).append(inv)

    # Remittance by payment
    rem_by_pmt: dict[str, list[dict]] = {}
    for rem in remittance_lines:
        rem_by_pmt.setdefault(rem["payment_id"], []).append(rem)

    applications: list[dict] = []
    matched_inv_ids: set[str] = set()
    matched_pmt_ids: set[str] = set()

    def _add_match(pmt_id: str, inv_id: str, amount: float) -> None:
        applications.append(
            {
                "payment_id": pmt_id,
                "invoice_id": inv_id,
                "applied_amount": amount,
            }
        )
        matched_inv_ids.add(inv_id)

    def _resolve_customer(payer_name: str) -> str | None:
        """Resolve payer name to customer_id using exact then fuzzy match."""
        norm = _normalize_name(payer_name)
        # Exact normalized match
        if norm in cust_id_by_name_norm:
            return cust_id_by_name_norm[norm]
        # Fuzzy match (Jaro-Winkler > 0.80)
        best_score = 0.0
        best_id = None
        for cand_norm, cand_id in cust_id_by_name_norm.items():
            jw = _jaro_winkler(norm, cand_norm)
            if jw > best_score:
                best_score = jw
                best_id = cand_id
        if best_score >= 0.80:
            return best_id
        # Prefix match (first word)
        norm_first = norm.split()[0] if norm else ""
        if len(norm_first) >= 4:
            for cand_norm, cand_id in cust_id_by_name_norm.items():
                cand_first = cand_norm.split()[0] if cand_norm else ""
                if norm_first == cand_first:
                    return cand_id
        return None

    def _find_invoice_by_ref(ref: str) -> str | None:
        """Find invoice ID by reference string (exact or fuzzy)."""
        # Exact match
        if ref in inv_by_id and ref not in matched_inv_ids:
            return ref
        # Extract number and match
        num = _extract_inv_number(ref)
        if num and num in inv_by_num:
            for inv_id in inv_by_num[num]:
                if inv_id not in matched_inv_ids:
                    return inv_id
        # Edit distance match (for transpositions)
        if ref.startswith("INV-"):
            ref_num = ref.split("-")[1].lstrip("0") or "0"
            for num_key, inv_ids in inv_by_num.items():
                if _edit_distance(ref_num, num_key) <= 1:
                    for inv_id in inv_ids:
                        if inv_id not in matched_inv_ids:
                            return inv_id
        return None

    # ── Pass 1: Exact remittance ref + exact amount ───────────────────────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue
        rems = rem_by_pmt.get(pmt["payment_id"], [])
        all_matched = True
        matched_this_pmt: list[tuple[str, str, float]] = []

        for rem in rems:
            inv_id = _find_invoice_by_ref(rem["invoice_ref"])
            if inv_id and abs(rem["amount"] - inv_by_id[inv_id]["amount"]) < 0.01:
                matched_this_pmt.append((pmt["payment_id"], inv_id, rem["amount"]))
            else:
                all_matched = False

        if matched_this_pmt and all_matched:
            for pmt_id, inv_id, amt in matched_this_pmt:
                _add_match(pmt_id, inv_id, amt)
            matched_pmt_ids.add(pmt["payment_id"])

    # ── Pass 2: Fuzzy remittance ref + amount tolerance ───────────────────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue
        rems = rem_by_pmt.get(pmt["payment_id"], [])
        if not rems:
            continue

        cust_id = _resolve_customer(pmt["payer_name"])
        matched_this_pmt = []

        for rem in rems:
            inv_id = _find_invoice_by_ref(rem["invoice_ref"])
            if inv_id:
                inv = inv_by_id[inv_id]
                # Allow amount tolerance for discounts/deductions (up to 15%)
                if abs(rem["amount"] - inv["amount"]) < 0.01:
                    matched_this_pmt.append((pmt["payment_id"], inv_id, rem["amount"]))
                elif inv["amount"] > 0 and 0.85 <= rem["amount"] / inv["amount"] <= 1.0:
                    matched_this_pmt.append((pmt["payment_id"], inv_id, rem["amount"]))
                elif inv["amount"] > 0 and 0.25 <= rem["amount"] / inv["amount"] < 0.85:
                    # Partial payment
                    matched_this_pmt.append((pmt["payment_id"], inv_id, rem["amount"]))

        if matched_this_pmt:
            for pmt_id, inv_id, amt in matched_this_pmt:
                _add_match(pmt_id, inv_id, amt)
            matched_pmt_ids.add(pmt["payment_id"])

    # ── Pass 3: Multi-invoice payment (remittance groups) ─────────────────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue
        rems = rem_by_pmt.get(pmt["payment_id"], [])
        if len(rems) < 2:
            continue

        cust_id = _resolve_customer(pmt["payer_name"])
        if not cust_id:
            continue

        cust_invs = [
            inv
            for inv in inv_by_cust.get(cust_id, [])
            if inv["invoice_id"] not in matched_inv_ids
        ]

        matched_this_pmt = []
        for rem in rems:
            # Try to find matching invoice by amount within this customer
            best_inv = None
            best_diff = float("inf")
            for inv in cust_invs:
                if inv["invoice_id"] in {m[1] for m in matched_this_pmt}:
                    continue
                diff = abs(rem["amount"] - inv["amount"])
                if diff < best_diff and (
                    diff < 0.01
                    or (
                        inv["amount"] > 0
                        and 0.25 <= rem["amount"] / inv["amount"] <= 1.05
                    )
                ):
                    best_diff = diff
                    best_inv = inv

            if best_inv:
                matched_this_pmt.append(
                    (pmt["payment_id"], best_inv["invoice_id"], rem["amount"])
                )

        if len(matched_this_pmt) >= 2:
            for pmt_id, inv_id, amt in matched_this_pmt:
                _add_match(pmt_id, inv_id, amt)
            matched_pmt_ids.add(pmt["payment_id"])

    # ── Pass 4: Exact amount + exact customer name ────────────────────────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        cust_id = _resolve_customer(pmt["payer_name"])
        if not cust_id:
            continue

        cust_invs = [
            inv
            for inv in inv_by_cust.get(cust_id, [])
            if inv["invoice_id"] not in matched_inv_ids
        ]

        for inv in cust_invs:
            if abs(pmt["amount"] - inv["amount"]) < 0.01:
                _add_match(pmt["payment_id"], inv["invoice_id"], inv["amount"])
                matched_pmt_ids.add(pmt["payment_id"])
                break

    # ── Pass 5: Discount detection (97-99.5% of invoice, within customer) ─
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        cust_id = _resolve_customer(pmt["payer_name"])
        if not cust_id:
            continue

        cust_invs = [
            inv
            for inv in inv_by_cust.get(cust_id, [])
            if inv["invoice_id"] not in matched_inv_ids and inv["amount"] > 0
        ]

        for inv in cust_invs:
            ratio = pmt["amount"] / inv["amount"]
            if 0.97 <= ratio <= 0.995:
                _add_match(pmt["payment_id"], inv["invoice_id"], pmt["amount"])
                matched_pmt_ids.add(pmt["payment_id"])
                break

    # ── Pass 6: Short-pay deduction (85-97% of invoice, within customer) ──
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        cust_id = _resolve_customer(pmt["payer_name"])
        if not cust_id:
            continue

        cust_invs = [
            inv
            for inv in inv_by_cust.get(cust_id, [])
            if inv["invoice_id"] not in matched_inv_ids and inv["amount"] > 0
        ]

        for inv in cust_invs:
            ratio = pmt["amount"] / inv["amount"]
            if 0.85 <= ratio < 0.97:
                _add_match(pmt["payment_id"], inv["invoice_id"], pmt["amount"])
                matched_pmt_ids.add(pmt["payment_id"])
                break

    # ── Pass 7: Overpayment detection (100-115% of invoice) ───────────────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        cust_id = _resolve_customer(pmt["payer_name"])
        if not cust_id:
            continue

        cust_invs = [
            inv
            for inv in inv_by_cust.get(cust_id, [])
            if inv["invoice_id"] not in matched_inv_ids and inv["amount"] > 0
        ]

        for inv in cust_invs:
            ratio = pmt["amount"] / inv["amount"]
            if 1.0 < ratio <= 1.15:
                _add_match(pmt["payment_id"], inv["invoice_id"], inv["amount"])
                matched_pmt_ids.add(pmt["payment_id"])
                break

    # ── Pass 8: Credit memo detection ─────────────────────────────────────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        cust_id = _resolve_customer(pmt["payer_name"])
        if not cust_id:
            continue

        cust_invs = [
            inv
            for inv in inv_by_cust.get(cust_id, [])
            if inv["invoice_id"] not in matched_inv_ids
        ]

        # Look for a positive invoice + negative credit that sum to payment amount
        pos_invs = [inv for inv in cust_invs if inv["amount"] > 0]
        neg_invs = [inv for inv in cust_invs if inv["amount"] < 0]

        for pos in pos_invs:
            for neg in neg_invs:
                expected_pmt = pos["amount"] + neg["amount"]
                if abs(pmt["amount"] - expected_pmt) < 0.01:
                    _add_match(pmt["payment_id"], pos["invoice_id"], pos["amount"])
                    _add_match(pmt["payment_id"], neg["invoice_id"], neg["amount"])
                    matched_pmt_ids.add(pmt["payment_id"])
                    break
            if pmt["payment_id"] in matched_pmt_ids:
                break

    # ── Pass 9: Partial payment (25-80% of invoice, within customer) ──────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        cust_id = _resolve_customer(pmt["payer_name"])
        if not cust_id:
            continue

        # Check if remittance mentions partial
        rems = rem_by_pmt.get(pmt["payment_id"], [])
        has_partial_hint = any(
            "partial" in (rem.get("memo", "") or "").lower()
            or "installment" in (rem.get("memo", "") or "").lower()
            or "progress" in (rem.get("memo", "") or "").lower()
            for rem in rems
        )

        if not has_partial_hint and not rems:
            # No hint and no remittance — skip partial detection here
            continue

        cust_invs = [
            inv
            for inv in inv_by_cust.get(cust_id, [])
            if inv["invoice_id"] not in matched_inv_ids and inv["amount"] > 0
        ]

        for inv in cust_invs:
            ratio = pmt["amount"] / inv["amount"]
            if 0.25 <= ratio < 0.85:
                _add_match(pmt["payment_id"], inv["invoice_id"], pmt["amount"])
                matched_pmt_ids.add(pmt["payment_id"])
                break

    # ── Pass 10: Exact amount + closest date (any customer, fallback) ─────
    for pmt in payments:
        if pmt["payment_id"] in matched_pmt_ids:
            continue

        best_inv = None
        best_gap = 999
        pmt_date = pmt["payment_date"]

        for inv in invoices:
            if inv["invoice_id"] in matched_inv_ids:
                continue
            if abs(pmt["amount"] - inv["amount"]) < 0.01:
                gap = abs((pmt_date - inv["invoice_date"]).days)
                if gap < best_gap and gap <= 90:
                    best_gap = gap
                    best_inv = inv

        if best_inv:
            _add_match(pmt["payment_id"], best_inv["invoice_id"], best_inv["amount"])
            matched_pmt_ids.add(pmt["payment_id"])

    # ── Identify unmatched / unapplied ────────────────────────────────────
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
    parser = argparse.ArgumentParser(
        description="Strategy 2: Fine-tuned multi-pass matcher"
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "problems" / "n100_seed42.json"),
        help="Path to dataset JSON",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text())

    # Convert date strings
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
