"""Score a bank reconciliation solver run against generated ground truth.

Re-generates the same problem (same seed/n/difficulty) and compares the
solver's output in the .db file against the known ground truth.

Usage:
    uv run python examples/score_bank_rec.py gen200.db  --n 200
    uv run python examples/score_bank_rec.py gen1000.db --n 1000
    uv run python examples/score_bank_rec.py gen1000.db --n 1000 --difficulty medium --seed 99
"""

from __future__ import annotations

import argparse
import sys

import duckdb

from examples.bank_rec_generator import generate, verify


def flatten_truth(gt: dict) -> dict:
    """Flatten ground truth into sets for comparison."""
    # All (bank_id, gl_id) pairs that should be matched
    matched_pairs: set[tuple[str, str]] = set()
    matched_by_type: dict[str, set[tuple[str, str]]] = {
        "1to1": set(),
        "batch": set(),
        "mismatch": set(),
    }
    for e in gt["matches_1to1"]:
        pair = (e["bank_id"], e["gl_id"])
        matched_pairs.add(pair)
        matched_by_type["1to1"].add(pair)
    for e in gt["matches_many_to_one"]:
        for gl_id in e["gl_ids"]:
            pair = (e["bank_id"], gl_id)
            matched_pairs.add(pair)
            matched_by_type["batch"].add(pair)
    for e in gt["matches_amount_mismatch"]:
        pair = (e["bank_id"], e["gl_id"])
        matched_pairs.add(pair)
        matched_by_type["mismatch"].add(pair)

    # Offsetting pairs (normalized: sorted tuple)
    offsetting: set[tuple[str, str]] = set()
    for e in gt["offsetting_gl_pairs"]:
        ids = sorted(e["gl_ids"])
        offsetting.add((ids[0], ids[1]))

    # Unmatched
    unmatched_bank = {e["bank_id"] for e in gt["unmatched_bank"]}
    unmatched_gl = {e["gl_id"] for e in gt["unmatched_gl"]}

    return {
        "matched_pairs": matched_pairs,
        "matched_by_type": matched_by_type,
        "offsetting": offsetting,
        "unmatched_bank": unmatched_bank,
        "unmatched_gl": unmatched_gl,
    }


def read_solver_output(db_path: str) -> dict:
    """Read solver results from the output database."""
    conn = duckdb.connect(db_path, read_only=True)

    # Matched pairs — auto-detect view name
    solver_pairs: set[tuple[str, str]] = set()
    matched_view = None
    for candidate_view in [
        "match_remaining_all_matched",
        "match_residual_all_matched",
        "match_hard_all_matched",
    ]:
        try:
            rows = conn.execute(
                f"SELECT bank_id, gl_id FROM {candidate_view}"
            ).fetchall()
            solver_pairs = {(r[0], r[1]) for r in rows}
            matched_view = candidate_view
            break
        except duckdb.CatalogException:
            continue
    if matched_view is None:
        print(
            "  WARNING: no matched-pairs view found (tried match_remaining_all_matched, match_residual_all_matched, match_hard_all_matched)"
        )
    else:
        print(f"  Using matched-pairs view: {matched_view}")

    # Offsetting pairs
    solver_offsetting: set[tuple[str, str]] = set()
    try:
        rows = conn.execute(
            "SELECT original_id, reversal_id FROM offsetting_pairs"
        ).fetchall()
        solver_offsetting = {(min(r[0], r[1]), max(r[0], r[1])) for r in rows}
    except duckdb.CatalogException:
        print("  WARNING: offsetting_pairs not found in DB")

    # Unmatched bank
    solver_unmatched_bank: set[str] = set()
    try:
        rows = conn.execute("SELECT id FROM report_unmatched_bank").fetchall()
        solver_unmatched_bank = {r[0] for r in rows}
    except duckdb.CatalogException:
        pass

    # Unmatched GL
    solver_unmatched_gl: set[str] = set()
    try:
        rows = conn.execute("SELECT id FROM report_unmatched_gl").fetchall()
        solver_unmatched_gl = {r[0] for r in rows}
    except duckdb.CatalogException:
        pass

    # Summary from report_summary
    summary = {}
    try:
        rows = conn.execute("SELECT * FROM report_summary").fetchall()
        cols = [d[0] for d in conn.description]
        if rows:
            summary = dict(zip(cols, rows[0]))
    except duckdb.CatalogException:
        pass

    # Node metadata
    node_meta = {}
    try:
        rows = conn.execute("SELECT node, meta_json FROM _node_meta").fetchall()
        import json

        for node, meta_json in rows:
            node_meta[node] = json.loads(meta_json)
    except duckdb.CatalogException:
        pass

    conn.close()
    return {
        "matched_pairs": solver_pairs,
        "offsetting": solver_offsetting,
        "unmatched_bank": solver_unmatched_bank,
        "unmatched_gl": solver_unmatched_gl,
        "summary": summary,
        "node_meta": node_meta,
    }


def score(truth: dict, solver: dict) -> dict:
    """Compare solver output against ground truth."""
    tp = truth["matched_pairs"] & solver["matched_pairs"]
    fp = solver["matched_pairs"] - truth["matched_pairs"]
    fn = truth["matched_pairs"] - solver["matched_pairs"]

    n_truth = len(truth["matched_pairs"])
    n_solver = len(solver["matched_pairs"])
    n_tp = len(tp)

    precision = 100.0 * n_tp / n_solver if n_solver else 0.0
    recall = 100.0 * n_tp / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # By type breakdown
    by_type = {}
    for typ, pairs in truth["matched_by_type"].items():
        found = pairs & solver["matched_pairs"]
        by_type[typ] = {
            "truth": len(pairs),
            "found": len(found),
            "missed": len(pairs) - len(found),
            "recall_pct": round(100.0 * len(found) / len(pairs), 1) if pairs else 100.0,
        }

    # Offsetting
    off_tp = truth["offsetting"] & solver["offsetting"]
    off_truth = len(truth["offsetting"])
    off_solver = len(solver["offsetting"])
    off_correct = len(off_tp)

    # Unmatched bank
    ub_tp = truth["unmatched_bank"] & solver["unmatched_bank"]
    ub_truth = len(truth["unmatched_bank"])
    ub_correct = len(ub_tp)

    # Unmatched GL
    ug_tp = truth["unmatched_gl"] & solver["unmatched_gl"]
    ug_truth = len(truth["unmatched_gl"])
    ug_correct = len(ug_tp)

    return {
        "pair_accuracy": {
            "truth_pairs": n_truth,
            "solver_pairs": n_solver,
            "true_positive": n_tp,
            "false_positive": len(fp),
            "false_negative": len(fn),
            "precision_pct": round(precision, 1),
            "recall_pct": round(recall, 1),
            "f1_pct": round(f1, 1),
        },
        "by_type": by_type,
        "offsetting": {
            "truth": off_truth,
            "solver": off_solver,
            "correct": off_correct,
            "recall_pct": round(100.0 * off_correct / off_truth, 1)
            if off_truth
            else 100.0,
            "precision_pct": round(100.0 * off_correct / off_solver, 1)
            if off_solver
            else 100.0,
        },
        "unmatched_bank": {
            "truth": ub_truth,
            "correct": ub_correct,
            "recall_pct": round(100.0 * ub_correct / ub_truth, 1)
            if ub_truth
            else 100.0,
        },
        "unmatched_gl": {
            "truth": ug_truth,
            "correct": ug_correct,
            "recall_pct": round(100.0 * ug_correct / ug_truth, 1)
            if ug_truth
            else 100.0,
        },
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
    }


def print_report(scores: dict, solver: dict, summary: dict):
    """Print a formatted scoring report."""
    pa = scores["pair_accuracy"]
    print()
    print("=" * 70)
    print("PAIR-LEVEL ACCURACY (bank_id, gl_id)")
    print("=" * 70)
    print(f"  Truth pairs:    {pa['truth_pairs']:>6d}")
    print(f"  Solver pairs:   {pa['solver_pairs']:>6d}")
    print(f"  True positive:  {pa['true_positive']:>6d}")
    print(f"  False positive: {pa['false_positive']:>6d}")
    print(f"  False negative: {pa['false_negative']:>6d}")
    print(f"  Precision:      {pa['precision_pct']:>6.1f}%")
    print(f"  Recall:         {pa['recall_pct']:>6.1f}%")
    print(f"  F1:             {pa['f1_pct']:>6.1f}%")

    print()
    print("BY MATCH TYPE")
    print("-" * 50)
    for typ, info in scores["by_type"].items():
        print(
            f"  {typ:<12s}  truth={info['truth']:>4d}  "
            f"found={info['found']:>4d}  "
            f"missed={info['missed']:>4d}  "
            f"recall={info['recall_pct']:.1f}%"
        )

    off = scores["offsetting"]
    print()
    print(f"OFFSETTING GL PAIRS")
    print(
        f"  Truth={off['truth']}  Solver={off['solver']}  "
        f"Correct={off['correct']}  "
        f"Recall={off['recall_pct']:.1f}%  Precision={off['precision_pct']:.1f}%"
    )

    ub = scores["unmatched_bank"]
    print(f"UNMATCHED BANK")
    print(
        f"  Truth={ub['truth']}  Correct={ub['correct']}  "
        f"Recall={ub['recall_pct']:.1f}%"
    )

    ug = scores["unmatched_gl"]
    print(f"UNMATCHED GL")
    print(
        f"  Truth={ug['truth']}  Correct={ug['correct']}  "
        f"Recall={ug['recall_pct']:.1f}%"
    )

    if scores["false_positives"]:
        print()
        print(f"FALSE POSITIVES (solver matched, truth disagrees) — showing up to 20:")
        for bank_id, gl_id in scores["false_positives"][:20]:
            print(f"  {bank_id} -> {gl_id}")

    if scores["false_negatives"]:
        print()
        print(f"FALSE NEGATIVES (truth matched, solver missed) — showing up to 20:")
        for bank_id, gl_id in scores["false_negatives"][:20]:
            print(f"  {bank_id} -> {gl_id}")

    # Node metadata
    meta = solver.get("node_meta", {})
    if meta:
        print()
        print("NODE METADATA")
        print("-" * 50)
        for node, m in sorted(meta.items()):
            elapsed = m.get("elapsed_s", "?")
            iters = m.get("iterations", "-")
            tokens = m.get("total_tokens", "-")
            status = m.get("validation_status", "?")
            print(
                f"  {node:<20s}  {elapsed:>6.1f}s  iters={iters}  "
                f"tokens={tokens}  validation={status}"
            )

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Score bank rec solver output")
    parser.add_argument("db_path", help="Path to solver output .db file")
    parser.add_argument("--n", type=int, default=200, help="n_bank used for generation")
    parser.add_argument("--difficulty", default="hard", help="Difficulty preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Regenerate ground truth with same parameters
    print(
        f"Regenerating ground truth: n={args.n} difficulty={args.difficulty} seed={args.seed}"
    )
    result = generate(n_bank=args.n, seed=args.seed, difficulty=args.difficulty)
    errors = verify(result)
    if errors:
        print(f"ERROR: Generator verification failed: {errors[:5]}")
        sys.exit(1)

    gt = result["ground_truth"]
    s = gt["summary"]
    print(
        f"  bank={s['bank_transactions']}  gl={s['gl_entries']}  "
        f"1:1={s['matches_1to1']}  batch={s['matches_many_to_one']}  "
        f"mismatch={s['matches_amount_mismatch']}  "
        f"offset={s['offsetting_gl_pairs']}  "
        f"ub={s['unmatched_bank']}  ug={s['unmatched_gl']}"
    )

    truth = flatten_truth(gt)

    # Read solver output
    print(f"Reading solver output from: {args.db_path}")
    solver = read_solver_output(args.db_path)
    print(
        f"  Solver matched {len(solver['matched_pairs'])} pairs, "
        f"{len(solver['offsetting'])} offsetting, "
        f"{len(solver['unmatched_bank'])} unmatched bank, "
        f"{len(solver['unmatched_gl'])} unmatched GL"
    )

    # Score
    scores = score(truth, solver)
    print_report(scores, solver, s)

    # Exit code: 0 if F1 > 50%, 1 otherwise
    f1 = scores["pair_accuracy"]["f1_pct"]
    sys.exit(0 if f1 > 50 else 1)


if __name__ == "__main__":
    main()
