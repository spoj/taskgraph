"""Fully deterministic bank reconciliation solver — no LLM.

Rule cascade + subset-sum solver. Replicates the entity normalization logic
from bank_rec_v4.py, then matches via:

  1. Check number + exact amount
  2. Entity-prefix + exact amount (1:1)
  3. Generic DEPOSIT/NSF pairing (linked by GL entity)
  4. Offsetting GL pairs (equal-and-opposite, entity-confirmed)
  5. Batch deposits — full entity group sum
  6. Batch deposits — subset-sum solver for partial groups
  7. Exact amount + date proximity (unique amounts only)
  8. Entity-prefix + tolerance (≤ $100)

Usage:
    uv run python examples/solve_bank_rec.py --n 1000 --seed 42 --difficulty hard
    uv run python examples/solve_bank_rec.py --n 200 --seed 42 --difficulty easy
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations

from examples.bank_rec_generator import generate, verify
from examples.score_bank_rec import flatten_truth, print_report, score


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Txn:
    id: str
    date: object  # datetime.date
    description: str
    amount: float
    ref: str = ""
    entry_type: str = ""
    # Derived
    desc_clean: str = ""
    check_no: int | None = None
    entity_norm: str = ""


@dataclass
class Match:
    bank_id: str
    gl_id: str
    bank_amount: float
    gl_amount: float
    match_type: str
    note: str = ""


@dataclass
class OffsetPair:
    original_id: str
    reversal_id: str


# ---------------------------------------------------------------------------
# Entity normalization (mirrors FEATURES_SQL from bank_rec_v4.py)
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = re.compile(
    r"(^|\s)(INCORPORATED|INC|LLC|LTD|LIMITED|CORP|CORPORATION|CO|COMPANY|LP|LLP|PLC)(\s|$)"
)
_MULTI_SPACE = re.compile(r"\s+")

_BANK_PREFIX = re.compile(r"^(ACH\s+(CREDIT|DEBIT|CR|DR)|WIRE\s+(TRF\s+)?(IN|OUT))\s+")
_BANK_REFNUMS = re.compile(r"\d{6,}\s*")
_BANK_TRAILING_FRAG = re.compile(r"\s+[A-Z]{1,3}$")

_CHECK_BANK = re.compile(r"CHECK\s*#?\s*(\d+)", re.IGNORECASE)
_CHECK_GL = re.compile(r"(?:CHK|CHECK)[-#]?(\d+)", re.IGNORECASE)

_VOID_ENTITY = re.compile(r"(?i)VOID\s*-\s*.*?\s+to\s+(.+?)\s*\(")


def _norm_bank_entity(desc: str) -> str:
    """Extract and normalize entity from a bank description."""
    s = desc.upper()
    s = _BANK_PREFIX.sub("", s)
    s = _BANK_REFNUMS.sub(" ", s)
    s = _LEGAL_SUFFIXES.sub(" ", s)
    s = _BANK_TRAILING_FRAG.sub("", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def _norm_gl_entity(desc: str) -> str:
    """Extract and normalize entity from a GL description."""
    # Take text before " - " or " / "
    for sep in (" - ", " / "):
        idx = desc.find(sep)
        if idx > 0:
            desc = desc[:idx].strip()
            break
    s = desc.upper()
    s = _LEGAL_SUFFIXES.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def _extract_check_bank(desc: str) -> int | None:
    m = _CHECK_BANK.search(desc)
    return int(m.group(1)) if m else None


def _extract_check_gl(ref: str) -> int | None:
    m = _CHECK_GL.search(ref)
    return int(m.group(1)) if m else None


def _extract_void_entity(desc: str) -> str | None:
    m = _VOID_ENTITY.search(desc)
    if not m:
        return None
    s = m.group(1).upper()
    s = _LEGAL_SUFFIXES.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def _jaro_winkler(s1: str, s2: str) -> float:
    """Simple Jaro-Winkler similarity."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

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

    # Winkler bonus
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def prepare_bank(raw: list[dict]) -> list[Txn]:
    out = []
    for r in raw:
        t = Txn(
            id=r["id"],
            date=r["date"],
            description=r["description"],
            amount=r["amount"],
        )
        t.desc_clean = re.sub(r"[^A-Z0-9 ]", " ", r["description"].upper())
        t.check_no = _extract_check_bank(r["description"])
        t.entity_norm = _norm_bank_entity(r["description"])
        out.append(t)
    return out


def prepare_gl(raw: list[dict]) -> list[Txn]:
    out = []
    for r in raw:
        t = Txn(
            id=r["id"],
            date=r["date"],
            description=r["description"],
            amount=r["amount"],
            ref=r.get("ref", ""),
            entry_type=r.get("entry_type", ""),
        )
        t.desc_clean = re.sub(r"[^A-Z0-9 ]", " ", r["description"].upper())
        t.check_no = _extract_check_gl(r.get("ref", ""))
        t.entity_norm = _norm_gl_entity(r["description"])
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------


def _date_diff(d1, d2) -> int:
    return abs((d1 - d2).days)


def _starts_with(gl_entity: str, bank_entity: str) -> bool:
    return gl_entity.startswith(bank_entity) and len(bank_entity) >= 4


def solve(bank_raw: list[dict], gl_raw: list[dict]) -> dict:
    """Run the deterministic solver. Returns matched pairs, offsetting, unmatched."""

    bank = prepare_bank(bank_raw)
    gl = prepare_gl(gl_raw)

    bank_by_id = {b.id: b for b in bank}
    gl_by_id = {g.id: g for g in gl}

    matched: list[Match] = []
    matched_bank: set[str] = set()
    matched_gl: set[str] = set()
    offset_pairs: list[OffsetPair] = []
    offset_gl: set[str] = set()

    def _add_match(bank_id: str, gl_id: str, mtype: str, note: str = ""):
        b = bank_by_id[bank_id]
        g = gl_by_id[gl_id]
        matched.append(Match(bank_id, gl_id, b.amount, g.amount, mtype, note))
        matched_bank.add(bank_id)
        matched_gl.add(gl_id)

    def _add_batch(bank_id: str, gl_ids: list[str], note: str = ""):
        b = bank_by_id[bank_id]
        matched_bank.add(bank_id)
        for gid in gl_ids:
            g = gl_by_id[gid]
            matched.append(Match(bank_id, gid, b.amount, g.amount, "batch", note))
            matched_gl.add(gid)

    # -----------------------------------------------------------------------
    # Pass 1: Check number + exact amount
    # -----------------------------------------------------------------------

    gl_by_check: dict[int, list[Txn]] = {}
    for g in gl:
        if g.check_no is not None:
            gl_by_check.setdefault(g.check_no, []).append(g)

    for b in bank:
        if b.check_no is None:
            continue
        candidates = gl_by_check.get(b.check_no, [])
        for g in candidates:
            if g.id not in matched_gl and b.amount == g.amount:
                _add_match(b.id, g.id, "check_number", f"Check #{b.check_no}")
                break

    # -----------------------------------------------------------------------
    # Pass 2: Entity-prefix + exact amount (1:1, conservative)
    # -----------------------------------------------------------------------

    # Build candidates, pick unambiguous matches
    ent_candidates: list[
        tuple[str, str, int, int]
    ] = []  # (bank_id, gl_id, date_gap, ent_len)
    for b in bank:
        if b.id in matched_bank:
            continue
        if len(b.entity_norm) < 4:
            continue
        for g in gl:
            if g.id in matched_gl:
                continue
            if b.amount != g.amount:
                continue
            if _date_diff(b.date, g.date) > 10:
                continue
            if not _starts_with(g.entity_norm, b.entity_norm):
                continue
            ent_candidates.append(
                (b.id, g.id, _date_diff(b.date, g.date), len(b.entity_norm))
            )

    # Count candidates per bank and per GL
    bank_cand_count: dict[str, int] = {}
    gl_cand_count: dict[str, int] = {}
    for bid, gid, _, _ in ent_candidates:
        bank_cand_count[bid] = bank_cand_count.get(bid, 0) + 1
        gl_cand_count[gid] = gl_cand_count.get(gid, 0) + 1

    # Rank by (date_gap ASC, entity_len DESC)
    ent_candidates.sort(key=lambda x: (x[2], -x[3]))

    # Greedy assignment: best candidate first, skip conflicts
    bank_best: dict[str, tuple[str, int, int]] = {}
    gl_best: dict[str, tuple[str, int, int]] = {}
    for bid, gid, dg, el in ent_candidates:
        if bid not in bank_best or (dg, -el) < (bank_best[bid][1], -bank_best[bid][2]):
            bank_best[bid] = (gid, dg, el)
        if gid not in gl_best or (dg, -el) < (gl_best[gid][1], -gl_best[gid][2]):
            gl_best[gid] = (bid, dg, el)

    for bid, (gid, dg, el) in sorted(bank_best.items()):
        if bid in matched_bank or gid in matched_gl:
            continue
        # Check this is the best for both sides, OR one side has only 1 candidate
        gl_top = gl_best.get(gid)
        if gl_top and gl_top[0] == bid:
            _add_match(bid, gid, "exact_1to1")
        elif bank_cand_count.get(bid, 0) == 1 or gl_cand_count.get(gid, 0) == 1:
            _add_match(bid, gid, "exact_1to1")

    # -----------------------------------------------------------------------
    # Pass 3: Generic DEPOSIT / NSF pairing
    # -----------------------------------------------------------------------
    # Bank "DEPOSIT" (positive) and "RETURNED ITEM - NSF" (negative) have no
    # entity signal.  On the GL side, the payment and NSF entries share the
    # same entity name.  Pair them: DEPOSIT(+X) <-> GL payment(+X, entity=E)
    # and NSF(-X) <-> GL NSF return(-X, entity=E).

    _remaining_bank3 = [b for b in bank if b.id not in matched_bank]
    _remaining_gl3 = [g for g in gl if g.id not in matched_gl]

    # Identify generic bank items
    _dep_bank = [
        b
        for b in _remaining_bank3
        if b.description.upper() == "DEPOSIT" and b.amount > 0
    ]
    _nsf_bank = [
        b
        for b in _remaining_bank3
        if "RETURNED ITEM" in b.description.upper() and b.amount < 0
    ]

    # Index GL by (sign, amount)
    _gl_pos_by_amt: dict[float, list[Txn]] = {}
    _gl_neg_by_amt: dict[float, list[Txn]] = {}
    for g in _remaining_gl3:
        if g.amount > 0:
            _gl_pos_by_amt.setdefault(g.amount, []).append(g)
        elif g.amount < 0:
            _gl_neg_by_amt.setdefault(g.amount, []).append(g)

    # Index bank by absolute amount
    _dep_by_abs: dict[float, list[Txn]] = {}
    for b in _dep_bank:
        _dep_by_abs.setdefault(b.amount, []).append(b)

    _nsf_by_abs: dict[float, list[Txn]] = {}
    for b in _nsf_bank:
        _nsf_by_abs.setdefault(abs(b.amount), []).append(b)

    _used_dep: set[str] = set()
    _used_nsf: set[str] = set()
    _used_gl3: set[str] = set()

    for abs_amt, dep_list in _dep_by_abs.items():
        nsf_list = _nsf_by_abs.get(abs_amt, [])
        if not nsf_list:
            continue

        # GL candidates for deposit (positive) and NSF (negative)
        gl_pos_cands = [
            g for g in _gl_pos_by_amt.get(abs_amt, []) if g.id not in matched_gl
        ]
        gl_neg_cands = [
            g for g in _gl_neg_by_amt.get(-abs_amt, []) if g.id not in matched_gl
        ]

        if not gl_pos_cands or not gl_neg_cands:
            continue

        # Find GL pairs that share the same entity
        for gp in gl_pos_cands:
            if gp.id in _used_gl3:
                continue
            for gn in gl_neg_cands:
                if gn.id in _used_gl3:
                    continue
                if gp.entity_norm != gn.entity_norm:
                    continue
                # Found a GL pair with same entity — now assign bank items
                dep = next(
                    (
                        b
                        for b in dep_list
                        if b.id not in _used_dep and b.id not in matched_bank
                    ),
                    None,
                )
                nsf = next(
                    (
                        b
                        for b in nsf_list
                        if b.id not in _used_nsf and b.id not in matched_bank
                    ),
                    None,
                )
                if dep and nsf:
                    # Date check: deposit should be close to GL payment
                    if (
                        _date_diff(dep.date, gp.date) <= 10
                        and _date_diff(nsf.date, gn.date) <= 10
                    ):
                        _add_match(
                            dep.id,
                            gp.id,
                            "deposit_nsf",
                            f"Deposit+NSF pair, entity={gp.entity_norm}",
                        )
                        _add_match(
                            nsf.id,
                            gn.id,
                            "deposit_nsf",
                            f"Deposit+NSF pair, entity={gn.entity_norm}",
                        )
                        _used_dep.add(dep.id)
                        _used_nsf.add(nsf.id)
                        _used_gl3.add(gp.id)
                        _used_gl3.add(gn.id)
                        break
            else:
                continue
            break

    # -----------------------------------------------------------------------
    # Pass 4: Offsetting GL pairs
    # -----------------------------------------------------------------------

    unmatched_gl_list = [g for g in gl if g.id not in matched_gl]

    # Build negative-amount index
    neg_gl: dict[float, list[Txn]] = {}
    for g in unmatched_gl_list:
        if g.amount < 0:
            neg_gl.setdefault(g.amount, []).append(g)

    for g_pos in unmatched_gl_list:
        if g_pos.amount >= 0:
            continue
        # Look for a partner with opposite amount
        opp_amount = -g_pos.amount
        # g_pos is negative, find positive partner
        pass

    # Re-approach: pair (a, b) where a.amount + b.amount = 0, a.amount < 0, a.id < b.id
    offset_candidates: list[tuple[Txn, Txn, bool]] = []
    neg_by_amount: dict[float, list[Txn]] = {}
    for g in unmatched_gl_list:
        if g.amount < 0:
            neg_by_amount.setdefault(g.amount, []).append(g)

    for g_pos in unmatched_gl_list:
        if g_pos.amount <= 0:
            continue
        # Find negative partners
        target = -g_pos.amount
        for g_neg in neg_by_amount.get(target, []):
            if g_neg.id in offset_gl or g_pos.id in offset_gl:
                continue
            if _date_diff(g_neg.date, g_pos.date) > 10:
                continue
            # Ensure a.id < b.id (a is negative)
            a, b = (g_neg, g_pos) if g_neg.id < g_pos.id else (g_pos, g_neg)
            if a.amount >= 0:
                a, b = b, a  # ensure a is the negative one
            if a.id >= b.id:
                a, b = b, a

            # Entity confirmation
            entity_confirmed = False
            if a.entity_norm == b.entity_norm:
                entity_confirmed = True
            else:
                # Check VOID entity pattern
                ve_a = _extract_void_entity(a.description)
                ve_b = _extract_void_entity(b.description)
                if ve_b and len(a.entity_norm) >= 4 and ve_b.startswith(a.entity_norm):
                    entity_confirmed = True
                if ve_a and len(b.entity_norm) >= 4 and ve_a.startswith(b.entity_norm):
                    entity_confirmed = True

            if not entity_confirmed:
                continue

            # Jaro-Winkler check
            jw = _jaro_winkler(a.desc_clean, b.desc_clean)
            if jw <= 0.4:
                continue

            offset_candidates.append((a, b, True))

    # Greedy assignment for offsets (avoid double-use)
    for a, b, _ in offset_candidates:
        if a.id in offset_gl or b.id in offset_gl:
            continue
        if a.id in matched_gl or b.id in matched_gl:
            continue
        # Ensure a.amount < 0
        neg, pos = (a, b) if a.amount < 0 else (b, a)
        offset_pairs.append(OffsetPair(neg.id, pos.id))
        offset_gl.add(a.id)
        offset_gl.add(b.id)

    # -----------------------------------------------------------------------
    # Pass 4: Batch deposits — full entity group
    # -----------------------------------------------------------------------

    remaining_bank_pos = [b for b in bank if b.id not in matched_bank and b.amount > 0]
    remaining_gl_pos = [
        g
        for g in gl
        if g.id not in matched_gl and g.id not in offset_gl and g.amount > 0
    ]

    # Group GL by entity_norm
    gl_by_entity: dict[str, list[Txn]] = {}
    for g in remaining_gl_pos:
        if g.entity_norm:
            gl_by_entity.setdefault(g.entity_norm, []).append(g)

    # Only groups with >= 2 entries
    gl_groups = {
        ent: entries for ent, entries in gl_by_entity.items() if len(entries) >= 2
    }

    # Try to match each group to a bank deposit
    used_bank_batch: set[str] = set()
    used_gl_batch: set[str] = set()

    for gl_ent, gl_entries in sorted(gl_groups.items()):
        gl_sum = sum(g.amount for g in gl_entries)
        gl_min_date = min(g.date for g in gl_entries)
        gl_max_date = max(g.date for g in gl_entries)

        best_bank = None
        for b in remaining_bank_pos:
            if b.id in matched_bank or b.id in used_bank_batch:
                continue
            if len(b.entity_norm) < 4:
                continue
            if not _starts_with(gl_ent, b.entity_norm):
                continue
            if abs(gl_sum - b.amount) >= 0.01:
                continue
            # Date window check
            from datetime import timedelta

            if b.date < gl_min_date - timedelta(days=7):
                continue
            if b.date > gl_max_date + timedelta(days=7):
                continue
            best_bank = b
            break

        if best_bank:
            gl_ids = [g.id for g in gl_entries]
            _add_batch(
                best_bank.id, gl_ids, f"Batch: {len(gl_entries)} items, entity={gl_ent}"
            )
            used_bank_batch.add(best_bank.id)
            for gid in gl_ids:
                used_gl_batch.add(gid)

    # -----------------------------------------------------------------------
    # Pass 5: Batch deposits — subset-sum solver
    # -----------------------------------------------------------------------
    # For bank deposits still unmatched, try finding subsets of GL entries
    # from the same entity that sum to the bank amount.

    remaining_bank_pos2 = [b for b in bank if b.id not in matched_bank and b.amount > 0]
    remaining_gl_pos2 = [
        g
        for g in gl
        if g.id not in matched_gl and g.id not in offset_gl and g.amount > 0
    ]

    # Rebuild GL by entity for remaining
    gl_by_entity2: dict[str, list[Txn]] = {}
    for g in remaining_gl_pos2:
        if g.entity_norm:
            gl_by_entity2.setdefault(g.entity_norm, []).append(g)

    for b in remaining_bank_pos2:
        if b.id in matched_bank:
            continue
        if len(b.entity_norm) < 4:
            continue

        # Find GL entries with matching entity prefix
        candidate_gl: list[Txn] = []
        for gl_ent, gl_entries in gl_by_entity2.items():
            if _starts_with(gl_ent, b.entity_norm):
                for g in gl_entries:
                    if g.id not in matched_gl:
                        candidate_gl.append(g)

        if len(candidate_gl) < 2:
            continue

        target = b.amount

        # Try subsets of size 2..min(len, 8) using DP or brute force
        # For small candidate sets (≤ 15), brute force is fine
        found_subset = None

        if len(candidate_gl) <= 20:
            # Brute force over subset sizes, smallest first
            for size in range(2, min(len(candidate_gl) + 1, 9)):
                for combo in combinations(candidate_gl, size):
                    s = sum(g.amount for g in combo)
                    if abs(s - target) < 0.01:
                        # Date check
                        from datetime import timedelta

                        min_d = min(g.date for g in combo)
                        max_d = max(g.date for g in combo)
                        if b.date >= min_d - timedelta(
                            days=7
                        ) and b.date <= max_d + timedelta(days=7):
                            found_subset = combo
                            break
                if found_subset:
                    break

        if found_subset:
            gl_ids = [g.id for g in found_subset]
            _add_batch(
                b.id,
                gl_ids,
                f"Subset batch: {len(gl_ids)} of {len(candidate_gl)} items",
            )
            # Remove used GL from entity index
            for g in found_subset:
                if g.entity_norm in gl_by_entity2:
                    gl_by_entity2[g.entity_norm] = [
                        x for x in gl_by_entity2[g.entity_norm] if x.id != g.id
                    ]

    # -----------------------------------------------------------------------
    # Pass 6: Exact amount + date proximity (unique amounts, no entity needed)
    # -----------------------------------------------------------------------

    remaining_bank6 = [b for b in bank if b.id not in matched_bank]
    remaining_gl6 = [g for g in gl if g.id not in matched_gl and g.id not in offset_gl]

    # Count amounts in each pool
    bank_amt_count: dict[float, int] = {}
    for b in remaining_bank6:
        bank_amt_count[b.amount] = bank_amt_count.get(b.amount, 0) + 1

    gl_amt_count: dict[float, int] = {}
    for g in remaining_gl6:
        gl_amt_count[g.amount] = gl_amt_count.get(g.amount, 0) + 1

    # Index GL by amount (only unique amounts)
    gl_by_amount: dict[float, list[Txn]] = {}
    for g in remaining_gl6:
        gl_by_amount.setdefault(g.amount, []).append(g)

    for b in remaining_bank6:
        if b.id in matched_bank:
            continue
        # Only match when amount is unique in bank pool
        if bank_amt_count.get(b.amount, 0) != 1:
            continue
        gl_cands = gl_by_amount.get(b.amount, [])
        # Only match when amount is unique in GL pool
        if gl_amt_count.get(b.amount, 0) != 1:
            continue
        # Find closest by date within 10 days
        best = None
        best_gap = 999
        for g in gl_cands:
            if g.id in matched_gl:
                continue
            gap = _date_diff(b.date, g.date)
            if gap <= 10 and gap < best_gap:
                best = g
                best_gap = gap
        if best:
            _add_match(
                b.id, best.id, "exact_amount_1to1", "Unique amount + date proximity"
            )
            # Update counts
            gl_by_amount[b.amount] = [
                g for g in gl_by_amount[b.amount] if g.id != best.id
            ]

    # -----------------------------------------------------------------------
    # Pass 7: Entity-prefix + tolerance (≤ $100)
    # -----------------------------------------------------------------------

    remaining_bank7 = [b for b in bank if b.id not in matched_bank]
    remaining_gl7 = [g for g in gl if g.id not in matched_gl and g.id not in offset_gl]

    tol_candidates: list[
        tuple[str, str, float, int]
    ] = []  # (bid, gid, amt_diff, date_gap)
    for b in remaining_bank7:
        if len(b.entity_norm) < 4:
            continue
        for g in remaining_gl7:
            if (b.amount > 0) != (g.amount > 0):
                continue
            amt_diff = abs(b.amount - g.amount)
            if amt_diff > 100 or amt_diff < 0.01:
                continue
            date_gap = _date_diff(b.date, g.date)
            if date_gap > 10:
                continue
            if not _starts_with(g.entity_norm, b.entity_norm):
                continue
            tol_candidates.append((b.id, g.id, amt_diff, date_gap))

    # Deduplicate: each bank and GL should appear at most once
    # Sort by amt_diff, then date_gap
    tol_candidates.sort(key=lambda x: (x[2], x[3]))

    tol_bank_used: set[str] = set()
    tol_gl_used: set[str] = set()
    for bid, gid, amt_diff, date_gap in tol_candidates:
        if bid in matched_bank or bid in tol_bank_used:
            continue
        if gid in matched_gl or gid in tol_gl_used:
            continue
        # Check uniqueness: only match if this is the sole candidate for this bank
        bank_options = sum(
            1
            for b2, g2, _, _ in tol_candidates
            if b2 == bid and g2 not in matched_gl and g2 not in tol_gl_used
        )
        gl_options = sum(
            1
            for b2, g2, _, _ in tol_candidates
            if g2 == gid and b2 not in matched_bank and b2 not in tol_bank_used
        )
        if bank_options == 1 or gl_options == 1:
            _add_match(
                bid,
                gid,
                "tolerance_entity",
                f"Entity prefix + amount diff ${amt_diff:.2f}",
            )
            tol_bank_used.add(bid)
            tol_gl_used.add(gid)

    # -----------------------------------------------------------------------
    # Build results
    # -----------------------------------------------------------------------

    all_matched_gl = matched_gl | offset_gl
    unmatched_bank_ids = {b.id for b in bank if b.id not in matched_bank}
    unmatched_gl_ids = {
        g.id for g in gl if g.id not in matched_gl and g.id not in offset_gl
    }

    # Normalize offsetting pairs for scorer: sorted tuple
    offsetting_set = set()
    for op in offset_pairs:
        ids = sorted([op.original_id, op.reversal_id])
        offsetting_set.add((ids[0], ids[1]))

    return {
        "matched_pairs": {(m.bank_id, m.gl_id) for m in matched},
        "offsetting": offsetting_set,
        "unmatched_bank": unmatched_bank_ids,
        "unmatched_gl": unmatched_gl_ids,
        "matches": matched,
        "offset_pairs": offset_pairs,
        "summary": {},
        "node_meta": {},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Deterministic bank rec solver")
    parser.add_argument("--n", type=int, default=200, help="n_bank for generation")
    parser.add_argument("--difficulty", default="hard", help="easy|medium|hard")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Generate
    print(f"Generating: n={args.n} difficulty={args.difficulty} seed={args.seed}")
    t0 = time.perf_counter()
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
    gen_time = time.perf_counter() - t0

    # Solve
    t1 = time.perf_counter()
    solver = solve(result["bank_transactions"], result["gl_entries"])
    solve_time = time.perf_counter() - t1

    print(
        f"  Solver: {len(solver['matched_pairs'])} pairs, "
        f"{len(solver['offsetting'])} offsetting, "
        f"{len(solver['unmatched_bank'])} unmatched bank, "
        f"{len(solver['unmatched_gl'])} unmatched GL"
    )
    print(
        f"  Time: generate={gen_time:.2f}s  solve={solve_time:.3f}s  total={gen_time + solve_time:.2f}s"
    )

    # Score
    truth = flatten_truth(gt)
    scores = score(truth, solver)
    print_report(scores, solver, s)

    # Match type breakdown
    by_type: dict[str, int] = {}
    for m in solver["matches"]:
        by_type[m.match_type] = by_type.get(m.match_type, 0) + 1
    print("SOLVER MATCH TYPES")
    print("-" * 40)
    for mt, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {mt:<25s} {cnt:>5d}")
    print()

    f1 = scores["pair_accuracy"]["f1_pct"]
    sys.exit(0 if f1 > 50 else 1)


if __name__ == "__main__":
    main()
