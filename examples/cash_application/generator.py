"""Cash Application Problem Generator.

Generates realistic accounts-receivable cash application datasets with
configurable size and difficulty. Produces open invoices, incoming payments
with optional remittance advice, and ground truth application mappings.

Cash application is the process of matching incoming customer payments to
outstanding invoices. This generator models real-world messiness:

- Partial payments (customer pays portion of invoice balance)
- Multi-invoice payments (single check covers several invoices)
- Early payment discounts (1-3% deducted by customer)
- Short-pay deductions (disputes, damage claims, shipping issues)
- Overpayments (customer pays more than owed)
- Duplicate payments (same invoice paid twice)
- Credit memo offsets (negative invoices reduce balance)
- Missing remittance advice (payment with no detail)
- Payer name mismatches (abbreviations, subsidiaries, typos)
- Invoice reference errors (transpositions, PO numbers, garbled refs)
- Unmatched payments (no corresponding invoice)
- Unapplied invoices (no payment received)

Ground truth structure:
  applications: list of {payment_id, invoice_id, applied_amount, match_type}
  unmatched_payments: list of payment_ids with no valid invoice
  unapplied_invoices: list of invoice_ids with no payment received
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass
from datetime import date, timedelta

# ── Company Name Pool ─────────────────────────────────────────────────────────
# (canonical_name, [payer_name_variations])
# Variations model real-world discrepancies: abbreviations, uppercasing,
# subsidiary names, truncations, legal suffix differences.

_COMPANIES: list[tuple[str, list[str]]] = [
    (
        "Meridian Technologies Inc",
        [
            "Meridian Tech",
            "MERIDIAN TECHNOLOGIES",
            "Meridian Technologies",
            "MeridianTech Inc",
        ],
    ),
    (
        "Pacific Coast Supply Co",
        [
            "Pacific Coast Supply",
            "PACIFIC COAST SUPPLY CO",
            "Pac Coast Supply",
            "PCS Co",
        ],
    ),
    (
        "Summit Healthcare Partners",
        ["Summit Healthcare", "SUMMIT HEALTH PARTNERS", "Summit Health", "SHP LLC"],
    ),
    (
        "Atlas Manufacturing Corp",
        ["Atlas Manufacturing", "ATLAS MFG CORP", "Atlas Mfg", "Atlas Mfg Corp"],
    ),
    (
        "Redwood Financial Services",
        ["Redwood Financial", "REDWOOD FINANCIAL SVCS", "Redwood Fin Svc", "RFS Inc"],
    ),
    (
        "Northern Star Logistics",
        ["Northern Star", "NORTHERN STAR LOGISTICS", "N Star Logistics", "NSL"],
    ),
    (
        "Cascade Energy Solutions",
        ["Cascade Energy", "CASCADE ENERGY SOLUTIONS", "Cascade Energy Sol", "CES LLC"],
    ),
    (
        "Horizon Media Group",
        ["Horizon Media", "HORIZON MEDIA GROUP", "Horizon Media Grp", "HMG"],
    ),
    (
        "Pinnacle Construction Ltd",
        ["Pinnacle Construction", "PINNACLE CONSTR LTD", "Pinnacle Constr", "PCL"],
    ),
    (
        "Evergreen Environmental Inc",
        [
            "Evergreen Environmental",
            "EVERGREEN ENVIRON INC",
            "Evergreen Environ",
            "EEI",
        ],
    ),
    (
        "Sterling Automotive Parts",
        ["Sterling Auto Parts", "STERLING AUTOMOTIVE", "Sterling Auto", "SAP Co"],
    ),
    (
        "Beacon Software Systems",
        ["Beacon Software", "BEACON SOFTWARE SYS", "Beacon SW", "BSS Inc"],
    ),
    (
        "Coastal Marine Services",
        ["Coastal Marine", "COASTAL MARINE SVCS", "Coastal Marine Svc", "CMS"],
    ),
    (
        "Ironbridge Industrial",
        ["Ironbridge", "IRONBRIDGE INDUSTRIAL", "Iron Bridge Ind", "IBI Corp"],
    ),
    (
        "Sapphire Electronics Ltd",
        ["Sapphire Electronics", "SAPPHIRE ELECTRONICS LTD", "Sapphire Elec", "SEL"],
    ),
    (
        "Woodland Properties Group",
        ["Woodland Properties", "WOODLAND PROP GROUP", "Woodland Prop Grp", "WPG"],
    ),
    (
        "Falcon Security Solutions",
        ["Falcon Security", "FALCON SECURITY SOL", "Falcon Sec", "FSS"],
    ),
    (
        "Quantum Research Labs",
        ["Quantum Research", "QUANTUM RESEARCH LABS", "Quantum Labs", "QRL"],
    ),
    (
        "Riverview Hospitality Inc",
        ["Riverview Hospitality", "RIVERVIEW HOSP INC", "Riverview Hosp", "RHI"],
    ),
    (
        "Crestline Food Distributors",
        ["Crestline Food", "CRESTLINE FOOD DIST", "Crestline Foods", "CFD"],
    ),
    (
        "Vanguard Legal Associates",
        ["Vanguard Legal", "VANGUARD LEGAL ASSOC", "Vanguard Law", "VLA"],
    ),
    (
        "Titan Aerospace Corp",
        ["Titan Aerospace", "TITAN AEROSPACE CORP", "Titan Aero", "TAC"],
    ),
    (
        "Silverline Telecom Inc",
        ["Silverline Telecom", "SILVERLINE TELECOM INC", "Silverline Tel", "SLT"],
    ),
    (
        "Prairie Agricultural Co",
        ["Prairie Agricultural", "PRAIRIE AGRI CO", "Prairie Ag", "PAC Inc"],
    ),
    (
        "Cobalt Mining Industries",
        ["Cobalt Mining", "COBALT MINING IND", "Cobalt Mine Ind", "CMI"],
    ),
    (
        "Aurora Pharmaceuticals",
        ["Aurora Pharma", "AURORA PHARMACEUTICALS", "Aurora Pharm", "APH"],
    ),
    (
        "Maple Leaf Consulting",
        ["Maple Leaf", "MAPLE LEAF CONSULTING", "Maple Leaf Consult", "MLC"],
    ),
    (
        "Obsidian Data Centers",
        ["Obsidian Data", "OBSIDIAN DATA CENTERS", "Obsidian DC", "ODC"],
    ),
    (
        "Trident Shipping Corp",
        ["Trident Shipping", "TRIDENT SHIPPING CORP", "Trident Ship", "TSC"],
    ),
    (
        "Phoenix Renewable Energy",
        ["Phoenix Renewable", "PHOENIX RENEWABLE ENERGY", "Phoenix Renew", "PRE"],
    ),
]


# ── Invoice Description Templates ────────────────────────────────────────────

_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

_PROJECTS = [
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Phoenix",
    "Titan",
    "Atlas",
    "Mercury",
    "Neptune",
    "Horizon",
    "Apex",
    "Summit",
    "Vanguard",
    "Catalyst",
]

_DESCRIPTION_TEMPLATES = [
    "Professional Services - {month} {year}",
    "Product Order #{num}",
    "Consulting Services - Project {project}",
    "Maintenance Agreement - Q{q} {year}",
    "Software License - Annual Renewal {year}",
    "Equipment Lease - {month} {year}",
    "Marketing Services - Campaign #{num}",
    "IT Support Services - {month} {year}",
    "Training Program - {num} attendees",
    "Freight & Shipping - Order #{num}",
    "Raw Materials - PO #{num}",
    "Office Supplies - {month} {year}",
    "Legal Services - Matter #{num}",
    "Audit & Compliance - FY {year}",
    "Insurance Premium - Q{q} {year}",
    "Temporary Staffing - {month} {year}",
    "Facility Management - {month} {year}",
    "Advertising Placement - {month} {year}",
    "Data Processing Services - {month} {year}",
    "Telecommunications - {month} {year}",
]


# ── Amount Ranges ─────────────────────────────────────────────────────────────
# Categories with (min, max) dollar ranges and selection weights.

_AMOUNT_RANGES = {
    "small": (100, 2_000),
    "medium": (2_000, 15_000),
    "large": (15_000, 75_000),
    "very_large": (75_000, 250_000),
}
_AMOUNT_WEIGHTS = [0.25, 0.40, 0.25, 0.10]

# ── Payment Methods ───────────────────────────────────────────────────────────

_METHODS = ["CHECK", "WIRE", "ACH", "EFT"]
_METHOD_WEIGHTS = [0.40, 0.25, 0.25, 0.10]

# ── Deduction Reasons (for short-pay events) ─────────────────────────────────

_DEDUCTION_REASONS = [
    "Damaged goods",
    "Shipping shortage",
    "Quality defect",
    "Pricing dispute",
    "Late delivery penalty",
    "Promotional allowance",
    "Volume rebate",
    "Warranty claim",
    "Return credit pending",
    "Tax adjustment",
    "Freight overcharge",
    "Missing items",
]


# ── Difficulty Profiles ───────────────────────────────────────────────────────


@dataclass
class DifficultyProfile:
    """Controls the mix of event types and messiness modifiers."""

    # Event type rates (should sum to ~1.0).
    # Each rate is the fraction of total events allocated to that type.
    simple_1to1: float = 0.25
    multi_invoice: float = 0.15
    partial_payment: float = 0.12
    discount_taken: float = 0.08
    short_pay_deduction: float = 0.07
    no_remittance: float = 0.10
    cross_reference: float = 0.05
    duplicate_payment: float = 0.03
    overpayment: float = 0.03
    credit_memo: float = 0.04
    unmatched_payment: float = 0.04
    unapplied_invoice: float = 0.04

    # Messiness modifiers
    name_variation_rate: float = (
        0.30  # fraction of payments with non-canonical payer name
    )
    typo_rate: float = 0.15  # fraction of remittance refs with typos
    duplicate_amount_rate: float = (
        0.10  # fraction of amounts reused across unrelated invoices
    )
    multi_partial_rate: float = (
        0.30  # fraction of multi-invoice events with partial last invoice
    )
    date_spread_days: int = 90  # invoice date spread in days before reference date


EASY = DifficultyProfile(
    simple_1to1=0.50,
    multi_invoice=0.10,
    partial_payment=0.05,
    discount_taken=0.05,
    short_pay_deduction=0.03,
    no_remittance=0.05,
    cross_reference=0.02,
    duplicate_payment=0.02,
    overpayment=0.02,
    credit_memo=0.02,
    unmatched_payment=0.06,
    unapplied_invoice=0.08,
    name_variation_rate=0.10,
    typo_rate=0.05,
    duplicate_amount_rate=0.05,
    multi_partial_rate=0.10,
    date_spread_days=60,
)

MEDIUM = DifficultyProfile(
    simple_1to1=0.35,
    multi_invoice=0.12,
    partial_payment=0.08,
    discount_taken=0.07,
    short_pay_deduction=0.05,
    no_remittance=0.07,
    cross_reference=0.04,
    duplicate_payment=0.03,
    overpayment=0.03,
    credit_memo=0.03,
    unmatched_payment=0.05,
    unapplied_invoice=0.08,
    name_variation_rate=0.25,
    typo_rate=0.12,
    duplicate_amount_rate=0.10,
    multi_partial_rate=0.25,
    date_spread_days=75,
)

HARD = DifficultyProfile(
    simple_1to1=0.25,
    multi_invoice=0.15,
    partial_payment=0.12,
    discount_taken=0.08,
    short_pay_deduction=0.07,
    no_remittance=0.10,
    cross_reference=0.05,
    duplicate_payment=0.03,
    overpayment=0.03,
    credit_memo=0.04,
    unmatched_payment=0.04,
    unapplied_invoice=0.04,
    name_variation_rate=0.40,
    typo_rate=0.20,
    duplicate_amount_rate=0.15,
    multi_partial_rate=0.40,
    date_spread_days=90,
)

_PROFILES = {"easy": EASY, "medium": MEDIUM, "hard": HARD}


# ── Internal Customer Dataclass ───────────────────────────────────────────────


@dataclass
class _Customer:
    customer_id: str
    name: str
    variations: list[str]


# ── Generator ─────────────────────────────────────────────────────────────────


class _Generator:
    """Stateful generator that produces a complete cash application problem set."""

    def __init__(
        self,
        rng: random.Random,
        difficulty: DifficultyProfile,
        year: int,
        month: int,
    ):
        self.rng = rng
        self.diff = difficulty
        self.year = year
        self.month = month

        # Sequence counters
        self._inv_seq = 0
        self._pmt_seq = 0
        self._rem_seq = 0
        self._cust_seq = 0
        self._check_seq = 1000

        # Company pool (shuffled for variety across seeds)
        self._company_pool = list(_COMPANIES)
        rng.shuffle(self._company_pool)
        self._company_idx = 0

        # Customer pool (reused across events for repeat-customer realism)
        self._customers: list[_Customer] = []

        # Generated records
        self.invoices: list[dict] = []
        self.payments: list[dict] = []
        self.remittance_lines: list[dict] = []

        # Ground truth
        self.applications: list[dict] = []
        self.unmatched_payments: list[str] = []
        self.unapplied_invoices: list[str] = []

        # Track amounts for duplicate-amount generation
        self._used_amounts: list[float] = []

    # ── ID Generators ─────────────────────────────────────────────────────

    def _next_inv_id(self) -> str:
        self._inv_seq += 1
        return f"INV-{self._inv_seq:05d}"

    def _next_pmt_id(self) -> str:
        self._pmt_seq += 1
        return f"PMT-{self._pmt_seq:05d}"

    def _next_rem_id(self) -> str:
        self._rem_seq += 1
        return f"REM-{self._rem_seq:05d}"

    def _next_check_num(self) -> str:
        self._check_seq += self.rng.randint(1, 7)
        return str(self._check_seq)

    # ── Data Generators ───────────────────────────────────────────────────

    def _get_or_create_customer(self, prefer_existing: bool = True) -> _Customer:
        """Get an existing customer or create a new one."""
        if prefer_existing and self._customers and self.rng.random() < 0.4:
            return self.rng.choice(self._customers)

        self._cust_seq += 1
        cust_id = f"CUST-{self._cust_seq:03d}"

        if self._company_idx < len(self._company_pool):
            name, variations = self._company_pool[self._company_idx]
            self._company_idx += 1
        else:
            # Wrap around with suffix for uniqueness
            base_name, base_vars = self._company_pool[
                self._company_idx % len(self._company_pool)
            ]
            suffix = f" Division {self._cust_seq}"
            name = base_name + suffix
            variations = [v + suffix for v in base_vars[:2]] + base_vars[2:]
            self._company_idx += 1

        cust = _Customer(cust_id, name, list(variations))
        self._customers.append(cust)
        return cust

    def _gen_amount(self) -> float:
        """Generate a realistic invoice amount using log-uniform distribution."""
        # Chance to reuse an existing amount (creates ambiguity for solvers)
        if self._used_amounts and self.rng.random() < self.diff.duplicate_amount_rate:
            return self.rng.choice(self._used_amounts)

        category = self.rng.choices(
            list(_AMOUNT_RANGES.keys()),
            weights=_AMOUNT_WEIGHTS,
        )[0]
        lo, hi = _AMOUNT_RANGES[category]
        val = math.exp(self.rng.uniform(math.log(lo), math.log(hi)))

        # Round to realistic values
        if val > 10_000:
            val = round(val / 100) * 100
        elif val > 1_000:
            val = round(val / 10) * 10
        elif val > 100:
            val = round(val, 0)
        else:
            val = round(val, 2)
        return float(max(val, 1.0))

    def _gen_invoice_date(self) -> date:
        """Generate a random invoice date within the problem's date range."""
        center = date(self.year, self.month, 15)
        offset = self.rng.randint(-self.diff.date_spread_days, 0)
        d = center + timedelta(days=offset)
        # Clamp to valid range
        return max(d, date(self.year - 1, 1, 1))

    def _gen_payment_date(self, invoice_date: date) -> date:
        """Generate a payment date after the invoice date."""
        delay = self.rng.randint(3, 60)
        return invoice_date + timedelta(days=delay)

    def _gen_description(self) -> str:
        """Generate a realistic invoice description."""
        template = self.rng.choice(_DESCRIPTION_TEMPLATES)
        return template.format(
            month=self.rng.choice(_MONTHS),
            year=self.year,
            num=self.rng.randint(10000, 99999),
            project=self.rng.choice(_PROJECTS),
            q=self.rng.randint(1, 4),
        )

    def _payer_name(self, customer: _Customer) -> str:
        """Return a payer name, sometimes using a variation."""
        if self.rng.random() < self.diff.name_variation_rate and customer.variations:
            return self.rng.choice(customer.variations)
        return customer.name

    def _garble_ref(self, ref: str) -> str:
        """Apply a realistic typo/garble to an invoice reference."""
        if not ref.startswith("INV-"):
            return ref

        num_part = ref[4:]
        garble_type = self.rng.choice(
            ["transpose", "digit", "prefix", "po", "drop_leading_zero"]
        )

        if garble_type == "transpose" and len(num_part) >= 2:
            chars = list(num_part)
            i = self.rng.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return f"INV-{''.join(chars)}"
        elif garble_type == "digit":
            chars = list(num_part)
            i = self.rng.randint(0, len(chars) - 1)
            chars[i] = str(self.rng.randint(0, 9))
            return f"INV-{''.join(chars)}"
        elif garble_type == "prefix":
            return f"Invoice #{num_part.lstrip('0') or '0'}"
        elif garble_type == "drop_leading_zero":
            return f"INV-{num_part.lstrip('0') or '0'}"
        else:  # po
            return f"PO-{self.rng.randint(10000, 99999)}"

    # ── Record Builders ───────────────────────────────────────────────────

    def _make_invoice(
        self,
        customer: _Customer,
        amount: float | None = None,
        inv_date: date | None = None,
        description: str | None = None,
    ) -> dict:
        inv_id = self._next_inv_id()
        if amount is None:
            amount = self._gen_amount()
        if inv_date is None:
            inv_date = self._gen_invoice_date()
        due_date = inv_date + timedelta(days=self.rng.choice([30, 45, 60, 90]))
        if description is None:
            description = self._gen_description()

        inv = {
            "invoice_id": inv_id,
            "customer_id": customer.customer_id,
            "customer_name": customer.name,
            "invoice_date": inv_date,
            "due_date": due_date,
            "amount": amount,
            "description": description,
        }
        self.invoices.append(inv)
        self._used_amounts.append(amount)
        return inv

    def _make_payment(
        self,
        amount: float,
        customer: _Customer,
        pmt_date: date | None = None,
        method: str | None = None,
        ref_info: str = "",
    ) -> dict:
        pmt_id = self._next_pmt_id()
        if pmt_date is None:
            pmt_date = self._gen_invoice_date() + timedelta(days=30)
        if method is None:
            method = self.rng.choices(_METHODS, weights=_METHOD_WEIGHTS)[0]

        if not ref_info:
            if method == "CHECK":
                ref_info = f"CK# {self._next_check_num()}"
            elif method == "WIRE":
                chars = self.rng.choices(
                    string.ascii_uppercase + string.digits,
                    k=10,
                )
                ref_info = f"Wire Ref: {''.join(chars)}"
            elif method == "ACH":
                ref_info = f"ACH {self._payer_name(customer)}"
            else:
                ref_info = f"EFT-{''.join(self.rng.choices(string.digits, k=8))}"

        pmt = {
            "payment_id": pmt_id,
            "payment_date": pmt_date,
            "payer_name": self._payer_name(customer),
            "amount": amount,
            "method": method,
            "reference_info": ref_info,
        }
        self.payments.append(pmt)
        return pmt

    def _make_remittance(
        self,
        payment_id: str,
        invoice_ref: str,
        amount: float,
        memo: str = "",
        garble: str = "never",
    ) -> dict:
        """Create a remittance line.

        garble: "never" | "maybe" | "always" — controls post-remap garbling.
        """
        rem_id = self._next_rem_id()
        rem = {
            "remittance_id": rem_id,
            "payment_id": payment_id,
            "invoice_ref": invoice_ref,
            "amount": amount,
            "memo": memo,
            "_garble": garble,
        }
        self.remittance_lines.append(rem)
        return rem

    # ── Event Generators ──────────────────────────────────────────────────
    # Each returns the number of invoices created (for target tracking).

    def _gen_simple_1to1(self) -> int:
        """One payment exactly matching one invoice, with remittance."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)
        pmt_date = self._gen_payment_date(inv["invoice_date"])
        pmt = self._make_payment(inv["amount"], cust, pmt_date=pmt_date)
        self._make_remittance(
            pmt["payment_id"],
            inv["invoice_id"],
            inv["amount"],
            garble="maybe",
        )
        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": inv["amount"],
                "match_type": "simple_1to1",
            }
        )
        return 1

    def _gen_multi_invoice(self) -> int:
        """One payment covers 2-4 invoices from the same customer."""
        cust = self._get_or_create_customer()
        n_inv = self.rng.randint(2, 4)
        base_date = self._gen_invoice_date()

        invoices = []
        for _ in range(n_inv):
            offset = self.rng.randint(-15, 15)
            inv_date = base_date + timedelta(days=offset)
            inv = self._make_invoice(cust, inv_date=inv_date)
            invoices.append(inv)

        # Decide if last invoice is partially paid
        partial_last = self.rng.random() < self.diff.multi_partial_rate

        partial_amt = 0.0
        if partial_last:
            frac = round(self.rng.uniform(0.3, 0.8), 2)
            partial_amt = round(invoices[-1]["amount"] * frac, 2)
            total = sum(inv["amount"] for inv in invoices[:-1]) + partial_amt
        else:
            total = sum(inv["amount"] for inv in invoices)

        latest_inv_date = max(inv["invoice_date"] for inv in invoices)
        pmt_date = self._gen_payment_date(latest_inv_date)
        pmt = self._make_payment(round(total, 2), cust, pmt_date=pmt_date)

        for i, inv in enumerate(invoices):
            if partial_last and i == len(invoices) - 1:
                applied = partial_amt
                match_type = "multi_invoice_partial"
            else:
                applied = inv["amount"]
                match_type = "multi_invoice"

            self._make_remittance(
                pmt["payment_id"],
                inv["invoice_id"],
                applied,
                garble="maybe",
            )
            self.applications.append(
                {
                    "payment_id": pmt["payment_id"],
                    "invoice_id": inv["invoice_id"],
                    "applied_amount": applied,
                    "match_type": match_type,
                }
            )

        return n_inv

    def _gen_partial_payment(self) -> int:
        """Payment covers only a fraction of the invoice balance."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)
        frac = round(self.rng.uniform(0.25, 0.80), 2)
        partial_amt = round(inv["amount"] * frac, 2)

        pmt_date = self._gen_payment_date(inv["invoice_date"])
        pmt = self._make_payment(partial_amt, cust, pmt_date=pmt_date)

        memo = self.rng.choice(
            [
                f"Partial payment - {inv['invoice_id']}",
                f"Pmt on acct {inv['invoice_id']}",
                "Partial payment per agreement",
                f"Progress payment #{self.rng.randint(1, 3)}",
                f"Installment {self.rng.randint(1, 4)} of {self.rng.randint(3, 6)}",
            ]
        )
        self._make_remittance(
            pmt["payment_id"],
            inv["invoice_id"],
            partial_amt,
            memo,
            garble="maybe",
        )
        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": partial_amt,
                "match_type": "partial_payment",
            }
        )
        return 1

    def _gen_discount_taken(self) -> int:
        """Customer takes an early payment discount (1-3%)."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)

        discount_pct = self.rng.choice([0.01, 0.015, 0.02, 0.025, 0.03])
        discount_amt = round(inv["amount"] * discount_pct, 2)
        paid_amt = round(inv["amount"] - discount_amt, 2)

        # Paid within discount window
        pmt_date = inv["invoice_date"] + timedelta(days=self.rng.randint(3, 15))
        pmt = self._make_payment(paid_amt, cust, pmt_date=pmt_date)

        memo = self.rng.choice(
            [
                f"{discount_pct * 100:.1f}% discount taken",
                f"Less {discount_pct * 100:.0f}% early pay discount",
                f"Discount ${discount_amt:.2f}",
                "Early payment terms applied",
                f"Net {100 - discount_pct * 100:.0f} terms",
            ]
        )
        self._make_remittance(
            pmt["payment_id"],
            inv["invoice_id"],
            paid_amt,
            memo,
            garble="maybe",
        )
        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": paid_amt,
                "match_type": "discount_taken",
            }
        )
        return 1

    def _gen_short_pay_deduction(self) -> int:
        """Customer deducts for a dispute or claim."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)

        deduction_pct = self.rng.uniform(0.03, 0.15)
        deduction_amt = round(inv["amount"] * deduction_pct, 2)
        paid_amt = round(inv["amount"] - deduction_amt, 2)
        reason = self.rng.choice(_DEDUCTION_REASONS)

        pmt_date = self._gen_payment_date(inv["invoice_date"])
        pmt = self._make_payment(paid_amt, cust, pmt_date=pmt_date)

        memo = f"Deduction: {reason} (-${deduction_amt:.2f})"
        self._make_remittance(
            pmt["payment_id"],
            inv["invoice_id"],
            paid_amt,
            memo,
            garble="maybe",
        )
        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": paid_amt,
                "match_type": "short_pay_deduction",
            }
        )
        return 1

    def _gen_no_remittance(self) -> int:
        """Payment with no remittance detail (exact amount match only)."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)

        pmt_date = self._gen_payment_date(inv["invoice_date"])
        pmt = self._make_payment(inv["amount"], cust, pmt_date=pmt_date)
        # No remittance line created — solver must match by amount + customer

        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": inv["amount"],
                "match_type": "no_remittance",
            }
        )
        return 1

    def _gen_cross_reference(self) -> int:
        """Payment with an incorrect invoice reference in remittance."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)

        pmt_date = self._gen_payment_date(inv["invoice_date"])
        pmt = self._make_payment(inv["amount"], cust, pmt_date=pmt_date)

        # Remittance has correct ref now; garbling applied post-remap
        self._make_remittance(
            pmt["payment_id"],
            inv["invoice_id"],
            inv["amount"],
            garble="always",
        )
        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": inv["amount"],
                "match_type": "cross_reference",
            }
        )
        return 1

    def _gen_duplicate_payment(self) -> int:
        """Customer accidentally pays the same invoice twice."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)

        # First payment (legitimate)
        pmt_date1 = self._gen_payment_date(inv["invoice_date"])
        pmt1 = self._make_payment(inv["amount"], cust, pmt_date=pmt_date1)
        self._make_remittance(pmt1["payment_id"], inv["invoice_id"], inv["amount"])

        # Second payment (duplicate, a few days later)
        pmt_date2 = pmt_date1 + timedelta(days=self.rng.randint(1, 10))
        pmt2 = self._make_payment(inv["amount"], cust, pmt_date=pmt_date2)
        self._make_remittance(pmt2["payment_id"], inv["invoice_id"], inv["amount"])

        # First payment applies; second is unmatched
        self.applications.append(
            {
                "payment_id": pmt1["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": inv["amount"],
                "match_type": "duplicate_payment",
            }
        )
        self.unmatched_payments.append(pmt2["payment_id"])
        return 1

    def _gen_overpayment(self) -> int:
        """Customer pays more than the invoice amount."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)

        overpay_pct = self.rng.uniform(0.02, 0.15)
        overpay_total = round(inv["amount"] * (1 + overpay_pct), 2)

        pmt_date = self._gen_payment_date(inv["invoice_date"])
        pmt = self._make_payment(overpay_total, cust, pmt_date=pmt_date)
        self._make_remittance(
            pmt["payment_id"],
            inv["invoice_id"],
            overpay_total,
            memo="Payment on account",
        )

        # Only the invoice amount is applied; excess is on-account
        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": inv["amount"],
                "match_type": "overpayment",
            }
        )
        return 1

    def _gen_credit_memo(self) -> int:
        """Invoice partially offset by a credit memo; payment for net amount."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)

        credit_pct = self.rng.uniform(0.10, 0.40)
        credit_amt = round(inv["amount"] * credit_pct, 2)
        net_amt = round(inv["amount"] - credit_amt, 2)

        # Credit memo is a negative invoice
        credit_date = inv["invoice_date"] + timedelta(days=self.rng.randint(1, 15))
        credit_desc = (
            f"Credit Memo - "
            f"{self.rng.choice(['Return', 'Adjustment', 'Correction', 'Allowance'])} "
            f"ref {inv['invoice_id']}"
        )
        credit = self._make_invoice(
            cust,
            amount=-credit_amt,
            inv_date=credit_date,
            description=credit_desc,
        )

        pmt_date = self._gen_payment_date(credit_date)
        pmt = self._make_payment(net_amt, cust, pmt_date=pmt_date)

        self._make_remittance(
            pmt["payment_id"],
            inv["invoice_id"],
            inv["amount"],
            memo=f"Less credit {credit['invoice_id']}",
            garble="maybe",
        )
        self._make_remittance(
            pmt["payment_id"],
            credit["invoice_id"],
            -credit_amt,
            memo="Credit applied",
        )

        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": inv["invoice_id"],
                "applied_amount": inv["amount"],
                "match_type": "credit_memo",
            }
        )
        self.applications.append(
            {
                "payment_id": pmt["payment_id"],
                "invoice_id": credit["invoice_id"],
                "applied_amount": -credit_amt,
                "match_type": "credit_memo",
            }
        )
        return 2  # invoice + credit memo

    def _gen_unmatched_payment(self) -> int:
        """Payment with no corresponding invoice (advance or wrong company)."""
        cust = self._get_or_create_customer(prefer_existing=False)
        amount = self._gen_amount()
        pmt_date = self._gen_invoice_date() + timedelta(days=self.rng.randint(5, 30))
        pmt = self._make_payment(amount, cust, pmt_date=pmt_date)

        if self.rng.random() < 0.5:
            fake_ref = f"INV-{self.rng.randint(90000, 99999):05d}"
            self._make_remittance(
                pmt["payment_id"],
                fake_ref,
                amount,
                memo="Advance payment",
            )

        self.unmatched_payments.append(pmt["payment_id"])
        return 0

    def _gen_unapplied_invoice(self) -> int:
        """Invoice with no payment (still open/aging)."""
        cust = self._get_or_create_customer()
        inv = self._make_invoice(cust)
        self.unapplied_invoices.append(inv["invoice_id"])
        return 1

    # ── Post-Processing ───────────────────────────────────────────────────

    def _remap_ids(self) -> None:
        """Sort records by date and reassign sequential IDs."""
        self.invoices.sort(key=lambda x: (x["invoice_date"], x["invoice_id"]))
        self.payments.sort(key=lambda x: (x["payment_date"], x["payment_id"]))

        # Build ID maps
        inv_id_map: dict[str, str] = {}
        for i, inv in enumerate(self.invoices, 1):
            old_id = inv["invoice_id"]
            new_id = f"INV-{i:05d}"
            inv_id_map[old_id] = new_id
            inv["invoice_id"] = new_id

        pmt_id_map: dict[str, str] = {}
        for i, pmt in enumerate(self.payments, 1):
            old_id = pmt["payment_id"]
            new_id = f"PMT-{i:05d}"
            pmt_id_map[old_id] = new_id
            pmt["payment_id"] = new_id

        for i, rem in enumerate(self.remittance_lines, 1):
            rem["remittance_id"] = f"REM-{i:05d}"
            rem["payment_id"] = pmt_id_map.get(rem["payment_id"], rem["payment_id"])
            # Remap exact invoice refs (garbled ones won't match)
            if rem["invoice_ref"] in inv_id_map:
                rem["invoice_ref"] = inv_id_map[rem["invoice_ref"]]

        # Remap ground truth
        for app in self.applications:
            app["payment_id"] = pmt_id_map.get(app["payment_id"], app["payment_id"])
            app["invoice_id"] = inv_id_map.get(app["invoice_id"], app["invoice_id"])

        self.unmatched_payments = [
            pmt_id_map.get(p, p) for p in self.unmatched_payments
        ]
        self.unapplied_invoices = [
            inv_id_map.get(i, i) for i in self.unapplied_invoices
        ]

    def _apply_garbling(self) -> None:
        """Apply ref typos/garbling after ID remapping (so garbles use final IDs)."""
        for rem in self.remittance_lines:
            garble = rem.pop("_garble")
            if garble == "always":
                rem["invoice_ref"] = self._garble_ref(rem["invoice_ref"])
            elif garble == "maybe" and self.rng.random() < self.diff.typo_rate:
                rem["invoice_ref"] = self._garble_ref(rem["invoice_ref"])

    # ── Main Entry Point ──────────────────────────────────────────────────

    def run(self, n_invoices: int) -> dict:
        """Generate events until approximately n_invoices invoices are created."""
        event_specs = [
            ("simple_1to1", self.diff.simple_1to1, 1.0),
            ("multi_invoice", self.diff.multi_invoice, 3.0),
            ("partial_payment", self.diff.partial_payment, 1.0),
            ("discount_taken", self.diff.discount_taken, 1.0),
            ("short_pay_deduction", self.diff.short_pay_deduction, 1.0),
            ("no_remittance", self.diff.no_remittance, 1.0),
            ("cross_reference", self.diff.cross_reference, 1.0),
            ("duplicate_payment", self.diff.duplicate_payment, 1.0),
            ("overpayment", self.diff.overpayment, 1.0),
            ("credit_memo", self.diff.credit_memo, 2.0),
            ("unmatched_payment", self.diff.unmatched_payment, 0.0),
            ("unapplied_invoice", self.diff.unapplied_invoice, 1.0),
        ]

        total_rate = sum(rate for _, rate, _ in event_specs)
        avg_inv = sum(rate / total_rate * ipv for _, rate, ipv in event_specs)
        n_events = max(1, round(n_invoices / avg_inv))

        # Allocate event counts proportionally
        counts: dict[str, int] = {}
        allocated = 0
        for name, rate, _ in event_specs[:-1]:
            c = max(0, round(n_events * rate / total_rate))
            counts[name] = c
            allocated += c
        counts[event_specs[-1][0]] = max(0, n_events - allocated)

        # For larger problems, ensure at least 1 of each non-trivial type
        if n_invoices >= 30:
            for name, rate, _ in event_specs:
                if rate > 0 and counts.get(name, 0) == 0:
                    counts[name] = 1

        # Build and shuffle event list
        generators = {
            "simple_1to1": self._gen_simple_1to1,
            "multi_invoice": self._gen_multi_invoice,
            "partial_payment": self._gen_partial_payment,
            "discount_taken": self._gen_discount_taken,
            "short_pay_deduction": self._gen_short_pay_deduction,
            "no_remittance": self._gen_no_remittance,
            "cross_reference": self._gen_cross_reference,
            "duplicate_payment": self._gen_duplicate_payment,
            "overpayment": self._gen_overpayment,
            "credit_memo": self._gen_credit_memo,
            "unmatched_payment": self._gen_unmatched_payment,
            "unapplied_invoice": self._gen_unapplied_invoice,
        }

        event_list = []
        for name, count in counts.items():
            event_list.extend([name] * count)
        self.rng.shuffle(event_list)

        for event_name in event_list:
            generators[event_name]()

        # Post-process
        self._remap_ids()
        self._apply_garbling()

        return {
            "invoices": self.invoices,
            "payments": self.payments,
            "remittance_lines": self.remittance_lines,
            "ground_truth": {
                "applications": self.applications,
                "unmatched_payments": self.unmatched_payments,
                "unapplied_invoices": self.unapplied_invoices,
            },
        }


# ── Public API ────────────────────────────────────────────────────────────────


def generate(
    n_invoices: int,
    seed: int,
    difficulty: str = "hard",
    year: int = 2025,
    month: int = 6,
) -> dict:
    """Generate a cash application problem set.

    Args:
        n_invoices: target number of invoices (actual count may vary slightly)
        seed: random seed for reproducibility
        difficulty: "easy", "medium", or "hard"
        year: reference year for dates
        month: reference month for dates

    Returns:
        dict with keys: invoices, payments, remittance_lines, ground_truth, metadata
    """
    profile = _PROFILES.get(difficulty, HARD)
    rng = random.Random(seed)
    gen = _Generator(rng, profile, year, month)
    result = gen.run(n_invoices)

    result["metadata"] = {
        "n_invoices": len(result["invoices"]),
        "n_payments": len(result["payments"]),
        "n_remittance_lines": len(result["remittance_lines"]),
        "n_applications": len(result["ground_truth"]["applications"]),
        "n_unmatched_payments": len(result["ground_truth"]["unmatched_payments"]),
        "n_unapplied_invoices": len(result["ground_truth"]["unapplied_invoices"]),
        "seed": seed,
        "difficulty": difficulty,
        "target_n": n_invoices,
    }

    # Count match types
    type_counts: dict[str, int] = {}
    for app in result["ground_truth"]["applications"]:
        t = app["match_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    result["metadata"]["match_type_counts"] = type_counts

    return result


def verify(result: dict) -> list[str]:
    """Verify internal consistency of a generated problem set.

    Returns a list of error strings (empty = valid).
    """
    errors = []

    inv_ids = {inv["invoice_id"] for inv in result["invoices"]}
    pmt_ids = {pmt["payment_id"] for pmt in result["payments"]}
    gt = result["ground_truth"]

    # Check all referenced IDs exist
    for app in gt["applications"]:
        if app["payment_id"] not in pmt_ids:
            errors.append(f"Application refs unknown payment: {app['payment_id']}")
        if app["invoice_id"] not in inv_ids:
            errors.append(f"Application refs unknown invoice: {app['invoice_id']}")

    for pmt_id in gt["unmatched_payments"]:
        if pmt_id not in pmt_ids:
            errors.append(f"Unmatched payment refs unknown: {pmt_id}")

    for inv_id in gt["unapplied_invoices"]:
        if inv_id not in inv_ids:
            errors.append(f"Unapplied invoice refs unknown: {inv_id}")

    # Every payment should appear in ground truth (applications or unmatched)
    pmt_in_apps = {app["payment_id"] for app in gt["applications"]}
    pmt_in_unmatched = set(gt["unmatched_payments"])
    pmt_accounted = pmt_in_apps | pmt_in_unmatched

    for pmt_id in pmt_ids:
        if pmt_id not in pmt_accounted:
            errors.append(f"Payment {pmt_id} not accounted for in ground truth")

    # Every invoice should appear in ground truth (applications or unapplied)
    inv_in_apps = {app["invoice_id"] for app in gt["applications"]}
    inv_in_unapplied = set(gt["unapplied_invoices"])
    inv_accounted = inv_in_apps | inv_in_unapplied

    for inv_id in inv_ids:
        if inv_id not in inv_accounted:
            errors.append(f"Invoice {inv_id} not accounted for in ground truth")

    # No ID should be in both applied and unmatched/unapplied
    for pmt_id in pmt_in_apps & pmt_in_unmatched:
        errors.append(f"Payment {pmt_id} in both applications and unmatched")

    for inv_id in inv_in_apps & inv_in_unapplied:
        errors.append(f"Invoice {inv_id} in both applications and unapplied")

    # Check applied amounts are reasonable
    inv_amounts = {inv["invoice_id"]: inv["amount"] for inv in result["invoices"]}
    for app in gt["applications"]:
        inv_amt = inv_amounts.get(app["invoice_id"], 0)
        if inv_amt > 0 and app["applied_amount"] > inv_amt * 1.01:
            errors.append(
                f"Applied amount {app['applied_amount']:.2f} exceeds invoice "
                f"amount {inv_amt:.2f} for {app['invoice_id']}"
            )

    return errors


# ── CLI Smoke Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for n in [10, 30, 100, 300, 1000]:
        result = generate(n, seed=42, difficulty="hard")
        errs = verify(result)
        m = result["metadata"]
        status = "PASS" if not errs else f"FAIL ({len(errs)} errors)"
        print(
            f"n={n:>4d}  invoices={m['n_invoices']:>4d}  "
            f"payments={m['n_payments']:>4d}  "
            f"remittance={m['n_remittance_lines']:>4d}  "
            f"applications={m['n_applications']:>4d}  "
            f"unmatched_pmt={m['n_unmatched_payments']:>2d}  "
            f"unapplied_inv={m['n_unapplied_invoices']:>2d}  "
            f"{status}"
        )
        if errs:
            for e in errs[:5]:
                print(f"  - {e}")
