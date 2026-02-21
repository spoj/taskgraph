"""Synthetic Bank Reconciliation Problem Generator

Generate bank reconciliation problems of arbitrary size (1K-10K+ transactions)
with configurable difficulty.  The core strategy:

1. Allocate N "base events" across mutation types (simple, batch, void/reissue, ...)
2. For each event, produce bank-side and GL-side records
3. Apply description/date/amount mutations to create realistic challenges
4. Return data + ground truth for validation

Mutation types (mirrors every difficulty feature in bank_rec_problem.py):
  SIMPLE 1:1      — clean match, varying description difficulty
  DATE_SHIFT      — bank clearing date differs from GL posting date
  CRYPTIC_DESC    — bank shows "AMZN MKTP US*...", "ADP TOFCPYRL ..."
  TRUNCATED_DESC  — bank truncates vendor name to 15-20 chars
  BATCH_DEPOSIT   — N GL entries deposited as one bank line
  AMOUNT_MISMATCH — small fee netted on one side ($10-$50 difference)
  VOID_REISSUE    — voided check + reissue (3 GL entries, 1 bank)
  SELF_CORRECTING — wrong payment -> reversal -> correct (3 bank, 3 GL)
  NSF_RETURN      — deposit + returned-item pair (2 bank, 2 GL)
  DUP_AMOUNT      — two different vendors, same amount, different dates
  SAME_AMT_DATE   — same amount AND date, description is only signal
  DUP_VENDOR_PMT  — same vendor, same amount, different check numbers
  PRIOR_PERIOD    — GL dated in prior month, bank clears current month
  UNMATCHED_BANK  — bank fees/interest with no GL counterpart
  UNMATCHED_GL    — outstanding checks / deposits in transit
  OFFSETTING_GL   — GL void pairs that net to zero

Usage:
    from examples.bank_rec_generator import generate, verify, EASY, MEDIUM, HARD

    result = generate(n_bank=1000, seed=42, difficulty="medium")
    bank_txns   = result["bank_transactions"]
    gl_entries  = result["gl_entries"]
    ground_truth = result["ground_truth"]

    errors = verify(result)  # [] if ground truth is consistent
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta


# ═══════════════════════════════════════════════════════════════════════════════
# Constants — name components, categories, description templates
# ═══════════════════════════════════════════════════════════════════════════════

_ROOTS = [
    "Apex",
    "Summit",
    "Pinnacle",
    "Cascade",
    "Sterling",
    "Meridian",
    "Pacific",
    "Atlantic",
    "Harbor",
    "Lakeside",
    "Mountain",
    "Valley",
    "Brightstar",
    "Northstar",
    "Skyline",
    "Ironwood",
    "Silverline",
    "Redwood",
    "Oakwood",
    "Cedarpoint",
    "Riverstone",
    "Greenfield",
    "Westfield",
    "Eastgate",
    "Crossroads",
    "Horizon",
    "Vanguard",
    "Keystone",
    "Cornerstone",
    "Milestone",
    "Baseline",
    "Ridgeline",
    "Crestview",
    "Fairview",
    "Grandview",
    "Lakeview",
    "Clearwater",
    "Blackstone",
    "Goldcrest",
    "Whitehorse",
    "Blueridge",
    "Greentree",
    "Firestone",
    "Ironclad",
    "Titanium",
    "Platinum",
    "Diamond",
    "Emerald",
    "Sapphire",
    "Granite",
    "Cobalt",
    "Quartz",
    "Crimson",
    "Amber",
    "Falcon",
    "Eagle",
    "Osprey",
    "Sequoia",
    "Magnolia",
    "Cypress",
    "Birchwood",
    "Maplewood",
    "Stonebridge",
    "Ridgeway",
    "Brookfield",
    "Ashland",
    "Thornton",
    "Waverly",
    "Prescott",
    "Stratton",
    "Belmont",
    "Kensington",
]

_INDUSTRIES = [
    "Technology",
    "Solutions",
    "Industries",
    "Services",
    "Manufacturing",
    "Engineering",
    "Consulting",
    "Systems",
    "Enterprises",
    "Holdings",
    "Partners",
    "Associates",
    "Group",
    "Logistics",
    "Distribution",
    "Supply",
    "Resources",
    "Analytics",
    "Digital",
    "Creative",
    "Strategic",
    "Professional",
    "Development",
    "Management",
    "Innovations",
    "Dynamics",
    "Networks",
    "Capital",
    "Ventures",
    "Global",
]

_SUFFIXES = ["Inc", "LLC", "Co", "Ltd", "Corp", "LLP", "LP"]

_LAST_NAMES = [
    "Johnson",
    "Smith",
    "Williams",
    "Brown",
    "Davis",
    "Miller",
    "Wilson",
    "Moore",
    "Taylor",
    "Anderson",
    "Thomas",
    "Jackson",
    "White",
    "Harris",
    "Martin",
    "Thompson",
    "Garcia",
    "Martinez",
    "Robinson",
    "Clark",
    "Lewis",
    "Lee",
    "Walker",
    "Hall",
    "Allen",
    "Young",
    "King",
    "Wright",
]

# Templates for confusable vendor names — {name} is replaced with a last name
_CONFUSABLE_TEMPLATES = [
    "{name} & Associates",
    "K. {name} Consulting Group",
    "{name} Associates LLC",
    "{name} Professional Services",
    "The {name} Group Inc",
    "{name} & Partners LLP",
    "{name} Management Co",
]

# Categories with default sign and amount range
_CATEGORIES = {
    "rent": {"sign": -1, "lo": 2_000, "hi": 25_000},
    "payroll": {"sign": -1, "lo": 30_000, "hi": 250_000},
    "utilities": {"sign": -1, "lo": 500, "hi": 15_000},
    "insurance": {"sign": -1, "lo": 3_000, "hi": 40_000},
    "supplies": {"sign": -1, "lo": 50, "hi": 5_000},
    "professional_svc": {"sign": -1, "lo": 1_000, "hi": 75_000},
    "equipment": {"sign": -1, "lo": 500, "hi": 50_000},
    "travel": {"sign": -1, "lo": 150, "hi": 12_000},
    "taxes": {"sign": -1, "lo": 500, "hi": 100_000},
    "subscription": {"sign": -1, "lo": 100, "hi": 10_000},
    "credit_card": {"sign": -1, "lo": 3_000, "hi": 60_000},
    "customer_payment": {"sign": 1, "lo": 1_000, "hi": 300_000},
    "refund": {"sign": 1, "lo": 100, "hi": 10_000},
    "marketing": {"sign": -1, "lo": 500, "hi": 50_000},
    "legal": {"sign": -1, "lo": 1_000, "hi": 100_000},
}

_CATEGORY_WEIGHTS = {
    "rent": 3,
    "payroll": 2,
    "utilities": 3,
    "insurance": 2,
    "supplies": 15,
    "professional_svc": 15,
    "equipment": 8,
    "travel": 8,
    "taxes": 3,
    "subscription": 5,
    "credit_card": 3,
    "customer_payment": 30,
    "refund": 3,
    "marketing": 8,
    "legal": 5,
}

# Bank-side cryptic description templates (by category)
_CRYPTIC_TEMPLATES = {
    "supplies": [
        "AMZN MKTP US*{code} AMZN.CO",
        "AMZN MKTP US*{code}",
        "TST* AMAZON {code} AWS",
        "SQ *{product} {code}",
    ],
    "subscription": [
        "MSFT *{product} {ref}",
        "GOOG *{product} {ref}",
        "INTUIT *{product}",
    ],
    "payroll": [
        "ADP TOFCPYRL {date_code} XXXXXX{last4}",
        "PAYCHEX E-PAY {ref}",
        "GUSTO PAY {date_code}",
    ],
    "travel": [
        "DELTA AIR {ref}",
        "UNITED {ref}",
        "AA {ref}",
        "UBER   *TRIP {ref}",
        "LYFT   *RIDE {ref}",
    ],
    "utilities": ["PG&E WEBPMT {ref}", "AT&T *PAYMENT {ref}", "COMCAST CABLE {ref}"],
    "insurance": ["STATE FARM RO {ref}", "GEICO *PAYMENT {ref}"],
}

_DETAIL_PHRASES = {
    "rent": ["Office space lease", "Monthly rent", "Facility rental"],
    "payroll": ["Net pay", "Payroll processing", "Employee compensation"],
    "utilities": ["Electric and gas", "Water and sewer", "Telecom services"],
    "insurance": [
        "General liability renewal",
        "Workers comp premium",
        "Property insurance",
    ],
    "supplies": [
        "Office supplies",
        "Cleaning supplies",
        "Maintenance parts",
        "Printer cartridges",
    ],
    "professional_svc": [
        "Consulting engagement",
        "Legal retainer",
        "Audit services",
        "IT support contract",
    ],
    "equipment": ["Equipment purchase", "Machinery maintenance", "Hardware upgrade"],
    "travel": ["Business travel", "Conference attendance", "Client site visit"],
    "taxes": ["Sales tax remittance", "Property tax", "Payroll tax"],
    "subscription": ["Annual license", "Monthly SaaS subscription", "Cloud platform"],
    "credit_card": ["Corporate card payment", "Credit card statement balance"],
    "customer_payment": [
        "Invoice payment",
        "Account receivable",
        "Contract milestone",
        "Service fee",
    ],
    "refund": ["Credit memo", "Returned goods", "Overpayment refund"],
    "marketing": [
        "Ad campaign",
        "Social media marketing",
        "SEO services",
        "Billboard advertising",
    ],
    "legal": ["Litigation retainer", "Contract review", "Trademark registration"],
}

_UNMATCHED_BANK_TYPES = [
    ("SERVICE CHARGE", -15, -75),
    ("WIRE TRF FEE", -15, -50),
    ("INTEREST PAYMENT", 3, 60),
    ("MONTHLY MAINTENANCE FEE", -10, -50),
    ("ANALYSIS CHARGE", -20, -100),
    ("FOREIGN TXN FEE", -5, -25),
    ("RETURNED ITEM FEE", -15, -35),
    ("OVERDRAFT FEE", -35, -35),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DifficultyProfile:
    """Controls the mix of mutation types and difficulty modifiers.

    *Event-type rates* (should roughly sum to 1.0) determine what fraction
    of base events become each structural pattern.

    *Modifier rates* are applied to simple 1:1 events to increase ambiguity.
    """

    # --- Event type rates (fraction of base events) ---
    simple: float = 0.58
    batch_deposit: float = 0.04  # 1 bank = sum of K GL entries
    void_reissue: float = 0.02  # 1 bank, 3 GL (orig + void + reissue)
    self_correcting: float = 0.02  # 3 bank, 3 GL
    nsf_return: float = 0.02  # 2 bank, 2 GL
    amount_mismatch: float = 0.02  # 1 bank, 1 GL, amounts differ
    dup_vendor_pmt: float = 0.02  # 2 bank, 2 GL, same vendor + amount
    unmatched_bank: float = 0.03  # 1 bank, 0 GL
    unmatched_gl: float = 0.03  # 0 bank, 1 GL
    offsetting_gl: float = 0.02  # 0 bank, 2 GL (net to zero)

    # --- Difficulty modifiers on simple 1:1 events ---
    dup_amount_frac: float = 0.12  # fraction in dup-amount pairs
    same_amt_date_frac: float = 0.04  # fraction in same-amt-same-date pairs
    weekly_recurring_frac: float = 0.0  # fraction in weekly recurring series
    prior_period_frac: float = 0.03  # fraction dated in prior month
    date_shift_frac: float = 0.30  # fraction where bank date != GL date
    cryptic_desc_frac: float = 0.15  # fraction with cryptic bank desc
    truncated_desc_frac: float = 0.20  # fraction with truncated bank desc
    typo_frac: float = 0.0  # fraction of GL descriptions with typos

    # --- Batch deposit sizing (capped at 5: solver uses brute-force
    #     k-way joins so k>5 is combinatorially infeasible) ---
    batch_size_min: int = 2
    batch_size_max: int = 3


# Presets ---------------------------------------------------------------

EASY = DifficultyProfile(
    simple=0.74,
    batch_deposit=0.02,
    void_reissue=0.01,
    self_correcting=0.01,
    nsf_return=0.01,
    amount_mismatch=0.01,
    dup_vendor_pmt=0.01,
    unmatched_bank=0.02,
    unmatched_gl=0.02,
    offsetting_gl=0.01,
    dup_amount_frac=0.06,
    same_amt_date_frac=0.01,
    weekly_recurring_frac=0.0,
    prior_period_frac=0.02,
    date_shift_frac=0.15,
    cryptic_desc_frac=0.08,
    truncated_desc_frac=0.10,
    batch_size_min=2,
    batch_size_max=2,
    typo_frac=0.0,
)

MEDIUM = DifficultyProfile(
    simple=0.65,
    batch_deposit=0.03,
    void_reissue=0.02,
    self_correcting=0.02,
    nsf_return=0.02,
    amount_mismatch=0.02,
    dup_vendor_pmt=0.02,
    unmatched_bank=0.02,
    unmatched_gl=0.02,
    offsetting_gl=0.02,
    dup_amount_frac=0.15,
    same_amt_date_frac=0.04,
    weekly_recurring_frac=0.08,
    prior_period_frac=0.03,
    date_shift_frac=0.25,
    cryptic_desc_frac=0.15,
    truncated_desc_frac=0.20,
    typo_frac=0.05,
    batch_size_min=2,
    batch_size_max=3,
)

HARD = DifficultyProfile(
    simple=0.42,
    batch_deposit=0.06,
    void_reissue=0.04,
    self_correcting=0.04,
    nsf_return=0.03,
    amount_mismatch=0.04,
    dup_vendor_pmt=0.04,
    unmatched_bank=0.04,
    unmatched_gl=0.04,
    offsetting_gl=0.03,
    dup_amount_frac=0.22,
    same_amt_date_frac=0.08,
    weekly_recurring_frac=0.12,
    prior_period_frac=0.05,
    date_shift_frac=0.40,
    cryptic_desc_frac=0.25,
    truncated_desc_frac=0.30,
    batch_size_min=2,
    batch_size_max=5,
    typo_frac=0.15,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal generator
# ═══════════════════════════════════════════════════════════════════════════════


class _Generator:
    """Stateful generator that accumulates bank/GL records and ground truth."""

    def __init__(
        self, n_bank: int, seed: int, profile: DifficultyProfile, year: int, month: int
    ):
        self.rng = random.Random(seed)
        self.p = profile
        self.year = year
        self.month = month
        self.n_bank_target = n_bank

        # Counters
        self._b_seq = 0
        self._g_seq = 0
        self._check_no = 4500 + self.rng.randint(0, 500)
        self._ref_seq = 1000

        # Accumulators
        self._bank: list[dict] = []
        self._gl: list[dict] = []

        # Ground truth buckets
        self._m1: list[dict] = []  # matches_1to1
        self._mm: list[dict] = []  # matches_many_to_one
        self._ma: list[dict] = []  # matches_amount_mismatch
        self._off: list[dict] = []  # offsetting_gl_pairs
        self._ub: list[dict] = []  # unmatched_bank
        self._ug: list[dict] = []  # unmatched_gl

        # Vendor dedup
        self._used_names: set[str] = set()
        self._used_amounts: set[float] = set()

        # Precompute days in month
        if month == 12:
            self._month_days = (date(year + 1, 1, 1) - date(year, month, 1)).days
        else:
            self._month_days = (date(year, month + 1, 1) - date(year, month, 1)).days

    # ------------------------------------------------------------------
    # ID / ref helpers
    # ------------------------------------------------------------------

    def _bid(self) -> str:
        self._b_seq += 1
        w = max(2, len(str(self.n_bank_target * 2)))
        return f"B{self._b_seq:0{w}d}"

    def _gid(self) -> str:
        self._g_seq += 1
        w = max(2, len(str(self.n_bank_target * 2)))
        return f"G{self._g_seq:0{w}d}"

    def _chk(self) -> int:
        self._check_no += 1
        return self._check_no

    def _ref(self, prefix: str = "AP") -> str:
        self._ref_seq += 1
        return f"{prefix}-{self._ref_seq}"

    # ------------------------------------------------------------------
    # Vendor generation
    # ------------------------------------------------------------------

    def _random_category(self) -> str:
        cats = list(_CATEGORY_WEIGHTS.keys())
        ws = [_CATEGORY_WEIGHTS[c] for c in cats]
        return self.rng.choices(cats, weights=ws, k=1)[0]

    def _vendor(self, category: str | None = None) -> tuple[str, str, str]:
        """Return (full_name, bank_short_name, category)."""
        if category is None:
            category = self._random_category()
        for _ in range(200):
            root = self.rng.choice(_ROOTS)
            ind = self.rng.choice(_INDUSTRIES)
            suf = self.rng.choice(_SUFFIXES)
            full = f"{root} {ind} {suf}"
            if full not in self._used_names:
                self._used_names.add(full)
                short = full.upper()[:25].rstrip()
                return full, short, category
        # fallback
        n = len(self._used_names)
        full = f"Vendor-{n} Services Inc"
        self._used_names.add(full)
        return full, full.upper()[:25], category

    def _confusable_pair(self, category: str) -> list[tuple[str, str, str]]:
        """Generate 2 vendors with confusingly similar names."""
        last = self.rng.choice(_LAST_NAMES)
        # avoid reuse of same last name root
        for _ in range(20):
            if not any(last in n for n in self._used_names):
                break
            last = self.rng.choice(_LAST_NAMES)
        templates = list(_CONFUSABLE_TEMPLATES)
        self.rng.shuffle(templates)
        pair = []
        for t in templates:
            name = t.format(name=last)
            if name not in self._used_names:
                self._used_names.add(name)
                pair.append((name, name.upper()[:25].rstrip(), category))
            if len(pair) == 2:
                break
        # fallback if not enough unique names
        while len(pair) < 2:
            pair.append(self._vendor(category))
        return pair

    # ------------------------------------------------------------------
    # Amount generation
    # ------------------------------------------------------------------

    def _amount(self, category: str, unique: bool = True) -> float:
        cat = _CATEGORIES[category]
        lo, hi = cat["lo"], cat["hi"]
        sign = cat["sign"]
        for _ in range(200):
            log_lo = math.log(max(lo, 1))
            log_hi = math.log(max(hi, 2))
            raw = math.exp(self.rng.uniform(log_lo, log_hi))
            # Round to realistic precision
            r = self.rng.random()
            if r < 0.2:
                # Flat hundred/thousand
                if raw >= 1000:
                    val = round(raw / 100) * 100.0
                else:
                    val = round(raw / 10) * 10.0
            elif r < 0.4:
                # Ending in .99 or .50
                val = (
                    round(raw) - 0.01 if self.rng.random() < 0.5 else round(raw) + 0.50
                )
            else:
                # Random cents
                val = round(raw, 2)
            if not unique or val not in self._used_amounts:
                self._used_amounts.add(val)
                return val * sign
        # fallback with cents
        val = round(self.rng.uniform(lo, hi), 2)
        self._used_amounts.add(val)
        return val * sign

    # ------------------------------------------------------------------
    # Date generation
    # ------------------------------------------------------------------

    def _date(self, month: int | None = None, year: int | None = None) -> date:
        m = month or self.month
        y = year or self.year
        days = self._month_days if (m == self.month and y == self.year) else 28
        return date(y, m, self.rng.randint(1, days))

    def _shift_weekdays(self, d: date, lo: int, hi: int) -> date:
        shift = self.rng.randint(lo, hi)
        target = d + timedelta(days=shift)
        # push forward if it lands on weekend
        if target.weekday() == 5:  # Saturday
            target += timedelta(days=2)
        elif target.weekday() == 6:  # Sunday
            target += timedelta(days=1)
        return target

    def _clamp_to_month(self, d: date) -> date:
        """Clamp a date to the target month."""
        first = date(self.year, self.month, 1)
        last = date(self.year, self.month, self._month_days)
        if d < first:
            return first
        if d > last:
            return last
        return d

    # ------------------------------------------------------------------
    # Description generation
    # ------------------------------------------------------------------

    def _bank_desc(
        self, short: str, category: str, is_credit: bool, style: str = "auto"
    ) -> str:
        """Generate a bank-side description."""
        if style == "auto":
            r = self.rng.random()
            if r < self.p.cryptic_desc_frac:
                c = self._cryptic(category)
                if c:
                    return c
                # Cryptic template unavailable for this category; re-roll
                # against truncated vs standard using a fresh random draw
                # to avoid biasing the truncated rate.
                r = self.rng.random()
            if r < self.p.truncated_desc_frac:
                style = "truncated"
            else:
                style = "standard"

        if style == "cryptic":
            c = self._cryptic(category)
            if c:
                return c
            style = "standard"

        prefix = (
            "ACH CREDIT"
            if is_credit
            else self.rng.choice(["ACH DEBIT", "POS DEBIT", "E-PAYMENT", "AUTO-PAY"])
        )
        if style == "truncated":
            trunc_len = self.rng.randint(12, 20)
            base = short[:trunc_len].rstrip()
            if self.rng.random() < 0.3:
                base = base.replace(" ", "")
            return f"{prefix} {base}"
        return f"{prefix} {short}"

    def _cryptic(self, category: str) -> str | None:
        templates = _CRYPTIC_TEMPLATES.get(category)
        if not templates:
            return None
        t = self.rng.choice(templates)
        code = "".join(self.rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=8))
        ref = f"{self.rng.randint(100000, 9999999):07d}"
        product = self.rng.choice(
            [
                "AZURE",
                "365",
                "TEAMS",
                "CLOUD",
                "ADS",
                "WORKSPACE",
                "QUICKBOOKS",
                "SUITE",
            ]
        )
        dc = f"{self.month:02d}{self.rng.randint(1, 28):02d}{self.year % 100}"
        last4 = f"{self.rng.randint(1000, 9999)}"
        return t.format(code=code, ref=ref, product=product, date_code=dc, last4=last4)

    def _gl_desc(self, full: str, category: str) -> str:
        detail = self.rng.choice(_DETAIL_PHRASES.get(category, ["Payment"]))
        inv = self._ref("INV")
        mo = date(self.year, self.month, 1).strftime("%B %Y")
        desc = f"{full} - {detail}, {inv}, {mo}"
        if (
            getattr(self.p, "typo_frac", 0.0) > 0
            and self.rng.random() < self.p.typo_frac
        ):
            if len(desc) > 10:
                chars = list(desc)
                idx = self.rng.randint(0, len(full) - 1)
                if self.rng.random() < 0.5:
                    chars.pop(idx)
                else:
                    if idx < len(chars) - 1:
                        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                desc = "".join(chars)
        return desc

    def _gl_ref(self, category: str, is_credit: bool = False) -> str:
        if is_credit or category in ("customer_payment", "refund"):
            return self._ref("AR")
        if category == "payroll":
            return f"PR-{self.year}-{self.month:02d}{chr(65 + self.rng.randint(0, 3))}"
        if category == "taxes":
            return self._ref("TAX")
        if category == "credit_card":
            return f"CC-{self.month:02d}{self.year % 100}"
        return self._ref("AP")

    # ------------------------------------------------------------------
    # Record builders
    # ------------------------------------------------------------------

    def _brec(self, d: date, desc: str, amt: float) -> dict:
        return {"id": self._bid(), "date": d, "description": desc, "amount": amt}

    def _grec(
        self, d: date, desc: str, amt: float, ref: str, etype: str = "normal"
    ) -> dict:
        return {
            "id": self._gid(),
            "date": d,
            "description": desc,
            "amount": amt,
            "ref": ref,
            "entry_type": etype,
        }

    # ------------------------------------------------------------------
    # Event generators
    # ------------------------------------------------------------------

    def _gen_simple(
        self, vendor=None, amount=None, gl_date=None, bank_date=None, note_extra=""
    ):
        """1:1 exact match with auto-selected description difficulty."""
        if vendor is None:
            vendor = self._vendor()
        full, short, cat = vendor
        if amount is None:
            amount = self._amount(cat)
        is_credit = amount > 0
        if gl_date is None:
            gl_date = self._date()
        if bank_date is None:
            is_check = not is_credit and self.rng.random() < 0.3
            if self.rng.random() < self.p.date_shift_frac:
                if is_check:
                    bank_date = self._clamp_to_month(
                        self._shift_weekdays(gl_date, 3, 14)
                    )
                else:
                    bank_date = self._clamp_to_month(
                        self._shift_weekdays(gl_date, 1, 4)
                    )
            else:
                bank_date = self._clamp_to_month(self._shift_weekdays(gl_date, 0, 1))

        bdesc = self._bank_desc(short, cat, is_credit)
        gdesc = self._gl_desc(full, cat)
        gref = self._gl_ref(cat, is_credit)

        b = self._brec(bank_date, bdesc, amount)
        g = self._grec(gl_date, gdesc, amount, gref)
        self._bank.append(b)
        self._gl.append(g)
        note = f"1:1 match, {full}, ${abs(amount):,.2f}"
        if note_extra:
            note += f" ({note_extra})"
        self._m1.append({"bank_id": b["id"], "gl_id": g["id"], "note": note})
        return b, g

    def _gen_batch_deposit(self):
        """1 bank record = sum of K GL entries from the SAME vendor/customer.

        Models the common B2B scenario where a single payment covers
        multiple invoices. Can be a batch deposit (AR) or batch payment (AP).
        """
        k = self.rng.randint(self.p.batch_size_min, self.p.batch_size_max)
        base = self._date()

        is_deposit = self.rng.random() < 0.5
        cat = "customer_payment" if is_deposit else "professional_svc"
        v = self._vendor(cat)
        full, short, cat = v

        gl_recs = []
        total = 0.0
        for _ in range(k):
            amt = abs(self._amount(cat))
            if not is_deposit:
                amt = -amt
            total += amt
            # GL dates cluster within 0-5 days *before* the base date
            gd = self._clamp_to_month(self._shift_weekdays(base, -5, 0))
            g = self._grec(
                gd,
                self._gl_desc(full, cat),
                amt,
                self._gl_ref(cat, is_credit=is_deposit),
            )
            gl_recs.append(g)
            self._gl.append(g)

        total = round(total, 2)
        bd = self._clamp_to_month(self._shift_weekdays(base, 0, 2))
        bdesc = self._bank_desc(short, cat, is_credit=is_deposit)
        b = self._brec(bd, bdesc, total)
        self._bank.append(b)
        self._mm.append(
            {
                "bank_id": b["id"],
                "gl_ids": [g["id"] for g in gl_recs],
                "bank_amount": total,
                "gl_amounts": {g["id"]: g["amount"] for g in gl_recs},
                "note": f"Batch {'deposit' if is_deposit else 'payment'}: {k} items totalling ${abs(total):,.2f}",
            }
        )

    def _gen_void_reissue(self):
        """Voided check + reissue: 3 GL entries, 1 bank entry."""
        v = self._vendor("professional_svc")
        full, short, _ = v
        amt = -abs(self._amount("professional_svc"))
        gd = self._date()

        orig_chk = self._chk()
        g_orig = self._grec(
            gd,
            f"{full} - {self.rng.choice(_DETAIL_PHRASES['professional_svc'])}, {self._ref('INV')}",
            amt,
            f"CHK-{orig_chk}",
        )
        void_d = self._clamp_to_month(gd + timedelta(days=self.rng.randint(2, 7)))
        g_void = self._grec(
            void_d,
            f"VOID - Check #{orig_chk} to {full} (incorrect information)",
            -amt,
            f"CHK-{orig_chk}-V",
            "void",
        )
        re_chk = self._chk()
        re_d = self._clamp_to_month(void_d + timedelta(days=self.rng.randint(1, 5)))
        g_re = self._grec(
            re_d,
            f"{full} - Reissued payment (replaces voided #{orig_chk})",
            amt,
            f"CHK-{re_chk}",
        )
        bd = self._clamp_to_month(re_d + timedelta(days=self.rng.randint(0, 3)))
        b = self._brec(bd, f"CHECK #{re_chk}", amt)

        self._gl.extend([g_orig, g_void, g_re])
        self._bank.append(b)
        self._m1.append(
            {
                "bank_id": b["id"],
                "gl_id": g_re["id"],
                "note": f"Reissued check #{re_chk}, {full}, ${abs(amt):,.2f}",
            }
        )
        self._off.append(
            {
                "gl_ids": [g_orig["id"], g_void["id"]],
                "amounts": {g_orig["id"]: amt, g_void["id"]: -amt},
                "note": f"Voided check #{orig_chk} to {full}",
            }
        )

    def _gen_self_correcting(self):
        """Wrong payment -> reversal -> correct payment (3 bank, 3 GL)."""
        wv = self._vendor("professional_svc")
        rv = self._vendor("professional_svc")
        wfull, wshort, _ = wv
        rfull, rshort, _ = rv
        amt = -abs(self._amount("professional_svc"))
        d0 = self._date()

        # wrong
        g_w = self._grec(
            d0,
            f"{wfull} - Payment entered in error (see correction)",
            amt,
            self._ref("AP"),
        )
        b_w = self._brec(
            self._clamp_to_month(d0 + timedelta(days=self.rng.randint(0, 2))),
            f"ACH DEBIT {wshort}",
            amt,
        )
        # reversal
        d1 = self._clamp_to_month(d0 + timedelta(days=self.rng.randint(1, 4)))
        g_r = self._grec(
            d1,
            f"{wfull} - Reversal of erroneous payment",
            -amt,
            f"JE-{self.rng.randint(5000, 9999)}",
            "reversal",
        )
        b_r = self._brec(
            self._clamp_to_month(d1 + timedelta(days=self.rng.randint(0, 2))),
            f"ACH CREDIT {wshort}",
            -amt,
        )
        # correct
        d2 = self._clamp_to_month(d1 + timedelta(days=self.rng.randint(1, 3)))
        g_c = self._grec(
            d2,
            f"{rfull} - Corrected payment (replaces erroneous to {wfull})",
            amt,
            self._ref("AP"),
        )
        b_c = self._brec(
            self._clamp_to_month(d2 + timedelta(days=self.rng.randint(0, 2))),
            f"ACH DEBIT {rshort}",
            amt,
        )

        self._gl.extend([g_w, g_r, g_c])
        self._bank.extend([b_w, b_r, b_c])
        for b, g, note in [
            (b_w, g_w, f"Wrong pmt to {wfull}"),
            (b_r, g_r, f"Reversal from {wfull}"),
            (b_c, g_c, f"Correct pmt to {rfull}"),
        ]:
            self._m1.append(
                {
                    "bank_id": b["id"],
                    "gl_id": g["id"],
                    "note": f"Self-correcting: {note}, ${abs(amt):,.2f}",
                }
            )

    def _gen_nsf(self):
        """Deposit + NSF return (2 bank, 2 GL)."""
        v = self._vendor("customer_payment")
        full, short, _ = v
        amt = abs(self._amount("customer_payment"))
        dd = self._date()

        g_dep = self._grec(
            dd, f"{full} - Payment received, {self._ref('INV')}", amt, self._ref("AR")
        )
        b_dep = self._brec(
            self._clamp_to_month(dd + timedelta(days=self.rng.randint(0, 1))),
            "DEPOSIT",
            amt,
        )
        rd = self._clamp_to_month(dd + timedelta(days=self.rng.randint(3, 10)))
        g_ret = self._grec(
            rd,
            f"{full} - NSF return of deposited check",
            -amt,
            f"JE-{self.rng.randint(5000, 9999)}",
        )
        b_ret = self._brec(
            self._clamp_to_month(rd + timedelta(days=self.rng.randint(0, 1))),
            "RETURNED ITEM - NSF",
            -amt,
        )

        self._gl.extend([g_dep, g_ret])
        self._bank.extend([b_dep, b_ret])
        self._m1.append(
            {
                "bank_id": b_dep["id"],
                "gl_id": g_dep["id"],
                "note": f"NSF deposit from {full}, ${amt:,.2f}",
            }
        )
        self._m1.append(
            {
                "bank_id": b_ret["id"],
                "gl_id": g_ret["id"],
                "note": f"NSF return from {full}, -${amt:,.2f}",
            }
        )

    def _gen_amount_mismatch(self):
        """Wire with small fee netted on GL side, OR a transposition error."""
        v = self._vendor("customer_payment")
        full, short, cat = v

        if self.rng.random() < 0.5:
            # Fee mismatch
            bank_amt = abs(self._amount(cat))
            fee = round(self.rng.uniform(10, 50), 2)
            gl_amt = round(bank_amt - fee, 2)
            diff = fee
            note_str = f"Wire from {full}: bank ${bank_amt:,.2f} vs GL ${gl_amt:,.2f} (${fee:.2f} fee)"
        else:
            # Transposition error mismatch
            gl_amt = self._amount(cat)

            # Generate transposed bank_amt
            s = f"{abs(gl_amt):.2f}"
            chars = list(s)
            valid_indices = [
                i
                for i in range(len(chars) - 1)
                if chars[i] != "." and chars[i + 1] != "." and chars[i] != chars[i + 1]
            ]
            if valid_indices:
                idx = self.rng.choice(valid_indices)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            new_amt = float("".join(chars))
            bank_amt = new_amt if gl_amt >= 0 else -new_amt

            diff = round(bank_amt - gl_amt, 2)
            note_str = (
                f"Transposition from {full}: bank ${bank_amt:,.2f} vs GL ${gl_amt:,.2f}"
            )

        gd = self._date()
        bd = self._clamp_to_month(self._shift_weekdays(gd, 0, 3))

        is_credit = bank_amt > 0
        wref = self.rng.randint(1_000_000, 9_999_999)
        g = self._grec(
            gd,
            f"{full} - Customer payment, {self._ref('AR')}",
            gl_amt,
            self._ref("AR"),
        )
        b = self._brec(bd, f"WIRE TRF IN {wref} {short[:15]}", bank_amt)
        self._gl.append(g)
        self._bank.append(b)
        self._ma.append(
            {
                "bank_id": b["id"],
                "gl_id": g["id"],
                "bank_amount": bank_amt,
                "gl_amount": gl_amt,
                "difference": diff,
                "note": note_str,
            }
        )

    def _gen_dup_vendor_pmt(self):
        """Same vendor, same amount, two check numbers."""
        v = self._vendor("professional_svc")
        full, short, _ = v
        amt = -abs(self._amount("professional_svc"))
        d1 = self._date()
        d2 = self._clamp_to_month(d1 + timedelta(days=self.rng.randint(3, 10)))

        c1, c2 = self._chk(), self._chk()
        detail = self.rng.choice(_DETAIL_PHRASES["professional_svc"])
        g1 = self._grec(d1, f"{full} - {detail}, {self._ref('INV')}", amt, f"CHK-{c1}")
        b1 = self._brec(
            self._clamp_to_month(d1 + timedelta(days=self.rng.randint(0, 3))),
            f"CHECK #{c1}",
            amt,
        )
        g2 = self._grec(
            d2, f"{full} - {detail} (duplicate pmt, see AP note)", amt, f"CHK-{c2}"
        )
        b2 = self._brec(
            self._clamp_to_month(d2 + timedelta(days=self.rng.randint(0, 3))),
            f"CHECK #{c2}",
            amt,
        )

        self._gl.extend([g1, g2])
        self._bank.extend([b1, b2])
        self._m1.append(
            {
                "bank_id": b1["id"],
                "gl_id": g1["id"],
                "note": f"Check #{c1}, {full}, ${abs(amt):,.2f} (dup vendor pmt #1)",
            }
        )
        self._m1.append(
            {
                "bank_id": b2["id"],
                "gl_id": g2["id"],
                "note": f"Check #{c2}, {full}, ${abs(amt):,.2f} (dup vendor pmt #2)",
            }
        )

    def _gen_unmatched_bank(self):
        """Bank fee / interest with no GL counterpart."""
        desc, lo, hi = self.rng.choice(_UNMATCHED_BANK_TYPES)
        amt = round(self.rng.uniform(min(lo, hi), max(lo, hi)), 2)
        b = self._brec(self._date(), desc, amt)
        self._bank.append(b)
        self._ub.append(
            {
                "bank_id": b["id"],
                "amount": amt,
                "note": f"{desc} - not yet recorded in GL",
            }
        )

    def _gen_unmatched_gl(self):
        """Outstanding check or deposit in transit."""
        v = self._vendor()
        full, short, cat = v
        amt = self._amount(cat)
        tag = "Deposit in transit" if amt > 0 else "Outstanding check"
        g = self._grec(
            self._date(),
            f"{full} - {tag}, {self._ref('INV')}",
            amt,
            self._gl_ref(cat, amt > 0),
        )
        self._gl.append(g)
        self._ug.append(
            {
                "gl_id": g["id"],
                "amount": amt,
                "note": f"{tag} - {full}, not on bank statement",
            }
        )

    def _gen_offsetting_gl(self):
        """GL void pair that nets to zero, no bank counterpart."""
        v = self._vendor("professional_svc")
        full, short, _ = v
        amt = -abs(self._amount("professional_svc"))
        gd = self._date()
        chk = self._chk()
        g1 = self._grec(
            gd,
            f"{full} - {self.rng.choice(_DETAIL_PHRASES['professional_svc'])}",
            amt,
            f"CHK-{chk}",
        )
        g2 = self._grec(
            self._clamp_to_month(gd + timedelta(days=self.rng.randint(1, 5))),
            f"VOID - Payment to {full} (cancelled)",
            -amt,
            f"CHK-{chk}-V",
            "void",
        )
        self._gl.extend([g1, g2])
        self._off.append(
            {
                "gl_ids": [g1["id"], g2["id"]],
                "amounts": {g1["id"]: amt, g2["id"]: -amt},
                "note": f"Cancelled payment to {full}, nets to zero",
            }
        )

    def _gen_prior_period(self):
        """GL dated in prior month, bank clears current month."""
        v = self._vendor()
        full, short, cat = v
        amt = self._amount(cat)
        is_credit = amt > 0

        # GL: last 5 days of prior month
        if self.month == 1:
            pm, py = 12, self.year - 1
        else:
            pm, py = self.month - 1, self.year
        last_day = (date(py, pm + 1, 1) - timedelta(days=1)).day if pm < 12 else 31
        gl_d = date(py, pm, self.rng.randint(max(1, last_day - 4), last_day))
        bank_d = date(self.year, self.month, self.rng.randint(1, 5))

        b = self._brec(bank_d, self._bank_desc(short, cat, is_credit), amt)
        g = self._grec(
            gl_d, self._gl_desc(full, cat), amt, self._gl_ref(cat, is_credit)
        )
        self._bank.append(b)
        self._gl.append(g)
        self._m1.append(
            {
                "bank_id": b["id"],
                "gl_id": g["id"],
                "note": f"Prior period: GL {gl_d} -> bank {bank_d}, {full}",
            }
        )

    # ------------------------------------------------------------------
    # Difficulty modifiers (generate paired simple events)
    # ------------------------------------------------------------------

    def _gen_dup_amount_pairs(self, n_pairs: int):
        """Pairs with same amount, different vendors, different dates."""
        for _ in range(n_pairs):
            cat = self._random_category()
            amt = self._amount(cat, unique=False)
            v1, v2 = self._vendor(cat), self._vendor(cat)
            d1 = self._date()
            offset = self.rng.randint(5, 15)
            d2 = d1 + timedelta(days=offset)
            d2 = self._clamp_to_month(d2)
            if d2 == d1:
                d2 = self._clamp_to_month(d1 - timedelta(days=offset))
            self._gen_simple(
                vendor=v1, amount=amt, gl_date=d1, note_extra="dup amount pair"
            )
            self._gen_simple(
                vendor=v2, amount=amt, gl_date=d2, note_extra="dup amount pair"
            )

    def _gen_same_amt_date_pairs(self, n_pairs: int):
        """Pairs with same amount AND date — description is only signal."""
        for _ in range(n_pairs):
            cat = "professional_svc"
            amt = self._amount(cat, unique=False)
            d = self._date()
            pair = self._confusable_pair(cat)
            self._gen_simple(
                vendor=pair[0],
                amount=amt,
                gl_date=d,
                bank_date=d,
                note_extra="same amt+date, description only signal",
            )
            self._gen_simple(
                vendor=pair[1],
                amount=amt,
                gl_date=d,
                bank_date=d,
                note_extra="same amt+date, description only signal",
            )

    def _gen_weekly_series_batch(self, n_series: int):
        """Generates sequences of 4 weekly payments to the same vendor."""
        cats = [
            "payroll",
            "subscription",
            "rent",
            "professional_svc",
            "utilities",
            "customer_payment",
        ]
        for _ in range(n_series):
            cat = self.rng.choice(cats)
            v = self._vendor(cat)
            base_amt = self._amount(cat, unique=False)
            is_fixed = (
                self.rng.random() < 0.6
            )  # 60% exact same amount, 40% slightly varied

            # Start in the first 7 days
            start_d = date(self.year, self.month, self.rng.randint(1, 7))

            for w in range(4):
                d = start_d + timedelta(days=w * 7)

                amt = base_amt
                if not is_fixed:
                    variance = base_amt * self.rng.uniform(-0.05, 0.05)
                    amt = round(base_amt + variance, 2)
                    if amt == 0:
                        amt = base_amt

                # Bank date is typically 0-2 days after GL date
                bd = self._clamp_to_month(self._shift_weekdays(d, 0, 2))

                self._gen_simple(
                    vendor=v,
                    amount=amt,
                    gl_date=d,
                    bank_date=bd,
                    note_extra=f"weekly recurring {w + 1}/4",
                )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self) -> dict:
        p = self.p
        n = self.n_bank_target

        # Bank-record multipliers per event type
        mults = {
            "simple": 1,
            "batch_deposit": 1,
            "void_reissue": 1,
            "self_correcting": 3,
            "nsf_return": 2,
            "amount_mismatch": 1,
            "dup_vendor_pmt": 2,
            "unmatched_bank": 1,
        }
        rates = {
            "simple": p.simple,
            "batch_deposit": p.batch_deposit,
            "void_reissue": p.void_reissue,
            "self_correcting": p.self_correcting,
            "nsf_return": p.nsf_return,
            "amount_mismatch": p.amount_mismatch,
            "dup_vendor_pmt": p.dup_vendor_pmt,
            "unmatched_bank": p.unmatched_bank,
        }
        # bank_per_n_events: expected bank records per unit of n_events
        # Since counts[k] = round(n_events * rate_k), total bank records ≈
        # n_events * sum(rate_k * mult_k).  Solve for n_events.
        bank_per_n = sum(rates[k] * mults[k] for k in rates)
        n_events = round(n / bank_per_n)

        counts = {k: max(1, round(n_events * v)) for k, v in rates.items()}
        counts["unmatched_gl"] = max(1, round(n_events * p.unmatched_gl))
        counts["offsetting_gl"] = max(1, round(n_events * p.offsetting_gl))
        counts["prior_period"] = max(1, round(counts["simple"] * p.prior_period_frac))

        # Pairs generated by modifiers (each pair = 2 simple events)
        n_dup_pairs = max(0, round(counts["simple"] * p.dup_amount_frac / 2))
        n_same_pairs = max(0, round(counts["simple"] * p.same_amt_date_frac / 2))
        n_weekly_series = max(0, round(counts["simple"] * p.weekly_recurring_frac / 4))

        modifier_events = (
            n_dup_pairs * 2
            + n_same_pairs * 2
            + n_weekly_series * 4
            + counts["prior_period"]
        )
        counts["simple"] = max(1, counts["simple"] - modifier_events)

        # Generate each event type
        for _ in range(counts["simple"]):
            self._gen_simple()
        for _ in range(counts["batch_deposit"]):
            self._gen_batch_deposit()
        for _ in range(counts["void_reissue"]):
            self._gen_void_reissue()
        for _ in range(counts["self_correcting"]):
            self._gen_self_correcting()
        for _ in range(counts["nsf_return"]):
            self._gen_nsf()
        for _ in range(counts["amount_mismatch"]):
            self._gen_amount_mismatch()
        for _ in range(counts["dup_vendor_pmt"]):
            self._gen_dup_vendor_pmt()
        for _ in range(counts["unmatched_bank"]):
            self._gen_unmatched_bank()
        for _ in range(counts["unmatched_gl"]):
            self._gen_unmatched_gl()
        for _ in range(counts["offsetting_gl"]):
            self._gen_offsetting_gl()
        for _ in range(counts["prior_period"]):
            self._gen_prior_period()

        # Difficulty modifiers (generate additional simple events)
        self._gen_dup_amount_pairs(n_dup_pairs)
        self._gen_same_amt_date_pairs(n_same_pairs)
        self._gen_weekly_series_batch(n_weekly_series)

        # Sort by date, re-assign sequential IDs
        self._bank.sort(key=lambda r: r["date"])
        self._gl.sort(key=lambda r: r["date"])

        bw = max(2, len(str(len(self._bank))))
        gw = max(2, len(str(len(self._gl))))
        idmap: dict[str, str] = {}
        for i, r in enumerate(self._bank, 1):
            new = f"B{i:0{bw}d}"
            idmap[r["id"]] = new
            r["id"] = new
        for i, r in enumerate(self._gl, 1):
            new = f"G{i:0{gw}d}"
            idmap[r["id"]] = new
            r["id"] = new

        # Remap IDs in ground truth
        self._remap(idmap)

        summary = self._summary(counts, n_dup_pairs, n_same_pairs, n_weekly_series)
        return {
            "bank_transactions": self._bank,
            "gl_entries": self._gl,
            "ground_truth": {
                "matches_1to1": self._m1,
                "matches_many_to_one": self._mm,
                "matches_amount_mismatch": self._ma,
                "offsetting_gl_pairs": self._off,
                "unmatched_bank": self._ub,
                "unmatched_gl": self._ug,
                "summary": summary,
            },
        }

    def _remap(self, m: dict[str, str]):
        for e in self._m1:
            e["bank_id"] = m.get(e["bank_id"], e["bank_id"])
            e["gl_id"] = m.get(e["gl_id"], e["gl_id"])
        for e in self._mm:
            e["bank_id"] = m.get(e["bank_id"], e["bank_id"])
            e["gl_ids"] = [m.get(x, x) for x in e["gl_ids"]]
            e["gl_amounts"] = {m.get(k, k): v for k, v in e["gl_amounts"].items()}
        for e in self._ma:
            e["bank_id"] = m.get(e["bank_id"], e["bank_id"])
            e["gl_id"] = m.get(e["gl_id"], e["gl_id"])
        for e in self._off:
            e["gl_ids"] = [m.get(x, x) for x in e["gl_ids"]]
            e["amounts"] = {m.get(k, k): v for k, v in e["amounts"].items()}
        for e in self._ub:
            e["bank_id"] = m.get(e["bank_id"], e["bank_id"])
        for e in self._ug:
            e["gl_id"] = m.get(e["gl_id"], e["gl_id"])

    def _summary(self, counts, n_dup_pairs, n_same_pairs, n_weekly_series) -> dict:
        nb = len(self._bank)
        ng = len(self._gl)
        n1 = len(self._m1)
        nmm = len(self._mm)
        nmm_gl = sum(len(e["gl_ids"]) for e in self._mm)
        nma = len(self._ma)
        noff = len(self._off)
        noff_gl = noff * 2
        nub = len(self._ub)
        nug = len(self._ug)
        return {
            "bank_transactions": nb,
            "gl_entries": ng,
            "matches_1to1": n1,
            "matches_many_to_one": nmm,
            "matches_many_to_one_gl_entries": nmm_gl,
            "matches_amount_mismatch": nma,
            "offsetting_gl_pairs": noff,
            "offsetting_gl_entries": noff_gl,
            "unmatched_bank": nub,
            "unmatched_gl": nug,
            "gl_accounted": n1 + nmm_gl + nma + noff_gl + nug,
            "bank_accounted": n1 + nmm + nma + nub,
            "event_counts": counts,
            "dup_amount_pairs": n_dup_pairs,
            "same_amt_date_pairs": n_same_pairs,
            "weekly_recurring_series": n_weekly_series,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def generate(
    n_bank: int = 1000,
    seed: int = 42,
    difficulty: str | DifficultyProfile = "medium",
    year: int = 2025,
    month: int = 1,
) -> dict:
    """Generate a synthetic bank reconciliation problem.

    Args:
        n_bank:     Target number of bank-side transactions (approximate).
        seed:       Random seed for reproducibility.
        difficulty: ``"easy"``, ``"medium"``, ``"hard"``, or a
                    :class:`DifficultyProfile` instance.
        year:       Year for generated dates.
        month:      Month for generated dates (1-12).

    Returns:
        dict with keys ``bank_transactions``, ``gl_entries``, ``ground_truth``.
        The ``ground_truth`` dict has the same structure as
        :data:`examples.bank_rec_problem.GROUND_TRUTH`.
    """
    if isinstance(difficulty, str):
        profile = {"easy": EASY, "medium": MEDIUM, "hard": HARD}[difficulty]
    else:
        profile = difficulty
    return _Generator(n_bank, seed, profile, year, month).run()


def verify(result: dict) -> list[str]:
    """Check ground-truth consistency.  Returns list of error strings (empty = OK)."""
    errors: list[str] = []
    gt = result["ground_truth"]
    bank_ids = {r["id"] for r in result["bank_transactions"]}
    gl_ids = {r["id"] for r in result["gl_entries"]}

    # Collect all referenced IDs
    seen_bank: list[str] = []
    seen_gl: list[str] = []

    for e in gt["matches_1to1"]:
        seen_bank.append(e["bank_id"])
        seen_gl.append(e["gl_id"])
    for e in gt["matches_many_to_one"]:
        seen_bank.append(e["bank_id"])
        seen_gl.extend(e["gl_ids"])
    for e in gt["matches_amount_mismatch"]:
        seen_bank.append(e["bank_id"])
        seen_gl.append(e["gl_id"])
    for e in gt["offsetting_gl_pairs"]:
        seen_gl.extend(e["gl_ids"])
    for e in gt["unmatched_bank"]:
        seen_bank.append(e["bank_id"])
    for e in gt["unmatched_gl"]:
        seen_gl.append(e["gl_id"])

    # Every referenced ID must exist in the data
    for bid in seen_bank:
        if bid not in bank_ids:
            errors.append(
                f"Ground truth references bank_id {bid} not in bank_transactions"
            )
    for gid in seen_gl:
        if gid not in gl_ids:
            errors.append(f"Ground truth references gl_id {gid} not in gl_entries")

    # Every bank/GL record should appear exactly once in ground truth
    bank_counts: dict[str, int] = {}
    for bid in seen_bank:
        bank_counts[bid] = bank_counts.get(bid, 0) + 1
    for bid in bank_ids:
        c = bank_counts.get(bid, 0)
        if c == 0:
            errors.append(f"Bank {bid} not accounted for in ground truth")
        elif c > 1:
            errors.append(f"Bank {bid} appears {c} times in ground truth")

    gl_counts: dict[str, int] = {}
    for gid in seen_gl:
        gl_counts[gid] = gl_counts.get(gid, 0) + 1
    for gid in gl_ids:
        c = gl_counts.get(gid, 0)
        if c == 0:
            errors.append(f"GL {gid} not accounted for in ground truth")
        elif c > 1:
            errors.append(f"GL {gid} appears {c} times in ground truth")

    # Summary counts should match
    s = gt["summary"]
    if s["bank_transactions"] != len(bank_ids):
        errors.append(
            f"Summary bank count {s['bank_transactions']} != actual {len(bank_ids)}"
        )
    if s["gl_entries"] != len(gl_ids):
        errors.append(f"Summary GL count {s['gl_entries']} != actual {len(gl_ids)}")
    if s["bank_accounted"] != len(bank_ids):
        errors.append(
            f"bank_accounted {s['bank_accounted']} != actual bank count {len(bank_ids)}"
        )
    if s["gl_accounted"] != len(gl_ids):
        errors.append(
            f"gl_accounted {s['gl_accounted']} != actual GL count {len(gl_ids)}"
        )

    # Batch deposit sums should match
    bank_by_id = {r["id"]: r for r in result["bank_transactions"]}
    gl_by_id = {r["id"]: r for r in result["gl_entries"]}
    for e in gt["matches_many_to_one"]:
        gl_sum = round(sum(gl_by_id[gid]["amount"] for gid in e["gl_ids"]), 2)
        bank_amt = bank_by_id[e["bank_id"]]["amount"]
        if abs(gl_sum - bank_amt) > 0.01:
            errors.append(f"Batch {e['bank_id']}: GL sum {gl_sum} != bank {bank_amt}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# CLI smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time

    sizes = [100, 1_000, 5_000, 10_000] if "--full" in sys.argv else [1_000]
    for n in sizes:
        for diff in ["easy", "medium", "hard"]:
            t0 = time.perf_counter()
            result = generate(n_bank=n, seed=42, difficulty=diff)
            elapsed = time.perf_counter() - t0
            errs = verify(result)
            s = result["ground_truth"]["summary"]
            status = "OK" if not errs else f"ERRORS: {len(errs)}"
            print(
                f"n={n:>6d}  diff={diff:<6s}  bank={s['bank_transactions']:>6d}  "
                f"gl={s['gl_entries']:>6d}  1:1={s['matches_1to1']:>5d}  "
                f"batch={s['matches_many_to_one']:>3d}  "
                f"mismatch={s['matches_amount_mismatch']:>3d}  "
                f"offset={s['offsetting_gl_pairs']:>3d}  "
                f"ub={s['unmatched_bank']:>3d}  ug={s['unmatched_gl']:>3d}  "
                f"{elapsed:.2f}s  [{status}]"
            )
            if errs:
                for e in errs[:10]:
                    print(f"  ! {e}")
