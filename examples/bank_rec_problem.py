"""Bank Reconciliation Problem --- January 2025

A mid-size company's monthly bank reconciliation for the operating account.
43 bank statement transactions, 49 general ledger entries.

This is DATA ONLY --- no solution logic. Import the data and build a solver
separately.

Difficulty features:
  - 8 pairs share the same amount on the same/close dates (description-only signal)
  - 2 batch deposits (one bank line = sum of multiple GL entries)
  - 1 amount mismatch ($75,000 bank vs $74,975 GL --- wire fee netted by accountant)
  - Self-correcting entries (wrong payment -> reversal -> correct payment, all $15,340)
  - Voided and reissued check ($4,200 appears 3x in GL, 1x on bank)
  - Duplicate vendor payment (same amount, same vendor, 4 days apart)
  - Prior-period outstanding check (GL dated Dec 30, clears bank Jan 3 --- 4-day gap)
  - NSF returned check (+$8,200 deposit then -$8,200 return)
  - Cryptic bank descriptions (AMZN MKTP US*..., ADP TOFCPYRL ..., PAYPAL *JRODRIGU)
  - Truncated entity names (BRIGHTSTAR ENTE, HARBOR REALTY PRTNRS, TECHFLOW SOLUTNS)
  - 3 "Johnson" entities with similar names and overlapping amounts
  - 3 bank items with no GL counterpart (wire fee, service charge, interest)
  - 2 GL items with no bank counterpart (outstanding check, deposit in transit)

Usage:
    from examples.bank_rec_problem import BANK_TRANSACTIONS, GL_ENTRIES, GROUND_TRUTH
"""

from datetime import date


# ===========================================================================
# Bank Statement Transactions
# ===========================================================================
# As they appear on the bank statement: terse, truncated, sometimes cryptic.
# Fields: id, date, description, amount (positive = deposit, negative = withdrawal)

BANK_TRANSACTIONS = [
    # --- Rent (2 payments, identical amount, same date) ---
    {
        "id": "B01",
        "date": date(2025, 1, 2),
        "description": "ACH DEBIT PINNACLE PROP MGMT",
        "amount": -8_500.00,
    },
    {
        "id": "B02",
        "date": date(2025, 1, 2),
        "description": "ACH DEBIT HARBOR REALTY PRTNRS",
        "amount": -8_500.00,
    },
    # --- Prior-period outstanding check (GL date: Dec 30, 4-day gap) ---
    {
        "id": "B03",
        "date": date(2025, 1, 3),
        "description": "CHECK #4518",
        "amount": -6_780.00,
    },
    # --- Customer payment #1 (TechFlow, same amount as B36) ---
    {
        "id": "B04",
        "date": date(2025, 1, 3),
        "description": "ACH CREDIT TECHFLOW SOLUTNS",
        "amount": 32_000.00,
    },
    # --- Wrong-vendor payment (will be reversed in B08) ---
    {
        "id": "B05",
        "date": date(2025, 1, 5),
        "description": "ACH DEBIT SUMMIT SUPPLY CO",
        "amount": -15_340.00,
    },
    # --- Amazon purchase #1 (cryptic, same amount as B26) ---
    {
        "id": "B06",
        "date": date(2025, 1, 6),
        "description": "AMZN MKTP US*RT4K29ZQ1 AMZN.CO",
        "amount": -1_247.53,
    },
    # --- Staples POS (same amount as B30 at Office Depot) ---
    {
        "id": "B07",
        "date": date(2025, 1, 7),
        "description": "POS PURCHASE STAPLES #4422",
        "amount": -892.15,
    },
    # --- Reversal of wrong payment B05 ---
    {
        "id": "B08",
        "date": date(2025, 1, 7),
        "description": "ACH CREDIT SUMMIT SUPPLY CO",
        "amount": 15_340.00,
    },
    # --- Correct payment (same absolute amount as B05 and B08) ---
    {
        "id": "B09",
        "date": date(2025, 1, 8),
        "description": "ACH DEBIT NORTHSTAR MFG INC",
        "amount": -15_340.00,
    },
    # --- Johnson #1 (same amount as B11, same date!) ---
    {
        "id": "B10",
        "date": date(2025, 1, 8),
        "description": "ACH DEBIT JOHNSON & ASSOC",
        "amount": -5_500.00,
    },
    # --- Johnson #2 (same amount as B10, same date!) ---
    {
        "id": "B11",
        "date": date(2025, 1, 8),
        "description": "ACH DEBIT K JOHNSON CONS GRP",
        "amount": -5_500.00,
    },
    # --- Large wire from customer ---
    {
        "id": "B12",
        "date": date(2025, 1, 9),
        "description": "WIRE TRF IN 8837462 ACME COR",
        "amount": 150_000.00,
    },
    # --- Wire fee (NO GL counterpart) ---
    {
        "id": "B13",
        "date": date(2025, 1, 9),
        "description": "WIRE TRF FEE",
        "amount": -25.00,
    },
    # --- Customer check deposit (will bounce as NSF, see B24) ---
    {
        "id": "B14",
        "date": date(2025, 1, 10),
        "description": "DEPOSIT",
        "amount": 8_200.00,
    },
    # --- Microsoft Azure (cryptic bank description) ---
    {
        "id": "B15",
        "date": date(2025, 1, 10),
        "description": "ACH DEBIT MSFT *AZURE 012025",
        "amount": -4_873.29,
    },
    # --- Johnson #3 (different amount from B10/B11, similar name) ---
    {
        "id": "B16",
        "date": date(2025, 1, 10),
        "description": "ACH DEBIT JOHNSON ASSOC LLC",
        "amount": -5_750.00,
    },
    # --- Delta Air #1 (same amount as B37) ---
    {
        "id": "B17",
        "date": date(2025, 1, 12),
        "description": "ACH DEBIT DELTA AIR 0062847",
        "amount": -2_847.60,
    },
    # --- Duplicate vendor payment (check #1, same vendor/amount as B19) ---
    {
        "id": "B18",
        "date": date(2025, 1, 12),
        "description": "CHECK #4519",
        "amount": -3_400.00,
    },
    # --- Duplicate vendor payment (check #2) ---
    {
        "id": "B19",
        "date": date(2025, 1, 12),
        "description": "CHECK #4520",
        "amount": -3_400.00,
    },
    # --- PayPal freelancer (truncated username) ---
    {
        "id": "B20",
        "date": date(2025, 1, 13),
        "description": "PAYPAL *JRODRIGU",
        "amount": -4_500.00,
    },
    # --- Reissued check (original #4515 voided, reissued as #4522) ---
    {
        "id": "B21",
        "date": date(2025, 1, 14),
        "description": "CHECK #4522",
        "amount": -4_200.00,
    },
    # --- BATCH DEPOSIT #1 (5 customer payments lumped into one line) ---
    {
        "id": "B22",
        "date": date(2025, 1, 15),
        "description": "DEPOSIT",
        "amount": 47_250.00,
    },
    # --- Payroll #1 (same amount as B40) ---
    {
        "id": "B23",
        "date": date(2025, 1, 15),
        "description": "ADP TOFCPYRL 011525 XXXXXX7890",
        "amount": -86_500.00,
    },
    # --- NSF return of check deposited in B14 ---
    {
        "id": "B24",
        "date": date(2025, 1, 15),
        "description": "RETURNED ITEM - Loss Draft",
        "amount": -8_200.00,
    },
    # --- Supplier payment (Globex, not to be confused with wire B28) ---
    {
        "id": "B25",
        "date": date(2025, 1, 16),
        "description": "ACH DEBIT GLOBEX INDUSTRIES",
        "amount": -12_500.00,
    },
    # --- Amazon purchase #2 (cryptic, same amount as B06) ---
    {
        "id": "B26",
        "date": date(2025, 1, 17),
        "description": "AMZN MKTP US*M87JQ2P15 AMZN.C",
        "amount": -1_247.53,
    },
    # --- Customer payment ---
    {
        "id": "B27",
        "date": date(2025, 1, 17),
        "description": "ACH CREDIT WESTFIELD CORP",
        "amount": 25_000.00,
    },
    # --- Wire from Globex customer (GL records $74,975 --- $25 wire fee netted!) ---
    {
        "id": "B28",
        "date": date(2025, 1, 19),
        "description": "WIRE TRF IN 8837501 GLOBEX IN",
        "amount": 75_000.00,
    },
    # --- Summit Supply regular order ---
    {
        "id": "B29",
        "date": date(2025, 1, 20),
        "description": "ACH DEBIT SUMMIT SUPPLY CO",
        "amount": -7_650.00,
    },
    # --- Office Depot POS (same amount as B07 at Staples) ---
    {
        "id": "B30",
        "date": date(2025, 1, 21),
        "description": "POS PURCHASE OFFICE DEPOT 881",
        "amount": -892.15,
    },
    # --- Customer payment (truncated entity name) ---
    {
        "id": "B31",
        "date": date(2025, 1, 22),
        "description": "ACH CREDIT BRIGHTSTAR ENTE",
        "amount": 18_500.00,
    },
    # --- Credit card payment ---
    {
        "id": "B32",
        "date": date(2025, 1, 22),
        "description": "ACH DEBIT CHASE CARD SVCS",
        "amount": -23_456.78,
    },
    # --- Pinnacle CAM charges (same vendor as B01 rent, different amount) ---
    {
        "id": "B33",
        "date": date(2025, 1, 23),
        "description": "ACH DEBIT PINNACLE PROP MGMT",
        "amount": -2_150.00,
    },
    # --- Venmo ad-hoc payment ---
    {
        "id": "B34",
        "date": date(2025, 1, 24),
        "description": "VENMO PAYMENT SARAH C",
        "amount": -1_500.00,
    },
    # --- Utilities ---
    {
        "id": "B35",
        "date": date(2025, 1, 24),
        "description": "ACH DEBIT CONSOLIDATED UTIL",
        "amount": -3_842.17,
    },
    # --- Customer payment #2 (TechFlow again, same amount as B04) ---
    {
        "id": "B36",
        "date": date(2025, 1, 25),
        "description": "ACH CREDIT TECHFLOW SOLUTNS",
        "amount": 32_000.00,
    },
    # --- Delta Air #2 (same amount as B17, different trip) ---
    {
        "id": "B37",
        "date": date(2025, 1, 27),
        "description": "ACH DEBIT DELTA AIR 0069921",
        "amount": -2_847.60,
    },
    # --- BATCH DEPOSIT #2 (2 customer payments lumped into one line) ---
    {
        "id": "B38",
        "date": date(2025, 1, 28),
        "description": "DEPOSIT",
        "amount": 15_750.00,
    },
    # --- Insurance ---
    {
        "id": "B39",
        "date": date(2025, 1, 29),
        "description": "ACH DEBIT PREMIUM INS GROUP",
        "amount": -11_200.00,
    },
    # --- Payroll #2 (same amount as B23) ---
    {
        "id": "B40",
        "date": date(2025, 1, 31),
        "description": "ADP TOFCPYRL 013125 XXXXXX7890",
        "amount": -86_500.00,
    },
    # --- Bank service charge (NO GL counterpart) ---
    {
        "id": "B41",
        "date": date(2025, 1, 31),
        "description": "SERVICE CHARGE",
        "amount": -35.00,
    },
    # --- Interest earned (NO GL counterpart) ---
    {
        "id": "B42",
        "date": date(2025, 1, 31),
        "description": "INTEREST PAYMENT",
        "amount": 12.47,
    },
    # --- Sales tax remittance ---
    {
        "id": "B43",
        "date": date(2025, 1, 31),
        "description": "ACH DEBIT STATE TAX AUTH",
        "amount": -4_250.00,
    },
]


# ===========================================================================
# General Ledger Entries (Bank/Cash Account)
# ===========================================================================
# As they appear in the accounting system: verbose, well-documented.
# Fields: id, date, description, amount, ref, entry_type

GL_ENTRIES = [
    # --- Rent for main office (matches B01) ---
    {
        "id": "G01",
        "date": date(2025, 1, 1),
        "description": "Pinnacle Property Management - Rent, Main Office 123 Industrial Pkwy, Jan 2025",
        "amount": -8_500.00,
        "ref": "AP-4401",
        "entry_type": "normal",
    },
    # --- Rent for satellite office (matches B02) ---
    {
        "id": "G02",
        "date": date(2025, 1, 1),
        "description": "Harbor Realty Partners - Rent, Satellite Office 456 Harbor Dr, Jan 2025",
        "amount": -8_500.00,
        "ref": "AP-4402",
        "entry_type": "normal",
    },
    # --- Prior-period check (matches B03, but 4-day gap!) ---
    {
        "id": "G03",
        "date": date(2024, 12, 30),
        "description": "Datalink Corporation - Server maintenance and support Q4 2024",
        "amount": -6_780.00,
        "ref": "CHK-4518",
        "entry_type": "normal",
    },
    # --- Customer receipt #1 (matches B04; same amount/entity as G41) ---
    {
        "id": "G04",
        "date": date(2025, 1, 3),
        "description": "TechFlow Solutions Inc - Invoice #TF-2024-089, Professional services Dec",
        "amount": 32_000.00,
        "ref": "AR-8801",
        "entry_type": "normal",
    },
    # --- WRONG VENDOR PAYMENT (matches B05) ---
    {
        "id": "G05",
        "date": date(2025, 1, 5),
        "description": "Summit Supply Co - Purchase order #8842 (ENTERED IN ERROR - see JE-5510)",
        "amount": -15_340.00,
        "ref": "AP-4403",
        "entry_type": "normal",
    },
    # --- Reversal of wrong payment (matches B08) ---
    {
        "id": "G06",
        "date": date(2025, 1, 7),
        "description": "Summit Supply Co - Reversal of AP-4403 per manager approval JE-5510",
        "amount": 15_340.00,
        "ref": "JE-5510",
        "entry_type": "reversal",
    },
    # --- Correct payment to right vendor (matches B09) ---
    {
        "id": "G07",
        "date": date(2025, 1, 8),
        "description": "Northstar Manufacturing Inc - Purchase order #8842, raw materials",
        "amount": -15_340.00,
        "ref": "AP-4404",
        "entry_type": "normal",
    },
    # --- Amazon office supplies (matches B06; same amount as G31) ---
    {
        "id": "G08",
        "date": date(2025, 1, 5),
        "description": "Amazon Business - Office supplies, toner, copy paper (Order #112-4829374)",
        "amount": -1_247.53,
        "ref": "PO-6601",
        "entry_type": "normal",
    },
    # --- Staples purchase (matches B07; same amount as G35) ---
    {
        "id": "G09",
        "date": date(2025, 1, 7),
        "description": "Staples Store #4422 - Printer ink cartridges and shipping labels",
        "amount": -892.15,
        "ref": "PO-6602",
        "entry_type": "normal",
    },
    # --- Johnson & Associates (matches B10; same amount/date as G11!) ---
    {
        "id": "G10",
        "date": date(2025, 1, 8),
        "description": "Johnson & Associates - Monthly legal retainer January 2025",
        "amount": -5_500.00,
        "ref": "AP-4405",
        "entry_type": "normal",
    },
    # --- K. Johnson Consulting (matches B11; same amount/date as G10!) ---
    {
        "id": "G11",
        "date": date(2025, 1, 8),
        "description": "K. Johnson Consulting Group - IT security audit Phase 2",
        "amount": -5_500.00,
        "ref": "AP-4406",
        "entry_type": "normal",
    },
    # --- Johnson Associates LLC (matches B16; different amount) ---
    {
        "id": "G12",
        "date": date(2025, 1, 10),
        "description": "Johnson Associates LLC - Management consulting  engagement  Q1",
        "amount": -5_750.00,
        "ref": "AP-4407",
        "entry_type": "normal",
    },
    # --- Acme wire (matches B12) ---
    {
        "id": "G13",
        "date": date(2025, 1, 8),
        "description": "Acme Corporation - Q1 prepayment per contract #AC-2024-007",
        "amount": 150_000.00,
        "ref": "AR-8802",
        "entry_type": "normal",
    },
    # --- Microsoft Azure (matches B15) ---
    {
        "id": "G14",
        "date": date(2025, 1, 10),
        "description": "Microsoft Azure - Cloud hosting and compute services January 2025",
        "amount": -4_873.29,
        "ref": "AP-4408",
        "entry_type": "normal",
    },
    # --- Customer check deposit (matches B14; same absolute amount as G29!) ---
    {
        "id": "G15",
        "date": date(2025, 1, 10),
        "description": "Riverside Medical Partners - Payment received, Invoice #RP-2025-441",
        "amount": 8_200.00,
        "ref": "AR-8803",
        "entry_type": "normal",
    },
    # --- Delta flight #1 (matches B17; same amount as G42) ---
    {
        "id": "G16",
        "date": date(2025, 1, 11),
        "description": "Delta Air Lines - SFO to JFK round trip, J. Smith, Conf #DL8834",
        "amount": -2_847.60,
        "ref": "EXP-7701",
        "entry_type": "normal",
    },
    # --- Meridian check #4519 (matches B18; same amount as G18) ---
    {
        "id": "G17",
        "date": date(2025, 1, 8),
        "description": "Meridian Services Group - Invoice #MS-2025-003, HVAC quarterly maintenance",
        "amount": -3_400.00,
        "ref": "CHK-4519",
        "entry_type": "normal",
    },
    # --- Meridian check #4520 DUPLICATE (matches B19; same amount as G17) ---
    {
        "id": "G18",
        "date": date(2025, 1, 12),
        "description": "Meridian Services Group - Invoice #MS-2025-003 (duplicate pmt, see AP note)",
        "amount": -3_400.00,
        "ref": "CHK-4520",
        "entry_type": "normal",
    },
    # --- PayPal freelancer (matches B20) ---
    {
        "id": "G19",
        "date": date(2025, 1, 13),
        "description": "PayPal - John Rodriguez, Freelance web development Phase 3",
        "amount": -4_500.00,
        "ref": "AP-4410",
        "entry_type": "normal",
    },
    # --- VOIDED CHECK #4515 original entry (OFFSETTING with G21) ---
    {
        "id": "G20",
        "date": date(2025, 1, 5),
        "description": "Apex Design Group - Logo redesign project, Invoice #ADG-2025-100",
        "amount": -4_200.00,
        "ref": "CHK-4515",
        "entry_type": "normal",
    },
    # --- VOID of check #4515 (OFFSETTING with G20) ---
    {
        "id": "G21",
        "date": date(2025, 1, 10),
        "description": "VOID - Check #4515 to Apex Design Group (incorrect mailing address)",
        "amount": 4_200.00,
        "ref": "CHK-4515-V",
        "entry_type": "void",
    },
    # --- Reissued check #4522 (matches B21) ---
    {
        "id": "G22",
        "date": date(2025, 1, 12),
        "description": "Apex Design Group - Logo redesign (reissued check, replaces voided #4515)",
        "amount": -4_200.00,
        "ref": "CHK-4522",
        "entry_type": "normal",
    },
    # --- Batch deposit #1 component 1/5 ---
    {
        "id": "G23",
        "date": date(2025, 1, 14),
        "description": "Cascade Industries - Invoice #CI-2025-880, Equipment parts",
        "amount": 12_000.00,
        "ref": "AR-8804",
        "entry_type": "normal",
    },
    # --- Batch deposit #1 component 2/5 ---
    {
        "id": "G24",
        "date": date(2025, 1, 14),
        "description": "Mountain View Technology - Invoice #MVT-2025-015, Annual support",
        "amount": 8_750.00,
        "ref": "AR-8805",
        "entry_type": "normal",
    },
    # --- Batch deposit #1 component 3/5 ---
    {
        "id": "G25",
        "date": date(2025, 1, 14),
        "description": "Sterling & Partners LLP - Retainer prepayment February 2025",
        "amount": 15_000.00,
        "ref": "AR-8806",
        "entry_type": "normal",
    },
    # --- Batch deposit #1 component 4/5 (note: same amount as G44!) ---
    {
        "id": "G26",
        "date": date(2025, 1, 14),
        "description": "Oakwood Supplies - Credit memo #CM-2025-442, returned goods",
        "amount": 6_500.00,
        "ref": "AR-8807",
        "entry_type": "normal",
    },
    # --- Batch deposit #1 component 5/5 ---
    {
        "id": "G27",
        "date": date(2025, 1, 14),
        "description": "Redline Logistics Corp - Invoice #RL-2025-219, Freight charges",
        "amount": 5_000.00,
        "ref": "AR-8808",
        "entry_type": "normal",
    },
    # --- Payroll #1 (matches B23; same amount as G46) ---
    {
        "id": "G28",
        "date": date(2025, 1, 15),
        "description": "ADP Payroll - Net pay, period 01/01-01/15/2025, 47 employees",
        "amount": -86_500.00,
        "ref": "PR-2025-01A",
        "entry_type": "normal",
    },
    # --- NSF return (matches B24; same absolute amount as G15!) ---
    {
        "id": "G29",
        "date": date(2025, 1, 15),
        "description": "Riverside Medical Partners - NSF return of deposited check, Inv #RP-2025-441",
        "amount": -8_200.00,
        "ref": "JE-5515",
        "entry_type": "normal",
    },
    # --- Globex supplier payment (matches B25) ---
    {
        "id": "G30",
        "date": date(2025, 1, 16),
        "description": "Globex Industries Ltd - Raw materials, PO #GM-7756",
        "amount": -12_500.00,
        "ref": "AP-4412",
        "entry_type": "normal",
    },
    # --- Amazon lab equipment (matches B26; same amount as G08!) ---
    {
        "id": "G31",
        "date": date(2025, 1, 17),
        "description": "Amazon Business - Laboratory equipment and safety supplies (Order #305-5019283)",
        "amount": -1_247.53,
        "ref": "PO-6605",
        "entry_type": "normal",
    },
    # --- Westfield customer receipt (matches B27) ---
    {
        "id": "G32",
        "date": date(2025, 1, 17),
        "description": "Westfield Corporation - Invoice #WC-2025-011, Consulting engagement",
        "amount": 25_000.00,
        "ref": "AR-8810",
        "entry_type": "normal",
    },
    # --- Globex wire AMOUNT MISMATCH (matches B28, but $74,975 vs $75,000!) ---
    {
        "id": "G33",
        "date": date(2025, 1, 19),
        "description": "Globex Industries Ltd - Customer prepayment Q1 (net of $25 incoming wire fee per controller memo)",
        "amount": 74_975.00,
        "ref": "AR-8811",
        "entry_type": "normal",
    },
    # --- Summit Supply regular order (matches B29) ---
    {
        "id": "G34",
        "date": date(2025, 1, 20),
        "description": "Summit Supply Co - PO #9015, Monthly maintenance supplies",
        "amount": -7_650.00,
        "ref": "AP-4413",
        "entry_type": "normal",
    },
    # --- Office Depot (matches B30; same amount as G09!) ---
    {
        "id": "G35",
        "date": date(2025, 1, 21),
        "description": "Office Depot - Desk, chair, and accessories for new hire setup",
        "amount": -892.15,
        "ref": "PO-6606",
        "entry_type": "normal",
    },
    # --- Brightstar customer receipt (matches B31) ---
    {
        "id": "G36",
        "date": date(2025, 1, 22),
        "description": "Brightstar Enterprises Inc - Equipment operating lease, Q1 2025",
        "amount": 18_500.00,
        "ref": "AR-8812",
        "entry_type": "normal",
    },
    # --- Chase credit card (matches B32) ---
    {
        "id": "G37",
        "date": date(2025, 1, 22),
        "description": "Chase Visa - Corporate credit card statement closing January 2025",
        "amount": -23_456.78,
        "ref": "CC-0125",
        "entry_type": "normal",
    },
    # --- Pinnacle CAM charges (matches B33) ---
    {
        "id": "G38",
        "date": date(2025, 1, 23),
        "description": "Pinnacle Property Management - Common area maintenance charges Q1 2025",
        "amount": -2_150.00,
        "ref": "AP-4414",
        "entry_type": "normal",
    },
    # --- Venmo photographer (matches B34) ---
    {
        "id": "G39",
        "date": date(2025, 1, 24),
        "description": "Venmo - Sarah Chen, Product photography for spring catalog",
        "amount": -1_500.00,
        "ref": "AP-4415",
        "entry_type": "normal",
    },
    # --- Utilities (matches B35) ---
    {
        "id": "G40",
        "date": date(2025, 1, 24),
        "description": "Consolidated Utilities Inc - Electric and natural gas, January 2025",
        "amount": -3_842.17,
        "ref": "AP-4416",
        "entry_type": "normal",
    },
    # --- Customer receipt #2 (matches B36; same amount/entity as G04) ---
    {
        "id": "G41",
        "date": date(2025, 1, 25),
        "description": "TechFlow Solutions Inc - Invoice #TF-2025-003, Platform license  renewal",
        "amount": 32_000.00,
        "ref": "AR-8815",
        "entry_type": "normal",
    },
    # --- Delta flight #2 (matches B37; same amount as G16) ---
    {
        "id": "G42",
        "date": date(2025, 1, 27),
        "description": "Delta Air Lines - LAX to ORD round trip, M. Johnson, Conf #DL9921",
        "amount": -2_847.60,
        "ref": "EXP-7705",
        "entry_type": "normal",
    },
    # --- Batch deposit #2 component 1/2 ---
    {
        "id": "G43",
        "date": date(2025, 1, 27),
        "description": "Lakeside Medical Group - Invoice #LM-2025-108, Medical supplies",
        "amount": 9_250.00,
        "ref": "AR-8816",
        "entry_type": "normal",
    },
    # --- Batch deposit #2 component 2/2 (note: same amount as G26!) ---
    {
        "id": "G44",
        "date": date(2025, 1, 27),
        "description": "Precision Tools Inc - Invoice #PT-2025-042, Custom tooling order",
        "amount": 6_500.00,
        "ref": "AR-8817",
        "entry_type": "normal",
    },
    # --- Insurance (matches B39) ---
    {
        "id": "G45",
        "date": date(2025, 1, 29),
        "description": "Premium Insurance Group - Commercial general liability policy renewal",
        "amount": -11_200.00,
        "ref": "AP-4418",
        "entry_type": "normal",
    },
    # --- Payroll #2 (matches B40; same amount as G28) ---
    {
        "id": "G46",
        "date": date(2025, 1, 31),
        "description": "ADP Payroll - Net pay, period 01/16-01/31/2025, 47 employees",
        "amount": -86_500.00,
        "ref": "PR-2025-01B",
        "entry_type": "normal",
    },
    # --- Sales tax (matches B43) ---
    {
        "id": "G47",
        "date": date(2025, 1, 31),
        "description": "State Department of Revenue - Sales tax remittance January 2025",
        "amount": -4_250.00,
        "ref": "TAX-2025-01",
        "entry_type": "normal",
    },
    # --- OUTSTANDING check (NO bank counterpart, hasn't cleared yet) ---
    {
        "id": "G48",
        "date": date(2025, 1, 28),
        "description": "Henderson Electric - Electrical wiring installation, Invoice #HE-2025-445",
        "amount": -3_200.00,
        "ref": "CHK-4521",
        "entry_type": "normal",
    },
    # --- DEPOSIT IN TRANSIT (NO bank counterpart, not yet credited) ---
    {
        "id": "G49",
        "date": date(2025, 1, 31),
        "description": "Quantum Analytics Corp - Invoice #QA-2025-018, Data analytics platform",
        "amount": 22_000.00,
        "ref": "AR-8820",
        "entry_type": "normal",
    },
]


# ===========================================================================
# Ground Truth
# ===========================================================================

GROUND_TRUTH = {
    # -----------------------------------------------------------------------
    # 1:1 exact-amount matches (37 pairs)
    # -----------------------------------------------------------------------
    "matches_1to1": [
        # --- Easy (unique amount, clear description, close dates) ---
        {
            "bank_id": "B03",
            "gl_id": "G03",
            "note": "Check #4518; 4-day gap (Dec 30 GL, Jan 3 bank) - exceeds typical 3-day window",
        },
        {
            "bank_id": "B12",
            "gl_id": "G13",
            "note": "Acme $150K wire; unique amount, 1-day gap",
        },
        {
            "bank_id": "B14",
            "gl_id": "G15",
            "note": "Riverside $8,200 deposit; same date, but +8200 vs -8200 also exists (NSF)",
        },
        {"bank_id": "B15", "gl_id": "G14", "note": "Azure $4,873.29; unique amount"},
        {
            "bank_id": "B16",
            "gl_id": "G12",
            "note": "Johnson Associates LLC $5,750; unique amount but 3 similar Johnson entities",
        },
        {
            "bank_id": "B20",
            "gl_id": "G19",
            "note": "PayPal $4,500; unique amount, bank shows 'PAYPAL *JRODRIGU'",
        },
        {
            "bank_id": "B25",
            "gl_id": "G30",
            "note": "Globex $12,500 supplier payment; unique amount",
        },
        {"bank_id": "B27", "gl_id": "G32", "note": "Westfield $25,000; unique amount"},
        {
            "bank_id": "B29",
            "gl_id": "G34",
            "note": "Summit Supply $7,650; unique amount",
        },
        {
            "bank_id": "B31",
            "gl_id": "G36",
            "note": "Brightstar $18,500; truncated bank desc 'BRIGHTSTAR ENTE'",
        },
        {
            "bank_id": "B32",
            "gl_id": "G37",
            "note": "Chase card $23,456.78; unique amount",
        },
        {
            "bank_id": "B33",
            "gl_id": "G38",
            "note": "Pinnacle CAM $2,150; unique amount, same vendor as B01 rent",
        },
        {
            "bank_id": "B34",
            "gl_id": "G39",
            "note": "Venmo $1,500; unique amount, bank shows 'VENMO PAYMENT SARAH C'",
        },
        {
            "bank_id": "B35",
            "gl_id": "G40",
            "note": "Utilities $3,842.17; unique amount",
        },
        {"bank_id": "B39", "gl_id": "G45", "note": "Insurance $11,200; unique amount"},
        {"bank_id": "B43", "gl_id": "G47", "note": "Sales tax $4,250; unique amount"},
        # --- Medium (same amount exists elsewhere, need date + description) ---
        {
            "bank_id": "B01",
            "gl_id": "G01",
            "note": "Rent $8,500 #1; SAME amount/date as B02/G02, must use PINNACLE <-> Pinnacle Property",
        },
        {
            "bank_id": "B02",
            "gl_id": "G02",
            "note": "Rent $8,500 #2; SAME amount/date as B01/G01, must use HARBOR <-> Harbor Realty",
        },
        {
            "bank_id": "B04",
            "gl_id": "G04",
            "note": "TechFlow $32K #1; SAME amount as B36/G41, disambiguate by date (Jan 3 vs Jan 25)",
        },
        {
            "bank_id": "B36",
            "gl_id": "G41",
            "note": "TechFlow $32K #2; SAME amount as B04/G04, disambiguate by date (Jan 25 vs Jan 3)",
        },
        {
            "bank_id": "B06",
            "gl_id": "G08",
            "note": "Amazon $1,247.53 #1; SAME amount as B26/G31, cryptic bank desc, date disambiguates (Jan 6/5)",
        },
        {
            "bank_id": "B26",
            "gl_id": "G31",
            "note": "Amazon $1,247.53 #2; SAME amount as B06/G08, cryptic bank desc, date disambiguates (Jan 17)",
        },
        {
            "bank_id": "B07",
            "gl_id": "G09",
            "note": "Staples $892.15; SAME amount as B30/G35 (Office Depot), date + desc disambiguate",
        },
        {
            "bank_id": "B30",
            "gl_id": "G35",
            "note": "Office Depot $892.15; SAME amount as B07/G09 (Staples), date + desc disambiguate",
        },
        {
            "bank_id": "B17",
            "gl_id": "G16",
            "note": "Delta $2,847.60 #1; SAME amount as B37/G42, date disambiguates (Jan 12 vs Jan 27)",
        },
        {
            "bank_id": "B37",
            "gl_id": "G42",
            "note": "Delta $2,847.60 #2; SAME amount as B17/G16, date disambiguates (Jan 27 vs Jan 12)",
        },
        {
            "bank_id": "B23",
            "gl_id": "G28",
            "note": "ADP Payroll $86,500 #1; SAME amount as B40/G46, date disambiguates (Jan 15)",
        },
        {
            "bank_id": "B40",
            "gl_id": "G46",
            "note": "ADP Payroll $86,500 #2; SAME amount as B23/G28, date disambiguates (Jan 31)",
        },
        # --- Hard (same amount AND same/close date, description is the only signal) ---
        {
            "bank_id": "B10",
            "gl_id": "G10",
            "note": "Johnson & Associates $5,500; SAME amount AND date as B11/G11, must match 'JOHNSON & ASSOC' -> 'Johnson & Associates'",
        },
        {
            "bank_id": "B11",
            "gl_id": "G11",
            "note": "K. Johnson Consulting $5,500; SAME amount AND date as B10/G10, must match 'K JOHNSON CONS GRP' -> 'K. Johnson Consulting Group'",
        },
        # --- Self-correcting entries (same absolute amount $15,340, different signs/entities) ---
        {
            "bank_id": "B05",
            "gl_id": "G05",
            "note": "Summit wrong payment -$15,340; same magnitude as B08(+) and B09(-), match by entity+sign+date",
        },
        {
            "bank_id": "B08",
            "gl_id": "G06",
            "note": "Summit reversal +$15,340; same magnitude as B05(-) and B09(-), match by sign+date",
        },
        {
            "bank_id": "B09",
            "gl_id": "G07",
            "note": "Northstar correct payment -$15,340; same magnitude as B05, different entity",
        },
        # --- NSF pair (same absolute amount, opposite signs) ---
        {
            "bank_id": "B24",
            "gl_id": "G29",
            "note": "NSF return -$8,200; same absolute amount as B14/G15 deposit, match by sign+date",
        },
        # --- Check number matches (same amount $3,400 x2 and $4,200) ---
        {
            "bank_id": "B18",
            "gl_id": "G17",
            "note": "Check #4519, Meridian $3,400; same amount as B19/G18, check number disambiguates",
        },
        {
            "bank_id": "B19",
            "gl_id": "G18",
            "note": "Check #4520, Meridian $3,400 duplicate; same amount as B18/G17, check number disambiguates",
        },
        {
            "bank_id": "B21",
            "gl_id": "G22",
            "note": "Check #4522, Apex $4,200 reissue; G20/G21 are offsetting void pair, check number disambiguates",
        },
    ],
    # -----------------------------------------------------------------------
    # Many-to-one matches (bank deposit = sum of GL entries)
    # -----------------------------------------------------------------------
    "matches_many_to_one": [
        {
            "bank_id": "B22",
            "gl_ids": ["G23", "G24", "G25", "G26", "G27"],
            "bank_amount": 47_250.00,
            "gl_amounts": {
                "G23": 12_000.00,
                "G24": 8_750.00,
                "G25": 15_000.00,
                "G26": 6_500.00,
                "G27": 5_000.00,
            },
            "note": "Batch deposit: 5 customer payments deposited together. G26 ($6,500) same amount as G44 in batch #2.",
        },
        {
            "bank_id": "B38",
            "gl_ids": ["G43", "G44"],
            "bank_amount": 15_750.00,
            "gl_amounts": {"G43": 9_250.00, "G44": 6_500.00},
            "note": "Batch deposit: 2 customer payments. G44 ($6,500) same amount as G26 in batch #1.",
        },
    ],
    # -----------------------------------------------------------------------
    # Amount mismatch (bank and GL amounts differ due to fee netting)
    # -----------------------------------------------------------------------
    "matches_amount_mismatch": [
        {
            "bank_id": "B28",
            "gl_id": "G33",
            "bank_amount": 75_000.00,
            "gl_amount": 74_975.00,
            "difference": 25.00,
            "note": "Globex wire: bank shows gross $75,000; GL shows $74,975 (accountant netted $25 wire fee). Missing $25 GL entry for wire fee expense.",
        },
    ],
    # -----------------------------------------------------------------------
    # Offsetting GL entries (net to zero, no bank counterpart)
    # -----------------------------------------------------------------------
    "offsetting_gl_pairs": [
        {
            "gl_ids": ["G20", "G21"],
            "amounts": {"G20": -4_200.00, "G21": 4_200.00},
            "note": "Voided check #4515 to Apex Design Group. Original (-$4,200) + void (+$4,200) = $0. Reissue is G22 -> B21.",
        },
    ],
    # -----------------------------------------------------------------------
    # Unmatched bank items (no GL counterpart exists)
    # -----------------------------------------------------------------------
    "unmatched_bank": [
        {
            "bank_id": "B13",
            "amount": -25.00,
            "note": "Wire transfer fee - not yet recorded in GL",
        },
        {
            "bank_id": "B41",
            "amount": -35.00,
            "note": "Monthly service charge - not yet recorded in GL",
        },
        {
            "bank_id": "B42",
            "amount": 12.47,
            "note": "Interest earned - not yet recorded in GL",
        },
    ],
    # -----------------------------------------------------------------------
    # Unmatched GL items (no bank counterpart in January statement)
    # -----------------------------------------------------------------------
    "unmatched_gl": [
        {
            "gl_id": "G48",
            "amount": -3_200.00,
            "note": "Outstanding check #4521 to Henderson Electric - not yet cleared",
        },
        {
            "gl_id": "G49",
            "amount": 22_000.00,
            "note": "Deposit in transit from Quantum Analytics - not yet credited by bank",
        },
    ],
    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    "summary": {
        "bank_transactions": 43,
        "gl_entries": 49,
        "matches_1to1": 37,
        "matches_many_to_one": 2,  # covering 7 GL entries
        "matches_amount_mismatch": 1,
        "offsetting_gl_pairs": 1,  # covering 2 GL entries
        "unmatched_bank": 3,
        "unmatched_gl": 2,
        # Verification: 37 + 7 + 1 + 2 + 2 = 49 GL entries accounted for
        # Verification: 37 + 2 + 1 + 3 = 43 bank transactions accounted for
    },
}


# ===========================================================================
# Difficulty Analysis
# ===========================================================================

DIFFICULTY_ANALYSIS = """
MATCHING DIFFICULTY BREAKDOWN
=============================

EASY (16 bank items) - Unique amount within date window, no ambiguity:
  B03 (CHK 4518 $6,780), B12 (Acme $150K), B14 ($8,200 deposit),
  B15 (Azure $4,873.29), B16 (Johnson LLC $5,750), B20 (PayPal $4,500),
  B25 (Globex $12,500), B27 (Westfield $25K), B29 (Summit $7,650),
  B31 (Brightstar $18,500), B32 (Chase $23,456.78), B33 (Pinnacle $2,150),
  B34 (Venmo $1,500), B35 (Utilities $3,842.17), B39 (Insurance $11,200),
  B43 (Sales tax $4,250)

MEDIUM (14 bank items) - Same amount exists elsewhere, need date OR description:
  B01/B02 (2x rent $8,500 - description needed, same date)
  B04/B36 (2x TechFlow $32K - date separates, 22 days apart)
  B06/B26 (2x Amazon $1,247.53 - date separates, 11 days apart; cryptic bank desc)
  B07/B30 (2x office supplies $892.15 - date separates, 14 days apart)
  B17/B37 (2x Delta $2,847.60 - date separates, 15 days apart)
  B23/B40 (2x ADP $86,500 - date separates, 16 days apart)
  B24 (NSF -$8,200 - sign disambiguates from deposit +$8,200)
  B05/B08/B09 (self-correcting $15,340 - entity + sign disambiguate)

HARD (7 bank items) - Requires sophisticated matching:
  B10/B11 (2x Johnson $5,500 - SAME amount AND date, description is only signal)
  B18/B19 (2x Meridian $3,400 - check number is only reliable signal)
  B21 (Apex $4,200 - must identify voided check offsetting pair first)

VERY HARD (3 bank items) - Cannot be solved with standard 1:1 exact-amount joins:
  B22 (batch deposit $47,250 = 5 GL entries; requires subset-sum matching)
  B38 (batch deposit $15,750 = 2 GL entries; subset-sum, G44 amount overlaps G26)
  B28 (Globex wire $75,000 vs GL $74,975; $25 amount mismatch, no exact join)

STRUCTURAL (3+2 items) - Items with no counterpart:
  B13, B41, B42 (bank fees/interest - no GL entries exist)
  G48, G49 (outstanding check, deposit in transit - not on bank statement)

WHAT PURELY DETERMINISTIC SQL *CAN* HANDLE:
  - Easy matches (16): trivial equi-join on amount + date window
  - Medium matches (14): equi-join + ROW_NUMBER + description similarity for ties
  - Hard/check-number matches (5): extract check numbers, equi-join on amount + check#
  - Offsetting detection (1 pair): find GL entries where A.amount + B.amount = 0
  Total: ~35 of 43 bank items (~81%)

WHAT REQUIRES REASONING OR ADVANCED TECHNIQUES:
  - Batch deposits (2): subset-sum over unmatched GL entries (recursive CTE possible but complex)
  - Amount mismatch (1): fuzzy/tolerance join or manual investigation
  - Johnson disambiguation (2): high-quality fuzzy string matching on truncated names
  Total: ~5 of 43 bank items (~12%)
  Without these: 3 bank items remain fully unmatched (fees/interest = 7%)
"""
