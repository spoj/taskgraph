"""Validation view demo workspace.

Demonstrates validation views with a categorize-and-balance-check pattern.
The agent must categorize rows and produce a validation view.

Usage:
    taskgraph run --spec tests.validation_view_demo -o output.db
"""

# --- Inputs ---

INPUTS = {
    "expenses": {
        "data": [
            {
                "id": 1,
                "date": "2025-01-05",
                "vendor": "Acme Corp",
                "amount": 1500.00,
                "description": "Office supplies",
            },
            {
                "id": 2,
                "date": "2025-01-08",
                "vendor": "Acme Corp",
                "amount": 2300.00,
                "description": "Office furniture",
            },
            {
                "id": 3,
                "date": "2025-01-10",
                "vendor": "CloudHost Inc",
                "amount": 4200.00,
                "description": "Monthly hosting",
            },
            {
                "id": 4,
                "date": "2025-01-12",
                "vendor": "CloudHost Inc",
                "amount": 800.00,
                "description": "SSL certificates",
            },
            {
                "id": 5,
                "date": "2025-01-15",
                "vendor": "TravelCo",
                "amount": 950.00,
                "description": "Flight booking",
            },
            {
                "id": 6,
                "date": "2025-01-18",
                "vendor": "TravelCo",
                "amount": 320.00,
                "description": "Hotel stay",
            },
            {
                "id": 7,
                "date": "2025-01-20",
                "vendor": "Acme Corp",
                "amount": 1100.00,
                "description": "Printer toner",
            },
            {
                "id": 8,
                "date": "2025-01-25",
                "vendor": "LegalEase LLP",
                "amount": 3500.00,
                "description": "Contract review",
            },
            {
                "id": 9,
                "date": "2025-01-28",
                "vendor": "CloudHost Inc",
                "amount": 4200.00,
                "description": "Monthly hosting",
            },
            {
                "id": 10,
                "date": "2025-01-30",
                "vendor": "TravelCo",
                "amount": 130.00,
                "description": "Parking",
            },
        ],
        "columns": ["id", "date", "vendor", "amount", "description"],
    },
    # Expected total for this dataset
    "expected_total": {
        "data": [
            {"label": "expected", "total": 19000.00},
        ],
        "columns": ["label", "total"],
    },
}

# Expected total = sum of all amounts: 1500+2300+4200+800+950+320+1100+3500+4200+130 = 19000

# --- Task ---

TASKS = [
    {
        "name": "categorize",
        "repair_context": (
            "Categorize each expense line into a spending category based on the\n"
            "vendor and description. Use these categories:\n"
            "- 'Office' for office supplies, furniture, toner (vendor: Acme Corp)\n"
            "- 'Technology' for hosting, certificates (vendor: CloudHost Inc)\n"
            "- 'Travel' for flights, hotels, parking (vendor: TravelCo)\n"
            "- 'Professional' for legal, consulting (vendor: LegalEase LLP)\n\n"
            "Create three views:\n"
            "1. 'categorized' — all input columns plus a 'category' column\n"
            "2. 'category_summary' — one row per category with: category, item_count,\n"
            "   total_amount. Include a final row with category='TOTAL' summing everything.\n"
            "3. 'categorize__validation' — rows with columns: status, message.\n"
            "   Add a row with status='fail' if any expense row is missing from categorized,\n"
            "   or if the categorized total differs from expected_total by > 0.01.\n"
        ),
        "sql": """
            CREATE OR REPLACE VIEW categorized AS
            SELECT
                e.*,
                CASE
                    WHEN e.vendor = 'Acme Corp' THEN 'Office'
                    WHEN e.vendor = 'CloudHost Inc' THEN 'Technology'
                    WHEN e.vendor = 'TravelCo' THEN 'Travel'
                    WHEN e.vendor = 'LegalEase LLP' THEN 'Professional'
                    ELSE 'Other'
                END AS category
            FROM expenses e;

            CREATE OR REPLACE VIEW category_summary AS
            SELECT
                category,
                COUNT(*) AS item_count,
                SUM(amount) AS total_amount
            FROM categorized
            GROUP BY category
            UNION ALL
            SELECT
                'TOTAL' AS category,
                COUNT(*) AS item_count,
                SUM(amount) AS total_amount
            FROM categorized;

            CREATE OR REPLACE VIEW categorize__validation AS
            WITH
            exp AS (SELECT COUNT(*) AS cnt FROM expenses),
            cat AS (SELECT COUNT(*) AS cnt FROM categorized),
            sumcat AS (SELECT SUM(amount) AS total_amount FROM categorized),
            tot AS (SELECT total FROM expected_total)
            SELECT
                'fail' AS status,
                'missing rows: expected ' || CAST(exp.cnt AS VARCHAR) ||
                ' got ' || CAST(cat.cnt AS VARCHAR) AS message
            FROM exp, cat
            WHERE exp.cnt <> cat.cnt

            UNION ALL
            SELECT
                'fail' AS status,
                'total mismatch: expected ' || CAST(tot.total AS VARCHAR) ||
                ' got ' || CAST(sumcat.total_amount AS VARCHAR) AS message
            FROM tot, sumcat
            WHERE ABS(tot.total - sumcat.total_amount) > 0.01

            UNION ALL
            SELECT 'pass' AS status, 'ok' AS message
            FROM exp, cat, tot, sumcat
            WHERE exp.cnt = cat.cnt
              AND ABS(tot.total - sumcat.total_amount) <= 0.01
            """,
        "inputs": ["expenses", "expected_total"],
        "outputs": ["categorized", "category_summary", "categorize__validation"],
        "output_columns": {
            "categorized": ["category"],
            "category_summary": ["category", "item_count", "total_amount"],
            "categorize__validation": ["status", "message"],
        },
    },
]
