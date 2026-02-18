"""Validation view demo workspace.

Demonstrates validation views with a categorize-and-balance-check pattern.
The agent must categorize rows and produce a validation view.

Usage:
    taskgraph run --spec tests.validation_view_demo -o output.db
"""

# Expected total = sum of all amounts: 1500+2300+4200+800+950+320+1100+3500+4200+130 = 19000

NODES = [
    # --- Sources ---
    {
        "name": "expenses",
        "source": [
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
    {
        "name": "expected_total",
        "source": [
            {"label": "expected", "total": 19000.00},
        ],
        "columns": ["label", "total"],
    },
    # --- Transform ---
    {
        "name": "categorize",
        "depends_on": ["expenses", "expected_total"],
        "sql": """
            CREATE OR REPLACE VIEW categorize_detail AS
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

            CREATE OR REPLACE VIEW categorize_summary AS
            SELECT
                category,
                COUNT(*) AS item_count,
                SUM(amount) AS total_amount
            FROM categorize_detail
            GROUP BY category
            UNION ALL
            SELECT
                'TOTAL' AS category,
                COUNT(*) AS item_count,
                SUM(amount) AS total_amount
            FROM categorize_detail;
            """,
        "validate_sql": """
            CREATE OR REPLACE VIEW categorize__validation AS
            WITH
            exp AS (SELECT COUNT(*) AS cnt FROM expenses),
            cat AS (SELECT COUNT(*) AS cnt FROM categorize_detail),
            sumcat AS (SELECT SUM(amount) AS total_amount FROM categorize_detail),
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
        "output_columns": {
            "categorize_detail": ["category"],
            "categorize_summary": ["category", "item_count", "total_amount"],
        },
    },
]
