"""Validation SQL demo workspace.

Demonstrates validate_sql with a categorize-and-balance-check pattern.
The agent must categorize rows and ensure they sum to an expected total.

Usage:
    taskgraph run --spec tests.validate_demo -o output.db
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
        "prompt": (
            "Categorize each expense line into a spending category based on the\n"
            "vendor and description. Use these categories:\n"
            "- 'Office' for office supplies, furniture, toner (vendor: Acme Corp)\n"
            "- 'Technology' for hosting, certificates (vendor: CloudHost Inc)\n"
            "- 'Travel' for flights, hotels, parking (vendor: TravelCo)\n"
            "- 'Professional' for legal, consulting (vendor: LegalEase LLP)\n\n"
            "Create two views:\n"
            "1. 'categorized' — all input columns plus a 'category' column\n"
            "2. 'category_summary' — one row per category with: category, item_count,\n"
            "   total_amount. Include a final row with category='TOTAL' summing everything.\n"
        ),
        "inputs": ["expenses", "expected_total"],
        "outputs": ["categorized", "category_summary"],
        "output_columns": {
            "categorized": ["category"],
            "category_summary": ["category", "item_count", "total_amount"],
        },
        "validate_sql": [
            # Every row must be categorized
            "SELECT 'uncategorized row _row_id=' || e._row_id "
            "FROM expenses e LEFT JOIN categorized c ON e._row_id = c._row_id "
            "WHERE c._row_id IS NULL",
            # Categorized total must match the expected total
            "SELECT 'total mismatch: categorized=' || CAST(c.total AS TEXT) "
            "  || ' expected=' || CAST(t.total AS TEXT) "
            "FROM (SELECT SUM(amount) AS total FROM categorized) c, "
            "     expected_total t "
            "WHERE ABS(c.total - t.total) > 0.01",
        ],
    },
]
