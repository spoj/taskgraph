"""Diamond DAG test workspace.

DAG shape:
       prep
      /    \\
  sales   costs
      \\    /
      report

Tests: DAG resolution, layer concurrency, namespace enforcement.

Usage:
    taskgraph run --spec examples/basics/diamond_dag/spec.py -o output.db
"""

NODES = [
    # --- Sources ---
    {
        "name": "transactions",
        "source": lambda: [
            {
                "id": 1,
                "date": "2025-01-05",
                "type": "sale",
                "product": "Widget A",
                "amount": 100.00,
                "region": "North",
            },
            {
                "id": 2,
                "date": "2025-01-06",
                "type": "cost",
                "product": "Widget A",
                "category": "materials",
                "amount": 40.00,
                "region": "North",
            },
            {
                "id": 3,
                "date": "2025-01-07",
                "type": "sale",
                "product": "Widget B",
                "amount": 250.00,
                "region": "South",
            },
            {
                "id": 4,
                "date": "2025-01-08",
                "type": "cost",
                "product": "Widget B",
                "category": "materials",
                "amount": 80.00,
                "region": "South",
            },
            {
                "id": 5,
                "date": "2025-01-09",
                "type": "sale",
                "product": "Widget A",
                "amount": 150.00,
                "region": "South",
            },
            {
                "id": 6,
                "date": "2025-01-10",
                "type": "cost",
                "product": "Widget A",
                "category": "labor",
                "amount": 30.00,
                "region": "North",
            },
            {
                "id": 7,
                "date": "2025-01-11",
                "type": "sale",
                "product": "Widget C",
                "amount": 300.00,
                "region": "North",
            },
            {
                "id": 8,
                "date": "2025-01-12",
                "type": "cost",
                "product": "Widget C",
                "category": "materials",
                "amount": 120.00,
                "region": "North",
            },
            {
                "id": 9,
                "date": "2025-01-13",
                "type": "cost",
                "product": "Widget B",
                "category": "labor",
                "amount": 50.00,
                "region": "South",
            },
            {
                "id": 10,
                "date": "2025-01-14",
                "type": "sale",
                "product": "Widget B",
                "amount": 200.00,
                "region": "North",
            },
        ],
        "columns": ["id", "date", "type", "product", "amount", "region"],
    },
    {
        "name": "products",
        "source": [
            {"name": "Widget A", "sku": "WA-001", "unit_price": 100.00},
            {"name": "Widget B", "sku": "WB-001", "unit_price": 250.00},
            {"name": "Widget C", "sku": "WC-001", "unit_price": 300.00},
        ],
        "columns": ["name", "sku", "unit_price"],
    },
    # --- Transforms: diamond DAG ---
    {
        "name": "prep",
        "depends_on": ["transactions"],
        "sql": """
            CREATE OR REPLACE VIEW prep_sales AS
            SELECT id, date, product, amount, region
            FROM transactions
            WHERE type = 'sale';

            CREATE OR REPLACE VIEW prep_costs AS
            SELECT id, date, product, category, amount, region
            FROM transactions
            WHERE type = 'cost'
            """,
        "output_columns": {
            "prep_sales": ["id", "date", "product", "amount", "region"],
            "prep_costs": ["id", "date", "product", "category", "amount", "region"],
        },
    },
    {
        "name": "sales",
        "depends_on": ["prep", "products"],
        "sql": """
            CREATE OR REPLACE VIEW sales_summary AS
            SELECT
                s.product,
                SUM(s.amount) AS total_sales,
                COUNT(*) AS num_transactions
            FROM prep_sales s
            GROUP BY s.product
            """,
        "output_columns": {
            "sales_summary": ["product", "total_sales", "num_transactions"],
        },
    },
    {
        "name": "costs",
        "depends_on": ["prep"],
        "sql": """
            CREATE OR REPLACE VIEW costs_summary AS
            SELECT
                product,
                SUM(amount) AS total_costs,
                SUM(CASE WHEN category = 'materials' THEN amount ELSE 0 END) AS materials_cost,
                SUM(CASE WHEN category = 'labor' THEN amount ELSE 0 END) AS labor_cost
            FROM prep_costs
            GROUP BY product
            """,
        "output_columns": {
            "costs_summary": ["product", "total_costs", "materials_cost", "labor_cost"],
        },
    },
    {
        "name": "report",
        "depends_on": ["sales", "costs"],
        "sql": """
            CREATE OR REPLACE VIEW report_profit AS
            SELECT
                s.product,
                s.total_sales,
                c.total_costs,
                s.total_sales - c.total_costs AS profit,
                ROUND((s.total_sales - c.total_costs) / s.total_sales * 100, 1) AS margin_pct
            FROM sales_summary s
            LEFT JOIN costs_summary c ON c.product = s.product
            """,
        "output_columns": {
            "report_profit": [
                "product",
                "total_sales",
                "total_costs",
                "profit",
                "margin_pct",
            ],
        },
    },
]
