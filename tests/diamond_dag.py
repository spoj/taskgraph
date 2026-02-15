"""Diamond DAG test workspace.

DAG shape:
       prep
      /    \\
  sales   costs
      \\    /
      report

Tests: DAG resolution, layer concurrency, namespace enforcement.
"""


# --- Inputs: synthetic data ---

INPUTS = {
    "transactions": {
        "data": lambda: [
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
    "products": {
        "data": [
            {"name": "Widget A", "sku": "WA-001", "unit_price": 100.00},
            {"name": "Widget B", "sku": "WB-001", "unit_price": 250.00},
            {"name": "Widget C", "sku": "WC-001", "unit_price": 300.00},
        ],
        "columns": ["name", "sku", "unit_price"],
    },
}


# --- Tasks: diamond DAG ---

prep = {
    "name": "prep",
    "prompt": (
        "Split the transactions table into two views based on the 'type' column:\n"
        "- 'prepared_sales': rows where type='sale', with columns: id, date, product, amount, region\n"
        "- 'prepared_costs': rows where type='cost', with columns: id, date, product, category, amount, region\n"
    ),
    "inputs": ["transactions"],
    "outputs": ["prepared_sales", "prepared_costs"],
    "output_columns": {
        "prepared_sales": ["id", "date", "product", "amount", "region"],
        "prepared_costs": ["id", "date", "product", "category", "amount", "region"],
    },
}

sales = {
    "name": "sales",
    "prompt": (
        "Summarize sales by product. Create a view 'sales_summary' with columns:\n"
        "- product: product name\n"
        "- total_sales: sum of amount\n"
        "- num_transactions: count of transactions\n"
        "Join with the products table to include the SKU.\n"
    ),
    "inputs": ["prepared_sales", "products"],
    "outputs": ["sales_summary"],
    "output_columns": {
        "sales_summary": ["product", "total_sales", "num_transactions"],
    },
}

costs = {
    "name": "costs",
    "prompt": (
        "Summarize costs by product. Create a view 'costs_summary' with columns:\n"
        "- product: product name\n"
        "- total_costs: sum of amount\n"
        "- materials_cost: sum where category='materials'\n"
        "- labor_cost: sum where category='labor'\n"
    ),
    "inputs": ["prepared_costs"],
    "outputs": ["costs_summary"],
    "output_columns": {
        "costs_summary": ["product", "total_costs", "materials_cost", "labor_cost"],
    },
}

report = {
    "name": "report",
    "prompt": (
        "Create a profit report combining sales and costs. Create a view 'profit_report' with columns:\n"
        "- product: product name\n"
        "- total_sales: from sales_summary\n"
        "- total_costs: from costs_summary\n"
        "- profit: total_sales - total_costs\n"
        "- margin_pct: round(profit / total_sales * 100, 1)\n"
        "Use LEFT JOIN so products with sales but no costs still appear.\n"
    ),
    "inputs": ["sales_summary", "costs_summary"],
    "outputs": ["profit_report"],
    "output_columns": {
        "profit_report": [
            "product",
            "total_sales",
            "total_costs",
            "profit",
            "margin_pct",
        ],
    },
}

TASKS = [prep, sales, costs, report]
