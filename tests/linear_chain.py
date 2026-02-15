"""Linear chain DAG test workspace.

DAG shape: parse -> enrich -> aggregate (strictly sequential)

Complements diamond_dag.py by testing a pure linear dependency chain
where each layer has exactly one task.

Usage:
    taskgraph run tests/linear_chain.py -o output.db
"""

# --- Inputs ---

INPUTS = {
    "raw_orders": {
        "data": [
            {
                "order_id": "ORD-001",
                "customer": "alice",
                "items": "widget_a:2,widget_b:1",
                "date": "2025-03-01",
            },
            {
                "order_id": "ORD-002",
                "customer": "bob",
                "items": "widget_c:5",
                "date": "2025-03-02",
            },
            {
                "order_id": "ORD-003",
                "customer": "alice",
                "items": "widget_a:1,widget_c:3",
                "date": "2025-03-03",
            },
            {
                "order_id": "ORD-004",
                "customer": "carol",
                "items": "widget_b:2,widget_a:4",
                "date": "2025-03-04",
            },
            {
                "order_id": "ORD-005",
                "customer": "bob",
                "items": "widget_b:1",
                "date": "2025-03-05",
            },
        ],
        "columns": ["order_id", "customer", "items", "date"],
    },
    "price_list": {
        "data": [
            {"sku": "widget_a", "unit_price": 10.00},
            {"sku": "widget_b", "unit_price": 25.00},
            {"sku": "widget_c", "unit_price": 7.50},
        ],
        "columns": ["sku", "unit_price"],
    },
}

# --- Tasks: linear chain ---

parse = {
    "name": "parse",
    "prompt": (
        "Parse the raw_orders table. The 'items' column contains comma-separated\n"
        "entries in the format 'sku:quantity' (e.g. 'widget_a:2,widget_b:1').\n\n"
        "Create a view 'order_lines' with one row per item per order:\n"
        "- order_id: from the order\n"
        "- customer: from the order\n"
        "- date: from the order\n"
        "- sku: the item SKU (e.g. 'widget_a')\n"
        "- quantity: the integer quantity\n\n"
        "Hint: Use string_split() and unnest() to explode the comma-separated\n"
        "items into individual rows.\n"
    ),
    "inputs": ["raw_orders"],
    "outputs": ["order_lines"],
    "output_columns": {
        "order_lines": ["order_id", "customer", "date", "sku", "quantity"],
    },
}

enrich = {
    "name": "enrich",
    "prompt": (
        "Join order_lines with price_list to compute line totals.\n\n"
        "Create a view 'enriched_lines' with all columns from order_lines plus:\n"
        "- unit_price: from the price_list\n"
        "- line_total: quantity * unit_price\n"
    ),
    "inputs": ["order_lines", "price_list"],
    "outputs": ["enriched_lines"],
    "output_columns": {
        "enriched_lines": ["unit_price", "line_total"],
    },
}

aggregate = {
    "name": "aggregate",
    "prompt": (
        "Create two summary views from enriched_lines:\n\n"
        "1. 'customer_totals' with columns:\n"
        "   - customer\n"
        "   - order_count: number of distinct orders\n"
        "   - total_items: sum of quantities\n"
        "   - total_spend: sum of line_total\n\n"
        "2. 'sku_totals' with columns:\n"
        "   - sku\n"
        "   - total_quantity: sum of quantities across all orders\n"
        "   - total_revenue: sum of line_total\n"
    ),
    "inputs": ["enriched_lines"],
    "outputs": ["customer_totals", "sku_totals"],
    "output_columns": {
        "customer_totals": ["customer", "order_count", "total_items", "total_spend"],
        "sku_totals": ["sku", "total_quantity", "total_revenue"],
    },
}

TASKS = [parse, enrich, aggregate]
