"""Linear chain DAG test workspace.

DAG shape: parse -> enrich -> aggregate (strictly sequential)

Complements diamond_dag.py by testing a pure linear dependency chain
where each layer has exactly one task.

Usage:
    taskgraph run --spec tests.linear_chain -o output.db
"""

NODES = [
    # --- Sources ---
    {
        "name": "raw_orders",
        "source": [
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
    {
        "name": "price_list",
        "source": [
            {"sku": "widget_a", "unit_price": 10.00},
            {"sku": "widget_b", "unit_price": 25.00},
            {"sku": "widget_c", "unit_price": 7.50},
        ],
        "columns": ["sku", "unit_price"],
    },
    # --- Transforms: linear chain ---
    {
        "name": "parse",
        "depends_on": ["raw_orders"],
        "sql": """
            CREATE OR REPLACE VIEW parse_order_lines AS
            SELECT
                o.order_id,
                o.customer,
                o.date,
                split_part(item, ':', 1) AS sku,
                CAST(split_part(item, ':', 2) AS INTEGER) AS quantity
            FROM raw_orders o,
            UNNEST(string_split(o.items, ',')) AS t(item)
            """,
        "output_columns": {
            "parse_order_lines": ["order_id", "customer", "date", "sku", "quantity"],
        },
    },
    {
        "name": "enrich",
        "depends_on": ["parse", "price_list"],
        "sql": """
            CREATE OR REPLACE VIEW enrich_lines AS
            SELECT
                l.*, p.unit_price, l.quantity * p.unit_price AS line_total
            FROM parse_order_lines l
            JOIN price_list p ON p.sku = l.sku
            """,
        "output_columns": {
            "enrich_lines": ["unit_price", "line_total"],
        },
    },
    {
        "name": "aggregate",
        "depends_on": ["enrich"],
        "sql": """
            CREATE OR REPLACE VIEW aggregate_customer_totals AS
            SELECT
                customer,
                COUNT(DISTINCT order_id) AS order_count,
                SUM(quantity) AS total_items,
                SUM(line_total) AS total_spend
            FROM enrich_lines
            GROUP BY customer;

            CREATE OR REPLACE VIEW aggregate_sku_totals AS
            SELECT
                sku,
                SUM(quantity) AS total_quantity,
                SUM(line_total) AS total_revenue
            FROM enrich_lines
            GROUP BY sku
            """,
        "output_columns": {
            "aggregate_customer_totals": [
                "customer",
                "order_count",
                "total_items",
                "total_spend",
            ],
            "aggregate_sku_totals": ["sku", "total_quantity", "total_revenue"],
        },
    },
]
