"""Minimal single-task workspace.

The simplest possible spec: one input, one task, one output.
Agent classifies employees by department and computes headcount.

Usage:
    taskgraph run --spec examples/basics/single_task/spec.py -o output.db
"""

NODES = [
    {
        "name": "employees",
        "source": [
            {"id": 1, "name": "Alice", "department": "Engineering", "salary": 120000},
            {"id": 2, "name": "Bob", "department": "Engineering", "salary": 110000},
            {"id": 3, "name": "Carol", "department": "Sales", "salary": 95000},
            {"id": 4, "name": "Dave", "department": "Sales", "salary": 90000},
            {"id": 5, "name": "Eve", "department": "Engineering", "salary": 130000},
            {"id": 6, "name": "Frank", "department": "HR", "salary": 85000},
        ],
        "columns": ["id", "name", "department", "salary"],
    },
    {
        "name": "summarize",
        "depends_on": ["employees"],
        "sql": """
            CREATE OR REPLACE VIEW summarize_department_summary AS
            SELECT
                department,
                COUNT(*) AS headcount,
                SUM(salary) AS total_salary,
                CAST(ROUND(AVG(salary), 0) AS INTEGER) AS avg_salary
            FROM employees
            GROUP BY department
            ORDER BY department
            """,
        "output_columns": {
            "summarize_department_summary": [
                "department",
                "headcount",
                "total_salary",
                "avg_salary",
            ],
        },
    },
]
