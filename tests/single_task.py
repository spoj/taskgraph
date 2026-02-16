"""Minimal single-task workspace.

The simplest possible spec: one input, one task, one output.
Agent classifies employees by department and computes headcount.

Usage:
    taskgraph run --spec tests.single_task -o output.db
"""

# --- Inputs ---

INPUTS = {
    "employees": {
        "data": [
            {"id": 1, "name": "Alice", "department": "Engineering", "salary": 120000},
            {"id": 2, "name": "Bob", "department": "Engineering", "salary": 110000},
            {"id": 3, "name": "Carol", "department": "Sales", "salary": 95000},
            {"id": 4, "name": "Dave", "department": "Sales", "salary": 90000},
            {"id": 5, "name": "Eve", "department": "Engineering", "salary": 130000},
            {"id": 6, "name": "Frank", "department": "HR", "salary": 85000},
        ],
        "columns": ["id", "name", "department", "salary"],
    },
}

# --- Task ---

TASKS = [
    {
        "name": "summarize",
        "intent": (
            "Summarize employees by department. Create a view 'department_summary' with:\n"
            "- department: department name\n"
            "- headcount: number of employees\n"
            "- total_salary: sum of salaries\n"
            "- avg_salary: average salary (rounded to nearest integer)\n"
            "Order by department name.\n"
        ),
        "sql": """
            CREATE OR REPLACE VIEW department_summary AS
            SELECT
                department,
                COUNT(*) AS headcount,
                SUM(salary) AS total_salary,
                CAST(ROUND(AVG(salary), 0) AS INTEGER) AS avg_salary
            FROM employees
            GROUP BY department
            ORDER BY department
            """,
        "inputs": ["employees"],
        "outputs": ["department_summary"],
        "output_columns": {
            "department_summary": [
                "department",
                "headcount",
                "total_salary",
                "avg_salary",
            ],
        },
    },
]
