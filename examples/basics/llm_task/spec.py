"""Simple LLM prompt workspace.

Demonstrates an LLM agent transforming data based on a natural language objective.

Usage:
    taskgraph run --spec examples/basics/llm_task/spec.py -m google/gemini-3.1-pro-preview -o output.db
"""

NODES = [
    {
        "name": "raw_feedback",
        "source": [
            {"id": 1, "text": "The UI is so fast, I love it!"},
            {"id": 2, "text": "App crashed when I clicked save."},
            {"id": 3, "text": "How do I reset my password?"},
        ],
        "columns": ["id", "text"],
    },
    {
        "name": "classify",
        "depends_on": ["raw_feedback"],
        "prompt": """
Analyze the raw_feedback and create a view named `classify_results`.
The view must have columns: `id`, `sentiment` (Positive, Negative, Neutral), 
and `category` (Performance, Bug, Support).
""",
        "output_columns": {
            "classify_results": ["id", "sentiment", "category"],
        },
    },
]
