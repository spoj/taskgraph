# Basic TaskGraph Examples

This directory contains simple, illustrative examples of how to define TaskGraph specifications. They are excellent references for learning the core mechanics of the framework.

- **`single_task/`**: The simplest possible spec. One source node with inline data and one `sql` node that summarizes it.
- **`linear_chain/`**: Demonstrates a simple sequential dependency chain (`parse -> enrich -> aggregate`).
- **`diamond_dag/`**: Demonstrates concurrent execution. Two nodes depend on a single upstream node, so TaskGraph will execute them simultaneously in parallel before joining them in a final report.
- **`llm_task/`**: A single LLM `prompt` node that classifies source data.
- **`validation_demo/`**: Demonstrates how to write deterministic `validate` SQL queries that act as guardrails, ensuring that nodes do not complete successfully unless certain row-counts or financial balances match.

### Running the examples

You can run any of these directly using the TaskGraph CLI:

```bash
taskgraph run --spec examples/basics/diamond_dag/spec.py -o output.db
```
