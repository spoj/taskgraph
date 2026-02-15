"""Task definition and DAG resolution for multi-task workspaces.

A Task declares what it reads (inputs), what it produces (outputs),
and how to validate results. Tasks form a DAG based on output->input
dependencies. resolve_task_deps() builds the raw dependency graph;
resolve_dag() produces topo-sorted layers for display/logging.

Example:

    tasks = [
        Task(
            name="prep",
            inputs=["raw_rows"],
            outputs=["rows_clean"],
            prompt="Clean and normalize raw rows...",
        ),
        Task(
            name="match",
            inputs=["rows_clean", "reference"],
            outputs=["matches", "match_summary"],
            prompt="Match normalized rows against a reference table...",
            validate_sql=["SELECT ... FROM ... WHERE mismatch_count > 0"],
        ),
    ]

    deps = resolve_task_deps(tasks)
    # deps = {"prep": set(), "t1353": {"prep"}}

    layers = resolve_dag(tasks)
    # layers[0] = [prep]       (no dependencies)
    # layers[1] = [t1353]      (depends on prep's output)
"""

import duckdb
from dataclasses import dataclass, field


@dataclass
class Task:
    """A single unit of work in a workspace.

    Attributes:
        name: Unique identifier, also used as namespace prefix for intermediate views.
        prompt: Task-specific instructions for the agent.
        inputs: Table/view names this task reads from. These must exist before the task runs.
        outputs: View names this task must produce. Other tasks can depend on these.
        validate_sql: SQL queries that must each return zero rows. Evaluated sequentially,
            short-circuits on first query that returns rows. Each returned row becomes an
            error message (single-column value, or col=val pairs for multi-column).
        output_columns: Optional schema check. Maps view_name -> list of required column
            names. Validation fails if a view is missing any declared column.
    """

    name: str
    prompt: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    validate_sql: list[str] = field(default_factory=list)
    output_columns: dict[str, list[str]] = field(default_factory=dict)

    def validate(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        """Run validation. Returns error messages (empty = pass).

        Checks run in order, short-circuiting on first failure:
        1. All declared output views exist.
        2. Output views have required columns (if output_columns specified).
        3. Each validate_sql query returns zero rows.
        """
        # 1. Check all declared outputs exist
        existing_views = {
            row[0]
            for row in conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE internal = false"
            ).fetchall()
        }

        missing = [o for o in self.outputs if o not in existing_views]
        if missing:
            return [f"Required output view '{o}' was not created." for o in missing]

        # 2. Check output columns
        if self.output_columns:
            for view_name, required_cols in self.output_columns.items():
                if view_name not in existing_views:
                    continue  # Already caught in step 1
                try:
                    actual_cols = {
                        row[0]
                        for row in conn.execute(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = ?",
                            [view_name],
                        ).fetchall()
                    }
                except duckdb.Error as e:
                    return [f"Schema check error for '{view_name}': {e}"]

                missing_cols = [c for c in required_cols if c not in actual_cols]
                if missing_cols:
                    return [
                        f"View '{view_name}' is missing required column(s): "
                        f"{', '.join(missing_cols)}. "
                        f"Actual columns: {', '.join(sorted(actual_cols))}"
                    ]

        # 3. Run SQL checks sequentially, short-circuit on first failure
        for sql in self.validate_sql:
            try:
                cursor = conn.execute(sql)
                rows = cursor.fetchall()
            except duckdb.Error as e:
                return [f"Validation query error: {e}"]
            if rows:
                cols = [d[0] for d in cursor.description]
                errors = []
                for row in rows:
                    if len(cols) == 1:
                        errors.append(str(row[0]))
                    else:
                        errors.append(", ".join(f"{c}={v}" for c, v in zip(cols, row)))
                return errors

        return []


def resolve_task_deps(tasks: list[Task]) -> dict[str, set[str]]:
    """Build the dependency graph for tasks.

    Returns a mapping of task_name -> set of task names it depends on.

    Raises ValueError if two tasks produce the same output view.
    """
    # Build output -> producing task mapping
    output_to_task: dict[str, str] = {}
    for t in tasks:
        for o in t.outputs:
            if o in output_to_task:
                raise ValueError(
                    f"Output '{o}' produced by both '{output_to_task[o]}' and '{t.name}'"
                )
            output_to_task[o] = t.name

    # Build dependency graph: task -> set of task names it depends on
    deps: dict[str, set[str]] = {t.name: set() for t in tasks}
    for t in tasks:
        for inp in t.inputs:
            if inp in output_to_task:
                producer = output_to_task[inp]
                if producer != t.name:  # Don't self-depend
                    deps[t.name].add(producer)

    return deps


def resolve_dag(tasks: list[Task]) -> list[list[Task]]:
    """Topologically sort tasks into execution layers (for display/logging).

    Tasks in the same layer have no dependencies on each other and can
    run concurrently. Layer 0 has no task dependencies, layer 1 depends
    only on layer 0, etc.

    Note: Actual execution uses resolve_task_deps() for finer-grained
    scheduling where each task starts as soon as its specific dependencies
    complete. This function is used for display and validation only.

    Raises ValueError if:
    - Two tasks produce the same output view
    - A cycle is detected
    - A task input references another task's output that doesn't exist

    Returns:
        List of layers, where each layer is a list of tasks.
    """
    task_by_name = {t.name: t for t in tasks}
    deps = {name: set(d) for name, d in resolve_task_deps(tasks).items()}

    # Kahn's algorithm producing layers
    in_degree = {name: len(d) for name, d in deps.items()}
    layers: list[list[Task]] = []

    remaining = set(task_by_name.keys())

    while remaining:
        # Find all tasks with in_degree 0
        layer_names = sorted(n for n in remaining if in_degree[n] == 0)

        if not layer_names:
            cycle_members = sorted(remaining)
            raise ValueError(f"Dependency cycle detected among tasks: {cycle_members}")

        layer = [task_by_name[n] for n in layer_names]
        layers.append(layer)

        # Remove this layer and update in_degrees
        for name in layer_names:
            remaining.remove(name)
            for other in remaining:
                if name in deps[other]:
                    deps[other].remove(name)
                    in_degree[other] -= 1

    return layers


def validate_task_graph(tasks: list[Task], available_tables: set[str]) -> list[str]:
    """Validate that all task inputs can be satisfied.

    Checks that every task input is either:
    - An available table (ingested data)
    - An output of another task

    Returns list of error messages (empty = valid).
    """
    errors = []

    # Collect all outputs
    all_outputs = set()
    for t in tasks:
        all_outputs.update(t.outputs)

    # Check all inputs are satisfiable
    for t in tasks:
        for inp in t.inputs:
            if inp not in available_tables and inp not in all_outputs:
                errors.append(
                    f"Task '{t.name}' requires input '{inp}' which is neither "
                    f"an ingested table nor an output of another task."
                )

    return errors
