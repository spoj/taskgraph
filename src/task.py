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
            sql="CREATE OR REPLACE VIEW rows_clean AS SELECT ...",
            validate_sql=(
                "CREATE OR REPLACE VIEW prep__validation AS "
                "SELECT 'pass' AS status, 'ok' AS message"
            ),
        ),
        Task(
            name="match",
            inputs=["rows_clean", "reference"],
            outputs=["matches", "match_summary"],
            prompt="Match normalized rows against a reference table...",
            validate_sql=(
                "CREATE OR REPLACE VIEW match__validation AS "
                "SELECT 'pass' AS status, 'ok' AS message"
            ),
        ),
    ]

    deps = resolve_task_deps(tasks)
    # deps = {"prep": set(), "match": {"prep"}}

    layers = resolve_dag(tasks)
    # layers[0] = [prep]       (no dependencies)
    # layers[1] = [match]      (depends on prep's output)
"""

import duckdb
import re
from dataclasses import dataclass, field
from .sql_utils import (
    get_parser_conn,
    split_sql_statements,
    extract_create_name,
    get_column_names,
)

_VALIDATION_VIEW_REQUIRED_COLS = ["status", "message"]
_VALIDATION_STATUS_ALLOWED = {"pass", "warn", "fail"}
MAX_INLINE_MESSAGES = 20  # Cap on messages shown inline in validation/warning output

# Constants and parser helpers moved to sql_utils.py


def validation_view_prefix(task_name: str) -> str:
    return f"{task_name}__validation"


def is_validation_view_for_task(view_name: str, task_name: str) -> bool:
    """Return True if view_name is a validation view for task_name.

    Convention: '{task_name}__validation' and '{task_name}__validation_*'
    """
    prefix = validation_view_prefix(task_name)
    return view_name == prefix or view_name.startswith(prefix + "_")


def validate_one_validation_view(
    conn: duckdb.DuckDBPyConnection, view_name: str
) -> tuple[list[str], bool]:
    """Validate a single validation view.

    Returns (errors, fatal).
    - fatal=True indicates a schema/query/contract problem that should stop immediately.
    - fatal=False with errors means validation 'fail' rows were found.

    The view must have columns 'status' and 'message'.
    Allowed status values: 'pass', 'warn', 'fail' (case-insensitive).
    Any row with lower(status)='fail' is reported as an error.
    """
    actual_cols = get_column_names(conn, view_name)
    if not actual_cols:
        return (
            [f"Validation view '{view_name}' not found."],
            True,
        )

    actual_lower = {c.lower() for c in actual_cols}
    missing = [c for c in _VALIDATION_VIEW_REQUIRED_COLS if c not in actual_lower]
    if missing:
        label = "column" if len(missing) == 1 else "columns"
        return (
            [
                f"Validation view '{view_name}' is missing required {label}: "
                f"{', '.join(missing)}. Actual columns: {', '.join(sorted(actual_cols))}"
            ],
            True,
        )

    # Enforce status domain
    try:
        bad = conn.execute(
            f'SELECT DISTINCT lower(status) AS status FROM "{view_name}" '
            "WHERE status IS NOT NULL AND lower(status) NOT IN ('pass','warn','fail')"
        ).fetchall()
    except duckdb.Error as e:
        return ([f"Validation view query error for '{view_name}': {e}"], True)

    if bad:
        bad_vals = ", ".join(sorted({str(r[0]) for r in bad}))
        allowed = ", ".join(sorted(_VALIDATION_STATUS_ALLOWED))
        return (
            [
                f"Validation view '{view_name}' has invalid status value(s): {bad_vals}. "
                f"Allowed: {allowed}"
            ],
            True,
        )

    has_evidence_view = "evidence_view" in actual_lower

    try:
        if has_evidence_view:
            rows = conn.execute(
                f'SELECT status, message, evidence_view FROM "{view_name}" '
                "WHERE lower(status) = 'fail'"
            ).fetchall()
        else:
            rows = conn.execute(
                f'SELECT status, message FROM "{view_name}" '
                "WHERE lower(status) = 'fail'"
            ).fetchall()
    except duckdb.Error as e:
        return ([f"Validation view query error for '{view_name}': {e}"], True)

    if rows:
        msgs: list[str] = []
        evidence_views: set[str] = set()
        if has_evidence_view:
            for _status, msg, ev in rows:
                msgs.append(str(msg))
                if ev:
                    evidence_views.add(str(ev))
        else:
            msgs = [str(msg) for _status, msg in rows]

        count = len(msgs)
        header = f"Failures in '{view_name}' ({count})"
        if evidence_views:
            header += f" [evidence: {', '.join(sorted(evidence_views))}]"

        sample = msgs[:MAX_INLINE_MESSAGES]
        detail = "\n".join(f"  {m}" for m in sample)
        if count > MAX_INLINE_MESSAGES:
            detail += f"\n  ... and {count - MAX_INLINE_MESSAGES} more"

        return ([f"{header}:\n{detail}"], False)

    return ([], False)


@dataclass
class Task:
    """A single unit of work in a workspace.

    Attributes:
        name: Unique identifier, also used as namespace prefix for intermediate views.
        inputs: Table/view names this task reads from. These must exist before the task runs.
        outputs: View names this task must produce. Other tasks can depend on these.
        Validation views: optional deterministic SQL that creates one or more views
            named '{name}__validation' and/or '{name}__validation_*'. Any row with
            lower(status)='fail' causes the task to fail.
        output_columns: Optional schema check. Maps view_name -> list of required column
            names. Validation fails if a view is missing any declared column.
        prompt: Objective text used for LLM-driven transform tasks.
        validate_sql: Deterministic SQL to create validation views.
    """

    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    output_columns: dict[str, list[str]] = field(default_factory=dict)

    # Deterministic SQL (views/macros only). If provided, the workspace
    # harness executes these statements directly.
    sql: str = ""

    # LLM-driven transform prompt.
    prompt: str = ""

    # Deterministic SQL used to create validation views.
    validate_sql: str = ""

    def transform_mode(self) -> str:
        """Return execution mode: 'sql' or 'prompt'.

        Spec parsing enforces exactly one of (sql, prompt) is provided.
        """
        return "sql" if self.sql else "prompt"

    def has_validation(self) -> bool:
        return bool(self.validate_sql and self.validate_sql.strip())

    def sql_statements(self) -> list[str]:
        """Return SQL statements for sql transform tasks."""
        return split_sql_statements(self.sql)

    def validate_sql_statements(self) -> list[str]:
        """Return SQL statements for validation SQL."""
        return split_sql_statements(self.validate_sql)

    def validation_view_names(self) -> list[str]:
        """Return validation view names derived from validate_sql."""
        if not self.has_validation():
            return []
        names: list[str] = []
        for stmt in self.validate_sql_statements():
            name = extract_create_name(stmt)
            if name and is_validation_view_for_task(name, self.name):
                names.append(name)
        return sorted(set(names))

    def validate_transform(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        """Validate transform outputs. Returns error messages (empty = pass).

        Checks run in order, short-circuiting on first failure:
        1. All declared output views/tables exist.
        2. Output views/tables have required columns (if output_columns specified).

        Outputs may be views (during task execution) or tables (after
        materialization), so both catalogs are checked.
        """
        # 1. Check all declared outputs exist (views or materialized tables)
        existing_views = {
            row[0]
            for row in conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE internal = false"
            ).fetchall()
        }
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT table_name FROM duckdb_tables() WHERE internal = false"
            ).fetchall()
        }
        existing = existing_views | existing_tables

        missing = [o for o in self.outputs if o not in existing]
        if missing:
            return [f"Required output view '{o}' was not created." for o in missing]

        # 2. Check output columns
        if self.output_columns:
            for view_name, required_cols in self.output_columns.items():
                if view_name not in existing:
                    continue  # Already caught in step 1
                actual_cols = get_column_names(conn, view_name)
                if not actual_cols:
                    # This shouldn't happen if it's in existing but stay safe
                    continue

                missing_cols = [c for c in required_cols if c not in actual_cols]
                if missing_cols:
                    label = "column" if len(missing_cols) == 1 else "columns"
                    return [
                        f"View '{view_name}' is missing required {label}: "
                        f"{', '.join(missing_cols)}. "
                        f"Actual columns: {', '.join(sorted(actual_cols))}"
                    ]

        return []

    def validate_validation_views(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        """Enforce validation views created by validate_sql.

        Each validation view must have columns:
        - status: pass|warn|fail (case-insensitive)
        - message: human-readable string

        Any row with lower(status)='fail' fails the task.
        """
        views = self.validation_view_names()
        if self.has_validation() and not views:
            return [
                f"validate_sql did not create any '{validation_view_prefix(self.name)}' views."
            ]
        if not views:
            return []

        errors: list[str] = []

        for view_name in views:
            view_errors, fatal = self._validate_one_validation_view(conn, view_name)
            if fatal and view_errors:
                return view_errors

            if view_errors:
                errors.extend(view_errors)

        return errors

    def validation_warnings(
        self, conn: duckdb.DuckDBPyConnection, limit: int | None = None
    ) -> tuple[int, list[str]]:
        """Return (total_warning_count, warning_messages) from validation views.

        Assumes validation views exist and are well-formed.
        Pass limit=None for no truncation.
        """
        views = self.validation_view_names()
        if not views:
            return 0, []

        warnings: list[str] = []
        total = 0
        remaining = None if limit is None else max(1, limit)

        for view_name in views:
            actual = get_column_names(conn, view_name)
            if not actual:
                continue

            actual_lower = {c.lower() for c in actual}
            if "status" not in actual_lower or "message" not in actual_lower:
                continue

            try:
                count_row = conn.execute(
                    f"SELECT COUNT(*) FROM \"{view_name}\" WHERE lower(status) = 'warn'"
                ).fetchone()
                view_count = int(count_row[0]) if count_row else 0
            except duckdb.Error:
                continue

            if view_count <= 0:
                continue

            total += view_count
            if remaining is not None and remaining <= 0:
                continue

            has_evidence_view = "evidence_view" in actual_lower

            try:
                if has_evidence_view:
                    query = (
                        f'SELECT status, message, evidence_view FROM "{view_name}" '
                        "WHERE lower(status) = 'warn'"
                    )
                else:
                    query = (
                        f'SELECT status, message FROM "{view_name}" '
                        "WHERE lower(status) = 'warn'"
                    )
                if remaining is not None:
                    query += f" LIMIT {remaining}"
                rows = conn.execute(query).fetchall()
            except duckdb.Error:
                continue

            if rows:
                msgs: list[str] = []
                evidence_views: set[str] = set()
                if has_evidence_view:
                    for _status, msg, ev in rows:
                        msgs.append(str(msg))
                        if ev:
                            evidence_views.add(str(ev))
                else:
                    msgs = [str(msg) for _status, msg in rows]

                header = f"Warnings via '{view_name}' ({view_count} row(s))"
                if evidence_views:
                    header += f" [evidence: {', '.join(sorted(evidence_views))}]"

                sample = msgs[:MAX_INLINE_MESSAGES]
                detail = "\n".join(f"  {m}" for m in sample)
                if view_count > MAX_INLINE_MESSAGES:
                    detail += f"\n  ... and {view_count - MAX_INLINE_MESSAGES} more"

                warnings.append(f"{header}:\n{detail}")
                if remaining is not None:
                    remaining -= len(msgs)

        return total, warnings

    def _validate_one_validation_view(
        self, conn: duckdb.DuckDBPyConnection, view_name: str
    ) -> tuple[list[str], bool]:
        """Validate a single validation view. Delegates to module-level function."""
        return validate_one_validation_view(conn, view_name)


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
