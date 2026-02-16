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
            intent="Clean and normalize raw rows...",
            sql="CREATE OR REPLACE VIEW rows_clean AS SELECT ...",
        ),
        Task(
            name="match",
            inputs=["rows_clean", "reference"],
            outputs=["matches", "match_summary", "match__validation"],
            intent="Match normalized rows against a reference table...",
            sql="CREATE OR REPLACE VIEW matches AS SELECT ...",
            # Create a view named 'match__validation' with status/message rows.
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


_VALIDATION_VIEW_REQUIRED_COLS = ["status", "message"]
_VALIDATION_STATUS_ALLOWED = {"pass", "warn", "fail"}
_MAX_INLINE_MESSAGES = 20  # Cap messages shown inline in validation/warning output


def validation_view_prefix(task_name: str) -> str:
    return f"{task_name}__validation"


def is_validation_view_for_task(view_name: str, task_name: str) -> bool:
    """Return True if view_name is a validation view for task_name.

    Convention: '{task_name}__validation' and '{task_name}__validation_*'
    """
    prefix = validation_view_prefix(task_name)
    return view_name == prefix or view_name.startswith(prefix + "_")


def validation_outputs(task: "Task") -> list[str]:
    """Return declared validation views for a task (outputs only)."""
    return sorted(
        [o for o in task.outputs if is_validation_view_for_task(o, task.name)]
    )


@dataclass
class Task:
    """A single unit of work in a workspace.

    Attributes:
        name: Unique identifier, also used as namespace prefix for intermediate views.
        inputs: Table/view names this task reads from. These must exist before the task runs.
        outputs: View names this task must produce. Other tasks can depend on these.
        Validation views: a task may declare one or more outputs named
            '{name}__validation' and/or '{name}__validation_*'. If present, they are
            enforced after the task runs. Any row with lower(status)='fail' causes the
            task to fail.
        output_columns: Optional schema check. Maps view_name -> list of required column
            names. Validation fails if a view is missing any declared column.
        intent: Objective text used when repairing failed SQL tasks.
        repair_on_warn: If True (default), warnings in validation views trigger
            LLM repair. Set to False for monitoring-only validation views where
            you want to log warnings without spending tokens on repair.
    """

    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    output_columns: dict[str, list[str]] = field(default_factory=dict)

    # Deterministic SQL (views/macros only). If provided, the workspace
    # harness executes these statements directly; on failure it may
    # optionally attempt LLM repair.
    sql: str = ""

    # Immutable deterministic SQL (no LLM repair). Same constraints as sql.
    sql_strict: str = ""

    # Objective for repair mode when sql fails.
    intent: str = ""

    # If True, validation warnings trigger LLM repair.
    repair_on_warn: bool = True

    def run_mode(self) -> str:
        """Return execution mode: 'sql_strict' or 'sql'.

        Spec parsing enforces exactly one of (sql_strict, sql) is provided.
        """
        return "sql_strict" if self.sql_strict else "sql"

    def sql_statements(self) -> list[str]:
        """Return SQL statements for sql/sql_strict tasks."""
        sql_text = self.sql_strict or self.sql
        sql_text = (sql_text or "").strip()
        if not sql_text:
            return []
        conn = duckdb.connect()
        try:
            statements = conn.extract_statements(sql_text)
        except duckdb.Error:
            return [sql_text]
        finally:
            conn.close()
        return [s.query.strip() for s in statements if s.query.strip()]

    def validate(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        """Run validation. Returns error messages (empty = pass).

        Checks run in order, short-circuiting on first failure:
        1. All declared output views exist.
        2. Output views have required columns (if output_columns specified).
        3. If validation views are declared in outputs, enforce them.
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

        # 3. Enforce validation view(s) if declared
        validation_errors = self._validate_validation_views(conn)
        if validation_errors:
            return validation_errors

        return []

    def _validate_validation_views(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        """Enforce declared validation view outputs.

        Each validation view must have columns:
        - status: pass|warn|fail (case-insensitive)
        - message: human-readable string

        Any row with lower(status)='fail' fails the task.
        """
        views = validation_outputs(self)
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
    ) -> list[str]:
        """Return warning messages from validation views.

        Assumes validation views exist and are well-formed.
        Pass limit=None for no truncation.
        """
        views = validation_outputs(self)
        if not views:
            return []

        warnings: list[str] = []
        remaining = None if limit is None else max(1, limit)

        for view_name in views:
            if remaining is not None and remaining <= 0:
                break

            try:
                cols = [
                    row[0]
                    for row in conn.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                        [view_name],
                    ).fetchall()
                ]
            except duckdb.Error:
                continue

            actual = {c.lower() for c in cols}
            if "status" not in actual or "message" not in actual:
                continue

            has_evidence_view = "evidence_view" in actual

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

                count = len(msgs)
                header = f"Warnings via '{view_name}' ({count} row(s))"
                if evidence_views:
                    header += f" [evidence: {', '.join(sorted(evidence_views))}]"

                sample = msgs[:_MAX_INLINE_MESSAGES]
                detail = "\n".join(f"  {m}" for m in sample)
                if count > _MAX_INLINE_MESSAGES:
                    detail += f"\n  ... and {count - _MAX_INLINE_MESSAGES} more"

                warnings.append(f"{header}:\n{detail}")
                if remaining is not None:
                    remaining -= count

        return warnings

    def _validate_one_validation_view(
        self, conn: duckdb.DuckDBPyConnection, view_name: str
    ) -> tuple[list[str], bool]:
        """Validate a single validation view.

        Returns (errors, fatal).
        - fatal=True indicates a schema/query/contract problem that should stop immediately.
        """
        try:
            cols = [
                row[0]
                for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                    [view_name],
                ).fetchall()
            ]
        except duckdb.Error as e:
            return (
                [f"Validation view schema check error for '{view_name}': {e}"],
                True,
            )

        actual = {c.lower() for c in cols}
        missing = [c for c in _VALIDATION_VIEW_REQUIRED_COLS if c not in actual]
        if missing:
            return (
                [
                    f"Validation view '{view_name}' is missing required column(s): "
                    f"{', '.join(missing)}. Actual columns: {', '.join(cols)}"
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

        has_evidence_view = "evidence_view" in actual

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
            header = f"Fail rows in '{view_name}' ({count})"
            if evidence_views:
                header += f" [evidence: {', '.join(sorted(evidence_views))}]"

            sample = msgs[:_MAX_INLINE_MESSAGES]
            detail = "\n".join(f"  {m}" for m in sample)
            if count > _MAX_INLINE_MESSAGES:
                detail += f"\n  ... and {count - _MAX_INLINE_MESSAGES} more"

            return ([f"{header}:\n{detail}"], False)

        return ([], False)


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
