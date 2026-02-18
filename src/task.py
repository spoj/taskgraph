"""Node definition and DAG resolution for workspaces.

A Node is the single unit of work.  Every node has a ``name`` and an
execution mode — exactly one of ``source``, ``sql``, or ``prompt``:

- **source** — ingests data (callable, raw data, or file path) into a
  table named ``{name}``.
- **sql** — executes deterministic SQL that creates ``{name}_*`` views.
- **prompt** — runs an LLM agent that creates ``{name}_*`` views.

Nodes form a DAG via explicit ``depends_on`` edges.  ``resolve_deps()``
builds the raw dependency graph; ``resolve_dag()`` produces topo-sorted
layers for display / scheduling.

Example::

    nodes = [
        Node(
            name="raw_rows",
            source="data/rows.csv",
            columns=["id", "amount"],
        ),
        Node(
            name="prep",
            depends_on=["raw_rows"],
            sql="CREATE OR REPLACE VIEW prep_clean AS SELECT * FROM raw_rows WHERE amount > 0",
        ),
        Node(
            name="match",
            depends_on=["prep"],
            prompt="Match cleaned rows against a reference table...",
            output_columns={"match_results": ["id", "score"]},
        ),
    ]

    deps = resolve_deps(nodes)
    # {"raw_rows": set(), "prep": set(), "match": {"prep"}}

    layers = resolve_dag(nodes)
    # layers[0] = [raw_rows, ...]   (no deps)
    # layers[1] = [prep]            (depends on raw_rows)
    # layers[2] = [match]           (depends on prep)
"""

from __future__ import annotations

import duckdb
from dataclasses import dataclass, field
from typing import Any

from .catalog import list_tables, list_views
from .sql_utils import (
    split_sql_statements,
    get_column_names,
)

_VALIDATION_VIEW_REQUIRED_COLS = ["status", "message"]
_VALIDATION_STATUS_ALLOWED = {"pass", "warn", "fail"}
MAX_INLINE_MESSAGES = 20  # Cap on messages shown inline in validation/warning output

# Sentinel for "no source provided" (since None/0/[]/etc. could be valid data).
_NO_SOURCE = object()


def validation_view_prefix(node_name: str) -> str:
    return f"{node_name}__validation"


def is_validation_view(view_name: str, node_name: str) -> bool:
    """Return True if *view_name* is a validation view for *node_name*.

    Convention: ``{node_name}__validation`` and ``{node_name}__validation_*``
    """
    prefix = validation_view_prefix(node_name)
    return view_name == prefix or view_name.startswith(prefix + "_")


def validate_node_name(name: str) -> str | None:
    """Validate a node name, returning an error message or None."""
    if "__" in name:
        return (
            f"Node name '{name}' must not contain '__' "
            "(reserved for __validation views)."
        )
    return None


def discover_validation_objects(
    conn: duckdb.DuckDBPyConnection, node_name: str
) -> list[str]:
    """Discover validation objects for *node_name* from the catalog.

    Includes both views and tables so this works both before and after
    materialization.
    """
    base = validation_view_prefix(node_name)
    prefix = base + "_"
    names: set[str] = set()

    for name in list_views(conn, exclude_prefixes=()):
        if name == base or name.startswith(prefix):
            names.add(name)

    for name in list_tables(conn, exclude_prefixes=()):
        if name == base or name.startswith(prefix):
            names.add(name)

    return sorted(names)


def _query_validation_rows(
    conn: duckdb.DuckDBPyConnection,
    view_name: str,
    status_filter: str,
    limit: int | None = None,
) -> tuple[list[str], set[str]]:
    """Query a validation view for rows matching a status, return (messages, evidence_views).

    Assumes the view has 'status' and 'message' columns (caller must verify).
    """
    actual_cols = get_column_names(conn, view_name)
    has_evidence = "evidence_view" in {c.lower() for c in actual_cols}

    cols = "status, message, evidence_view" if has_evidence else "status, message"
    query = (
        f"SELECT {cols} FROM \"{view_name}\" WHERE lower(status) = '{status_filter}'"
    )
    if limit is not None:
        query += f" LIMIT {limit}"

    try:
        rows = conn.execute(query).fetchall()
    except duckdb.Error:
        return [], set()

    msgs: list[str] = []
    evidence_views: set[str] = set()
    for row in rows:
        msgs.append(str(row[1]))
        if has_evidence and row[2]:
            evidence_views.add(str(row[2]))
    return msgs, evidence_views


def _format_validation_block(
    view_name: str, msgs: list[str], evidence_views: set[str], total: int, label: str
) -> str:
    """Format a validation message block with header and sample."""
    header = f"{label} in '{view_name}' ({total})"
    if evidence_views:
        header += f" [evidence: {', '.join(sorted(evidence_views))}]"
    sample = msgs[:MAX_INLINE_MESSAGES]
    detail = "\n".join(f"  {m}" for m in sample)
    if total > MAX_INLINE_MESSAGES:
        detail += f"\n  ... and {total - MAX_INLINE_MESSAGES} more"
    return f"{header}:\n{detail}"


def validate_one_validation_view(
    conn: duckdb.DuckDBPyConnection, view_name: str
) -> tuple[list[str], bool]:
    """Validate a single validation view.

    Returns (errors, fatal).
    - fatal=True indicates a schema/query/contract problem.
    - fatal=False with errors means validation 'fail' rows were found.
    """
    actual_cols = get_column_names(conn, view_name)
    if not actual_cols:
        return ([f"Validation view '{view_name}' not found."], True)

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

    msgs, evidence_views = _query_validation_rows(conn, view_name, "fail")
    if msgs:
        return (
            [
                _format_validation_block(
                    view_name, msgs, evidence_views, len(msgs), "Failures"
                )
            ],
            False,
        )

    return ([], False)


@dataclass
class Node:
    """A single node in a workspace DAG.

    Exactly one of ``source``, ``sql``, or ``prompt`` must be set,
    determining the node type:

    - **source**: Data ingestion node.  ``source`` holds a callable,
      raw data (list[dict] / dict[str,list]), or a file path string.
      The ingested table is named ``{name}``.
    - **sql**: Deterministic SQL transform.  Creates views/macros
      in the ``{name}_*`` namespace.
    - **prompt**: LLM-driven transform.  The agent creates views
      in the ``{name}_*`` namespace.

    Attributes:
        name: Unique identifier.  For source nodes this becomes the
            table name; for sql/prompt nodes it is the namespace prefix
            (views must be ``{name}_*``).
        depends_on: Node names that must complete before this node runs.
        source: Data source for ingestion (callable / data / file path).
            Use the sentinel ``_NO_SOURCE`` default to distinguish
            "not provided" from ``None``-as-valid-data.
        sql: Deterministic SQL statements.
        prompt: LLM objective text.
        columns: (source nodes) Required column names on the ingested
            table.  Checked after ingestion.
        output_columns: (sql/prompt nodes) Maps view_name → required
            column list.  Keys define which views must exist; values
            define required columns per view.
        validate_sql: SQL to create ``{name}__validation*`` views.
            Works for all node types.
    """

    name: str
    depends_on: list[str] = field(default_factory=list)

    # Exactly one of these three:
    source: Any = field(default=_NO_SOURCE, repr=False)
    sql: str = ""
    prompt: str = ""

    # Schema validation
    columns: list[str] = field(default_factory=list)
    output_columns: dict[str, list[str]] = field(default_factory=dict)

    # Validation SQL (all node types)
    validate_sql: str = ""

    def node_type(self) -> str:
        """Return ``'source'``, ``'sql'``, or ``'prompt'``."""
        if self.source is not _NO_SOURCE:
            return "source"
        if self.sql:
            return "sql"
        return "prompt"

    def is_source(self) -> bool:
        return self.source is not _NO_SOURCE

    def has_validation(self) -> bool:
        return bool(self.validate_sql and self.validate_sql.strip())

    def sql_statements(self) -> list[str]:
        """Return SQL statements for sql transform nodes."""
        return split_sql_statements(self.sql)

    def validate_sql_statements(self) -> list[str]:
        """Return SQL statements for validation SQL."""
        return split_sql_statements(self.validate_sql)

    # ------------------------------------------------------------------
    # Output validation (unified for all node types)
    # ------------------------------------------------------------------

    def _required_schema(self) -> dict[str, list[str]]:
        """Build unified schema map: ``{name: [required_columns]}``.

        Source nodes: ``{self.name: self.columns}``.
        SQL/prompt nodes: ``self.output_columns``.
        """
        if self.is_source():
            return {self.name: self.columns} if self.columns else {}
        return self.output_columns

    def validate_outputs(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        """Validate node outputs: required tables/views exist with required columns."""
        schema = self._required_schema()
        if not schema:
            return []

        errors: list[str] = []
        for name, required_cols in schema.items():
            actual_cols = get_column_names(conn, name)
            if not actual_cols:
                errors.append(f"Required output '{name}' was not created.")
                continue
            if not required_cols:
                continue
            # Hide internal _row_id from display (added by ingestion)
            display_cols = [c for c in actual_cols if c != "_row_id"]
            actual_lower = {c.lower() for c in actual_cols}
            missing = [c for c in required_cols if c.lower() not in actual_lower]
            if missing:
                label = "column" if len(missing) == 1 else "columns"
                errors.append(
                    f"Output '{name}' is missing required {label}: "
                    f"{', '.join(missing)}. "
                    f"Actual columns: {', '.join(sorted(display_cols))}"
                )
        return errors

    # ------------------------------------------------------------------
    # Validation views (same for all node types)
    # ------------------------------------------------------------------

    def validate_validation_views(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        """Enforce validation views created by validate_sql.

        Each validation view must have columns:
        - status: pass|warn|fail (case-insensitive)
        - message: human-readable string

        Any row with lower(status)='fail' fails the node.
        """
        if not self.has_validation():
            return []

        views = discover_validation_objects(conn, self.name)
        if not views:
            return [
                f"validate_sql did not create any '{validation_view_prefix(self.name)}' views."
            ]

        errors: list[str] = []

        for view_name in views:
            view_errors, fatal = validate_one_validation_view(conn, view_name)
            if fatal and view_errors:
                return view_errors

            if view_errors:
                errors.extend(view_errors)

        return errors

    def validation_warnings(
        self, conn: duckdb.DuckDBPyConnection, limit: int | None = None
    ) -> tuple[int, list[str]]:
        """Return (total_warning_count, warning_messages) from validation views."""
        if not self.has_validation():
            return 0, []

        views = discover_validation_objects(conn, self.name)
        if not views:
            return 0, []

        warnings: list[str] = []
        total = 0
        remaining = None if limit is None else max(1, limit)

        for view_name in views:
            actual = get_column_names(conn, view_name)
            if not actual or not {"status", "message"} <= {c.lower() for c in actual}:
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

            msgs, evidence_views = _query_validation_rows(
                conn, view_name, "warn", limit=remaining
            )
            if msgs:
                warnings.append(
                    _format_validation_block(
                        view_name, msgs, evidence_views, view_count, "Warnings via"
                    )
                )
                if remaining is not None:
                    remaining -= len(msgs)

        return total, warnings


# ---------------------------------------------------------------------------
# DAG resolution
# ---------------------------------------------------------------------------


def resolve_deps(nodes: list[Node]) -> dict[str, set[str]]:
    """Build the dependency graph for all nodes.

    Returns ``{node_name: set_of_node_names_it_depends_on}``.
    Only references to other nodes in *nodes* are included; unknown
    names in ``depends_on`` are silently dropped (caught by
    ``validate_graph`` instead).
    """
    all_names = {n.name for n in nodes}
    return {n.name: {d for d in n.depends_on if d in all_names} for n in nodes}


def resolve_dag(nodes: list[Node]) -> list[list[Node]]:
    """Topologically sort nodes into execution layers.

    Nodes in the same layer have no dependencies on each other and can
    run concurrently.  Layer 0 has no dependencies, layer 1 depends
    only on layer 0, etc.

    Raises ValueError if a cycle is detected.
    """
    node_by_name = {n.name: n for n in nodes}
    deps = {name: set(d) for name, d in resolve_deps(nodes).items()}

    # Kahn's algorithm producing layers
    in_degree = {name: len(d) for name, d in deps.items()}
    layers: list[list[Node]] = []

    remaining = set(node_by_name.keys())

    while remaining:
        layer_names = sorted(n for n in remaining if in_degree[n] == 0)

        if not layer_names:
            cycle_members = sorted(remaining)
            raise ValueError(f"Dependency cycle detected among nodes: {cycle_members}")

        layer = [node_by_name[n] for n in layer_names]
        layers.append(layer)

        for name in layer_names:
            remaining.remove(name)
            for other in remaining:
                if name in deps[other]:
                    deps[other].remove(name)
                    in_degree[other] -= 1

    return layers


def validate_graph(nodes: list[Node]) -> list[str]:
    """Validate the node graph structure.

    Checks:
    1. All ``depends_on`` references point to existing node names.
    2. No cycles (delegated to ``resolve_dag``).
    3. ``output_columns`` keys start with ``{name}_`` (namespace).
    4. No node name is a prefix of another (namespace collision).
    5. Source nodes don't have ``output_columns``; sql/prompt nodes
       don't have ``columns``.
    """
    errors: list[str] = []
    all_names = {n.name for n in nodes}

    # 0. Node name restrictions
    for n in nodes:
        msg = validate_node_name(n.name)
        if msg:
            errors.append(msg)

    # 0b. Exactly one mode must be set: source/sql/prompt
    for n in nodes:
        modes: list[str] = []
        if n.is_source():
            modes.append("source")
        if n.sql and n.sql.strip():
            modes.append("sql")
        if n.prompt and n.prompt.strip():
            modes.append("prompt")
        if len(modes) != 1:
            errors.append(
                f"Node '{n.name}' must have exactly one of: source, sql, prompt. "
                f"Found: {modes or '(none)'}"
            )

    # 1. depends_on refs
    for n in nodes:
        for dep in n.depends_on:
            if dep not in all_names:
                errors.append(
                    f"Node '{n.name}' depends on '{dep}' which is not a known node."
                )

    # 2. output_columns keys start with {name}_
    for n in nodes:
        if not n.output_columns:
            continue
        prefix = f"{n.name}_"
        for key in n.output_columns:
            if not key.startswith(prefix):
                errors.append(
                    f"Node '{n.name}' output_columns key '{key}' must start "
                    f"with '{prefix}' (namespace enforcement)."
                )

    # 3. No name is a prefix of another
    sorted_names = sorted(all_names)
    for i, name_a in enumerate(sorted_names):
        for name_b in sorted_names[i + 1 :]:
            if name_b.startswith(name_a + "_"):
                errors.append(
                    f"Node name '{name_a}' is a prefix of '{name_b}'. "
                    f"This would cause namespace collisions "
                    f"('{name_b}_foo' matches both '{name_a}_*' and '{name_b}_*')."
                )

    # 4. Cross-type field misuse
    for n in nodes:
        if n.is_source() and n.output_columns:
            errors.append(
                f"Source node '{n.name}' should not have output_columns "
                f"(use 'columns' for source schema validation)."
            )
        if not n.is_source() and n.columns:
            errors.append(
                f"Node '{n.name}' (type={n.node_type()}) should not have "
                f"'columns' (use 'output_columns' for sql/prompt nodes)."
            )

    # 5. Cycles (only if no ref errors)
    if not errors:
        try:
            resolve_dag(nodes)
        except ValueError as e:
            errors.append(str(e))

    return errors
