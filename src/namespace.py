"""Namespace enforcement for DDL name control.

A Namespace controls what names a node (or input validator) can use in
CREATE/DROP VIEW|MACRO statements.  Three factory methods encode the
project's naming conventions:

- ``Namespace.for_node(node)`` — node transform: ``{name}_*``
  prefix only, validation views forbidden.
- ``Namespace.for_validation(node)`` — validation SQL:
  ``{name}__validation_*`` prefix.
- ``Namespace.for_source_validation(input_name)`` — source node
  validation SQL:
  ``{input}__validation_*`` prefix.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from .task import is_validation_view

if TYPE_CHECKING:
    from .task import Node


class Namespace:
    """Controls what DDL names can be created/dropped.

    Owns the naming rules (allowed names, prefix, forbidden check) and
    provides a single ``check_name()`` entry point used by
    ``is_sql_allowed()`` in the agent module.

    Attributes:
        allowed_names: Explicit set of permitted names.
        prefix: Names starting with ``{prefix}_`` are also allowed.
        forbidden: Optional callable that returns True for names that
            should be blocked even if they match allowed/prefix.
        forbidden_msg: Human-readable explanation when ``forbidden``
            triggers.
    """

    __slots__ = ("allowed_names", "prefix", "forbidden", "forbidden_msg")

    def __init__(
        self,
        allowed_names: frozenset[str],
        prefix: str,
        *,
        forbidden: Callable[[str], bool] | None = None,
        forbidden_msg: str = "Name is forbidden in this context.",
    ) -> None:
        self.allowed_names = allowed_names
        self.prefix = prefix
        self.forbidden = forbidden
        self.forbidden_msg = forbidden_msg

    # --- Factory methods ---

    @classmethod
    def for_node(cls, node: Node) -> Namespace:
        """Namespace for node transform execution.

        Allows only ``{node.name}_*`` prefixed views (underscore required —
        bare ``{node.name}`` is NOT a valid view name).  Blocks validation
        views — those are created separately by the validation phase.
        """
        return cls(
            allowed_names=frozenset(),
            prefix=node.name,
            forbidden=lambda name: is_validation_view(name, node.name),
            forbidden_msg=(
                "Validation views cannot be created during transform; "
                "they are created by validation."
            ),
        )

    @classmethod
    def for_validation(cls, node: Node) -> Namespace:
        """Namespace for node validation SQL execution.

        Allows only ``{node.name}__validation_*`` prefixed names.
        """
        base = f"{node.name}__validation"
        return cls(
            allowed_names=frozenset(),
            prefix=base,
        )

    @classmethod
    def for_source_validation(cls, input_name: str) -> Namespace:
        """Namespace for source node validation SQL execution."""
        base = f"{input_name}__validation"
        return cls(allowed_names=frozenset(), prefix=base)

    # --- Name checking ---

    def is_name_allowed(self, name: str) -> bool:
        """Return True if *name* is permitted by this namespace."""
        return name in self.allowed_names or name.startswith(f"{self.prefix}_")

    def check_name(
        self, name: str | None, kind_label: str, action: str
    ) -> tuple[bool, str]:
        """Check if a DDL name is allowed.  Returns ``(ok, error_msg)``.

        Evaluation order:
        1. ``None`` name (extraction failed) → blocked.
        2. Forbidden check → blocked with ``forbidden_msg``.
        3. Namespace check → blocked with allowed-names summary.
        4. Otherwise → allowed.
        """
        if name is None:
            return (
                False,
                f"Could not extract {kind_label} name from {action} statement.",
            )
        if self.forbidden and self.forbidden(name):
            return False, f"Cannot {action} '{name}': {self.forbidden_msg}"
        if not self.is_name_allowed(name):
            return (
                False,
                f"Cannot {action} {kind_label} '{name}'. "
                f"Allowed names: {self.format_allowed()}",
            )
        return True, ""

    def format_allowed(self) -> str:
        """Format allowed names for error messages."""
        parts: list[str] = []
        if self.allowed_names:
            parts.append(", ".join(sorted(self.allowed_names)))
        if self.prefix:
            parts.append(f"{self.prefix}_* (prefix)")
        return " | ".join(parts) if parts else "(none)"

    def __repr__(self) -> str:
        return (
            f"Namespace(allowed_names={self.allowed_names!r}, prefix={self.prefix!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Namespace):
            return NotImplemented
        return (
            self.allowed_names == other.allowed_names
            and self.prefix == other.prefix
            and self.forbidden_msg == other.forbidden_msg
            # forbidden callables compared by identity only
        )
