"""Workspace spec loader.

A spec is a Python module defining NODES and optional EXPORTS.

NODES is a list of dicts, each defining a single node in the workspace DAG.
Node type is determined by which of ``source`` / ``sql`` / ``prompt`` is
present (exactly one required):

- **source** — data ingestion node (callable, raw data, or file path).
- **sql** — deterministic SQL transform.
- **prompt** — LLM-driven transform.

All nodes share: ``name``, ``depends_on``, and optional ``validate``.
Source nodes additionally: ``source``, ``columns``.
SQL/prompt nodes additionally: ``sql`` OR ``prompt``, ``output_columns``.

Validation is expressed as a mapping of ``check_name -> query``.
Each query is wrapped by Taskgraph into a validation view named:

  ``{node}__validation_{check_name}``
"""

import importlib
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from .ingest import (
    FileInput,
    is_supported_file_string,
    parse_file_path,
    parse_file_string,
)
from .task import Node, validate_node_name

from .sql_utils import SqlParseError, parse_one_statement

_SOURCE_FIELDS = {"name", "depends_on", "source", "columns", "validate"}
_SQL_PROMPT_FIELDS = {
    "name",
    "depends_on",
    "sql",
    "prompt",
    "output_columns",
    "validate",
}


def resolve_module_path(module_path: str) -> Path:
    """Resolve a module path to a source file path without importing it."""
    spec = importlib.util.find_spec(module_path)
    if spec is None or spec.origin is None:
        raise ValueError(f"Cannot resolve spec module: {module_path}")
    if spec.origin in {"built-in", "frozen"}:
        raise ValueError(f"Spec module has no source file: {module_path}")
    return Path(spec.origin).resolve()


def _resolve_source_path(source: Any, spec_dir: Path) -> Any:
    """Resolve a source value, turning file paths/strings into FileInput.

    """
    if isinstance(source, FileInput):
        return source
    if isinstance(source, Path):
        return parse_file_path(source, base_dir=spec_dir)
    if isinstance(source, str) and is_supported_file_string(source):
        return parse_file_string(source, base_dir=spec_dir)
    return source


def _parse_module(module: ModuleType) -> tuple[list[Node], dict[str, Any]]:
    """Parse NODES and EXPORTS from a loaded module.

    Returns ``(nodes, exports)`` where *nodes* is a list of :class:`Node`
    objects and *exports* is a dict of export functions.
    """
    if not hasattr(module, "NODES"):
        raise ValueError("Spec module must define NODES list.")
    raw_nodes = module.NODES
    if not isinstance(raw_nodes, list):
        raise ValueError("NODES must be a list.")

    spec_dir = Path(module.__file__).resolve().parent if module.__file__ else Path.cwd()

    nodes: list[Node] = []
    seen_names: set[str] = set()

    for i, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise ValueError(f"NODES[{i}] must be a dict, got {type(raw).__name__}.")

        name = raw.get("name")
        if not name:
            raise ValueError(f"NODES[{i}] is missing required 'name' field.")
        if not isinstance(name, str):
            raise ValueError(
                f"NODES[{i}] 'name' must be a string, got {type(name).__name__}."
            )
        name = name.strip()
        if not name:
            raise ValueError(f"NODES[{i}] 'name' must not be empty.")

        name_err = validate_node_name(name)
        if name_err:
            raise ValueError(name_err)

        if name in seen_names:
            raise ValueError(f"Duplicate node name: '{name}'.")
        seen_names.add(name)

        # Determine mode: exactly one of source / sql / prompt
        modes = [k for k in ("source", "sql", "prompt") if k in raw]
        if len(modes) != 1:
            if len(modes) == 0:
                raise ValueError(
                    f"Node '{name}' must have exactly one of: source, sql, prompt."
                )
            else:
                raise ValueError(
                    f"Node '{name}' has multiple modes: {modes}. Only one allowed."
                )

        mode = modes[0]

        # Field validation
        allowed = _SOURCE_FIELDS if mode == "source" else _SQL_PROMPT_FIELDS
        unknown = set(raw.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Node '{name}' has unknown fields: "
                f"{', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(sorted(allowed))}"
            )

        # Build Node kwargs
        kwargs: dict[str, Any] = {"name": name}

        if "depends_on" in raw:
            deps = raw["depends_on"]
            if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
                raise ValueError(
                    f"Node '{name}': depends_on must be a list of strings."
                )
            kwargs["depends_on"] = deps

        if mode == "source":
            kwargs["source"] = _resolve_source_path(raw["source"], spec_dir)

            if "columns" in raw:
                cols = raw["columns"]
                if not isinstance(cols, list) or not all(
                    isinstance(c, str) for c in cols
                ):
                    raise ValueError(
                        f"Node '{name}': columns must be a list of strings."
                    )
                kwargs["columns"] = cols

        elif mode in ("sql", "prompt"):
            val = raw[mode]
            if not isinstance(val, str):
                raise ValueError(f"Node '{name}': {mode} must be a non-empty string.")
            val = val.strip()
            if not val:
                raise ValueError(f"Node '{name}': {mode} must be a non-empty string.")
            kwargs[mode] = val

        if "output_columns" in raw:
            oc = raw["output_columns"]
            if not isinstance(oc, dict):
                raise ValueError(
                    f"Node '{name}': output_columns must be a dict "
                    f"mapping view_name -> [column_names], "
                    f"got {type(oc).__name__}"
                )
            for k, v in oc.items():
                if not isinstance(v, list) or not all(isinstance(c, str) for c in v):
                    raise ValueError(
                        f"Node '{name}': output_columns['{k}'] must be a list of strings."
                    )
            kwargs["output_columns"] = oc

        if "validate" in raw:
            val = raw["validate"]
            if not isinstance(val, dict):
                raise ValueError(
                    f"Node '{name}': validate must be a dict mapping suffix -> query, "
                    f"got {type(val).__name__}"
                )

            validate: dict[str, str] = {}
            for suffix, query in val.items():
                if not isinstance(suffix, str):
                    raise ValueError(
                        f"Node '{name}': validate keys must be strings (suffixes)."
                    )
                if not isinstance(query, str):
                    raise ValueError(
                        f"Node '{name}': validate['{suffix}'] must be a string query."
                    )
                suffix = suffix.strip()
                query = query.strip()
                if not suffix:
                    raise ValueError(
                        f"Node '{name}': validate keys must be non-empty strings (e.g. 'main')."
                    )
                if suffix and not re.match(r"^[A-Za-z0-9_]+$", suffix):
                    raise ValueError(
                        f"Node '{name}': validate suffix '{suffix}' must match [A-Za-z0-9_]+."
                    )
                if "__" in suffix:
                    raise ValueError(
                        f"Node '{name}': validate suffix '{suffix}' must not contain '__'."
                    )
                if not query:
                    raise ValueError(
                        f"Node '{name}': validate['{suffix}'] must be a non-empty query."
                    )

                # Ensure the query is syntactically valid inside CREATE VIEW.
                # Parsing does not bind names, so it can reference tables/views
                # that are created at runtime.
                wrapped = (
                    f"CREATE VIEW __tg_validate_tmp AS {query.rstrip(';').strip()}"
                )
                try:
                    parse_one_statement(wrapped)
                except SqlParseError as e:
                    raise ValueError(
                        f"Node '{name}': validate['{suffix}'] must be a single query usable in CREATE VIEW: {e}"
                    ) from e

                validate[suffix] = query

            if validate:
                kwargs["validate"] = validate

        nodes.append(Node(**kwargs))

    exports = getattr(module, "EXPORTS", {})
    if not isinstance(exports, dict):
        raise ValueError("EXPORTS must be a dict mapping path -> function.")
    return nodes, exports


def load_spec_from_module(module_path: str) -> tuple[list[Node], dict[str, Any]]:
    """Load a workspace spec from a module path.

    Returns ``(nodes, exports)`` where *nodes* is a list of :class:`Node`
    objects and *exports* is a dict of export functions.

    Raises ValueError if the spec is invalid.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(f"Cannot import spec module '{module_path}': {e}") from e
    except Exception as e:
        raise ValueError(f"Error loading spec module '{module_path}': {e}") from e
    return _parse_module(module)
