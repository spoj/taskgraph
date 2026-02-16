"""Workspace spec loader — shared between CLI and web interface.

A spec is a Python module defining INPUTS, TASKS, and optional EXPORTS.

INPUTS values can be:
- Simple: callable or raw data
- Rich: dict with "data" key + optional "columns" and "validate_sql"

Task prompts must be strings.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .task import Task


def _load_module(spec_path: Path) -> ModuleType:
    """Load a Python module from a file path."""
    spec_dir = str(spec_path.resolve().parent)
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)

    spec = importlib.util.spec_from_file_location("workspace_spec", spec_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load spec from {spec_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_module_path(module_path: str) -> Path:
    """Resolve a module path to a source file path without importing it."""
    spec = importlib.util.find_spec(module_path)
    if spec is None or spec.origin is None:
        raise ValueError(f"Cannot resolve spec module: {module_path}")
    if spec.origin in {"built-in", "frozen"}:
        raise ValueError(f"Spec module has no source file: {module_path}")
    return Path(spec.origin).resolve()


def _parse_module(module: ModuleType) -> dict[str, Any]:
    """Parse INPUTS, TASKS, EXPORTS from a loaded module.

    Returns dict with 'inputs', 'tasks', 'exports',
    'input_columns', 'input_validate_sql' keys.
    """
    if not hasattr(module, "INPUTS"):
        raise ValueError("Spec must define INPUTS")
    if not hasattr(module, "TASKS"):
        raise ValueError("Spec must define TASKS")

    raw_inputs = module.INPUTS

    # Parse INPUTS: separate data from validation metadata
    inputs: dict[str, Any] = {}
    input_columns: dict[str, list[str]] = {}
    input_validate_sql: dict[str, list[str]] = {}

    for name, value in raw_inputs.items():
        if isinstance(value, dict) and "data" in value:
            # Rich format: {"data": ..., "columns": [...], "validate_sql": [...]}
            inputs[name] = value["data"]
            if "columns" in value:
                input_columns[name] = value["columns"]
            if "validate_sql" in value:
                input_validate_sql[name] = value["validate_sql"]
        else:
            # Simple format: callable or raw data
            inputs[name] = value

    # Accept tasks as dicts or Task objects
    tasks = []
    for t in module.TASKS:
        if isinstance(t, dict):
            t = dict(t)  # shallow copy to avoid mutating the original
            task_name = t.get("name", "")
            if "prompt" in t and not isinstance(t["prompt"], str):
                raise ValueError(
                    f"Task '{task_name}' prompt must be a string, "
                    f"got {type(t['prompt']).__name__}"
                )
            tasks.append(Task(**t))
        elif isinstance(t, Task):
            tasks.append(t)
        else:
            raise ValueError(
                f"Each task must be a dict or Task, got {type(t).__name__}"
            )

    return {
        "inputs": inputs,
        "tasks": tasks,
        "exports": getattr(module, "EXPORTS", {}),
        "input_columns": input_columns,
        "input_validate_sql": input_validate_sql,
    }


def load_spec_from_module(module_path: str) -> dict[str, Any]:
    """Load a workspace spec from a module path.

    Returns dict with 'inputs', 'tasks', 'exports',
    'input_columns', 'input_validate_sql' keys.

    Raises ValueError if the spec is invalid.
    """
    module = importlib.import_module(module_path)
    result = _parse_module(module)
    return result
