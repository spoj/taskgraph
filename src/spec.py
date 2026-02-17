"""Workspace spec loader.

A spec is a Python module defining INPUTS, TASKS, and optional EXPORTS.

INPUTS values can be:
- Simple: callable, raw data, or file path
- Rich: dict with "source" key + optional "columns" and "validate_sql"

Task prompt values must be strings.
"""

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from .ingest import (
    FileInput,
    is_supported_file_string,
    parse_file_path,
    parse_file_string,
)
from .task import Task


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
    spec_dir = Path(module.__file__).resolve().parent if module.__file__ else Path.cwd()

    # Parse INPUTS: separate data from validation metadata
    inputs: dict[str, Any] = {}
    input_columns: dict[str, list[str]] = {}
    input_validate_sql: dict[str, str] = {}

    def _resolve_source(source: Any) -> Any:
        if isinstance(source, FileInput):
            return source
        if isinstance(source, Path):
            return parse_file_path(source, base_dir=spec_dir)
        if isinstance(source, str) and is_supported_file_string(source):
            return parse_file_string(source, base_dir=spec_dir)
        return source

    for name, value in raw_inputs.items():
        if isinstance(value, dict) and "source" in value:
            # Rich format: {"source": ..., "columns": [...], "validate_sql": [...]}
            inputs[name] = _resolve_source(value["source"])
            if "columns" in value:
                input_columns[name] = value["columns"]
            if "validate_sql" in value:
                vs = value["validate_sql"]
                if not isinstance(vs, str):
                    raise ValueError(
                        f"Input '{name}' validate_sql must be a string, "
                        f"got {type(vs).__name__}"
                    )
                vs = vs.strip()
                if vs:
                    input_validate_sql[name] = vs
            continue

        if isinstance(value, dict) and "data" in value:
            raise ValueError(
                f"Input '{name}' uses deprecated key 'data'; use 'source' instead"
            )

        # Simple format: callable or raw data
        inputs[name] = _resolve_source(value)

    # Accept tasks as dicts or Task objects
    tasks = []

    def _normalize_text(task_name: str, value: Any, key: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Task '{task_name}' {key} must be a string")
        s2 = value.strip()
        if not s2:
            raise ValueError(f"Task '{task_name}' {key} must not be empty")
        return s2

    for t in module.TASKS:
        if isinstance(t, dict):
            t = dict(t)  # shallow copy to avoid mutating the original
            task_name = t.get("name", "")

            # Deterministic SQL (optional). If present, the harness will
            # execute it directly (no LLM). Accept str.
            if "sql" in t:
                t["sql"] = _normalize_text(task_name, t.get("sql", ""), "sql")

            # Prompt-driven transform (optional).
            if "prompt" in t:
                t["prompt"] = _normalize_text(task_name, t.get("prompt", ""), "prompt")

            # Validation SQL (optional).
            if "validate_sql" in t:
                t["validate_sql"] = _normalize_text(
                    task_name, t.get("validate_sql", ""), "validate_sql"
                )

            sql_statements = (t.get("sql") or "").strip()
            prompt_text = (t.get("prompt") or "").strip()

            if sql_statements and prompt_text:
                raise ValueError(
                    f"Task '{task_name}' must not specify both 'sql' and 'prompt'"
                )
            if not sql_statements and not prompt_text:
                raise ValueError(
                    f"Task '{task_name}' must specify exactly one of 'sql' or 'prompt'"
                )

            tasks.append(Task(**t))
        elif isinstance(t, Task):
            tasks.append(t)
        else:
            raise ValueError(
                f"Each task must be a dict or Task, got {type(t).__name__}"
            )

    for t in tasks:
        if not isinstance(t.sql, str):
            raise ValueError(
                f"Task '{t.name}' sql must be a string, got {type(t.sql).__name__}"
            )
        if not isinstance(t.prompt, str):
            raise ValueError(
                f"Task '{t.name}' prompt must be a string, got {type(t.prompt).__name__}"
            )
        if not isinstance(t.validate_sql, str):
            raise ValueError(
                f"Task '{t.name}' validate_sql must be a string, got {type(t.validate_sql).__name__}"
            )
        sql_statements = (t.sql or "").strip()
        prompt_text = (t.prompt or "").strip()
        if sql_statements and prompt_text:
            raise ValueError(
                f"Task '{t.name}' must not specify both 'sql' and 'prompt'"
            )
        if not sql_statements and not prompt_text:
            raise ValueError(
                f"Task '{t.name}' must specify exactly one of 'sql' or 'prompt'"
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
