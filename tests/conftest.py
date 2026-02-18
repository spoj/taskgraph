"""Shared fixtures and helpers for Taskgraph test suite."""

import importlib
import sys
import uuid
from pathlib import Path

import duckdb
import pytest

from src.agent import init_trace_table
from src.task import Node


@pytest.fixture
def conn():
    """In-memory DuckDB connection for each test."""
    c = duckdb.connect(":memory:")
    init_trace_table(c)
    yield c
    c.close()


def _make_node(**kwargs) -> Node:
    """Helper to create a Node with defaults."""
    defaults = {
        "name": "t",
        "sql": "SELECT 1",
    }
    defaults.update(kwargs)
    return Node(**defaults)


def _write_spec_module(tmp_path: Path, source: str) -> str:
    """Create a temporary spec module and return its module path."""
    module_name = f"spec_{uuid.uuid4().hex}"
    module_dir = tmp_path / module_name
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text(source)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    return module_name
