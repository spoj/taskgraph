"""Taskgraph workspace — core modules."""

from .agent_loop import run_agent_loop, AgentResult
from .api import OpenRouterClient, create_model_callable
from .agent import run_task_agent
from .task import Task, resolve_dag, resolve_task_deps, validate_task_graph
from .workspace import Workspace, WorkspaceResult
from .ingest import ingest_table, coerce_to_dataframe, get_schema_info_for_tables

__all__ = [
    # Agent loop
    "run_agent_loop",
    "AgentResult",
    # API client
    "OpenRouterClient",
    "create_model_callable",
    # Task agent
    "run_task_agent",
    # Task DAG
    "Task",
    "resolve_dag",
    "resolve_task_deps",
    "validate_task_graph",
    # Workspace orchestrator
    "Workspace",
    "WorkspaceResult",
    # Ingestion
    "ingest_table",
    "coerce_to_dataframe",
    "get_schema_info_for_tables",
]
