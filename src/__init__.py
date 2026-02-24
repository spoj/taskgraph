"""Taskgraph workspace — core modules."""

from .agent_loop import run_agent_loop, AgentResult
from .api import OpenRouterClient, create_model_callable
from .agent import run_node_agent
from .task import Node, resolve_dag, resolve_deps, validate_graph
from .workspace import Workspace, WorkspaceResult
from .ingest import (
    ingest_table,
    coerce_to_dataframe,
    llm,
    LLMSource,
    llm_pages,
    LLMPagesSource,
)

__all__ = [
    # Agent loop
    "run_agent_loop",
    "AgentResult",
    # API client
    "OpenRouterClient",
    "create_model_callable",
    # Node agent
    "run_node_agent",
    # Node DAG
    "Node",
    "resolve_dag",
    "resolve_deps",
    "validate_graph",
    # Workspace orchestrator
    "Workspace",
    "WorkspaceResult",
    # Ingestion
    "ingest_table",
    "coerce_to_dataframe",
    "llm",
    "LLMSource",
    "llm_pages",
    "LLMPagesSource",
]
