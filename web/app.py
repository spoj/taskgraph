"""Taskgraph web interface — upload files, pick a spec, run workspace, download results.

Usage:
    uvicorn web.app:app --reload
    # or: python -m web.app
"""

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.api import OpenRouterClient
from src.spec import load_spec_from_module, resolve_module_path
from src.spec_repo import get_spec_repo_info
from src.workspace import Workspace

log = logging.getLogger(__name__)

# --- Run state ---

# Each run gets a unique ID. Stores: {run_id: RunState}
_runs: dict[str, dict[str, Any]] = {}

# Built-in specs (module paths)
BUILTIN_SPECS = {
    "diamond_dag": "tests.diamond_dag",
    "single_task": "tests.single_task",
    "validate_demo": "tests.validate_demo",
    "linear_chain": "tests.linear_chain",
}

# --- App ---

app = FastAPI(title="Taskgraph", docs_url=None, redoc_url=None)
app.mount(
    "/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static"
)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.get("/api/specs")
async def list_specs():
    """List available built-in specs."""
    return [{"id": k, "path": v} for k, v in BUILTIN_SPECS.items()]


@app.post("/api/run")
async def start_run(
    spec_id: str = Form(default=""),
    spec_file: UploadFile | None = File(default=None),
    data_files: list[UploadFile] = File(default=[]),
    model: str = Form(default="openai/gpt-5.2"),
    reasoning_effort: str = Form(default="low"),
):
    """Start a workspace run. Returns a run_id for SSE streaming."""
    run_id = str(uuid.uuid4())[:8]
    run_dir = Path(tempfile.mkdtemp(prefix=f"taskgraph_{run_id}_"))

    # Save uploaded data files
    upload_dir = run_dir / "uploads"
    upload_dir.mkdir()
    for f in data_files:
        if not f.filename:
            continue
        dest = upload_dir / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)

    # Determine spec
    if spec_file and spec_file.filename:
        raise HTTPException(400, "spec_file is not supported; use spec_id")
    if spec_id and spec_id in BUILTIN_SPECS:
        spec_module = BUILTIN_SPECS[spec_id]
    else:
        raise HTTPException(400, "Must provide spec_id")

    db_path = run_dir / "output.db"
    export_dir = run_dir / "exports"
    export_dir.mkdir()

    _runs[run_id] = {
        "status": "pending",
        "spec_module": spec_module,
        "db_path": str(db_path),
        "run_dir": str(run_dir),
        "export_dir": str(export_dir),
        "log_lines": [],
        "result": None,
        "model": model,
        "reasoning_effort": reasoning_effort,
    }

    # Start the run in background
    asyncio.create_task(
        _execute_run(run_id, spec_module, db_path, export_dir, model, reasoning_effort)
    )

    return {"run_id": run_id}


async def _execute_run(
    run_id: str,
    spec_module: str,
    db_path: Path,
    export_dir: Path,
    model: str,
    reasoning_effort: str,
):
    """Execute workspace run, capturing log output."""
    state = _runs[run_id]
    state["status"] = "running"

    # Set up log capture
    handler = _ListHandler(state["log_lines"])
    handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    try:
        _append_log(state, f"Loading spec: {spec_module}")
        spec_path = resolve_module_path(spec_module)
        repo_info = get_spec_repo_info(spec_path)
        if repo_info.dirty:
            _append_log(
                state,
                f"WARNING: spec repo is dirty ({repo_info.root}); run is not strictly reproducible.",
            )
        spec = load_spec_from_module(spec_module)

        # Rewrite export paths to use export_dir
        exports = {}
        for name, fn in spec["exports"].items():
            export_path = export_dir / Path(name).name
            exports[str(export_path)] = fn

        workspace = Workspace(
            db_path=db_path,
            inputs=spec["inputs"],
            tasks=spec["tasks"],
            exports=exports,
            input_columns=spec.get("input_columns", {}),
            input_validate_sql=spec.get("input_validate_sql", {}),
            spec_module=spec_module,
            spec_git_commit=repo_info.commit,
            spec_git_root=str(repo_info.root),
            spec_git_dirty=repo_info.dirty,
        )

        re = reasoning_effort if reasoning_effort != "low" else None
        async with OpenRouterClient(reasoning_effort=re) as client:
            result = await workspace.run(
                client=client,
                model=model,
                max_iterations=200,
            )

        state["result"] = {
            "success": result.success,
            "elapsed_s": result.elapsed_s,
            "dag_layers": result.dag_layers,
            "tasks": {
                name: {
                    "success": r.success,
                    "iterations": r.iterations,
                    "tool_calls": r.tool_calls_count,
                    "tokens": r.usage["prompt_tokens"] + r.usage["completion_tokens"],
                }
                for name, r in result.task_results.items()
            },
        }
        state["status"] = "done"
        _append_log(
            state,
            f"\nDone: {'ALL PASSED' if result.success else 'SOME FAILED'} ({result.elapsed_s:.1f}s)",
        )

    except Exception as e:
        state["status"] = "error"
        _append_log(state, f"\nERROR: {e}")
        log.exception("Run %s failed", run_id)

    finally:
        root_logger.removeHandler(handler)


def _append_log(state: dict[str, Any], msg: str) -> None:
    state["log_lines"].append(msg)


class _ListHandler(logging.Handler):
    """Logging handler that appends to a list."""

    def __init__(self, lines: list[str]):
        super().__init__()
        self.lines = lines

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE stream of log lines for a run."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        state: dict[str, Any] = _runs[run_id]
        sent = 0
        while True:
            lines = state["log_lines"]
            while sent < len(lines):
                yield {"event": "log", "data": lines[sent]}
                sent += 1

            if state["status"] in ("done", "error"):
                yield {"event": "done", "data": state["status"]}
                if state["result"]:
                    import json

                    yield {"event": "result", "data": json.dumps(state["result"])}
                break

            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


@app.get("/api/run/{run_id}/download/db")
async def download_db(run_id: str):
    """Download the output .db file."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    db_path = Path(_runs[run_id]["db_path"])
    if not db_path.exists():
        raise HTTPException(404, "Database not ready")
    return FileResponse(
        db_path, filename="output.db", media_type="application/octet-stream"
    )


@app.get("/api/run/{run_id}/exports")
async def list_exports(run_id: str) -> list[dict[str, str]]:
    """List exported files for a run."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    export_dir = Path(_runs[run_id]["export_dir"])
    if not export_dir.exists():
        return []
    return [{"name": f.name} for f in export_dir.iterdir() if f.is_file()]


@app.get("/api/run/{run_id}/download/export/{filename}")
async def download_export(run_id: str, filename: str):
    """Download an exported file."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    export_dir = Path(_runs[run_id]["export_dir"])
    file_path = export_dir / filename
    if not file_path.exists() or not file_path.is_relative_to(export_dir):
        raise HTTPException(404, "File not found")
    return FileResponse(file_path, filename=filename)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
