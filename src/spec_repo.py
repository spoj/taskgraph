"""Spec repository helpers (git-based reproducibility)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpecRepoInfo:
    root: Path
    commit: str
    dirty: bool
    status_porcelain: str


def _run_git(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise ValueError("git is required but was not found on PATH") from e

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(detail or "git command failed")

    return result.stdout.strip()


def get_spec_repo_info(module_file: Path) -> SpecRepoInfo:
    """Resolve git root + commit for a spec module file.

    Raises ValueError if the module is not in a git repo.
    """
    if not module_file.exists():
        raise ValueError(f"Spec module file not found: {module_file}")
    module_dir = module_file.parent
    root_str = _run_git(["rev-parse", "--show-toplevel"], cwd=module_dir)
    root = Path(root_str).resolve()

    status = _run_git(["status", "--porcelain"], cwd=root)
    commit = _run_git(["rev-parse", "HEAD"], cwd=root)
    return SpecRepoInfo(
        root=root, commit=commit, dirty=bool(status), status_porcelain=status
    )
