"""Helpers for resolving representation provenance metadata."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def resolve_representation_code_version() -> str:
    """Resolve a reproducible code version string for representation provenance."""

    override = os.getenv("ADRF_CODE_VERSION") or os.getenv("GIT_COMMIT")
    if override:
        return override

    try:
        commit_sha = _git_output("rev-parse", "HEAD")
        dirty = bool(_git_output("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return "working-tree"

    return f"{commit_sha}-dirty" if dirty else commit_sha


def _git_output(*args: str) -> str:
    """Run a git command relative to the repository root and return stripped stdout."""

    completed = subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
