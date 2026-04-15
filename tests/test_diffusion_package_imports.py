"""Smoke tests for clean diffusion package imports."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_diffusion_submodules_import_cleanly_in_fresh_interpreter() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    script = (
        "from adrf.diffusion.schedulers import make_scheduler; "
        "from adrf.diffusion.adapters import DiffusersNoisePredictorAdapter; "
        "assert callable(make_scheduler); "
        "assert DiffusersNoisePredictorAdapter is not None"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
