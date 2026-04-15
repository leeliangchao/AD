"""Tests for lazy imports from the top-level utils package."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_importing_utils_config_does_not_eagerly_import_runtime() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    script = (
        "import sys; "
        "import adrf.utils.config; "
        "raise SystemExit(1 if 'adrf.utils.runtime' in sys.modules else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
