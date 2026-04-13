"""Smoke test for the process-style diffusion path-cost baseline."""

from pathlib import Path
import subprocess
import sys


def test_run_diffusion_pathcost_baseline_script_executes() -> None:
    """The diffusion path-cost baseline script should run with its default config."""

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_diffusion_pathcost_baseline.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "image_auroc" in result.stdout

