"""Smoke test for ablation matrix execution and summary export."""

import json
from pathlib import Path
import subprocess
import sys


def test_ablation_scripts_execute_end_to_end() -> None:
    """Ablation scripts should run a matrix and export its summary."""

    project_root = Path(__file__).resolve().parents[1]
    run_result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation_matrix.py",
            "--config",
            "configs/ablation/diffusion_evidence_matrix.yaml",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr

    payload = json.loads(run_result.stdout)
    matrix_dir = payload["matrix_dir"]

    export_result = subprocess.run(
        [sys.executable, "scripts/export_ablation_summary.py", matrix_dir],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_result.returncode == 0, export_result.stderr
    exported = json.loads(export_result.stdout)
    assert exported["markdown"].endswith("ablation_summary.md")

