"""Smoke test for multi-seed matrix execution and paper table export."""

import json
from pathlib import Path
import subprocess
import sys


def test_multiseed_scripts_execute_end_to_end() -> None:
    """Multi-seed matrix and paper-table scripts should run end-to-end."""

    project_root = Path(__file__).resolve().parents[1]
    run_result = subprocess.run(
        [
            sys.executable,
            "scripts/run_multiseed_matrix.py",
            "--config",
            "configs/ablation/diffusion_evidence_multiseed.yaml",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr
    payload = json.loads(run_result.stdout)
    matrix_dir = payload["matrix_dir"]
    assert payload["aggregated_results"]

    export_result = subprocess.run(
        [sys.executable, "scripts/export_paper_tables.py", matrix_dir],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_result.returncode == 0, export_result.stderr
    outputs = json.loads(export_result.stdout)
    assert outputs["markdown"].endswith("paper_table.md")

