"""Smoke test for the budgeted paper baseline matrix workflow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_budgeted_paper_matrix_scripts_execute_end_to_end() -> None:
    """Budgeted matrix and grouped export scripts should run end-to-end."""

    project_root = Path(__file__).resolve().parents[1]
    run_result = subprocess.run(
        [
            sys.executable,
            "scripts/run_budgeted_paper_matrix.py",
            "--config",
            "configs/ablation/paper_baseline_matrix_v2_budgeted_smoke.yaml",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr

    payload = json.loads(run_result.stdout)
    matrix_dir = Path(payload["matrix_dir"])
    assert payload["aggregated_results"]
    assert matrix_dir.joinpath("paper_table.md").exists()
    assert matrix_dir.joinpath("paper_table_category_mean.md").exists()
    assert matrix_dir.joinpath("paper_table_by_axis.md").exists()

    completed_runs = [record for record in payload["experiments"] if record.get("status") == "completed"]
    assert completed_runs
    trainable_run = next(
        record for record in completed_runs if record.get("normality") == "diffusion_inversion_basic"
    )
    assert trainable_run["budget"]["max_epochs"] == 3
    assert trainable_run["budget"]["batch_size"] == 8

    export_result = subprocess.run(
        [sys.executable, "scripts/export_grouped_paper_tables.py", str(matrix_dir)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_result.returncode == 0, export_result.stderr
    outputs = json.loads(export_result.stdout)
    assert outputs["paper_markdown"].endswith("paper_table.md")
    assert outputs["category_mean_markdown"].endswith("paper_table_category_mean.md")
    assert outputs["by_axis_markdown"].endswith("paper_table_by_axis.md")
