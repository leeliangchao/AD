"""Smoke test for the audited paper baseline matrix workflow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_audited_paper_matrix_scripts_execute_end_to_end() -> None:
    """Audited matrix, audit tables, and failure-analysis scripts should run end-to-end."""

    project_root = Path(__file__).resolve().parents[1]
    run_result = subprocess.run(
        [
            sys.executable,
            "scripts/run_audited_paper_matrix.py",
            "--config",
            "configs/ablation/paper_baseline_matrix_v3_audited_smoke.yaml",
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
    diffusion_runs = [record for record in completed_runs if record.get("normality") == "diffusion_basic"]
    assert diffusion_runs[0]["budget"]["backend"] == "legacy"

    export_tables = subprocess.run(
        [sys.executable, "scripts/export_audit_tables.py", str(matrix_dir)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_tables.returncode == 0, export_tables.stderr
    table_outputs = json.loads(export_tables.stdout)
    assert table_outputs["paper_markdown"].endswith("paper_table.md")
    assert table_outputs["category_mean_markdown"].endswith("paper_table_category_mean.md")
    assert table_outputs["by_axis_markdown"].endswith("paper_table_by_axis.md")

    export_failure = subprocess.run(
        [sys.executable, "scripts/export_failure_analysis.py", str(matrix_dir), "2"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_failure.returncode == 0, export_failure.stderr
    failure_outputs = json.loads(export_failure.stdout)
    failure_root = Path(failure_outputs["failure_analysis_dir"])
    assert failure_root.exists()
    assert failure_outputs["groups"]
