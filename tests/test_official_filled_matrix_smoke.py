"""Smoke test for the stronger official filled baseline matrix."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_run_official_filled_matrix_script_executes_smoke_config() -> None:
    """The official filled smoke matrix should run end-to-end via the wrapper script."""

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "ablation" / "paper_baseline_matrix_official_v1_filled_smoke.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_official_filled_matrix.py",
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["matrix_name"] == "paper_baseline_matrix_official_v1_filled_smoke"
    assert len(payload["experiments"]) == 7
    assert all(record["status"] == "completed" for record in payload["experiments"])
    matrix_dir = Path(payload["matrix_dir"])
    assert (matrix_dir / "paper_table.md").exists()
    assert (matrix_dir / "paper_table_category_mean.md").exists()
    assert (matrix_dir / "paper_table_by_axis.md").exists()
