"""Tests for the generic single-experiment launcher script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


def _load_run_experiment_module():
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_experiment_script_executes_default_config() -> None:
    """The generic experiment launcher should run its default config."""

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"train"' in result.stdout
    assert '"evaluation"' in result.stdout


def test_run_experiment_script_writes_outputs_under_repo_root_when_launched_from_scripts_dir() -> None:
    """Right-click style launches from scripts/ should still write outputs under the repo root."""

    project_root = Path(__file__).resolve().parents[1]
    repo_outputs = project_root / "outputs" / "runs"
    scripts_outputs = project_root / "scripts" / "outputs" / "runs"
    repo_before = len(list(repo_outputs.iterdir())) if repo_outputs.exists() else 0
    scripts_before = len(list(scripts_outputs.iterdir())) if scripts_outputs.exists() else 0

    result = subprocess.run(
        [sys.executable, "run_experiment.py"],
        cwd=project_root / "scripts",
        capture_output=True,
        text=True,
        check=False,
    )

    repo_after = len(list(repo_outputs.iterdir())) if repo_outputs.exists() else 0
    scripts_after = len(list(scripts_outputs.iterdir())) if scripts_outputs.exists() else 0

    assert result.returncode == 0, result.stderr
    assert repo_after == repo_before + 1
    assert scripts_after == scripts_before


def test_run_experiment_resolve_config_path_raises_for_missing_config() -> None:
    """The script should raise a clear error when the configured YAML file is missing."""

    module = _load_run_experiment_module()

    try:
        module.resolve_config_path("does_not_exist.yaml")
    except FileNotFoundError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("Expected resolve_config_path() to raise FileNotFoundError.")

    assert "does_not_exist.yaml" in message
    assert "configs/experiment" in message
