"""Tests for researcher-facing docs and entrypoint scripts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_PATHS = [
    PROJECT_ROOT / "docs" / "getting_started_research.md",
    PROJECT_ROOT / "docs" / "official_baseline_contract.md",
    PROJECT_ROOT / "docs" / "result_reading_guide.md",
]


def _run_script(script_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, f"scripts/{script_name}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_research_docs_exist() -> None:
    """Researcher onboarding docs should exist at stable official paths."""

    for path in DOC_PATHS:
        assert path.exists(), f"Missing researcher doc: {path}"


def test_check_research_env_script_executes() -> None:
    """The environment check script should print the minimal research checklist."""

    result = _run_script("check_research_env.py")

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    for expected in (
        "python:",
        "torch:",
        "cuda_available:",
        "gpu_name:",
        "data/mvtec:",
        "bottle:",
        "capsule:",
        "grid:",
        "official_contract:",
        "official_filled_smoke:",
    ):
        assert expected in stdout


def test_print_official_contract_script_executes() -> None:
    """The contract printer should expose the official comparison contract."""

    result = _run_script("print_official_contract.py")

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    for expected in (
        "official_contract:",
        "datasets:",
        "mvtec_bottle",
        "mvtec_capsule",
        "mvtec_grid",
        "seeds: [0, 1, 2]",
        "runtime_config: configs/runtime/real.yaml",
        "backend: legacy",
        "metrics:",
        "feature_memory + feature_distance",
        "diffusion_basic + noise_residual",
        "diffusion_inversion_basic + path_cost",
        "reference_diffusion_basic + noise_residual",
    ):
        assert expected in stdout
