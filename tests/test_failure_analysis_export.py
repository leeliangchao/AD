"""Tests for failure-analysis export."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from adrf.runner.experiment_runner import ExperimentRunner


def _load_export_failure_analysis() -> object:
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "scripts" / "export_failure_analysis.py"
    spec = importlib.util.spec_from_file_location("export_failure_analysis_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_failure_analysis


def _write_matrix_results(
    matrix_dir: Path,
    run_path: Path,
    *,
    experiment_name: str,
    dataset: str,
    normality: str,
    evidence: str,
) -> None:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    (matrix_dir / "matrix_results.json").write_text(
        json.dumps(
            {
                "matrix_name": "paper_baseline_matrix_v3_audited_smoke",
                "experiments": [
                    {
                        "experiment_name": experiment_name,
                        "group_name": experiment_name,
                        "dataset": dataset,
                        "representation": "pixel" if normality != "feature_memory" else "feature",
                        "normality": normality,
                        "evidence": evidence,
                        "seed": 0,
                        "status": "completed",
                        "run_path": str(run_path),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _run_demo_experiment(tmp_path: Path, *, normality: dict[str, object], evidence: dict[str, object]) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    config = {
        "name": f"{normality['name']}_{evidence['name']}_audit_demo",
        "runtime_config": str(project_root / "configs" / "runtime" / "smoke.yaml"),
        "datamodule": {
            "name": "mvtec_single_class",
            "params": {
                "root": str(project_root / "tests" / "fixtures" / "mvtec"),
                "category": "bottle",
                "reference_index": 0,
                "image_size": [32, 32],
                "batch_size": 2,
                "num_workers": 0,
                "normalize": False,
            },
        },
        "representation": {"name": "pixel", "params": {}},
        "normality": normality,
        "evidence": evidence,
        "evaluator": {"name": "basic_ad", "params": {}},
        "protocol": {"name": "one_class", "params": {}},
    }
    runner = ExperimentRunner(config, output_root=tmp_path / "runs")
    runner.run()
    assert runner.run_dir is not None
    return runner.run_dir


def test_failure_analysis_export_writes_minimum_case_bundle(tmp_path: Path) -> None:
    """Failure analysis should export input, mask, anomaly map, and score even without diffusion extras."""

    run_dir = _run_demo_experiment(
        tmp_path,
        normality={
            "name": "autoencoder",
            "params": {
                "input_channels": 3,
                "hidden_channels": 4,
                "latent_channels": 8,
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
            },
        },
        evidence={"name": "reconstruction_residual", "params": {"aggregator": "mean"}},
    )
    matrix_dir = tmp_path / "matrix_autoencoder"
    _write_matrix_results(
        matrix_dir,
        run_dir,
        experiment_name="mvtec_bottle__pixel__autoencoder__reconstruction_residual",
        dataset="mvtec_bottle",
        normality="autoencoder",
        evidence="reconstruction_residual",
    )

    export_failure_analysis = _load_export_failure_analysis()
    outputs = export_failure_analysis(matrix_dir, max_cases_per_group=2)

    group_dir = Path(outputs["groups"]["autoencoder__reconstruction_residual"])
    case_dirs = sorted(path for path in group_dir.iterdir() if path.is_dir())
    assert case_dirs
    case_dir = case_dirs[0]
    assert (case_dir / "input.png").exists()
    assert (case_dir / "anomaly_map.png").exists()
    assert (case_dir / "score.json").exists()
    assert (case_dir / "mask.png").exists()


def test_failure_analysis_export_writes_available_diffusion_artifacts(tmp_path: Path) -> None:
    """Failure analysis should export diffusion-specific tensors when the run exposes them."""

    run_dir = _run_demo_experiment(
        tmp_path,
        normality={
            "name": "diffusion_basic",
            "params": {
                "backend": "legacy",
                "input_channels": 3,
                "hidden_channels": 8,
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "noise_level": 0.2,
            },
        },
        evidence={"name": "noise_residual", "params": {"aggregator": "mean"}},
    )
    matrix_dir = tmp_path / "matrix_diffusion"
    _write_matrix_results(
        matrix_dir,
        run_dir,
        experiment_name="mvtec_bottle__pixel__diffusion_basic__noise_residual",
        dataset="mvtec_bottle",
        normality="diffusion_basic",
        evidence="noise_residual",
    )

    export_failure_analysis = _load_export_failure_analysis()
    outputs = export_failure_analysis(matrix_dir, max_cases_per_group=2)

    group_dir = Path(outputs["groups"]["diffusion_basic__noise_residual"])
    case_dirs = sorted(path for path in group_dir.iterdir() if path.is_dir())
    assert case_dirs
    case_dir = case_dirs[0]
    assert (case_dir / "predicted_noise.png").exists()
    assert (case_dir / "target_noise.png").exists()
