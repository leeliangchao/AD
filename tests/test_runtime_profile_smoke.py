"""Smoke tests for runtime profile integration."""

import json
from pathlib import Path

from adrf.runner.experiment_runner import ExperimentRunner


def test_runtime_profile_smoke_records_runtime_info(tmp_path: Path) -> None:
    """A smoke run should record runtime/device information into run_info.json."""

    project_root = Path(__file__).resolve().parents[1]
    config = {
        "name": "runtime_smoke",
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
        "normality": {
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
        "evidence": {"name": "reconstruction_residual", "params": {"aggregator": "mean"}},
        "evaluator": {"name": "basic_ad", "params": {}},
        "protocol": {"name": "one_class", "params": {}},
    }

    runner = ExperimentRunner(config, output_root=tmp_path / "runs")
    runner.run()

    run_info = json.loads((runner.run_dir / "run_info.json").read_text(encoding="utf-8"))
    runtime_info = run_info["runtime"]

    assert runtime_info["profile_name"] == "smoke"
    assert "requested_device" in runtime_info
    assert "actual_device" in runtime_info
    assert "amp_enabled" in runtime_info
    assert "total_time_s" in runtime_info
    assert "train_time_s" in runtime_info
    assert "eval_time_s" in runtime_info

