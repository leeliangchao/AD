"""Tests for the local run logger."""

import json
from pathlib import Path

from adrf.logging.run_logger import RunLogger


def test_run_logger_creates_run_dir_and_persists_core_files(tmp_path: Path) -> None:
    """RunLogger should create a run directory and persist run assets."""

    logger = RunLogger(base_dir=tmp_path / "runs")
    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("artifact", encoding="utf-8")

    logger.start_run(
        "demo_run",
        config={"normality": {"name": "autoencoder"}},
        run_info={"components": {"normality": "autoencoder"}},
    )
    logger.log_metrics({"evaluation": {"image_auroc": 1.0}})
    logger.log_artifact(artifact_path, artifact_type="prediction")
    logger.finish_run()

    assert logger.run_dir is not None
    run_dir = logger.run_dir
    assert (run_dir / "config_snapshot.yaml").exists()
    assert (run_dir / "run_info.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "predictions" / "artifact.txt").exists()

    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert run_info["status"] == "completed"
    assert metrics["latest"]["evaluation.image_auroc"] == 1.0

