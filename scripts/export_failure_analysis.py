"""Export minimal failure-analysis bundles for completed audited runs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.checkpoint.io import load_model_checkpoint
from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.reporting.summary import find_latest_ablation_dir
from adrf.runner.experiment_runner import ExperimentRunner
from adrf.utils.config import load_yaml_config


def export_failure_analysis(
    matrix_dir: str | Path,
    max_cases_per_group: int | None = None,
) -> dict[str, Any]:
    """Re-run completed experiments and export minimal per-case audit bundles."""

    matrix_path = Path(matrix_dir).resolve()
    matrix_results = json.loads((matrix_path / "matrix_results.json").read_text(encoding="utf-8"))
    configured_max_cases = _resolve_max_cases(matrix_path, matrix_results)
    case_limit = int(max_cases_per_group or configured_max_cases)

    failure_root = matrix_path / "failure_analysis"
    failure_root.mkdir(parents=True, exist_ok=True)

    exported_groups: dict[str, str] = {}
    group_summaries: dict[str, Any] = {}
    for record in matrix_results.get("experiments", []):
        if not isinstance(record, dict):
            continue
        if record.get("status") != "completed":
            continue
        run_path = record.get("run_path")
        if not isinstance(run_path, str) or not run_path:
            continue

        normality = str(record.get("normality", "unknown"))
        evidence = str(record.get("evidence", "unknown"))
        group_key = f"{normality}__{evidence}"
        group_dir = failure_root / group_key
        group_dir.mkdir(parents=True, exist_ok=True)

        exported_cases = _export_group_cases(Path(run_path), group_dir, evidence=evidence, max_cases=case_limit)
        exported_groups[group_key] = str(group_dir)
        group_summaries[group_key] = {
            "run_path": str(Path(run_path).resolve()),
            "exported_cases": exported_cases,
        }

    summary_path = failure_root / "summary.json"
    summary_path.write_text(
        json.dumps({"failure_analysis_dir": str(failure_root), "groups": group_summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "failure_analysis_dir": str(failure_root),
        "summary_path": str(summary_path),
        "groups": exported_groups,
    }


def _export_group_cases(
    run_path: Path,
    group_dir: Path,
    *,
    evidence: str,
    max_cases: int,
) -> list[dict[str, Any]]:
    """Replay one run and export the top-ranked audit cases."""

    config = load_yaml_config(run_path / "config_snapshot.yaml")
    runner = ExperimentRunner(config, output_root=run_path.parent)
    runner.setup()
    runner._initialize_seed()
    if not _restore_or_train_normality(runner, run_path):
        runner.train()

    cases = _collect_cases(runner, evidence=evidence)
    ranked_cases = sorted(cases, key=_case_sort_key, reverse=True)[:max_cases]
    exported_cases: list[dict[str, Any]] = []
    for index, case in enumerate(ranked_cases, start=1):
        case_dir = group_dir / f"{index:02d}_{_safe_name(case['sample_id'])}"
        case_dir.mkdir(parents=True, exist_ok=True)
        exported_cases.append(_write_case_bundle(case_dir, case, evidence=evidence))
    return exported_cases


def _restore_or_train_normality(runner: ExperimentRunner, run_path: Path) -> bool:
    """Restore a saved checkpoint when available, else signal that training is required."""

    checkpoint_path = run_path / "checkpoints" / "normality.pt"
    if checkpoint_path.exists():
        return load_model_checkpoint(runner.normality, checkpoint_path)
    return False


def _collect_cases(runner: ExperimentRunner, *, evidence: str) -> list[dict[str, Any]]:
    """Collect per-sample artifacts and predictions for one configured runner."""

    del evidence
    dataloader = runner.datamodule.test_dataloader()
    cases: list[dict[str, Any]] = []
    for batch in dataloader:
        for sample in batch:
            representation = runner.representation(sample)
            artifacts = runner.normality.infer(sample, representation)
            prediction = runner.evidence.predict(sample, artifacts)
            cases.append(
                {
                    "sample": sample,
                    "sample_id": str(sample.sample_id or "sample"),
                    "image_score": float(prediction["image_score"]),
                    "prediction": prediction,
                    "artifacts": artifacts,
                }
            )
    return cases


def _write_case_bundle(case_dir: Path, case: dict[str, Any], *, evidence: str) -> dict[str, Any]:
    """Write one case bundle to disk and return its summary payload."""

    sample = case["sample"]
    prediction = case["prediction"]
    artifacts = case["artifacts"]

    _save_tensor_image(case_dir / "input.png", sample.image, mode="image")
    if sample.mask is not None:
        _save_tensor_image(case_dir / "mask.png", sample.mask, mode="mask")
    if sample.reference is not None:
        _save_tensor_image(case_dir / "reference.png", sample.reference, mode="image")
    _save_tensor_image(case_dir / "anomaly_map.png", prediction["anomaly_map"], mode="map")
    _save_optional_artifacts(case_dir, artifacts, evidence=evidence, anomaly_map=prediction["anomaly_map"])

    score_payload = {
        "sample_id": case["sample_id"],
        "category": sample.category,
        "label": _sample_label(sample),
        "image_score": case["image_score"],
        "aux_scores": prediction.get("aux_scores", {}),
    }
    score_path = case_dir / "score.json"
    score_path.write_text(json.dumps(score_payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"case_dir": str(case_dir), "score_path": str(score_path), **score_payload}


def _save_optional_artifacts(
    case_dir: Path,
    artifacts: NormalityArtifacts,
    *,
    evidence: str,
    anomaly_map: Any,
) -> None:
    """Export optional artifact tensors when the run exposes them."""

    optional_tensors = {
        "memory_distance.png": artifacts.get_aux("memory_distance"),
        "reconstruction.png": artifacts.get_primary("reconstruction"),
        "reference_projection.png": artifacts.get_primary("reference_projection"),
        "conditional_alignment.png": artifacts.get_aux("conditional_alignment"),
        "predicted_noise.png": artifacts.get_aux("predicted_noise"),
        "target_noise.png": artifacts.get_aux("target_noise"),
    }
    for filename, payload in optional_tensors.items():
        if payload is not None:
            _save_tensor_image(case_dir / filename, payload, mode="map" if filename.endswith("noise.png") or "alignment" in filename or "distance" in filename else "image")

    if evidence == "path_cost":
        _save_tensor_image(case_dir / "path_cost_map.png", anomaly_map, mode="map")
    if evidence == "direction_mismatch":
        _save_tensor_image(case_dir / "direction_mismatch_map.png", anomaly_map, mode="map")
    if evidence == "noise_residual":
        _save_tensor_image(case_dir / "noise_residual_map.png", anomaly_map, mode="map")

    step_costs = artifacts.get_aux("step_costs")
    if isinstance(step_costs, list):
        for index, step_cost in enumerate(step_costs[:3], start=1):
            _save_tensor_image(case_dir / f"step_cost_{index:02d}.png", step_cost, mode="map")

    trajectory = artifacts.get_aux("trajectory")
    if isinstance(trajectory, list) and trajectory:
        _save_tensor_image(case_dir / "trajectory_start.png", trajectory[0], mode="map")
        _save_tensor_image(case_dir / "trajectory_end.png", trajectory[-1], mode="map")


def _save_tensor_image(path: Path, value: Any, *, mode: str) -> None:
    """Serialize a tensor-like payload into a PNG file."""

    array = _to_numpy(value, mode=mode)
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    image.save(path)


def _to_numpy(value: Any, *, mode: str) -> np.ndarray:
    """Convert tensors, arrays, or PIL images into displayable numpy arrays."""

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB" if mode == "image" else "L"))
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().float().numpy()
    else:
        array = np.asarray(value)

    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    elif array.ndim == 3 and array.shape[-1] not in {1, 3}:
        array = array.mean(axis=0)

    if mode == "image":
        if array.ndim == 2:
            normalized = _normalize_to_uint8(array)
            return np.stack([normalized, normalized, normalized], axis=-1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        clipped = np.clip(array, 0.0, 1.0)
        return (clipped * 255.0).round().astype(np.uint8)
    return _normalize_to_uint8(array)


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize a scalar map into 8-bit grayscale."""

    map_array = array.astype(np.float32, copy=False)
    if map_array.ndim == 0:
        map_array = map_array.reshape(1, 1)
    elif map_array.ndim == 1:
        map_array = map_array.reshape(1, -1)
    if map_array.ndim == 3 and map_array.shape[-1] in {1, 3}:
        map_array = map_array.mean(axis=-1)
    if map_array.ndim != 2:
        raise ValueError(f"Expected a 2D map after normalization, got shape {map_array.shape}.")

    min_value = float(np.min(map_array))
    max_value = float(np.max(map_array))
    if max_value <= min_value:
        return np.zeros_like(map_array, dtype=np.uint8)
    normalized = (map_array - min_value) / (max_value - min_value)
    return (normalized * 255.0).round().astype(np.uint8)


def _case_sort_key(case: dict[str, Any]) -> tuple[int, float]:
    """Rank anomalous samples ahead of normal samples, then by image score."""

    sample = case["sample"]
    return (_sample_label(sample), float(case["image_score"]))


def _sample_label(sample: Sample) -> int:
    """Resolve a binary anomaly label from the sample metadata."""

    if sample.label is not None:
        return int(sample.label)
    if sample.mask is not None:
        mask = _to_numpy(sample.mask, mode="mask")
        return int(mask.max() > 0)
    return 0


def _resolve_max_cases(matrix_path: Path, matrix_results: dict[str, Any]) -> int:
    """Resolve the audit case limit from the matrix snapshot when present."""

    del matrix_results
    snapshot_path = matrix_path / "matrix_config_snapshot.yaml"
    if not snapshot_path.exists():
        return 8
    snapshot = load_yaml_config(snapshot_path)
    audit = snapshot.get("audit", {})
    if not isinstance(audit, dict):
        return 8
    value = audit.get("max_cases_per_group", 8)
    return int(value) if isinstance(value, int) else 8


def _safe_name(value: str) -> str:
    """Convert a value into a filesystem-safe token."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "sample"


def main() -> int:
    """Export failure-analysis bundles for the latest or given matrix directory."""

    matrix_dir = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else find_latest_ablation_dir(PROJECT_ROOT / "outputs" / "ablations")
    )
    max_cases = int(sys.argv[2]) if len(sys.argv) > 2 else None
    outputs = export_failure_analysis(matrix_dir, max_cases_per_group=max_cases)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
