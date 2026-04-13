"""Benchmark summary export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _write_summary_files(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    title: str,
    stem: str,
) -> dict[str, Path]:
    """Write markdown, CSV, and JSON summaries for experiment collections."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    markdown_path = output_path / f"{stem}.md"
    csv_path = output_path / f"{stem}.csv"
    json_path = output_path / f"{stem}.json"

    lines = [
        f"# {title}",
        "",
        "| Experiment | Status | Normality | Evidence | Device | Budget | Total Time (s) | image_auroc | pixel_auroc | pixel_aupr | Run Path | Report Path |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for record in records:
        metrics = record.get("metrics", {})
        run_info = _load_run_info(record.get("run_path", ""))
        runtime = _extract_runtime_info(run_info)
        budget = _extract_budget_info(run_info)
        artifact_info = _extract_artifact_info(run_info)
        lines.append(
            "| {experiment_name} | {status} | {normality} | {evidence} | {device} | {budget} | {total_time_s} | {image_auroc} | {pixel_auroc} | {pixel_aupr} | {run_path} | {report_path} |".format(
                experiment_name=record.get("experiment_name", "unknown"),
                status=record.get("status", "unknown"),
                normality=record.get("normality", ""),
                evidence=record.get("evidence", ""),
                device=runtime.get("actual_device", ""),
                budget=_format_budget_summary(budget),
                total_time_s=runtime.get("total_time_s", ""),
                image_auroc=metrics.get("image_auroc", ""),
                pixel_auroc=metrics.get("pixel_auroc", ""),
                pixel_aupr=metrics.get("pixel_aupr", ""),
                run_path=record.get("run_path", ""),
                report_path=artifact_info.get("report_path", ""),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "experiment_name",
        "status",
        "dataset",
        "representation",
        "normality",
        "evidence",
        "device",
        "budget_max_epochs",
        "budget_batch_size",
        "budget_lr",
        "budget_num_steps",
        "budget_backend",
        "budget_image_size",
        "budget_runtime_config",
        "total_time_s",
        "image_auroc",
        "pixel_auroc",
        "pixel_aupr",
        "run_path",
        "report_path",
        "metrics_path",
        "checkpoint_path",
        "config_path",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics", {})
            run_info = _load_run_info(record.get("run_path", ""))
            runtime = _extract_runtime_info(run_info)
            budget = _extract_budget_info(run_info)
            artifact_info = _extract_artifact_info(run_info)
            writer.writerow(
                {
                    "experiment_name": record.get("experiment_name", ""),
                    "status": record.get("status", ""),
                    "dataset": record.get("dataset", ""),
                    "representation": record.get("representation", ""),
                    "normality": record.get("normality", ""),
                    "evidence": record.get("evidence", ""),
                    "device": runtime.get("actual_device", ""),
                    "budget_max_epochs": budget.get("max_epochs", ""),
                    "budget_batch_size": budget.get("batch_size", ""),
                    "budget_lr": budget.get("lr", ""),
                    "budget_num_steps": budget.get("num_steps", ""),
                    "budget_backend": budget.get("backend", ""),
                    "budget_image_size": json.dumps(budget.get("image_size", "")),
                    "budget_runtime_config": budget.get("runtime_config", ""),
                    "total_time_s": runtime.get("total_time_s", ""),
                    "image_auroc": metrics.get("image_auroc", ""),
                    "pixel_auroc": metrics.get("pixel_auroc", ""),
                    "pixel_aupr": metrics.get("pixel_aupr", ""),
                    "run_path": record.get("run_path", ""),
                    "report_path": artifact_info.get("report_path", ""),
                    "metrics_path": artifact_info.get("metrics_path", ""),
                    "checkpoint_path": artifact_info.get("checkpoint_path", ""),
                    "config_path": record.get("config_path", ""),
                    "error": record.get("error", ""),
                }
            )

    json_path.write_text(json.dumps({"title": title, "records": records}, indent=2, sort_keys=True), encoding="utf-8")
    return {"markdown": markdown_path, "csv": csv_path, "json": json_path}


def write_benchmark_summary(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    suite_name: str,
) -> dict[str, Path]:
    """Write markdown and CSV summaries for one benchmark suite."""

    outputs = _write_summary_files(records, output_dir, suite_name, "benchmark_summary")
    outputs["json"].write_text(
        json.dumps({"suite_name": suite_name, "records": records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs


def write_ablation_summary(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    matrix_name: str,
) -> dict[str, Path]:
    """Write markdown and CSV summaries for one ablation matrix run."""

    outputs = _write_summary_files(records, output_dir, matrix_name, "ablation_summary")
    outputs["json"].write_text(
        json.dumps({"matrix_name": matrix_name, "records": records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs


def export_benchmark_summary(suite_dir: str | Path) -> dict[str, Path]:
    """Generate markdown/CSV summaries from a stored benchmark suite result file."""

    suite_path = Path(suite_dir).resolve()
    suite_results = json.loads((suite_path / "suite_results.json").read_text(encoding="utf-8"))
    return write_benchmark_summary(
        records=list(suite_results.get("experiments", [])),
        output_dir=suite_path,
        suite_name=str(suite_results.get("suite_name", suite_path.name)),
    )


def export_ablation_summary(matrix_dir: str | Path) -> dict[str, Path]:
    """Generate markdown/CSV summaries from a stored ablation matrix result file."""

    matrix_path = Path(matrix_dir).resolve()
    matrix_results = json.loads((matrix_path / "matrix_results.json").read_text(encoding="utf-8"))
    return write_ablation_summary(
        records=list(matrix_results.get("experiments", [])),
        output_dir=matrix_path,
        matrix_name=str(matrix_results.get("matrix_name", matrix_path.name)),
    )


def find_latest_benchmark_dir(base_dir: str | Path = "outputs/benchmarks") -> Path:
    """Return the most recently modified benchmark suite directory."""

    root = Path(base_dir)
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No benchmark directories found under {root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_ablation_dir(base_dir: str | Path = "outputs/ablations") -> Path:
    """Return the most recently modified ablation matrix directory."""

    root = Path(base_dir)
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No ablation directories found under {root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_runtime_info(run_path: str | Path) -> dict[str, Any]:
    """Load runtime metadata from a run directory when available."""

    run_info = _load_run_info(run_path)
    return _extract_runtime_info(run_info)


def _load_run_info(run_path: str | Path) -> dict[str, Any]:
    """Load run_info.json from a run directory when available."""

    if not run_path:
        return {}
    run_info_path = Path(run_path) / "run_info.json"
    if not run_info_path.exists():
        return {}
    try:
        return json.loads(run_info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _extract_runtime_info(run_info: dict[str, Any]) -> dict[str, Any]:
    """Extract runtime metadata from a loaded run-info payload."""

    runtime = run_info.get("runtime", {})
    return runtime if isinstance(runtime, dict) else {}


def _extract_budget_info(run_info: dict[str, Any]) -> dict[str, Any]:
    """Extract budget metadata from a loaded run-info payload."""

    budget = run_info.get("budget", {})
    return budget if isinstance(budget, dict) else {}


def _extract_artifact_info(run_info: dict[str, Any]) -> dict[str, Any]:
    """Extract artifact path metadata from a loaded run-info payload."""

    artifacts = run_info.get("artifacts", {})
    return artifacts if isinstance(artifacts, dict) else {}


def _format_budget_summary(budget: dict[str, Any]) -> str:
    """Render one concise budget summary for markdown tables."""

    if not budget:
        return ""

    parts: list[str] = []
    if "max_epochs" in budget:
        parts.append(f"ep={budget['max_epochs']}")
    if "batch_size" in budget:
        parts.append(f"bs={budget['batch_size']}")
    if "lr" in budget:
        parts.append(f"lr={budget['lr']}")
    if "num_steps" in budget:
        parts.append(f"steps={budget['num_steps']}")
    if "backend" in budget:
        parts.append(f"backend={budget['backend']}")
    image_size = budget.get("image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        parts.append(f"img={image_size[0]}x{image_size[1]}")
    runtime_config = budget.get("runtime_config")
    if isinstance(runtime_config, str) and runtime_config:
        parts.append(f"runtime={Path(runtime_config).stem}")
    return ", ".join(parts)
