"""Paper-friendly table export helpers for aggregated experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def write_paper_tables(
    aggregated_records: list[dict[str, Any]],
    output_dir: str | Path,
    title: str,
) -> dict[str, Path]:
    """Write markdown and CSV paper tables for aggregated experiment results."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    markdown_path = output_path / "paper_table.md"
    csv_path = output_path / "paper_table.csv"
    json_path = output_path / "paper_table.json"

    lines = [
        f"# {title}",
        "",
        "| Experiment | Status | Dataset | Normality | Evidence | Image AUROC | Pixel AUROC | Pixel AUPR | Train Time | Eval Time | Total Time |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in aggregated_records:
        metrics = record.get("aggregated_metrics", {})
        lines.append(
            "| {experiment} | {status} | {dataset} | {normality} | {evidence} | {image_auroc} | {pixel_auroc} | {pixel_aupr} | {train_time} | {eval_time} | {total_time} |".format(
                experiment=record.get("experiment_name", ""),
                status=record.get("status", ""),
                dataset=record.get("dataset", ""),
                normality=record.get("normality", ""),
                evidence=record.get("evidence", ""),
                image_auroc=_format_metric(metrics.get("image_auroc")),
                pixel_auroc=_format_metric(metrics.get("pixel_auroc")),
                pixel_aupr=_format_metric(metrics.get("pixel_aupr")),
                train_time=_format_metric(metrics.get("train_time")),
                eval_time=_format_metric(metrics.get("eval_time")),
                total_time=_format_metric(metrics.get("total_time")),
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
        "backend",
        "image_auroc_mean",
        "image_auroc_std",
        "image_auroc_display",
        "pixel_auroc_mean",
        "pixel_auroc_std",
        "pixel_auroc_display",
        "pixel_aupr_mean",
        "pixel_aupr_std",
        "pixel_aupr_display",
        "train_time_mean",
        "train_time_std",
        "train_time_display",
        "eval_time_mean",
        "eval_time_std",
        "eval_time_display",
        "total_time_mean",
        "total_time_std",
        "total_time_display",
        "seed_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in aggregated_records:
            metrics = record.get("aggregated_metrics", {})
            writer.writerow(
                {
                    "experiment_name": record.get("experiment_name", ""),
                    "status": record.get("status", ""),
                    "dataset": record.get("dataset", ""),
                    "representation": record.get("representation", ""),
                    "normality": record.get("normality", ""),
                    "evidence": record.get("evidence", ""),
                    "backend": _budget_value(record.get("budget"), "backend"),
                    "image_auroc_mean": _metric_value(metrics.get("image_auroc"), "mean"),
                    "image_auroc_std": _metric_value(metrics.get("image_auroc"), "std"),
                    "image_auroc_display": _format_metric(metrics.get("image_auroc")),
                    "pixel_auroc_mean": _metric_value(metrics.get("pixel_auroc"), "mean"),
                    "pixel_auroc_std": _metric_value(metrics.get("pixel_auroc"), "std"),
                    "pixel_auroc_display": _format_metric(metrics.get("pixel_auroc")),
                    "pixel_aupr_mean": _metric_value(metrics.get("pixel_aupr"), "mean"),
                    "pixel_aupr_std": _metric_value(metrics.get("pixel_aupr"), "std"),
                    "pixel_aupr_display": _format_metric(metrics.get("pixel_aupr")),
                    "train_time_mean": _metric_value(metrics.get("train_time"), "mean"),
                    "train_time_std": _metric_value(metrics.get("train_time"), "std"),
                    "train_time_display": _format_metric(metrics.get("train_time")),
                    "eval_time_mean": _metric_value(metrics.get("eval_time"), "mean"),
                    "eval_time_std": _metric_value(metrics.get("eval_time"), "std"),
                    "eval_time_display": _format_metric(metrics.get("eval_time")),
                    "total_time_mean": _metric_value(metrics.get("total_time"), "mean"),
                    "total_time_std": _metric_value(metrics.get("total_time"), "std"),
                    "total_time_display": _format_metric(metrics.get("total_time")),
                    "seed_count": record.get("seed_count", 0),
                }
            )

    json_path.write_text(
        json.dumps({"title": title, "records": aggregated_records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "csv": csv_path, "json": json_path}


def export_paper_tables(matrix_dir: str | Path) -> dict[str, Path]:
    """Export paper-ready tables from a stored aggregated matrix result file."""

    matrix_path = Path(matrix_dir).resolve()
    aggregated = json.loads((matrix_path / "matrix_aggregated.json").read_text(encoding="utf-8"))
    return write_paper_tables(
        aggregated_records=list(aggregated.get("aggregated_results", [])),
        output_dir=matrix_path,
        title=str(aggregated.get("matrix_name", matrix_path.name)),
    )


def write_grouped_paper_tables(
    aggregated_records: list[dict[str, Any]],
    output_dir: str | Path,
    title: str,
) -> dict[str, Path]:
    """Write paper, category-mean, and axis-grouped tables for one matrix run."""

    paper_outputs = write_paper_tables(aggregated_records, output_dir=output_dir, title=title)
    category_mean_records = _build_category_mean_records(aggregated_records)
    category_outputs = _write_metric_table(
        records=category_mean_records,
        output_dir=output_dir,
        stem="paper_table_category_mean",
        title=f"{title} Category Mean",
        markdown_columns=(
            ("method", "Method"),
            ("status", "Status"),
            ("normality", "Normality"),
            ("evidence", "Evidence"),
            ("dataset_count", "Datasets"),
            ("image_auroc_display", "Image AUROC"),
            ("pixel_auroc_display", "Pixel AUROC"),
            ("pixel_aupr_display", "Pixel AUPR"),
            ("train_time_display", "Train Time"),
            ("total_time_display", "Total Time"),
        ),
        csv_fieldnames=(
            "method",
            "status",
            "normality",
            "evidence",
            "representation",
            "backend",
            "dataset_count",
            "image_auroc_mean",
            "image_auroc_std",
            "image_auroc_display",
            "pixel_auroc_mean",
            "pixel_auroc_std",
            "pixel_auroc_display",
            "pixel_aupr_mean",
            "pixel_aupr_std",
            "pixel_aupr_display",
            "train_time_mean",
            "train_time_std",
            "train_time_display",
            "total_time_mean",
            "total_time_std",
            "total_time_display",
        ),
    )
    axis_outputs = _write_axis_grouped_table(
        category_mean_records=category_mean_records,
        output_dir=output_dir,
        title=f"{title} By Axis",
    )
    return {
        "paper_markdown": paper_outputs["markdown"],
        "paper_csv": paper_outputs["csv"],
        "paper_json": paper_outputs["json"],
        "category_mean_markdown": category_outputs["markdown"],
        "category_mean_csv": category_outputs["csv"],
        "category_mean_json": category_outputs["json"],
        "by_axis_markdown": axis_outputs["markdown"],
        "by_axis_csv": axis_outputs["csv"],
        "by_axis_json": axis_outputs["json"],
    }


def export_grouped_paper_tables(matrix_dir: str | Path) -> dict[str, Path]:
    """Export paper, category-mean, and axis-grouped tables from stored results."""

    matrix_path = Path(matrix_dir).resolve()
    aggregated = json.loads((matrix_path / "matrix_aggregated.json").read_text(encoding="utf-8"))
    return write_grouped_paper_tables(
        aggregated_records=list(aggregated.get("aggregated_results", [])),
        output_dir=matrix_path,
        title=str(aggregated.get("matrix_name", matrix_path.name)),
    )


def _format_metric(metric: Any) -> str:
    """Format one aggregated metric as `mean ± std`."""

    if not isinstance(metric, dict):
        return ""
    mean_value = metric.get("mean")
    std_value = metric.get("std")
    if not isinstance(mean_value, (int, float)) or not isinstance(std_value, (int, float)):
        return ""
    return f"{mean_value:.3f} ± {std_value:.3f}"


def _metric_value(metric: Any, key: str) -> float | str:
    """Extract one numeric metric summary field for CSV export."""

    if not isinstance(metric, dict):
        return ""
    value = metric.get(key)
    return value if isinstance(value, (int, float)) else ""


def _build_category_mean_records(aggregated_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-dataset aggregated records into one category-mean row per method."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in aggregated_records:
        key = (
            str(record.get("normality", "")),
            str(record.get("evidence", "")),
            str(record.get("representation", "")),
            str(_budget_value(record.get("budget"), "backend")),
        )
        grouped.setdefault(key, []).append(record)

    category_mean_records: list[dict[str, Any]] = []
    for (_normality, _evidence, _representation, _backend), records in sorted(grouped.items()):
        first = records[0]
        metrics = _aggregate_metrics_across_records(records)
        category_mean_records.append(
            {
                "method": _method_label(first),
                "status": _coalesce_status(records),
                "normality": first.get("normality", ""),
                "evidence": first.get("evidence", ""),
                "representation": first.get("representation", ""),
                "backend": _budget_value(first.get("budget"), "backend"),
                "dataset_count": len(records),
                "aggregated_metrics": metrics,
                "image_auroc_mean": _metric_value(metrics.get("image_auroc"), "mean"),
                "image_auroc_std": _metric_value(metrics.get("image_auroc"), "std"),
                "image_auroc_display": _format_metric(metrics.get("image_auroc")),
                "pixel_auroc_mean": _metric_value(metrics.get("pixel_auroc"), "mean"),
                "pixel_auroc_std": _metric_value(metrics.get("pixel_auroc"), "std"),
                "pixel_auroc_display": _format_metric(metrics.get("pixel_auroc")),
                "pixel_aupr_mean": _metric_value(metrics.get("pixel_aupr"), "mean"),
                "pixel_aupr_std": _metric_value(metrics.get("pixel_aupr"), "std"),
                "pixel_aupr_display": _format_metric(metrics.get("pixel_aupr")),
                "train_time_mean": _metric_value(metrics.get("train_time"), "mean"),
                "train_time_std": _metric_value(metrics.get("train_time"), "std"),
                "train_time_display": _format_metric(metrics.get("train_time")),
                "total_time_mean": _metric_value(metrics.get("total_time"), "mean"),
                "total_time_std": _metric_value(metrics.get("total_time"), "std"),
                "total_time_display": _format_metric(metrics.get("total_time")),
            }
        )

    return category_mean_records


def _write_axis_grouped_table(
    category_mean_records: list[dict[str, Any]],
    output_dir: str | Path,
    title: str,
) -> dict[str, Path]:
    """Write markdown, CSV, and JSON tables grouped by research axis."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    markdown_path = output_path / "paper_table_by_axis.md"
    csv_path = output_path / "paper_table_by_axis.csv"
    json_path = output_path / "paper_table_by_axis.json"

    sections = _axis_sections(category_mean_records)
    lines = [f"# {title}", ""]
    csv_rows: list[dict[str, Any]] = []
    for section_name, rows in sections.items():
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append(
            "| Axis Bucket | Method | Status | Normality | Evidence | Image AUROC | Pixel AUROC | Pixel AUPR | Train Time | Total Time |"
        )
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                "| {axis_bucket} | {method} | {status} | {normality} | {evidence} | {image_auroc} | {pixel_auroc} | {pixel_aupr} | {train_time} | {total_time} |".format(
                    axis_bucket=row.get("axis_bucket", ""),
                    method=row.get("method", ""),
                    status=row.get("status", ""),
                    normality=row.get("normality", ""),
                    evidence=row.get("evidence", ""),
                    image_auroc=row.get("image_auroc_display", ""),
                    pixel_auroc=row.get("pixel_auroc_display", ""),
                    pixel_aupr=row.get("pixel_aupr_display", ""),
                    train_time=row.get("train_time_display", ""),
                    total_time=row.get("total_time_display", ""),
                )
            )
            csv_rows.append(row)
        lines.append("")
    markdown_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    fieldnames = [
        "section",
        "axis_bucket",
        "method",
        "status",
        "normality",
        "evidence",
        "representation",
        "backend",
        "dataset_count",
        "image_auroc_mean",
        "image_auroc_std",
        "image_auroc_display",
        "pixel_auroc_mean",
        "pixel_auroc_std",
        "pixel_auroc_display",
        "pixel_aupr_mean",
        "pixel_aupr_std",
        "pixel_aupr_display",
        "train_time_mean",
        "train_time_std",
        "train_time_display",
        "total_time_mean",
        "total_time_std",
        "total_time_display",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    json_path.write_text(json.dumps({"title": title, "sections": sections}, indent=2, sort_keys=True), encoding="utf-8")
    return {"markdown": markdown_path, "csv": csv_path, "json": json_path}


def _write_metric_table(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    stem: str,
    title: str,
    markdown_columns: tuple[tuple[str, str], ...],
    csv_fieldnames: tuple[str, ...],
) -> dict[str, Path]:
    """Write a generic markdown/CSV/JSON metric table from flat row mappings."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    markdown_path = output_path / f"{stem}.md"
    csv_path = output_path / f"{stem}.csv"
    json_path = output_path / f"{stem}.json"

    lines = [f"# {title}", ""]
    header = " | ".join(label for _key, label in markdown_columns)
    alignment = " | ".join("---:" if "Time" in label or "AU" in label or label == "Datasets" else "---" for _key, label in markdown_columns)
    lines.append(f"| {header} |")
    lines.append(f"| {alignment} |")
    for record in records:
        rendered = " | ".join(str(record.get(key, "")) for key, _label in markdown_columns)
        lines.append(f"| {rendered} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_fieldnames))
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in csv_fieldnames})

    json_path.write_text(json.dumps({"title": title, "records": records}, indent=2, sort_keys=True), encoding="utf-8")
    return {"markdown": markdown_path, "csv": csv_path, "json": json_path}


def _aggregate_metrics_across_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate dataset-level means into category-level means and deviations."""

    metric_values: dict[str, list[float]] = {}
    for record in records:
        aggregated_metrics = record.get("aggregated_metrics", {})
        if not isinstance(aggregated_metrics, dict):
            continue
        for metric_name, metric_payload in aggregated_metrics.items():
            mean_value = _metric_value(metric_payload, "mean")
            if isinstance(mean_value, (int, float)):
                metric_values.setdefault(metric_name, []).append(float(mean_value))

    aggregated: dict[str, dict[str, float]] = {}
    for metric_name, values in metric_values.items():
        aggregated[metric_name] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "count": float(len(values)),
        }
    return aggregated


def _axis_sections(category_mean_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group category-mean rows into the required research-axis sections."""

    sections = {
        "classical_vs_diffusion_static": {"classical", "diffusion_static"},
        "diffusion_static_vs_process": {"diffusion_static", "diffusion_process"},
        "unconditional_vs_conditional": {"unconditional", "conditional"},
    }
    grouped_sections: dict[str, list[dict[str, Any]]] = {name: [] for name in sections}
    for record in category_mean_records:
        axes = _classify_axes(record)
        for section_name, allowed_buckets in sections.items():
            axis_bucket = axes["family"] if section_name != "unconditional_vs_conditional" else axes["conditioning"]
            if axis_bucket in allowed_buckets:
                grouped_sections[section_name].append(
                    {
                        "section": section_name,
                        "axis_bucket": axis_bucket,
                        **record,
                    }
                )

    for section_name, rows in grouped_sections.items():
        grouped_sections[section_name] = sorted(rows, key=lambda row: (str(row.get("axis_bucket", "")), str(row.get("method", ""))))
    return grouped_sections


def _classify_axes(record: dict[str, Any]) -> dict[str, str]:
    """Map one method row onto the requested research-axis buckets."""

    normality = str(record.get("normality", ""))
    family = "other"
    conditioning = "conditional" if normality.startswith("reference_") or normality == "reference_basic" else "unconditional"
    if normality in {"feature_memory", "autoencoder", "reference_basic"}:
        family = "classical"
    elif normality in {"diffusion_basic", "reference_diffusion_basic"}:
        family = "diffusion_static"
    elif normality == "diffusion_inversion_basic":
        family = "diffusion_process"
    return {"family": family, "conditioning": conditioning}


def _method_label(record: dict[str, Any]) -> str:
    """Render one concise method label."""

    return "{normality} + {evidence}".format(
        normality=record.get("normality", ""),
        evidence=record.get("evidence", ""),
    )


def _coalesce_status(records: list[dict[str, Any]]) -> str:
    """Summarize grouped record status for paper-facing exports."""

    statuses = {str(record.get("status", "")) for record in records if record.get("status")}
    if not statuses:
        return ""
    if statuses == {"completed"}:
        return "completed"
    if statuses == {"failed"}:
        return "failed"
    if "partial_failed" in statuses or ("completed" in statuses and "failed" in statuses):
        return "partial_failed"
    return ",".join(sorted(statuses))


def _budget_value(budget: Any, key: str) -> Any:
    """Read one value from an optional budget mapping."""

    if not isinstance(budget, dict):
        return ""
    return budget.get(key, "")
