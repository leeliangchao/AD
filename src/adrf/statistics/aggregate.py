"""Statistical aggregation helpers for multi-seed experiment results."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any


def aggregate_seed_metrics(seed_results: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate per-seed metric dictionaries into summary statistics."""

    metric_values: dict[str, list[float]] = defaultdict(list)
    for result in seed_results:
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_values[key].append(float(value))

    aggregated: dict[str, dict[str, float]] = {}
    for metric_name, values in metric_values.items():
        if not values:
            continue
        aggregated[metric_name] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "best": max(values),
            "median": median(values),
            "count": float(len(values)),
        }
    return aggregated


def aggregate_grouped_seed_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group per-seed records by experiment group and aggregate their metrics."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group_name = str(record.get("group_name") or record.get("experiment_name"))
        grouped[group_name].append(record)

    aggregated_records: list[dict[str, Any]] = []
    for group_name, group_records in grouped.items():
        first = group_records[0]
        completed_records = [record for record in group_records if record.get("status") == "completed"]
        all_completed = len(completed_records) == len(group_records)
        if all_completed:
            group_status = "completed"
        elif completed_records:
            group_status = "partial_failed"
        else:
            group_status = "failed"
        aggregated_metrics = aggregate_seed_metrics(completed_records)
        aggregated_records.append(
            {
                "group_name": group_name,
                "experiment_name": group_name,
                "dataset": first.get("dataset", ""),
                "representation": first.get("representation", ""),
                "normality": first.get("normality", ""),
                "evidence": first.get("evidence", ""),
                "status": group_status,
                "seed_count": len(group_records),
                "aggregated_metrics": aggregated_metrics,
                "seed_runs": [
                    {
                        "seed": record.get("seed"),
                        "status": record.get("status"),
                        "run_path": record.get("run_path", ""),
                    }
                    for record in group_records
                ],
            }
        )

    aggregated_records.sort(key=lambda item: str(item.get("group_name", "")))
    return aggregated_records
