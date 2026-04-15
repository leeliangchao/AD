"""Tests for multi-seed metric aggregation."""

import pytest

from adrf.statistics.aggregate import aggregate_grouped_seed_results, aggregate_seed_metrics


def test_aggregate_seed_metrics_computes_mean_std_best_and_count() -> None:
    """Per-seed metrics should aggregate into the expected summary statistics."""

    aggregated = aggregate_seed_metrics(
        [
            {"seed": 0, "metrics": {"image_auroc": 0.91, "pixel_auroc": 0.87}},
            {"seed": 1, "metrics": {"image_auroc": 0.93, "pixel_auroc": 0.88}},
            {"seed": 2, "metrics": {"image_auroc": 0.90, "pixel_auroc": 0.86}},
        ]
    )

    assert aggregated["image_auroc"]["mean"] == pytest.approx((0.91 + 0.93 + 0.90) / 3)
    assert aggregated["image_auroc"]["best"] == pytest.approx(0.93)
    assert aggregated["image_auroc"]["count"] == 3.0
    assert aggregated["pixel_auroc"]["std"] >= 0.0


def test_aggregate_grouped_seed_results_groups_by_combo_name() -> None:
    """Per-seed records should be grouped and aggregated by group_name."""

    aggregated = aggregate_grouped_seed_results(
        [
            {
                "group_name": "combo_a",
                "experiment_name": "combo_a__seed0",
                "dataset": "mvtec_bottle",
                "representation": "pixel",
                "normality": "diffusion_basic",
                "evidence": "noise_residual",
                "status": "completed",
                "seed": 0,
                "metrics": {"image_auroc": 0.9},
            },
            {
                "group_name": "combo_a",
                "experiment_name": "combo_a__seed1",
                "dataset": "mvtec_bottle",
                "representation": "pixel",
                "normality": "diffusion_basic",
                "evidence": "noise_residual",
                "status": "completed",
                "seed": 1,
                "metrics": {"image_auroc": 0.8},
            },
        ]
    )

    assert len(aggregated) == 1
    assert aggregated[0]["group_name"] == "combo_a"
    assert aggregated[0]["aggregated_metrics"]["image_auroc"]["mean"] == pytest.approx(0.85)
    assert aggregated[0]["seed_count"] == 2


def test_aggregate_grouped_seed_results_marks_partial_failures_explicitly() -> None:
    aggregated = aggregate_grouped_seed_results(
        [
            {
                "group_name": "combo_a",
                "experiment_name": "combo_a__seed0",
                "dataset": "mvtec_bottle",
                "representation": "pixel",
                "normality": "diffusion_basic",
                "evidence": "noise_residual",
                "status": "completed",
                "seed": 0,
                "metrics": {"image_auroc": 0.9},
            },
            {
                "group_name": "combo_a",
                "experiment_name": "combo_a__seed1",
                "dataset": "mvtec_bottle",
                "representation": "pixel",
                "normality": "diffusion_basic",
                "evidence": "noise_residual",
                "status": "failed",
                "seed": 1,
            },
        ]
    )

    assert aggregated[0]["status"] == "partial_failed"
