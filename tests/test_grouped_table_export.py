"""Tests for grouped paper table export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from adrf.statistics.table_export import export_grouped_paper_tables, write_grouped_paper_tables


def _aggregated_record(
    dataset: str,
    normality: str,
    evidence: str,
    image_auroc: float,
    pixel_auroc: float,
    pixel_aupr: float,
    train_time: float,
    total_time: float,
    *,
    representation: str | None = None,
    backend: str | None = None,
) -> dict[str, object]:
    return {
        "experiment_name": f"{dataset}__{normality}__{evidence}",
        "dataset": dataset,
        "representation": representation or ("feature" if normality == "feature_memory" else "pixel"),
        "normality": normality,
        "evidence": evidence,
        "budget": {"backend": backend} if backend is not None else {},
        "seed_count": 3,
        "aggregated_metrics": {
            "image_auroc": {"mean": image_auroc, "std": 0.01},
            "pixel_auroc": {"mean": pixel_auroc, "std": 0.02},
            "pixel_aupr": {"mean": pixel_aupr, "std": 0.03},
            "train_time": {"mean": train_time, "std": 0.1},
            "total_time": {"mean": total_time, "std": 0.2},
        },
    }


def _demo_aggregated_records() -> list[dict[str, object]]:
    datasets = ("mvtec_bottle", "mvtec_capsule", "mvtec_grid")
    records: list[dict[str, object]] = []
    feature_scores = (0.80, 0.85, 0.90)
    static_scores = (0.84, 0.87, 0.91)
    process_scores = (0.82, 0.86, 0.89)
    conditional_scores = (0.78, 0.80, 0.83)
    conditional_diffusion_scores = (0.81, 0.84, 0.88)

    for dataset, image_score, static_score, process_score, conditional_score, conditional_diffusion_score in zip(
        datasets,
        feature_scores,
        static_scores,
        process_scores,
        conditional_scores,
        conditional_diffusion_scores,
        strict=True,
    ):
        records.append(
            _aggregated_record(
                dataset=dataset,
                normality="feature_memory",
                evidence="feature_distance",
                image_auroc=image_score,
                pixel_auroc=image_score - 0.05,
                pixel_aupr=image_score - 0.10,
                train_time=1.0,
                total_time=1.3,
            )
        )
        records.append(
            _aggregated_record(
                dataset=dataset,
                normality="diffusion_basic",
                evidence="noise_residual",
                image_auroc=static_score,
                pixel_auroc=static_score - 0.04,
                pixel_aupr=static_score - 0.08,
                train_time=1.4,
                total_time=1.8,
            )
        )
        records.append(
            _aggregated_record(
                dataset=dataset,
                normality="diffusion_inversion_basic",
                evidence="path_cost",
                image_auroc=process_score,
                pixel_auroc=process_score - 0.04,
                pixel_aupr=process_score - 0.08,
                train_time=1.8,
                total_time=2.4,
            )
        )
        records.append(
            _aggregated_record(
                dataset=dataset,
                normality="reference_basic",
                evidence="conditional_violation",
                image_auroc=conditional_score,
                pixel_auroc=conditional_score - 0.05,
                pixel_aupr=conditional_score - 0.10,
                train_time=1.2,
                total_time=1.7,
            )
        )
        records.append(
            _aggregated_record(
                dataset=dataset,
                normality="reference_diffusion_basic",
                evidence="noise_residual",
                image_auroc=conditional_diffusion_score,
                pixel_auroc=conditional_diffusion_score - 0.04,
                pixel_aupr=conditional_diffusion_score - 0.08,
                train_time=1.6,
                total_time=2.1,
            )
        )

    return records


def test_write_grouped_paper_tables_exports_category_mean_and_axis_views(tmp_path: Path) -> None:
    """Grouped export should write category-mean and by-axis paper tables."""

    outputs = write_grouped_paper_tables(
        aggregated_records=_demo_aggregated_records(),
        output_dir=tmp_path / "tables",
        title="budgeted_demo",
    )

    assert outputs["paper_markdown"].exists()
    assert outputs["category_mean_markdown"].exists()
    assert outputs["by_axis_markdown"].exists()

    category_mean_markdown = outputs["category_mean_markdown"].read_text(encoding="utf-8")
    assert "0.850 ± 0.041" in category_mean_markdown
    assert "feature_memory + feature_distance" in category_mean_markdown

    by_axis_markdown = outputs["by_axis_markdown"].read_text(encoding="utf-8")
    assert "## classical_vs_diffusion_static" in by_axis_markdown
    assert "## diffusion_static_vs_process" in by_axis_markdown
    assert "## unconditional_vs_conditional" in by_axis_markdown
    assert "diffusion_process" in by_axis_markdown
    assert "conditional" in by_axis_markdown

    with outputs["category_mean_csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["dataset_count"] == "3"
    assert rows[0]["image_auroc_mean"]
    assert rows[0]["image_auroc_display"].count("±") == 1


def test_export_grouped_paper_tables_reads_matrix_aggregated_payload(tmp_path: Path) -> None:
    """Grouped export should load `matrix_aggregated.json` and emit all table variants."""

    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    (matrix_dir / "matrix_aggregated.json").write_text(
        json.dumps(
            {
                "matrix_name": "budgeted_demo",
                "aggregated_results": _demo_aggregated_records(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    outputs = export_grouped_paper_tables(matrix_dir)

    assert outputs["paper_csv"].exists()
    assert outputs["category_mean_csv"].exists()
    assert outputs["by_axis_csv"].exists()


def test_write_grouped_paper_tables_keeps_distinct_representation_backend_rows_separate(tmp_path: Path) -> None:
    outputs = write_grouped_paper_tables(
        aggregated_records=[
            _aggregated_record(
                dataset="mvtec_bottle",
                normality="diffusion_basic",
                evidence="noise_residual",
                representation="pixel",
                backend="legacy",
                image_auroc=0.90,
                pixel_auroc=0.80,
                pixel_aupr=0.70,
                train_time=1.0,
                total_time=1.5,
            ),
            _aggregated_record(
                dataset="mvtec_capsule",
                normality="diffusion_basic",
                evidence="noise_residual",
                representation="feature",
                backend="diffusers",
                image_auroc=0.80,
                pixel_auroc=0.70,
                pixel_aupr=0.60,
                train_time=1.2,
                total_time=1.7,
            ),
        ],
        output_dir=tmp_path / "tables",
        title="distinct_methods",
    )

    with outputs["category_mean_csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert {(row["representation"], row["backend"]) for row in rows} == {
        ("pixel", "legacy"),
        ("feature", "diffusers"),
    }
