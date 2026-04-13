"""Tests for paper table export."""

import json
from pathlib import Path

from adrf.statistics.table_export import export_paper_tables, write_paper_tables


def test_write_paper_tables_exports_markdown_and_csv(tmp_path: Path) -> None:
    """Aggregated experiment records should export into paper-friendly tables."""

    outputs = write_paper_tables(
        aggregated_records=[
            {
                "experiment_name": "diffusion_basic__noise_residual",
                "dataset": "mvtec_bottle",
                "representation": "pixel",
                "normality": "diffusion_basic",
                "evidence": "noise_residual",
                "seed_count": 3,
                "aggregated_metrics": {
                    "image_auroc": {"mean": 0.912, "std": 0.013},
                    "pixel_auroc": {"mean": 0.873, "std": 0.009},
                    "pixel_aupr": {"mean": 0.801, "std": 0.015},
                },
            }
        ],
        output_dir=tmp_path / "tables",
        title="demo_table",
    )

    assert outputs["markdown"].exists()
    assert outputs["csv"].exists()
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "0.912 ± 0.013" in markdown
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["title"] == "demo_table"


def test_export_paper_tables_reads_aggregated_matrix_file(tmp_path: Path) -> None:
    """Paper table export should load precomputed aggregated matrix results."""

    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    (matrix_dir / "matrix_aggregated.json").write_text(
        json.dumps(
            {
                "matrix_name": "demo_matrix",
                "aggregated_results": [
                    {
                        "experiment_name": "combo",
                        "dataset": "mvtec_bottle",
                        "representation": "pixel",
                        "normality": "diffusion_basic",
                        "evidence": "noise_residual",
                        "seed_count": 2,
                        "aggregated_metrics": {
                            "image_auroc": {"mean": 1.0, "std": 0.0},
                            "pixel_auroc": {"mean": 0.9, "std": 0.1},
                            "pixel_aupr": {"mean": 0.8, "std": 0.2},
                        },
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    outputs = export_paper_tables(matrix_dir)

    assert outputs["markdown"].exists()
    assert outputs["csv"].exists()

