"""Sequential runner for ablation experiment matrices."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from adrf.ablation.matrix import AblationMatrix
from adrf.runner.experiment_runner import ExperimentRunner
from adrf.reporting.summary import write_ablation_summary
from adrf.statistics.aggregate import aggregate_grouped_seed_results
from adrf.statistics.table_export import write_grouped_paper_tables, write_paper_tables


class AblationRunner:
    """Run a matrix of valid experiment combinations and export a summary."""

    def __init__(
        self,
        matrix: str | Path | AblationMatrix,
        output_root: str | Path = "outputs",
    ) -> None:
        self.matrix = AblationMatrix.from_yaml(matrix) if isinstance(matrix, (str, Path)) else matrix
        self.output_root = Path(output_root)
        self.matrix_dir: Path | None = None

    def run(self) -> dict[str, Any]:
        """Run the expanded matrix sequentially and write summary assets."""

        valid_specs, invalid_specs = self.matrix.expand_with_invalid()
        matrix_dir = self._create_matrix_dir()
        self.matrix_dir = matrix_dir
        (matrix_dir / "matrix_config_snapshot.yaml").write_text(
            yaml.safe_dump(self.matrix.config, sort_keys=False),
            encoding="utf-8",
        )

        records: list[dict[str, Any]] = []
        started_at = datetime.now(timezone.utc).isoformat()
        continue_on_error = bool(self.matrix.config.get("continue_on_error", True))
        for spec in valid_specs:
            runner = ExperimentRunner(
                spec["config"],
                output_root=self.output_root / "runs",
                run_name=spec["name"],
            )
            record: dict[str, Any] = {
                "experiment_name": spec["name"],
                "group_name": spec.get("group_name", spec["name"]),
                "dataset": spec["dataset"],
                "representation": spec["representation"],
                "normality": spec["normality"],
                "evidence": spec["evidence"],
                "seed": spec.get("seed"),
            }
            try:
                results = runner.run()
                record["status"] = "completed"
                record["metrics"] = {
                    **dict(results.get("evaluation", {})),
                    **self._runtime_metric_payload(runner.runtime_stats),
                }
                record["budget"] = dict(runner.budget_info)
                record["run_path"] = str(runner.run_dir) if runner.run_dir is not None else ""
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                record["metrics"] = {}
                record["budget"] = dict(getattr(runner, "budget_info", {}))
                record["run_path"] = str(runner.run_dir) if runner.run_dir is not None else ""
                if not continue_on_error:
                    records.append(record)
                    break
            records.append(record)

        aggregated_results = self._attach_group_metadata(
            aggregate_grouped_seed_results(records),
            records,
        )
        audit_config = self.matrix.config.get("audit", {})

        matrix_results = {
            "matrix_name": self.matrix.name,
            "matrix_dir": str(matrix_dir),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "audit": dict(audit_config) if isinstance(audit_config, dict) else {},
            "experiments": records,
            "aggregated_results": aggregated_results,
            "invalid_combinations": invalid_specs,
        }
        (matrix_dir / "matrix_results.json").write_text(
            json.dumps(matrix_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (matrix_dir / "matrix_aggregated.json").write_text(
            json.dumps(
                {
                    "matrix_name": self.matrix.name,
                    "audit": dict(audit_config) if isinstance(audit_config, dict) else {},
                    "aggregated_results": aggregated_results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        write_ablation_summary(records, matrix_dir, self.matrix.name)
        self._export_report_tables(matrix_dir, aggregated_results)
        return matrix_results

    def _create_matrix_dir(self) -> Path:
        """Create the output directory for one ablation matrix execution."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        matrix_dir = self.output_root / "ablations" / f"{timestamp}_{self.matrix.name}"
        suffix = 1
        while matrix_dir.exists():
            matrix_dir = self.output_root / "ablations" / f"{timestamp}_{self.matrix.name}_{suffix}"
            suffix += 1
        matrix_dir.mkdir(parents=True, exist_ok=False)
        return matrix_dir

    def _runtime_metric_payload(self, runtime_stats: dict[str, Any]) -> dict[str, float]:
        """Map runtime timing stats onto aggregate-friendly metric names."""

        metrics: dict[str, float] = {}
        mapping = {
            "train_time_s": "train_time",
            "eval_time_s": "eval_time",
            "total_time_s": "total_time",
        }
        for source_key, target_key in mapping.items():
            value = runtime_stats.get(source_key)
            if isinstance(value, (int, float)):
                metrics[target_key] = float(value)
        return metrics

    def _export_report_tables(self, matrix_dir: Path, aggregated_results: list[dict[str, Any]]) -> None:
        """Export paper-ready tables requested by the matrix report config."""

        report_config = self.matrix.config.get("report", {})
        if not isinstance(report_config, dict):
            return

        export_grouped = bool(report_config.get("grouped_tables")) or bool(report_config.get("category_mean_table"))
        if export_grouped:
            write_grouped_paper_tables(aggregated_results, output_dir=matrix_dir, title=self.matrix.name)
            return

        if bool(report_config.get("paper_table")):
            write_paper_tables(aggregated_results, output_dir=matrix_dir, title=self.matrix.name)

    def _attach_group_metadata(
        self,
        aggregated_results: list[dict[str, Any]],
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Carry representative budget metadata from seed records into grouped outputs."""

        grouped_records: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            group_name = str(record.get("group_name") or record.get("experiment_name", ""))
            grouped_records.setdefault(group_name, []).append(record)

        enriched: list[dict[str, Any]] = []
        for aggregated_record in aggregated_results:
            group_name = str(aggregated_record.get("group_name") or aggregated_record.get("experiment_name", ""))
            matching_records = grouped_records.get(group_name, [])
            representative = next((record for record in matching_records if record.get("status") == "completed"), None)
            if representative is None and matching_records:
                representative = matching_records[0]
            merged = dict(aggregated_record)
            if representative is not None:
                merged["budget"] = dict(representative.get("budget", {}))
                merged["run_path"] = representative.get("run_path", "")
            enriched.append(merged)
        return enriched
