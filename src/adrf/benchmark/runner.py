"""Sequential benchmark suite runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from adrf.benchmark.suite import BenchmarkSuite
from adrf.runner.launching import run_experiment_with_runtime_launch
from adrf.reporting.summary import write_benchmark_summary
from adrf.utils.config import load_yaml_config


class BenchmarkRunner:
    """Run a benchmark suite sequentially and export a suite summary."""

    def __init__(
        self,
        suite: str | Path | BenchmarkSuite,
        output_root: str | Path = "outputs",
    ) -> None:
        self.suite = BenchmarkSuite.from_yaml(suite) if isinstance(suite, (str, Path)) else suite
        self.output_root = Path(output_root)
        self.suite_dir: Path | None = None

    def run(self) -> dict[str, Any]:
        """Run every experiment in the suite and write summary assets."""

        suite_dir = self._create_suite_dir()
        self.suite_dir = suite_dir

        records: list[dict[str, Any]] = []
        resolved_experiments: list[dict[str, Any]] = []
        started_at = datetime.now(timezone.utc).isoformat()
        for experiment_path in self.suite.experiments:
            try:
                config = load_yaml_config(experiment_path)
                resolved_experiments.append({"path": str(experiment_path), "config": config})
                record = self._build_record(experiment_path, config)
                results, run_dir, _run_info = run_experiment_with_runtime_launch(
                    experiment_path,
                    output_root=self.output_root / "runs",
                )
                record["status"] = "completed"
                record["metrics"] = dict(results.get("evaluation", {}))
                record["run_path"] = str(run_dir) if run_dir is not None else ""
            except Exception as exc:
                record = {
                    "experiment_name": experiment_path.stem,
                    "config_path": str(experiment_path),
                    "dataset": "",
                    "representation": "",
                    "normality": "",
                    "evidence": "",
                }
                resolved_experiments.append({"path": str(experiment_path), "error": str(exc)})
                record["status"] = "failed"
                record["error"] = str(exc)
                record["metrics"] = {}
                record["run_path"] = ""
                if not self.suite.continue_on_error:
                    records.append(record)
                    break
            records.append(record)

        (suite_dir / "suite_config_snapshot.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": self.suite.name,
                    "continue_on_error": self.suite.continue_on_error,
                    "experiments": [str(path) for path in self.suite.experiments],
                    "resolved_experiments": resolved_experiments,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        suite_results = {
            "suite_name": self.suite.name,
            "suite_dir": str(suite_dir),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "experiments": records,
        }
        (suite_dir / "suite_results.json").write_text(
            json.dumps(suite_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_benchmark_summary(records, suite_dir, self.suite.name)
        return suite_results

    def _create_suite_dir(self) -> Path:
        """Create the output directory for one benchmark suite execution."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suite_dir = self.output_root / "benchmarks" / f"{timestamp}_{self.suite.name}"
        suffix = 1
        while suite_dir.exists():
            suite_dir = self.output_root / "benchmarks" / f"{timestamp}_{self.suite.name}_{suffix}"
            suffix += 1
        suite_dir.mkdir(parents=True, exist_ok=False)
        return suite_dir

    def _build_record(self, experiment_path: Path, config: dict[str, Any]) -> dict[str, Any]:
        """Build the summary record for one experiment config."""

        return {
            "experiment_name": experiment_path.stem,
            "config_path": str(experiment_path),
            "dataset": self._resolve_dataset_name(config),
            "representation": self._resolve_component_name(config, "representation"),
            "normality": self._resolve_component_name(config, "normality"),
            "evidence": self._resolve_component_name(config, "evidence"),
        }

    @staticmethod
    def _resolve_component_name(config: dict[str, Any], key: str) -> str:
        """Resolve a configured component name when present."""

        spec = config.get(key, {})
        if isinstance(spec, dict):
            name = spec.get("name")
            if isinstance(name, str):
                return name
        return ""

    def _resolve_dataset_name(self, config: dict[str, Any]) -> str:
        """Resolve a stable dataset label for benchmark summaries."""

        datamodule = config.get("datamodule", {})
        if not isinstance(datamodule, dict):
            return ""

        params = datamodule.get("params", {})
        if not isinstance(params, dict):
            return ""

        category = params.get("category")
        if isinstance(category, str) and category:
            return f"mvtec_{category}"

        name = datamodule.get("name")
        return name if isinstance(name, str) else ""
