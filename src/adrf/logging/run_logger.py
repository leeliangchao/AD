"""Local filesystem logger for experiment runs."""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from adrf.logging.base import BaseLogger


class RunLogger(BaseLogger):
    """Persist experiment metadata, metrics, and artifacts under one run directory."""

    def __init__(self, base_dir: str | Path = "outputs/runs") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.run_dir: Path | None = None
        self.run_info: dict[str, Any] = {}
        self.metrics_history: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []

    def start_run(
        self,
        run_name: str,
        config: dict[str, Any],
        run_info: dict[str, Any] | None = None,
    ) -> None:
        """Create the run directory structure and write initial metadata."""

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_name = self._sanitize_name(run_name)
        run_dir = self.base_dir / f"{timestamp}_{safe_name}"
        suffix = 1
        while run_dir.exists():
            run_dir = self.base_dir / f"{timestamp}_{safe_name}_{suffix}"
            suffix += 1

        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for directory in ("checkpoints", "visualizations", "predictions", "artifacts"):
            (self.run_dir / directory).mkdir(parents=True, exist_ok=True)

        config_snapshot_path = self.run_dir / "config_snapshot.yaml"
        config_snapshot_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

        base_run_info = {
            "run_name": safe_name,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": self._default_artifact_paths(self.run_dir),
        }
        self.run_info = self._merge_nested(base_run_info, run_info or {})
        self._write_json(self.run_dir / "run_info.json", self.run_info)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Append one metric event and refresh metrics outputs."""

        run_dir = self._require_run_dir()
        flat_metrics = self._flatten_mapping(metrics)
        entry: dict[str, Any] = {"step": step}
        entry.update(flat_metrics)
        self.metrics_history.append(entry)

        latest_metrics = dict(flat_metrics)
        payload = {
            "latest": latest_metrics,
            "history": self.metrics_history,
        }
        self._write_json(run_dir / "metrics.json", payload)
        self._write_metrics_csv(run_dir / "metrics.csv")

    def log_run_info(self, updates: dict[str, Any]) -> None:
        """Merge additional run metadata into run_info.json."""

        run_dir = self._require_run_dir()
        self.run_info = self._merge_nested(self.run_info, updates)
        self._write_json(run_dir / "run_info.json", self.run_info)

    def log_artifact(self, path: str | Path, artifact_type: str | None = None) -> None:
        """Copy an artifact into the run directory or register it if already inside."""

        run_dir = self._require_run_dir()
        source = Path(path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Artifact does not exist: {source}")

        target_dir = self._artifact_dir(artifact_type)
        if self._is_relative_to(source, run_dir):
            target = source
        else:
            target = (target_dir / source.name).resolve()
            if source == target:
                self.artifacts.append(
                    {
                        "type": artifact_type or "artifact",
                        "path": str(target),
                    }
                )
                self._write_json(run_dir / "artifacts.json", {"artifacts": self.artifacts})
                return
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)

        self.artifacts.append(
            {
                "type": artifact_type or "artifact",
                "path": str(target),
            }
        )
        self._write_json(run_dir / "artifacts.json", {"artifacts": self.artifacts})

    def finish_run(self, status: str = "completed") -> None:
        """Update run metadata with the final status."""

        if self.run_dir is None:
            return
        run_dir = self._require_run_dir()
        self.run_info["status"] = status
        self.run_info["finished_at"] = datetime.now(timezone.utc).isoformat()
        if self.metrics_history:
            self.run_info["latest_metrics"] = {
                key: value
                for key, value in self.metrics_history[-1].items()
                if key != "step"
            }
        self._write_json(run_dir / "run_info.json", self.run_info)

    def resolve_artifact_path(self, relative_path: str | Path) -> Path:
        """Return a writable path inside the current run directory."""

        run_dir = self._require_run_dir()
        destination = run_dir / Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def _artifact_dir(self, artifact_type: str | None) -> Path:
        """Map an artifact type onto its target directory."""

        run_dir = self._require_run_dir()
        mapping = {
            "checkpoint": run_dir / "checkpoints",
            "checkpoints": run_dir / "checkpoints",
            "visualization": run_dir / "visualizations",
            "visualizations": run_dir / "visualizations",
            "prediction": run_dir / "predictions",
            "predictions": run_dir / "predictions",
        }
        directory = mapping.get(artifact_type or "", run_dir / "artifacts")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _require_run_dir(self) -> Path:
        """Ensure that start_run has been called."""

        if self.run_dir is None:
            raise RuntimeError("RunLogger.start_run must be called before logging.")
        return self.run_dir.resolve()

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Convert a run name into a filesystem-friendly token."""

        return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "run"

    @staticmethod
    def _flatten_mapping(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested metric mappings using dot-separated keys."""

        flattened: dict[str, Any] = {}
        for key, value in mapping.items():
            flat_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(RunLogger._flatten_mapping(value, prefix=flat_key))
            else:
                flattened[flat_key] = value
        return flattened

    def _write_metrics_csv(self, path: Path) -> None:
        """Write metric history in wide CSV form."""

        fieldnames = ["step"]
        for entry in self.metrics_history:
            for key in entry:
                if key != "step" and key not in fieldnames:
                    fieldnames.append(key)

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.metrics_history:
                writer.writerow(entry)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        """Write a JSON payload with deterministic formatting."""

        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        """Return whether one path is inside another."""

        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _default_artifact_paths(run_dir: Path) -> dict[str, str]:
        """Return the standard filesystem paths associated with one run."""

        return {
            "run_dir": str(run_dir.resolve()),
            "config_snapshot_path": str((run_dir / "config_snapshot.yaml").resolve()),
            "run_info_path": str((run_dir / "run_info.json").resolve()),
            "metrics_path": str((run_dir / "metrics.json").resolve()),
            "metrics_csv_path": str((run_dir / "metrics.csv").resolve()),
            "report_path": str((run_dir / "report.md").resolve()),
            "checkpoint_path": str((run_dir / "checkpoints" / "normality.pt").resolve()),
            "predictions_dir": str((run_dir / "predictions").resolve()),
            "visualizations_dir": str((run_dir / "visualizations").resolve()),
            "artifacts_manifest_path": str((run_dir / "artifacts.json").resolve()),
        }

    @staticmethod
    def _merge_nested(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge nested dictionaries for run metadata updates."""

        merged = dict(base)
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = RunLogger._merge_nested(merged[key], value)
            else:
                merged[key] = value
        return merged
