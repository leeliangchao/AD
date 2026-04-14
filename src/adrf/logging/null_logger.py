"""No-op logger used on non-primary distributed ranks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from adrf.logging.base import BaseLogger


class NullLogger(BaseLogger):
    """Discard logging and artifact writes for non-primary worker processes."""

    def start_run(
        self,
        run_name: str,
        config: dict[str, Any],
        run_info: dict[str, Any] | None = None,
    ) -> None:
        del run_name, config, run_info

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        del metrics, step

    def log_run_info(self, updates: dict[str, Any]) -> None:
        del updates

    def log_artifact(self, path: str | Path, artifact_type: str | None = None) -> None:
        del path, artifact_type

    def finish_run(self, status: str = "completed") -> None:
        del status

    def resolve_artifact_path(self, relative_path: str | Path) -> Path:
        return Path(relative_path)
