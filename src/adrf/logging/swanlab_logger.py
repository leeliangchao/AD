"""Optional SwanLab logger adapter."""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Any

from adrf.logging.base import BaseLogger


class SwanLabLoggerAdapter(BaseLogger):
    """Adapter that forwards run lifecycle events to SwanLab when available."""

    def __init__(
        self,
        project: str = "adrf",
        module: Any | None = None,
        strict: bool = False,
    ) -> None:
        self.project = project
        self._module = module
        self._strict = strict
        self._run: Any | None = None
        self.enabled = True

        if self._module is None:
            try:
                self._module = importlib.import_module("swanlab")
            except ImportError as exc:
                if strict:
                    raise
                warnings.warn(
                    "SwanLab is not installed; SwanLabLoggerAdapter will run in disabled mode.",
                    stacklevel=2,
                )
                self.enabled = False
                self._module = None

    @property
    def is_remote(self) -> bool:
        """Return whether this logger targets a remote backend."""

        return True

    def start_run(
        self,
        run_name: str,
        config: dict[str, Any],
        run_info: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a SwanLab run when the SDK is available."""

        if not self.enabled or self._module is None:
            return

        init_fn = getattr(self._module, "init", None)
        if callable(init_fn):
            self._run = init_fn(project=self.project, name=run_name, config=config, metadata=run_info or {})
        else:
            warnings.warn("SwanLab module does not expose init(); skipping run start.", stacklevel=2)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log scalar metrics to SwanLab when available."""

        if not self.enabled:
            return

        target = self._run or self._module
        log_fn = getattr(target, "log", None)
        if callable(log_fn):
            log_fn(metrics, step=step)

    def log_run_info(self, updates: dict[str, Any]) -> None:
        """Log run metadata updates when the backend supports it."""

        if not self.enabled:
            return

        target = self._run or self._module
        update_fn = getattr(target, "update_metadata", None)
        if callable(update_fn):
            update_fn(updates)

    def log_artifact(self, path: str | Path, artifact_type: str | None = None) -> None:
        """Log artifacts when the backend exposes an artifact API, else ignore."""

        if not self.enabled:
            return

        target = self._run or self._module
        log_artifact_fn = getattr(target, "log_artifact", None)
        if callable(log_artifact_fn):
            log_artifact_fn(str(path), artifact_type=artifact_type)

    def finish_run(self, status: str = "completed") -> None:
        """Finalize the SwanLab run when available."""

        if not self.enabled:
            return

        target = self._run or self._module
        finish_fn = getattr(target, "finish", None)
        if callable(finish_fn):
            finish_fn(status=status)

    def resolve_artifact_path(self, relative_path: str | Path) -> Path:
        """Remote logger does not own local artifact paths."""

        raise RuntimeError("SwanLabLoggerAdapter does not provide local artifact paths.")
