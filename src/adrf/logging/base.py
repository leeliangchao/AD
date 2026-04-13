"""Abstract logger interface for experiment management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseLogger(ABC):
    """Minimal logger abstraction for local and future remote backends."""

    @abstractmethod
    def start_run(
        self,
        run_name: str,
        config: dict[str, Any],
        run_info: dict[str, Any] | None = None,
    ) -> None:
        """Start a run and persist its initial metadata."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Persist scalar metrics for the active run."""

    @abstractmethod
    def log_run_info(self, updates: dict[str, Any]) -> None:
        """Update run-level metadata for the active run."""

    @abstractmethod
    def log_artifact(self, path: str | Path, artifact_type: str | None = None) -> None:
        """Register or copy an artifact into the active run directory."""

    @abstractmethod
    def finish_run(self, status: str = "completed") -> None:
        """Mark the active run as completed or failed."""

    @abstractmethod
    def resolve_artifact_path(self, relative_path: str | Path) -> Path:
        """Resolve a writable path inside the active run directory."""

    @property
    def is_remote(self) -> bool:
        """Return whether the logger primarily targets a remote backend."""

        return False
