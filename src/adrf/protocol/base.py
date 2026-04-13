"""Base protocol helpers for experiment orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseProtocol(ABC):
    """Minimal protocol interface for training and evaluation orchestration."""

    @abstractmethod
    def train_epoch(self, runner: Any) -> dict[str, Any]:
        """Execute the training phase for a configured runner."""

    @abstractmethod
    def evaluate(self, runner: Any) -> dict[str, float]:
        """Execute evaluation and return metric values."""

    def run(self, runner: Any) -> dict[str, Any]:
        """Execute training followed by evaluation."""

        return {
            "train": self.train_epoch(runner),
            "evaluation": self.evaluate(runner),
        }

