from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TrainSummary:
    num_train_batches: int = 0
    num_train_samples: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_train_batches": self.num_train_batches,
            "num_train_samples": self.num_train_samples,
            **self.metrics,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainSummary":
        metrics = {
            str(key): float(value)
            for key, value in payload.items()
            if key not in {"num_train_batches", "num_train_samples"} and isinstance(value, (int, float))
        }
        return cls(
            num_train_batches=int(payload.get("num_train_batches", 0)),
            num_train_samples=int(payload.get("num_train_samples", 0)),
            metrics=metrics,
        )


@dataclass(slots=True)
class EvaluationSummary:
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return dict(self.metrics)
