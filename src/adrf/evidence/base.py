"""Base helpers for evidence models."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.interfaces import EvidenceModel
from adrf.core.sample import Sample
from adrf.evaluation.aggregators import max_pool_score, mean_pool_score, topk_mean_score


AggregatorFn = Callable[[torch.Tensor | np.ndarray], float]


class BaseEvidenceModel(EvidenceModel, ABC):
    """Common capability checks and score aggregation for evidence models."""

    required_capabilities: set[str] = set()
    aggregator_name: str
    aggregator_fn: AggregatorFn

    def __init__(self, aggregator: str = "max") -> None:
        self.aggregator_name = aggregator
        self.aggregator_fn = self._resolve_aggregator(aggregator)

    def ensure_required_capabilities(self, artifacts: NormalityArtifacts) -> None:
        """Validate that all subclass-declared capabilities are present."""

        if self.required_capabilities:
            artifacts.require(*sorted(self.required_capabilities))

    def build_prediction(
        self,
        anomaly_map: torch.Tensor | np.ndarray,
        *,
        image_score: float | None = None,
        aux_scores: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the unified evidence prediction payload."""

        score = image_score if image_score is not None else self.aggregator_fn(anomaly_map)
        payload = dict(aux_scores or {})
        payload.setdefault("aggregator", self.aggregator_name)
        return {
            "anomaly_map": anomaly_map,
            "image_score": float(score),
            "aux_scores": payload,
        }

    @staticmethod
    def require_image_tensor(sample: Sample) -> torch.Tensor:
        """Return the sample image tensor or raise when it is unavailable."""

        if not isinstance(sample.image, torch.Tensor):
            raise TypeError("Evidence models expect sample.image to be a torch.Tensor.")
        return sample.image

    @staticmethod
    def _resolve_aggregator(name: str) -> AggregatorFn:
        """Map an aggregator name onto its implementation."""

        aggregators: dict[str, AggregatorFn] = {
            "max": max_pool_score,
            "mean": mean_pool_score,
            "topk_mean": topk_mean_score,
        }
        if name not in aggregators:
            available = ", ".join(sorted(aggregators))
            raise ValueError(f"Unknown aggregator '{name}'. Available aggregators: {available}.")
        return aggregators[name]

