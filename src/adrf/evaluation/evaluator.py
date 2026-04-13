"""Evaluator for anomaly-detection predictions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from adrf.core.interfaces import Evaluator
from adrf.core.sample import Sample
from adrf.evaluation.metrics import compute_image_auroc, compute_pixel_aupr, compute_pixel_auroc


class BasicADEvaluator(Evaluator):
    """Accumulate predictions and compute minimal AD metrics."""

    def __init__(self) -> None:
        self.reset()

    def compute(self) -> dict[str, float]:
        """Compute image- and pixel-level anomaly metrics."""

        if not self.image_scores:
            raise ValueError("No predictions have been recorded.")

        return {
            "image_auroc": compute_image_auroc(self.image_labels, self.image_scores),
            "pixel_auroc": compute_pixel_auroc(self.pixel_masks, self.pixel_maps),
            "pixel_aupr": compute_pixel_aupr(self.pixel_masks, self.pixel_maps),
        }

    def reset(self) -> None:
        """Clear accumulated predictions and labels."""

        self.image_labels: list[int] = []
        self.image_scores: list[float] = []
        self.pixel_masks: list[np.ndarray] = []
        self.pixel_maps: list[np.ndarray] = []

    @staticmethod
    def _resolve_image_label(sample: Sample) -> int:
        """Resolve the image-level ground-truth anomaly label."""

        if sample.label is not None:
            return int(sample.label)
        if sample.mask is not None:
            mask = BasicADEvaluator._to_map_array(sample.mask, name="mask")
            return int(mask.max() > 0)
        raise ValueError("Sample must define either label or mask for evaluation.")

    @staticmethod
    def _to_scalar_score(value: Any) -> float:
        """Convert a scalar-like image score into a plain float."""

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Prediction image_score tensor must contain exactly one element.")
            return float(value.detach().cpu().item())
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError("Prediction image_score array must contain exactly one element.")
            return float(value.reshape(-1)[0])
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError("Prediction image_score must be a scalar number or single-element tensor/array.")

    @staticmethod
    def _to_map_array(value: Any, *, name: str) -> np.ndarray:
        """Convert tensor-like anomaly maps or masks into 2D numpy arrays."""

        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)

        squeezed = np.squeeze(array)
        if squeezed.ndim == 0:
            return squeezed.reshape(1, 1)
        if squeezed.ndim == 1:
            return squeezed
        if squeezed.ndim == 2:
            return squeezed
        raise ValueError(f"{name} must be 1D or 2D after squeezing.")

    def update(self, prediction: Mapping[str, Any], sample: Sample) -> None:
        """Store one prediction/sample pair for later metric computation."""

        if "anomaly_map" not in prediction or "image_score" not in prediction:
            raise KeyError("Prediction must contain 'anomaly_map' and 'image_score'.")

        raw_anomaly_map = self._to_map_array(prediction["anomaly_map"], name="anomaly_map")
        image_score = self._to_scalar_score(prediction["image_score"])
        image_gt = self._resolve_image_label(sample)

        if sample.mask is not None:
            mask = self._to_map_array(sample.mask, name="mask")
            anomaly_map = self._align_map_to_shape(raw_anomaly_map, mask.shape)
            pixel_gt = (mask > 0).astype(int, copy=False)
        else:
            anomaly_map = raw_anomaly_map
            pixel_gt = np.zeros(anomaly_map.shape, dtype=int)

        self.image_labels.append(image_gt)
        self.image_scores.append(image_score)
        self.pixel_masks.append(pixel_gt)
        self.pixel_maps.append(anomaly_map)

    @staticmethod
    def _align_map_to_shape(anomaly_map: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        """Resize an anomaly map to the target spatial shape when needed."""

        if anomaly_map.shape == target_shape:
            return anomaly_map.astype(float, copy=False)

        tensor = torch.as_tensor(anomaly_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = functional.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
        return resized.squeeze(0).squeeze(0).cpu().numpy().astype(float, copy=False)
