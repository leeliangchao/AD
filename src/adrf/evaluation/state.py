from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ADEvaluatorState:
    image_labels: list[int]
    image_scores: list[float]
    pixel_masks: list[np.ndarray]
    pixel_maps: list[np.ndarray]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_labels": list(self.image_labels),
            "image_scores": list(self.image_scores),
            "pixel_masks": [np.asarray(mask).copy() for mask in self.pixel_masks],
            "pixel_maps": [np.asarray(anomaly_map).copy() for anomaly_map in self.pixel_maps],
        }

    @classmethod
    def from_mapping(cls, state: Mapping[str, Any]) -> "ADEvaluatorState":
        return cls(
            image_labels=[int(value) for value in state.get("image_labels", [])],
            image_scores=[float(value) for value in state.get("image_scores", [])],
            pixel_masks=[np.asarray(mask) for mask in state.get("pixel_masks", [])],
            pixel_maps=[np.asarray(anomaly_map) for anomaly_map in state.get("pixel_maps", [])],
        )

    @classmethod
    def merge(cls, states: list[Mapping[str, Any]]) -> "ADEvaluatorState":
        image_labels: list[int] = []
        image_scores: list[float] = []
        pixel_masks: list[np.ndarray] = []
        pixel_maps: list[np.ndarray] = []
        for state in states:
            normalized = cls.from_mapping(state)
            image_labels.extend(normalized.image_labels)
            image_scores.extend(normalized.image_scores)
            pixel_masks.extend(normalized.pixel_masks)
            pixel_maps.extend(normalized.pixel_maps)
        return cls(
            image_labels=image_labels,
            image_scores=image_scores,
            pixel_masks=pixel_masks,
            pixel_maps=pixel_maps,
        )
