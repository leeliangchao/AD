from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidencePrediction:
    anomaly_map: Any
    image_score: float
    aux_scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_map": self.anomaly_map,
            "image_score": float(self.image_score),
            "aux_scores": dict(self.aux_scores),
        }


def normalize_evidence_prediction_input(
    prediction: EvidencePrediction | Mapping[str, Any],
) -> EvidencePrediction:
    if isinstance(prediction, EvidencePrediction):
        return prediction
    if not isinstance(prediction, Mapping):
        raise TypeError(f"Evidence predictions must be mappings or EvidencePrediction, got {type(prediction).__name__}.")
    if "anomaly_map" not in prediction or "image_score" not in prediction:
        missing = [key for key in ("anomaly_map", "image_score") if key not in prediction]
        raise KeyError(", ".join(missing))
    return EvidencePrediction(
        anomaly_map=prediction["anomaly_map"],
        image_score=float(prediction["image_score"]),
        aux_scores=dict(prediction.get("aux_scores", {})),
    )
