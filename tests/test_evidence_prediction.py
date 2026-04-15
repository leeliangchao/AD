from __future__ import annotations

import numpy as np
import pytest
import torch

from adrf.evidence.prediction import EvidencePrediction, normalize_evidence_prediction_input


def test_evidence_prediction_to_dict_preserves_prediction_shape() -> None:
    prediction = EvidencePrediction(
        anomaly_map=torch.ones(2, 2),
        image_score=0.5,
        aux_scores={"aggregator": "mean"},
    )

    assert prediction.to_dict() == {
        "anomaly_map": prediction.anomaly_map,
        "image_score": 0.5,
        "aux_scores": {"aggregator": "mean"},
    }


def test_normalize_evidence_prediction_input_accepts_mapping() -> None:
    normalized = normalize_evidence_prediction_input(
        {
            "anomaly_map": np.zeros((2, 2), dtype=float),
            "image_score": 1,
            "aux_scores": {"aggregator": "max"},
        }
    )

    assert isinstance(normalized, EvidencePrediction)
    assert np.array_equal(normalized.anomaly_map, np.zeros((2, 2), dtype=float))
    assert normalized.image_score == 1.0
    assert normalized.aux_scores == {"aggregator": "max"}


def test_normalize_evidence_prediction_input_rejects_missing_required_keys() -> None:
    with pytest.raises(KeyError, match="image_score"):
        normalize_evidence_prediction_input({"anomaly_map": torch.zeros(2, 2)})
