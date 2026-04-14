"""Tests for the basic anomaly detection evaluator."""

import pytest
import torch

from adrf.core.sample import Sample
from adrf.evaluation.evaluator import BasicADEvaluator


def test_basic_evaluator_accumulates_and_computes_metrics() -> None:
    """Evaluator should collect image- and pixel-level signals and compute metrics."""

    evaluator = BasicADEvaluator()

    evaluator.update(
        {"anomaly_map": torch.tensor([[0.1, 0.2], [0.2, 0.1]]), "image_score": 0.2, "aux_scores": {}},
        Sample(image=torch.zeros(1, 2, 2), label=0, mask=torch.zeros(1, 2, 2)),
    )
    evaluator.update(
        {"anomaly_map": torch.tensor([[0.1, 0.2], [0.9, 0.8]]), "image_score": 0.9, "aux_scores": {}},
        Sample(image=torch.zeros(1, 2, 2), label=1, mask=torch.tensor([[[0, 0], [1, 1]]], dtype=torch.float32)),
    )

    metrics = evaluator.compute()

    assert metrics["image_auroc"] == pytest.approx(1.0)
    assert metrics["pixel_auroc"] == pytest.approx(1.0)
    assert metrics["pixel_aupr"] == pytest.approx(1.0)


def test_basic_evaluator_reset_clears_cached_state() -> None:
    """Reset should clear accumulated predictions and labels."""

    evaluator = BasicADEvaluator()
    evaluator.update(
        {"anomaly_map": torch.zeros(2, 2), "image_score": 0.1, "aux_scores": {}},
        Sample(image=torch.zeros(1, 2, 2), label=0, mask=torch.zeros(1, 2, 2)),
    )

    evaluator.reset()

    with pytest.raises(ValueError, match="No predictions"):
        evaluator.compute()


def test_basic_evaluator_state_roundtrip_and_merge() -> None:
    """Evaluator state should be serializable and mergeable across worker processes."""

    first = BasicADEvaluator()
    first.update(
        {"anomaly_map": torch.zeros(2, 2), "image_score": 0.1, "aux_scores": {}},
        Sample(image=torch.zeros(1, 2, 2), label=0, mask=torch.zeros(1, 2, 2)),
    )
    second = BasicADEvaluator()
    second.update(
        {"anomaly_map": torch.ones(2, 2), "image_score": 0.9, "aux_scores": {}},
        Sample(image=torch.zeros(1, 2, 2), label=1, mask=torch.ones(1, 2, 2)),
    )

    merged = BasicADEvaluator.merge_states([first.state_dict(), second.state_dict()])
    restored = BasicADEvaluator()
    restored.load_state_dict(merged)
    metrics = restored.compute()

    assert metrics["image_auroc"] == pytest.approx(1.0)
    assert metrics["pixel_auroc"] == pytest.approx(1.0)
    assert metrics["pixel_aupr"] == pytest.approx(1.0)
