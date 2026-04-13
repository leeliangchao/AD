"""Tests for anomaly-map score aggregators."""

import math

import numpy as np
import pytest
import torch

from adrf.evaluation.aggregators import max_pool_score, mean_pool_score, topk_mean_score


def test_aggregators_accept_numpy_and_torch_inputs() -> None:
    """Aggregators should return scalar floats for tensor-like inputs."""

    anomaly_map = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    numpy_map = anomaly_map.numpy()

    assert max_pool_score(anomaly_map) == pytest.approx(4.0)
    assert mean_pool_score(numpy_map) == pytest.approx(2.5)
    assert topk_mean_score(numpy_map, k_ratio=0.5) == pytest.approx(3.5)


def test_topk_mean_score_rejects_invalid_ratios() -> None:
    """Top-k aggregation should validate the requested ratio."""

    with pytest.raises(ValueError, match="k_ratio"):
        topk_mean_score(np.array([1.0, 2.0]), k_ratio=0.0)


def test_aggregators_reject_empty_inputs() -> None:
    """Empty maps should fail with a clear error."""

    with pytest.raises(ValueError, match="empty"):
        max_pool_score(torch.tensor([]))
    with pytest.raises(ValueError, match="empty"):
        mean_pool_score(np.array([]))
    with pytest.raises(ValueError, match="empty"):
        topk_mean_score(np.array([]))

