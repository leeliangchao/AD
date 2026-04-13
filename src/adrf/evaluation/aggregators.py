"""Aggregation helpers for converting anomaly maps into image-level scores."""

from __future__ import annotations

import math

import numpy as np
import torch


def max_pool_score(anomaly_map: torch.Tensor | np.ndarray) -> float:
    """Return the maximum value from an anomaly map."""

    values = _flatten_input(anomaly_map)
    return float(values.max())


def mean_pool_score(anomaly_map: torch.Tensor | np.ndarray) -> float:
    """Return the mean value from an anomaly map."""

    values = _flatten_input(anomaly_map)
    return float(values.mean())


def topk_mean_score(anomaly_map: torch.Tensor | np.ndarray, k_ratio: float = 0.01) -> float:
    """Return the mean over the highest-scoring top-k fraction of the map."""

    if not 0.0 < k_ratio <= 1.0:
        raise ValueError("k_ratio must be in the interval (0, 1].")
    values = _flatten_input(anomaly_map)
    k = max(1, int(math.ceil(values.size * k_ratio)))
    topk_values = np.partition(values, values.size - k)[-k:]
    return float(topk_values.mean())


def _flatten_input(anomaly_map: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert an anomaly map into a flat numpy array."""

    if isinstance(anomaly_map, torch.Tensor):
        values = anomaly_map.detach().cpu().numpy()
    else:
        values = np.asarray(anomaly_map)

    flattened = values.astype(float, copy=False).reshape(-1)
    if flattened.size == 0:
        raise ValueError("Cannot aggregate an empty anomaly map.")
    return flattened

