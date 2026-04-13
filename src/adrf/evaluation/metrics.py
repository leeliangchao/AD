"""Minimal anomaly-detection metrics for image- and pixel-level evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


ArrayLike = torch.Tensor | np.ndarray | Sequence[float] | Sequence[int]


def compute_image_auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Compute image-level AUROC for binary anomaly labels."""

    y_true = _to_flat_numpy(labels, name="labels").astype(int, copy=False)
    y_score = _to_flat_numpy(scores, name="scores").astype(float, copy=False)
    _validate_binary_problem(y_true, y_score)
    return _binary_auroc(y_true, y_score)


def compute_pixel_auroc(masks: Sequence[ArrayLike], anomaly_maps: Sequence[ArrayLike]) -> float:
    """Compute pixel-level AUROC across one or more masks and anomaly maps."""

    y_true, y_score = _flatten_pixel_inputs(masks, anomaly_maps)
    _validate_binary_problem(y_true, y_score)
    return _binary_auroc(y_true, y_score)


def compute_pixel_aupr(masks: Sequence[ArrayLike], anomaly_maps: Sequence[ArrayLike]) -> float:
    """Compute pixel-level AUPR across one or more masks and anomaly maps."""

    y_true, y_score = _flatten_pixel_inputs(masks, anomaly_maps)
    _validate_binary_problem(y_true, y_score)
    return _binary_average_precision(y_true, y_score)


def _flatten_pixel_inputs(
    masks: Sequence[ArrayLike],
    anomaly_maps: Sequence[ArrayLike],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and flatten pixel-level inputs into one binary task."""

    if len(masks) == 0 or len(anomaly_maps) == 0:
        raise ValueError("Pixel metrics cannot be computed from empty inputs.")
    if len(masks) != len(anomaly_maps):
        raise ValueError("Pixel metric inputs must have the same length.")

    flat_masks: list[np.ndarray] = []
    flat_scores: list[np.ndarray] = []
    for mask, anomaly_map in zip(masks, anomaly_maps, strict=True):
        mask_array = _to_numpy_array(mask, name="mask")
        score_array = _to_numpy_array(anomaly_map, name="anomaly_map")
        if mask_array.shape != score_array.shape:
            raise ValueError("Pixel metric mask and anomaly map must have the same shape.")
        flat_masks.append(mask_array.reshape(-1).astype(int, copy=False))
        flat_scores.append(score_array.reshape(-1).astype(float, copy=False))
    return np.concatenate(flat_masks), np.concatenate(flat_scores)


def _validate_binary_problem(y_true: np.ndarray, y_score: np.ndarray) -> None:
    """Validate binary labels and aligned prediction scores."""

    if y_true.size == 0 or y_score.size == 0:
        raise ValueError("Metrics cannot be computed from empty inputs.")
    if y_true.shape != y_score.shape:
        raise ValueError("Ground-truth labels and scores must have the same length.")
    unique_values = np.unique(y_true)
    if not np.isin(unique_values, [0, 1]).all():
        raise ValueError("Ground-truth labels must be binary with values in {0, 1}.")
    if unique_values.size != 2:
        raise ValueError("Metrics require both positive and negative examples.")


def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC for a validated binary classification task."""

    order = np.argsort(y_score, kind="mergesort")[::-1]
    sorted_true = y_true[order]
    sorted_score = y_score[order]

    distinct_indices = np.where(np.diff(sorted_score))[0]
    threshold_indices = np.r_[distinct_indices, sorted_true.size - 1]
    true_positives = np.cumsum(sorted_true)[threshold_indices]
    false_positives = (threshold_indices + 1) - true_positives

    positives = int(sorted_true.sum())
    negatives = int(sorted_true.size - positives)
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    return float(np.trapezoid(tpr, fpr))


def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision for a validated binary classification task."""

    order = np.argsort(y_score, kind="mergesort")[::-1]
    sorted_true = y_true[order]
    sorted_score = y_score[order]

    distinct_indices = np.where(np.diff(sorted_score))[0]
    threshold_indices = np.r_[distinct_indices, sorted_true.size - 1]
    true_positives = np.cumsum(sorted_true)[threshold_indices]
    false_positives = (threshold_indices + 1) - true_positives

    positives = int(sorted_true.sum())
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positives
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _to_flat_numpy(values: ArrayLike, *, name: str) -> np.ndarray:
    """Convert a vector-like input into a flat numpy array."""

    array = _to_numpy_array(values, name=name).reshape(-1)
    return array


def _to_numpy_array(values: ArrayLike, *, name: str) -> np.ndarray:
    """Convert supported tensor-like values into a numpy array."""

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    squeezed = np.squeeze(array)
    if squeezed.ndim == 0:
        return squeezed.reshape(1)
    if squeezed.ndim == 3 and squeezed.shape[0] == 1:
        return squeezed[0]
    if squeezed.ndim > 2:
        raise ValueError(f"{name} must be at most 2-dimensional after squeezing.")
    return squeezed

