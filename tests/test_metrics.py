"""Tests for image-level and pixel-level anomaly metrics."""

import numpy as np
import pytest

from adrf.evaluation.metrics import compute_image_auroc, compute_pixel_aupr, compute_pixel_auroc


def test_metrics_compute_expected_scores_for_separable_predictions() -> None:
    """Basic metric functions should return 1.0 for perfectly separable inputs."""

    assert compute_image_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert compute_pixel_auroc(
        [np.array([[0, 0], [1, 1]])],
        [np.array([[0.1, 0.2], [0.8, 0.9]])],
    ) == pytest.approx(1.0)
    assert compute_pixel_aupr(
        [np.array([[0, 0], [1, 1]])],
        [np.array([[0.1, 0.2], [0.8, 0.9]])],
    ) == pytest.approx(1.0)


def test_metrics_validate_empty_inputs_and_shape_mismatch() -> None:
    """Metrics should fail fast on invalid inputs."""

    with pytest.raises(ValueError, match="empty"):
        compute_image_auroc([], [])
    with pytest.raises(ValueError, match="same length"):
        compute_image_auroc([0], [0.1, 0.2])
    with pytest.raises(ValueError, match="same shape"):
        compute_pixel_auroc([np.zeros((2, 2))], [np.zeros((3, 3))])


def test_pixel_metrics_require_positive_and_negative_examples() -> None:
    """Degenerate masks should fail instead of returning misleading scores."""

    masks = [np.zeros((2, 2), dtype=int)]
    scores = [np.zeros((2, 2), dtype=float)]

    with pytest.raises(ValueError, match="both positive and negative"):
        compute_pixel_auroc(masks, scores)
    with pytest.raises(ValueError, match="both positive and negative"):
        compute_pixel_aupr(masks, scores)

