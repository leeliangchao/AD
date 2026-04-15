from __future__ import annotations

import numpy as np

from adrf.evaluation.state import ADEvaluatorState


def test_evaluator_state_round_trips_serialized_payload() -> None:
    state = ADEvaluatorState(
        image_labels=[0, 1],
        image_scores=[0.1, 0.9],
        pixel_masks=[np.zeros((2, 2), dtype=int)],
        pixel_maps=[np.ones((2, 2), dtype=float)],
    )

    restored = ADEvaluatorState.from_mapping(state.to_dict())

    assert restored.image_labels == [0, 1]
    assert restored.image_scores == [0.1, 0.9]
    assert np.array_equal(restored.pixel_masks[0], np.zeros((2, 2), dtype=int))
    assert np.array_equal(restored.pixel_maps[0], np.ones((2, 2), dtype=float))


def test_evaluator_state_merge_concatenates_multiple_states() -> None:
    merged = ADEvaluatorState.merge(
        [
            {"image_labels": [0], "image_scores": [0.1], "pixel_masks": [np.zeros((1, 1))], "pixel_maps": [np.zeros((1, 1))]},
            {"image_labels": [1], "image_scores": [0.9], "pixel_masks": [np.ones((1, 1))], "pixel_maps": [np.ones((1, 1))]},
        ]
    )

    assert merged.image_labels == [0, 1]
    assert merged.image_scores == [0.1, 0.9]
    assert len(merged.pixel_masks) == 2
    assert len(merged.pixel_maps) == 2
