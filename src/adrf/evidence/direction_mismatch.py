"""Evidence model based on step-to-step trajectory mismatch."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel


class DirectionMismatchEvidence(BaseEvidenceModel):
    """Convert trajectory step differences into anomaly evidence."""

    required_capabilities = {"trajectory"}

    def __init__(self, aggregator: str = "max", direction_reduce: str = "sum") -> None:
        super().__init__(aggregator=aggregator)
        if direction_reduce not in {"sum", "mean"}:
            raise ValueError("direction_reduce must be either 'sum' or 'mean'.")
        self.direction_reduce = direction_reduce

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build anomaly evidence from trajectory-only process artifacts."""

        del sample
        self.ensure_required_capabilities(artifacts)
        trajectory = artifacts.get_aux("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise TypeError("DirectionMismatchEvidence expects artifacts['trajectory'] to be a non-empty list.")
        if len(trajectory) < 2:
            raise ValueError("DirectionMismatchEvidence requires at least two trajectory states.")

        transition_maps = []
        for previous_state, current_state in zip(trajectory[:-1], trajectory[1:], strict=True):
            if not isinstance(previous_state, torch.Tensor) or not isinstance(current_state, torch.Tensor):
                raise TypeError("Each trajectory state must be a torch.Tensor.")
            if previous_state.shape != current_state.shape:
                raise ValueError("All trajectory states must share the same shape.")
            transition = torch.abs(current_state.float() - previous_state.float())
            transition_maps.append(self._reduce_channels(transition))

        stacked = torch.stack(transition_maps, dim=0)
        if self.direction_reduce == "sum":
            anomaly_map = stacked.sum(dim=0)
        else:
            anomaly_map = stacked.mean(dim=0)

        return self.build_prediction(
            anomaly_map,
            aux_scores={"num_steps": len(trajectory)},
        )

    @staticmethod
    def _reduce_channels(tensor: torch.Tensor) -> torch.Tensor:
        """Reduce a trajectory difference tensor into a 2D map."""

        if tensor.ndim == 3:
            return tensor.mean(dim=0)
        if tensor.ndim == 2:
            return tensor
        raise ValueError("Trajectory states must be 2D or 3D tensors.")
