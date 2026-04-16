"""Evidence model based on step-to-step trajectory mismatch."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel
from adrf.evidence.diffusion_scorers import score_direction_mismatch_from_step_updates


class DirectionMismatchEvidence(BaseEvidenceModel):
    """Convert trajectory step differences into anomaly evidence."""

    required_capabilities = set()

    def __init__(self, aggregator: str = "max", direction_reduce: str = "sum") -> None:
        super().__init__(aggregator=aggregator)
        if direction_reduce not in {"sum", "mean"}:
            raise ValueError("direction_reduce must be either 'sum' or 'mean'.")
        self.direction_reduce = direction_reduce

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build anomaly evidence from trajectory-only process artifacts."""

        del sample
        transitions = self._resolve_transitions(artifacts)
        anomaly_map = score_direction_mismatch_from_step_updates(
            transitions,
            direction_reduce=self.direction_reduce,
        )

        return self.build_prediction(
            anomaly_map,
            aux_scores={"num_steps": len(transitions) + 1},
        )

    def _resolve_transitions(self, artifacts: NormalityArtifacts) -> list[torch.Tensor]:
        """Resolve canonical step updates first, then fall back to trajectory differences."""

        raw_step_updates = artifacts.get_aux("step_updates")
        if isinstance(raw_step_updates, list) and raw_step_updates:
            updates = []
            for update in raw_step_updates:
                if not isinstance(update, torch.Tensor):
                    raise TypeError("Each step update must be a torch.Tensor.")
                updates.append(update.float())
            if len(updates) < 2:
                raise ValueError("DirectionMismatchEvidence requires at least two step updates.")
            return updates

        if not artifacts.has("trajectory"):
            raise KeyError("Missing required capabilities: trajectory")
        self.ensure_required_capabilities(artifacts)
        trajectory = artifacts.get_aux("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise TypeError("DirectionMismatchEvidence expects artifacts['trajectory'] to be a non-empty list.")
        if len(trajectory) < 2:
            raise ValueError("DirectionMismatchEvidence requires at least two trajectory states.")

        transitions = []
        for previous_state, current_state in zip(trajectory[:-1], trajectory[1:], strict=True):
            if not isinstance(previous_state, torch.Tensor) or not isinstance(current_state, torch.Tensor):
                raise TypeError("Each trajectory state must be a torch.Tensor.")
            if previous_state.shape != current_state.shape:
                raise ValueError("All trajectory states must share the same shape.")
            transitions.append(current_state.float() - previous_state.float())
        return transitions
