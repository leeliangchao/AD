"""Evidence model based on accumulated process costs along a diffusion path."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel


class PathCostEvidence(BaseEvidenceModel):
    """Accumulate step-aligned cost maps into anomaly evidence."""

    required_capabilities = set()

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build anomaly evidence from a sequence of step cost maps."""

        del sample
        raw_step_updates = artifacts.get_aux("step_updates")
        if isinstance(raw_step_updates, list) and raw_step_updates:
            step_cost_maps = [self._normalize_step_cost(update) for update in raw_step_updates]
        else:
            if not artifacts.has("step_costs"):
                raise KeyError("Missing required capabilities: step_costs")
            self.ensure_required_capabilities(artifacts)
            raw_step_costs = artifacts.get_aux("step_costs")
            if not isinstance(raw_step_costs, list) or not raw_step_costs:
                raise TypeError("PathCostEvidence expects artifacts['step_costs'] to be a non-empty list.")
            step_cost_maps = [self._normalize_step_cost(cost_map) for cost_map in raw_step_costs]
        anomaly_map = torch.stack(step_cost_maps, dim=0).sum(dim=0)
        return self.build_prediction(
            anomaly_map,
            aux_scores={"path_cost_total": float(anomaly_map.sum().item())},
        )

    @staticmethod
    def _normalize_step_cost(step_cost: torch.Tensor) -> torch.Tensor:
        """Convert a step cost tensor into a 2D map."""

        if not isinstance(step_cost, torch.Tensor):
            raise TypeError("Each step cost must be a torch.Tensor.")
        if step_cost.ndim == 2:
            return step_cost.float()
        if step_cost.ndim == 3 and step_cost.shape[0] == 1:
            return step_cost.squeeze(0).float()
        if step_cost.ndim == 3:
            return step_cost.float().mean(dim=0)
        raise ValueError("Step cost tensors must be 2D or 3D.")
