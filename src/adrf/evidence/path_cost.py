"""Evidence model based on accumulated process costs along a diffusion path."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel
from adrf.evidence.diffusion_scorers import score_path_cost_from_step_costs, score_path_cost_from_step_updates


class PathCostEvidence(BaseEvidenceModel):
    """Accumulate step-aligned cost maps into anomaly evidence."""

    required_capabilities = set()

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build anomaly evidence from a sequence of step cost maps."""

        del sample
        raw_step_updates = artifacts.get_aux("step_updates")
        if isinstance(raw_step_updates, list) and raw_step_updates:
            anomaly_map = score_path_cost_from_step_updates(raw_step_updates)
        else:
            if not artifacts.has("step_costs"):
                raise KeyError("Missing required capabilities: step_costs")
            self.ensure_required_capabilities(artifacts)
            raw_step_costs = artifacts.get_aux("step_costs")
            if not isinstance(raw_step_costs, list) or not raw_step_costs:
                raise TypeError("PathCostEvidence expects artifacts['step_costs'] to be a non-empty list.")
            anomaly_map = score_path_cost_from_step_costs(raw_step_costs)
        return self.build_prediction(
            anomaly_map,
            aux_scores={"path_cost_total": float(anomaly_map.sum().item())},
        )
