"""Evidence model based on memory distances in feature space."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel


class FeatureDistanceEvidence(BaseEvidenceModel):
    """Convert feature-memory distances into an anomaly map and image score."""

    required_capabilities = {"feature_response", "memory_distance"}

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build evidence from the memory-distance artifact payload."""

        del sample
        self.ensure_required_capabilities(artifacts)
        anomaly_map = artifacts.get_aux("memory_distance")
        if not isinstance(anomaly_map, torch.Tensor):
            raise TypeError("FeatureDistanceEvidence expects artifacts['memory_distance'] to be a torch.Tensor.")
        return self.build_prediction(anomaly_map)

