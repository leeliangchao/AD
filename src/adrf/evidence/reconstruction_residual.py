"""Evidence model based on reconstruction residuals."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel


class ReconstructionResidualEvidence(BaseEvidenceModel):
    """Compute L1 reconstruction residuals from normality artifacts."""

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build evidence from a reconstruction or projection artifact."""

        reconstruction = self._resolve_reconstruction(artifacts)
        image = self.require_image_tensor(sample).float()
        reconstruction = reconstruction.float()
        if image.shape != reconstruction.shape:
            raise ValueError("Sample image and reconstruction must have the same shape.")
        anomaly_map = torch.abs(image - reconstruction).mean(dim=0)
        return self.build_prediction(anomaly_map)

    @staticmethod
    def _resolve_reconstruction(artifacts: NormalityArtifacts) -> torch.Tensor:
        """Read reconstruction-like content from artifacts with fallback behavior."""

        if artifacts.has("reconstruction"):
            reconstruction = artifacts.get_primary("reconstruction")
        elif artifacts.has("projection"):
            reconstruction = artifacts.get_primary("projection")
        else:
            raise KeyError("ReconstructionResidualEvidence requires 'reconstruction' or 'projection'.")

        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError("ReconstructionResidualEvidence expects tensor-valued reconstruction artifacts.")
        return reconstruction

