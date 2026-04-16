"""Evidence model based on reconstruction residuals."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel
from adrf.evidence.diffusion_scorers import score_reconstruction_residual


class ReconstructionResidualEvidence(BaseEvidenceModel):
    """Compute L1 reconstruction residuals from normality artifacts."""

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build evidence from a reconstruction or projection artifact."""

        reconstruction = self._resolve_reconstruction(artifacts)
        image = self.require_image_tensor(sample).float()
        reconstruction = reconstruction.float()
        anomaly_map = score_reconstruction_residual(image, reconstruction)
        return self.build_prediction(anomaly_map)

    @staticmethod
    def _resolve_reconstruction(artifacts: NormalityArtifacts) -> torch.Tensor:
        """Read reconstruction-like content from artifacts with fallback behavior."""

        if "reconstruction" in artifacts.primary:
            reconstruction = artifacts.get_primary("reconstruction")
        elif artifacts.has("reconstruction"):
            reconstruction = artifacts.get_primary("reconstruction")
        elif "projection" in artifacts.primary:
            reconstruction = artifacts.get_primary("projection")
        elif artifacts.has("projection"):
            reconstruction = artifacts.get_primary("projection")
        else:
            raise KeyError("ReconstructionResidualEvidence requires 'reconstruction' or 'projection'.")

        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError("ReconstructionResidualEvidence expects tensor-valued reconstruction artifacts.")
        return reconstruction
