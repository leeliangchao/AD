"""Evidence model based on residual error between predicted and target noise."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel


class NoiseResidualEvidence(BaseEvidenceModel):
    """Compute a pixel anomaly map from predicted-vs-target noise residuals."""

    required_capabilities = {"predicted_noise", "target_noise"}

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build anomaly evidence from diffusion-style noise artifacts."""

        del sample
        self.ensure_required_capabilities(artifacts)
        predicted_noise = artifacts.get_aux("predicted_noise")
        target_noise = artifacts.get_aux("target_noise")
        if not isinstance(predicted_noise, torch.Tensor) or not isinstance(target_noise, torch.Tensor):
            raise TypeError("NoiseResidualEvidence expects tensor-valued predicted_noise and target_noise.")
        if predicted_noise.shape != target_noise.shape:
            raise ValueError("predicted_noise and target_noise must have the same shape.")

        residual = torch.abs(predicted_noise.float() - target_noise.float())
        if residual.ndim == 3:
            anomaly_map = residual.mean(dim=0)
        elif residual.ndim == 2:
            anomaly_map = residual
        else:
            raise ValueError("NoiseResidualEvidence expects 2D or 3D noise tensors.")
        return self.build_prediction(anomaly_map)
