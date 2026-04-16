"""Evidence model based on residual error between predicted and target noise."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel
from adrf.evidence.diffusion_scorers import score_noise_residual


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
        anomaly_map = score_noise_residual(predicted_noise, target_noise)
        return self.build_prediction(anomaly_map)
