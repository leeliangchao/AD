"""Evidence model for reference-conditioned violation scoring."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.base import BaseEvidenceModel


class ConditionalViolationEvidence(BaseEvidenceModel):
    """Measure deviation from a reference-conditioned normal projection."""

    required_capabilities = {"reference_projection"}

    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Build anomaly evidence from the conditional projection."""

        self.ensure_required_capabilities(artifacts)
        image = self.require_image_tensor(sample).float()
        reference_projection = artifacts.get_primary("reference_projection")
        if not isinstance(reference_projection, torch.Tensor):
            raise TypeError("ConditionalViolationEvidence expects tensor-valued reference_projection.")
        if image.shape != reference_projection.shape:
            raise ValueError("sample.image and reference_projection must have the same shape.")

        anomaly_map = torch.abs(image - reference_projection.float()).mean(dim=0)
        return self.build_prediction(anomaly_map)
