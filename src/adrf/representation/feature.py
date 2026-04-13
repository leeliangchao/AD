"""Feature-space representation built on top of a torchvision backbone."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from adrf.core.sample import Sample
from adrf.representation.base import BaseRepresentation


class FeatureRepresentation(BaseRepresentation):
    """Extract convolutional feature maps from a ResNet18 backbone."""

    def __init__(self, pretrained: bool = True, freeze: bool = True) -> None:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.feature_dim = backbone.fc.in_features
        self.freeze = freeze
        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.backbone.eval()

    def __call__(self, sample: Sample) -> dict[str, Any]:
        """Return a feature-space representation for one sample."""

        image = self.require_image_tensor(sample)
        with torch.no_grad():
            features = self.backbone(image.unsqueeze(0)).squeeze(0)
        return {
            "representation": features,
            "space_type": "feature",
            "spatial_shape": tuple(features.shape[-2:]),
            "feature_dim": int(features.shape[0]),
        }
