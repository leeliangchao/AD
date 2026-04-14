"""Feature-space representation built on top of a torchvision backbone."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationProvenance


class FeatureRepresentation(BaseRepresentation):
    """Extract convolutional feature maps from a ResNet18 backbone."""

    space = "feature"

    def __init__(
        self,
        weights: str | None = "imagenet1k_v1",
        trainable: bool = False,
        input_image_size: tuple[int, int] = (256, 256),
        input_normalize: bool = False,
    ) -> None:
        super().__init__(input_image_size=input_image_size, input_normalize=input_normalize)
        resolved_weights = ResNet18_Weights.DEFAULT if weights == "imagenet1k_v1" else None
        backbone = resnet18(weights=resolved_weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.feature_dim = backbone.fc.in_features
        self.weights = weights
        self.trainable = trainable
        if not self.trainable:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _encode_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            return self.backbone(batch)

        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(batch)
        self.backbone.train(was_training)
        return features

    def describe(self) -> RepresentationProvenance:
        return RepresentationProvenance(
            representation_name="feature",
            backbone_name="resnet18",
            weights_source=self.weights,
            feature_layer="layer4",
            pooling=None,
            trainable=self.trainable,
            frozen_submodules=() if self.trainable else ("backbone",),
            input_image_size=self.input_image_size,
            input_normalize=self.input_normalize,
            normalize_mean=None,
            normalize_std=None,
            code_version="working-tree",
            config_fingerprint=f"feature:{self.weights}:{self.trainable}:{self.input_image_size}:{self.input_normalize}",
        )
