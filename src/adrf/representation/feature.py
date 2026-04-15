"""Feature-space representation built on top of a torchvision backbone."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationProvenance
from adrf.representation import provenance as representation_provenance

_UNSET = object()


class FeatureRepresentation(BaseRepresentation):
    """Extract convolutional feature maps from a ResNet18 backbone."""

    space = "feature"

    def __init__(
        self,
        weights: str | None | object = _UNSET,
        trainable: bool | object = _UNSET,
        input_image_size: tuple[int, int] = (256, 256),
        input_normalize: bool = False,
        *,
        pretrained: bool | None = None,
        freeze: bool | None = None,
    ) -> None:
        super().__init__(input_image_size=input_image_size, input_normalize=input_normalize)
        resolved_weight_name = self._resolve_weights(weights=weights, pretrained=pretrained)
        resolved_trainable = self._resolve_trainable(trainable=trainable, freeze=freeze)
        resolved_weights = ResNet18_Weights.DEFAULT if resolved_weight_name == "imagenet1k_v1" else None
        backbone = resnet18(weights=resolved_weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.feature_dim = backbone.fc.in_features
        self.weights = resolved_weight_name
        self.trainable = resolved_trainable
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
            code_version=representation_provenance.resolve_representation_code_version(),
            config_fingerprint=f"feature:{self.weights}:{self.trainable}:{self.input_image_size}:{self.input_normalize}",
        )

    @staticmethod
    def _resolve_weights(weights: str | None | object, pretrained: bool | None) -> str | None:
        if weights is _UNSET:
            return "imagenet1k_v1" if pretrained is not False else None
        if pretrained is None:
            FeatureRepresentation._validate_supported_weights(weights)
            return weights  # type: ignore[return-value]

        expected_weights = "imagenet1k_v1" if pretrained else None
        if weights != expected_weights:
            raise ValueError("FeatureRepresentation received conflicting 'weights' and legacy 'pretrained' arguments.")
        FeatureRepresentation._validate_supported_weights(weights)
        return weights  # type: ignore[return-value]

    @staticmethod
    def _resolve_trainable(trainable: bool | object, freeze: bool | None) -> bool:
        if trainable is _UNSET:
            return False if freeze is None else not freeze
        if freeze is None:
            return bool(trainable)

        expected_trainable = not freeze
        if bool(trainable) != expected_trainable:
            raise ValueError("FeatureRepresentation received conflicting 'trainable' and legacy 'freeze' arguments.")
        return bool(trainable)

    @staticmethod
    def _validate_supported_weights(weights: str | None | object) -> None:
        if weights not in {None, "imagenet1k_v1"}:
            raise ValueError("FeatureRepresentation received unsupported weights. Supported values: None, 'imagenet1k_v1'.")
