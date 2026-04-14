from __future__ import annotations

import torch

from adrf.representation.contracts import RepresentationOutput, RepresentationProvenance


def make_pixel_output(image: torch.Tensor, sample_id: str = "sample") -> RepresentationOutput:
    return RepresentationOutput(
        tensor=image,
        space="pixel",
        spatial_shape=tuple(image.shape[-2:]),
        feature_dim=int(image.shape[0]),
        sample_id=sample_id,
        requires_grad=image.requires_grad,
        device=str(image.device),
        dtype=str(image.dtype),
        provenance=RepresentationProvenance(
            representation_name="pixel",
            backbone_name=None,
            weights_source=None,
            feature_layer=None,
            pooling=None,
            trainable=False,
            frozen_submodules=(),
            input_image_size=tuple(image.shape[-2:]),
            input_normalize=False,
            normalize_mean=None,
            normalize_std=None,
            code_version="test-sha",
            config_fingerprint="pixel-fixture",
        ),
    )


def make_feature_output(
    tensor: torch.Tensor,
    sample_id: str = "sample",
    requires_grad: bool | None = None,
) -> RepresentationOutput:
    grad_flag = tensor.requires_grad if requires_grad is None else requires_grad
    return RepresentationOutput(
        tensor=tensor,
        space="feature",
        spatial_shape=tuple(tensor.shape[-2:]) if tensor.ndim == 3 else None,
        feature_dim=int(tensor.shape[0] if tensor.ndim == 3 else tensor.shape[-1]),
        sample_id=sample_id,
        requires_grad=grad_flag,
        device=str(tensor.device),
        dtype=str(tensor.dtype),
        provenance=RepresentationProvenance(
            representation_name="feature",
            backbone_name="resnet18",
            weights_source="fixture",
            feature_layer="layer4",
            pooling=None,
            trainable=grad_flag,
            frozen_submodules=() if grad_flag else ("backbone",),
            input_image_size=(64, 64),
            input_normalize=False,
            normalize_mean=None,
            normalize_std=None,
            code_version="test-sha",
            config_fingerprint="feature-fixture",
        ),
    )
