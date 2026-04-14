import torch

from adrf.core.sample import Sample
from adrf.representation.feature import FeatureRepresentation


def test_feature_representation_trainable_mode_preserves_gradients() -> None:
    model = FeatureRepresentation(weights=None, trainable=True, input_image_size=(64, 64), input_normalize=False)
    output = model.encode_sample(Sample(image=torch.rand(3, 64, 64), sample_id="trainable"))

    loss = output.tensor.mean()
    loss.backward()

    first_trainable = next(parameter for parameter in model.backbone.parameters() if parameter.requires_grad)
    assert output.requires_grad is True
    assert first_trainable.grad is not None


def test_feature_representation_frozen_mode_detaches_outputs() -> None:
    model = FeatureRepresentation(weights=None, trainable=False, input_image_size=(64, 64), input_normalize=False)
    output = model.encode_sample(Sample(image=torch.rand(3, 64, 64), sample_id="frozen"))

    assert output.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in model.backbone.parameters())
