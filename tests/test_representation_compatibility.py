from pathlib import Path
import sys

import pytest
import torch

from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.feature_memory import FeatureMemoryNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_feature_output, make_pixel_output


def test_feature_memory_rejects_pixel_representations() -> None:
    model = FeatureMemoryNormality()

    with pytest.raises(ValueError, match="space `feature`"):
        model.fit([make_pixel_output(torch.ones(3, 8, 8))])


def test_feature_memory_rejects_grad_carrying_feature_outputs() -> None:
    tensor = torch.ones(4, 2, 2, requires_grad=True)
    model = FeatureMemoryNormality()

    with pytest.raises(ValueError, match="detached"):
        model.fit([make_feature_output(tensor, requires_grad=True)])


def test_autoencoder_rejects_feature_representations() -> None:
    model = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8)

    with pytest.raises(ValueError, match="space `pixel`"):
        model.fit([make_feature_output(torch.ones(4, 2, 2))])


def test_autoencoder_rejects_grad_carrying_pixel_outputs() -> None:
    tensor = torch.ones(3, 8, 8, requires_grad=True)
    model = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8)

    with pytest.raises(ValueError, match="detached"):
        model.fit([make_pixel_output(tensor)])
