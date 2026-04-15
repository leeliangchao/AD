"""Tests for the minimal reference-conditioned diffusion normality model."""

from collections.abc import Iterable
from pathlib import Path
import sys
from typing import get_type_hints

from PIL import Image
import torch

from adrf.core.sample import Sample
from adrf.data.transforms import SampleTransform
from adrf.normality.reference_diffusion_basic import ReferenceDiffusionBasicNormality
from adrf.representation.contracts import RepresentationOutput

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_reference_diffusion_public_annotations_expose_representation_output_only() -> None:
    """ReferenceDiffusionBasicNormality should expose typed RepresentationOutput annotations."""

    fit_hints = get_type_hints(ReferenceDiffusionBasicNormality.fit)
    infer_hints = get_type_hints(ReferenceDiffusionBasicNormality.infer)

    assert fit_hints["representations"] == Iterable[RepresentationOutput]
    assert infer_hints["representation"] is RepresentationOutput


def test_reference_diffusion_basic_fit_and_infer_accept_representation_output() -> None:
    """ReferenceDiffusionBasicNormality should emit diffusion artifacts for typed pixel outputs."""

    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id=f"sample-{index + 1:03d}",
        )
        for index in range(2)
    ]
    representations = [make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in samples]
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
    )

    model.fit(representations, samples)
    artifacts = model.infer(samples[0], representations[0])

    assert artifacts.has("predicted_noise")
    assert artifacts.has("target_noise")
    assert artifacts.has("reconstruction")
    assert artifacts.has("reference_projection")
    assert artifacts.has("conditional_alignment")
    assert "anomaly_map" not in artifacts.primary
    assert "image_score" not in artifacts.primary

    predicted_noise = artifacts.get_aux("predicted_noise")
    target_noise = artifacts.get_aux("target_noise")
    reference_projection = artifacts.get_primary("reference_projection")
    reconstruction = artifacts.get_primary("reconstruction")
    conditional_alignment = artifacts.get_aux("conditional_alignment")
    assert isinstance(predicted_noise, torch.Tensor)
    assert isinstance(target_noise, torch.Tensor)
    assert isinstance(reference_projection, torch.Tensor)
    assert isinstance(reconstruction, torch.Tensor)
    assert isinstance(conditional_alignment, torch.Tensor)
    assert predicted_noise.shape == target_noise.shape == (3, 16, 16)
    assert reconstruction.shape == (3, 16, 16)
    assert reference_projection.shape == (3, 16, 16)
    assert conditional_alignment.shape == (16, 16)
    assert artifacts.get_diag("time_embed_dim") == 32
    assert artifacts.get_diag("num_train_timesteps") == 32
    assert artifacts.representation == representations[0].to_artifact_dict()
    assert artifacts.representation["space"] == "pixel"
    assert artifacts.representation["sample_id"] == "sample-001"
    assert ReferenceDiffusionBasicNormality.accepted_spaces == frozenset({"pixel"})
    assert ReferenceDiffusionBasicNormality.accepted_tensor_ranks == frozenset({3})


def test_reference_diffusion_artifacts_flow_into_reconstruction_residual_evidence() -> None:
    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id="sample-001",
        )
    ]
    representations = [make_pixel_output(samples[0].image, sample_id=samples[0].sample_id or "sample")]
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
    )
    model.fit(representations, samples)

    artifacts = model.infer(samples[0], representations[0])

    from adrf.evidence.reconstruction_residual import ReconstructionResidualEvidence

    prediction = ReconstructionResidualEvidence(aggregator="mean").predict(samples[0], artifacts)

    assert prediction["anomaly_map"].shape == (16, 16)
    assert isinstance(prediction["image_score"], float)


def test_reference_diffusion_infer_is_deterministic_for_the_same_sample() -> None:
    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id="sample-001",
        )
    ]
    representations = [make_pixel_output(samples[0].image, sample_id=samples[0].sample_id or "sample")]
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
    )

    model.fit(representations, samples)
    first = model.infer(samples[0], representations[0])
    second = model.infer(samples[0], representations[0])

    assert torch.allclose(first.get_aux("predicted_noise"), second.get_aux("predicted_noise"))
    assert torch.allclose(first.get_aux("target_noise"), second.get_aux("target_noise"))
    assert torch.allclose(first.get_primary("reference_projection"), second.get_primary("reference_projection"))


def test_reference_diffusion_basic_accepts_pil_reference_fallback() -> None:
    """ReferenceDiffusionBasicNormality should keep working with raw PIL references."""

    torch.manual_seed(0)
    sample = Sample(
        image=torch.rand(3, 16, 16),
        reference=Image.new("RGB", (12, 10), color=(0, 255, 0)),
        sample_id="sample-pil",
    )
    representation = make_pixel_output(sample.image, sample_id="sample-pil")
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
    )

    model.fit([representation], [sample])
    artifacts = model.infer(sample, representation)

    assert artifacts.get_primary("reference_projection").shape == (3, 16, 16)
    assert artifacts.representation["sample_id"] == "sample-pil"


def test_reference_diffusion_treats_uint8_tensor_reference_like_pil_reference() -> None:
    image = torch.rand(3, 16, 16)
    pil_reference = Image.new("RGB", (16, 16), color=(255, 128, 0))
    tensor_reference = torch.tensor([255, 128, 0], dtype=torch.uint8).view(3, 1, 1).expand(3, 16, 16)
    representation = make_pixel_output(image, sample_id="sample-ref")
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
    )

    pil_tensor = model._prepare_reference_tensor(
        Sample(image=image, reference=pil_reference, sample_id="sample-ref"),
        representation,
    )
    uint8_tensor = model._prepare_reference_tensor(
        Sample(image=image, reference=tensor_reference, sample_id="sample-ref"),
        representation,
    )

    assert torch.allclose(pil_tensor, uint8_tensor)


def test_reference_diffusion_preserves_already_normalized_float_reference_tensor() -> None:
    image = torch.rand(3, 16, 16)
    normalized_reference = torch.full((3, 16, 16), 2.25, dtype=torch.float32)
    representation = make_pixel_output(image, sample_id="sample-ref")
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
    )

    prepared = model._prepare_reference_tensor(
        Sample(image=image, reference=normalized_reference, sample_id="sample-ref"),
        representation,
    )

    assert torch.allclose(prepared, normalized_reference)


def test_reference_diffusion_applies_sample_normalization_policy_to_raw_pil_reference() -> None:
    raw_sample = Sample(
        image=Image.new("RGB", (16, 16), color=(255, 255, 255)),
        reference=Image.new("RGB", (16, 16), color=(255, 128, 0)),
        sample_id="sample-ref",
    )
    transform = SampleTransform(
        image_size=(16, 16),
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    transformed = transform(raw_sample)
    representation = make_pixel_output(transformed.image, sample_id="sample-ref")
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
    )

    prepared = model._prepare_reference_tensor(
        Sample(
            image=transformed.image,
            reference=raw_sample.reference,
            sample_id="sample-ref",
            metadata=transformed.metadata,
        ),
        representation,
    )

    assert torch.allclose(prepared, transformed.reference)


def test_reference_diffusion_legacy_mapping_emits_normalized_representation_payload() -> None:
    """ReferenceDiffusionBasicNormality should normalize legacy mapping inputs in emitted artifacts."""

    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id=f"legacy-{index:03d}",
        )
        for index in range(2)
    ]
    typed_representations = [make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in samples]
    representations = [
        {
            "representation": representation.tensor,
            "space_type": "pixel",
            "spatial_shape": representation.spatial_shape,
            "feature_dim": representation.feature_dim,
            "sample_id": representation.sample_id,
            "requires_grad": representation.requires_grad,
            "device": representation.device,
            "dtype": representation.dtype,
            "provenance": representation.provenance,
        }
        for representation in typed_representations
    ]
    expected_representation = {
        "tensor": typed_representations[0].tensor,
        "space": typed_representations[0].space,
        "spatial_shape": typed_representations[0].spatial_shape,
        "feature_dim": typed_representations[0].feature_dim,
        "sample_id": typed_representations[0].sample_id,
        "requires_grad": typed_representations[0].requires_grad,
        "device": typed_representations[0].device,
        "dtype": typed_representations[0].dtype,
        "provenance": typed_representations[0].provenance.to_dict(),
    }
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=4,
        time_embed_dim=16,
        num_train_timesteps=8,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
    )

    model.fit(representations, samples)
    artifacts = model.infer(samples[0], representations[0])

    assert artifacts.representation == expected_representation
