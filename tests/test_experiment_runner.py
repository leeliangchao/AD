"""Tests for the experiment runner."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from adrf.normality.base import BaseNormalityModel
from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationProvenance
from adrf.runner.experiment_runner import ExperimentRunner, build_default_registry


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _write_mask_image(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=value).save(path)


def _write_fixture_dataset(root: Path) -> None:
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (0, 0, 0))
    _write_rgb_image(root / "bottle" / "train" / "good" / "001.png", (10, 10, 10))
    _write_rgb_image(root / "bottle" / "test" / "good" / "002.png", (0, 0, 0))
    _write_rgb_image(root / "bottle" / "test" / "broken_large" / "003.png", (255, 255, 255))
    _write_mask_image(root / "bottle" / "ground_truth" / "broken_large" / "003_mask.png", 255)


class _TrainableFeatureRepresentation(BaseRepresentation):
    space = "feature"
    trainable = True

    def __init__(self) -> None:
        super().__init__(input_image_size=(32, 32), input_normalize=False)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _encode_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch[:, :1, :, :] * self.scale

    def describe(self) -> RepresentationProvenance:
        return RepresentationProvenance(
            representation_name="feature",
            backbone_name="toy",
            weights_source=None,
            feature_layer="scale",
            pooling=None,
            trainable=True,
            frozen_submodules=(),
            input_image_size=self.input_image_size,
            input_normalize=self.input_normalize,
            normalize_mean=None,
            normalize_std=None,
            code_version="tests",
            config_fingerprint="runner-toy-feature",
        )


class _OfflineToyNormality(nn.Module, BaseNormalityModel):
    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = True

    def fit(self, representations, samples=None) -> None:
        del representations, samples

    def infer(self, sample, representation):
        del sample, representation
        raise AssertionError("test does not exercise inference.")


class _BatchNormFeatureRepresentation(BaseRepresentation):
    space = "feature"
    trainable = True

    def __init__(self) -> None:
        super().__init__(input_image_size=(32, 32), input_normalize=False)
        self.batch_norm = nn.BatchNorm2d(1)

    def _encode_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(batch[:, :1, :, :])

    def describe(self) -> RepresentationProvenance:
        return RepresentationProvenance(
            representation_name="feature",
            backbone_name="toy-bn",
            weights_source=None,
            feature_layer="batch_norm",
            pooling=None,
            trainable=True,
            frozen_submodules=(),
            input_image_size=self.input_image_size,
            input_normalize=self.input_normalize,
            normalize_mean=None,
            normalize_std=None,
            code_version="tests",
            config_fingerprint="runner-toy-batchnorm-feature",
        )


class _CompatibleToyNormality(nn.Module, BaseNormalityModel):
    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = False

    def fit(self, representations, samples=None) -> None:
        del representations, samples

    def infer(self, sample, representation):
        del sample, representation
        raise AssertionError("test does not exercise inference.")


def _write_runner_config(config_path: Path, dataset_root: Path, *, representation_name: str, normality_name: str) -> None:
    config_path.write_text(
        "\n".join(
            [
                "datamodule:",
                "  name: mvtec_single_class",
                "  params:",
                f"    root: {dataset_root.as_posix()}",
                "    category: bottle",
                "    image_size: [32, 32]",
                "    batch_size: 2",
                "    num_workers: 0",
                "    normalize: false",
                "representation:",
                f"  name: {representation_name}",
                "  params: {}",
                "normality:",
                f"  name: {normality_name}",
                "  params: {}",
                "evidence:",
                "  name: reconstruction_residual",
                "  params:",
                "    aggregator: mean",
                "evaluator:",
                "  name: basic_ad",
                "  params: {}",
                "protocol:",
                "  name: one_class",
                "  params: {}",
            ]
        ),
        encoding="utf-8",
    )


def test_experiment_runner_builds_components_and_runs(tmp_path: Path) -> None:
    """ExperimentRunner should assemble the configured pipeline and return results."""

    dataset_root = tmp_path / "mvtec"
    _write_fixture_dataset(dataset_root)
    config_path = tmp_path / "recon.yaml"
    config_path.write_text(
        "\n".join(
            [
                "datamodule:",
                "  name: mvtec_single_class",
                "  params:",
                f"    root: {dataset_root.as_posix()}",
                "    category: bottle",
                "    image_size: [32, 32]",
                "    batch_size: 2",
                "    num_workers: 0",
                "    normalize: false",
                "representation:",
                "  name: pixel",
                "  params: {}",
                "normality:",
                "  name: autoencoder",
                "  params:",
                "    input_channels: 3",
                "    hidden_channels: 4",
                "    latent_channels: 8",
                "    epochs: 1",
                "    batch_size: 2",
                "evidence:",
                "  name: reconstruction_residual",
                "  params:",
                "    aggregator: mean",
                "evaluator:",
                "  name: basic_ad",
                "  params: {}",
                "protocol:",
                "  name: one_class",
                "  params: {}",
            ]
        ),
        encoding="utf-8",
    )

    runner = ExperimentRunner(config_path)
    results = runner.run()

    assert runner.datamodule is not None
    assert runner.protocol is not None
    assert set(results) == {"train", "evaluation"}
    assert results["train"]["num_train_samples"] == 2
    assert set(results["evaluation"]) == {"image_auroc", "pixel_auroc", "pixel_aupr"}


def test_experiment_runner_rejects_trainable_offline_representation_contract(tmp_path: Path) -> None:
    """ExperimentRunner should reject trainable representations paired with detached offline normality."""

    dataset_root = tmp_path / "mvtec"
    _write_fixture_dataset(dataset_root)
    config_path = tmp_path / "invalid_contract.yaml"
    _write_runner_config(
        config_path,
        dataset_root,
        representation_name="trainable_feature",
        normality_name="offline_toy",
    )
    registry = build_default_registry()
    registry.register("representation", "trainable_feature", _TrainableFeatureRepresentation)
    registry.register("normality", "offline_toy", _OfflineToyNormality)

    runner = ExperimentRunner(config_path, registry=registry)

    with pytest.raises(ValueError, match="trainable representation"):
        runner.setup()


def test_experiment_runner_setup_validation_does_not_mutate_trainable_representation_state(tmp_path: Path) -> None:
    """ExperimentRunner setup validation should preserve trainable representation state."""

    dataset_root = tmp_path / "mvtec"
    _write_fixture_dataset(dataset_root)
    config_path = tmp_path / "stateful_contract.yaml"
    _write_runner_config(
        config_path,
        dataset_root,
        representation_name="batchnorm_feature",
        normality_name="compatible_toy",
    )
    registry = build_default_registry()
    registry.register("representation", "batchnorm_feature", _BatchNormFeatureRepresentation)
    registry.register("normality", "compatible_toy", _CompatibleToyNormality)

    runner = ExperimentRunner(config_path, registry=registry)
    representation = _BatchNormFeatureRepresentation()
    runner.representation = representation
    runner.normality = _CompatibleToyNormality()
    before_batches = int(representation.batch_norm.num_batches_tracked.item())

    runner._validate_representation_normality_contract()

    assert representation.training is True
    assert representation.batch_norm.training is True
    assert int(representation.batch_norm.num_batches_tracked.item()) == before_batches
