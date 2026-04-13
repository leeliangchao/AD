"""Tests for the one-class protocol orchestration."""

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from adrf.data.datamodule import MVTecDataModule
from adrf.evaluation.evaluator import BasicADEvaluator
from adrf.evidence.feature_distance import FeatureDistanceEvidence
from adrf.evidence.reconstruction_residual import ReconstructionResidualEvidence
from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.feature_memory import FeatureMemoryNormality
from adrf.protocol.one_class import OneClassProtocol
from adrf.representation.feature import FeatureRepresentation
from adrf.representation.pixel import PixelRepresentation


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _write_mask_image(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=value).save(path)


def _build_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec"
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (0, 0, 0))
    _write_rgb_image(root / "bottle" / "train" / "good" / "001.png", (5, 5, 5))
    _write_rgb_image(root / "bottle" / "test" / "good" / "002.png", (0, 0, 0))
    _write_rgb_image(root / "bottle" / "test" / "broken_large" / "003.png", (255, 255, 255))
    _write_mask_image(root / "bottle" / "ground_truth" / "broken_large" / "003_mask.png", 255)
    return root


def test_one_class_protocol_runs_feature_baseline(tmp_path: Path) -> None:
    """OneClassProtocol should orchestrate the feature-memory baseline."""

    torch.manual_seed(0)
    root = _build_fixture_root(tmp_path)
    runner = SimpleNamespace(
        datamodule=MVTecDataModule(
            root=root,
            category="bottle",
            image_size=(64, 64),
            batch_size=2,
            num_workers=0,
            normalize=False,
        ),
        representation=FeatureRepresentation(pretrained=False, freeze=True),
        normality=FeatureMemoryNormality(),
        evidence=FeatureDistanceEvidence(),
        evaluator=BasicADEvaluator(),
    )
    protocol = OneClassProtocol()

    train_summary = protocol.train_epoch(runner)
    metrics = protocol.evaluate(runner)

    assert train_summary["num_train_samples"] == 2
    assert set(metrics) == {"image_auroc", "pixel_auroc", "pixel_aupr"}


def test_one_class_protocol_runs_reconstruction_baseline(tmp_path: Path) -> None:
    """OneClassProtocol should orchestrate the reconstruction baseline."""

    torch.manual_seed(0)
    root = _build_fixture_root(tmp_path)
    runner = SimpleNamespace(
        datamodule=MVTecDataModule(
            root=root,
            category="bottle",
            image_size=(32, 32),
            batch_size=2,
            num_workers=0,
            normalize=False,
        ),
        representation=PixelRepresentation(),
        normality=AutoEncoderNormality(
            input_channels=3,
            hidden_channels=4,
            latent_channels=8,
            epochs=1,
            batch_size=2,
        ),
        evidence=ReconstructionResidualEvidence(),
        evaluator=BasicADEvaluator(),
    )
    protocol = OneClassProtocol()

    train_summary = protocol.train_epoch(runner)
    metrics = protocol.evaluate(runner)

    assert train_summary["num_train_samples"] == 2
    assert set(metrics) == {"image_auroc", "pixel_auroc", "pixel_aupr"}

