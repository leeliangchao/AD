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


class _RecordingOfflineNormality:
    def __init__(self) -> None:
        self.fit_calls: list[tuple[int, int]] = []

    def fit(self, representations, samples=None) -> None:
        representation_list = list(representations)
        sample_list = [] if samples is None else list(samples)
        self.fit_calls.append((len(representation_list), len(sample_list)))

    def infer(self, sample, representation):
        del sample, representation
        raise AssertionError("test does not exercise inference.")


class _BatchOnlyFeatureRepresentation(FeatureRepresentation):
    def __init__(self) -> None:
        super().__init__(pretrained=False, freeze=True, input_image_size=(64, 64), input_normalize=False)
        self.encode_batch_calls: list[tuple[str | None, ...]] = []

    def encode_batch(self, samples):
        self.encode_batch_calls.append(tuple(sample.sample_id for sample in samples))
        return super().encode_batch(samples)

    def encode_sample(self, sample):
        raise AssertionError(f"Protocol should use encode_batch(), not encode_sample() for {sample.sample_id}.")


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
    representation = _BatchOnlyFeatureRepresentation()
    runner = SimpleNamespace(
        datamodule=MVTecDataModule(
            root=root,
            category="bottle",
            image_size=(64, 64),
            batch_size=2,
            num_workers=0,
            normalize=False,
        ),
        representation=representation,
        normality=FeatureMemoryNormality(),
        evidence=FeatureDistanceEvidence(),
        evaluator=BasicADEvaluator(),
    )
    protocol = OneClassProtocol()

    train_summary = protocol.train_epoch(runner)
    metrics = protocol.evaluate(runner)

    assert train_summary["num_train_samples"] == 2
    assert train_summary["num_train_batches"] == 1
    assert len(representation.encode_batch_calls) == 2
    assert all(len(sample_ids) == 2 for sample_ids in representation.encode_batch_calls)
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


def test_one_class_protocol_offline_fit_aggregates_distributed_train_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _build_fixture_root(tmp_path)
    normality = _RecordingOfflineNormality()
    runner = SimpleNamespace(
        datamodule=MVTecDataModule(
            root=root,
            category="bottle",
            image_size=(64, 64),
            batch_size=2,
            num_workers=0,
            normalize=False,
        ),
        representation=_BatchOnlyFeatureRepresentation(),
        normality=normality,
        distributed_context=SimpleNamespace(enabled=True, world_size=2),
        distributed_training_enabled=False,
    )
    protocol = OneClassProtocol()

    def _fake_all_gather(payload, context):
        del context
        if "samples" in payload:
            return [
                payload,
                {
                    "samples": [object(), object(), object()],
                    "representations": [object(), object(), object()],
                },
            ]
        return [
            payload,
            {"num_train_batches": 2, "num_train_samples": 3},
        ]

    monkeypatch.setattr("adrf.protocol.one_class.all_gather_objects", _fake_all_gather)

    train_summary = protocol.train_epoch(runner)

    assert normality.fit_calls == [(5, 5)]
    assert train_summary == {
        "num_train_batches": 3,
        "num_train_samples": 5,
    }
