from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from adrf.data.datamodule import MVTecDataModule
from adrf.normality.base import BaseNormalityModel
from adrf.protocol.one_class import OneClassProtocol
from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationBatch, RepresentationProvenance


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _build_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec"
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (0, 0, 0))
    _write_rgb_image(root / "bottle" / "train" / "good" / "001.png", (16, 16, 16))
    _write_rgb_image(root / "bottle" / "test" / "good" / "002.png", (0, 0, 0))
    return root


class _TrainableToyRepresentation(BaseRepresentation):
    space = "feature"
    trainable = True

    def __init__(self) -> None:
        super().__init__(input_image_size=(32, 32), input_normalize=False)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.encode_batch_calls = 0

    def encode_batch(self, samples: Sequence) -> RepresentationBatch:
        self.encode_batch_calls += 1
        return super().encode_batch(samples)

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
            config_fingerprint="toy-trainable-feature",
        )


class _JointToyNormality(nn.Module, BaseNormalityModel):
    fit_mode = "joint"
    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = False

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.optimizer: torch.optim.Optimizer | None = None

    def configure_joint_training(self, representation_model: nn.Module) -> None:
        self.optimizer = torch.optim.SGD(
            list(self.parameters())
            + [parameter for parameter in representation_model.parameters() if parameter.requires_grad],
            lr=1e-2,
        )

    def fit_batch(self, representations: RepresentationBatch, samples) -> dict[str, float]:
        del samples
        self.validate_representation_batch(representations)
        assert self.optimizer is not None
        loss = (representations.tensor.mean() * self.scale).square()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().item())}

    def fit(self, representations, samples=None) -> None:
        raise AssertionError("joint mode should not call fit().")

    def infer(self, sample, representation):
        raise AssertionError("test does not exercise inference.")


def test_one_class_protocol_joint_fit_updates_trainable_representation_parameter(tmp_path: Path) -> None:
    torch.manual_seed(0)
    root = _build_fixture_root(tmp_path)
    representation = _TrainableToyRepresentation()
    normality = _JointToyNormality()
    runner = SimpleNamespace(
        datamodule=MVTecDataModule(
            root=root,
            category="bottle",
            image_size=(32, 32),
            batch_size=2,
            num_workers=0,
            normalize=False,
        ),
        representation=representation,
        normality=normality,
    )
    protocol = OneClassProtocol()
    initial_scale = representation.scale.detach().clone()

    train_summary = protocol.train_epoch(runner)

    assert train_summary["num_train_samples"] == 2
    assert train_summary["num_train_batches"] == 1
    assert representation.encode_batch_calls == 1
    assert not torch.allclose(representation.scale.detach(), initial_scale)
