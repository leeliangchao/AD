"""Tests for the minimal MVTec datamodule."""

from pathlib import Path

import torch
from PIL import Image

from adrf.core.sample import Sample
from adrf.data.datamodule import MVTecDataModule


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def _write_mask_image(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (20, 20), color=value).save(path)


def test_datamodule_builds_train_and_test_dataloaders(tmp_path: Path) -> None:
    """The datamodule should yield batches of transformed Sample objects."""

    root = tmp_path / "mvtec"
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (20, 30, 40))
    _write_rgb_image(root / "bottle" / "train" / "good" / "001.png", (30, 40, 50))
    _write_rgb_image(root / "bottle" / "test" / "good" / "002.png", (40, 50, 60))
    _write_rgb_image(root / "bottle" / "test" / "broken_large" / "003.png", (50, 60, 70))
    _write_mask_image(root / "bottle" / "ground_truth" / "broken_large" / "003_mask.png")

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        image_size=(16, 16),
        batch_size=2,
        num_workers=0,
        normalize=False,
    )

    train_batch = next(iter(datamodule.train_dataloader()))
    test_batch = next(iter(datamodule.test_dataloader()))

    assert isinstance(train_batch, list)
    assert isinstance(test_batch, list)
    assert all(isinstance(sample, Sample) for sample in train_batch)
    assert all(isinstance(sample.image, torch.Tensor) for sample in train_batch)
    assert train_batch[0].image.shape == (3, 16, 16)
    assert all(isinstance(sample.reference, torch.Tensor) for sample in train_batch)
    assert train_batch[0].reference.shape == (3, 16, 16)
    assert len(test_batch) == 2
