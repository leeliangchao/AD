"""Tests for the minimal MVTec datamodule."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from adrf.core.sample import Sample
from adrf.data.datamodule import MVTecDataModule
from adrf.data.datasets.mvtec import MVTecSingleClassDataset


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def _write_mask_image(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (20, 20), color=value).save(path)


def _write_mvtec_split_fixture(root: Path) -> None:
    """Create the synthetic MVTec layout used by the split regression tests."""

    for index, color in enumerate(
        [(20, 30, 40), (30, 40, 50), (40, 50, 60), (50, 60, 70), (60, 70, 80)],
    ):
        _write_rgb_image(root / "bottle" / "train" / "good" / f"{index:03d}.png", color)
    _write_rgb_image(root / "bottle" / "test" / "good" / "010.png", (70, 80, 90))


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


def test_datamodule_supports_validation_and_calibration_splits(tmp_path: Path) -> None:
    """The datamodule should deterministically carve val/calibration subsets from train/good."""

    root = tmp_path / "mvtec"
    _write_mvtec_split_fixture(root)

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        image_size=(16, 16),
        batch_size=2,
        num_workers=0,
        normalize=False,
        val_split=0.2,
        calibration_split=0.2,
        split_seed=7,
    )
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.calibration_dataset is not None
    assert len(datamodule.train_dataset) == 3
    assert len(datamodule.val_dataset) == 1
    assert len(datamodule.calibration_dataset) == 1

    val_batch = next(iter(datamodule.val_dataloader()))
    calibration_batch = next(iter(datamodule.calibration_dataloader()))
    assert all(isinstance(sample.image, torch.Tensor) for sample in val_batch)
    assert all(isinstance(sample.reference, torch.Tensor) for sample in calibration_batch)


def test_datamodule_uses_reference_from_retained_train_subset(tmp_path: Path) -> None:
    """Held-out splits should not be able to become the fixed reference image."""

    root = tmp_path / "mvtec"
    _write_mvtec_split_fixture(root)

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        batch_size=2,
        num_workers=0,
        normalize=False,
        val_split=0.2,
        calibration_split=0.2,
        split_seed=7,
        reference_index=0,
    )
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.calibration_dataset is not None

    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    calibration_batch = next(iter(datamodule.calibration_dataloader()))
    test_batch = next(iter(datamodule.test_dataloader()))
    reference_path = Path(train_batch[0].metadata["reference_path"])
    retained_train_paths = {
        Path(datamodule.train_dataset[index].metadata["image_path"])
        for index in range(len(datamodule.train_dataset))
    }
    held_out_paths = {
        Path(sample.metadata["image_path"])
        for sample in val_batch
    }
    held_out_paths.update(Path(sample.metadata["image_path"]) for sample in calibration_batch)

    assert reference_path in retained_train_paths
    assert reference_path not in held_out_paths

    assert all(
        Path(sample.metadata["reference_path"]) == reference_path
        for sample in calibration_batch
    )
    assert all(
        Path(sample.metadata["reference_path"]) == reference_path
        for sample in test_batch
    )


def test_datamodule_uses_nonzero_reference_index_with_retained_train_subset(tmp_path: Path) -> None:
    """Held-out splits should preserve non-zero reference indexing within retained train order."""

    root = tmp_path / "mvtec"
    _write_mvtec_split_fixture(root)

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        batch_size=2,
        num_workers=0,
        normalize=False,
        val_split=0.2,
        calibration_split=0.2,
        split_seed=7,
        reference_index=1,
    )
    datamodule.setup()

    assert datamodule.train_dataset is not None

    retained_train_names = [
        Path(datamodule.train_dataset[index].metadata["image_path"]).name
        for index in range(len(datamodule.train_dataset))
    ]
    assert len(retained_train_names) == 3

    expected_reference_name = retained_train_names[1]
    reference_names = {
        Path(datamodule.train_dataset[index].metadata["reference_path"]).name
        for index in range(len(datamodule.train_dataset))
    }

    assert reference_names == {expected_reference_name}


def test_datamodule_no_holdout_preserves_sorted_train_order_and_reference_semantics(
    tmp_path: Path,
) -> None:
    """Without holdouts, train order and reference selection should stay aligned to sorted train/good."""

    root = tmp_path / "mvtec"
    _write_mvtec_split_fixture(root)

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        batch_size=2,
        num_workers=0,
        normalize=False,
        reference_index=0,
        val_split=0.0,
        calibration_split=0.0,
    )
    datamodule.setup()

    assert datamodule.train_dataset is not None

    train_image_names = [
        Path(datamodule.train_dataset[index].metadata["image_path"]).name
        for index in range(len(datamodule.train_dataset))
    ]
    train_reference_names = {
        Path(datamodule.train_dataset[index].metadata["reference_path"]).name
        for index in range(len(datamodule.train_dataset))
    }

    assert train_image_names == ["000.png", "001.png", "002.png", "003.png", "004.png"]
    assert train_reference_names == {"000.png"}


def test_datamodule_relabels_held_out_split_metadata(tmp_path: Path) -> None:
    """Held-out dataloaders should report their own split metadata."""

    root = tmp_path / "mvtec"
    _write_mvtec_split_fixture(root)

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        batch_size=2,
        num_workers=0,
        normalize=False,
        val_split=0.2,
        calibration_split=0.2,
        split_seed=7,
    )
    datamodule.setup()

    val_batch = next(iter(datamodule.val_dataloader()))
    calibration_batch = next(iter(datamodule.calibration_dataloader()))

    assert {sample.metadata["split"] for sample in val_batch} == {"val"}
    assert {sample.metadata["split"] for sample in calibration_batch} == {"calibration"}


def test_datamodule_relabels_held_out_split_metadata_before_custom_transform(
    tmp_path: Path,
) -> None:
    """Held-out custom transforms should observe the relabeled split before running."""

    root = tmp_path / "mvtec"
    _write_mvtec_split_fixture(root)
    seen_splits: list[str] = []

    def record_split(sample: Sample) -> Sample:
        seen_splits.append(sample.metadata["split"])
        return sample

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        batch_size=1,
        num_workers=0,
        normalize=False,
        val_split=0.2,
        calibration_split=0.2,
        split_seed=7,
        transform=record_split,
    )
    datamodule.setup()

    val_batch = next(iter(datamodule.val_dataloader()))
    assert seen_splits == ["val"]
    assert [sample.metadata["split"] for sample in val_batch] == ["val"]

    seen_splits.clear()
    calibration_batch = next(iter(datamodule.calibration_dataloader()))
    assert seen_splits == ["calibration"]
    assert [sample.metadata["split"] for sample in calibration_batch] == ["calibration"]


def test_datamodule_rejects_overcommitted_train_splits(tmp_path: Path) -> None:
    """Validation and calibration splits must leave at least one training example."""

    root = tmp_path / "mvtec"
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (20, 30, 40))
    _write_rgb_image(root / "bottle" / "train" / "good" / "001.png", (30, 40, 50))
    _write_rgb_image(root / "bottle" / "test" / "good" / "002.png", (40, 50, 60))

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        batch_size=2,
        num_workers=0,
        normalize=False,
        val_split=0.5,
        calibration_split=0.5,
    )

    with pytest.raises(ValueError, match="leave at least one training sample"):
        datamodule.setup()


def test_datamodule_validates_reference_index_against_retained_train_subset(tmp_path: Path) -> None:
    """reference_index should be checked against the retained training subset only."""

    root = tmp_path / "mvtec"
    _write_mvtec_split_fixture(root)

    datamodule = MVTecDataModule(
        root=root,
        category="bottle",
        batch_size=2,
        num_workers=0,
        normalize=False,
        val_split=0.2,
        calibration_split=0.2,
        split_seed=7,
        reference_index=3,
    )

    with pytest.raises(IndexError, match="reference_index=3 is out of range for 3"):
        datamodule.setup()


def test_dataset_rejects_missing_explicit_reference_path_eagerly(tmp_path: Path) -> None:
    """Explicit reference overrides should fail at dataset construction time."""

    root = tmp_path / "mvtec"
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (20, 30, 40))

    missing_reference_path = root / "bottle" / "train" / "good" / "missing.png"

    with pytest.raises(FileNotFoundError, match=str(missing_reference_path)):
        MVTecSingleClassDataset(
            root=root,
            category="bottle",
            split="train",
            reference_path=missing_reference_path,
        )
