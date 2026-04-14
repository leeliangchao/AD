"""Tests for the minimal MVTec AD dataset adapter."""

from pathlib import Path

import pytest
from PIL import Image

from adrf.core.sample import Sample
from adrf.data.datasets.mvtec import MVTecSingleClassDataset


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=color).save(path)


def _write_mask_image(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (24, 24), color=value).save(path)


def test_mvtec_single_class_dataset_returns_samples_for_train_and_test(tmp_path: Path) -> None:
    """The dataset should read one category and emit unified Sample objects."""

    root = tmp_path / "mvtec"
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (20, 30, 40))
    _write_rgb_image(root / "bottle" / "test" / "good" / "001.png", (50, 60, 70))
    _write_rgb_image(root / "bottle" / "test" / "broken_large" / "002.png", (80, 90, 100))
    _write_mask_image(root / "bottle" / "ground_truth" / "broken_large" / "002_mask.png")

    train_dataset = MVTecSingleClassDataset(root=root, category="bottle", split="train")
    test_dataset = MVTecSingleClassDataset(root=root, category="bottle", split="test")

    assert len(train_dataset) == 1
    assert len(test_dataset) == 2

    train_sample = train_dataset[0]
    test_samples = [test_dataset[index] for index in range(len(test_dataset))]
    anomaly_sample = next(sample for sample in test_samples if sample.label == 1)
    good_sample = next(sample for sample in test_samples if sample.label == 0)

    assert isinstance(train_sample, Sample)
    assert train_sample.label == 0
    assert train_sample.mask is None
    assert train_sample.category == "bottle"
    assert train_sample.metadata["split"] == "train"
    assert train_sample.sample_id == "train/good/000.png"

    assert good_sample.mask is None
    assert anomaly_sample.mask is not None
    assert anomaly_sample.metadata["split"] == "test"
    assert anomaly_sample.metadata["defect_type"] == "broken_large"
    assert anomaly_sample.sample_id == "test/broken_large/002.png"


def test_mvtec_single_class_dataset_caches_reference_image_per_dataset_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dataset should avoid reopening the fixed reference for every sample."""

    root = tmp_path / "mvtec"
    reference_path = root / "bottle" / "train" / "good" / "000.png"
    _write_rgb_image(reference_path, (20, 30, 40))
    _write_rgb_image(root / "bottle" / "train" / "good" / "001.png", (50, 60, 70))

    open_calls: list[str] = []
    original_open = Image.open

    def tracking_open(path: str | Path, *args: object, **kwargs: object):
        open_calls.append(str(Path(path)))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("adrf.data.datasets.mvtec.Image.open", tracking_open)

    dataset = MVTecSingleClassDataset(root=root, category="bottle", split="train", reference_index=1)
    dataset[0]
    dataset[0]

    reference_opens = [path for path in open_calls if path == str(root / "bottle" / "train" / "good" / "001.png")]
    assert len(reference_opens) == 1
