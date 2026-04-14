"""Tests for the data-layer usage demo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image


def _load_data_usage_demo_module():
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "examples" / "data_usage_demo.py"
    spec = importlib.util.spec_from_file_location("data_usage_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), color=color).save(path)


def _write_mvtec_split_fixture(root: Path) -> None:
    """Create the synthetic MVTec layout used by the demo test."""

    for index, color in enumerate(
        [(20, 30, 40), (30, 40, 50), (40, 50, 60), (50, 60, 70), (60, 70, 80)],
    ):
        _write_rgb_image(root / "bottle" / "train" / "good" / f"{index:03d}.png", color)
    _write_rgb_image(root / "bottle" / "test" / "good" / "010.png", (70, 80, 90))


def test_data_usage_demo_collects_split_and_transform_examples(tmp_path: Path) -> None:
    """The demo should expose both basic split usage and transform-time split semantics."""

    root = tmp_path / "mvtec"
    _write_mvtec_split_fixture(root)

    module = _load_data_usage_demo_module()
    payload = module.collect_demo_payload(
        root=root,
        category="bottle",
        reference_index=1,
        val_split=0.2,
        calibration_split=0.2,
        split_seed=7,
    )

    assert payload["category"] == "bottle"
    assert payload["split_lengths"] == {"train": 3, "val": 1, "calibration": 1, "test": 1}
    assert payload["basic_demo"]["samples"]["train"]["split"] == "train"
    assert payload["basic_demo"]["samples"]["val"]["split"] == "val"
    assert payload["basic_demo"]["samples"]["calibration"]["split"] == "calibration"
    assert payload["basic_demo"]["samples"]["test"]["split"] == "test"
    assert payload["basic_demo"]["samples"]["val"]["reference_name"] == "002.png"
    assert payload["transform_demo"]["observed_splits"] == {"val": ["val"], "calibration": ["calibration"]}
    assert payload["transform_demo"]["samples"]["val"]["observed_split_before_transform"] == "val"
    assert payload["transform_demo"]["samples"]["calibration"]["observed_split_before_transform"] == "calibration"
