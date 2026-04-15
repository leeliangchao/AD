"""Demonstrate the data-layer contract against the repository's MVTec data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from adrf.core.sample import Sample
from adrf.data.datamodule import MVTecDataModule
from adrf.data.transforms import build_sample_transform


def project_root() -> Path:
    """Return the repository root that contains this example."""
    """ 返回包含此示例的存储库根目录。"""

    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    """Return the repository-local MVTec root used by the example."""
    """返回示例使用的存储库本地 MVTec 根目录。"""

    return project_root() / "data" / "mvtec"


def available_categories(root: Path) -> list[str]:
    """Return available MVTec categories under the provided root."""
    """返回提供的根目录下可用的 MVTec 类别。"""

    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _sample_snapshot(sample: Sample) -> dict[str, Any]:
    """Build a compact summary of one Sample payload."""
    """ 构建一个 Sample 负载的紧凑摘要。"""

    image_shape = list(sample.image.shape) if hasattr(sample.image, "shape") else None
    reference_shape = list(sample.reference.shape) if hasattr(sample.reference, "shape") else None
    snapshot = {
        "sample_id": sample.sample_id,
        "label": sample.label,
        "split": sample.metadata.get("split"),
        "image_name": Path(sample.metadata["image_path"]).name,
        "reference_name": Path(sample.metadata["reference_path"]).name,
        "image_type": type(sample.image).__name__,
        "image_shape": image_shape,
        "reference_type": type(sample.reference).__name__ if sample.reference is not None else None,
        "reference_shape": reference_shape,
        "has_reference": sample.reference is not None,
    }
    if "observed_split_before_transform" in sample.metadata:
        snapshot["observed_split_before_transform"] = sample.metadata["observed_split_before_transform"]
    return snapshot


class TraceSplitTransform:
    """Wrap the default transform and record which split it observed before execution."""
    """ 包装默认变换并记录它在执行前观察到的拆分。"""

    def __init__(self) -> None:
        self.base_transform = build_sample_transform(image_size=(32, 32), normalize=False)
        #   记录每个拆分被观察到的次数，以验证数据加载器是否正确应用了变换。
        self.seen_splits: dict[str, list[str]] = {}

    def __call__(self, sample: Sample) -> Sample:
        split = str(sample.metadata["split"])
        self.seen_splits.setdefault(split, []).append(split)
        transformed = self.base_transform(sample)
        payload = transformed.to_dict()
        payload["metadata"]["observed_split_before_transform"] = split
        return Sample(**payload)


def collect_demo_payload(
    *,
    root: str | Path,
    category: str = "bottle",
    reference_index: int = 1,
    val_split: float = 0.2,
    calibration_split: float = 0.2,
    split_seed: int = 7,
) -> dict[str, Any]:
    """Collect a compact demonstration payload for the current data-layer contract."""

    root_path = Path(root).resolve()
    basic_dm = MVTecDataModule(
        root=root_path,
        category=category,
        image_size=(32, 32),
        batch_size=1,
        num_workers=0,
        normalize=False,
        reference_index=reference_index,
        val_split=val_split,
        calibration_split=calibration_split,
        split_seed=split_seed,
    )
    basic_dm.setup()

    basic_samples = {
        "train": _sample_snapshot(next(iter(basic_dm.train_dataloader()))[0]),
        "val": _sample_snapshot(next(iter(basic_dm.val_dataloader()))[0]),
        "calibration": _sample_snapshot(next(iter(basic_dm.calibration_dataloader()))[0]),
        "test": _sample_snapshot(next(iter(basic_dm.test_dataloader()))[0]),
    }

    tracing_transform = TraceSplitTransform()
    transform_dm = MVTecDataModule(
        root=root_path,
        category=category,
        batch_size=1,
        num_workers=0,
        normalize=False,
        reference_index=reference_index,
        val_split=val_split,
        calibration_split=calibration_split,
        split_seed=split_seed,
        transform=tracing_transform,
    )
    transform_dm.setup()
    traced_val = next(iter(transform_dm.val_dataloader()))[0]
    traced_calibration = next(iter(transform_dm.calibration_dataloader()))[0]

    return {
        "root": str(root_path),
        "category": category,
        "available_categories": available_categories(root_path),
        "split_lengths": {
            "train": len(basic_dm.train_dataset),
            "val": len(basic_dm.val_dataset),
            "calibration": len(basic_dm.calibration_dataset),
            "test": len(basic_dm.test_dataset),
        },
        "basic_demo": {
            "config": {
                "reference_index": reference_index,
                "val_split": val_split,
                "calibration_split": calibration_split,
                "split_seed": split_seed,
            },
            "samples": basic_samples,
        },
        "transform_demo": {
            "observed_splits": {
                "val": tracing_transform.seen_splits.get("val", []),
                "calibration": tracing_transform.seen_splits.get("calibration", []),
            },
            "samples": {
                "val": _sample_snapshot(traced_val),
                "calibration": _sample_snapshot(traced_calibration),
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the data usage demo."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_data_root(), help="Path to the MVTec root.")
    parser.add_argument("--category", default="bottle", help="MVTec category to inspect.")
    parser.add_argument("--reference-index", type=int, default=1, help="Fixed reference index within retained train.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Held-out validation ratio from train/good.")
    parser.add_argument(
        "--calibration-split",
        type=float,
        default=0.2,
        help="Held-out calibration ratio from train/good.",
    )
    parser.add_argument("--split-seed", type=int, default=7, help="Deterministic split seed.")
    return parser


def main() -> None:
    """Run the data-layer demo and print a JSON payload."""

    args = build_parser().parse_args()
    payload = collect_demo_payload(
        root=args.root,
        category=args.category,
        reference_index=args.reference_index,
        val_split=args.val_split,
        calibration_split=args.calibration_split,
        split_seed=args.split_seed,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
