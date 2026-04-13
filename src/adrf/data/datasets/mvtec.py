"""Minimal MVTec AD single-category dataset adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from PIL import Image
from torch.utils.data import Dataset

from adrf.core.sample import Sample

Split = Literal["train", "test"]


@dataclass(slots=True)
class _SampleRecord:
    """Internal index entry for one MVTec sample."""

    image_path: Path
    label: int
    split: Split
    defect_type: str
    mask_path: Path | None


class MVTecSingleClassDataset(Dataset[Sample]):
    """Load one MVTec AD category and emit unified Sample instances."""

    def __init__(
        self,
        root: str | Path,
        category: str = "bottle",
        split: Split = "train",
        reference_index: int = 0,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.reference_index = reference_index
        self.transform = transform
        self.category_root = self.root / self.category

        if not self.category_root.exists():
            raise FileNotFoundError(f"MVTec category path does not exist: {self.category_root}")

        self.reference_path = self._resolve_reference_path()
        self._records = self._index_records()

    def __len__(self) -> int:
        """Return the number of indexed samples."""

        return len(self._records)

    def __getitem__(self, index: int) -> Sample:
        """Load one sample and apply the optional sample transform."""

        record = self._records[index]
        image = Image.open(record.image_path).convert("RGB")
        reference = Image.open(self.reference_path).convert("RGB")
        mask = Image.open(record.mask_path).convert("L") if record.mask_path is not None else None
        sample = Sample(
            image=image,
            label=record.label,
            mask=mask,
            category=self.category,
            sample_id=str(record.image_path.relative_to(self.category_root).as_posix()),
            reference=reference,
            metadata={
                "split": record.split,
                "defect_type": record.defect_type,
                "image_path": str(record.image_path),
                "reference_path": str(self.reference_path),
                "mask_path": str(record.mask_path) if record.mask_path is not None else None,
            },
        )
        return self.transform(sample) if self.transform is not None else sample

    def _index_records(self) -> list[_SampleRecord]:
        """Collect sample records for the configured split."""

        if self.split == "train":
            image_paths = self._sorted_image_paths(self.category_root / "train" / "good")
            return [
                _SampleRecord(
                    image_path=image_path,
                    label=0,
                    split="train",
                    defect_type="good",
                    mask_path=None,
                )
                for image_path in image_paths
            ]

        records: list[_SampleRecord] = []
        test_root = self.category_root / "test"
        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for image_path in self._sorted_image_paths(defect_dir):
                mask_path: Path | None = None
                if label == 1:
                    mask_path = self._resolve_mask_path(defect_type, image_path)
                records.append(
                    _SampleRecord(
                        image_path=image_path,
                        label=label,
                        split="test",
                        defect_type=defect_type,
                        mask_path=mask_path,
                    )
                )
        return records

    def _resolve_mask_path(self, defect_type: str, image_path: Path) -> Path:
        """Resolve the expected mask path for an anomalous image."""

        candidates = [
            self.category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png",
            self.category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.pgm",
            self.category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.ppm",
        ]
        for mask_path in candidates:
            if mask_path.exists():
                return mask_path
        raise FileNotFoundError(
            "Ground-truth mask does not exist. Checked: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def _resolve_reference_path(self) -> Path:
        """Resolve the fixed category-level reference image path."""

        candidates = self._sorted_image_paths(self.category_root / "train" / "good")
        if not candidates:
            raise FileNotFoundError(
                f"No train/good reference candidates found under {self.category_root / 'train' / 'good'}."
            )
        if not 0 <= self.reference_index < len(candidates):
            raise IndexError(
                f"reference_index={self.reference_index} is out of range for {len(candidates)} candidates."
            )
        return candidates[self.reference_index]

    @staticmethod
    def _sorted_image_paths(directory: Path) -> list[Path]:
        """Return sorted image files from a directory."""

        if not directory.exists():
            return []

        allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}
        return sorted(
            path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed_suffixes
        )
