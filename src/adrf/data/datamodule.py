"""Minimal datamodule for the MVTec single-category workflow."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from adrf.core.sample import Sample
from adrf.data.datasets.mvtec import MVTecSingleClassDataset
from adrf.data.datasets.mvtec import sorted_image_paths
from adrf.data.transforms import SampleTransform, build_sample_transform
from adrf.utils.runtime import resolve_dataloader_runtime


def _collate_samples(batch: list[Sample]) -> list[Sample]:
    """Keep dataloader outputs as a list of Sample objects."""

    return batch


class _SampleDatasetView(Dataset[Sample]):
    """Wrap a dataset to override metadata and apply datamodule-level transforms."""

    def __init__(
        self,
        dataset: Dataset[Sample],
        *,
        split: str | None = None,
        transform: SampleTransform | None = None,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        """Return the wrapped dataset length."""

        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        """Return a sample with optional split override and transform applied."""

        sample = self.dataset[index]
        if self.split is not None:
            payload = sample.to_dict()
            payload["metadata"]["split"] = self.split
            sample = Sample(**payload)
        return self.transform(sample) if self.transform is not None else sample


@dataclass(slots=True, frozen=True)
class _TrainSplitPlan:
    """Deterministic split plan for the train/good pool."""

    train_indices: list[int]
    val_indices: list[int]
    calibration_indices: list[int]


@dataclass(slots=True)
class MVTecDataModule:
    """Build train and test dataloaders for one MVTec category."""

    root: str | Path
    category: str = "bottle"
    reference_index: int = 0
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 0
    normalize: bool = True
    val_split: float = 0.0
    calibration_split: float = 0.0
    split_seed: int = 0
    runtime: dict[str, object] | None = None
    transform: SampleTransform | None = None
    train_dataset: Dataset[Sample] | None = field(default=None, init=False)
    val_dataset: Dataset[Sample] | None = field(default=None, init=False)
    calibration_dataset: Dataset[Sample] | None = field(default=None, init=False)
    test_dataset: Dataset[Sample] | None = field(default=None, init=False)

    def setup(self) -> None:
        """Instantiate train and test datasets if they are not built yet."""

        transform = self.transform or build_sample_transform(
            image_size=self.image_size,
            normalize=self.normalize,
        )
        train_candidates = self._collect_train_good_candidates()
        split_plan = self._plan_train_split(len(train_candidates))
        reference_path = self._select_reference_path(train_candidates, split_plan.train_indices)
        base_train_dataset = MVTecSingleClassDataset(
            root=self.root,
            category=self.category,
            split="train",
            reference_index=self.reference_index,
            reference_path=reference_path,
        )
        self.train_dataset = _SampleDatasetView(
            Subset(base_train_dataset, split_plan.train_indices),
            transform=transform,
        )
        self.val_dataset = _SampleDatasetView(
            Subset(base_train_dataset, split_plan.val_indices),
            split="val",
            transform=transform,
        )
        self.calibration_dataset = _SampleDatasetView(
            Subset(base_train_dataset, split_plan.calibration_indices),
            split="calibration",
            transform=transform,
        )
        self.test_dataset = MVTecSingleClassDataset(
            root=self.root,
            category=self.category,
            split="test",
            reference_index=self.reference_index,
            reference_path=reference_path,
            transform=transform,
        )

    def train_dataloader(self) -> DataLoader[list[Sample]]:
        """Return the dataloader over normal training samples."""

        if self.train_dataset is None:
            self.setup()
        loader_runtime = self._loader_runtime()
        sampler = self._distributed_sampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=int(loader_runtime["batch_size"]),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(loader_runtime["num_workers"]),
            pin_memory=bool(loader_runtime["pin_memory"]),
            persistent_workers=bool(loader_runtime["persistent_workers"]),
            collate_fn=_collate_samples,
        )

    def test_dataloader(self) -> DataLoader[list[Sample]]:
        """Return the dataloader over test samples."""

        if self.test_dataset is None:
            self.setup()
        loader_runtime = self._loader_runtime()
        sampler = self._distributed_sampler(self.test_dataset, shuffle=False)
        return DataLoader(
            self.test_dataset,
            batch_size=int(loader_runtime["batch_size"]),
            shuffle=False,
            sampler=sampler,
            num_workers=int(loader_runtime["num_workers"]),
            pin_memory=bool(loader_runtime["pin_memory"]),
            persistent_workers=bool(loader_runtime["persistent_workers"]),
            collate_fn=_collate_samples,
        )

    def val_dataloader(self) -> DataLoader[list[Sample]]:
        """Return the dataloader over the validation split carved from train/good."""

        if self.val_dataset is None:
            self.setup()
        loader_runtime = self._loader_runtime()
        sampler = self._distributed_sampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=int(loader_runtime["batch_size"]),
            shuffle=False,
            sampler=sampler,
            num_workers=int(loader_runtime["num_workers"]),
            pin_memory=bool(loader_runtime["pin_memory"]),
            persistent_workers=bool(loader_runtime["persistent_workers"]),
            collate_fn=_collate_samples,
        )

    def calibration_dataloader(self) -> DataLoader[list[Sample]]:
        """Return the dataloader over the calibration split carved from train/good."""

        if self.calibration_dataset is None:
            self.setup()
        loader_runtime = self._loader_runtime()
        sampler = self._distributed_sampler(self.calibration_dataset, shuffle=False)
        return DataLoader(
            self.calibration_dataset,
            batch_size=int(loader_runtime["batch_size"]),
            shuffle=False,
            sampler=sampler,
            num_workers=int(loader_runtime["num_workers"]),
            pin_memory=bool(loader_runtime["pin_memory"]),
            persistent_workers=bool(loader_runtime["persistent_workers"]),
            collate_fn=_collate_samples,
        )

    def _loader_runtime(self) -> dict[str, object]:
        """Resolve runtime-aware DataLoader kwargs."""

        return resolve_dataloader_runtime(
            self.runtime or {},
            defaults={
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "pin_memory": False,
                "persistent_workers": False,
                "non_blocking": False,
            },
        )

    def _distributed_sampler(
        self,
        dataset: Dataset[Sample],
        *,
        shuffle: bool,
    ) -> DistributedSampler[Sample] | None:
        """Build a DistributedSampler when the runtime requests multi-process execution."""

        distributed_cfg = (self.runtime or {}).get("distributed", {})
        if not isinstance(distributed_cfg, dict):
            return None
        world_size = int(distributed_cfg.get("world_size", 1))
        if world_size <= 1:
            return None
        rank = int(distributed_cfg.get("rank", 0))
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )

    def _collect_train_good_candidates(self) -> list[Path]:
        """Collect train/good candidates before building datasets."""

        candidate_root = Path(self.root) / self.category / "train" / "good"
        candidates = sorted_image_paths(candidate_root)
        if not candidates:
            raise FileNotFoundError(f"No train/good reference candidates found under {candidate_root}.")
        return candidates

    def _plan_train_split(self, total_samples: int) -> _TrainSplitPlan:
        """Build the deterministic train/val/calibration split indices."""

        self._validate_split_ratios()
        indices = list(range(total_samples))
        generator = random.Random(self.split_seed)
        generator.shuffle(indices)

        val_count = int(total_samples * self.val_split)
        calibration_count = int(total_samples * self.calibration_split)
        reserved_count = val_count + calibration_count
        if reserved_count >= total_samples:
            raise ValueError("validation/calibration splits must leave at least one training sample.")

        val_indices = sorted(indices[:val_count])
        calibration_indices = sorted(indices[val_count : val_count + calibration_count])
        train_indices = sorted(indices[val_count + calibration_count :])
        return _TrainSplitPlan(
            train_indices=train_indices,
            val_indices=val_indices,
            calibration_indices=calibration_indices,
        )

    def _select_reference_path(self, candidates: list[Path], train_indices: list[int]) -> Path:
        """Select the fixed reference from the retained training subset only."""

        retained_train_candidates = [candidates[index] for index in train_indices]
        if not 0 <= self.reference_index < len(retained_train_candidates):
            raise IndexError(
                f"reference_index={self.reference_index} is out of range for {len(retained_train_candidates)} "
                "retained training candidates."
            )
        return retained_train_candidates[self.reference_index]

    def _validate_split_ratios(self) -> None:
        """Validate the configured validation and calibration split ratios."""

        for name, value in (("val_split", self.val_split), ("calibration_split", self.calibration_split)):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in the interval [0.0, 1.0).")
