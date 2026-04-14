"""Minimal datamodule for the MVTec single-category workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from adrf.core.sample import Sample
from adrf.data.datasets.mvtec import MVTecSingleClassDataset
from adrf.data.transforms import SampleTransform, build_sample_transform
from adrf.utils.runtime import resolve_dataloader_runtime


def _collate_samples(batch: list[Sample]) -> list[Sample]:
    """Keep dataloader outputs as a list of Sample objects."""

    return batch


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
    runtime: dict[str, object] | None = None
    transform: SampleTransform | None = None
    train_dataset: MVTecSingleClassDataset | None = field(default=None, init=False)
    test_dataset: MVTecSingleClassDataset | None = field(default=None, init=False)

    def setup(self) -> None:
        """Instantiate train and test datasets if they are not built yet."""

        transform = self.transform or build_sample_transform(
            image_size=self.image_size,
            normalize=self.normalize,
        )
        self.train_dataset = MVTecSingleClassDataset(
            root=self.root,
            category=self.category,
            split="train",
            reference_index=self.reference_index,
            transform=transform,
        )
        self.test_dataset = MVTecSingleClassDataset(
            root=self.root,
            category=self.category,
            split="test",
            reference_index=self.reference_index,
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
        dataset: MVTecSingleClassDataset,
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
