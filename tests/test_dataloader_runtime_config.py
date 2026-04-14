"""Tests for dataloader runtime profile overrides."""

from pathlib import Path

from torch.utils.data.distributed import DistributedSampler

from adrf.data.datamodule import MVTecDataModule


def test_datamodule_uses_runtime_profile_loader_overrides_without_taking_batch_size() -> None:
    """Runtime dataloader settings should not override experiment-level batch size."""

    project_root = Path(__file__).resolve().parents[1]
    datamodule = MVTecDataModule(
        root=project_root / "tests" / "fixtures" / "mvtec",
        category="bottle",
        runtime={
            "dataloader": {
                "batch_size": 3,
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
            }
        },
        batch_size=2,
        num_workers=0,
        normalize=False,
    )

    train_loader = datamodule.train_dataloader()

    assert train_loader.batch_size == 2
    assert train_loader.num_workers == 2
    assert train_loader.pin_memory is True
    assert train_loader.persistent_workers is True


def test_datamodule_uses_distributed_sampler_when_runtime_requests_it() -> None:
    """Distributed runtime metadata should switch train/test loaders to DistributedSampler."""

    project_root = Path(__file__).resolve().parents[1]
    datamodule = MVTecDataModule(
        root=project_root / "tests" / "fixtures" / "mvtec",
        category="bottle",
        runtime={
            "dataloader": {"num_workers": 0},
            "distributed": {"rank": 1, "world_size": 2},
        },
        batch_size=2,
        num_workers=0,
        normalize=False,
    )

    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    assert isinstance(train_loader.sampler, DistributedSampler)
    assert isinstance(test_loader.sampler, DistributedSampler)
    assert train_loader.sampler.rank == 1
    assert train_loader.sampler.num_replicas == 2
