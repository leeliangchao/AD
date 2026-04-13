"""Tests for dataloader runtime profile overrides."""

from pathlib import Path

from adrf.data.datamodule import MVTecDataModule


def test_datamodule_uses_runtime_profile_loader_overrides() -> None:
    """DataLoader kwargs should be derived from the runtime profile when provided."""

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

    assert train_loader.batch_size == 3
    assert train_loader.num_workers == 2
    assert train_loader.pin_memory is True
    assert train_loader.persistent_workers is True

