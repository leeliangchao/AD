"""Smoke test for the reference-conditioned baseline."""

from pathlib import Path
import subprocess
import sys

from adrf.data.datamodule import MVTecDataModule


def test_reference_baseline_uses_fixed_reference_and_script_executes() -> None:
    """Dataset should populate a fixed category reference and the script should run."""

    project_root = Path(__file__).resolve().parents[1]
    datamodule = MVTecDataModule(
        root=project_root / "tests" / "fixtures" / "mvtec",
        category="bottle",
        reference_index=0,
        image_size=(32, 32),
        batch_size=2,
        num_workers=0,
        normalize=False,
    )
    sample = next(iter(datamodule.train_dataloader()))[0]
    assert sample.has_reference() is True
    assert sample.metadata["reference_path"].endswith("train/good/000.ppm")

    result = subprocess.run(
        [sys.executable, "scripts/run_reference_baseline.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "image_auroc" in result.stdout

