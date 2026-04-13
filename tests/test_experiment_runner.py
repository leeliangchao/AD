"""Tests for the experiment runner."""

from pathlib import Path

from PIL import Image

from adrf.runner.experiment_runner import ExperimentRunner


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _write_mask_image(path: Path, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (32, 32), color=value).save(path)


def _write_fixture_dataset(root: Path) -> None:
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (0, 0, 0))
    _write_rgb_image(root / "bottle" / "train" / "good" / "001.png", (10, 10, 10))
    _write_rgb_image(root / "bottle" / "test" / "good" / "002.png", (0, 0, 0))
    _write_rgb_image(root / "bottle" / "test" / "broken_large" / "003.png", (255, 255, 255))
    _write_mask_image(root / "bottle" / "ground_truth" / "broken_large" / "003_mask.png", 255)


def test_experiment_runner_builds_components_and_runs(tmp_path: Path) -> None:
    """ExperimentRunner should assemble the configured pipeline and return results."""

    dataset_root = tmp_path / "mvtec"
    _write_fixture_dataset(dataset_root)
    config_path = tmp_path / "recon.yaml"
    config_path.write_text(
        "\n".join(
            [
                "datamodule:",
                "  name: mvtec_single_class",
                "  params:",
                f"    root: {dataset_root.as_posix()}",
                "    category: bottle",
                "    image_size: [32, 32]",
                "    batch_size: 2",
                "    num_workers: 0",
                "    normalize: false",
                "representation:",
                "  name: pixel",
                "  params: {}",
                "normality:",
                "  name: autoencoder",
                "  params:",
                "    input_channels: 3",
                "    hidden_channels: 4",
                "    latent_channels: 8",
                "    epochs: 1",
                "    batch_size: 2",
                "evidence:",
                "  name: reconstruction_residual",
                "  params:",
                "    aggregator: mean",
                "evaluator:",
                "  name: basic_ad",
                "  params: {}",
                "protocol:",
                "  name: one_class",
                "  params: {}",
            ]
        ),
        encoding="utf-8",
    )

    runner = ExperimentRunner(config_path)
    results = runner.run()

    assert runner.datamodule is not None
    assert runner.protocol is not None
    assert set(results) == {"train", "evaluation"}
    assert results["train"]["num_train_samples"] == 2
    assert set(results["evaluation"]) == {"image_auroc", "pixel_auroc", "pixel_aupr"}

