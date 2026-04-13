"""One-class protocol for the MVP anomaly-detection pipeline."""

from __future__ import annotations

from typing import Any

from adrf.protocol.base import BaseProtocol


class OneClassProtocol(BaseProtocol):
    """Train on normal data only, then evaluate the full AD pipeline on test data."""

    def train_epoch(self, runner: Any) -> dict[str, Any]:
        """Collect train representations and fit the configured normality model."""

        dataloader = runner.datamodule.train_dataloader()
        train_samples = []
        train_representations = []
        num_batches = 0
        for batch in dataloader:
            num_batches += 1
            for sample in batch:
                representation = runner.representation(sample)
                train_samples.append(sample)
                train_representations.append(representation)

        runner.normality.fit(train_representations, train_samples)
        return {
            "num_train_batches": num_batches,
            "num_train_samples": len(train_samples),
        }

    def evaluate(self, runner: Any) -> dict[str, float]:
        """Run the inference pipeline over the test split and compute metrics."""

        runner.evaluator.reset()
        dataloader = runner.datamodule.test_dataloader()
        for batch in dataloader:
            for sample in batch:
                representation = runner.representation(sample)
                artifacts = runner.normality.infer(sample, representation)
                prediction = runner.evidence.predict(sample, artifacts)
                runner.evaluator.update(prediction, sample)
        return runner.evaluator.compute()

