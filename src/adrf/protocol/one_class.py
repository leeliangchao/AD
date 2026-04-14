"""One-class protocol for the MVP anomaly-detection pipeline."""

from __future__ import annotations

from typing import Any

from adrf.protocol.base import BaseProtocol
from adrf.utils.distributed import all_gather_objects


class OneClassProtocol(BaseProtocol):
    """Runner-based one-class protocol for the shared AD experiment lifecycle."""

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

        distributed_context = getattr(runner, "distributed_context", None)
        if (
            distributed_context is not None
            and distributed_context.enabled
            and distributed_context.world_size > 1
            and not bool(getattr(runner, "distributed_training_enabled", False))
        ):
            gathered = all_gather_objects(
                {
                    "samples": train_samples,
                    "representations": train_representations,
                },
                distributed_context,
            )
            train_samples = []
            train_representations = []
            for payload in gathered:
                train_samples.extend(payload.get("samples", []))
                train_representations.extend(payload.get("representations", []))

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

        distributed_context = getattr(runner, "distributed_context", None)
        if (
            distributed_context is not None
            and distributed_context.enabled
            and distributed_context.world_size > 1
        ):
            gathered_states = all_gather_objects(runner.evaluator.state_dict(), distributed_context)
            merged_state = runner.evaluator.merge_states(gathered_states)
            runner.evaluator.load_state_dict(merged_state)
        return runner.evaluator.compute()
