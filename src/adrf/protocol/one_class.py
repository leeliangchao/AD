"""One-class protocol for the MVP anomaly-detection pipeline."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any

from torch import nn

from adrf.protocol.base import BaseProtocol
from adrf.utils.distributed import all_gather_objects


class OneClassProtocol(BaseProtocol):
    """Runner-based one-class protocol for the shared AD experiment lifecycle."""

    def train_epoch(self, runner: Any) -> dict[str, Any]:
        """Collect train representations and fit the configured normality model."""

        dataloader = runner.datamodule.train_dataloader()
        fit_mode = str(getattr(runner.normality, "fit_mode", "offline"))
        if fit_mode not in {"offline", "joint"}:
            raise ValueError(f"Unsupported normality fit mode: {fit_mode}")

        train_samples = []
        train_representations = []
        num_batches = 0
        num_train_samples = 0
        joint_metric_totals: dict[str, float] = defaultdict(float)
        representation_model = getattr(runner.representation, "representation", runner.representation)
        if fit_mode == "joint":
            runner.normality.configure_joint_training(representation_model)

        for batch in dataloader:
            num_batches += 1
            num_train_samples += len(batch)
            batch_representations = runner.representation.encode_batch(batch)
            if fit_mode == "offline":
                train_samples.extend(batch)
                train_representations.extend(batch_representations.unbind())
                continue

            batch_metrics = runner.normality.fit_batch(batch_representations, batch)
            for key, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    joint_metric_totals[str(key)] += float(value)

        distributed_context = getattr(runner, "distributed_context", None)
        if (
            fit_mode == "offline"
            and distributed_context is not None
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

        if fit_mode == "offline":
            runner.normality.fit(train_representations, train_samples)
            return {
                "num_train_batches": num_batches,
                "num_train_samples": len(train_samples),
            }

        train_summary: dict[str, Any] = {
            "num_train_batches": num_batches,
            "num_train_samples": num_train_samples,
        }
        for key, total in joint_metric_totals.items():
            train_summary[key] = total / max(num_batches, 1)
        return train_summary

    def evaluate(self, runner: Any) -> dict[str, float]:
        """Run the inference pipeline over the test split and compute metrics."""

        runner.evaluator.reset()
        dataloader = runner.datamodule.test_dataloader()
        with _temporarily_switch_modules_to_eval(
            getattr(runner.representation, "representation", runner.representation),
            runner.normality,
        ):
            for batch in dataloader:
                batch_representations = runner.representation.encode_batch(batch).unbind()
                for sample, representation in zip(batch, batch_representations, strict=True):
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


@contextmanager
def _temporarily_switch_modules_to_eval(*objects: object):
    """Run a block with the provided module trees in eval mode, then restore prior state."""

    roots: list[nn.Module] = []
    for obj in objects:
        if isinstance(obj, nn.Module) and all(existing is not obj for existing in roots):
            roots.append(obj)

    if not roots:
        yield
        return

    training_states = {
        module: module.training
        for root in roots
        for module in root.modules()
    }
    try:
        for root in roots:
            root.eval()
        yield
    finally:
        for module, was_training in training_states.items():
            module.train(was_training)
