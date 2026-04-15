from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from adrf.protocol.context import ProtocolContext
from adrf.protocol.results import TrainSummary
from adrf.protocol.runtime_support import merge_distributed_train_summary
from adrf.utils.distributed import all_gather_objects


class TrainingStrategy(ABC):
    @abstractmethod
    def train(self, context: ProtocolContext) -> TrainSummary:
        """Execute one training phase."""


class OfflineTrainingStrategy(TrainingStrategy):
    def train(self, context: ProtocolContext) -> TrainSummary:
        train_samples = []
        train_representations = []
        num_batches = 0
        num_train_samples = 0

        for batch in context.train_loader:
            num_batches += 1
            num_train_samples += len(batch)
            batch_representations = context.representation.encode_batch(batch)
            train_samples.extend(batch)
            train_representations.extend(batch_representations.unbind())

        if (
            context.distributed_context is not None
            and getattr(context.distributed_context, "enabled", False)
            and getattr(context.distributed_context, "world_size", 1) > 1
            and not context.distributed_training_enabled
        ):
            gathered = all_gather_objects(
                {
                    "samples": train_samples,
                    "representations": train_representations,
                },
                context.distributed_context,
            )
            train_samples = []
            train_representations = []
            for payload in gathered:
                train_samples.extend(payload.get("samples", []))
                train_representations.extend(payload.get("representations", []))

        context.normality.fit(train_representations, train_samples)
        return merge_distributed_train_summary(
            TrainSummary(
                num_train_batches=num_batches,
                num_train_samples=num_train_samples,
            ),
            context.distributed_context,
        )


class JointTrainingStrategy(TrainingStrategy):
    def train(self, context: ProtocolContext) -> TrainSummary:
        context.normality.configure_joint_training(context.representation_model)

        num_batches = 0
        num_train_samples = 0
        metric_totals: dict[str, float] = defaultdict(float)

        for batch in context.train_loader:
            num_batches += 1
            num_train_samples += len(batch)
            batch_representations = context.representation.encode_batch(batch)
            batch_metrics = context.normality.fit_batch(batch_representations, batch)
            for key, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    metric_totals[str(key)] += float(value)

        summary = TrainSummary(
            num_train_batches=num_batches,
            num_train_samples=num_train_samples,
            metrics={
                key: total / max(num_batches, 1)
                for key, total in metric_totals.items()
            },
        )
        return merge_distributed_train_summary(summary, context.distributed_context)


def resolve_training_strategy(context: ProtocolContext) -> TrainingStrategy:
    fit_mode = str(getattr(context.normality, "fit_mode", "offline"))
    if fit_mode == "offline":
        return OfflineTrainingStrategy()
    if fit_mode == "joint":
        return JointTrainingStrategy()
    raise ValueError(f"Unsupported normality fit mode: {fit_mode}")
