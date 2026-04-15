from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from adrf.normality.training import (
    JointNormalityTrainingAdapter,
    OfflineNormalityTrainingAdapter,
    resolve_normality_training_adapter,
)
from adrf.protocol.context import ProtocolContext
from adrf.protocol.results import TrainSummary
from adrf.protocol import runtime_support
from adrf.utils.distributed import all_gather_objects


class TrainingStrategy(ABC):
    @abstractmethod
    def train(self, context: ProtocolContext) -> TrainSummary:
        """Execute one training phase."""


class OfflineTrainingStrategy(TrainingStrategy):
    def train(self, context: ProtocolContext) -> TrainSummary:
        adapter = resolve_normality_training_adapter(context.normality)
        if not isinstance(adapter, OfflineNormalityTrainingAdapter):
            raise RuntimeError(f"{type(context.normality).__name__} does not support offline training.")

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

        adapter.fit(train_representations, train_samples)
        return runtime_support.merge_distributed_train_summary(
            TrainSummary(
                num_train_batches=num_batches,
                num_train_samples=num_train_samples,
            ),
            context.distributed_context,
        )


class JointTrainingStrategy(TrainingStrategy):
    def train(self, context: ProtocolContext) -> TrainSummary:
        adapter = resolve_normality_training_adapter(context.normality)
        if not isinstance(adapter, JointNormalityTrainingAdapter):
            raise RuntimeError(f"{type(context.normality).__name__} does not support joint training.")

        adapter.configure_joint_training(context.representation_model)

        num_batches = 0
        num_train_samples = 0
        metric_totals: dict[str, float] = defaultdict(float)
        metric_counts: dict[str, int] = defaultdict(int)

        for batch in context.train_loader:
            num_batches += 1
            num_train_samples += len(batch)
            batch_representations = context.representation.encode_batch(batch)
            batch_metrics = adapter.fit_batch(batch_representations, batch)
            for key, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    metric_key = str(key)
                    metric_totals[metric_key] += float(value)
                    metric_counts[metric_key] += 1

        summary = TrainSummary(
            num_train_batches=num_batches,
            num_train_samples=num_train_samples,
            metrics={
                key: total / max(metric_counts[key], 1)
                for key, total in metric_totals.items()
            },
            metric_weights={
                key: float(metric_counts[key])
                for key in metric_totals
            },
        )
        return runtime_support.merge_distributed_train_summary(summary, context.distributed_context)


def resolve_training_strategy(context: ProtocolContext) -> TrainingStrategy:
    adapter = resolve_normality_training_adapter(context.normality)
    if isinstance(adapter, OfflineNormalityTrainingAdapter):
        return OfflineTrainingStrategy()
    if isinstance(adapter, JointNormalityTrainingAdapter):
        return JointTrainingStrategy()
    raise ValueError(f"Unsupported normality fit mode: {getattr(context.normality, 'fit_mode', None)}")
