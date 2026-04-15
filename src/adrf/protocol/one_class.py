"""One-class protocol for the MVP anomaly-detection pipeline."""

from __future__ import annotations

from typing import Any

from adrf.protocol.base import BaseProtocol
from adrf.protocol.context import ProtocolContext
from adrf.protocol.results import EvaluationSummary, TrainSummary
from adrf.protocol.runtime_support import (
    merge_distributed_evaluator_state,
    temporarily_switch_modules_to_eval,
)
from adrf.protocol.training import resolve_training_strategy


class OneClassProtocol(BaseProtocol):
    """Runner-based one-class protocol for the shared AD experiment lifecycle."""

    def train(self, context: ProtocolContext) -> TrainSummary:
        return resolve_training_strategy(context).train(context)

    def evaluate_context(self, context: ProtocolContext) -> EvaluationSummary:
        return evaluate_one_class(context)


def evaluate_one_class(context: ProtocolContext) -> EvaluationSummary:
    context.evaluator.reset()
    with temporarily_switch_modules_to_eval(context.representation_model, context.normality):
        for batch in context.test_loader:
            batch_representations = context.representation.encode_batch(batch).unbind()
            for sample, representation in zip(batch, batch_representations, strict=True):
                artifacts = context.normality.infer(sample, representation)
                prediction = context.evidence.predict(sample, artifacts)
                context.evaluator.update(prediction, sample)

    merge_distributed_evaluator_state(context.evaluator, context.distributed_context)
    metrics = {
        str(key): float(value)
        for key, value in context.evaluator.compute().items()
        if isinstance(value, (int, float))
    }
    return EvaluationSummary(metrics=metrics)
