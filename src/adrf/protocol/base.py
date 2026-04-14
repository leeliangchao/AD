"""Base protocol helpers for experiment orchestration."""

from __future__ import annotations

from abc import ABC
from typing import Any

from adrf.core.interfaces import Protocol as ProtocolContract
from adrf.protocol.context import ProtocolContext
from adrf.protocol.results import EvaluationSummary, TrainSummary


class BaseProtocol(ProtocolContract, ABC):
    """Protocol shell that keeps the public runner contract but uses typed internals."""

    def train(self, context: ProtocolContext) -> TrainSummary:
        """Execute the training phase for an explicit protocol context."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement train(context) or override train_epoch(runner)."
        )

    def evaluate_context(self, context: ProtocolContext) -> EvaluationSummary:
        """Execute evaluation for an explicit protocol context."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement evaluate_context(context) or override evaluate(runner)."
        )

    def build_context(self, runner: Any) -> ProtocolContext:
        return ProtocolContext.from_runner(runner)

    def train_epoch(self, runner: Any) -> dict[str, Any]:
        return self.train(self.build_context(runner)).to_dict()

    def evaluate(self, runner: Any) -> dict[str, float]:
        return self.evaluate_context(self.build_context(runner)).to_dict()

    def run(self, runner: Any) -> dict[str, Any]:
        context: ProtocolContext | None = None

        def get_context() -> ProtocolContext:
            nonlocal context
            if context is None:
                context = self.build_context(runner)
            return context

        if self._uses_legacy_train_epoch():
            train_result = self.train_epoch(runner)
        else:
            train_result = self.train(get_context()).to_dict()

        if self._uses_legacy_evaluate():
            evaluation_result = self.evaluate(runner)
        else:
            evaluation_result = self.evaluate_context(get_context()).to_dict()

        return {
            "train": train_result,
            "evaluation": evaluation_result,
        }

    def _uses_legacy_train_epoch(self) -> bool:
        return type(self).train_epoch is not BaseProtocol.train_epoch

    def _uses_legacy_evaluate(self) -> bool:
        return type(self).evaluate is not BaseProtocol.evaluate
