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

    def build_train_context(self, runner: Any) -> ProtocolContext:
        if type(self).build_context is not BaseProtocol.build_context:
            return self.build_context(runner)
        return ProtocolContext.from_runner(runner, phase="train")

    def build_evaluate_context(self, runner: Any) -> ProtocolContext:
        if type(self).build_context is not BaseProtocol.build_context:
            return self.build_context(runner)
        return ProtocolContext.from_runner(runner, phase="evaluate")

    def train_epoch(self, runner: Any) -> dict[str, Any]:
        return self.train(self.build_train_context(runner)).to_dict()

    def evaluate(self, runner: Any) -> dict[str, float]:
        return self.evaluate_context(self.build_evaluate_context(runner)).to_dict()

    def run(self, runner: Any) -> dict[str, Any]:
        mode = self._resolve_mode()

        if mode == "legacy":
            train_result = self.train_epoch(runner)
            evaluation_result = self.evaluate(runner)
        elif mode == "typed":
            train_result = self.train(self.build_train_context(runner)).to_dict()
            evaluation_result = self.evaluate_context(self.build_evaluate_context(runner)).to_dict()
        else:
            raise RuntimeError(mode)

        return {
            "train": train_result,
            "evaluation": evaluation_result,
        }

    def _resolve_mode(self) -> str:
        typed_train = type(self).train is not BaseProtocol.train
        typed_evaluate = type(self).evaluate_context is not BaseProtocol.evaluate_context
        legacy_train = type(self).train_epoch is not BaseProtocol.train_epoch
        legacy_evaluate = type(self).evaluate is not BaseProtocol.evaluate

        typed_mode = typed_train and typed_evaluate and not legacy_train and not legacy_evaluate
        legacy_mode = legacy_train and legacy_evaluate and not typed_train and not typed_evaluate

        if typed_mode:
            return "typed"
        if legacy_mode:
            return "legacy"
        if typed_train or typed_evaluate or legacy_train or legacy_evaluate:
            raise RuntimeError(
                f"{type(self).__name__} mixes typed and legacy protocol phases; "
                "implement either train/evaluate_context or train_epoch/evaluate."
            )
        raise RuntimeError(
            f"{type(self).__name__} must implement either train/evaluate_context or train_epoch/evaluate."
        )
