from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ProtocolContext:
    train_loader: Any
    test_loader: Any
    representation: Any
    normality: Any
    evidence: Any
    evaluator: Any
    distributed_context: Any
    distributed_training_enabled: bool

    @property
    def representation_model(self) -> Any:
        return getattr(self.representation, "representation", self.representation)

    @classmethod
    def from_runner(cls, runner: Any) -> "ProtocolContext":
        datamodule = getattr(runner, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("ProtocolContext requires runner.datamodule.")

        representation = getattr(runner, "representation", None)
        if representation is None:
            raise RuntimeError("ProtocolContext requires runner.representation.")

        normality = getattr(runner, "normality", None)
        if normality is None:
            raise RuntimeError("ProtocolContext requires runner.normality.")

        evidence = getattr(runner, "evidence", None)
        if evidence is None:
            raise RuntimeError("ProtocolContext requires runner.evidence.")

        evaluator = getattr(runner, "evaluator", None)
        if evaluator is None:
            raise RuntimeError("ProtocolContext requires runner.evaluator.")

        if not hasattr(datamodule, "train_dataloader") or not hasattr(datamodule, "test_dataloader"):
            raise RuntimeError("ProtocolContext requires train_dataloader() and test_dataloader() on the datamodule.")

        return cls(
            train_loader=datamodule.train_dataloader(),
            test_loader=datamodule.test_dataloader(),
            representation=representation,
            normality=normality,
            evidence=evidence,
            evaluator=evaluator,
            distributed_context=getattr(runner, "distributed_context", None),
            distributed_training_enabled=bool(getattr(runner, "distributed_training_enabled", False)),
        )
