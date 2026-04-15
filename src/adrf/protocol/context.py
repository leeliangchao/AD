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
    def from_runner(cls, runner: Any, phase: str | None = None) -> "ProtocolContext":
        datamodule = getattr(runner, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("ProtocolContext requires runner.datamodule.")

        representation = getattr(runner, "representation", None)
        if representation is None:
            raise RuntimeError("ProtocolContext requires runner.representation.")

        normality = getattr(runner, "normality", None)
        if normality is None:
            raise RuntimeError("ProtocolContext requires runner.normality.")

        if phase not in {None, "train", "evaluate"}:
            raise RuntimeError(f"ProtocolContext does not support phase={phase!r}.")

        if phase in {None, "evaluate"}:
            evidence = getattr(runner, "evidence", None)
            if evidence is None:
                raise RuntimeError("ProtocolContext requires runner.evidence.")

            evaluator = getattr(runner, "evaluator", None)
            if evaluator is None:
                raise RuntimeError("ProtocolContext requires runner.evaluator.")
        else:
            evidence = getattr(runner, "evidence", None)
            evaluator = getattr(runner, "evaluator", None)

        train_loader = None
        test_loader = None
        if phase in {None, "train"}:
            if not hasattr(datamodule, "train_dataloader"):
                raise RuntimeError("ProtocolContext requires train_dataloader() on the datamodule.")
            train_loader = datamodule.train_dataloader()
        if phase in {None, "evaluate"}:
            if not hasattr(datamodule, "test_dataloader"):
                raise RuntimeError("ProtocolContext requires test_dataloader() on the datamodule.")
            test_loader = datamodule.test_dataloader()

        if phase is None:
            if evidence is None:
                raise RuntimeError("ProtocolContext requires runner.evidence.")
            if evaluator is None:
                raise RuntimeError("ProtocolContext requires runner.evaluator.")

        return cls(
            train_loader=train_loader,
            test_loader=test_loader,
            representation=representation,
            normality=normality,
            evidence=evidence,
            evaluator=evaluator,
            distributed_context=getattr(runner, "distributed_context", None),
            distributed_training_enabled=bool(getattr(runner, "distributed_training_enabled", False)),
        )
