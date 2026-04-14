from types import SimpleNamespace

import pytest

from adrf.protocol.context import ProtocolContext


def test_protocol_context_from_runner_extracts_protocol_dependencies() -> None:
    train_loader = object()
    test_loader = object()
    runner = SimpleNamespace(
        datamodule=SimpleNamespace(
            train_dataloader=lambda: train_loader,
            test_dataloader=lambda: test_loader,
        ),
        representation=object(),
        normality=object(),
        evidence=object(),
        evaluator=object(),
        distributed_context=SimpleNamespace(enabled=False, world_size=1),
        distributed_training_enabled=False,
    )

    context = ProtocolContext.from_runner(runner)

    assert context.train_loader is train_loader
    assert context.test_loader is test_loader
    assert context.representation is runner.representation
    assert context.normality is runner.normality
    assert context.evidence is runner.evidence
    assert context.evaluator is runner.evaluator
    assert context.distributed_context is runner.distributed_context
    assert context.distributed_training_enabled is False


def test_protocol_context_from_runner_requires_protocol_dependencies() -> None:
    runner = SimpleNamespace(
        datamodule=None,
        representation=object(),
        normality=object(),
        evidence=object(),
        evaluator=object(),
    )

    with pytest.raises(RuntimeError, match="datamodule"):
        ProtocolContext.from_runner(runner)
