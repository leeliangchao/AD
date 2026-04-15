from types import SimpleNamespace

from adrf.protocol.one_class import OneClassProtocol
from adrf.protocol.results import EvaluationSummary, TrainSummary


class _RecordingStrategy:
    def __init__(self) -> None:
        self.contexts = []

    def train(self, context):
        self.contexts.append(context)
        return TrainSummary(
            num_train_batches=1,
            num_train_samples=2,
            metrics={"loss": 0.5},
        )


def test_one_class_protocol_run_uses_resolved_strategy_and_serializes_results(monkeypatch) -> None:
    protocol = OneClassProtocol()
    strategy = _RecordingStrategy()
    context = SimpleNamespace()

    monkeypatch.setattr(protocol, "build_context", lambda runner: context)
    monkeypatch.setattr("adrf.protocol.one_class.resolve_training_strategy", lambda context_arg: strategy)
    monkeypatch.setattr(
        "adrf.protocol.one_class.evaluate_one_class",
        lambda context_arg: EvaluationSummary(metrics={"image_auroc": 0.9}),
    )

    results = protocol.run(SimpleNamespace())

    assert strategy.contexts == [context]
    assert results == {
        "train": {"num_train_batches": 1, "num_train_samples": 2, "loss": 0.5},
        "evaluation": {"image_auroc": 0.9},
    }
