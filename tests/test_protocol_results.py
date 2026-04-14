from adrf.protocol.results import EvaluationSummary, TrainSummary


def test_train_summary_to_dict_flattens_metrics_into_existing_protocol_shape() -> None:
    summary = TrainSummary(
        num_train_batches=2,
        num_train_samples=5,
        metrics={"loss": 0.25},
    )

    assert summary.to_dict() == {
        "num_train_batches": 2,
        "num_train_samples": 5,
        "loss": 0.25,
    }


def test_train_summary_from_mapping_round_trips_existing_protocol_payload() -> None:
    payload = {
        "num_train_batches": 3,
        "num_train_samples": 7,
        "loss": 1.5,
    }

    assert TrainSummary.from_mapping(payload).to_dict() == payload


def test_evaluation_summary_to_dict_returns_metric_mapping() -> None:
    summary = EvaluationSummary(metrics={"image_auroc": 0.9, "pixel_auroc": 0.8})

    assert summary.to_dict() == {
        "image_auroc": 0.9,
        "pixel_auroc": 0.8,
    }
