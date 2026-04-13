"""Tests for the unified sample schema."""

from adrf.core.sample import Sample


def test_sample_to_dict_returns_all_fields_and_copies_metadata() -> None:
    """Sample.to_dict should expose the full schema without aliasing metadata."""

    sample = Sample(
        image="image-array",
        label=1,
        mask="mask-array",
        category="bottle",
        sample_id="sample-001",
        reference="ref-image",
        views={"front": "front-image"},
        metadata={"split": "test"},
    )

    payload = sample.to_dict()

    assert payload == {
        "image": "image-array",
        "label": 1,
        "mask": "mask-array",
        "category": "bottle",
        "sample_id": "sample-001",
        "reference": "ref-image",
        "views": {"front": "front-image"},
        "metadata": {"split": "test"},
    }
    assert payload["metadata"] == sample.metadata
    assert payload["metadata"] is not sample.metadata


def test_sample_helpers_report_reference_and_views_presence() -> None:
    """Reference and multi-view helpers should reflect optional fields."""

    rich_sample = Sample(image="image-array", reference="ref-image", views={"left": "left-image"})
    plain_sample = Sample(image="image-array", views={})

    assert rich_sample.has_reference() is True
    assert rich_sample.has_views() is True
    assert plain_sample.has_reference() is False
    assert plain_sample.has_views() is False

