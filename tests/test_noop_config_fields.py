"""Tests that legacy no-op fields do not enter the official baseline contract."""

from __future__ import annotations

from pathlib import Path

from adrf.utils.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MATRIX_PATH = PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1.yaml"


def test_official_matrix_uses_only_effective_contract_sections() -> None:
    """The official matrix should omit legacy planning/reporting-only sections."""

    config = load_yaml_config(OFFICIAL_MATRIX_PATH)

    assert set(config) == {
        "name",
        "description",
        "datasets",
        "normality",
        "evidence",
        "representation_map",
        "compatibility",
        "protocol",
        "evaluation",
        "runtime_config",
        "seeds",
        "overrides",
    }


def test_official_matrix_excludes_legacy_noop_fields() -> None:
    """Legacy pseudo-fields should not appear in the official baseline config."""

    config = load_yaml_config(OFFICIAL_MATRIX_PATH)

    assert "defaults" not in config
    assert "output" not in config
    for dotted_path in (
        ("output", "summary_dir"),
        ("output", "keep_per_seed_runs"),
        ("defaults", "run_mode"),
        ("defaults", "save_checkpoint"),
        ("defaults", "export_report"),
        ("defaults", "export_predictions"),
        ("defaults", "logger"),
    ):
        cursor = config
        for key in dotted_path:
            if not isinstance(cursor, dict) or key not in cursor:
                break
            cursor = cursor[key]
        else:  # pragma: no cover - defensive guard for unexpected nested presence
            raise AssertionError(f"Unexpected official contract field present: {'.'.join(dotted_path)}")
