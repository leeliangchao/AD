"""Export grouped audit tables from an aggregated ablation matrix run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.reporting.summary import find_latest_ablation_dir
from adrf.statistics.table_export import export_grouped_paper_tables


def export_audit_tables(matrix_dir: str | Path) -> dict[str, str]:
    """Export the grouped paper tables used for stage-16 audit review."""

    outputs = export_grouped_paper_tables(matrix_dir)
    return {key: str(value) for key, value in outputs.items()}


def main() -> int:
    """Export grouped audit tables for the latest or given matrix directory."""

    matrix_dir = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else find_latest_ablation_dir(PROJECT_ROOT / "outputs" / "ablations")
    )
    outputs = export_audit_tables(matrix_dir)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
