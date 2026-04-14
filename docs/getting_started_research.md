# Getting Started For Research

This repository's current mainline is the official baseline matrix workflow:

- Claim-bearing contract: `configs/ablation/paper_baseline_matrix_official_v1.yaml`
- First smoke run: `configs/ablation/paper_baseline_matrix_official_v1_filled_smoke.yaml`
- Full filled run: `configs/ablation/paper_baseline_matrix_official_v1_filled.yaml`
- Runner wrapper: `uv run python scripts/run_official_filled_matrix.py --config <matrix.yaml>`

## What Is Official

If you are using this repo for research, treat these as the official entrypoints:

- Contract definition: `docs/official_baseline_contract.md`
- Result reading guide: `docs/result_reading_guide.md`
- Environment/path check: `uv run python scripts/check_research_env.py`
- Contract printer: `uv run python scripts/print_official_contract.py`

Do not treat ad hoc configs, benchmark suites, or older matrix variants as the main research contract.

## Where Real Data Goes

Real MVTec data should live at:

```text
data/mvtec/
```

Expected category directories for the official baseline contract:

```text
data/mvtec/bottle/
data/mvtec/capsule/
data/mvtec/grid/
```

The filled smoke config uses fixture data under `tests/fixtures/mvtec` so that you can validate the pipeline before touching real data. The full filled config points at `data/mvtec`.

## Minimum Command Path

For first use, run these commands in order:

```bash
uv sync
uv run python scripts/check_research_env.py
uv run python scripts/print_official_contract.py
uv run python scripts/run_official_filled_matrix.py --config configs/ablation/paper_baseline_matrix_official_v1_filled_smoke.yaml
uv run python scripts/export_failure_analysis.py
```

After the smoke run looks sane, move to the real contract-sized filled matrix:

```bash
uv run python scripts/run_official_filled_matrix.py --config configs/ablation/paper_baseline_matrix_official_v1_filled.yaml
uv run python scripts/export_failure_analysis.py
```

## What To Look At First

After a matrix run, start in the latest directory under:

```text
outputs/ablations/<timestamp>_paper_baseline_matrix_official_v1_filled_smoke/
```

or:

```text
outputs/ablations/<timestamp>_paper_baseline_matrix_official_v1_filled/
```

Read results in this order:

1. `paper_table.md`
2. `paper_table_category_mean.md`
3. `paper_table_by_axis.md`
4. `failure_analysis/summary.json`
5. `matrix_results.json`

`paper_table.md` is the fastest way to confirm that all seven official baseline combinations finished. `paper_table_category_mean.md` and `paper_table_by_axis.md` are the first files that should inform research judgment. `failure_analysis/summary.json` is for audit and interpretation, not for replacing the table-level decision.
