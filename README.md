# AD Research Framework

Industrial anomaly detection research framework for running single experiments, benchmark suites, ablation matrices, and audited report exports.

## Current Status

This repository is no longer just a round-1 scaffold. The implemented surface includes:

- Config-driven experiment assembly with registry-backed components
- MVTec single-category data loading and one-class training/evaluation
- Two representation families: `pixel`, `feature`
- Six normality model families:
  - `feature_memory`
  - `autoencoder`
  - `diffusion_basic`
  - `diffusion_inversion_basic`
  - `reference_basic`
  - `reference_diffusion_basic`
- Six evidence model families:
  - `feature_distance`
  - `reconstruction_residual`
  - `noise_residual`
  - `path_cost`
  - `direction_mismatch`
  - `conditional_violation`
- Local run logging plus optional SwanLab integration
- Benchmark suites, ablation matrices, multi-seed aggregation, paper-table export, and failure-analysis export

## Setup

Use `uv` for environment and test execution:

```bash
uv sync
uv run pytest
```

## CLI

After `uv sync`, the package exposes installable CLI entrypoints:

```bash
uv run adrf experiment configs/experiment/feature_baseline.yaml
uv run adrf benchmark configs/benchmark/baseline_suite.yaml
uv run adrf ablation configs/ablation/paper_baseline_matrix_v3_audited.yaml
uv run adrf report --kind experiment --path outputs/runs/<run_dir>
uv run adrf report --kind benchmark --path outputs/benchmarks/<suite_dir>
uv run adrf report --kind ablation --path outputs/ablations/<matrix_dir>
```

Dedicated entrypoints are also available:

```bash
uv run adrf-experiment configs/experiment/recon_baseline.yaml
uv run adrf-benchmark configs/benchmark/diffusion_suite.yaml
uv run adrf-ablation configs/ablation/diffusion_evidence_matrix.yaml
uv run adrf-report --kind benchmark
```

The repository root entrypoint is equivalent:

```bash
uv run python main.py benchmark
uv run python main.py report --kind ablation
```

## Script Entry Points

Legacy script wrappers remain available for reproducibility and batch workflows:

```bash
uv run python scripts/run_feature_baseline.py
uv run python scripts/run_recon_baseline.py
uv run python scripts/run_diffusion_baseline.py
uv run python scripts/run_diffusion_pathcost_baseline.py
uv run python scripts/run_diffusion_direction_baseline.py
uv run python scripts/run_reference_diffusion_baseline.py
uv run python scripts/run_reference_baseline.py
uv run python scripts/run_benchmark_suite.py
uv run python scripts/run_ablation_matrix.py --config configs/ablation/diffusion_evidence_matrix.yaml
uv run python scripts/run_multiseed_matrix.py --config configs/ablation/diffusion_evidence_multiseed.yaml
uv run python scripts/run_budgeted_paper_matrix.py --config configs/ablation/paper_baseline_matrix_v2_budgeted_smoke.yaml
uv run python scripts/run_audited_paper_matrix.py --config configs/ablation/paper_baseline_matrix_v3_audited_smoke.yaml
uv run python scripts/export_experiment_report.py
uv run python scripts/export_benchmark_summary.py
uv run python scripts/export_ablation_summary.py
uv run python scripts/export_paper_tables.py
uv run python scripts/export_grouped_paper_tables.py
uv run python scripts/export_audit_tables.py
uv run python scripts/export_failure_analysis.py
```

## Runtime Profiles

Available runtime presets:

```bash
configs/runtime/smoke.yaml
configs/runtime/real.yaml
```

## Outputs

Successful runs write artifacts under:

- `outputs/runs`: per-experiment run directories with config snapshots, metrics, checkpoints, and markdown reports
- `outputs/benchmarks`: suite-level summaries
- `outputs/ablations`: matrix-level summaries, grouped tables, and audited failure-analysis assets

## Diffusion Backends

- `DiffusionBasicNormality` supports `backend: legacy` and `backend: diffusers`
- `DiffusionInversionBasicNormality` supports `backend: legacy` and `backend: diffusers`
- Current experiment configs default to `backend: legacy`
