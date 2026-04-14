# Official Baseline Contract

## Claim-Bearing Contract

The only claim-bearing comparison contract in this repository is:

```text
configs/ablation/paper_baseline_matrix_official_v1.yaml
```

Anything else is either a smoke config, a filled implementation config, or a convenience wrapper around this contract.

## Fixed Values

The official contract fixes the comparison scope to:

| Field | Fixed Value |
| --- | --- |
| datasets | `mvtec_bottle`, `mvtec_capsule`, `mvtec_grid` |
| seeds | `[0, 1, 2]` |
| protocol | `one_class` |
| evaluator | `basic_ad` |
| runtime_config | `configs/runtime/real.yaml` |
| trainer.max_epochs | `10` |
| dataloader.batch_size | `32` |
| optimization.lr | `1.0e-4` |
| dataset.image_size | `[256, 256]` |
| diffusion.num_steps | `10` |
| normality.backend | `legacy` |
| fixed metrics | `image_auroc`, `pixel_auroc`, `pixel_aupr`, `train_time`, `total_time` |

## Official Core Baseline Combinations

The official matrix carries exactly seven claim-bearing baseline combinations:

1. `feature_memory + feature_distance`
2. `autoencoder + reconstruction_residual`
3. `diffusion_basic + noise_residual`
4. `diffusion_inversion_basic + path_cost`
5. `diffusion_inversion_basic + direction_mismatch`
6. `reference_basic + conditional_violation`
7. `reference_diffusion_basic + noise_residual`

## What Is Part Of The Official Contract

These fields are part of the official comparison contract:

- Dataset set and dataset config identities
- Seed set
- Normality family set
- Evidence family set
- `representation_map`
- `compatibility`
- `protocol`
- `evaluation`
- `runtime_config`
- The fixed override values that enter execution today
- The emitted comparison metrics listed above

## What Is Not Part Of The Official Contract

These are not claim-bearing by themselves:

- `configs/ablation/paper_baseline_matrix_official_v1_filled_smoke.yaml`
- `configs/ablation/paper_baseline_matrix_official_v1_filled.yaml`
- Wrapper scripts such as `scripts/run_official_filled_matrix.py`
- Failure-analysis assets
- Benchmark suites, audited matrices, budgeted matrices, and ad hoc experiment configs
- Legacy pseudo-fields that do not enter the current execution path
  - `output.summary_dir`
  - `output.keep_per_seed_runs`
  - `defaults.run_mode`
  - `defaults.save_checkpoint`
  - `defaults.export_report`
  - `defaults.export_predictions`
  - `defaults.logger`

## Practical Reading

Use the official contract to decide what comparison is allowed to support a research claim. Use filled configs only to supply stronger baseline implementations while preserving the same comparison boundary.
