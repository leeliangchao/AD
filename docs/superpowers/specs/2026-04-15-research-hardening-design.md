# Research Hardening Design

## Goal

Move the framework from a test-first prototype into a research-ready execution system with clearer module boundaries, stronger runtime contracts, and fewer implicit cross-layer assumptions.

## Scope

This design covers the full `src/adrf` tree at the architectural level, with implementation priority focused on the layers that currently carry the most ambiguity:

- `protocol`
- `normality`
- `evidence / evaluator / artifact`
- `runner / runtime / distributed`
- `benchmark / ablation / reporting / logging`
- `data / representation`
- `registry / config`

## Current Architectural Assessment

### 1. Protocol

The protocol layer was originally too thin and too weak. That has already been partially corrected:

- explicit `ProtocolContext`
- typed train/evaluation summaries
- shared protocol runtime helpers
- extracted training strategies

The remaining protocol risk is mostly transitional-shell cleanup, not core orchestration design.

### 2. Normality

The normality layer was the real owner of training semantics but expressed that through implicit capabilities:

- `fit_mode`
- `fit`
- `configure_joint_training`
- `fit_batch`
- runtime duck-typing

This has now been substantially hardened through:

- explicit normality training adapters
- explicit normality runtime specs
- explicit normality runtime state
- explicit normality representation adapters
- earlier contract validation

The remaining work is to keep shrinking legacy compatibility paths and unify more execution semantics around the explicit state objects.

### 3. Evidence / Evaluator / Artifact

The evidence/evaluator/artifact boundary was still passing around anonymous dictionaries. That has now been partially hardened:

- explicit `EvidencePrediction`
- explicit `ADEvaluatorState`
- explicit `NormalityArtifacts` serialization
- explicit artifact capability validation

The remaining work is around stronger typed prediction payloads, explicit map/scalar validation boundaries, and more structured report/export metadata.

### 4. Runner / Runtime / Distributed

This is still the largest remaining structural risk.

Observed weaknesses:

- large `ExperimentRunner`
- runtime and distributed semantics split across runner, CLI, and utility helpers
- benchmark/ablation execution historically bypassed launcher semantics
- local logging lifecycle can still mask root failures in some paths

The runtime/runner area needs the next round of consolidation.

### 5. Data / Representation

This layer is more stable than protocol/normality, but still carries prototype-era choices:

- datamodule and dataset assumptions are strongly MVTec-specific
- representation runtime behavior is partly hidden in adapters
- train/val/calibration splitting logic is embedded in the datamodule

This is not the first blocker for research execution, but it is a medium-priority area for future cleanup.

### 6. Benchmark / Ablation / Reporting / Logging

This area is functional but still contains operational thin spots:

- benchmark and ablation runners own a lot of orchestration glue
- reporting and table-export layers are file-heavy and weakly typed
- local logging is central but lifecycle-sensitive

This is the highest-value next integration layer because it determines whether “research-ready” actually holds under real experiment orchestration.

### 7. Registry / Config

Registry/config are still minimal, which is good for flexibility but weaker for safety:

- component specs remain shallow mappings
- schema validation is limited
- some config-level mistakes still surface too late

This should be improved, but after the runtime/orchestration layer is stabilized.

## Design Direction

The system should move toward explicit contracts per layer:

1. **Protocol contract**
   - explicit context
   - explicit summaries
   - explicit phase helpers

2. **Normality contract**
   - explicit training adapters
   - explicit runtime specs
   - explicit runtime state
   - explicit representation normalization

3. **Evidence/evaluator contract**
   - explicit prediction object
   - explicit evaluator state object
   - explicit artifact serialization and validation

4. **Execution/orchestration contract**
   - explicit experiment launch mode
   - explicit distributed launch boundary
   - explicit logger lifecycle guarantees
   - explicit report/export inputs

5. **Configuration contract**
   - clearer schema validation
   - fewer late runtime failures

## Priority Order

### P0: Finish execution reliability

This is the next most important area after the protocol/normality/evidence hardening already in progress.

Focus:

- `ExperimentRunner`
- launcher helpers
- benchmark runner
- ablation runner
- run logger lifecycle

Desired outcome:

- no hidden launcher/runtime mismatch
- no masked root errors
- no experiment orchestration path that behaves differently just because it bypasses CLI glue

### P1: Strengthen reporting/export boundaries

Focus:

- `reporting/summary.py`
- `statistics/table_export.py`
- report/export scripts

Desired outcome:

- clearer summary contracts
- less implicit dependence on raw record dict shapes
- easier downstream analysis without brittle field guessing

### P2: Strengthen data/representation boundaries

Focus:

- datamodule
- representation runtime adapter
- dataset split planning

Desired outcome:

- clearer dataset split semantics
- less hidden runtime mutation
- easier extension to non-MVTec experimental settings

### P3: Strengthen registry/config validation

Focus:

- registry
- config loading
- component spec validation

Desired outcome:

- config mistakes fail earlier
- more stable experiment authoring
- fewer runtime surprises in long experiment batches

## Concrete Program

The practical program is:

1. Continue shipping focused hardening slices as small, reviewable PRs.
2. Keep each slice attached to one layer boundary, not “mega refactors”.
3. Require fresh targeted regression runs per slice.
4. Use higher-level orchestration tests to prove that layer changes survive realistic experiment execution.

## Definition Of “Research-Ready”

The project should be considered meaningfully closer to research-ready when:

- orchestration works for single-process and launcher-based multi-process paths
- core contracts are explicit across protocol, normality, evidence, evaluator, and artifact boundaries
- experiment failures report the true root cause
- benchmark/ablation/reporting paths use the same execution semantics as standalone experiments
- module responsibilities are understandable without reading half the repo

## Status

As of this design revision:

- protocol hardening: substantially advanced
- normality hardening: substantially advanced
- evidence/evaluator/artifact hardening: started and advancing
- runner/runtime/launch/reporting hardening: next major focus

The recommended next implementation work is therefore not “more normality first”, but execution reliability across runner, launcher, ablation, benchmark, and logging.
