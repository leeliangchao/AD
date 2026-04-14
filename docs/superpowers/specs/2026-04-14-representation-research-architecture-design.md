# Representation Research Architecture Design

**Date:** 2026-04-14

## Goal

Refactor the `representation` layer from a weak, dictionary-based feature extraction helper into a research-grade subsystem that supports both frozen and trainable representations, exposes explicit experiment semantics, and enforces compatibility across the `representation -> normality -> evidence` pipeline.

## Why This Change Is Necessary

The current implementation is sufficient for smoke tests but not for credible research iteration.

Observed issues in the current codebase:

- `FeatureRepresentation` exposes a `freeze` flag but still forces `eval()` and `torch.no_grad()` for all calls, so the apparent trainable path is silently non-trainable.
- Representation outputs are untyped `dict[str, Any]`, which allows incompatible consumers to accept invalid payloads and continue running.
- `FeatureMemoryNormality` can consume pixel tensors without failing, which means invalid experiment combinations can produce numbers instead of explicit errors.
- Representation metadata is too weak to reconstruct what a run actually used, especially for backbone choice, preprocessing, feature layer, and trainability.
- The pipeline is sample-wise at the representation boundary, which prevents efficient batched feature extraction and complicates future trainable end-to-end experiments.

This refactor intentionally introduces a breaking change in order to establish explicit contracts and long-term research correctness.

## Design Principles

- Research semantics must be explicit in code, not implicit in conventions.
- Invalid experimental combinations must fail fast before producing results.
- Frozen and trainable representation modes must both be first-class, not simulated through flags that do not change execution semantics.
- Representation outputs must carry provenance sufficient for experiment auditability and report generation.
- Batch computation is the default execution mode for neural representations.

## Scope

In scope:

- New representation interfaces and typed output objects
- Refactoring `PixelRepresentation` and `FeatureRepresentation`
- Runner and protocol changes needed to support batched representation execution
- Normality-side compatibility validation for representation space and tensor shape
- Provenance capture in runtime artifacts and experiment metadata
- Migration of immediate consumers that rely on representation outputs

Out of scope for this change:

- Adding new representation families beyond pixel and ResNet feature baselines
- Introducing an offline feature cache or dataset-side embedding storage
- Reworking evaluator math or evidence scoring logic beyond adapting to the new artifacts contract

## Architecture Overview

The representation layer will become a first-class subsystem built from four pieces:

1. `RepresentationModel`
   A shared abstraction for all representations. It will be an `nn.Module` and will own representation-specific execution semantics such as batch encoding, trainability, and provenance reporting.

2. `RepresentationOutput`
   A typed single-sample object that replaces open-ended dictionaries. It represents one sample's encoded output plus fixed metadata fields.

3. `RepresentationBatch`
   A typed batched object used by the runner, protocol, and trainable normality models. It will support conversion to per-sample outputs via `unbind()`.

4. `RepresentationProvenance`
   A structured description of the representation definition used in a run. It captures experiment semantics that must be visible in artifacts and reports.

These objects replace loose dictionary payloads and become the only supported representation contract.

## Core Contracts

### `RepresentationModel`

Every representation implementation must:

- inherit from `torch.nn.Module`
- declare a stable `space` identifier
- implement batch encoding over transformed `Sample` inputs
- expose whether it is trainable in the current configuration
- expose provenance for the encoded outputs

Required behavior:

- frozen representations may execute under `no_grad` and evaluation mode
- trainable representations must preserve gradients and obey normal `train()` / `eval()` transitions
- batch encoding must be numerically equivalent to repeated single-sample encoding, modulo normal floating-point tolerance

### `RepresentationOutput`

Single-sample representation payload with fixed fields:

- `tensor`
- `space`
- `spatial_shape`
- `feature_dim`
- `sample_id`
- `requires_grad`
- `device`
- `dtype`
- `provenance`

The object is the canonical payload stored inside downstream artifacts.

### `RepresentationBatch`

Batched representation payload with fixed fields:

- `tensor`
- `space`
- `spatial_shape`
- `feature_dim`
- `batch_size`
- `sample_ids`
- `requires_grad`
- `device`
- `dtype`
- `provenance`

Required helper methods:

- `unbind() -> list[RepresentationOutput]`
- shape validation for batch rank and metadata consistency

### `RepresentationProvenance`

Required fields:

- `representation_name`
- `backbone_name`
- `weights_source`
- `feature_layer`
- `pooling`
- `trainable`
- `frozen_submodules`
- `input_image_size`
- `input_normalize`
- `normalize_mean`
- `normalize_std`
- `code_version`
- `config_fingerprint`

This object must be deterministic for a fixed configuration and code revision.

## Representation Implementations

### `PixelRepresentation`

This remains the identity representation over transformed image tensors, but it must adopt the new contract.

Requirements:

- emit `space="pixel"`
- produce batched outputs with tensor shape `[B, C, H, W]`
- expose provenance describing image-space representation semantics and preprocessing policy
- always report `trainable=False`

### `FeatureRepresentation`

This becomes a real research module rather than a thin helper around a torchvision backbone.

Requirements:

- inherit from `nn.Module`
- support `trainable=False` and `trainable=True` as materially different execution modes
- use `eval() + no_grad()` only when `trainable=False`
- preserve gradients when `trainable=True`
- move cleanly through runtime device placement, checkpointing, and distributed wrapping
- emit `space="feature"`
- expose the exact backbone, weights, feature layer, and preprocessing policy through provenance

The current `freeze` parameter will be replaced or redefined in a way that matches actual execution semantics. The important invariant is that a "trainable" configuration must genuinely participate in optimization.

## Training Modes

The pipeline must support two explicit fit modes.

### Offline Mode

Purpose:

- frozen or detached representations used as inputs to a separate normality model

Execution:

- protocol encodes batches with the configured representation
- outputs are explicitly detached when required by the target normality model
- batched outputs are unbound or otherwise collected into the normality model's fit input

Typical compatible methods:

- `PixelRepresentation` with `AutoEncoderNormality`
- frozen `FeatureRepresentation` with `FeatureMemoryNormality`

### Joint Mode

Purpose:

- end-to-end or partially trainable experiments where representation parameters and normality parameters are optimized together

Execution:

- protocol iterates per batch and passes `RepresentationBatch` directly into the normality model's training step
- gradients propagate through representation outputs when the representation is trainable
- optimizer construction and runtime wrapping treat representation and normality as trainable modules in one experiment definition

Typical compatible methods:

- future trainable feature + trainable normality combinations

The framework must reject any configuration that appears joint in intent but uses an offline-only normality model.

## Consumer Compatibility Contract

Every normality model must declare the representation contract it accepts.

Required declarations:

- supported representation spaces
- accepted tensor ranks
- fit mode: `offline` or `joint`
- whether detached representations are required

Validation must happen before meaningful training starts, ideally at runner setup or at the first batch boundary.

Examples:

- `FeatureMemoryNormality` accepts only `space="feature"` and detached feature tensors
- `AutoEncoderNormality` accepts only `space="pixel"` and image tensors shaped as batched spatial inputs
- `FeatureRepresentation(trainable=True)` combined with an offline-only consumer must raise an explicit configuration error

This validation replaces the current behavior where incompatible tensors can be processed accidentally.

## Runtime and Protocol Changes

The protocol and runner will become batch-first at the representation boundary.

Required changes:

- the protocol must call a batch encoding method instead of looping sample-by-sample through `representation(sample)`
- evaluation may still operate per sample after `RepresentationBatch.unbind()`, because artifacts and evidence remain sample-oriented
- setup must run compatibility checks between representation and normality before the training loop
- runtime helpers must move representation modules to device using normal `nn.Module` semantics rather than special casing around helper attributes

The intent is to make neural representations efficient and make future joint training possible without another architecture rewrite.

## Provenance, Logging, and Reporting

Representation provenance is part of the scientific record of each run.

Required outcomes:

- run metadata must include serialized representation provenance
- artifacts emitted by normality models must retain the `RepresentationOutput` information needed for downstream inspection
- reports and summaries must expose enough representation metadata to distinguish runs that differ in backbone, weights, feature layer, trainability, or preprocessing

At minimum, two runs with different representation semantics must no longer look identical in stored metadata simply because both were labeled `"feature"`.

## Migration Strategy

Migration will happen in three stages.

### Stage 1: Introduce the New Contract

- add the new representation data objects and abstract base
- keep old dict paths only long enough to migrate immediate consumers in one controlled refactor branch

### Stage 2: Migrate Producers and Consumers

- rewrite `PixelRepresentation` and `FeatureRepresentation`
- update protocol and runner to use batched representation execution
- migrate `FeatureMemoryNormality` and `AutoEncoderNormality` to validate and consume the new contract
- adapt artifacts flow and evidence entry points where needed

### Stage 3: Remove Legacy Paths

- delete support for representation dictionaries such as `representation["representation"]`
- remove weak compatibility assumptions from tests and helper functions
- make the typed contract the only valid representation API

The branch should not be merged in a half-migrated state where both APIs are treated as first-class.

## Error Handling and Fail-Fast Rules

The new system must reject invalid states early.

Mandatory failures:

- non-tensor or wrongly ranked image inputs at representation boundaries
- inconsistent batch metadata
- unsupported representation space for a given normality model
- missing required provenance fields
- trainable representation paired with offline-only normality configuration
- shape mismatches between representation output and consumer expectations

These failures must use explicit error messages that identify the incompatible component names and the violated contract.

## Testing and Acceptance Criteria

The refactor is complete only when the following are true.

### Contract Tests

- invalid representation spaces fail before fit or inference proceeds
- invalid tensor rank fails with component-specific messages
- missing provenance fields fail at the runner or consumer boundary

### Training Semantics Tests

- frozen feature representation emits outputs with `requires_grad=False`
- trainable feature representation emits outputs with `requires_grad=True`
- trainable feature representation parameters can be updated by an optimizer in a minimal integration test

### Batch Equivalence Tests

- batched encoding and repeated single-sample encoding agree on tensor values and metadata
- `RepresentationBatch.unbind()` produces valid `RepresentationOutput` objects for every sample

### Integration Tests

- `pixel + autoencoder` baseline still runs under the new contract
- `feature + feature_memory` baseline still runs under the new contract
- incompatible combinations fail explicitly

### Reproducibility and Reporting Tests

- representation provenance is present in run metadata and artifacts
- fixed configuration and seed produce stable representation metadata

## Risks and Trade-Offs

- The refactor will touch multiple subsystems and temporarily increase migration cost.
- Some current tests will need full replacement rather than patching because they assert the old weak contract.
- Joint training support may expose additional design issues in trainable normality models that were previously hidden by the offline-only pipeline.

These costs are acceptable because they pay down a structural research debt that would otherwise contaminate future baselines and ablations.

## Success Definition

This design succeeds when the representation layer is no longer a convenience wrapper, but a scientific interface with explicit semantics.

Concrete success conditions:

- trainable and frozen representations are both real, not simulated
- the pipeline encodes representation compatibility in code
- invalid experimental combinations are rejected early
- batch execution is standard for neural representations
- every run records enough representation provenance to be scientifically auditable
