# Research Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incrementally harden the framework from a prototype toward research-ready execution by tightening module contracts and orchestration reliability across the full `src/adrf` stack.

**Architecture:** Use small, layer-focused slices instead of a single giant rewrite. Protocol, normality, and evidence/evaluator/artifact contracts are already being hardened; the next major effort focuses on runner/runtime/launcher/logging reliability, then reporting/export boundaries, then data/representation and registry/config validation.

**Tech Stack:** Python 3.12, PyTorch, pytest, YAML configs, filesystem-backed run logging

---

## Program Tracks

### Track 1: Execution Reliability

**Files:**
- Modify: `src/adrf/runner/experiment_runner.py`
- Modify: `src/adrf/runner/launching.py`
- Modify: `src/adrf/ablation/runner.py`
- Modify: `src/adrf/benchmark/runner.py`
- Modify: `src/adrf/logging/run_logger.py`
- Test: `tests/test_official_contract.py`
- Test: `tests/test_ablation_runner.py`
- Test: `tests/test_benchmark_smoke.py`

- [ ] Write failing tests for launcher/runtime/logging failure modes
- [ ] Verify they fail for the right reason
- [ ] Implement minimal runner/launcher/logging fixes
- [ ] Re-run focused orchestration tests
- [ ] Commit the slice

### Track 2: Reporting And Export Contracts

**Files:**
- Modify: `src/adrf/reporting/summary.py`
- Modify: `src/adrf/statistics/table_export.py`
- Modify: `src/adrf/reporting/report.py`
- Test: reporting/statistics summary tests

- [ ] Write failing tests for summary/export contract drift
- [ ] Verify the failures
- [ ] Introduce explicit summary/export adapters or state objects where needed
- [ ] Re-run focused reporting tests
- [ ] Commit the slice

### Track 3: Data And Representation Boundaries

**Files:**
- Modify: `src/adrf/data/datamodule.py`
- Modify: `src/adrf/representation/base.py`
- Modify: `src/adrf/utils/runtime.py`
- Test: datamodule/representation/runtime tests

- [ ] Write failing tests for split/runtime boundary ambiguity
- [ ] Verify the failures
- [ ] Make split planning / runtime adapter boundaries more explicit
- [ ] Re-run focused data/representation tests
- [ ] Commit the slice

### Track 4: Registry And Config Validation

**Files:**
- Modify: `src/adrf/registry/registry.py`
- Modify: `src/adrf/utils/config.py`
- Test: registry/config tests

- [ ] Write failing tests for late config/registry failures
- [ ] Verify the failures
- [ ] Harden spec validation with minimal explicit checks
- [ ] Re-run focused config tests
- [ ] Commit the slice

## Current Completed Foundations

- protocol contract hardening
- normality training/runtime/representation hardening
- evidence/evaluator/artifact contract hardening

## Success Criteria

- standalone experiment execution and orchestrated benchmark/ablation execution use consistent runtime semantics
- critical layer boundaries are explicit rather than inferred from raw dicts or duck-typed attributes
- failures happen earlier and surface the real root cause
- the main research pipeline can run in a way that is stable enough for repeated comparative experiments
