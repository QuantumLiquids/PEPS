# Optimizer Readability Refactor (Periodic Selector API + LR Trajectory Rename)

**Date**: 2026-02-24  
**Type**: Breaking Change  
**Status**: COMPLETED  

## Summary

This refactor improves optimizer readability and removes dead API surface while preserving optimizer runtime behavior.

## Intentional Source-Level API Changes

1. Removed `Optimizer::BoundedGradientUpdate(...)` (orphaned API, not used by `IterativeOptimize` dispatch).
2. Renamed `OptimizerParamsBuilder::SetAutoStepSelector(...)` to `SetPeriodicStepSelector(...)`.
3. Renamed `OptimizationResult::step_length_trajectory` to `learning_rate_trajectory`.

## Migration Guide

- Replace `.SetAutoStepSelector(...)` with `.SetPeriodicStepSelector(...)`.
- Replace `result.step_length_trajectory` with `result.learning_rate_trajectory`.
- Remove direct calls to `BoundedGradientUpdate(...)` (no replacement API).

## Scope Notes

- `SITPST` was removed from `optimizer_impl.h` and replaced with `WaveFunctionT` for implementation readability.
- Public compatibility alias `SITPST` in `optimizer.h` remains intentionally available.

## Verification Snapshot

- `ctest -R test_optimizer_double --output-on-failure`
- `ctest -R "test_optimizer_params|test_auto_step_selector_policy" --output-on-failure`
