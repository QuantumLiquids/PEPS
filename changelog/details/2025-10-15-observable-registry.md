# Observable Registry & Psi Summary Integration

**Date**: 2025-10-15  
**Type**: Enhancement  
**Status**: COMPLETED  
**Commits**: 87e59ee, 15c91d0, a970c67, d38abc8, 892ff49

## Executive Summary

We replaced the bespoke Monte Carlo measurer dump pipeline with a registry-based observable API, aligning implementation with RFC `2025-09-11-observable-registry-and-results-organization`. The rewrite introduces stable per-key metadata (`ObservableMeta`), first-class `psi` sample summaries, and precise CSV dumps while simplifying solver responsibilities.

## Key Data Contracts

- `ObservableMeta`: runtime shape/index labels supplied per solver via `DescribeObservables(size_t ly, size_t lx)`.
- `ObservableMap<T>`: flat value vectors returned per key for each sample.
- `PsiSummary`: sample-level amplitude mean + relative spread returned outside the registry (`samples/psi.csv`).

## Major Code Changes

1. **Measurement Base Layer**
   - `ModelMeasurementSolver` now caches `PsiSummary` and exposes `ComputePsiSummary()` helpers.
   - `SquareNNNModelMeasurementSolver` and specialisations emit registry keys with explicit shapes.

2. **`MCPEPSMeasurer` Rewrite**
   - Registry-aware sample buffering + MPI aggregation (`registry_stats_`).
   - CSV dump uses metadata for layout validation and naming.
   - Added high-precision CSV writers and per-sample `psi.csv` emitter.

3. **Model Solver Updates**
   - All square-lattice solvers (Hubbard, t-J, XXZ variants, transverse Ising, etc.) implement the new metadata contracts and registry evaluation.
   - Added superconducting bond order support conditioned on solver flags.

4. **Test & Tooling Refresh**
   - Rebuilt `test_mc_peps_measure.cpp` to assert on registry outputs rather than legacy structs.
   - Added QuSpin ED script `tests/tools/tJ_OBC.py` for deterministic reference data.

5. **Documentation**
   - Updated developer guide `docs/dev/guides/custom_measurement_solver_guide.md` and user docs to document registry keys, metadata shape conventions, and dump formats.
   - RFC marked ready for closure; validation log recorded under `docs/dev/testing/2025-10-15-observable-registry-validation.md`.

## Testing

- `ctest --test-dir build -R mc_peps_measure`
  - `test_mc_peps_measure_double`
  - `test_mc_peps_measure_complex`
Passed.

## Impact Assessment

- ✅ Stable CSV/registry interface with explicit metadata, removing shape guesswork.
- ✅ Psi consistency warnings clarified and limited to configured thresholds; results dumped per sample.
- ✅ Legacy `ObservablesLocal` code paths removed, reducing complexity and eliminating silent fallback bugs.
- ⚠️ Downstream data processing code must migrate to registry lookups; compatibility shims retained only for energy access.

## Follow-ups

- Implement SE binning automation per RFC `2025-09-11-mc-binning-ips-se-estimation.md`.
- Add contract tests ensuring `DescribeObservables()` shapes match runtime data for every solver (scaffold already in tests).

