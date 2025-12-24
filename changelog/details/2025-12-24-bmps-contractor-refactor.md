# BMPS/TRG Contractor Refactor (TensorNetwork2D Split)

**Date**: 2025-12-24  
**Type**: Refactor  
**Status**: COMPLETED  
**Scope**: `include/qlpeps/two_dim_tn/tensor_network_2d/*`

## Summary

- `TensorNetwork2D` is now a **data container** for a 2D finite-size tensor network grid.
- BMPS algorithm state (boundary MPS, boundary tensors, growth/move/trace/punch-hole) is moved to `BMPSContractor`.
- TRG-related logic is implemented in `TRGContractor`.

## Compatibility Notes

- This is an **intentional breaking API change**: contraction/workspace methods previously on `TensorNetwork2D`
  are no longer available there. Callers must use `BMPSContractor` / `TRGContractor`.
- Numerical results may change in corner cases due to bug fixes (e.g. BMPS growth fallthrough fix).

## Testing

- `ctest --test-dir build` (non-MPI tests) passed.
- `test_tJ_model_solver` is currently skipped until valid `SplitIndexTPS` test data is added.


