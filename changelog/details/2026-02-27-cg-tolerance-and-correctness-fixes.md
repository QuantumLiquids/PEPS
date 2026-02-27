# CG Tolerance Semantics and Correctness Fixes

**Date**: 2026-02-27
**Type**: Breaking Change + Bug Fix

## Summary

Four correctness/safety issues fixed. The CG tolerance API changes from squared-residual
semantics to standard norm-space semantics, which is a source-breaking rename.

## Breaking Changes

### `ConjugateGradientParams::tolerance` renamed to `relative_tolerance`

The CG stopping criterion now uses norm-space semantics:

```
||r|| <= max(relative_tolerance * ||b||, absolute_tolerance)
```

Previously, the field `tolerance` was used in squared-residual form
(`tolerance * ||b||^2`), which was non-standard and easy to misuse.

**Migration**: rename the field and convert values via `new_value = sqrt(old_value)`.

| Old (squared-space) | New (norm-space) |
|---------------------|------------------|
| `1e-10`             | `1e-5`           |
| `1e-5`              | `3e-3`           |
| `1e-4`              | `1e-2`           |

Default changed from `1e-5` (squared) to `1e-4` (norm-space).

## Bug Fixes

1. **`normalize_update` double-sqrt**: `learning_rate /= std::sqrt(natural_grad_norm)` applied
   sqrt twice (once in computing `natural_grad_norm`, once in the division). Fixed to
   `learning_rate /= natural_grad_norm`, with a zero guard added.

2. **MPI CG `rkp1_2norm` uninitialized**: when `max_iter == 0`, the MPI master path left
   `rkp1_2norm` uninitialized, causing undefined behavior in the non-convergence return.
   Now initialized to `rk_2norm`.

3. **Empty `Ostar_samples` dereference**: the non-selector SR path silently fell back to
   empty vectors when `Ostar_samples`/`Ostar_mean` were null, leading to a zero-size
   S-matrix and wrong results. Now throws `std::invalid_argument` early.
