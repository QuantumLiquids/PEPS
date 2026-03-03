# ExactSummationMeasurer Design

## Goal

`ExactSumMeasurerMPI` is a test-only utility that iterates over all configurations
to produce deterministic measurement results. It golden-regresses every observable
returned by each model's `EvaluateObservables()` against QuSpin ED benchmarks.

## API

Free function parallel to `ExactSumEnergyEvaluatorMPI`:

```cpp
template<typename MeasurementSolverT, typename TenElemT, typename QNT,
         template<typename, typename> class ContractorT = BMPSContractor>
ObservableMap<TenElemT> ExactSumMeasurerMPI(
    const SplitIndexTPS<TenElemT, QNT> &split_index_tps,
    const std::vector<Configuration> &all_configs,
    const typename ContractorT<TenElemT, QNT>::TruncateParams &trun_para,
    MeasurementSolverT &solver,
    size_t Ly, size_t Lx,
    const MPI_Comm &comm, int rank, int mpi_size);
```

Returns `ObservableMap<TenElemT>` with exact means (valid on master rank only).

## MPI Pattern

Same as `ExactSumEnergyEvaluatorMPI`:
1. Broadcast `SplitIndexTPS` from master to all ranks
2. Stripe-distribute configurations: rank processes configs `[rank, rank+mpi_size, ...]`
3. Per-config: construct `TPSWaveFunctionComponent`, call `EvaluateObservables()`
4. Accumulate `weight * obs_values` locally, then `MPI_Reduce` to master
5. Master divides by total weight to get exact means

## Observable Reduction

Since `ObservableMap` is `std::unordered_map` (iteration order undefined), reduction
uses a deterministic sorted key vector broadcast from rank 0:

1. Rank 0 builds sorted key list + value sizes from its local `ObservableMap`
2. Broadcast key count, key names, and sizes to all ranks (ensures idle ranks
   with zero configs can participate in collectives)
3. For each key in sorted order: `MPI_Reduce(local_buf, global_buf, n, MPI_SUM)`
   (idle ranks contribute zero buffers)
4. Reduce total weight: `MPI_Reduce(local_weight, total_weight, 1, MPI_SUM)`
5. Master divides: `result[key] = global_sum[key] / total_weight`

## Configuration Generation

TFIM needs `GenerateAllBinaryConfigs(Lx, Ly)` for full Hilbert space enumeration
(no conserved quantum number). Other models use existing `GenerateAllPermutationConfigs()`.

## Module Layout

| File | Purpose |
|------|---------|
| `include/qlpeps/algorithm/vmc_update/exact_summation_measurer.h` | `ExactSumMeasurerMPI` free function |
| `tests/test_algorithm/test_exact_summation_measurer.cpp` | Golden tests for all 4 OBC models |
| `tests/tools/quspin_exact_2x2_obc_benchmarks.py` | QuSpin ED benchmark script (validated) |
| `tests/tools/exact_2x2_obc_benchmarks.json` | QuSpin ED reference values |

## Models Covered (OBC, 2x2)

| Model | Config Generator | Hilbert Dim | Observable Keys |
|-------|-----------------|-------------|-----------------|
| Heisenberg | `GenerateAllPermutationConfigs({2,2})` | 6 | energy, spin_z, bond_energy_h/v, SzSz_all2all, SmSp_row, SpSm_row |
| TFIM | `GenerateAllBinaryConfigs()` | 16 | energy, spin_z, sigma_x, SzSz_row |
| Spinless fermion | `GenerateAllPermutationConfigs({2,2})` | 6 | energy, charge, bond_energy_h/v/dr/ur |
| t-J | `GenerateAllPermutationConfigs({1,1,2})` | 12 | energy, spin_z, charge, bond_energy_h/v/dr/ur |

## Golden Value Source

QuSpin ED on 2x2 OBC lattice (`tests/tools/quspin_exact_2x2_obc_benchmarks.py`).
All 4 models validated against analytical/reference energies to machine precision.

Note: These are ground-state observables. The test TPS data comes from simple update,
so TPS measurement values will differ from ED ground truth. The test golden values
are whatever the exact summation over the TPS wavefunction produces, not the
ED values. The ED values serve as a cross-check for physical reasonableness.

## Testing Strategy

1. Load existing simple-update TPS data (same as `test_exact_summation_evaluator.cpp`)
2. Call `ExactSumMeasurerMPI` with the measurement solver for each model
3. Golden-regress each observable value against hardcoded `constexpr` arrays
4. Use `kPrintGolden = true` pattern: first run prints values, then hardcode
5. For symmetry-constrained observables (spin_z in SU(2) models): use golden match,
   not near-zero assertion. Simple-update TPS may break SU(2), giving nonzero values.
6. For complex type: real parts must match golden, imaginary parts asserted near zero
   (exact summation with real Hamiltonians produces real observables)

## Key Reference Files

| What | Path |
|------|------|
| Existing exact evaluator (template) | `include/qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h` |
| Measurement solver base | `include/qlpeps/algorithm/vmc_update/model_measurement_solver.h` |
| Algorithm includes | `include/qlpeps/algorithm/algorithm_all.h` |
| Existing evaluator tests (pattern) | `tests/test_algorithm/test_exact_summation_evaluator.cpp` |
| Model: Spinless fermion | `include/qlpeps/algorithm/vmc_update/model_solvers/square_spinless_fermion.h` |
| Model: Heisenberg OBC | `include/qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_xxz_obc.h` |
| Model: TFIM OBC | `include/qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square_obc.h` |
| Model: t-J | `include/qlpeps/algorithm/vmc_update/model_solvers/square_tJ_model.h` |

## Constraints

- Header-only implementation (consistent with project convention)
- No new external dependencies
- Existing tests unchanged
- PBC models: lower priority, deferred to follow-up
