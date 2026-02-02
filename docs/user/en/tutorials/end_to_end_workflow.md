# End-to-end workflow: Simple Update → VMC → Measurement

This page describes the recommended **end-to-end workflow** in this repo:

1. Use **Simple Update** to get a reasonable initial PEPS (fast, no MPI).
2. Convert to `SplitIndexTPS` (the representation used for Monte Carlo sampling).
3. Run **VMC optimization** (MPI-parallel sampling, SR/first-order optimizers).
4. Run **Monte Carlo measurement** on the optimized state.

## Scope

- Finite-size square lattice workflows are the most complete today (especially OBC/BMPS).
- The model solver and Monte Carlo updater are pluggable as long as they satisfy the required interfaces.
- Supports single- or multi-process MPI runs (sampling is embarrassingly parallel).

If you just want to run the provided TFIM demo, follow the tutorials:

- Simple Update (TFIM): `simple_update_tfim.md`
- VMC optimize (TFIM): `vmc_optimize_tfim.md`
- Monte Carlo measurement (TFIM): `mc_measure_tfim.md`

## Step-by-step (what each phase produces)

### Step 1: Simple Update (initial state)

- Output: a PEPS dump directory (example uses `./peps/`).
- Purpose: generate a “usable” initial state quickly; VMC typically improves energy significantly afterwards.

### Step 2: Convert to `SplitIndexTPS` (for Monte Carlo)

- Recommended: use explicit conversion functions (`#include "qlpeps/api/conversions.h"`).
- Recommended in practice: normalize/scale site tensors so typical amplitudes in the sampled sector are \(O(1)\).

Related: `../howto/state_conversions.md`.

### Step 3: VMC optimization

- Input: `SplitIndexTPS`, model energy solver, updater, and parameter structs:
  - `OptimizerParams`, `MonteCarloParams`, `PEPSParams`, bundled into `VMCPEPSOptimizerParams`.
- Output: optimized `SplitIndexTPS` on disk (example uses `./optimized_tps/`).

Related: `../explanation/optimizer_algorithms.md`, `../howto/choose_mc_updater.md`, `../howto/data_persistence.md`.

### Step 4: Monte Carlo measurement

- Input: optimized `SplitIndexTPS` + a measurement solver (often the same model class also implements measurement).
- Output: aggregated CSV stats under `measurement_data_dump_path` (example uses `./mc_measure_output/stats/`).

Related: `../reference/model_observables_registry.md`.

## Outputs (default example paths)

The TFIM examples use these default directories:

- Simple Update dumps PEPS to `./peps/`
- VMC optimization dumps optimized `SplitIndexTPS` to `./optimized_tps/`
- Measurement dumps aggregated CSV stats to `./mc_measure_output/stats/`

## Common pitfalls

- **Amplitude scale issues** after conversion: if amplitudes are extremely small/large, sampling becomes numerically fragile.
- **Too-aggressive contraction truncation**: large psi-consistency warnings usually mean you need higher BMPS/TRG accuracy.
- **Updater/solver mismatch**: updater enforces a conserved sector but solver assumes full space (or vice versa).

## Related

- State conversions: `../howto/state_conversions.md`
- Top-level APIs: `../howto/top_level_apis.md`
- Data persistence and dump paths: `../howto/data_persistence.md`
