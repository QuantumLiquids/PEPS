# Monte Carlo Measurement (TFIM) — measure the TPS produced by VMC

This tutorial shows how to **measure observables** for the transverse-field Ising model (TFIM) using Monte Carlo sampling, starting from the optimized state produced by:

- `examples/transverse_field_ising_vmc_optimize.cpp`

We will load the dumped `SplitIndexTPS` from disk and run `MonteCarloMeasure(...)`.

## Prerequisites

1. You have already run:
   - `examples/transverse_field_ising_simple_update.cpp` (produces a `peps/` directory)
   - `examples/transverse_field_ising_vmc_optimize.cpp` (produces an `optimized_tps/` directory)
2. You can build and run MPI programs on your machine / cluster.

## What you will get

The measurer dumps CSV results to:

- `./mc_measure_output/stats/`

For TFIM the built-in solver registers (at least) these keys:

- `energy` → `stats/energy.csv` (scalar, written as a flat table)
- `spin_z` → `stats/spin_z_mean.csv`, `stats/spin_z_stderr.csv` (Ly×Lx)
- `sigma_x` → `stats/sigma_x_mean.csv`, `stats/sigma_x_stderr.csv` (Ly×Lx)
- `SzSz_row` → `stats/SzSz_row.csv` (flat; middle-row correlations)

> Tip: registry keys are the authoritative interface. See `docs/user/en/reference/model_observables_registry.md`.

## Example program

This repo provides a matching example:

- `examples/transverse_field_ising_mc_measure.cpp`

It:
- loads `SplitIndexTPS` from `./optimized_tps` (or a path you pass),
- builds `MCMeasurementParams`,
- runs `MonteCarloMeasure(...)` with `TransverseFieldIsingSquareOBC(h)` and `MCUpdateSquareNNFullSpaceUpdate`,
- dumps the results to `./mc_measure_output/`.

## Build

Build the examples as a small standalone CMake project:

```bash
cd examples
mkdir -p build && cd build
cmake ..
make -j
```

## Run

Run with MPI (each rank runs an independent Markov chain; statistics are aggregated):

```bash
cd examples/build
mpirun -n 4 ./transverse_field_ising_mc_measure
```

### Choosing which state to measure

If you also saved the “best state” during VMC (e.g. via optimizer `DumpData()`), measure that directory first.

For the provided VMC example, the default output directory is:

- `./optimized_tps`

To measure a different directory, pass it as the first argument:

```bash
mpirun -n 4 ./transverse_field_ising_mc_measure /path/to/your/sitps_dir
```

## Notes / pitfalls

- **Lattice size must match**: `Configuration(Ly,Lx)` must match the loaded TPS dimensions.
- **Warmup matters**: if you see unstable observables, increase `num_warmup_sweeps` and/or `total_samples`.
- **Output location**: look under `mc_measure_output/stats/` (master rank writes the aggregated stats).
