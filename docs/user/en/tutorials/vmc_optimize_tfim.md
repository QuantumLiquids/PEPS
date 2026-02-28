# VMC optimize (TFIM) tutorial

This tutorial runs VMC optimization for TFIM, starting from the Simple Update output (`./peps/`).

Source code:

- `examples/transverse_field_ising_vmc_optimize.cpp`

## What you’ll build

- A working example binary: `transverse_field_ising_vmc_optimize`
- An optimized `SplitIndexTPS` dump directory: `./optimized_tps/`

## Prerequisites

1. You have already run the Simple Update tutorial and have `./peps/`:
   - `simple_update_tfim.md`
2. See build notes: `../howto/build_and_link.md`

## Key concepts (state types)

In this repo, three state representations show up frequently:

- **PEPS**: a PEPS form that may carry explicit bond weights (typical Simple Update output).
- **TPS**: a tensor product state without explicit bond-weight tensors (used for “global” algorithms).
- **SplitIndexTPS**: a TPS variant where the physical index is split in advance for fast Monte Carlo projection / amplitude evaluation.

For Monte Carlo sampling APIs, `SplitIndexTPS` is the main workhorse.

Related reading:

- State conversions: `../howto/state_conversions.md`
- Glossary: `../reference/glossary.md`

## What VMC needs (inputs)

To run VMC you provide:

1. A model solver (Hamiltonian logic): TFIM uses `TransverseFieldIsingSquareOBC`.
2. A Monte Carlo updater + sampling parameters (how configurations evolve).
3. An optimizer algorithm + its parameters (SR / Adam / SGD / etc.).
4. Contraction parameters (`PEPSParams`, BMPS/TRG) controlling accuracy vs cost.

The example hard-codes these settings for a small 4×4 demo.

## Walkthrough: what the example does

The TFIM VMC example follows this sequence:

1. Load the `SquareLatticePEPS` dumped by Simple Update from `./peps/`.
2. Convert it to `SplitIndexTPS` via `ToSplitIndexTPS(...)` (`#include "qlpeps/api/conversions.h"`).
3. Construct:
   - `MonteCarloParams` (samples, warmup, sweeps-between-samples, initial `Configuration`, optional `config_dump_path`)
   - `PEPSParams` (BMPS truncation strategy/accuracy)
   - `OptimizerParams` (SR/SGD/Adam/etc; the TFIM example uses SR)
4. Bundle them into `VMCPEPSOptimizerParams` and run the one-call wrapper:
   - `auto result = VmcOptimize(params, sitps, MPI_COMM_WORLD, model, MCUpdateSquareNNFullSpaceUpdate{});`
5. Dump the optimized state:
   - `result.state.Dump(params.tps_dump_path);`

Note on performance hygiene:

- The example forces **one thread per MPI rank** via `hp_numeric::SetTensorManipulationThreads(1)` to avoid oversubscription on clusters.

## Steps

### 1) Build the examples (if you haven’t)

```bash
cd examples
mkdir -p build && cd build
cmake ..
make -j
```

### 2) Run VMC optimization with MPI

```bash
cd examples/build
mpirun -n 4 ./transverse_field_ising_vmc_optimize
```

The example is hard-coded to:

- lattice: 4×4
- local dim: 2
- field: `h = 0.5`

Expected output:

- An `examples/build/optimized_tps/` directory containing the optimized `SplitIndexTPS`.
- An optional `./vmc_configs/` directory if `MonteCarloParams.config_dump_path` is set in the example.

## Notes on parameters (quick intuition)

- `MonteCarloParams.total_samples`: total Monte Carlo samples across all MPI ranks per evaluation; the engine computes per-rank samples as `ceil(total_samples / mpi_size)`.
- `num_warmup_sweeps`: warm up the Markov chain before collecting samples.
- `sweeps_between_samples`: decorrelation between samples.

For the optimizer:

- SR (stochastic reconfiguration) solves a linear system each step (CG parameters matter).
- If the run is noisy/unstable, first increase contraction accuracy and warmup/samples.

### Parameter cheat sheet (as used by the example)

Monte Carlo parameters:

| Field | Meaning |
|---|---|
| `total_samples` | total number of samples across all ranks per evaluation |
| `num_warmup_sweeps` | warm-up sweeps before sampling |
| `sweeps_between_samples` | sweeps between collected samples |
| `initial_config` | initial configuration (random half-up/half-down for TFIM example) |
| `is_warmed_up` | whether the initial configuration is already equilibrated |
| `config_dump_path` | optional dump path for the final configuration |

SR (stochastic reconfiguration) uses CG parameters (`ConjugateGradientParams`) and SR parameters (`StochasticReconfigurationParams`):

| Field | Location | Meaning |
|---|---|---|
| `max_iter` | `ConjugateGradientParams` | maximum CG iterations |
| `relative_tolerance` | `ConjugateGradientParams` | CG residual tolerance (relative to initial residual norm) |
| `residual_recompute_interval` | `ConjugateGradientParams` | interval for recomputing exact residual (avoids drift) |
| `diag_shift` | `StochasticReconfigurationParams` | diagonal regularization for conditioning |

## Next steps

- Measure observables: `mc_measure_tfim.md`
