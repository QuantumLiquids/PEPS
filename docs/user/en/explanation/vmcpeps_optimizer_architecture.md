# VMCPEPSOptimizer: architecture and extension points

## What it is

`VMCPEPSOptimizer` is the execution engine for **variational Monte Carlo (VMC) optimization** of PEPS/TPS states in this repository.
It is designed to be small and predictable: you plug in *strategy components*, and the executor orchestrates them.

At a high level, every VMC run combines three ideas:

1. **How configurations evolve** (a Monte Carlo sweep updater)
2. **How energy/gradients are evaluated** (a model energy solver)
3. **How parameters are updated** (an optimizer algorithm + its params)

The main user-facing entry points are:

- One-call wrapper: `VmcOptimize(...)` in `include/qlpeps/api/vmc_api.h`
- The underlying class template: `VMCPEPSOptimizer<...>` in `include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h`

## Mental model (what happens during a run)

The execution loop is conceptually:

1. **Initialize** a `SplitIndexTPS` state and an initial `Configuration`.
2. For each optimization iteration:
   1. Each MPI rank runs Monte Carlo sweeps to generate samples (independent Markov chains).
   2. For each sample, the **energy solver** computes:
      - local energy \(E_{\mathrm{loc}}(S)\), and
      - optionally “holes” for gradient/SR evaluation.
   3. Sample statistics are **aggregated across MPI ranks**.
   4. The **optimizer algorithm** updates the state parameters (SR / Adam / SGD / …).
   5. The updated state is made available to all ranks for the next iteration.

Two practical implications:

- Contraction truncation (BMPS/TRG) can introduce “psi-consistency” mismatch across contraction positions. When warnings show up, increase contraction accuracy. See: `model_energy_solver_math.md`.
- Numerical stability often improves after conversion with light normalization/scaling. See: `../howto/state_conversions.md`.

## Component contracts (what you are expected to provide)

### 1) Monte Carlo sweep updater

The updater is a functor responsible for proposing/accepting configuration updates and keeping cached data consistent.

Contract (conceptual responsibilities):

- Update the configuration (`tps_component.config`).
- Update the cached wavefunction amplitude (`tps_component.amplitude`) and any cached contraction objects the framework expects.
- Maintain detailed balance for correct sampling (and be ergodic in the target sector).
- Write acceptance-rate diagnostics.

Start here:

- Choosing an updater: `../howto/choose_mc_updater.md`
- Writing a custom updater (PXP example): `../howto/write_mc_updater_pxp.md`
- Top-level APIs: `../howto/top_level_apis.md`

### 2) Model energy solver

The model energy solver owns Hamiltonian logic and the “local-energy convention” (especially important for complex-valued wavefunctions).

Start here:

- Math/conventions: `model_energy_solver_math.md`
- Writing a custom solver: `../howto/write_custom_energy_solver.md`

### 3) Optimizer algorithm + parameters

The optimizer is selected by `OptimizerParams` (SR / Adam / SGD / AdaGrad / L-BFGS, plus learning-rate schedulers and optional gradient preprocessing).

Start here:

- Algorithms and math: `optimizer_algorithms.md`
- Parameter setup: `../howto/set_optimizer_parameter.md`

## MPI semantics (what to expect)

Sampling is embarrassingly parallel, but the run is not “purely independent ranks”:

- Each rank runs its own Markov chain and contributes samples.
- Aggregation is collective (energy/gradient estimates require cross-rank reduction).
- State updates are coordinated so that all ranks continue from the same updated wavefunction.

Practical note for clusters:

- Avoid thread oversubscription: in MPI examples we force one thread per rank via `hp_numeric::SetTensorManipulationThreads(1)`.

## Where to start (for users)

- End-to-end: `../tutorials/end_to_end_workflow.md`
- Tutorials: `../tutorials/index.md`
- How-to: `../howto/index.md`
- Reference: `../reference/index.md`
