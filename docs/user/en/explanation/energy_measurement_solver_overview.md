# Energy and measurement solver overview

Use this page to understand how “energy solvers” and “measurement solvers” fit into the VMC + measurement pipeline.

## Backend note (OBC/BMPS vs PBC/TRG)

- OBC workflows use BMPS contraction.
- PBC workflows use TRG contraction.

In the public wrappers (`VmcOptimize` / `MonteCarloMeasure`), the boundary condition of `SplitIndexTPS` is cross-checked against `PEPSParams` and mismatches are rejected.

## Energy solver (VMC)

An energy solver is responsible for:

- computing the local energy for a Monte Carlo configuration, and
- (optionally) computing gradient-related tensors (“holes”) for optimization.

It is passed as the `EnergySolver` template parameter to `VMCPEPSOptimizer` / `VmcOptimize(...)`.

## Measurement solver (measurement)

A measurement solver is responsible for:

- computing observables for each Monte Carlo configuration, and
- exposing metadata via `DescribeObservables()` (registry keys + shapes).

It is passed to `MCPEPSMeasurer` / `MonteCarloMeasure(...)`.

## Related

- Math and conventions (complex gradients): `model_energy_solver_math.md`
- Write a custom energy solver: `../howto/write_custom_energy_solver.md`
- Registry keys (reference): `../reference/model_observables_registry.md`
