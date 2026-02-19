# Top-level APIs (Simple Update / VMC / Measurement)

This repository is primarily used in two pipelines:

- **VMC optimization**: `VmcOptimize(...)`
- **Monte Carlo measurement**: `MonteCarloMeasure(...)`

Both wrappers live in:

- `include/qlpeps/api/vmc_api.h`

Simple Update is implemented as executors under:

- `include/qlpeps/algorithm/simple_update/`

## What these APIs do (and do not do)

- `VmcOptimize(...)` and `MonteCarloMeasure(...)` operate on an in-memory `SplitIndexTPS`.
- They **do not** load TPS from disk for you. Loading is a method on the state type (`SplitIndexTPS::Load`).
- Output dumping is explicit via params (`tps_dump_path`, `measurement_data_dump_path`, `config_dump_path`).

## Backend (OBC/BMPS vs PBC/TRG)

Both wrappers infer the backend from:

- `sitps.GetBoundaryCondition()` (OBC vs PBC)

and **fail fast** if it conflicts with the truncation backend carried by your params:

- OBC requires BMPS truncation params in `PEPSParams`
- PBC requires TRG truncation params in `PEPSParams`

This is intentional: it prevents “silently running the wrong backend”.

## VMC optimization: `VmcOptimize(...)`

Minimal skeleton:

```cpp
#include <mpi.h>
#include "qlten/qlten.h"
#include "qlpeps/api/vmc_api.h"                  // VmcOptimize
#include "qlpeps/optimizer/optimizer_params.h"   // OptimizerFactory, ConjugateGradientParams
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h" // MCUpdateSquareNNFullSpaceUpdate

using TenElemT = qlten::QLTEN_Double;
using QNT = qlten::special_qn::TrivialRepQN;

// sitps: SplitIndexTPS<TenElemT, QNT> (already constructed)
// solver: a model energy solver instance (e.g. TransverseFieldIsingSquareOBC(h))

MonteCarloParams mc_params(
    /*total_samples=*/500,
    /*num_warmup_sweeps=*/200,
    /*sweeps_between_samples=*/2,
    /*initial_config=*/Configuration(/*Ly=*/4, /*Lx=*/4),
    /*is_warmed_up=*/false,
    /*config_dump_path=*/"");

auto trunc = BMPSTruncateParams<double>::SVD(/*D_min=*/2, /*D_max=*/8, /*trunc_err=*/1e-14);
PEPSParams peps_params(trunc);

auto opt_params = OptimizerFactory::CreateStochasticReconfiguration(
    /*max_iterations=*/40,
    ConjugateGradientParams(/*max_iter=*/100, /*tolerance=*/1e-5, /*restart=*/20, /*diag_shift=*/1e-3),
    /*learning_rate=*/0.1);

VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, /*tps_dump_path=*/"./optimized_tps");

auto result = qlpeps::VmcOptimize(params, sitps, MPI_COMM_WORLD, solver,
                                 MCUpdateSquareNNFullSpaceUpdate{});
```

## Monte Carlo measurement: `MonteCarloMeasure(...)`

Minimal skeleton:

```cpp
#include <mpi.h>
#include "qlten/qlten.h"
#include "qlpeps/api/vmc_api.h" // MonteCarloMeasure
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h" // MCUpdateSquareNNFullSpaceUpdate

MCMeasurementParams meas_params(
    /*mc_params=*/mc_params,
    /*peps_params=*/peps_params,
    /*measurement_data_dump_path=*/"./mc_measure_output");

auto meas = qlpeps::MonteCarloMeasure(sitps, meas_params, MPI_COMM_WORLD, solver,
                                     MCUpdateSquareNNFullSpaceUpdate{});
```

Notes:

- Observable shapes/keys are defined by each model’s `DescribeObservables()` metadata.
- See registry keys: `../reference/model_observables_registry.md`.

## State I/O: load from disk (explicit)

Loading is a method on the state:

```cpp
SplitIndexTPS<TenElemT, QNT> sitps;
if (!sitps.Load("./optimized_tps")) {
  throw std::runtime_error("Failed to load SplitIndexTPS.");
}
```

## Simple Update (executor API)

There is no one-call wrapper like `VmcOptimize(...)` for Simple Update today. You use the executor directly.

Example reference implementation:

- `examples/transverse_field_ising_simple_update.cpp`

Main include (square lattice NN Simple Update):

- `qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h`

Parameter modes:

1. Simple fixed-step mode (backward-compatible behavior):

```cpp
SimpleUpdatePara su_para(
    /*steps=*/100,
    /*tau=*/0.05,
    /*Dmin=*/1,
    /*Dmax=*/4,
    /*Trunc_err=*/1e-14);
```

2. Advanced automatic-stop mode (opt-in):

```cpp
auto su_para = SimpleUpdatePara::Advanced(
    /*steps=*/1000,      // hard cap, still enforced
    /*tau=*/0.05,
    /*Dmin=*/1,
    /*Dmax=*/4,
    /*Trunc_err=*/1e-14,
    /*energy_abs_tol=*/1e-8,
    /*energy_rel_tol=*/1e-10,
    /*lambda_rel_tol=*/1e-6,
    /*patience=*/3,      // consecutive sweeps required
    /*min_steps=*/10);   // do not stop before this many sweeps
```

Advanced stop semantics:

- Convergence gate is `energy AND lambda`.
- Energy uses hybrid tolerance:
  `|ΔE| <= max(energy_abs_tol, energy_rel_tol * max(1, |E_prev|, |E_curr|))`.
- Lambda uses per-bond relative L2 diagonal drift, then takes the global maximum.
- If bond dimensions change between two sweeps, lambda drift is skipped and convergence streak resets.

Run summary getters:

```cpp
executor.Execute();

bool converged = executor.LastRunConverged();
size_t sweeps_done = executor.LastRunExecutedSteps();
auto summary = executor.GetLastRunSummary();
```

## Loop Update (executor API)

Loop update is also executor-based. Existing fixed-step usage stays unchanged.

Main include:

- `qlpeps/algorithm/loop_update/loop_update.h`

Fixed-step mode (backward-compatible behavior):

```cpp
LoopUpdatePara lu_para(
    /*truncate_para=*/loop_truncate_para,
    /*steps=*/50,
    /*tau=*/0.01,
    /*gate_type=*/LoopGateType::kFirstOrder);
```

Advanced automatic-stop mode (opt-in):

```cpp
auto lu_para = LoopUpdatePara::Advanced(
    /*truncate_para=*/loop_truncate_para,
    /*steps=*/1000,      // hard cap, still enforced
    /*tau=*/0.01,
    /*energy_abs_tol=*/1e-8,
    /*energy_rel_tol=*/1e-10,
    /*lambda_rel_tol=*/1e-6,
    /*patience=*/3,      // consecutive sweeps required
    /*min_steps=*/10,    // do not stop before this many sweeps
    /*gate_type=*/LoopGateType::kFirstOrder);
```

Advanced stop semantics:

- Convergence gate is `energy AND lambda`.
- Energy uses hybrid tolerance:
  `|ΔE0| <= max(energy_abs_tol, energy_rel_tol * max(1, |E_prev|, |E_curr|))`.
- Lambda uses per-bond relative L2 diagonal drift, then takes the global maximum.
- If bond dimensions change between two sweeps, lambda drift is skipped and convergence streak resets.

Run summary getters:

```cpp
executor.Execute();

bool converged = executor.LastRunConverged();
size_t sweeps_done = executor.LastRunExecutedSteps();
auto summary = executor.GetLastRunSummary();
```

Per-step observability:

```cpp
const auto &metrics = executor.GetStepMetrics();  // vector<LoopUpdateStepMetrics<RealT>>
```

Optional step callback and machine-readable logs:

```cpp
lu_para.step_observer = [](const LoopUpdateStepMetrics<double> &m) {
  // consume metrics
};
lu_para.emit_machine_readable_metrics = true;
```

When machine-readable logs are enabled, one line is emitted per sweep:

```text
LU_METRIC step=<i> tau=<tau> e0=<estimated_e0> en=<estimated_en> trunc_err=<value-or-N/A> elapsed_sec=<t>
```

Defaults:

- Advanced stop is disabled unless `advanced_stop` is explicitly configured.
- Existing fixed-step behavior (`steps` hard cap) is unchanged by default.

## Related

- End-to-end workflow: `../tutorials/end_to_end_workflow.md`
- Choose an updater: `choose_mc_updater.md`
- State conversions: `state_conversions.md`
