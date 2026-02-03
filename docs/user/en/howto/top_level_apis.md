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

## Related

- End-to-end workflow: `../tutorials/end_to_end_workflow.md`
- Choose an updater: `choose_mc_updater.md`
- State conversions: `state_conversions.md`
