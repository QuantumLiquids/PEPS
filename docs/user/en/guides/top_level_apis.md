# PEPS Executors Tutorial

This tutorial presents the three high-level executors exposed by the library and shows how to use them correctly:

- Simple Update for imaginary-time projection on finite Square-Lattice PEPS
- VMC PEPS Optimizer Executor for variational Monte Carlo optimization
- Monte Carlo Measurement Executor for parallel observable measurements

The classes are header-only templates. Replace template parameters with the concrete types used in your project (tensor element type and quantum number type/symmetry).

## Prerequisites

- A compiled PEPS project (see the top-level README for building)
- Familiarity with `TPS`/`SplitIndexTPS` and basic MPI use
- Include convenience umbrella header when unsure: `qlpeps/algorithm/algorithm_all.h`

---

## Simple Update (imaginary-time projection)

- Header(s):
  - `qlpeps/algorithm/simple_update/simple_update.h` (abstract base)
  - `qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h` (nearest-neighbor model on the square lattice)
- Key classes:
  - `qlpeps::SimpleUpdateExecutor<TenElemT, QNT>` (abstract)
  - `qlpeps::SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>` (concrete)

### What it does

Performs Trotter steps with local truncation on a finite Square-Lattice PEPS by applying exp(-τ H_bond) gates and truncating bonds within `[Dmin, Dmax]` with target truncation error. Supports uniform nearest-neighbor terms plus optional uniform or site-dependent on-site terms.

### Important API

- Parameters: `SimpleUpdatePara { size_t steps, double tau, size_t Dmin, size_t Dmax, double Trunc_err }`
- Constructors (typical):
  - Uniform on-site term (optional):
    - `SquareLatticeNNSimpleUpdateExecutor(const SimpleUpdatePara&, const PEPST&, const Tensor& ham_nn, const Tensor& ham_onsite = Tensor())`
  - Non-uniform on-site terms:
    - `SquareLatticeNNSimpleUpdateExecutor(const SimpleUpdatePara&, const PEPST&, const Tensor& ham_nn, const TenMatrix<Tensor>& ham_onsite_terms)`
- Execution:
  - `void Execute()` builds evolution gates via `TaylorExpMatrix(tau, H_bond)` and sweeps vertical then horizontal bonds using `SquareLatticePEPS::NearestNeighborSiteProject`.
- Utilities:
  - `void ResetStepLenth(double tau)` updates τ and rebuilds gates on next execute
  - `const PEPST& GetPEPS() const`
  - `bool DumpResult(std::string path, bool release_mem)`
  - `double GetEstimatedEnergy() const`

### Notes and correctness details

- Gates are computed by Taylor expanding exp(-τ H) with index reordering that matches projection conventions implemented in `TaylorExpMatrix`.
- For models with on-site terms (e.g. TFIM, t–J with chemical potential), bond Hamiltonians are constructed as
  `H_ij = H_two_site + h_i ⊗ I + I ⊗ h_j` with boundary weights 0.5/0.375/0.25 as in the implementation.
- Diagnostics printed each sweep include middle-bond λ spectra, energy estimates from local expectation values and from norm decay, truncation error at the middle bond, and sweep timing.

### Minimal usage example

```cpp
#include "qlten/qlten.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"
using namespace qlten;
using namespace qlpeps;

using TenElemT = QLTEN_Double;     // or QLTEN_Complex
using QNT = qlten::special_qn::TrivialRepQN;  // Use TrivialRepQN for no symmetry (trivial quantum number)

using Tensor = QLTensor<TenElemT, QNT>;
using PEPST  = SquareLatticePEPS<TenElemT, QNT>;

// Build Hamiltonian pieces (user responsibility)
Tensor ham_nn = /* two-site operator with 4 legs (in-out per site) */;
Tensor ham_onsite = /* on-site operator with 2 legs (in-out) */; // optional

// Initial PEPS
PEPST peps_init(/* ly, lx, bond_dim, phys_dim, ... */);

// Parameters
SimpleUpdatePara su_para(/*steps=*/100, /*tau=*/1e-2, /*Dmin=*/8, /*Dmax=*/10, /*Trunc_err=*/1e-8);

// Executor and run
SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT> su(su_para, peps_init, ham_nn, ham_onsite);
su.Execute();

// Results
double e_est = su.GetEstimatedEnergy();
su.DumpResult("output/peps_", /*release_mem=*/false);
```

---

## VMC PEPS Optimizer (variational optimization)

- Header: `qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h`
- Base class (sampling core): `qlpeps::MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>`
- Executor: `qlpeps::VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>`

### What it does

Optimizes a TPS/PEPS wavefunction using Monte Carlo sampling of local energies and gradients. Supports line search schemes and iterative updates including Stochastic Reconfiguration (SR). Sampling is MPI-parallel; state updates happen on master then are broadcast to all ranks.

### Parameters

Use the consolidated parameter struct:

`VMCPEPSOptimizerParams { OptimizerParams optimizer_params; MonteCarloParams mc_params; PEPSParams peps_params; }`

- `OptimizerParams` controls update scheme, step lengths, and (for SR) conjugate gradient parameters.
- `MonteCarloParams` controls number of samples, warmup sweeps, sweeps between samples, and initial configuration.
- `PEPSParams` controls BMPS truncation and IO path (`wavefunction_path`).

Built-in Monte Carlo sweep updaters and model energy solvers are provided under:

- `qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h`
- `qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h`
- Exact summation (small systems) helper: `qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h`

### Execution flow (correctness-critical behavior)

1. Warm up Markov chains if needed, validate configurations across all ranks.
2. For each optimizer step, build an energy evaluator that:
   - broadcasts the current TPS state to all ranks,
   - updates the wavefunction component to reflect the new TPS,
   - normalizes TPS so wavefunction amplitude is O(1) across ranks,
   - runs `num_samples` sweeps, accumulating local energy and per-site gradient tensors,
   - gathers statistics (energy mean/error, mean gradient, and for SR the mean g-tensors) to master.
3. Master updates the state (line search or SR), validates, broadcasts, then continues.
4. Dumps current and best TPS, configurations, sample energies and trajectories.

### Minimal usage example

```cpp
#include "mpi.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
using namespace qlten;
using namespace qlpeps;

using TenElemT = QLTEN_Double;
using QNT      = qlten::special_qn::TrivialRepQN;

using MonteCarloSweepUpdater = MCUpdateSquareNNExchange<TenElemT, QNT>; // example updater
using EnergySolver          = ExactSummationEnergyEvaluator;            // or a model-specific solver

// Fill params appropriately
OptimizerParams opt_params;             // step_lengths, update_scheme, cg_params, ...
MonteCarloParams mc_params;             // num_samples, warmup, sweeps_between_samples, init config
PEPSParams peps_params;                 // BMPS truncate_para, wavefunction_path
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

// Initial state
SplitIndexTPS<TenElemT, QNT> tps_init(/* ly, lx */);

MPI_Comm comm = MPI_COMM_WORLD;
EnergySolver solver;

VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver> exe(params, tps_init, comm, solver);
exe.Execute();

// Access results
const auto &state_opt  = exe.GetOptimizedState();
const auto &state_best = exe.GetBestState();
double E_min           = exe.GetMinEnergy();
exe.DumpData("output/vmc_peps_", /*release_mem=*/false);
```

---

## Monte Carlo Measurement (observables)

- Header: `qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h`
- Base class (sampling core): `qlpeps::MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>`
- Executor: `qlpeps::MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>`

### What it does

Runs MPI-parallel Monte Carlo sampling to estimate energy, bond energies, one- and two-point functions, and short-time auto-correlations with error bars. Provides a replica test facility to probe ergodicity.

### Parameters and solvers

- `MCMeasurementParams` combines `MonteCarloParams` and `PEPSParams` for measurement runs.
- Registry-based API: implement
  `std::vector<ObservableMeta> DescribeObservables() const` and
  `ObservableMap<TenElemT> EvaluateObservables(const SplitIndexTPS<TenElemT,QNT>*, TPSWaveFunctionComponent<TenElemT,QNT>*)`.
  The executor invokes `EvaluateObservables` per sample and aggregates by observable `key`.
  Note: `psi_list` is only used internally as a transient value and is never dumped.
  The executor converts it to:
  - `psi_mean` (complex scalar): mean wavefunction amplitude
  - `psi_rel_err` (real scalar): relative radius defined as `radius_rel = max_i |psi_i - mean| / |mean|`.
  See the RFC “Observable Registry and Results Organization”.
  Built-in model measurement solvers are available under
  `qlpeps/algorithm/vmc_update/model_measurement_solver.h` and
  `qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h`.

### Execution flow and outputs

- `Execute()` warms up (if needed), measures sample-by-sample, then gathers statistics across MPI ranks.
- Dumps both binary and CSV outputs, including:
  - `energy_statistics` and `energy_statistics.csv` (energy and error bar),
  - `bond_energys.csv`, `one_point_functions.csv`, `two_point_functions.csv`,
  - raw samples per-rank in `energy_sample_data/`, `wave_function_amplitudes/`,
  - one-/two-point function samples as CSV in `one_point_function_samples/` and `two_point_function_samples/`.
- Supports emergency stop via `MPISignalGuard` to dump intermediate results safely.

### Minimal usage example

```cpp
#include "mpi.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
using namespace qlten;
using namespace qlpeps;

using TenElemT = QLTEN_Double;
using QNT      = qlten::special_qn::TrivialRepQN;

using MonteCarloSweepUpdater = MCUpdateSquareNNExchange<TenElemT, QNT>;
using MeasurementSolver      = SomeModelMeasurementSolver; // choose a concrete one

MCMeasurementParams meas_para; // set mc_samples, warmup, sweeps_between_samples, BMPS truncate

size_t ly = /*...*/, lx = /*...*/;
MPI_Comm comm = MPI_COMM_WORLD;
MeasurementSolver solver;

// Load TPS from file path or use in-memory TPS
SplitIndexTPS<TenElemT, QNT> tps(ly, lx);
tps.Load(meas_para.peps_params.wavefunction_path); // or load from your path

MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>
  meas(tps, meas_para, comm, solver);
meas.Execute();

auto [E, dE] = meas.OutputEnergy();
auto energy_estimate = meas.GetEnergyEstimate(); // query registry statistics by key, e.g., "energy"
// Registry CSVs are dumped under stats/<key>.csv
```

---

## Choosing an executor

- Use Simple Update to quickly obtain a projected PEPS state from a reasonable initial guess for local/NN models.
- Use VMC Optimizer to variationally refine TPS/PEPS with rigorous stochastic gradients and SR.
- Use MC Measurement to compute observables (with errors) for a given TPS/PEPS state.

## Common pitfalls

- Ensure Hamiltonian tensor index orders match the required conventions (see `TaylorExpMatrix` docs and examples).
- For fermionic systems, rely on the built-in parity operations and `CalGTenForFermionicTensors`; do not re-apply signs manually.
- Set `wavefunction_path` so that executors can dump/load TPS and configurations consistently.
- In MPI runs, avoid rank-dependent file paths except where documented (per-rank raw samples and configurations).


### State conversions (PEPS/TPS/SplitIndexTPS)

Use explicit named free functions in `qlpeps/api/conversions.h`.

```cpp
#include "qlpeps/api/conversions.h"
using qlten::special_qn::U1QN;

// PEPS -> TPS
auto tps = qlpeps::ToTPS<double, U1QN>(peps);

// TPS -> SplitIndexTPS
auto sitps = qlpeps::ToSplitIndexTPS<double, U1QN>(tps);

// PEPS -> SplitIndexTPS (direct)
auto sitps2 = qlpeps::ToSplitIndexTPS<double, U1QN>(peps);
```

Notes:
- Legacy interfaces `SquareLatticePEPS::operator TPS()` and `SplitIndexTPS(const TPS&)` are kept but marked deprecated. Prefer the explicit APIs above.


