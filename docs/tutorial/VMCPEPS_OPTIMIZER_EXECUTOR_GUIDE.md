# VMCPEPSOptimizerExecutor Comprehensive Guide

## Overview

`VMCPEPSOptimizerExecutor` is the **unified execution engine** for variational Monte Carlo optimization of PEPS (Projected Entangled Pair States). It orchestrates three fundamental components through a clean, template-based architecture that eliminates complexity rather than managing it.

**Core Design Philosophy**: One executor, three strategies, zero special cases.

## Architecture: Three-Strategy Composition

### The Essential Components

```cpp
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
class VMCPEPSOptimizerExecutor
```

The executor combines exactly three strategic components:

1. **Monte Carlo Updater** - How configurations evolve
2. **Energy Solver** - How energies and gradients are computed  
3. **Optimizer Algorithm** - How parameters are updated

**Design Insight**: Each component has a single responsibility. The executor simply orchestrates their interaction without complex conditional logic.

---

## Component 1: Monte Carlo Updater

### Concept: Configuration Evolution Strategy

The Monte Carlo Updater defines **how particle configurations change** during sampling. It's a functor that updates configurations while maintaining detailed balance for correct statistical sampling.

### Why This Matters

In VMC optimization, we need to sample different particle configurations to estimate:
- Energy expectation values: ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
- Gradient information: ∂⟨E⟩/∂θ for parameter updates

The updater strategy determines sampling efficiency and ergodicity.

### Built-in Updater Types

Located in: `include/qlpeps/vmc_basic/configuration_update_strategies/`

#### 1. Nearest-Neighbor Exchange (`MCUpdateSquareNNExchange`)
```cpp
// Suitable for: Heisenberg models, t-J model with particle number conservation
using UpdaterType = MCUpdateSquareNNExchange;
```

**Algorithm**: Randomly select neighboring site pairs, propose particle exchange, accept/reject based on amplitude ratio.

**Use case**: 
- Spin-1/2 Heisenberg model with U(1) symmetry
- t-J model where particle number is conserved
- Any model with exchange-based dynamics

#### 2. Full Configuration Space (`MCUpdateSquareNNFullSpaceUpdate`)
```cpp
// Suitable for: Models without strict conservation laws
using UpdaterType = MCUpdateSquareNNFullSpaceUpdate;
```

**Algorithm**: For each bond, consider all possible local configurations, sample according to probability weights.

**Use case**:
- Models without particle number conservation
- Bose-Hubbard models
- Systems where local quantum number can change arbitrarily

#### 3. Three-Site Triangular Update (`MCUpdateSquareTNN3SiteExchange`)  
```cpp
// Suitable for: Frustrated systems, triangular plaquette dynamics
using UpdaterType = MCUpdateSquareTNN3SiteExchange;
```

**Algorithm**: Update three sites in triangular arrangements, useful for frustrated systems.

**Use case**:
- Triangular lattice models
- Systems with frustration requiring three-site moves
- J1-J2 Heisenberg models on square lattice

### Updater Interface Contract

All updaters must implement:
```cpp
template<typename TenElemT, typename QNT>
void operator()(const SplitIndexTPS<TenElemT, QNT>& sitps,
                TPSWaveFunctionComponent<TenElemT, QNT>& tps_component,
                std::vector<double>& accept_rates);
```

**Responsibilities**:
- Update `tps_component.config` (particle configuration)
- Update `tps_component.amplitude` (wave function amplitude)
- Record acceptance rates for diagnostics
- Maintain detailed balance for correct sampling

### Choosing the Right Updater

**Decision Tree**:
```
Does your model conserve particle number?
├── Yes → Do you need only nearest-neighbor moves?
│   ├── Yes → MCUpdateSquareNNExchange (most efficient)
│   └── No → MCUpdateSquareTNN3SiteExchange (for frustration)
└── No → MCUpdateSquareNNFullSpaceUpdate (full configuration space)
```

---

## Component 2: Model Energy Solver

### Concept: Energy and Gradient Computation Engine

The Model Energy Solver calculates **local energies and gradient information** for specific particle configurations and TPS states. It encapsulates all model-specific Hamiltonian details.

### Why This Matters

VMC optimization requires:
- **Local Energy**: H_loc = ⟨config|H|ψ⟩ / ⟨config|ψ⟩ for each sampled configuration
- **Gradient Information**: ∂ln|ψ⟩/∂θ_i (logarithmic derivatives) for parameter updates
- **Hole Tensors**: For efficient gradient computation in TPS framework

The energy solver hides all the complex tensor network contractions behind a clean interface.

### Base Interface: CRTP Pattern

Located in: `include/qlpeps/algorithm/vmc_update/model_energy_solver.h`

```cpp
template<typename ConcreteModelSolver>
class ModelEnergySolver {
public:
  template<typename TenElemT, typename QNT, bool calchols>
  TenElemT CalEnergyAndHoles(
      const SplitIndexTPS<TenElemT, QNT>* sitps,
      TPSWaveFunctionComponent<TenElemT, QNT>* tps_sample,
      TensorNetwork2D<TenElemT, QNT>& hole_res
  );
};
```

**Design Pattern**: Uses CRTP (Curiously Recurring Template Pattern) for static polymorphism. Each concrete solver implements `CalEnergyAndHolesImpl()`.

### Built-in Energy Solvers

Located in: `include/qlpeps/algorithm/vmc_update/model_solvers/`

#### 1. Square Lattice XXZ Models
```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_xxz_model.h"
using SolverType = SquareSpinOneHalfXXZModel;
```

**Hamiltonian**: H = J_z ∑⟨i,j⟩ S^z_i S^z_j + J_{xy} ∑⟨i,j⟩ (S^x_i S^x_j + S^y_i S^y_j)
**Features**: XXZ nearest-neighbor interactions, optional magnetic field
**Quantum Numbers**: Optional no symmetry or U(1) symmetry

#### 2. Triangular Lattice Models
```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_triangle_heisenberg_sqrpeps.h"
using SolverType = SpinOneHalfTriangleHeisenbergSquarePEPS;
```

**Hamiltonian**: Heisenberg model on triangular lattice geometry
**Features**: Handles complex triangle-square mapping for PEPS

#### 3. J1-J2 Models
```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_triangle_heisenbergJ1J2_sqrpeps.h"
using SolverType = SpinOneHalfTriangleHeisenbergJ1J2SquarePEPS;
```

**Hamiltonian**: H = J1 ∑⟨i,j⟩ S⃗_i · S⃗_j + J2 ∑⟨⟨i,j⟩⟩ S⃗_i · S⃗_j
**Features**: Competing interactions, frustration effects

#### 4. t-J Models
```cpp
#include "qlpeps/algorithm/vmc_update/model_solvers/square_tJ_model.h"
using SolverType = SquaretJModel;
```

**Hamiltonian**: H = -t∑⟨i,j⟩,σ (c†_{i,σ} c_{j,σ} + h.c.) + J ∑⟨i,j⟩ (S⃗_i · S⃗_j - 1/4 n_i n_j) - μN
**Features**: Fermionic statistics, charge conservation, strongly correlated systems

### Energy Solver Workflow

**For Each Monte Carlo Sample**:
1. **Input**: Current TPS state + particle configuration
2. **Process**: 
   - Contract tensor networks to compute ⟨config|H|ψ⟩
   - Calculate local energy: E_loc = ⟨config|H|ψ⟩ / ⟨config|ψ⟩
   - Optionally compute gradient holes for optimization
3. **Output**: Local energy value + gradient information

**Key Optimization**: Reuses boundary MPS structures across samples for efficiency.

### Custom Energy Solver Development

To implement a new model, inherit from the appropriate base class:

```cpp
// For nearest-neighbor models
class MyCustomSolver : public SquareNNModelEnergySolver<MyCustomSolver> {
public:
  template<typename TenElemT, typename QNT, bool calchols>
  TenElemT CalEnergyAndHolesImpl(/*...*/);
  
  // Define model-specific energy evaluation
  TenElemT EvaluateBondEnergy(/* site1, site2, ... */);
  TenElemT EvaluateOnsiteEnergy(/* site, ... */);
};
```

**Required Methods**:
- `EvaluateBondEnergy()`: Compute energy for each bond
- `EvaluateOnsiteEnergy()`: Compute on-site energy terms
- `CalEnergyAndHolesImpl()`: Orchestrate the calculation

---

## Component 3: Optimizer Algorithm

The third component is the optimization algorithm itself. This is covered in detail in the [Optimizer Guide](OPTIMIZER_GUIDE.md), but here's how it integrates:

### Integration with VMC Framework

```cpp
VMCPEPSOptimizerParams params{
  optimizer_params,  // Algorithm choice (Adam, SGD, Stochastic Reconfiguration)
  mc_params,         // Monte Carlo sampling parameters
  peps_params        // PEPS bond dimension and truncation
};
```

**Workflow Integration**:
1. **Sampling Phase**: Monte Carlo Updater generates configurations
2. **Evaluation Phase**: Energy Solver computes energies and gradients
3. **Update Phase**: Optimizer Algorithm updates TPS parameters
4. **Repeat**: Until convergence

---

## Complete Integration Example

### Basic Usage Pattern

```cpp
#include "qlpeps/qlpeps.h"

using TenElemT = qlten::QLTEN_Complex;
using QNT = qlten::QNZ2;

// 1. Choose your three strategies
using MonteCarloUpdater = MCUpdateSquareNNExchange;
using EnergySolver = SquareSpinOneHalfXXZModel;

// 2. Configure parameters
OptimizerParams opt_params = /* See Optimizer Guide */;
MonteCarloParams mc_params = /* See Monte Carlo API Guide */;
PEPSParams peps_params = /* See VMC Data Persistence Guide */;

VMCPEPSOptimizerParams vmc_params{opt_params, mc_params, peps_params, "output"};

// 3. Initialize solver
EnergySolver energy_solver(ly, lx, J_coupling);

// 4. Create and execute
VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloUpdater, EnergySolver> 
  executor(vmc_params, initial_tps, MPI_COMM_WORLD, energy_solver);

executor.Execute();
```

### Advanced: Custom Component Combination

```cpp
// Custom three-site updater for frustrated system
using CustomUpdater = MCUpdateSquareTNN3SiteExchange;

// Custom J1-J2 energy solver
using CustomSolver = SpinOneHalfTriangleHeisenbergJ1J2SquarePEPS;

// Stochastic Reconfiguration for high precision
auto opt_params = OptimizerFactory::CreateStochasticReconfiguration(
  1000, cg_params, 0.1
);

VMCPEPSOptimizerExecutor<TenElemT, QNT, CustomUpdater, CustomSolver>
  executor(vmc_params, initial_tps, MPI_COMM_WORLD, custom_solver);
```

---

## Component Interaction Workflow

### Detailed Execution Flow

```
For each optimization iteration:
  ┌─────────────────────────────────────────┐
  │ 1. SAMPLING PHASE                       │
  │   MonteCarloUpdater generates           │
  │   new configurations via detailed       │
  │   balance, updates amplitudes           │
  └─────────────────────────────────────────┘
                      ↓
  ┌─────────────────────────────────────────┐
  │ 2. EVALUATION PHASE                     │
  │   EnergySolver computes local energies  │
  │   and gradient holes for each sample    │
  └─────────────────────────────────────────┘
                      ↓
  ┌─────────────────────────────────────────┐
  │ 3. AGGREGATION PHASE                    │
  │   MPI_Allreduce gathers statistics      │
  │   across all processes                  │
  └─────────────────────────────────────────┘
                      ↓
  ┌─────────────────────────────────────────┐
  │ 4. OPTIMIZATION PHASE                   │
  │   Optimizer updates TPS parameters      │
  │   based on energy and gradient info     │
  └─────────────────────────────────────────┘
                      ↓
  ┌─────────────────────────────────────────┐
  │ 5. BROADCAST PHASE                      │
  │   Updated TPS broadcast to all ranks    │
  │   for next iteration                    │
  └─────────────────────────────────────────┘
```

### MPI Parallelization Strategy

**Embarrassingly Parallel Sampling**:
- Each MPI rank runs independent Monte Carlo chains
- Sampling scales perfectly with processor count
- No communication during sampling phase

**Collective Optimization**:
- Energy/gradient statistics gathered via `MPI_Allreduce`
- Parameter updates computed on master rank
- Updated parameters broadcast via `MPI_Bcast`

---

## Design Principles: Why This Architecture Works

### 1. Single Responsibility Principle
- **Monte Carlo Updater**: Only handles configuration updates
- **Energy Solver**: Only handles energy/gradient computation  
- **Optimizer**: Only handles parameter updates
- **Executor**: Only orchestrates the interaction

### 2. Strategy Pattern with Templates
```cpp
template<typename MonteCarloUpdater, typename EnergySolver>
```
- Compile-time strategy selection
- Zero runtime overhead
- Type-safe composition
- Easy to extend with new strategies

### 3. CRTP for Performance
- Static polymorphism via CRTP
- No virtual function overhead
- Compile-time optimization opportunities
- Clean interfaces without runtime cost

### 4. Fail-Fast Validation
- Configuration validation at initialization
- Amplitude consistency checking
- Early termination on invalid states
- Clear error diagnostics

---

## Common Usage Patterns

### 1. Quick Prototyping
```cpp
// Use defaults for rapid testing
using SimpleUpdater = MCUpdateSquareNNExchange;
using SimpleSolver = SpinOneHalfHeisenbergSquare;
auto adam_params = OptimizerFactory::CreateAdam(500, 1e-3);
```

### 2. High-Precision Production
```cpp
// Stochastic Reconfiguration for best accuracy
using PrecisionSolver = SpinOneHalfHeisenbergSquare;
ConjugateGradientParams cg_params{100, 1e-6, 20, 0.001};
auto sr_params = OptimizerFactory::CreateStochasticReconfiguration(
  2000, cg_params, 0.05
);
```

### 3. Frustrated Systems
```cpp
// Three-site updates for frustration
using FrustrationUpdater = MCUpdateSquareTNN3SiteExchange;
using FrustrationSolver = SpinOneHalfTriangleHeisenbergJ1J2SquarePEPS;
```

### 4. Custom Models
```cpp
// Implement your own solver
class MyQuantumSolver : public SquareNNModelEnergySolver<MyQuantumSolver> {
  // Define model-specific energy evaluation
};
```

---

## Integration with Other Components

### Connection to Existing Tutorials

This executor integrates with several other system components:

1. **Parameter Management**: See [Optimizer Guide](OPTIMIZER_GUIDE.md) for algorithm details
2. **Data Persistence**: See [VMC Data Persistence Guide](VMC_DATA_PERSISTENCE_GUIDE.md) for I/O control
3. **API Patterns**: See [Monte Carlo PEPS API Guide](MONTE_CARLO_PEPS_API_GUIDE.md) for construction patterns
4. **High-Level Overview**: See [Top Level APIs](TOP_LEVEL_APIs.md) for ecosystem context

### Measurement Integration

The same Monte Carlo Updater and Energy Solver strategies can be reused for measurement:

```cpp
// Optimization phase
VMCPEPSOptimizerExecutor<TenElemT, QNT, UpdaterType, SolverType> 
  optimizer(opt_params, initial_tps, comm, solver);
optimizer.Execute();

// Measurement phase using same strategies
MonteCarloMeasurementExecutor<TenElemT, QNT, UpdaterType, MeasurementSolverType>
  measurement(measurement_params, optimized_tps, comm, measurement_solver);
measurement.Execute();
```

---

## Troubleshooting and Best Practices

### Component Selection Guidelines

**Choose Monte Carlo Updater based on**:
- Conservation laws in your model
- Required ergodicity properties
- Computational efficiency needs

**Choose Energy Solver based on**:
- Your specific Hamiltonian
- Required accuracy
- Available built-in solvers vs. custom development

**Choose Optimizer Algorithm based on**:
- Desired convergence speed vs. stability
- System size and computational budget
- Gradient noise characteristics

### Common Pitfalls

1. **Mismatched Components**: Ensure your updater respects the same conservation laws as your energy solver
2. Generally speaking, the Hilbert subspace constraints imposed by Monte Carlo updaters can be more restrictive than the tensor network symmetry constraints, but usually not less restrictive.

### Performance Optimization

1. **Component Choice**: Right strategy selection often more important than parameter tuning
2. **Sampling Efficiency**: Monitor acceptance rates (target ~50% for most updaters)
3. **Memory Management**: Use appropriate PEPS bond dimensions
4. **Load Balancing**: Ensure MPI ranks have similar computational loads

---

## Summary

`VMCPEPSOptimizerExecutor` exemplifies good software design:

- **Composition over inheritance**: Combines three strategy components
- **Single responsibility**: Each component has one well-defined job
- **Template-based flexibility**: Easy to extend without code changes
- **No special cases**: Clean, uniform interface regardless of component choices

The executor's power comes from **eliminating complexity at the architecture level** rather than trying to manage it. By clearly separating the three concerns (sampling, energy evaluation, parameter updates), it becomes straightforward to understand, extend, and optimize each component independently.

**Remember**: The best complexity is the complexity you never have to think about.
