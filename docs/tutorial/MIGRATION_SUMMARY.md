# Migration Guide: From VMCPEPSExecutor to VMCPEPSOptimizerExecutor

## Overview

This guide helps you migrate from the legacy `VMCPEPSExecutor` to the new `VMCPEPSOptimizerExecutor`. The new executor provides better modularity, cleaner separation of concerns, and improved maintainability while maintaining the same interface.

## TL;DR - What Changes

### 1. **Header Files**
```cpp
// OLD
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"

// NEW  
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/optimizer/optimizer_params.h"
```

### 2. **Parameter Structure**
```cpp
// OLD: Single structure
VMCOptimizePara optimize_para(/* all parameters */);

// NEW: Separate structures
MonteCarloParams mc_params(/* MC parameters */);
PEPSParams peps_params(/* PEPS parameters */);
OptimizerParams opt_params(/* optimization parameters */);
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);
```

### 3. **Executor Type**
```cpp
// OLD
VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    optimize_para, tps_init, comm, model);

// NEW
VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    params, tps_init, comm, model);
```

## What Stays the Same

- **Template Parameters**: Same `TenElemT`, `QNT`, `MonteCarloSweepUpdater`, `EnergySolver`
- **Public Interface**: `Execute()`, `GetState()`, `DumpData()` methods
- **Functionality**: Same optimization algorithms and Monte Carlo sampling
- **Results**: Same physical results and convergence behavior

## Key Differences

### Architecture
- **VMCPEPSExecutor**: Monolithic design with optimization logic embedded
- **VMCPEPSOptimizerExecutor**: Clean separation using the `Optimizer` class for all optimization logic

### Parameter Structure
- **VMCPEPSExecutor**: Uses `VMCOptimizePara` (legacy structure)
- **VMCPEPSOptimizerExecutor**: Uses `VMCPEPSOptimizerParams` with separate `OptimizerParams`, `MonteCarloParams`, and `PEPSParams`

### MPI Behavior
- **VMCPEPSOptimizerExecutor**: Better MPI coordination with master rank handling state updates and broadcasting to all ranks

## Migration Steps

### Step 1: Update Parameter Structure

#### Before (VMCPEPSExecutor):
```cpp
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"

VMCOptimizePara optimize_para(
    BMPSTruncatePara(/* truncation parameters */),
    1000,  // num_samples
    100,   // num_warmup_sweeps
    10,    // sweeps_between_samples
    {1, 1, 1, 1},  // occupancy
    4, 4,  // ly, lx
    {0.01, 0.01, 0.01},  // step_lengths
    StochasticReconfiguration,  // update_scheme
    ConjugateGradientParams(),  // cg_params
    "wavefunction_path"  // wavefunction_path
);

VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    optimize_para, tps_init, comm, model);
```

#### After (VMCPEPSOptimizerExecutor):
```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/optimizer/optimizer_params.h"

// Create separate parameter structures
MonteCarloParams mc_params(
    1000,  // num_samples
    100,   // num_warmup_sweeps
    10,    // sweeps_between_samples
    "config_path",  // config_path
    Configuration(4, 4)  // alternative_init_config
);

PEPSParams peps_params(
    BMPSTruncatePara(/* truncation parameters */),
    "wavefunction_path"  // wavefunction_path
);

OptimizerParams opt_params;
opt_params.core_params.step_lengths = {0.01, 0.01, 0.01};
opt_params.update_scheme = StochasticReconfiguration;
opt_params.cg_params = ConjugateGradientParams();

VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    params, tps_init, comm, model);
```

### Step 2: Update Constructor Calls

The constructor signatures remain the same, but use the new parameter structure:

```cpp
// Constructor with TPS
VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    params, tps_init, comm, model);

// Constructor with SplitIndexTPS
VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    params, sitpst_init, comm, model);

// Constructor with dimensions
VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    params, ly, lx, comm, model);
```

### Step 3: Update Header Includes

#### Before:
```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
```

#### After:
```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/optimizer/optimizer_params.h"
```

## Complete Migration Example

### Before (Legacy Code):
```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

using namespace qlpeps;
using namespace qlten;

// Setup parameters
VMCOptimizePara optimize_para(
    BMPSTruncatePara(8, 1e-12, 1000),
    1000,  // MC samples
    100,   // warmup sweeps
    10,    // sweeps between samples
    {1, 1, 1, 1},  // occupancy
    4, 4,  // lattice size
    {0.01, 0.01, 0.01},  // step lengths
    StochasticReconfiguration,
    ConjugateGradientParams(1e-6, 1000, 1e-8),
    "wavefunction_data"
);

// Create executor
using Model = SquareSpinOneHalfXXZModel;
using MCUpdater = MCUpdateSquareNNExchange;
Model model;

VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    optimize_para, 4, 4, comm, model);

// Execute optimization
executor.Execute();
```

### After (New Code):
```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

using namespace qlpeps;
using namespace qlten;

// Setup separate parameter structures
MonteCarloParams mc_params(
    1000,  // MC samples
    100,   // warmup sweeps
    10,    // sweeps between samples
    "config_path",  // config path
    Configuration(4, 4)  // alternative init config
);

PEPSParams peps_params(
    BMPSTruncatePara(8, 1e-12, 1000),
    "wavefunction_data"  // wavefunction path
);

OptimizerParams opt_params;
opt_params.core_params.step_lengths = {0.01, 0.01, 0.01};
opt_params.update_scheme = StochasticReconfiguration;
opt_params.cg_params = ConjugateGradientParams(1e-6, 1000, 1e-8);

VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

// Create executor
using Model = SquareSpinOneHalfXXZModel;
using MCUpdater = MCUpdateSquareNNExchange;
Model model;

VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    params, 4, 4, comm, model);

// Execute optimization
executor.Execute();
```

## Benefits of Migration

### 1. **Better Modularity**
- Optimization logic is separated into the `Optimizer` class
- Clear separation between Monte Carlo, PEPS, and optimization parameters

### 2. **Improved Testability**
- Each component can be tested independently
- Easier to mock and test optimization strategies

### 3. **Enhanced Maintainability**
- Cleaner code structure
- Easier to add new optimization algorithms
- Better separation of concerns

### 4. **Superior MPI Coordination**
- Master rank handles state updates
- Automatic broadcasting to maintain synchronization
- Better handling of stochastic reconfiguration

### 5. **Future-Proof Design**
- Built on the modern `Optimizer` framework
- Easier to extend with new features
- Better alignment with modern C++ practices

## Migration Checklist

- [ ] Update header includes
- [ ] Convert `VMCOptimizePara` to separate parameter structures
- [ ] Change executor type from `VMCPEPSExecutor` to `VMCPEPSOptimizerExecutor`
- [ ] Update constructor calls to use new parameter structure
- [ ] Test compilation
- [ ] Test runtime execution
- [ ] Verify results match

## Common Issues and Solutions

### 1. **Missing optimizer headers**
**Problem**: Compilation errors due to missing optimizer headers  
**Solution**: Include `optimizer_params.h`

### 2. **Parameter structure mismatch**
**Problem**: Trying to use `VMCOptimizePara` with the new executor  
**Solution**: Convert to `VMCPEPSOptimizerParams` structure

### 3. **Configuration initialization**
**Problem**: Configuration not properly initialized in new parameter structure  
**Solution**: Use `Configuration(ly, lx)` constructor and set occupancy

## Backward Compatibility

The `VMCPEPSOptimizerExecutor` maintains the same public interface as `VMCPEPSExecutor`:

- `Execute()` method
- `GetState()` method  
- `DumpData()` methods
- Same template parameters

This means you can often replace the executor type without changing the rest of your code.

## Testing Your Migration

After migration, test that:

1. **Compilation**: Code compiles without errors
2. **Runtime**: Optimization executes successfully
3. **Results**: Energy convergence matches expected behavior
4. **MPI**: Works correctly across multiple ranks

## Conclusion

Migrating to `VMCPEPSOptimizerExecutor` provides significant benefits in terms of code quality, maintainability, and future extensibility. The migration process is straightforward and the new executor maintains backward compatibility while offering improved architecture.

For questions or issues during migration, refer to the test files in `tests/test_algorithm/` for working examples.
