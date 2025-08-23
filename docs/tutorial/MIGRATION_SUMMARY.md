# Migration Guide: From VMCPEPSExecutor to VMCPEPSOptimizer

## Overview

This guide helps you migrate from the legacy `VMCPEPSExecutor` to the new `VMCPEPSOptimizer`. The new executor provides better modularity, cleaner separation of concerns, and improved maintainability while maintaining the same interface.

## TL;DR - What Changes

### 1. **Header Files**
```cpp
// OLD
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"

// NEW - Simple single header includes everything
#include "qlpeps/qlpeps.h"
```

### 2. **Parameter Structure**
```cpp
// OLD: Single structure with lattice size and occupancy arrays
VMCOptimizePara optimize_para(
    truncate_para, num_samples, num_warmup_sweeps, sweeps_between_samples,
    {1, 1, 1, 1}, // occupancy array
    4, 4,         // ly, lx
    step_lengths, update_scheme, cg_params, wavefunction_path);

// NEW: Separate structures with Configuration object
Configuration initial_config(4, 4, OccupancyNum({1, 1, 2})); // ly, lx, occupancy
MonteCarloParams mc_params(num_samples, num_warmup_sweeps, sweeps_between_samples, 
                          initial_config, false); // config, is_warmed_up
PEPSParams peps_params(truncate_para); // simplified structure
OptimizerParams opt_params; // step_lengths, update_scheme, cg_params set separately
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, "./"); // tps_dump_path
```

### 3. **Executor Construction**
```cpp
// OLD: Multiple constructor patterns
VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    optimize_para, tps_init, comm, model);

// NEW: Two clean patterns
// Pattern A: Explicit TPS provided by user
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params, tps_init, comm, model);

// Pattern B: TPS loaded from file path (ly, lx inferred from initial_config)
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params, "/path/to/tps", comm, model);
```

## What Stays the Same

- **Template Parameters**: Same `TenElemT`, `QNT`, `MonteCarloSweepUpdater`, `EnergySolver`
- **Public Interface**: `Execute()`, `GetState()`, `DumpData()` methods
- **Functionality**: Same optimization algorithms and Monte Carlo sampling
- **Results**: Same physical results and convergence behavior

## Key Differences

### Architecture
- **VMCPEPSExecutor**: Monolithic design with optimization logic embedded
- **VMCPEPSOptimizer**: Clean separation using the `Optimizer` class for all optimization logic

### Parameter Structure
- **VMCPEPSExecutor**: Uses `VMCOptimizePara` (legacy structure with occupancy arrays and lattice dimensions)
- **VMCPEPSOptimizer**: Uses `VMCPEPSOptimizerParams` with separate `OptimizerParams`, `MonteCarloParams`, and `PEPSParams`

### Configuration Handling
- **VMCPEPSExecutor**: Used occupancy arrays `{num_up, num_down, ...}` and separate `ly`, `lx` parameters
- **VMCPEPSOptimizer**: Uses `Configuration` objects that encapsulate lattice size and occupancy in a unified structure

### Constructor Patterns  
- **VMCPEPSExecutor**: Multiple overloaded constructors with different parameter combinations
- **VMCPEPSOptimizer**: Two clean patterns - explicit TPS vs. file path loading with automatic lattice size inference

### MPI Behavior
- **VMCPEPSOptimizer**: Better MPI coordination with master rank handling state updates and broadcasting to all ranks

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
    {1, 1, 1, 1},  // occupancy array
    4, 4,  // ly, lx
    {0.01, 0.01, 0.01},  // step_lengths
    StochasticReconfiguration,  // update_scheme
    ConjugateGradientParams(),  // cg_params
    "wavefunction_path"  // wavefunction_path
);

VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    optimize_para, tps_init, comm, model);
```

#### After (VMCPEPSOptimizer):
```cpp
#include "qlpeps/qlpeps.h"

// Step 1: Create Configuration object (replaces occupancy array + ly/lx)
Configuration initial_config(4, 4, OccupancyNum({1, 1, 2})); // ly, lx, occupancy

// Step 2: Create separate parameter structures
MonteCarloParams mc_params(
    1000,  // num_samples
    100,   // num_warmup_sweeps
    10,    // sweeps_between_samples
    initial_config,  // initial configuration
    false,  // is_warmed_up
    ""      // config_dump_path (optional)
);

PEPSParams peps_params(
    BMPSTruncatePara(/* truncation parameters */)
);

OptimizerParams opt_params;
opt_params.core_params.step_lengths = {0.01, 0.01, 0.01};
opt_params.update_scheme = StochasticReconfiguration;
opt_params.cg_params = ConjugateGradientParams();

// Step 3: Combine into unified parameter structure
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, "./"); // tps_dump_path

// Step 4: Create executor (same TPS, just new parameters)
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params, tps_init, comm, model);
```

### Step 2: Update Constructor Calls

The new API provides two clean constructor patterns:

```cpp
// Pattern A: Explicit TPS provided by user
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params,        // VMCPEPSOptimizerParams
    sitpst_init,   // SplitIndexTPS provided by user
    comm,          // MPI communicator
    model          // Energy solver
);

// Pattern B: TPS loaded from file path (recommended for saved states)
VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params,        // VMCPEPSOptimizerParams (ly, lx inferred from initial_config)
    "/path/to/tps", // TPS path - loaded automatically
    comm,          // MPI communicator
    model          // Energy solver
);
```

**Key Changes:**
- No more separate `ly`, `lx` constructor - lattice size is inferred from `initial_config`
- File path constructor automatically loads TPS and determines lattice size
- All constructors use the same unified parameter structure

### Step 3: Update Header Includes

#### Before:
```cpp
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
```

#### After:
```cpp
#include "qlpeps/qlpeps.h"  // Single header includes everything needed
```

**Why this works**: The `qlpeps.h` header automatically includes:
- `vmc_peps_optimizer.h` 
- `monte_carlo_peps_measurement.h`
- `model_solvers/build_in_model_solvers_all.h`
- `configuration_update_strategies/monte_carlo_sweep_updater_all.h`
- All other necessary headers

## Complete Migration Example

### Before (Legacy Code):
```cpp
#include "qlpeps/qlpeps.h"

using namespace qlpeps;
using namespace qlten;

// Setup parameters with all-in-one structure
VMCOptimizePara optimize_para(
    BMPSTruncatePara(8, 1e-12, 1000),
    1000,  // MC samples
    100,   // warmup sweeps
    10,    // sweeps between samples
    {1, 1, 1, 1},  // occupancy array
    4, 4,  // lattice size
    {0.01, 0.01, 0.01},  // step lengths
    StochasticReconfiguration,
    ConjugateGradientParams(1e-6, 1000, 1e-8),
    "wavefunction_data"
);

// Create executor with lattice dimensions
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
#include "qlpeps/qlpeps.h"

using namespace qlpeps;
using namespace qlten;

// Create Configuration object (replaces occupancy array + ly/lx)
Configuration initial_config(4, 4, OccupancyNum({1, 1, 2})); // ly=4, lx=4, occupancy

// Setup separate parameter structures
MonteCarloParams mc_params(
    1000,  // MC samples
    100,   // warmup sweeps
    10,    // sweeps between samples
    initial_config,  // initial configuration
    false  // is_warmed_up
);

PEPSParams peps_params(
    BMPSTruncatePara(8, 1e-12, 1000)
);

OptimizerParams opt_params;
opt_params.core_params.step_lengths = {0.01, 0.01, 0.01};
opt_params.update_scheme = StochasticReconfiguration;
opt_params.cg_params = ConjugateGradientParams(1e-6, 1000, 1e-8);

VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, "./"); // tps_dump_path

// Create executor - Option A: from file path (ly, lx auto-inferred from initial_config)
using Model = SquareSpinOneHalfXXZModel;
using MCUpdater = MCUpdateSquareNNExchange;
Model model;

VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    params, "wavefunction_data", comm, model);

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

- [ ] **Headers**: Use single `#include "qlpeps/qlpeps.h"` (no changes needed if already using this)
- [ ] **Configuration**: Replace occupancy arrays `{num_up, num_down, ...}` with `Configuration(ly, lx, OccupancyNum({...}))` objects  
- [ ] **Parameters**: Convert `VMCOptimizePara` to separate `MonteCarloParams`, `PEPSParams`, and `OptimizerParams`
- [ ] **Unification**: Combine parameter structures into unified `VMCPEPSOptimizerParams`
- [ ] **Lattice Size**: Remove explicit `ly`, `lx` parameters - use Configuration object for lattice size
- [ ] **Executor Type**: Change from `VMCPEPSExecutor` to `VMCPEPSOptimizer`
- [ ] **Constructor Pattern**: Choose explicit TPS vs. file path loading (both patterns work)
- [ ] **Testing**: Test compilation
- [ ] **Testing**: Test runtime execution  
- [ ] **Validation**: Verify energy convergence matches expected behavior

## Common Issues and Solutions

### 1. **Missing optimizer headers**
**Problem**: Compilation errors due to missing optimizer headers  
**Solution**: Include `#include "qlpeps/optimizer/optimizer_params.h"`

### 2. **Parameter structure mismatch**
**Problem**: Trying to use `VMCOptimizePara` with the new executor  
**Solution**: Convert to separate `MonteCarloParams`, `PEPSParams`, and `OptimizerParams` structures

### 3. **Configuration initialization**
**Problem**: Configuration not properly initialized - lattice size undefined  
**Solution**: Use `Configuration(ly, lx, OccupancyNum({num_up, num_down, num_holes}))` constructor

### 4. **Lattice size mismatch**
**Problem**: TPS file size doesn't match configuration size  
**Solution**: Ensure `initial_config.rows()` and `initial_config.cols()` match the saved TPS dimensions

### 5. **Constructor pattern confusion**
**Problem**: Trying to pass `ly`, `lx` directly to constructor  
**Solution**: Use file path constructor - lattice size is inferred from `initial_config` automatically

## Backward Compatibility

The `VMCPEPSOptimizer` maintains the same public interface as `VMCPEPSExecutor`:

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

Migrating to `VMCPEPSOptimizer` provides significant benefits in terms of code quality, maintainability, and future extensibility. The migration process is straightforward and the new executor maintains backward compatibility while offering improved architecture.

For questions or issues during migration, refer to the test files in `tests/test_algorithm/` for working examples.
