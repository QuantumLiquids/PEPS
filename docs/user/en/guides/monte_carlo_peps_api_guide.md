# Monte Carlo PEPS API Usage Guide

## Overview

The refactored Monte Carlo PEPS API provides two clear construction patterns:
1. **Explicit Data Control**: User provides TPS and configuration explicitly
2. **Convenient File Loading**: User provides file path, system loads TPS automatically

Both patterns maintain the same **unified parameter structure** and **clear initialization logic**.

## Pattern 1: Explicit Data Control 

**Best for**: Custom TPS creation, in-memory calculations, precise control over initialization

### For Measurement

```cpp
#include "qlpeps/qlpeps.h"
#include "qlpeps/api/conversions.h" // explicit conversions PEPS/TPS/SITPS

// User creates or loads TPS explicitly
SplitIndexTPS<TenElemT, QNT> user_tps(ly, lx);
user_tps.Random();  // or Load(), or custom creation

// User creates configuration with full control  
Configuration user_config(ly, lx, OccupancyNum({num_up, num_down, num_empty}));

// Unified parameter structure
MonteCarloParams mc_params(1000, 100, 5, user_config, false);  // samples, warmup, sweeps, config, is_warmed_up
PEPSParams peps_params(BMPSTruncatePara(/*...*/));
MCMeasurementParams params(mc_params, peps_params, "./output");  // includes dump path

// Clean, explicit constructor
MCPEPSMeasurer<TenElemT, QNT, UpdaterType, SolverType> executor(
    user_tps,        // explicit TPS
    params,          // unified parameters
    comm,            // MPI communicator
    solver           // measurement solver
);
```

### For Optimization

```cpp
#include "qlpeps/qlpeps.h"

// User provides TPS and configuration explicitly
SplitIndexTPS<TenElemT, QNT> initial_tps(ly, lx);
initial_tps.Load("/path/to/initial/tps");

// Load configuration from file with fallback
Configuration initial_config(ly, lx);
bool load_success = initial_config.Load("/path/to/config.dat", 0);
if (!load_success) {
    // Create fallback configuration with proper occupancy
    initial_config = Configuration(ly, lx, OccupancyNum({num_up, num_down, num_holes}));
    load_success = false;  // Still not warmed up since we created new config
}
// Note: warm up status is determined separately in MonteCarloParams

// Create optimization parameters
OptimizerParams opt_params(/*...*/);
MonteCarloParams mc_params(500, 100, 3, initial_config, load_success);
PEPSParams peps_params(BMPSTruncatePara(/*...*/));
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

// Explicit constructor
VMCPEPSOptimizer<TenElemT, QNT, UpdaterType, SolverType> optimizer(
    params,          // unified parameters
    initial_tps,     // explicit TPS
    comm,            // MPI communicator
    solver           // energy solver
);
```

## Pattern 2: Convenient File Loading

**Best for**: Standard workflows, loading from previous calculations, typical research scenarios

### For Measurement

```cpp
// Configuration determines lattice size automatically
Configuration analysis_config(4, 4);  // ly=4, lx=4 inferred
analysis_config.Random(std::vector<size_t>(2, 8));  // proper initialization

// Unified parameters - lattice size inferred from config
MonteCarloParams mc_params(2000, 200, 10, analysis_config, false);
PEPSParams peps_params(BMPSTruncatePara(/*...*/));
MCMeasurementParams params(mc_params, peps_params, "./measurement_output");

// Convenient constructor - TPS loaded from path
MonteCarloMeasurementExecutor<TenElemT, QNT, UpdaterType, SolverType> executor(
    "/path/to/saved/tps",  // TPS path - system loads automatically
    params,                // unified parameters (ly, lx inferred from config)
    comm,                  // MPI communicator  
    solver                 // measurement solver
);
```

### For Optimization  

```cpp
// Configuration size determines lattice dimensions
// Create t-J compatible configuration (6x6 lattice with 18 up + 18 down electrons)
size_t total_sites = 6 * 6;
size_t num_holes = total_sites - 18 - 18;
Configuration opt_config(6, 6, OccupancyNum({18, 18, num_holes}));

// Create parameters
OptimizerParams opt_params(/*...*/);
MonteCarloParams mc_params(1000, 200, 5, opt_config, false);
PEPSParams peps_params(BMPSTruncatePara(/*...*/));
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

// Convenient constructor
VMCPEPSOptimizer<TenElemT, QNT, UpdaterType, SolverType> optimizer(
    params,                // unified parameters
    "/path/to/initial/tps", // TPS path - loaded automatically, ly/lx from config
    comm,                  // MPI communicator
    solver                 // energy solver
);
```

## Key Design Principles

### 1. **Lattice Size Inference**
- **Automatic**: `ly = initial_config.rows()`, `lx = initial_config.cols()`
- **No Guessing**: User must provide properly-sized configuration
- **Clear Error**: System fails fast if configuration size is invalid

### 2. **Unified Parameters**
- **No Special Cases**: Same parameter structure for both patterns
- **User Responsibility**: Explicit `dump_path` specification  
- **Clear Ownership**: User knows exactly what data they're providing

### 3. **Consistent Initialization**
- **Same Process**: Both patterns use identical initialization logic
- **Predictable Behavior**: No hidden differences between patterns
- **Full Control**: User can always inspect and override defaults

## Migration from Legacy API

### Old Confused Pattern
```cpp
// User confusion: which config is used? what if path fails?
MCMeasurementParams para(mc_params, peps_params); // using unified parameter structure 
                       fallback_config,        // maybe used?
                       "/mystery/tps/path");   // maybe works?
```

### New Clear Patterns
```cpp
// Option A: User controls everything
MonteCarloMeasurementExecutor executor(user_tps, params, comm, solver);

// Option B: User specifies file, size inferred from config  
MonteCarloMeasurementExecutor executor("/path/to/tps", params, comm, solver);
```

## Best Practices

1. **Use Pattern 2 for typical research workflows** - most convenient
2. **Use Pattern 1 for precise control** - when you need custom TPS or special initialization
3. **Always specify dump_path** - don't rely on defaults for important data
4. **Use Configuration constructors directly** - for common configuration patterns
5. **Check configuration size** - ensure it matches your intended lattice dimensions

This design eliminates confusion while maintaining convenience - serving users by giving them control without complexity.
