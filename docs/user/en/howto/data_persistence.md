# VMC Data Persistence Guide

## Overview

This guide explains the **unified data persistence system** for VMC (Variational Monte Carlo) executors, including optimization and measurement. The design provides explicit user control over data dumping with clear, predictable I/O patterns.

## Design Principles

### 1. **Explicit User Control**
- Users specify exactly what and where to dump
- Empty path = no dump (clean opt-out mechanism)
- No hidden fallbacks or magic paths

### 2. **Logical Separation by Data Type**
- **Configuration dumps**: Controlled via `MonteCarloParams.config_dump_path`
- **TPS dumps**: Controlled via `VMCPEPSOptimizerParams.tps_dump_base_name` (optimizer) or explicit paths (measurement)
- **Measurement data dumps**: Controlled via `MCMeasurementParams.measurement_data_dump_path`
- **Optimization trajectory**: Energy samples, gradients, etc. (optimizer only)

### 3. **Context-Appropriate Defaults**
- **Configuration dumps**: Empty by default (opt-in)
- **TPS dumps**: Use `kTpsPathBase` constant ("tps") from `consts.h` by default
- **Measurement data**: Use `"./"`  by default (current directory)
- **Energy trajectory**: CSV dumped to `./energy/energy_trajectory.csv` during optimization

## Architecture by Component

### MonteCarloParams - Configuration Dumps

```cpp
struct MonteCarloParams {
  // ... other params ...
  std::string config_dump_path;  ///< Path for dumping final configuration (empty = no dump)
  
  MonteCarloParams(/*...*/, const std::string &config_dump_path = "")
    : /*...*/, config_dump_path(config_dump_path) {}
};
```

**Usage Pattern:**
- **Both VMC and Measurement** can dump final configurations
- User explicitly sets path if they want configuration saved
- Empty string means no configuration dump (saves disk space)

**Example:**
```cpp
// Dump configuration to specific path
MonteCarloParams mc_params(1000, 100, 5, config, false, "./final_config");

// Don't dump configuration (save space)  
MonteCarloParams mc_params(1000, 100, 5, config, false, "");  // or omit last param
```

### VMCPEPSOptimizerParams - TPS Dumps with Base Name + Postfix

```cpp
struct VMCPEPSOptimizerParams {
  // ... other params ...
  std::string tps_dump_base_name;  ///< Base name for TPS dumps (postfixes: final, lowest). Empty = no dump

  VMCPEPSOptimizerParams(const OptimizerParams &opt_params,
                         const MonteCarloParams &mc_params,
                         const PEPSParams &peps_params,
                         const std::string &tps_dump_path = "./");
};
```

**TPS Dump Strategy:**
VMC optimization generates **two important TPS states**:
1. **Final TPS**: Tail state held by the optimizer when the run finishes → `base_name + "final"`
2. **Lowest Energy TPS**: Snapshot with the lowest observed MC energy estimate → `base_name + "lowest"`

**Default Behavior:**
- Uses `kTpsPath` constant (`"tps"`) as default base name
- Generates: `"tpsfinal"` and `"tpslowest"`
- Empty base name disables all TPS dumping

**Example:**
```cpp
// Standard case - use default base name
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);
// → Will dump to "tpsfinal" and "tpslowest"

// Custom base name  
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);
params.tps_dump_base_name = "experiment_A";
// → Will dump to "experiment_Afinal" and "experiment_Alowest"

// No TPS dumping
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);
params.tps_dump_base_name = "";
// → No TPS dumps created
```

### MCMeasurementParams - Measurement Data Dumps

```cpp
struct MCMeasurementParams {
  MonteCarloParams mc_params;
  PEPSParams peps_params;
  std::string measurement_data_dump_path;  ///< Path for dumping measurement results (empty = current dir)

  MCMeasurementParams(/*...*/, const std::string &measurement_data_dump_path = "./")
    : /*...*/, measurement_data_dump_path(measurement_data_dump_path) {}
};
```

**Measurement Data Strategy:**
- **No TPS dumps** (measurements don't modify TPS)
- **Configuration dumps** via `mc_params.config_dump_path` (if specified)
- **Measurement results** → organized subdirectories under `measurement_data_dump_path`

**Generated Structure:**
```
{measurement_data_dump_path}/
├── metadata.txt              # run metadata, sample counts, MPI size, lattice size
├── stats/                    # aggregated observable statistics
│   ├── energy.csv
│   ├── <flat_observable>.csv
│   ├── <matrix_observable>_mean.csv
│   └── <matrix_observable>_stderr.csv
└── samples/
    └── psi.csv               # per-sample psi consistency summary on master rank
```

## Implementation Details

### Helper Methods in Base Classes

```cpp
// Simple, direct path generation - no wrapper functions needed
std::string final_tps_path = tps_base_name + "final";    // e.g., "tpsfinal"
std::string lowest_tps_path = tps_base_name + "lowest";  // e.g., "tpslowest"

// Empty base name = no dump (explicit user control)
if (tps_base_name.empty()) {
  // Skip TPS dumping entirely
}
```

### Dump Logic Examples

#### VMC Optimizer Dumping
```cpp
void VMCPEPSOptimizer::DumpData(const std::string& tps_base_name, bool release_mem) {
  // Generate paths with consistent naming
  std::string final_tps_path = tps_base_name + "final";   // Final tail state
  std::string lowest_tps_path = tps_base_name + "lowest"; // Lowest observed MC energy estimate
  std::string energy_data_path = "./energy";             // Optimization trajectory
  
  if (rank_ == kMPIMasterRank) {
    // Dump TPS states if base name provided
    if (!tps_base_name.empty()) {
      monte_carlo_engine_.State().Dump(final_tps_path, release_mem);
      tps_lowest_.Dump(lowest_tps_path, release_mem);
    }
    
    // Energy trajectory data
    EnsureDirectoryExists(energy_data_path);
  }
  
  MPI_Barrier(comm_);
  
  // Dump final configuration if requested
  if (!params_.mc_params.config_dump_path.empty()) {
    tps_sample_.config.Dump(params_.mc_params.config_dump_path, rank_);
  }
  
  if (rank_ == kMPIMasterRank) {
    WriteEnergyTrajectoryCsv_(energy_data_path + "/energy_trajectory.csv");
  }
}
```

## VMC Dump Mechanisms by Executor Type

### 1. VMC Optimizer (`VMCPEPSOptimizer`)

The optimizer dumps multiple types of data with different control mechanisms:

#### TPS States
- **Final TPS**: `tps_dump_base_name + "final"` (final tail state)
- **Lowest Energy TPS**: `tps_dump_base_name + "lowest"` (lowest observed MC energy estimate)
- **Control**: Via `VMCPEPSOptimizerParams.tps_dump_base_name`

#### Configuration Snapshots  
- **Final Monte Carlo configuration**
- **Control**: Via `MonteCarloParams.config_dump_path`

#### Optimization Trajectory
- **Global energy trajectory**: `./energy/energy_trajectory.csv`
- **Checkpoint live trajectory**: `{checkpoint_base_path}/energy_trajectory.csv`
- **Checkpoint step snapshot**: `{checkpoint_base_path}/step_<k>/trajectory_snapshot.csv`
- **Control**: global VMC trajectory is always dumped to `./energy/`; checkpoint CSVs are written only when `checkpoint_params` is enabled and use its `base_path`

#### Parameter Setup Example
```cpp
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);

// TPS dumping control
params.tps_dump_base_name = "experiment_01";  // → "experiment_01final", "experiment_01lowest"
// or params.tps_dump_base_name = "";         // No TPS dumps

// Configuration dumping control  
params.mc_params.config_dump_path = "final_configs/exp01";  // Explicit config path
// or params.mc_params.config_dump_path = "";               // No config dump

VMCPEPSOptimizer executor(tps, params, energy_solver, /* comm */);
executor.Execute();  // Automatically calls DumpData() at end
```

### 2. Measurement Executor (`MCPEPSMeasurer`)

#### Configuration Snapshots
```cpp
void MCPEPSMeasurer::DumpData(const std::string &measurement_data_path) {
  // Dump configuration if path is specified in MonteCarloParams
  if (!mc_measure_params.mc_params.config_dump_path.empty()) {
    tps_sample_.config.Dump(mc_measure_params.mc_params.config_dump_path, rank_);
  }

  // Dump measurement results to organized subdirectories
  const std::string base_path = measurement_data_path.empty() ? "./" : measurement_data_path + "/";
  // Dumps metadata.txt, stats/*.csv, and samples/psi.csv
}
```

## Usage Examples

### Typical Research Workflow

```cpp
// Setup for production run with full data preservation
// Create t-J compatible configuration for 8x8 lattice
size_t total_sites = 8 * 8;
size_t num_holes = total_sites - 32 - 32;
Configuration research_config(8, 8, OccupancyNum({32, 32, num_holes}));

MonteCarloParams mc_params(
  5000,                           // samples
  1000,                           // warmup  
  10,                             // sweeps between samples
  research_config,                // initial config
  false,                          // not warmed up
  "production_run_final_config"   // save final config
);

VMCPEPSOptimizerParams opt_params(
  optimizer_params,
  mc_params, 
  peps_params,
  "production_run_tps"            // explicit single-path dump for result.state if used
);
```

### Quick Testing Without Disk Usage

```cpp
// Setup for testing - minimal disk usage
Configuration test_config(4, 4);

MonteCarloParams mc_params(
  100,                            // small sample size
  50,                             // quick warmup
  2,                              // fast sampling  
  test_config,                    // test config
  false,                          // not warmed up
  ""                              // no config dump
);

VMCPEPSOptimizerParams opt_params(
  optimizer_params,
  mc_params,
  peps_params
);
opt_params.tps_dump_base_name = "";        // no final/lowest TPS dumps
```

## Best Practices for VMC Data Management

### 1. Use Descriptive Names
```cpp
// Good: Descriptive TPS base names
params.tps_dump_base_name = "heisenberg_D8_L4x4_sweep1000";
params.mc_params.config_dump_path = "configs/heisenberg_run01";

// Outputs:
// - heisenberg_D8_L4x4_sweep1000final/   (final tail TPS)
// - heisenberg_D8_L4x4_sweep1000lowest/  (lowest observed MC energy TPS)
// - configs/heisenberg_run01{rank}       (final configurations)
// - energy/energy_trajectory.csv         (optimization data)
```

### 2. Optimize Storage Usage
```cpp
// For parameter scans: disable TPS dumps to save space
params.tps_dump_base_name = "";         // No TPS dumps
params.mc_params.config_dump_path = ""; // No config dumps
// Only energy trajectory will be saved
```

### 3. Chain Optimization and Measurement
```cpp
// Step 1: Optimization with config dump
opt_params.mc_params.config_dump_path = "warmed_configs/step1";

// Step 2: Measurement using warmed config
measurement_params.mc_params.initial_config = /* load from step1 */;
measurement_params.mc_params.is_warmed_up = true;
```

### 4. Constants and Defaults
```cpp
// Default TPS base name from consts.h
#include "qlpeps/consts.h"

// Available constants:
const std::string kTpsPath = "tps";         // For single TPS dumps
const std::string kTpsPathBase = "tps";     // For base name (appends "final"/"lowest")

// Default usage:
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);
// params.tps_dump_base_name defaults to kTpsPathBase
```

## Summary

This unified dump system provides:
- **Clear separation** between different data types
- **Explicit user control** over what gets dumped where  
- **Consistent patterns** across optimization and measurement
- **Storage optimization** through selective dumping
- **Easy workflow chaining** via configuration persistence

The design eliminates "magic paths" and gives users full control over their data persistence strategy.

### Analysis-Only Measurement

```cpp
// Load existing TPS for analysis
Configuration analysis_config(6, 6);

MonteCarloParams mc_params(
  10000,                          // high statistics
  500,                            // thorough warmup
  5,                              // good sampling
  analysis_config,                // analysis config  
  false,                          // not warmed up
  ""                              // don't dump config (analysis only)
);

MCMeasurementParams measure_params(
  mc_params,
  peps_params,
  "analysis_results"              // organized measurement data
);

MCPEPSMeasurer executor(
  "/path/to/converged/tps",       // load existing TPS
  measure_params,
  comm,
  solver
);
```

## Migration Notes

### Old API Problems
```cpp
// Old: Unclear what gets dumped where
VMCOptimizePara para(truncate_para, samples, warmup, sweeps, 
                     fallback_config, step_lengths, update_scheme, 
                     cg_params, "/mystery/wavefunction/path");
// → Where does final TPS go? What about lowest energy TPS? Configuration?
```

### New API Solution
```cpp
// New: Crystal clear control
MonteCarloParams mc_params(samples, warmup, sweeps, config, false, "final_config");
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);
params.tps_dump_base_name = "experiment_tps";
// → final_config, experiment_tpsfinal, experiment_tpslowest
```

## Best Practices

1. **Use meaningful base names** - include experiment/run identifiers
2. **Set config dump paths for important runs** - configurations are small but crucial  
3. **Use empty paths for testing** - avoid cluttering disk during development
4. **Organize measurement data** - use descriptive measurement_data_dump_path
5. **Follow consistent naming** - establish patterns across your research workflow

This architecture serves users by eliminating surprises while maintaining convenience through sensible defaults.
