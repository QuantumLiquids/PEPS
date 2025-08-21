# Monte Carlo PEPS API Unification & Modernization

**Date**: 2025-08-18  
**Type**: Breaking Change  
**Status**: COMPLETED  
**Commits**: a4765ac, a0daad3, 5eadbca

## Executive Summary

Complete modernization of Monte Carlo PEPS API with unified parameter structures, improved error handling, explicit user control, and elimination of legacy patterns. This addresses long-standing issues with implicit behaviors, inconsistent APIs, and library design anti-patterns.

## Core Philosophy Changes

**From**: "Magic library that guesses user intent"  
**To**: "Explicit tools where user controls data and behavior"

- ✅ Single responsibility: each component does one thing well
- ✅ User control: explicit data ownership, no magic  
- ✅ Fail fast: clear errors instead of silent fallbacks
- ✅ Type safety: explicit conversions, no operator overloads

## Major Changes

### 1. Unified Parameter Structures (BREAKING)

#### Removed
- `MCMeasurementPara` class
- `VMCOptimizePara` legacy structure  
- `VMCPEPSExecutor` class (deprecated)
- Implicit type conversion operators
- Silent configuration loading/replacement

#### Added
- `MCMeasurementParams` - clean composition of MC + PEPS parameters
- `VMCPEPSOptimizerParams` - unified optimizer parameter structure
- Explicit accessor methods (`GetMCParams()`, `GetPEPSParams()`, etc.)
- Explicit state management (`is_warmed_up` flag)
- Factory pattern for executor creation

### 2. Error Handling Modernization (CRITICAL)

#### Before: Library Exit Anti-pattern
```cpp
// Library would call exit() on errors - user has no control
if (error_condition) {
  std::cerr << "Error!" << std::endl;
  exit(EXIT_FAILURE);  // BAD: Library should never exit
}
```

#### After: Exception-based Error Handling
```cpp
// Library throws exceptions - user controls error handling
if (error_condition) {
  throw std::runtime_error("Configuration load failed: " + path);
}
```

### 3. Constructor to Factory Pattern Migration

#### Before: Complex Constructors with Hidden I/O
```cpp
// Constructor secretly loads files from disk
MonteCarloMeasurementExecutor executor(tps_path, measurement_params, comm);
```

#### After: Explicit User Control
```cpp
// User explicitly loads TPS data
auto tps = SplitIndexTPS<>::LoadFromFile(tps_path);
MonteCarloMeasurementExecutor executor(tps, measurement_params, comm);

// OR: Use convenience factory if preferred
auto executor = MonteCarloMeasurementExecutor::CreateByLoadingTPS(
    tps_path, measurement_params, comm, solver);
```

### 4. Configuration Validation Enhancement

Renamed `RescueInvalidConfigurations_()` → `EnsureConfigurationValidity_()` for accuracy.

**Function behavior** (unchanged but clarified):
- Validates wave function amplitudes across all MPI ranks
- Auto-rescues invalid configurations from valid ranks when possible
- Provides clear diagnostics on complete failure
- No silent failures or undefined behavior

## API Migration Guide

### Parameter Structures

#### Before
```cpp
MCMeasurementPara para(trunc_para, 50, 50, 1, {1,1,2}, 2, 2);
```

#### After
```cpp
Configuration config(2, 2);
config.Random({1, 1, 2});
MonteCarloParams mc_params(50, 50, 1, config, false);
PEPSParams peps_params(trunc_para);
MCMeasurementParams params(mc_params, peps_params);
```

### Measurement Executor

#### Before
```cpp
MonteCarloMeasurementExecutor<> executor(tps_path, params, comm, solver);
```

#### After (Explicit TPS loading)
```cpp
auto tps = SplitIndexTPS<>::LoadFromFile(tps_path);
MonteCarloMeasurementExecutor<> executor(tps, params, comm, solver);
```

#### After (Factory convenience)
```cpp
auto executor = MonteCarloMeasurementExecutor<>::CreateByLoadingTPS(
    tps_path, params, comm, solver);
```

### VMC Optimizer

#### Before
```cpp
VMCPEPSExecutor<> optimizer(vmc_params, tps, comm, solver);
```

#### After
```cpp
VMCPEPSOptimizerExecutor<> optimizer(vmc_params, tps, comm, solver);
```

## Files Modified (48 total)

### Core API Headers
- `monte_carlo_peps_params.h` - unified parameter structures
- `monte_carlo_peps_base.h` - improved error handling & state access  
- `monte_carlo_peps_measurement.h` - factory pattern implementation
- `vmc_peps_optimizer*.h` - compatibility with unified parameters

### Test Files (All Updated)
- `test_algorithm/test_mc_peps_measure.cpp`
- `test_algorithm/test_vmc_peps_optimizer.cpp`
- `test_optimizer/test_optimizer_adagrad_exact_sum.cpp`
- `integration_tests/` (6 files)
- `slow_tests/` (3 files)  
- `test_deprecated/test_loop_update_deprecated.cpp`

### Documentation & Examples
- Complete Chinese documentation suite
- Updated migration examples
- New API guides and tutorials

### Legacy Cleanup
- Removed `vmc_peps.h` and `vmc_peps_impl.h`
- Deleted deprecated VMCPEPSExecutor implementation
- Cleaned up circular header dependencies
- Fixed test data pollution bugs

## Validation Results

- ✅ **All tests pass** - Zero physics result changes
- ✅ **No performance regression** - Minimal memory overhead  
- ✅ **Clean compilation** - No deprecation warnings
- ✅ **MPI compatibility maintained** - All parallel tests pass
- ✅ **Memory safety improved** - Exception-based error handling
- ✅ **Type safety enhanced** - Explicit conversions only

## Breaking Changes Summary

1. **Parameter struct names changed** - `MCMeasurementPara` → `MCMeasurementParams`
2. **Implicit conversions removed** - Must use explicit getter methods
3. **Constructor signatures changed** - TPS loading now explicit or via factory
4. **Error handling changed** - Library throws exceptions instead of exit()
5. **Class names changed** - `VMCPEPSExecutor` → `VMCPEPSOptimizerExecutor`

## Migration Support

- All existing patterns have clear migration paths
- Factory functions provide convenience when appropriate
- Exception handling allows graceful error recovery
- Explicit state management eliminates guesswork

## Impact Assessment

**Positive**:
- Eliminates library design anti-patterns
- Gives users complete control over their data
- Enables proper error handling in user applications
- Reduces API surface complexity through unification
- Improves code maintainability and testability

**Risk Mitigation**:
- Comprehensive test coverage ensures physics correctness
- Performance testing confirms zero regression
- Documentation provides clear migration paths
- Factory functions ease transition for common use cases

---

**Design Principle**: *"Do one thing and do it well. User owns their data."*

This refactor transforms the Monte Carlo PEPS API from an implicit, magic-heavy interface to an explicit, user-controlled tool that follows modern C++ best practices.
