# Optimizer Parameters API Refactor

**Date**: 2025-01-29  
**Type**: Breaking Change  
**Status**: COMPLETED

## Changes

### Removed
- WAVEFUNCTION_UPDATE_SCHEME enum → std::variant
- VMCOptimizePara struct → VMCPEPSOptimizerParams  
- VMCPEPSExecutor class → VMCPEPSOptimizer
- Default parameter constructors (Google C++ style compliance)
- Legacy compatibility fields in OptimizerParams

### Added
- Unified learning_rate parameter in BaseParams
- Learning rate schedulers: ConstantLR, ExponentialDecayLR, StepLR, PlateauLR
- Type-safe algorithm parameters using std::variant
- OptimizerFactory for common configurations
- OptimizerParamsBuilder for complex configurations

### API Migration

#### Before
```cpp
VMCOptimizePara params(/*many parameters*/);
VMCPEPSExecutor<TenElemT, QNT, Updater, Model> executor(params, tps, comm, model);
```

#### After
```cpp
ConjugateGradientParams cg_params{.max_iter = 100, .relative_tolerance = 1e-5,
                                   .residual_recompute_interval = 20};
StochasticReconfigurationParams sr_params{.cg_params = cg_params, .diag_shift = 0.001};
OptimizerParams opt_params = OptimizerFactory::CreateStochasticReconfiguration(
  1000, sr_params, 0.01);
VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);
VMCPEPSOptimizer<TenElemT, QNT, Updater, Model> executor(params, tps, comm, model);
```

## Files Modified

### Core Implementation
- `include/qlpeps/optimizer/optimizer_params.h`
- `include/qlpeps/optimizer/optimizer.h`  
- `include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h`

### Tests  
- `tests/test_algorithm/test_vmc_peps_optimizer.cpp`
- `tests/test_algorithm/test_optimizer.cpp`
- All integration tests

### Examples & Documentation
- `examples/migration_example.cpp`
- API documentation

## Impact

- No performance regression
- Minimal memory increase (scheduler objects)
- All tests pass
- Physics results unchanged
- MPI compatibility maintained

## Migration

1. Replace `VMCOptimizePara` → `VMCPEPSOptimizerParams`
2. Replace `VMCPEPSExecutor` → `VMCPEPSOptimizer`
3. Use `OptimizerFactory` for common configurations  
4. Specify all parameters explicitly
