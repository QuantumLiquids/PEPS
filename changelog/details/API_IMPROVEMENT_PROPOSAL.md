# API Improvement Proposal: Monte Carlo PEPS Measurement Parameters

## Date: 2025-08-06
## Status: PROPOSED (Not yet implemented)

## Summary

Proposed improvement to the `MCMeasurementPara` API to make parameter configuration clearer, safer, and more maintainable.

## Current Issues

The current `MCMeasurementPara` constructor has several problems:

1. **Too many parameters**: 8+ parameters in a single constructor
2. **Unclear parameter meanings**: Hard to understand what each number represents
3. **No validation**: Parameters aren't checked for consistency
4. **Model-specific confusion**: Different models need different particle configurations
5. **Hard to debug**: Configuration errors are difficult to trace

## CRITICAL ISSUE: Configuration-Path Mismatch

### Problem Description

The current API has a **critical flaw** that makes it error-prone and can cause silent failures:

```cpp
// Constructor 1: Random configuration with occupancy numbers
MCMeasurementPara(BMPSTruncatePara, samples, warmup, sweeps, 
                  std::vector<size_t> occupancy, rows, cols, path = default);

// Constructor 2: Manual configuration  
MCMeasurementPara(BMPSTruncatePara, samples, warmup, sweeps, 
                  Configuration config, path = default);
```

**Critical Issues:**
1. **Default path confusion**: If users forget to specify the path, the default path is used
2. **Configuration-path mismatch**: The manually specified configuration might not match the data at the default path
3. **Warm-up state ambiguity**: Users don't know if the loaded configuration is already warmed up
4. **Silent failures**: Configuration gets silently replaced during `SyncValidConfiguration_()` without user knowledge

### Impact

This issue caused the test failures we encountered where:
- Users specified a configuration with 2 up + 2 down spins
- The configuration was silently replaced with 1 up + 3 down spins
- Monte Carlo sampling failed (accept rate = 0.00)
- Energy measurements were incorrect

### Root Cause

The `SyncValidConfiguration_()` method in `MonteCarloPEPSBaseExecutor` replaces configurations when wave function amplitudes are invalid, but this happens silently without user notification.

## Proposed Changes

### 1. New Builder Pattern API

**Before:**
```cpp
MCMeasurementPara para(
    BMPSTruncatePara(4, 8, 1e-15, CompressMPSScheme::SVD_COMPRESS, 
                     std::make_optional<double>(1e-14), std::make_optional<size_t>(10)),
    50, 50, 1, std::vector<size_t>({1, 1, 2}), 2, 2);
```

**After:**
```cpp
auto params = MCMeasurementParams::Builder()
    .set_system_size(2, 2)
    .set_bmps_truncation(4, 8, 1e-15)
    .set_mc_sampling(50, 50, 1)
    .set_fermion_configuration(1, 1, 2)  // 1 up, 1 down, 2 empty
    .build();
```

### 2. Model-Specific Factory Methods

```cpp
// Predefined configurations for common models
auto heisenberg_params = MCMeasurementParams::for_heisenberg_model(2, 2, 4);
auto fermion_params = MCMeasurementParams::for_spinless_fermion_model(2, 2, 4);
auto tj_params = MCMeasurementParams::for_tj_model(2, 2, 4, 1, 1, 2);
```

### 3. Automatic Parameter Validation

The new API will automatically validate:
- System size must be positive
- Particle numbers must be consistent with system size
- Fermion models: total particles must equal total sites
- Boson models: total particles cannot exceed total sites

## Migration Strategy

### Phase 1: Add New API (Backward Compatible)
- Add new `MCMeasurementParams` class with builder pattern
- Keep existing `MCMeasurementPara` constructors working
- Add deprecation warnings to old API

### Phase 2: Update Tests and Examples
- Convert all tests to use new API
- Update documentation and examples
- Create migration guide

### Phase 3: Deprecate Old API
- Mark old constructors as `[[deprecated]]`
- Remove in next major version

## Benefits

1. **Clarity**: Parameter meanings are explicit and self-documenting
2. **Safety**: Automatic validation prevents configuration errors
3. **Maintainability**: Easy to add new parameters without breaking existing code
4. **Debugging**: Clear error messages for invalid configurations
5. **Model-specific**: Tailored configurations for different physics models

## Impact on Users

### Breaking Changes
- None in Phase 1 (backward compatible)
- Old API will be deprecated in Phase 2
- Old API will be removed in Phase 3

### Required Changes
- Users will need to update their code to use the new API
- Migration guide will be provided
- Examples will be updated

## Implementation Timeline

- **Phase 1**: 1-2 weeks (add new API)
- **Phase 2**: 2-3 weeks (update tests and docs)
- **Phase 3**: Next major release (deprecate old API)

## Files to Modify

- `include/qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h`
- `include/qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.cpp`
- All test files using `MCMeasurementPara`
- Documentation files

## Related Issues

- Addresses configuration debugging difficulties
- Improves code readability and maintainability
- Reduces potential for configuration errors 