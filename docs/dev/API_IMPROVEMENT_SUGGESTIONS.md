# API Improvement Proposal: Monte Carlo PEPS Measurement Parameters

## Date: 2025-08-06
## Status: PROPOSED (Not yet implemented)

## Summary

Proposed improvement to the `MCMeasurementPara` API to make parameter configuration clearer, safer, and more maintainable. This document combines the formal proposal with detailed implementation suggestions and examples.

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

## Proposed Solution: Explicit Configuration Management

**Single Responsibility Principle**: Each constructor should have one clear purpose.

```cpp
// NEW API: Explicit configuration management
class MCMeasurementParams {
public:
    // Constructor 1: Manual configuration with explicit warm-up state
    MCMeasurementParams(BMPSTruncatePara trunc_para,
                       size_t samples, size_t warmup, size_t sweeps,
                       Configuration initial_config,
                       bool is_warmed_up,  // EXPLICIT warm-up state
                       std::string tps_path);

    // Constructor 2: Load configuration from file
    MCMeasurementParams(BMPSTruncatePara trunc_para,
                       size_t samples, size_t warmup, size_t sweeps,
                       std::string config_path,  // Path to configuration file
                       std::string tps_path);

    // Constructor 3: Random configuration (for testing only)
    MCMeasurementParams(BMPSTruncatePara trunc_para,
                       size_t samples, size_t warmup, size_t sweeps,
                       std::vector<size_t> occupancy,  // Random configuration
                       size_t rows, size_t cols,
                       std::string tps_path);
};
```

### Benefits of New API

1. **Explicit Warm-up State**: Users must specify whether the configuration is warmed up
2. **No Default Path Confusion**: All paths must be explicitly specified
3. **Configuration-Data Consistency**: Clear separation between configuration and data loading
4. **Single Responsibility**: Each constructor has one clear purpose
5. **Better Error Messages**: Can provide specific error messages for configuration mismatches

## Proposed API Improvements

### 1. Builder Pattern for Parameter Construction

**Before:**
```cpp
MCMeasurementPara para(
    BMPSTruncatePara(4, 8, 1e-15, CompressMPSScheme::SVD_COMPRESS, 
                     std::make_optional<double>(1e-14), std::make_optional<size_t>(10)),
    50, 50, 1, std::vector<size_t>({1, 1, 2}), 2, 2);
```

**After:**
```cpp
// New API using builder pattern
auto params = MCMeasurementParams::Builder()
    .set_system_size(2, 2)                    // Lx, Ly
    .set_bmps_truncation(4, 8, 1e-15)         // Dpeps, Db_max, tolerance
    .set_mc_sampling(50, 50, 1)               // samples, warmup, sweeps
    .set_particle_configuration(1, 1, 2)      // 1 up, 1 down, 2 empty
    .build();

// For fermion models
auto fermion_params = MCMeasurementParams::Builder()
    .set_system_size(2, 2)
    .set_bmps_truncation(4, 8, 1e-15)
    .set_mc_sampling(50, 50, 1)
    .set_fermion_configuration(1, 1, 2)       // 1 up, 1 down, 2 empty
    .build();
```

### 2. Model-Specific Factory Methods

```cpp
// Predefined configurations for common models
auto heisenberg_params = MCMeasurementParams::for_heisenberg_model(2, 2, 4);
auto fermion_params = MCMeasurementParams::for_spinless_fermion_model(2, 2, 4);
auto tj_params = MCMeasurementParams::for_tj_model(2, 2, 4, 1, 1, 2); // 1 up, 1 down, 2 empty
```

### 3. Configuration Validation

```cpp
// Automatic validation
auto params = MCMeasurementParams::Builder()
    .set_system_size(2, 2)
    .set_particle_configuration(5, 5)  // 10 particles for 4 sites - ERROR!
    .build();  // Throws std::invalid_argument
```

### 4. Clear Parameter Names

```cpp
// Instead of anonymous numbers
MCMeasurementPara para(trunc_para, 50, 50, 1, particles, Ly, Lx);

// Use named parameters
auto params = MCMeasurementParams::Builder()
    .set_monte_carlo_samples(50)
    .set_warmup_samples(50)
    .set_sweep_repeats(1)
    .build();
```

### 5. Configuration Classes for Different Models

```cpp
// Model-specific configuration classes
class HeisenbergConfig {
    size_t Lx, Ly;
    double J;
    size_t Dpeps;
public:
    static MCMeasurementParams create_params(size_t Lx, size_t Ly, double J, size_t Dpeps);
};

class FermionConfig {
    size_t Lx, Ly;
    double t, t2;
    size_t Dpeps;
    std::vector<size_t> particle_numbers;
public:
    static MCMeasurementParams create_params(size_t Lx, size_t Ly, double t, double t2, 
                                           size_t Dpeps, const std::vector<size_t>& particles);
};
```

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

## Example Implementation

```cpp
class MCMeasurementParams {
public:
    class Builder {
    private:
        size_t Lx_ = 0, Ly_ = 0;
        size_t Dpeps_ = 4, Db_max_ = 8;
        double tolerance_ = 1e-15;
        size_t mc_samples_ = 50, warmup_samples_ = 50, sweep_repeats_ = 1;
        std::vector<size_t> particle_numbers_;
        bool is_fermion_ = false;
        
    public:
        Builder& set_system_size(size_t Lx, size_t Ly) {
            Lx_ = Lx; Ly_ = Ly;
            return *this;
        }
        
        Builder& set_bmps_truncation(size_t Dpeps, size_t Db_max, double tolerance) {
            Dpeps_ = Dpeps; Db_max_ = Db_max; tolerance_ = tolerance;
            return *this;
        }
        
        Builder& set_mc_sampling(size_t samples, size_t warmup, size_t sweeps) {
            mc_samples_ = samples; warmup_samples_ = warmup; sweep_repeats_ = sweeps;
            return *this;
        }
        
        Builder& set_particle_configuration(const std::vector<size_t>& particles) {
            particle_numbers_ = particles;
            is_fermion_ = false;
            return *this;
        }
        
        Builder& set_fermion_configuration(size_t n_up, size_t n_down, size_t n_empty) {
            particle_numbers_ = {n_up, n_down, n_empty};
            is_fermion_ = true;
            return *this;
        }
        
        MCMeasurementParams build() const {
            validate_parameters();
            return MCMeasurementParams(Lx_, Ly_, Dpeps_, Db_max_, tolerance_,
                                     mc_samples_, warmup_samples_, sweep_repeats_,
                                     particle_numbers_, is_fermion_);
        }
        
    private:
        void validate_parameters() const {
            if (Lx_ == 0 || Ly_ == 0) {
                throw std::invalid_argument("System size must be positive");
            }
            
            size_t total_sites = Lx_ * Ly_;
            size_t total_particles = std::accumulate(particle_numbers_.begin(), 
                                                   particle_numbers_.end(), 0);
            
            if (is_fermion_) {
                if (total_particles != total_sites) {
                    throw std::invalid_argument(
                        "For fermion models, total particles must equal total sites. "
                        "Got " + std::to_string(total_particles) + " particles for " + 
                        std::to_string(total_sites) + " sites");
                }
            } else {
                if (total_particles > total_sites) {
                    throw std::invalid_argument(
                        "Total particles cannot exceed total sites. "
                        "Got " + std::to_string(total_particles) + " particles for " + 
                        std::to_string(total_sites) + " sites");
                }
            }
        }
    };
    
    // Static factory methods for common models
    static MCMeasurementParams for_heisenberg_model(size_t Lx, size_t Ly, size_t Dpeps) {
        return Builder()
            .set_system_size(Lx, Ly)
            .set_bmps_truncation(Dpeps, 2*Dpeps, 1e-15)
            .set_mc_sampling(50, 50, 1)
            .set_particle_configuration({Lx*Ly/2, Lx*Ly/2})  // Half up, half down
            .build();
    }
    
    static MCMeasurementParams for_spinless_fermion_model(size_t Lx, size_t Ly, size_t Dpeps) {
        return Builder()
            .set_system_size(Lx, Ly)
            .set_bmps_truncation(Dpeps, 2*Dpeps, 1e-15)
            .set_mc_sampling(50, 50, 1)
            .set_fermion_configuration(Lx*Ly/2, Lx*Ly/2, 0)  // Half occupied, half empty
            .build();
    }
    
    static MCMeasurementParams for_tj_model(size_t Lx, size_t Ly, size_t Dpeps, 
                                           size_t n_up, size_t n_down, size_t n_empty) {
        return Builder()
            .set_system_size(Lx, Ly)
            .set_bmps_truncation(Dpeps, 2*Dpeps, 1e-15)
            .set_mc_sampling(50, 50, 1)
            .set_fermion_configuration(n_up, n_down, n_empty)
            .build();
    }
};
```

## Usage Examples

### Current (Messy)
```cpp
MCMeasurementPara para(
    BMPSTruncatePara(4, 8, 1e-15, CompressMPSScheme::SVD_COMPRESS, 
                     std::make_optional<double>(1e-14), std::make_optional<size_t>(10)),
    50, 50, 1, std::vector<size_t>({1, 1, 2}), 2, 2);
```

### Proposed (Clean)
```cpp
auto params = MCMeasurementParams::for_tj_model(2, 2, 4, 1, 1, 2);
```

### Migration Example

```cpp
// OLD API (error-prone)
MCMeasurementPara para(trunc_para, 50, 50, 1, 
                      Configuration(2, 2, {2, 2}));  // Default path used!

// NEW API (explicit)
MCMeasurementParams para(trunc_para, 50, 50, 1,
                        Configuration(2, 2, {2, 2}), 
                        false,  // Not warmed up
                        "path/to/tps/data");  // Explicit path
```

This makes the code much more readable and less error-prone!

## Related Issues

- Addresses configuration debugging difficulties
- Improves code readability and maintainability
- Reduces potential for configuration errors
- Solves the critical configuration-path mismatch problem 