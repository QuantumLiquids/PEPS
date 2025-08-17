# Monte Carlo PEPS API Refactor Proposal

## Date: 2025-01-29  
## Status: APPROVED FOR IMPLEMENTATION

## Executive Summary

**Problem**: Current Monte Carlo PEPS API is a mess with 8+ parameter constructors, silent configuration replacement, and inconsistent parameter structures.

**Solution**: Unify all parameter structures, eliminate special cases, give users complete control over their data.

**Approach**: Direct refactor without legacy compatibility (users can use v0.0.1 if needed).

## Core Design Changes

### 1. Unified Parameter Structure
```cpp
// Single MonteCarloParams for all use cases
struct MonteCarloParams {
  size_t num_samples;
  size_t num_warmup_sweeps; 
  size_t sweeps_between_samples;
  Configuration initial_config;  // User-provided
  bool is_warmed_up;            // Explicit state
};

// Simplified PEPSParams 
struct PEPSParams {
  BMPSTruncatePara truncate_para;
  // Remove wavefunction_path - user handles I/O
};

// Unified measurement parameters
struct MCMeasurementParams {
  MonteCarloParams mc_params;
  PEPSParams peps_params;
};
```

### 2. Single Responsibility Constructors
```cpp
// Only one constructor: user provides TPS
MonteCarloPEPSBaseExecutor(const SITPST& sitps,
                           const MonteCarloParams& mc_params, 
                           const PEPSParams& peps_params,
                           const MPI_Comm& comm);

// State access
const Configuration& GetCurrentConfiguration() const;
const SITPST& GetCurrentTPS() const;

// Optional persistence  
void DumpConfiguration(const std::string& path) const;
void DumpTPS(const std::string& path) const;
```

## Implementation Tasks

### Phase 1: Core Parameter Refactor (3 days)

**Files to Modify:**
1. `include/qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h`
   - ✅ Update `MonteCarloParams` structure
   - ✅ Simplify `PEPSParams` structure  
   - ✅ Add `MCMeasurementParams`
   - ❌ Remove `MCMeasurementPara` completely

2. `include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h`
   - ✅ Verify compatibility with new `MonteCarloParams`/`PEPSParams`
   - ✅ Update conversion operators if needed

**Testing:**
- Unit tests for parameter construction and conversion
- Verify no compilation errors in dependent headers

### Phase 2: Base Class Refactor (2 days)

**Files to Modify:**
1. `include/qlpeps/algorithm/vmc_update/monte_carlo_peps_base.h`
   - ❌ Remove constructor with TPS path loading
   - ✅ Keep only user-provided TPS constructor
   - ❌ Remove complex `InitConfigs_()` with fallback logic
   - ✅ Add simple `Initialize()` with separated responsibilities 
   - ❌ Remove silent configuration replacement in `SyncValidConfiguration_()`
   - ✅ Add `GetCurrentConfiguration()`, `GetCurrentTPS()`
   - ✅ Add `DumpConfiguration()`, `DumpTPS()`

**Testing:**
- Unit tests for direct initialization
- Verify proper error handling for invalid configurations
- Test state access methods

### Phase 3: Measurement Executor Refactor (2 days)

**Files to Modify:**
1. `include/qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h`
   - ✅ Update to use `MCMeasurementParams` instead of `MCMeasurementPara`
   - ✅ Update all measurement functions to new parameter structure
   - ✅ Remove any TPS loading logic

**Testing:**
- Integration tests for measurement functionality
- Verify measurement accuracy unchanged

### Phase 4: Optimizer Executor Refactor (1 day)

**Files to Modify:**
1. `include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h`
2. `include/qlpeps/algorithm/vmc_update/vmc_peps_optimizer_impl.h`
   - ✅ Verify compatibility with unified parameter structure
   - ✅ Update constructor calls to base class

**Testing:**
- Integration tests for optimization functionality
- Performance regression tests

### Phase 5: Update All Tests (3-4 days)

**Test Files to Update (16 files):**

**Unit Tests:**
- `tests/test_algorithm/test_mc_peps_measure.cpp`
- `tests/test_algorithm/test_vmc_peps_optimizer.cpp`
- `tests/test_algorithm/test_exact_sum_optimization.cpp`

**Integration Tests (6 files):**
- `tests/integration_tests/test_square_heisenberg.cpp`
- `tests/integration_tests/test_square_j1j2_xxz.cpp`  
- `tests/integration_tests/test_square_nn_spinless_free_fermion.cpp`
- `tests/integration_tests/test_square_tj_model.cpp`
- `tests/integration_tests/test_triangle_heisenberg.cpp`
- `tests/integration_tests/test_triangle_j1j2_heisenberg.cpp`
- `tests/integration_tests/integration_test_framework.h`

**Slow Tests (3 files):**
- `tests/slow_tests/test_boson_mc_peps_measure.cpp`
- `tests/slow_tests/test_fermion_mc_peps_measure.cpp`  
- `tests/slow_tests/test_fermion_mc_updater_large_size.cpp`

**Deprecated Tests:**
- `tests/test_deprecated/test_loop_update_deprecated.cpp` (update or mark as broken)

**Update Strategy for Each Test:**
```cpp
// OLD
MCMeasurementPara para(trunc_para, 50, 50, 1, {1,1,2}, 2, 2);

// NEW  
auto config = Configuration(2, 2);
config.Random({1, 1, 2});
MonteCarloParams mc_params(50, 50, 1, config, false);
PEPSParams peps_params(trunc_para);
MCMeasurementParams params(mc_params, peps_params);

// Load TPS explicitly
auto sitps = SplitIndexTPS<FPTYPE, U1QN>::LoadFromFile("path/to/tps");
MeasurementExecutor executor(sitps, mc_params, peps_params, MPI_COMM_WORLD);
```

### Phase 6: User Utility Classes (2 days)

**New Files to Create:**
1. `include/qlpeps/utility/configuration_manager.h`
2. `include/qlpeps/utility/tps_manager.h`

```cpp
// Configuration management utilities
class ConfigurationManager {
public:
  static Configuration LoadFromFile(const std::string& path);
  static void SaveToFile(const Configuration& config, const std::string& path);
  static Configuration CreateRandom(size_t Lx, size_t Ly, const std::vector<size_t>& occupancy);
  static Configuration CreateForModel(const std::string& model_type, size_t Lx, size_t Ly, const std::vector<double>& params);
};

// TPS management utilities
class TPSManager {
public:
  template<typename TenElemT, typename QNT>
  static SplitIndexTPS<TenElemT, QNT> LoadFromFile(const std::string& path);
  
  template<typename TenElemT, typename QNT>  
  static void SaveToFile(const SplitIndexTPS<TenElemT, QNT>& tps, const std::string& path);
};
```

**Testing:**
- Unit tests for utility functions
- Integration tests with actual file I/O

### Phase 7: Documentation Update (1 day)

**Files to Update:**
1. `docs/tutorial/TOP_LEVEL_APIs.md`
   - ✅ Update API examples
   - ✅ Add migration guide from old API

2. `docs/tutorial/MIGRATION_SUMMARY.md`
   - ✅ Document breaking changes
   - ✅ Provide conversion examples

3. `examples/migration_example.cpp`
   - ✅ Update to use new API
   - ✅ Show best practices

4. **New Documentation:**
   - `docs/dev/API_DESIGN_PRINCIPLES.md` - Document design philosophy
   - `docs/tutorial/CONFIGURATION_MANAGEMENT.md` - User guide for configuration handling

## Testing Strategy

### 1. Unit Tests
- **Parameter structures**: Construction, conversion, validation
- **Base class methods**: State access, initialization, error handling
- **Utility classes**: File I/O, configuration generation

### 2. Integration Tests  
- **End-to-end workflows**: Load TPS → Configure → Run measurement/optimization
- **Model-specific tests**: Heisenberg, t-J, spinless fermion models
- **Error scenarios**: Invalid configurations, missing files

### 3. Performance Tests
- **Regression tests**: Ensure no performance degradation
- **Memory usage**: Verify no memory leaks with new object lifetimes

### 4. Test Data Management
Following existing practice of placing test data in `build/` directory:
```bash
# Test data should be generated in build directory
mkdir -p build/test_data/
# Configuration files, TPS files generated during testing
```

## Risk Assessment

### High Risk
- **Break all existing code**: Since we're not maintaining compatibility
- **Test failures**: 16+ test files need updating

### Medium Risk  
- **Performance regression**: New object creation patterns
- **Memory leaks**: Changed object lifetime management

### Low Risk
- **API usability**: New API is cleaner and more explicit

## Mitigation Strategies

1. **Incremental Implementation**: Complete each phase with full testing before moving to next
2. **Automated Testing**: Run full test suite after each file modification  
3. **Performance Monitoring**: Benchmark before/after for critical paths
4. **Documentation**: Update docs in parallel with code changes

## Timeline Estimate

**Total: 12-14 days**

- Phase 1 (Core Parameters): 3 days
- Phase 2 (Base Class): 2 days  
- Phase 3 (Measurement): 2 days
- Phase 4 (Optimizer): 1 day
- Phase 5 (Tests): 3-4 days
- Phase 6 (Utilities): 2 days
- Phase 7 (Documentation): 1 day

## Success Criteria

1. ✅ All tests pass with new API
2. ✅ No performance regression (< 5% slowdown acceptable)
3. ✅ Clean, consistent parameter structures across all executors
4. ✅ User complete control over configuration and TPS data
5. ✅ Clear error messages for invalid inputs
6. ✅ Updated documentation with examples

## Breaking Changes Summary

**Removed:**
- `MCMeasurementPara` class
- TPS loading constructors in base class
- Silent configuration replacement logic
- `config_path` and `alternative_init_config` in `MonteCarloParams`

**Added:**
- `MCMeasurementParams` class
- `is_warmed_up` field in `MonteCarloParams`  
- State access methods (`GetCurrentConfiguration`, `GetCurrentTPS`)
- Utility classes (`ConfigurationManager`, `TPSManager`)

**Changed:**
- All measurement and optimization APIs now use unified parameter structure
- User must explicitly load TPS and configuration data
- Explicit warm-up state management

---

**Implementation Principle: "Do one thing and do it well."**

Each component has a single, clear responsibility. No magic. No guessing. User owns their data.
