# API Refactor Proposal: Optimizer Parameters Modernization

## Date: 2025-01-29
## Status: PROPOSED (Not yet implemented)

## Summary

Comprehensive refactoring of the `OptimizerParams` API to follow modern C++ and machine learning best practices. This refactor addresses Google C++ style violations, improves extensibility for future algorithms (Adam, L-BFGS, etc.), and introduces learning rate scheduling capabilities similar to PyTorch's design.

**Implementation Strategy**: Direct API replacement without backward compatibility, taking advantage of the active development stage with no external users. This approach enables a clean, efficient transition to the optimal design.

## Current Issues

The existing `OptimizerParams` structure has several critical problems:

### 1. **Google C++ Style Violations**
```cpp
// CURRENT: Violates Google style with default parameters
CoreParams(size_t max_iter, double energy_tol = 1e-15, double grad_tol = 1e-30,
           size_t patience = 20, const std::vector<double> &steps = {0.1},
           bool logging = true, size_t log_freq = 10, const std::string &log_path = "")
```

### 2. **Inconsistent Terminology**
- Uses "step_lengths" instead of standard ML term "learning_rate"  
- Makes future algorithm integration (Adam, etc.) awkward
- Conflicts with physics literature where "step length" means parameter change magnitude

### 3. **Poor Extensibility**
- `std::vector<double> step_lengths` is algorithm-agnostic but makes no sense for some methods
- No support for learning rate scheduling (essential for PEPS optimization)
- Hard-coded algorithm selection via enum

### 4. **Confusing API Design**
- Nested `CoreParams` structure adds unnecessary complexity
- Algorithm-specific parameters scattered across different places
- No clear separation between base settings and algorithm specifics

### 5. **Unused Parameters** ✅ **COMPLETED**
- ~~`enable_logging`, `log_frequency`, `log_file_path` are defined but completely unused~~ **REMOVED**
- ~~Logging always happens to stdout regardless of these settings~~ **FIXED**
- ~~Dead code that adds complexity without functionality~~ **ELIMINATED**

### 6. **Missing Modern Features**
- No learning rate scheduling (critical for convergence)
- No support for modern optimizers (Adam, RMSprop, L-BFGS)
- No parameter validation

## Technical Considerations

### Physics Requirements
- **Stochastic Reconfiguration**: Learning rate varies (0.1, 0.01, 0.001), needs CG parameters and diagonal shift
- **Gradient Methods**: Need adaptive learning rates and line search for stability
- **Second-Order Methods**: Different parameter requirements (history size for L-BFGS)

### ML Framework Patterns
- **PyTorch**: Separate optimizer constructors + lr_scheduler system
- **TensorFlow**: OptimizerConfig with algorithm-specific parameters
- **Scientific Computing**: Settings/Parameters naming conventions

### C++ Best Practices
- **Google Style**: No default parameters, explicit construction
- **Type Safety**: Use variants for algorithm selection
- **Extensibility**: Easy addition of new algorithms

## Proposed Solution

### New Architecture Overview

```cpp
// Base parameters for all optimization algorithms
struct BaseParams {
  size_t max_iterations;
  double energy_tolerance; 
  double gradient_tolerance;
  size_t plateau_patience;
  double learning_rate;                                    // Universal parameter for all algorithms
  std::unique_ptr<LearningRateScheduler> lr_scheduler;     // Optional learning rate scheduling
  
  // No default constructor - force explicit specification
  BaseParams(size_t max_iter, double energy_tol, double grad_tol,
             size_t patience, double learning_rate,
             std::unique_ptr<LearningRateScheduler> scheduler = nullptr);
};

// Learning rate scheduler interface (like PyTorch)
class LearningRateScheduler {
public:
  virtual ~LearningRateScheduler() = default;
  virtual double GetLearningRate(size_t iteration, double current_energy = 0.0) const = 0;
  virtual void Step() {}  // Called after each optimization step
  virtual std::unique_ptr<LearningRateScheduler> Clone() const = 0;
};

// Algorithm-specific parameter structures
struct SGDParams {
  double momentum;          // For future SGD with momentum
  bool nesterov;           // For Nesterov acceleration
  
  SGDParams(double momentum = 0.0, bool nesterov = false) 
    : momentum(momentum), nesterov(nesterov) {}
};

struct AdamParams {
  double beta1;         // First moment decay rate  
  double beta2;         // Second moment decay rate
  double epsilon;       // Numerical stability
  double weight_decay;  // L2 regularization
  
  AdamParams(double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 0.0)
    : beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd) {}
};

struct StochasticReconfigurationParams {
  ConjugateGradientParams cg_params;                     // CG solver configuration  
  bool normalize_update;                                 // For normalized SR
  double adaptive_diagonal_shift;                        // Dynamic regularization
  
  StochasticReconfigurationParams(const ConjugateGradientParams& cg_params,
                                 bool normalize = false, double adaptive_shift = 0.0)
    : cg_params(cg_params), normalize_update(normalize), adaptive_diagonal_shift(adaptive_shift) {}
};

struct LBFGSParams {
  size_t history_size;         // Limited memory size (typically 5-20)
  double tolerance_grad;       // Gradient tolerance for line search
  double tolerance_change;     // Parameter change tolerance  
  size_t max_eval;            // Max function evaluations per step
  
  LBFGSParams(size_t hist = 10, double tol_grad = 1e-5,
              double tol_change = 1e-9, size_t max_eval = 20)
    : history_size(hist), tolerance_grad(tol_grad),
      tolerance_change(tol_change), max_eval(max_eval) {}
};

// Type-safe algorithm parameter variant
using AlgorithmParams = std::variant<
  SGDParams,
  AdamParams,
  StochasticReconfigurationParams,
  LBFGSParams
>;

// Main optimizer parameters structure  
struct OptimizerParams {
  BaseParams base_params;
  AlgorithmParams algorithm_params;
  
  // Constructor requires explicit parameters (no defaults)
  OptimizerParams(const BaseParams& base_params, const AlgorithmParams& algo_params)
    : base_params(base_params), algorithm_params(algo_params) {}
    
  // Template getter for type safety
  template<typename T>
  const T& GetAlgorithmParams() const {
    return std::get<T>(algorithm_params);
  }
  
  // Algorithm identification
  template<typename T>
  bool IsAlgorithm() const {
    return std::holds_alternative<T>(algorithm_params);
  }
};
```

### Learning Rate Schedulers Implementation

```cpp
// Common scheduler implementations
class ConstantLR : public LearningRateScheduler {
  double learning_rate_;
public:
  explicit ConstantLR(double lr) : learning_rate_(lr) {}
  double GetLearningRate(size_t iteration, double current_energy) const override {
    return learning_rate_;
  }
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<ConstantLR>(learning_rate_);
  }
};

class ExponentialDecayLR : public LearningRateScheduler {
  double initial_lr_;
  double decay_rate_;
  size_t decay_steps_;
  
public:
  ExponentialDecayLR(double initial_lr, double decay_rate, size_t decay_steps)
    : initial_lr_(initial_lr), decay_rate_(decay_rate), decay_steps_(decay_steps) {}
    
  double GetLearningRate(size_t iteration, double current_energy) const override {
    return initial_lr_ * std::pow(decay_rate_, iteration / static_cast<double>(decay_steps_));
  }
  
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<ExponentialDecayLR>(initial_lr_, decay_rate_, decay_steps_);
  }
};

class StepLR : public LearningRateScheduler {
  double initial_lr_;
  double gamma_;
  size_t step_size_;
  
public:
  StepLR(double initial_lr, size_t step_size, double gamma = 0.1)
    : initial_lr_(initial_lr), step_size_(step_size), gamma_(gamma) {}
    
  double GetLearningRate(size_t iteration, double current_energy) const override {
    return initial_lr_ * std::pow(gamma_, iteration / step_size_);
  }
  
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    return std::make_unique<StepLR>(initial_lr_, step_size_, gamma_);
  }
};

// Physics-aware scheduler for energy plateaus
class PlateauLR : public LearningRateScheduler {
private:
  mutable double current_lr_;
  double factor_;
  size_t patience_;
  mutable size_t patience_counter_;
  mutable double best_energy_;
  double threshold_;
  
public:
  PlateauLR(double initial_lr, double factor = 0.5, size_t patience = 10, double threshold = 1e-4)
    : current_lr_(initial_lr), factor_(factor), patience_(patience), 
      patience_counter_(0), best_energy_(std::numeric_limits<double>::max()), threshold_(threshold) {}
      
  double GetLearningRate(size_t iteration, double current_energy) const override {
    // Update learning rate based on energy plateau detection
    if (current_energy < best_energy_ - threshold_) {
      best_energy_ = current_energy;
      patience_counter_ = 0;
    } else {
      patience_counter_++;
      if (patience_counter_ >= patience_) {
        current_lr_ *= factor_;
        patience_counter_ = 0;
        best_energy_ = current_energy;  // Reset reference
      }
    }
    return current_lr_;
  }
  
  std::unique_ptr<LearningRateScheduler> Clone() const override {
    auto clone = std::make_unique<PlateauLR>(current_lr_, factor_, patience_, threshold_);
    clone->best_energy_ = best_energy_;
    clone->patience_counter_ = patience_counter_;
    return clone;
  }
};
```

### User-Friendly Factory Methods

```cpp
class OptimizerFactory {
public:
  // Stochastic Reconfiguration (most common for PEPS)
  static OptimizerParams CreateStochasticReconfiguration(
      size_t max_iterations,
      double energy_tolerance,
      const ConjugateGradientParams& cg_params,
      double learning_rate,
      std::unique_ptr<LearningRateScheduler> scheduler = nullptr) {
    
    BaseParams base_params(max_iterations, energy_tolerance, 1e-15, 20, 
                          learning_rate, std::move(scheduler));
    StochasticReconfigurationParams sr_params(cg_params);
    return OptimizerParams(base_params, sr_params);
  }
  
  // SGD with exponential decay (common for gradient methods)
  static OptimizerParams CreateSGDWithDecay(
      size_t max_iterations,
      double energy_tolerance,
      double initial_learning_rate,
      double decay_rate = 0.95,
      size_t decay_steps = 100) {
    
    auto scheduler = std::make_unique<ExponentialDecayLR>(initial_learning_rate, decay_rate, decay_steps);
    BaseParams base_params(max_iterations, energy_tolerance, 1e-15, 20, 
                          initial_learning_rate, std::move(scheduler));
    SGDParams sgd_params{};
    return OptimizerParams(base_params, sgd_params);
  }
  
  // Adam optimizer (future algorithm)
  static OptimizerParams CreateAdam(
      size_t max_iterations,
      double energy_tolerance,
      double learning_rate,
      double beta1 = 0.9,
      double beta2 = 0.999,
      std::unique_ptr<LearningRateScheduler> scheduler = nullptr) {
    
    BaseParams base_params(max_iterations, energy_tolerance, 1e-15, 20, 
                          learning_rate, std::move(scheduler));
    AdamParams adam_params(beta1, beta2);
    return OptimizerParams(base_params, adam_params);
  }
  
  // L-BFGS for second-order optimization (future algorithm)
  static OptimizerParams CreateLBFGS(
      size_t max_iterations,
      double energy_tolerance,
      double learning_rate,
      size_t history_size = 10) {
    
    BaseParams base_params(max_iterations, energy_tolerance, 1e-15, 20, learning_rate);
    LBFGSParams lbfgs_params(history_size);
    return OptimizerParams(base_params, lbfgs_params);
  }
};
```

### Builder Pattern for Complex Configurations

```cpp
class OptimizerParamsBuilder {
private:
  std::optional<BaseParams> base_params_;
  std::optional<AlgorithmParams> algorithm_params_;
  
public:
  // Base parameter setters
  OptimizerParamsBuilder& SetMaxIterations(size_t max_iter) {
    if (!base_params_) {
      base_params_ = BaseParams(max_iter, 1e-10, 1e-15, 20, 0.01);  // Default learning rate
    } else {
      base_params_->max_iterations = max_iter;
    }
    return *this;
  }
  
  OptimizerParamsBuilder& SetEnergyTolerance(double tol) {
    if (!base_params_) {
      base_params_ = BaseParams(1000, tol, 1e-15, 20, 0.01);  // Default learning rate
    } else {
      base_params_->energy_tolerance = tol;
    }
    return *this;
  }
  
  OptimizerParamsBuilder& SetLearningRate(double learning_rate, 
                                         std::unique_ptr<LearningRateScheduler> scheduler = nullptr) {
    if (!base_params_) {
      base_params_ = BaseParams(1000, 1e-10, 1e-15, 20, learning_rate, std::move(scheduler));
    } else {
      base_params_->learning_rate = learning_rate;
      if (scheduler) {
        base_params_->lr_scheduler = std::move(scheduler);
      }
    }
    return *this;
  }
  
  OptimizerParamsBuilder& SetPlateauPatience(size_t patience) {
    if (!base_params_) {
      base_params_ = BaseParams(1000, 1e-10, 1e-15, patience, 0.01);  // Default learning rate
    } else {
      base_params_->plateau_patience = patience;
    }
    return *this;
  }
  
  // Algorithm parameter setters
  OptimizerParamsBuilder& WithSGD(double momentum = 0.0, bool nesterov = false) {
    SGDParams sgd_params(momentum, nesterov);
    algorithm_params_ = sgd_params;
    return *this;
  }
  
  OptimizerParamsBuilder& WithStochasticReconfiguration(const ConjugateGradientParams& cg_params,
                                                       bool normalize = false, 
                                                       double adaptive_shift = 0.0) {
    StochasticReconfigurationParams sr_params(cg_params, normalize, adaptive_shift);
    algorithm_params_ = sr_params;
    return *this;
  }
  
  OptimizerParamsBuilder& WithAdam(double beta1 = 0.9, double beta2 = 0.999, 
                                  double epsilon = 1e-8, double weight_decay = 0.0) {
    AdamParams adam_params(beta1, beta2, epsilon, weight_decay);
    algorithm_params_ = adam_params;
    return *this;
  }
  
  OptimizerParams Build() const {
    if (!base_params_ || !algorithm_params_) {
      throw std::invalid_argument("Both base parameters and algorithm parameters must be specified");
    }
    return OptimizerParams(*base_params_, *algorithm_params_);
  }
};
```

## Usage Examples

### Example 1: Stochastic Reconfiguration with Plateau Scheduler
```cpp
// Create CG parameters for SR
ConjugateGradientParams cg_params(100, 1e-5, 10, 0.01);

// Create plateau-based learning rate scheduler
auto plateau_scheduler = std::make_unique<PlateauLR>(0.01, 0.5, 20, 1e-6);

// Create optimizer parameters using factory method
auto params = OptimizerFactory::CreateStochasticReconfiguration(
    1000,       // max_iterations
    1e-10,      // energy_tolerance
    cg_params,  // CG solver parameters
    0.01,       // learning_rate (system dependent: 0.1, 0.01, 0.001)
    std::move(plateau_scheduler)  // learning rate scheduler
);

// Use with optimizer
Optimizer<double, U1QN> optimizer(params, comm, rank, size);
```

### Example 2: SGD with Exponential Decay
```cpp
// Simple factory method call
auto params = OptimizerFactory::CreateSGDWithDecay(
    1000,  // max_iterations  
    1e-10, // energy_tolerance
    0.1,   // initial_learning_rate
    0.95,  // decay_rate
    50     // decay_steps
);

Optimizer<double, U1QN> optimizer(params, comm, rank, size);
```

### Example 3: Custom Configuration with Builder
```cpp
// Complex configuration using builder pattern
ConjugateGradientParams cg_params(200, 1e-6, 15, 0.005);
auto step_scheduler = std::make_unique<StepLR>(0.05, 100, 0.8);

auto params = OptimizerParamsBuilder()
    .SetMaxIterations(2000)
    .SetEnergyTolerance(1e-12)
    .SetLearningRate(0.05, std::move(step_scheduler))
    .SetPlateauPatience(50)
    .WithStochasticReconfiguration(cg_params)
    .Build();

Optimizer<double, U1QN> optimizer(params, comm, rank, size);
```

### Example 4: Future Adam Optimizer
```cpp
// When Adam is implemented, the API will be ready
auto cosine_scheduler = std::make_unique<CosineAnnealingLR>(0.001, 500);

auto params = OptimizerFactory::CreateAdam(
    1000,   // max_iterations
    1e-10,  // energy_tolerance  
    0.001,  // learning_rate
    0.9,    // beta1
    0.999,  // beta2
    std::move(cosine_scheduler)
);

Optimizer<double, U1QN> optimizer(params, comm, rank, size);
```

## Implementation Plan

**Note**: Since this is an active development project with no external users, we can implement a direct replacement approach for maximum efficiency and code cleanliness. This reduces the timeline from 10 weeks to 6 weeks by eliminating migration complexity.

**Accurate Scope**: The refactor affects **~284 matches across ~22 files** (focused only on current OptimizerParams-related structures, excluding legacy):

**Core Structures to Refactor:**
- `OptimizerParams`: 126 matches, 15 files
- `VMCPEPSOptimizerParams`: 39 matches, 11 files  
- `BaseParams`: 46 matches, 4 files
- `ConjugateGradientParams`/`AdaGradParams`: 73 matches, 21 files

**Excluded from count**: `VMCOptimizePara` (40 matches, 14 files) - legacy structure to be removed, not relevant to this refactor.

### Phase 1: Core Infrastructure Implementation (Week 1-2)
1. **Create new parameter structures**
   - `BaseParams` struct with unified learning rate and no default constructor
   - Algorithm-specific parameter structs (SGD, Adam, SR, L-BFGS)
   - `AlgorithmParams` variant type
   - New `OptimizerParams` struct

2. **Implement learning rate scheduler interface**
   - `LearningRateScheduler` base class with virtual interface
   - Core implementations: `ConstantLR`, `ExponentialDecayLR`, `StepLR`, `PlateauLR`
   - Physics-aware `PlateauLR` for energy-based scheduling

3. **Create user-friendly API**
   - `OptimizerFactory` class with physics-aware factory methods
   - `OptimizerParamsBuilder` for complex configurations

### Phase 2: Direct Optimizer Replacement (Week 3)
1. **Replace Optimizer class implementation**
   - Update constructor to accept new `OptimizerParams` structure
   - Implement visitor pattern for algorithm-specific parameter handling
   - Integrate learning rate scheduler into optimization loops
   - Remove old parameter handling code entirely

2. **Update algorithm implementations**
   - Modify existing SGD and SR implementations to use unified learning rate
   - Add infrastructure for future Adam/L-BFGS algorithms
   - Implement scheduler integration in optimization steps

### Phase 3: Direct Dependent Code Updates (Week 4)
1. **Replace VMCPEPSOptimizerParams**
   - Direct conversion to use new `OptimizerParams` structure
   - Update all factory methods and constructors
   - Remove old parameter structures completely

2. **Update all test files (atomic replacement)**
   - `test_optimizer.cpp`: Replace parameter construction patterns
   - `test_vmc_peps_optimizer.cpp`: Use new factory methods throughout
   - Integration tests: Update parameter initialization
   - Slow tests: Convert existing configurations to new API

### Phase 4: Documentation and Validation (Week 5)
1. **Update documentation**
   - Doxygen comments for all new classes and methods
   - Update API examples to show new patterns
   - Performance optimization guidelines

2. **Update examples and comprehensive testing**
   - Update `migration_example.cpp` to showcase new API
   - Integration test examples with new parameter system
   - Unit tests for all new components
   - Performance regression testing
   - Memory leak validation

### Phase 5: Final Cleanup and Release (Week 6)
1. **Complete cleanup**
   - Remove all old parameter code (`CoreParams`, old constructors, etc.)
   - Update enum-based algorithm selection
   - Clean up any remaining legacy references

2. **Release preparation**
   - Final testing with real physics problems
   - Documentation review and cleanup
   - Version tagging and release notes

## Migration Strategy

**Direct Replacement Approach**: Since this is an active development project with no external users, we implement a clean, direct replacement strategy.

### Implementation Strategy
1. **Feature Branch Development**
   ```bash
   git checkout -b optimizer-params-refactor
   # Implement entire new API
   # Update all dependent code atomically
   # Comprehensive testing
   git merge to main  # Clean, complete transition
   ```

2. **Atomic Code Replacement**
   - Replace old parameter structures entirely
   - Update all usage sites in same commit/PR
   - No deprecated code or compatibility layers needed
   - Single, clean API transition

3. **Documentation Strategy**
   ```markdown
   ## Breaking Change Notice (v2.0)
   - Replaced: `CoreParams` → `BaseParams` with unified learning_rate
   - Simplified: Removed unused logging parameters ✅ **COMPLETED**
   - Enhanced: Added learning rate scheduling support
   - Updated: All factory methods and builders
   ```

### Benefits of Direct Replacement
- ✅ **Faster Development**: No time spent on compatibility layers
- ✅ **Cleaner Codebase**: No deprecated code or legacy baggage  
- ✅ **Better Testing**: Focus testing on the correct, new API
- ✅ **Team Agility**: Quick adaptation and immediate benefits

## Risk Assessment

### Low Risk (Development Project Advantages)
- **Code Updates**: Internal team can adapt quickly, no external dependencies
- **Test Updates**: Comprehensive test suite can be updated atomically
- **Team Coordination**: Small, focused team can implement changes efficiently

### Medium Risk  
- **Performance Impact**: New virtual function calls in schedulers (likely negligible)
- **Memory Overhead**: Smart pointers and scheduler objects (minimal impact)
- **Implementation Bugs**: New code always carries risk of bugs

### Very Low Risk
- **Breaking External Users**: No external users to break
- **API Stability**: No compatibility promises to maintain
- **Complex Migration**: Direct replacement eliminates migration complexity

### Mitigation Strategies
1. **Comprehensive Testing**: Test all physics problems with new parameters before merge
2. **Performance Monitoring**: Benchmark critical optimization paths
3. **Atomic Implementation**: Everything updated together in single PR
4. **Code Review**: Thorough review of all changes before merge

## Learning Rate Management Design

### Responsibility Assignment

**Decision**: The **Optimizer** manages learning rate scheduling logic, not the `OptimizerParams`.

### Rationale

The Optimizer is responsible for scheduling because:
- ✅ **Has optimization context**: iteration count, current energy, convergence state
- ✅ **Clear ownership**: Optimizer controls the optimization process
- ✅ **Stateless parameters**: OptimizerParams remain immutable and thread-safe
- ✅ **Energy-aware scheduling**: Can pass current energy to plateau-based schedulers

### Implementation Pattern

```cpp
template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::OptimizationResult
Optimizer<TenElemT, QNT>::IterativeOptimize(...) {
  
  for (size_t iter = 0; iter < params_.base_params.max_iterations; ++iter) {
    // Evaluate energy and gradient
    auto [current_energy, current_gradient, current_error] = energy_evaluator(current_state);
    
    // Determine learning rate for this iteration
    double learning_rate = GetCurrentLearningRate(iter, Real(current_energy));
    
    // Apply optimization update with current learning rate
    updated_state = ApplyUpdate(current_state, current_gradient, learning_rate);
    
    // ... rest of optimization logic
  }
}

private:
double Optimizer<TenElemT, QNT>::GetCurrentLearningRate(size_t iteration, double current_energy) {
  double learning_rate = params_.base_params.learning_rate;  // Default/base rate
  
  if (params_.base_params.lr_scheduler) {
    learning_rate = params_.base_params.lr_scheduler->GetLearningRate(iteration, current_energy);
    
    // Update scheduler internal state (for PlateauLR, etc.)
    params_.base_params.lr_scheduler->Step();
  }
  
  return learning_rate;
}
```

### Algorithm Integration

All algorithms use the same learning rate interface:

```cpp
// Algorithm-specific parameter updates
switch (algorithm_type) {
  case SGD:
    updated_state = current_state - learning_rate * gradient;
    break;
    
  case Adam:
    // Adam uses learning_rate as global scaling factor
    adam_update = adam_algorithm.ComputeUpdate(gradient, learning_rate);
    updated_state = current_state + adam_update;
    break;
    
  case StochasticReconfiguration:
    // SR uses learning_rate (commonly 0.01-0.1), can be scheduled for adaptation
    natural_gradient = SolveCGSystem(gradient);
    updated_state = current_state - learning_rate * natural_gradient;
    break;
}
```

### Comparison to Previous Design

```cpp
// OLD (limited but simple):
double step_length = params_.base_params.step_lengths[iter % step_lengths.size()];

// NEW (flexible and physics-aware):
double learning_rate = GetCurrentLearningRate(iter, current_energy);
```

The new system maintains simplicity for constant learning rates while enabling sophisticated adaptive scheduling when needed.

## Benefits Summary

### Key Design Changes
- ✅ **Unified Learning Rate**: Moved to BaseParams since all algorithms need it
- ✅ **Removed Dead Code**: Eliminated unused logging parameters ✅ **COMPLETED**
- ✅ **Simplified Algorithm Parameters**: Only algorithm-specific settings remain
- ✅ **Consistent Scheduler Support**: All algorithms benefit from learning rate scheduling

### Code Quality
- ✅ **Google C++ Style Compliant**: No default parameters
- ✅ **Type Safety**: Compile-time algorithm parameter validation
- ✅ **Modern C++**: Uses variants, smart pointers, constexpr

### Extensibility  
- ✅ **Easy Algorithm Addition**: Add Adam, RMSprop, L-BFGS with minimal changes
- ✅ **Learning Rate Scheduling**: Full PyTorch-style scheduler support
- ✅ **Physics-Aware**: Factory methods encode PEPS optimization best practices

### User Experience
- ✅ **Multiple Configuration Methods**: Factory, builder, or direct construction
- ✅ **Unified Learning Rate**: Single learning_rate parameter in BaseParams for all algorithms
- ✅ **Cleaner Algorithm Parameters**: Only algorithm-specific settings (momentum, betas, etc.)
- ✅ **Standard Terminology**: Uses "learning_rate" instead of "step_lengths"

### Maintainability
- ✅ **Clear Responsibilities**: Each class has a single, well-defined purpose
- ✅ **Easy Testing**: Mockable interfaces and dependency injection
- ✅ **Future-Proof**: Ready for modern ML algorithms and techniques

This refactor positions the PEPS optimizer API for long-term success while maintaining the physics-aware design that makes it effective for quantum many-body calculations.

## Current Default Parameter Values

**CRITICAL**: When removing default parameters from constructors (Google C++ style), use these values in test cases to maintain backward compatibility and ensure existing test cases continue working.

### OptimizerParams::BaseParams Current Defaults
```cpp
BaseParams(size_t max_iter, double energy_tol = 1e-15, double grad_tol = 1e-30,
           size_t patience = 20, const std::vector<double> &steps = {0.1})
```

**Test Migration Values:**
```cpp
// OLD (with defaults):
OptimizerParams::BaseParams params(1000);

// NEW (explicit values):
OptimizerParams::BaseParams params(1000, 1e-15, 1e-30, 20, {0.1});
```

### ConjugateGradientParams Current Defaults
```cpp
ConjugateGradientParams(void) : max_iter(0), tolerance(0.0), residue_restart_step(0), diag_shift(0.0) {}
```

**Test Migration Values:**
```cpp
// OLD (default constructor):
ConjugateGradientParams cg_params;

// NEW (explicit values for empty/default state):
ConjugateGradientParams cg_params(0, 0.0, 0, 0.0);  // For non-SR methods

// NEW (typical values for SR methods):
ConjugateGradientParams cg_params(100, 1e-5, 10, 0.01);  // Common SR values
```

### AdaGradParams Current Defaults
```cpp
AdaGradParams(double epsilon = 1e-8, double initial_accumulator_value = 0.0)
```

**Test Migration Values:**
```cpp
// OLD (with defaults):
AdaGradParams adagrad_params;

// NEW (explicit values):
AdaGradParams adagrad_params(1e-8, 0.0);
```

### OptimizerParams Factory Method Defaults
The current factory methods use these hardcoded defaults:
```cpp
// OptimizerParams::CreateStochasticGradient():
BaseParams(max_iterations, 1e-15, 1e-15, 20, step_lengths)

// OptimizerParams::CreateAdaGrad():
BaseParams(max_iterations, 1e-15, 1e-15, 20, {step_length})

// OptimizerParams::CreateStochasticReconfiguration():
BaseParams(max_iterations, 1e-15, 1e-15, 20, step_lengths)
```

### VMCPEPSOptimizerParams Structure
**Note**: `VMCPEPSOptimizerParams` itself has no default parameters - it simply aggregates component structures. The refactor will maintain the same interface:

```cpp
// CURRENT AND FUTURE (interface unchanged):
VMCPEPSOptimizerParams params(optimizer_params, mc_params, peps_params);
```

The changes are internal to the `OptimizerParams` component, not the `VMCPEPSOptimizerParams` container.

### Common Parameter Values in Current Codebase

**Common Step Lengths:**
- `{0.01}` - Single step, most common for SR
- `{0.01, 0.01, 0.01}` - Multiple iterations with same step
- `{0.1, 0.01, 0.001}` - Decreasing step schedule

**Common ConjugateGradientParams for SR:**
- `ConjugateGradientParams(100, 1e-5, 10, 0.01)` - Typical for production
- `ConjugateGradientParams(50, 1e-6, 5, 0.005)` - Conservative settings
- `ConjugateGradientParams(200, 1e-4, 20, 0.02)` - Aggressive settings

**Common Update Schemes:**
- `StochasticReconfiguration` - Most common (requires CG params)
- `StochasticGradient` - Second most common
- `AdaGrad` - Used in specific tests

This documentation ensures that when default parameters are removed from the component structures, all existing test cases can be updated with explicit values to maintain exactly the same behavior.
