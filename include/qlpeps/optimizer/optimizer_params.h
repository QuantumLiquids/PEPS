// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Refactored: 2025-01-29 - Modern C++ optimizer parameters with learning rate scheduling
*
* Description: QuantumLiquids/PEPS project. Modern optimizer parameters structure.
*/

#ifndef QLPEPS_OPTIMIZER_OPTIMIZER_PARAMS_H
#define QLPEPS_OPTIMIZER_OPTIMIZER_PARAMS_H

#include <vector>
#include <variant>
#include <string>
#include <memory>
#include <optional>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cmath>

namespace qlpeps {

// =============================================================================
// LEARNING RATE SCHEDULER INTERFACE
// =============================================================================

/**
 * @class LearningRateScheduler
 * @brief Abstract base class for learning rate scheduling algorithms.
 * 
 * Learning rate scheduling is crucial for PEPS optimization convergence.
 * This interface provides PyTorch-style scheduling capabilities with 
 * physics-aware features like energy plateau detection.
 */
class LearningRateScheduler {
public:
  virtual ~LearningRateScheduler() = default;
  
  /**
   * @brief Get the learning rate for the current iteration
   * @param iteration Current optimization iteration (0-based)
   * @param current_energy Current energy value (for energy-aware schedulers)
   * @return Learning rate for this iteration
   */
  virtual double GetLearningRate(size_t iteration, double current_energy = 0.0) const = 0;
  
  /**
   * @brief Called after each optimization step to update scheduler state
   */
  virtual void Step() {}
  
  /**
   * @brief Create a deep copy of the scheduler
   * @return Unique pointer to a cloned scheduler
   */
  virtual std::unique_ptr<LearningRateScheduler> Clone() const = 0;
};

/**
 * @class ConstantLR
 * @brief Constant learning rate scheduler
 */
class ConstantLR : public LearningRateScheduler {
private:
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

/**
 * @class ExponentialDecayLR
 * @brief Exponential decay learning rate scheduler
 * 
 * Learning rate decays exponentially: lr = initial_lr * decay_rate^(iteration/decay_steps)
 */
class ExponentialDecayLR : public LearningRateScheduler {
private:
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

/**
 * @class StepLR
 * @brief Step-wise learning rate scheduler
 * 
 * Learning rate is multiplied by gamma every step_size iterations
 */
class StepLR : public LearningRateScheduler {
private:
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

/**
 * @class PlateauLR
 * @brief Energy plateau-aware learning rate scheduler
 * 
 * Reduces learning rate when energy plateaus, essential for PEPS optimization
 */
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

// =============================================================================
// CORE PARAMETER STRUCTURES
// =============================================================================

/**
 * @struct ConjugateGradientParams
 * @brief Parameters for conjugate gradient solver used in stochastic reconfiguration.
 * 
 * No default constructor following Google C++ style - forces explicit parameter specification.
 */
struct ConjugateGradientParams {
  size_t max_iter;
  double tolerance;
  int residue_restart_step;
  double diag_shift;

  // No default constructor - forces explicit specification
  ConjugateGradientParams(size_t max_iter, double tolerance, int residue_restart_step, double diag_shift)
      : max_iter(max_iter), tolerance(tolerance), residue_restart_step(residue_restart_step), diag_shift(diag_shift) {}
};

/**
 * @struct BaseParams
 * @brief Core optimization parameters shared by all algorithms.
 * 
 * Contains unified learning_rate parameter and optional learning rate scheduler.
 * No default constructor following Google C++ style.
 */
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
             std::unique_ptr<LearningRateScheduler> scheduler = nullptr)
    : max_iterations(max_iter), energy_tolerance(energy_tol), gradient_tolerance(grad_tol),
      plateau_patience(patience), learning_rate(learning_rate), lr_scheduler(std::move(scheduler)) {}
      
  // Copy constructor
  BaseParams(const BaseParams& other)
    : max_iterations(other.max_iterations), energy_tolerance(other.energy_tolerance), 
      gradient_tolerance(other.gradient_tolerance), plateau_patience(other.plateau_patience),
      learning_rate(other.learning_rate),
      lr_scheduler(other.lr_scheduler ? other.lr_scheduler->Clone() : nullptr) {}
      
  // Copy assignment operator
  BaseParams& operator=(const BaseParams& other) {
    if (this != &other) {
      max_iterations = other.max_iterations;
      energy_tolerance = other.energy_tolerance;
      gradient_tolerance = other.gradient_tolerance;
      plateau_patience = other.plateau_patience;
      learning_rate = other.learning_rate;
      lr_scheduler = other.lr_scheduler ? other.lr_scheduler->Clone() : nullptr;
    }
    return *this;
  }
  
  // Move constructor and assignment are implicitly generated
  BaseParams(BaseParams&&) = default;
  BaseParams& operator=(BaseParams&&) = default;
};

// =============================================================================
// ALGORITHM-SPECIFIC PARAMETER STRUCTURES
// =============================================================================

/**
 * @struct SGDParams
 * @brief Parameters for Stochastic Gradient Descent algorithm
 */
struct SGDParams {
  double momentum;          // For future SGD with momentum
  bool nesterov;           // For Nesterov acceleration
  
  SGDParams(double momentum = 0.0, bool nesterov = false) 
    : momentum(momentum), nesterov(nesterov) {}
};

/**
 * @struct AdamParams
 * @brief Parameters for Adam optimization algorithm
 */
struct AdamParams {
  double beta1;         // First moment decay rate  
  double beta2;         // Second moment decay rate
  double epsilon;       // Numerical stability
  double weight_decay;  // L2 regularization
  
  AdamParams(double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, double wd = 0.0)
    : beta1(b1), beta2(b2), epsilon(eps), weight_decay(wd) {}
};

/**
 * @struct StochasticReconfigurationParams
 * @brief Parameters for Stochastic Reconfiguration (Natural Gradient) algorithm
 */
struct StochasticReconfigurationParams {
  ConjugateGradientParams cg_params;                     // CG solver configuration  
  bool normalize_update;                                 // For normalized SR
  double adaptive_diagonal_shift;                        // Dynamic regularization
  
  StochasticReconfigurationParams(const ConjugateGradientParams& cg_params,
                                 bool normalize = false, double adaptive_shift = 0.0)
    : cg_params(cg_params), normalize_update(normalize), adaptive_diagonal_shift(adaptive_shift) {}
};

/**
 * @struct LBFGSParams
 * @brief Parameters for L-BFGS optimization algorithm
 */
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

/**
 * @struct AdaGradParams
 * @brief Parameters for AdaGrad optimization algorithm
 * 
 * Updated to remove default parameters following Google C++ style
 */
struct AdaGradParams {
  double epsilon;
  double initial_accumulator_value;

  AdaGradParams(double epsilon, double initial_accumulator_value)
      : epsilon(epsilon), initial_accumulator_value(initial_accumulator_value) {}
};

// =============================================================================
// TYPE-SAFE ALGORITHM PARAMETER VARIANT
// =============================================================================

/**
 * @brief Type-safe algorithm parameter variant
 * 
 * Uses std::variant for compile-time type safety and efficient algorithm dispatch
 */
using AlgorithmParams = std::variant<
  SGDParams,
  AdamParams,
  StochasticReconfigurationParams,
  LBFGSParams,
  AdaGradParams
>;

// =============================================================================
// MAIN OPTIMIZER PARAMETERS STRUCTURE
// =============================================================================

/**
 * @struct OptimizerParams
 * @brief Main optimizer parameters structure with modern C++ design
 * 
 * Features:
 * - No default constructor (Google C++ style)
 * - Type-safe algorithm parameter handling
 * - Learning rate scheduling support
 * - Template-based parameter access
 */
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

// =============================================================================
// USER-FRIENDLY FACTORY METHODS
// =============================================================================

/**
 * @class OptimizerFactory
 * @brief Factory class for creating common optimizer parameter configurations
 * 
 * Provides physics-aware factory methods that encode PEPS optimization best practices
 */
class OptimizerFactory {
public:
  /**
   * @brief Create Stochastic Reconfiguration optimizer (most common for PEPS)
   */
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
  
  /**
   * @brief Create SGD with exponential decay (common for gradient methods)
   */
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
  
  /**
   * @brief Create Adam optimizer (future algorithm)
   */
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
  
  /**
   * @brief Create AdaGrad optimizer  
   */
  static OptimizerParams CreateAdaGrad(
      size_t max_iterations,
      double energy_tolerance,
      double learning_rate,
      double epsilon = 1e-8,
      double initial_accumulator = 0.0) {
    
    BaseParams base_params(max_iterations, energy_tolerance, 1e-15, 20, learning_rate);
    AdaGradParams adagrad_params(epsilon, initial_accumulator);
    return OptimizerParams(base_params, adagrad_params);
  }
  
  /**
   * @brief Create L-BFGS for second-order optimization (future algorithm)
   */
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

// =============================================================================
// BUILDER PATTERN FOR COMPLEX CONFIGURATIONS
// =============================================================================

/**
 * @class OptimizerParamsBuilder
 * @brief Builder pattern for complex optimizer parameter configurations
 * 
 * Allows step-by-step construction of optimizer parameters with validation
 */
class OptimizerParamsBuilder {
private:
  std::optional<BaseParams> base_params_;
  std::optional<AlgorithmParams> algorithm_params_;
  
public:
  /**
   * @brief Set maximum iterations
   */
  OptimizerParamsBuilder& SetMaxIterations(size_t max_iter) {
    if (!base_params_) {
      base_params_ = BaseParams(max_iter, 1e-10, 1e-15, 20, 0.01);
    } else {
      base_params_->max_iterations = max_iter;
    }
    return *this;
  }
  
  /**
   * @brief Set energy tolerance
   */
  OptimizerParamsBuilder& SetEnergyTolerance(double tol) {
    if (!base_params_) {
      base_params_ = BaseParams(1000, tol, 1e-15, 20, 0.01);
    } else {
      base_params_->energy_tolerance = tol;
    }
    return *this;
  }
  
  /**
   * @brief Set learning rate with optional scheduler
   */
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
  
  /**
   * @brief Set plateau patience
   */
  OptimizerParamsBuilder& SetPlateauPatience(size_t patience) {
    if (!base_params_) {
      base_params_ = BaseParams(1000, 1e-10, 1e-15, patience, 0.01);
    } else {
      base_params_->plateau_patience = patience;
    }
    return *this;
  }
  
  /**
   * @brief Configure for SGD algorithm
   */
  OptimizerParamsBuilder& WithSGD(double momentum = 0.0, bool nesterov = false) {
    SGDParams sgd_params(momentum, nesterov);
    algorithm_params_ = sgd_params;
    return *this;
  }
  
  /**
   * @brief Configure for Stochastic Reconfiguration algorithm
   */
  OptimizerParamsBuilder& WithStochasticReconfiguration(const ConjugateGradientParams& cg_params,
                                                       bool normalize = false, 
                                                       double adaptive_shift = 0.0) {
    StochasticReconfigurationParams sr_params(cg_params, normalize, adaptive_shift);
    algorithm_params_ = sr_params;
    return *this;
  }
  
  /**
   * @brief Configure for Adam algorithm
   */
  OptimizerParamsBuilder& WithAdam(double beta1 = 0.9, double beta2 = 0.999, 
                                  double epsilon = 1e-8, double weight_decay = 0.0) {
    AdamParams adam_params(beta1, beta2, epsilon, weight_decay);
    algorithm_params_ = adam_params;
    return *this;
  }
  
  /**
   * @brief Build the final OptimizerParams object
   */
  OptimizerParams Build() const {
    if (!base_params_ || !algorithm_params_) {
      throw std::invalid_argument("Both base parameters and algorithm parameters must be specified");
    }
    return OptimizerParams(*base_params_, *algorithm_params_);
  }
};

} // namespace qlpeps

#endif // QLPEPS_OPTIMIZER_OPTIMIZER_PARAMS_H 