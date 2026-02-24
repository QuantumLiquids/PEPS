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

// =============================================================================
// LEARNING RATE SCHEDULERS (moved to dedicated header)
// =============================================================================
#include "qlpeps/optimizer/lr_schedulers.h"

namespace qlpeps {

/**
 * @struct ConjugateGradientParams
 * @brief Parameters for conjugate gradient solver.
 *
 * Used by both stochastic reconfiguration (SR) and loop update full-environment
 * truncation.
 *
 * Current stopping criterion:
 *   ||r||^2 <= max(tolerance * ||b||^2, absolute_tolerance^2).
 *
 * Notes:
 * - \c tolerance is used in squared-residual form, so the equivalent norm-space
 *   relative tolerance is sqrt(tolerance).
 * - \c absolute_tolerance is in residual 2-norm (not squared).
 * - Default \c absolute_tolerance is 0.0 to preserve legacy relative-only behavior.
 * - A future API revision is planned to align with standard norm-space
 *   conventions (e.g. ||r|| <= atol + rtol * ||b||).
 */
struct ConjugateGradientParams {
  size_t max_iter;              ///< Maximum CG iterations
  double tolerance;             ///< Relative tolerance used in squared criterion (tolerance * ||b||^2)
  double absolute_tolerance;    ///< Absolute tolerance in residual norm ||r|| (default 0.0)
  int residue_restart_step;     ///< Periodically recompute residual (0 to disable)
  double diag_shift;            ///< Diagonal shift for regularization (SR only)

  // Default constructor for backward compatibility
  ConjugateGradientParams()
      : max_iter(100), tolerance(1e-5), absolute_tolerance(0.0),
        residue_restart_step(20), diag_shift(0.001) {}

  ConjugateGradientParams(
      size_t max_iter, double tolerance, int residue_restart_step,
      double diag_shift, double absolute_tolerance = 0.0)
      : max_iter(max_iter), tolerance(tolerance),
        absolute_tolerance(absolute_tolerance),
        residue_restart_step(residue_restart_step), diag_shift(diag_shift) {}
};

// BaseParams is now nested inside OptimizerParams - see below

// =============================================================================
// ALGORITHM-SPECIFIC PARAMETER STRUCTURES
// =============================================================================

/**
 * @struct SGDParams
 * @brief Parameters for Stochastic Gradient Descent algorithm
 */
struct SGDParams {
  double momentum;          // Momentum strength μ
  bool nesterov;           // Nesterov acceleration flag
  double weight_decay;     // L2 regularization λ (decoupled off scheduler)
  
  SGDParams(double momentum = 0.0, bool nesterov = false, double weight_decay = 0.0) 
    : momentum(momentum), nesterov(nesterov), weight_decay(weight_decay) {}
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
 * @enum LBFGSStepMode
 * @brief Step-length policy for L-BFGS updates.
 */
enum class LBFGSStepMode {
  kFixed = 0,       ///< Fixed step length from base learning rate.
  kStrongWolfe = 1  ///< Strong Wolfe line search (deterministic use recommended).
};

/**
 * @struct LBFGSParams
 * @brief Parameters for L-BFGS optimization algorithm
 */
struct LBFGSParams {
  size_t history_size;         // Limited memory size (typically 5-20)
  double tolerance_grad;       // Absolute floor for |phi'(alpha)| in strong-Wolfe curvature check
  double tolerance_change;     // Line-search interval/step tolerance
  size_t max_eval;             // Max objective evaluations per step/line-search
  LBFGSStepMode step_mode;     // Fixed-step or strong-Wolfe
  double wolfe_c1;             // Armijo constant (0,1)
  double wolfe_c2;             // Curvature constant (c1,1)
  double min_step;             // Minimal admissible step length
  double max_step;             // Maximal admissible step length
  double min_curvature;        // Minimal Re(<s,y>) to accept a history pair
  bool use_damping;            // Apply simple damping before dropping low-curvature pairs
  double max_direction_norm;   // Cap search direction norm for robustness
  bool allow_fallback_to_fixed_step;  // Explicit opt-in fallback if strong-Wolfe fails
  double fallback_fixed_step_scale;   // alpha = scale * learning_rate for fallback

  LBFGSParams(size_t hist = 10, double tol_grad = 1e-5,
              double tol_change = 1e-9, size_t max_eval = 20,
              LBFGSStepMode step_mode = LBFGSStepMode::kFixed,
              double wolfe_c1 = 1e-4, double wolfe_c2 = 0.9,
              double min_step = 1e-8, double max_step = 1.0,
              double min_curvature = 1e-12, bool use_damping = true,
              double max_direction_norm = 1e3,
              bool allow_fallback_to_fixed_step = false,
              double fallback_fixed_step_scale = 0.2)
    : history_size(hist), tolerance_grad(tol_grad),
      tolerance_change(tol_change), max_eval(max_eval),
      step_mode(step_mode), wolfe_c1(wolfe_c1), wolfe_c2(wolfe_c2),
      min_step(min_step), max_step(max_step),
      min_curvature(min_curvature), use_damping(use_damping),
      max_direction_norm(max_direction_norm),
      allow_fallback_to_fixed_step(allow_fallback_to_fixed_step),
      fallback_fixed_step_scale(fallback_fixed_step_scale) {}
};

/**
 * @struct PeriodicStepSelectorParams
 * @brief Optional MC-oriented automatic step-size selector for IterativeOptimize.
 *
 * Semantics:
 * - enabled=false: disabled, optimizer uses existing learning-rate path unchanged.
 * - every_n_steps: trigger interval for candidate step-size selection.
 * - phase_switch_ratio: early/late policy boundary as a fraction of max_iterations.
 * - enable_in_deterministic: allow running selector when evaluator error bars are unavailable.
 */
struct PeriodicStepSelectorParams {
  bool enabled = false;
  size_t every_n_steps = 10;
  double phase_switch_ratio = 0.3;
  bool enable_in_deterministic = false;
};

/**
 * @struct InitialStepSelectorParams
 * @brief Optional initial-step selector used at the first optimization step.
 *
 * Semantics:
 * - enabled=false: disabled, optimizer uses the configured learning rate directly.
 * - max_line_search_steps: evaluate cumulative candidates {eta, 2*eta, ..., k*eta}.
 * - enable_in_deterministic: allow running selector when evaluator error bars are unavailable.
 */
struct InitialStepSelectorParams {
  bool enabled = false;
  size_t max_line_search_steps = 3;
  bool enable_in_deterministic = false;
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
// CHECKPOINT PARAMETERS
// =============================================================================

/**
 * @struct CheckpointParams
 * @brief Parameters for periodic TPS checkpointing during optimization.
 *
 * When enabled, the optimizer saves the current TPS state every N steps
 * to base_path/step_{k}/. Resume is manual: load the TPS and pass it
 * to VmcOptimize() as the initial state.
 */
struct CheckpointParams {
  size_t every_n_steps = 0;   ///< 0 = disabled
  std::string base_path;      ///< base_path/step_{k}/

  bool IsEnabled() const { return every_n_steps > 0 && !base_path.empty(); }
};

// =============================================================================
// SPIKE RECOVERY PARAMETERS
// =============================================================================

/**
 * @struct SpikeRecoveryParams
 * @brief Parameters controlling automatic spike detection and recovery in VMC optimization.
 *
 * S1-S3 detect anomalous energy errors, gradient norms, and natural gradient anomalies
 * and trigger MC resampling. S4 (opt-in) detects upward energy spikes for state rollback.
 *
 * Detection thresholds are expressed as multiplicative factors relative to EMA statistics.
 * For example, S1 triggers when error > factor_err * EMA(error).
 */
struct SpikeRecoveryParams {
  bool enable_auto_recover = true;       ///< S1-S3 resample master switch
  size_t redo_mc_max_retries = 2;        ///< max resample attempts per step
  double factor_err = 100.0;             ///< S1: error > factor * EMA(error)
  double factor_grad = 1e10;             ///< S2: grad_norm > factor * EMA(grad_norm)
  double factor_ngrad = 10.0;            ///< S3: ngrad_norm > factor * EMA(ngrad_norm)
  size_t sr_min_iters_suspicious = 1;    ///< CG iters <= this is suspicious (S3)
  /// S4 master switch (default OFF).
  /// Note: rollback restores the wavefunction state only; optimizer-internal
  /// accumulators (momentum/Adam/AdaGrad) are not restored.
  bool enable_rollback = false;
  size_t ema_window = 50;               ///< EMA window size for all trackers
  double sigma_k = 10.0;                ///< S4: E - EMA(E) > sigma_k * EMA_std(E)
  std::string log_trigger_csv_path;     ///< optional CSV log path (empty = no CSV)
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
 * - Nested BaseParams to avoid namespace pollution
 */
struct OptimizerParams {
  /**
   * @struct BaseParams
   * @brief Core optimization parameters shared by all algorithms.
   * 
   * Contains unified learning_rate parameter and optional learning rate scheduler.
   * Also provides optional gradient clipping for first-order optimizers (SGD/AdaGrad/Adam).
   *
   * Gradient clipping semantics:
   * - Per-element magnitude clipping (complex-safe): if |g| > c then g ← polar(c, arg(g)); otherwise unchanged.
   * - Global L2 norm clipping: let r = sqrt(Σ |g_j|^2). If r > C then g_j ← (C/r) g_j for all elements; otherwise unchanged.
   *
   * Scope and defaults:
   * - Clipping applies only to first-order optimizers (SGD/AdaGrad/Adam). SR/L-BFGS do not use clipping.
   * - Clipping is disabled by default (std::nullopt values).
   * No default constructor following Google C++ style.
   */
  struct BaseParams {
    size_t max_iterations;
    double energy_tolerance;
    double gradient_tolerance;
    size_t plateau_patience;
    double learning_rate;                                    // Universal parameter for all algorithms
    std::unique_ptr<LearningRateScheduler> lr_scheduler;     // Optional learning rate scheduling
    /// Optional gradient clipping (first-order optimizers only)
    /// Per-element magnitude clip (complex-safe, preserve phase). unset -> no clipping
    std::optional<double> clip_value;
    /// Optional global L2 norm clipping threshold. unset -> no clipping
    std::optional<double> clip_norm;
    /// Optional initial-step selector config (disabled by default)
    InitialStepSelectorParams initial_step_selector;
    /// Optional periodic step-size selector config (disabled by default)
    PeriodicStepSelectorParams periodic_step_selector;
    
    // No default constructor - force explicit specification
    BaseParams(size_t max_iter, double energy_tol, double grad_tol,
               size_t patience, double learning_rate,
               std::unique_ptr<LearningRateScheduler> scheduler = nullptr)
      : max_iterations(max_iter), energy_tolerance(energy_tol), gradient_tolerance(grad_tol),
        plateau_patience(patience), learning_rate(learning_rate), lr_scheduler(std::move(scheduler)),
        clip_value(std::nullopt), clip_norm(std::nullopt),
        initial_step_selector(), periodic_step_selector() {}
        
    // Copy constructor
    BaseParams(const BaseParams& other)
      : max_iterations(other.max_iterations), energy_tolerance(other.energy_tolerance), 
        gradient_tolerance(other.gradient_tolerance), plateau_patience(other.plateau_patience),
        learning_rate(other.learning_rate),
        lr_scheduler(other.lr_scheduler ? other.lr_scheduler->Clone() : nullptr),
        clip_value(other.clip_value), clip_norm(other.clip_norm),
        initial_step_selector(other.initial_step_selector),
        periodic_step_selector(other.periodic_step_selector) {}
        
    // Copy assignment operator
    BaseParams& operator=(const BaseParams& other) {
      if (this != &other) {
        max_iterations = other.max_iterations;
        energy_tolerance = other.energy_tolerance;
        gradient_tolerance = other.gradient_tolerance;
        plateau_patience = other.plateau_patience;
        learning_rate = other.learning_rate;
        lr_scheduler = other.lr_scheduler ? other.lr_scheduler->Clone() : nullptr;
        clip_value = other.clip_value;
        clip_norm = other.clip_norm;
        initial_step_selector = other.initial_step_selector;
        periodic_step_selector = other.periodic_step_selector;
      }
      return *this;
    }
    
    // Move constructor and assignment are implicitly generated
    BaseParams(BaseParams&&) = default;
    BaseParams& operator=(BaseParams&&) = default;
  };

  BaseParams base_params;
  AlgorithmParams algorithm_params;
  CheckpointParams checkpoint_params;           ///< default: disabled
  SpikeRecoveryParams spike_recovery_params;    ///< default: S1-S3 on, S4 off

  // WARNING: TESTING-ONLY DEFAULT CONSTRUCTOR - DO NOT USE IN PRODUCTION.
  // Uses obviously invalid values to make misuse immediately obvious
  OptimizerParams()
    : base_params(1, 999.0, 999.0, 1, 999.0),  // Obviously wrong values
      algorithm_params(SGDParams()) {}

  // Constructor requires explicit parameters (no defaults)
  OptimizerParams(const BaseParams& base_params, const AlgorithmParams& algo_params)
    : base_params(base_params), algorithm_params(algo_params) {}

  // Constructor with checkpoint and spike recovery params
  OptimizerParams(const BaseParams& base_params, const AlgorithmParams& algo_params,
                  const CheckpointParams& ckpt, const SpikeRecoveryParams& spike)
    : base_params(base_params), algorithm_params(algo_params),
      checkpoint_params(ckpt), spike_recovery_params(spike) {}

  // Legacy compatibility fields removed - use new variant-based API instead

public:
    
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

  /**
   * @brief Check whether the selected algorithm is first-order (SGD/AdaGrad/Adam)
   * 
   * Clipping scope relies on this check: if true, gradient pre-processing
   * (clip_value/clip_norm) may be applied by the optimizer implementation.
   */
  bool IsFirstOrder() const {
    return IsAlgorithm<SGDParams>() || IsAlgorithm<AdaGradParams>() || IsAlgorithm<AdamParams>();
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
   * @brief Create Stochastic Reconfiguration optimizer - Simple version (advanced stopping disabled)
   */
  static OptimizerParams CreateStochasticReconfiguration(
      size_t max_iterations,
      const ConjugateGradientParams& cg_params,
      double learning_rate = 0.1) {
    
    // Set tolerances to 0 to disable advanced stopping
    OptimizerParams::BaseParams base_params(max_iterations, 0.0, 0.0, max_iterations, learning_rate);
    StochasticReconfigurationParams sr_params(cg_params);
    return OptimizerParams(base_params, sr_params);
  }

  /**
   * @brief Create Stochastic Reconfiguration optimizer - Full version with all parameters
   */
  static OptimizerParams CreateStochasticReconfigurationAdvanced(
      size_t max_iterations,
      double energy_tolerance,
      double gradient_tolerance,
      size_t plateau_patience,
      const ConjugateGradientParams& cg_params,
      double learning_rate = 0.1,
      std::unique_ptr<LearningRateScheduler> scheduler = nullptr) {
    
    OptimizerParams::BaseParams base_params(max_iterations, energy_tolerance, gradient_tolerance, plateau_patience, 
                          learning_rate, std::move(scheduler));
    StochasticReconfigurationParams sr_params(cg_params);
    return OptimizerParams(base_params, sr_params);
  }
  
  /**
   * @brief Create SGD with exponential decay - Simple version (advanced stopping disabled)
   */
  static OptimizerParams CreateSGDWithDecay(
      size_t max_iterations,
      double initial_learning_rate = 0.01,
      double decay_rate = 0.95,
      size_t decay_steps = 100) {
    
    auto scheduler = std::make_unique<ExponentialDecayLR>(initial_learning_rate, decay_rate, decay_steps);
    // Set tolerances to 0 to disable advanced stopping
    OptimizerParams::BaseParams base_params(max_iterations, 0.0, 0.0, max_iterations, 
                          initial_learning_rate, std::move(scheduler));
    SGDParams sgd_params{};
    return OptimizerParams(base_params, sgd_params);
  }

  /**
   * @brief Create SGD with exponential decay - Full version with all parameters
   */
  static OptimizerParams CreateSGDWithDecayAdvanced(
      size_t max_iterations,
      double energy_tolerance,
      double gradient_tolerance,
      size_t plateau_patience,
      double initial_learning_rate = 0.01,
      double decay_rate = 0.95,
      size_t decay_steps = 100) {
    
    auto scheduler = std::make_unique<ExponentialDecayLR>(initial_learning_rate, decay_rate, decay_steps);
    OptimizerParams::BaseParams base_params(max_iterations, energy_tolerance, gradient_tolerance, plateau_patience, 
                          initial_learning_rate, std::move(scheduler));
    SGDParams sgd_params{};
    return OptimizerParams(base_params, sgd_params);
  }


  
  /**
   * @brief Create Adam optimizer - Simple version (advanced stopping disabled)
   */
  static OptimizerParams CreateAdam(
      size_t max_iterations,
      double learning_rate = 1e-3,
      double beta1 = 0.9,
      double beta2 = 0.999) {
    
    // Set tolerances to 0 to disable advanced stopping
    OptimizerParams::BaseParams base_params(max_iterations, 0.0, 0.0, max_iterations, learning_rate);
    AdamParams adam_params(beta1, beta2);
    return OptimizerParams(base_params, adam_params);
  }

  /**
   * @brief Create Adam optimizer - Full version with all parameters
   */
  static OptimizerParams CreateAdamAdvanced(
      size_t max_iterations,
      double energy_tolerance,
      double gradient_tolerance,
      size_t plateau_patience,
      double learning_rate = 1e-3,
      double beta1 = 0.9,
      double beta2 = 0.999,
      std::unique_ptr<LearningRateScheduler> scheduler = nullptr) {
    
    OptimizerParams::BaseParams base_params(max_iterations, energy_tolerance, gradient_tolerance, plateau_patience, 
                          learning_rate, std::move(scheduler));
    AdamParams adam_params(beta1, beta2);
    return OptimizerParams(base_params, adam_params);
  }
  
  /**
   * @brief Create AdaGrad optimizer - Simple version (advanced stopping disabled)
   */
  static OptimizerParams CreateAdaGrad(
      size_t max_iterations,
      double learning_rate = 0.01,
      double epsilon = 1e-8,
      double initial_accumulator = 0.0) {
    
    // Set tolerances to 0 to disable advanced stopping
    OptimizerParams::BaseParams base_params(max_iterations, 0.0, 0.0, max_iterations, learning_rate);
    AdaGradParams adagrad_params(epsilon, initial_accumulator);
    return OptimizerParams(base_params, adagrad_params);
  }

  /**
   * @brief Create AdaGrad optimizer - Full version with all parameters
   */
  static OptimizerParams CreateAdaGradAdvanced(
      size_t max_iterations,
      double energy_tolerance,
      double gradient_tolerance,
      size_t plateau_patience,
      double learning_rate = 0.01,
      double epsilon = 1e-8,
      double initial_accumulator = 0.0) {
    
    OptimizerParams::BaseParams base_params(max_iterations, energy_tolerance, gradient_tolerance, plateau_patience, learning_rate);
    AdaGradParams adagrad_params(epsilon, initial_accumulator);
    return OptimizerParams(base_params, adagrad_params);
  }
  
  /**
   * @brief Create L-BFGS optimizer - Simple version (advanced stopping disabled)
   */
  static OptimizerParams CreateLBFGS(
      size_t max_iterations,
      double learning_rate = 1.0,
      size_t history_size = 10) {
    
    // Set tolerances to 0 to disable advanced stopping
    OptimizerParams::BaseParams base_params(max_iterations, 0.0, 0.0, max_iterations, learning_rate);
    LBFGSParams lbfgs_params(history_size, 1e-5, 1e-9, 20, LBFGSStepMode::kFixed);
    return OptimizerParams(base_params, lbfgs_params);
  }

  /**
   * @brief Create L-BFGS optimizer - Full version with all parameters
   */
  static OptimizerParams CreateLBFGSAdvanced(
      size_t max_iterations,
      double energy_tolerance,
      double gradient_tolerance,
      size_t plateau_patience,
      double learning_rate = 1.0,
      size_t history_size = 10,
      std::unique_ptr<LearningRateScheduler> scheduler = nullptr) {
    
    OptimizerParams::BaseParams base_params(max_iterations, energy_tolerance, gradient_tolerance, plateau_patience, 
                          learning_rate, std::move(scheduler));
    LBFGSParams lbfgs_params(history_size, 1e-5, 1e-9, 20, LBFGSStepMode::kFixed);
    return OptimizerParams(base_params, lbfgs_params);
  }

  /**
   * @brief Create L-BFGS optimizer - Full version with explicit LBFGS options
   */
  static OptimizerParams CreateLBFGSAdvanced(
      size_t max_iterations,
      double energy_tolerance,
      double gradient_tolerance,
      size_t plateau_patience,
      double learning_rate,
      const LBFGSParams& lbfgs_params,
      std::unique_ptr<LearningRateScheduler> scheduler = nullptr) {

    OptimizerParams::BaseParams base_params(max_iterations, energy_tolerance, gradient_tolerance, plateau_patience,
                                            learning_rate, std::move(scheduler));
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
  std::optional<OptimizerParams::BaseParams> base_params_;
  std::optional<AlgorithmParams> algorithm_params_;
  std::optional<CheckpointParams> checkpoint_params_;
  std::optional<SpikeRecoveryParams> spike_recovery_params_;

public:
  /**
   * @brief Set maximum iterations
   */
  OptimizerParamsBuilder& SetMaxIterations(size_t max_iter) {
    if (!base_params_) {
      base_params_ = OptimizerParams::BaseParams(max_iter, 1e-10, 1e-30, 20, 0.01);
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
      base_params_ = OptimizerParams::BaseParams(1000, tol, 1e-30, 20, 0.01);
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
      base_params_ = OptimizerParams::BaseParams(1000, 1e-10, 1e-30, 20, learning_rate, std::move(scheduler));
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
      base_params_ = OptimizerParams::BaseParams(1000, 1e-10, 1e-15, patience, 0.01);
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
   * @brief Configure for L-BFGS algorithm
   */
  OptimizerParamsBuilder& WithLBFGS(const LBFGSParams& lbfgs_params) {
    algorithm_params_ = lbfgs_params;
    return *this;
  }
  
  /**
   * @brief Configure for AdaGrad algorithm
   */
  OptimizerParamsBuilder& WithAdaGrad(double epsilon = 1e-8, double initial_accumulator = 0.0) {
    AdaGradParams adagrad_params(epsilon, initial_accumulator);
    algorithm_params_ = adagrad_params;
    return *this;
  }

  /**
   * @brief Enable per-element magnitude clipping (complex-safe, preserve phase)
   * 
   * Mathematical definition:
   * - For each element g: if |g| > c then g ← polar(c, arg(g)); otherwise unchanged.
   * - Real-valued tensors reduce to sign-preserving absolute value clipping.
   *
   * Constraints and scope:
   * - Requires BaseParams to be created first; throws std::invalid_argument otherwise.
   * - Applied only for first-order optimizers (SGD/AdaGrad/Adam) by the optimizer implementation.
   *
   * @param clip_value Per-element magnitude threshold (>0)
   * @return Builder reference for chaining
   */
  OptimizerParamsBuilder& SetClipValue(double clip_value) {
    if (!base_params_) {
      throw std::invalid_argument("BaseParams must be set before SetClipValue");
    }
    base_params_->clip_value = clip_value;
    return *this;
  }

  /**
   * @brief Enable global L2 norm clipping
   * 
   * Mathematical definition:
   * - Let r = sqrt(Σ |g_j|^2) across all parameters. If r > C, then scale g ← (C/r) g.
   * - Preserves direction while limiting update magnitude.
   *
   * Constraints and scope:
   * - Requires BaseParams to be created first; throws std::invalid_argument otherwise.
   * - Applied only for first-order optimizers (SGD/AdaGrad/Adam) by the optimizer implementation.
   *
   * @param clip_norm Global L2 norm threshold (>0)
   * @return Builder reference for chaining
   */
  OptimizerParamsBuilder& SetClipNorm(double clip_norm) {
    if (!base_params_) {
      throw std::invalid_argument("BaseParams must be set before SetClipNorm");
    }
    base_params_->clip_norm = clip_norm;
    return *this;
  }

  /**
   * @brief Configure optional periodic step-size selector for IterativeOptimize.
   *
   * Requires BaseParams to be set first.
   */
  OptimizerParamsBuilder& SetPeriodicStepSelector(bool enabled,
                                              size_t every_n_steps = 10,
                                              double phase_switch_ratio = 0.3,
                                              bool enable_in_deterministic = false) {
    if (!base_params_) {
      throw std::invalid_argument("BaseParams must be set before SetPeriodicStepSelector");
    }
    base_params_->periodic_step_selector = PeriodicStepSelectorParams{
        enabled, every_n_steps, phase_switch_ratio, enable_in_deterministic};
    return *this;
  }

  /**
   * @brief Configure optional initial-step selector for IterativeOptimize.
   *
   * Requires BaseParams to be set first.
   */
  OptimizerParamsBuilder& SetInitialStepSelector(bool enabled,
                                                 size_t max_line_search_steps = 3,
                                                 bool enable_in_deterministic = false) {
    if (!base_params_) {
      throw std::invalid_argument("BaseParams must be set before SetInitialStepSelector");
    }
    base_params_->initial_step_selector = InitialStepSelectorParams{
        enabled, max_line_search_steps, enable_in_deterministic};
    return *this;
  }

  /**
   * @brief Set checkpoint parameters for periodic TPS saving.
   * @param every_n_steps Save every N optimization steps (0 = disabled)
   * @param base_path Directory base path; checkpoints go to base_path/step_{k}/
   */
  OptimizerParamsBuilder& SetCheckpoint(size_t every_n_steps, const std::string& base_path) {
    checkpoint_params_ = CheckpointParams{every_n_steps, base_path};
    return *this;
  }

  /**
   * @brief Set spike recovery parameters for automatic anomaly handling.
   */
  OptimizerParamsBuilder& SetSpikeRecovery(const SpikeRecoveryParams& params) {
    spike_recovery_params_ = params;
    return *this;
  }

  /**
   * @brief Disable all spike recovery (S1-S3 resample and S4 rollback).
   */
  OptimizerParamsBuilder& DisableSpikeRecovery() {
    SpikeRecoveryParams disabled;
    disabled.enable_auto_recover = false;
    disabled.enable_rollback = false;
    spike_recovery_params_ = disabled;
    return *this;
  }

  /**
   * @brief Build the final OptimizerParams object
   */
  OptimizerParams Build() const {
    if (!base_params_ || !algorithm_params_) {
      throw std::invalid_argument("Both base parameters and algorithm parameters must be specified");
    }
    OptimizerParams result(*base_params_, *algorithm_params_);
    if (checkpoint_params_) {
      result.checkpoint_params = *checkpoint_params_;
    }
    if (spike_recovery_params_) {
      result.spike_recovery_params = *spike_recovery_params_;
    }
    return result;
  }
};

} // namespace qlpeps

#endif // QLPEPS_OPTIMIZER_OPTIMIZER_PARAMS_H 
