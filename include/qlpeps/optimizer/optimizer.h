// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Optimizer for VMC PEPS.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_H

#include <vector>
#include <memory>
#include <functional>
#include <deque>
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/optimizer/spike_detection.h"
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "qlpeps/optimizer/stochastic_reconfiguration_smatrix.h"

namespace qlpeps {

// Narrow import: prefer direct symbol using over local alias
using qlten::QLTensor;

/**
 * @brief Optimizer for VMC PEPS that handles different optimization strategies with MPI support
 * 
 * This class provides a unified interface for different optimization strategies.
 * It supports line search and iterative optimization, with or without energy error support.
 * 
 * The optimization strategies are:
 * 1. SGD (with momentum and Nesterov acceleration)
 * 2. AdaGrad (Adaptive Gradient)
 * 3. Stochastic reconfiguration (natural gradient)
 * 4. Bounded gradient element update
 * 5. Adam (with AdamW-style decoupled weight decay)
 * 6. L-BFGS (fixed-step and strong-Wolfe modes)
 * 
 * ⚠️ CRITICAL MPI RESPONSIBILITY SEPARATION:
 * 
 * 【CORE PRINCIPLE】: Clear MPI responsibility separation between state distribution and algorithm computation.
 * 
 * This design eliminates triple redundant broadcasts and follows "good taste" principles:
 * "Good taste is about eliminating special cases and making the normal case work correctly."
 * 
 * ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
 * │   Optimizer     │    │ Energy Evaluator │    │  VMC Executor   │
 * │                 │    │                  │    │                 │
 * │ • Algorithm logic│───▶│ • Receives state │    │ • Final sync    │
 * │ • SR uses MPI   │    │ • Broadcasts     │    │ • End guarantee │
 * │ • NO state bcast│    │ • Manages MC     │    │ • Cleanup       │
 * └─────────────────┘    └──────────────────┘    └─────────────────┘
 *                               │
 *                               ▼
 *                       All ranks synchronized
 *                       for Monte Carlo sampling
 * 
 * 【RESPONSIBILITY MATRIX】:
 * | Component         | Responsibility              | MPI Behavior                    |
 * |-------------------|----------------------------|---------------------------------|
 * | Optimizer         | Algorithm logic & updates  | ✅ SR/CG solver, ❌ NO state bcast|
 * | Energy Evaluator  | MC sampling coordination   | ✅ Sole state broadcast owner    |
 * | VMC Executor      | High-level orchestration   | ✅ Final state guarantee         |
 * 
 * 【MPI DATA DISTRIBUTION】:
 * 
 * INPUT (to energy_evaluator):
 * - current_state: Valid ONLY on master rank (optimizer output)
 * 
 * OUTPUT (from energy_evaluator):  
 * - energy: Valid on ALL ranks (broadcast by energy_evaluator)
 * - gradient: Valid ONLY on master rank (gathered by energy_evaluator)
 * - energy_error: Valid ONLY on master rank
 * 
 * INTERNAL STATE:
 * - algorithm_state (velocity, accumulated_gradients): Master rank ONLY
 * - optimization_statistics: Master rank ONLY
 * 
 * OUTPUT:
 * - optimized_state: Valid on all ranks (final broadcast by executor)
 * - trajectories: Master rank ONLY
 * 
 * 【ITERATION FLOW】:
 * 1. Optimizer updates state on master rank           ← NO state broadcast
 * 2. SR algorithm: MPI-distributed CG solving         ← Optimizer internal MPI
 * 3. Energy evaluator receives state (master only)    ← NO state broadcast yet
 * 4. Energy evaluator broadcasts for MC sampling      ← SINGLE state broadcast  
 * 5. Distributed Monte Carlo execution                ← All ranks
 * 6. Energy evaluator gathers gradients to master     ← Standard gather
 * 7. Energy evaluator broadcasts energy to all ranks  ← Standard broadcast
 * 8. Return to step 1                                 ← Loop continues
 * 
 * 【DESIGN BENEFITS】:
 * - 67% reduction in state broadcast overhead (3→1 broadcasts per iteration)
 * - Clear responsibility separation (single responsibility principle)
 * - Eliminates "special cases" and redundant MPI calls
 * - Better scalability for large MPI configurations
 * 
 * THREAD SAFETY:
 * - All gradient updates are performed only on master rank
 * - Updated states are broadcast to all ranks after each iteration
 * - This ensures deterministic behavior across different MPI configurations
 * 
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 */
template<typename TenElemT, typename QNT>
class Optimizer {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using WaveFunctionT = SITPST;
  
  struct OptimizationResult {
    WaveFunctionT optimized_state;
    double final_energy;
    double min_energy;
    std::vector<TenElemT> energy_trajectory;
    std::vector<double> energy_error_trajectory;
    std::vector<double> gradient_norms;
    size_t total_iterations;
    bool converged;
    SpikeStatistics spike_stats;   ///< Spike detection statistics for the run
  };

  struct OptimizationCallback {
    std::function<void(size_t iteration, double energy, double energy_error, double gradient_norm)> 
        on_iteration;
    std::function<void(const WaveFunctionT& state, double energy)> 
        on_best_state_found;
    std::function<bool(size_t iteration, double energy, double energy_error)> 
        should_stop;
  };

  Optimizer(const OptimizerParams& params, 
            const MPI_Comm& comm,
            int rank,
            int mpi_size);

  /**
   * @brief Perform line search optimization
   * 
   * @param initial_state Initial TPS state
   * @param energy_evaluator Function to evaluate energy and gradient
   * @param callback Optional callback for monitoring progress
   * @return Optimization result
   */
  OptimizationResult LineSearchOptimize(
      const WaveFunctionT& initial_state,
      std::function<std::pair<TenElemT, WaveFunctionT>(const WaveFunctionT&)> energy_evaluator,
      const OptimizationCallback& callback = OptimizationCallback{});

  /**
   * @brief Perform line search optimization with energy error support
   * 
   * @param initial_state Initial TPS state
   * @param energy_evaluator Function to evaluate energy, gradient, and error
   * @param callback Optional callback for monitoring progress
   * @return Optimization result
   */
  OptimizationResult LineSearchOptimize(
      const WaveFunctionT& initial_state,
      std::function<std::tuple<TenElemT, WaveFunctionT, double>(const WaveFunctionT&)> energy_evaluator,
      const OptimizationCallback& callback = OptimizationCallback{});

  /**
   * @brief Perform iterative optimization with energy error support
   * 
   * @param initial_state Initial TPS state
   * @param energy_evaluator Function to evaluate energy, gradient, and error
   * @param callback Optional callback for monitoring progress
   * @param Ostar_samples Optional O^*(S) samples for stochastic reconfiguration
   * @param Ostar_mean Optional average O^* tensor for stochastic reconfiguration
   * @return Optimization result
   */
  OptimizationResult IterativeOptimize(
      const WaveFunctionT& initial_state,
      std::function<std::tuple<TenElemT, WaveFunctionT, double>(const WaveFunctionT&)> energy_evaluator,
      const OptimizationCallback& callback = OptimizationCallback{},
      const std::vector<WaveFunctionT>* Ostar_samples = nullptr,
      const WaveFunctionT* Ostar_mean = nullptr);



  /**
   * @brief Calculate natural gradient using stochastic reconfiguration
   * 
   * ✅ MPI COMMUNICATION REQUIRED: This function DOES use MPI for distributed CG solving!
   * 
   * Unlike state update functions, SR natural gradient calculation requires distributed
   * computation across all ranks:
   * - Master rank coordinates CG iterations 
   * - All ranks participate in matrix-vector multiplications
   * - Broadcast/gather operations for CG algorithm convergence
   * 
   * This is algorithm-internal MPI communication, NOT state distribution.
   * 
   * @param gradient Standard gradient (valid ONLY on master rank)
   * @param Ostar_samples O^*(S) tensor samples (distributed across ranks)
   * @param Ostar_mean Average O^* tensor (valid ONLY on master rank)
   * @param init_guess Initial guess for conjugate gradient solver
   * @return Natural gradient and number of CG iterations (valid ONLY on master rank)
   */
  std::pair<WaveFunctionT, size_t> CalculateNaturalGradient(
      const WaveFunctionT& gradient,
      const std::vector<WaveFunctionT>& Ostar_samples,
      const WaveFunctionT& Ostar_mean,
      const WaveFunctionT& init_guess);

  /**
   * @brief Apply stochastic reconfiguration update
   * 
   * @param current_state Current TPS state
   * @param gradient Standard gradient
   * @param Ostar_samples O^*(S) tensor samples
   * @param Ostar_mean Average O^* tensor
   * @param step_length Step length
   * @param init_guess Initial guess for CG solver
   * @param normalize Whether to normalize the natural gradient
   * @return Updated state, natural gradient norm, and CG iterations
   */
  std::tuple<WaveFunctionT, double, size_t> StochasticReconfigurationUpdate(
      const WaveFunctionT& current_state,
      const WaveFunctionT& gradient,
      const std::vector<WaveFunctionT>& Ostar_samples,
      const WaveFunctionT& Ostar_mean,
      double step_length,
      const WaveFunctionT& init_guess,
      bool normalize = false);

  /**
   * @brief Apply bounded gradient element update
   * 
   * 🚫 MPI RESPONSIBILITY: This function does NOT broadcast the updated state!
   * Energy evaluator handles all state distribution for Monte Carlo sampling.
   * 
   * @param current_state Current TPS state (valid on all ranks)
   * @param gradient Gradient (valid ONLY on master rank)
   * @param step_length Step length
   * @return Updated TPS state (valid ONLY on master rank)
   */
  WaveFunctionT BoundedGradientUpdate(const WaveFunctionT& current_state,
                               const WaveFunctionT& gradient,
                               double step_length);

  /**
   * @brief Apply AdaGrad update with adaptive learning rates
   * 
   * 🚫 MPI RESPONSIBILITY: This function does NOT broadcast the updated state!
   * Energy evaluator handles all state distribution for Monte Carlo sampling.
   * 
   * @param current_state Current TPS state (valid on all ranks)
   * @param gradient Gradient (valid ONLY on master rank)
   * @param step_length Step length (base learning rate)
   * @return Updated TPS state (valid ONLY on master rank)
   */
  WaveFunctionT AdaGradUpdate(const WaveFunctionT& current_state,
                       const WaveFunctionT& gradient,
                       double step_length);

  /**
   * @brief Apply Adam update with bias-corrected moment estimates
   * 
   * MPI RESPONSIBILITY: This function does NOT broadcast the updated state!
   * Energy evaluator handles all state distribution for Monte Carlo sampling.
   * 
   * Algorithm:
   *   m_t = β₁ m_{t-1} + (1-β₁) g_t
   *   v_t = β₂ v_{t-1} + (1-β₂) g_t²
   *   m̂_t = m_t / (1 - β₁^t)
   *   v̂_t = v_t / (1 - β₂^t)
   *   θ_{t+1} = θ_t - α m̂_t / (√v̂_t + ε)
   * 
   * With optional decoupled weight decay (AdamW):
   *   θ_{t+1} = (1 - αλ) θ_t - α m̂_t / (√v̂_t + ε)
   * 
   * @param current_state Current TPS state (valid on all ranks)
   * @param gradient Gradient (valid ONLY on master rank)
   * @param step_length Step length (learning rate)
   * @param params Adam parameters (beta1, beta2, epsilon, weight_decay)
   * @return Updated TPS state (valid ONLY on master rank)
   */
  WaveFunctionT AdamUpdate(const WaveFunctionT& current_state,
                           const WaveFunctionT& gradient,
                           double step_length,
                           const AdamParams& params);

  /**
   * @brief Get current learning rate based on iteration and scheduler
   * 
   * @param iteration Current optimization iteration
   * @param current_energy Current energy value (for energy-aware schedulers)
   * @return Learning rate for this iteration
   */
  double GetCurrentLearningRate(size_t iteration, double current_energy = 0.0) const;

  // =============================================================================
  // BACKWARD COMPATIBILITY: Static methods for legacy VMC PEPS code
  // =============================================================================
  
  // Legacy enum-based helper methods removed - use OptimizerParams.IsAlgorithm<T>() instead

  /**
   * @brief Get spike detection statistics from the last optimization run.
   */
  const SpikeStatistics &GetSpikeStats() const { return spike_stats_; }

  /**
   * @brief Check if optimization should stop based on convergence criteria
   * 
   * @param current_energy Current energy value
   * @param previous_energy Previous energy value  
   * @param gradient_norm Current gradient norm
   * @param iterations_without_improvement Number of iterations without improvement
   * @return True if optimization should stop
   */
  bool ShouldStop(double current_energy, double previous_energy, double gradient_norm, 
                  size_t iterations_without_improvement);

  /**
   * @brief Clear up intermediate data and reset optimization state
   * 
   * This method should be called after optimization is complete to:
   * - Free memory used by intermediate data structures
   * - Reset algorithm-specific state variables
   * - Prepare for potential future optimizations
   * 
   * Note: This method is automatically called at the end of optimization methods
   * (LineSearchOptimize, IterativeOptimize). Users typically don't need to call
   * it manually unless they want to explicitly clean up state between operations.
   */
  void ClearUp();

 private:
  struct LBFGSHistoryPair {
    WaveFunctionT s;
    WaveFunctionT y;
    double rho = 0.0;  // 1 / Re(<s, y>)
  };

  struct LBFGSSnapshot {
    bool valid = false;
    std::deque<LBFGSHistoryPair> history;
    WaveFunctionT anchor_state;
    WaveFunctionT anchor_gradient;
    bool has_anchor = false;
  };

  enum class WolfeStepStatus {
    kAccepted = 0,
    kFailed = 1
  };

  OptimizerParams params_;
  MPI_Comm comm_;
  int rank_;
  int mpi_size_;
  
  // No RNG state needed inside Optimizer; stochasticity belongs to Monte Carlo components.

  // AdaGrad state
  WaveFunctionT accumulated_gradients_;
  bool adagrad_initialized_;
  
  // SGD Momentum state
  WaveFunctionT velocity_;
  bool sgd_momentum_initialized_;

  // Adam state (master rank ONLY)
  WaveFunctionT first_moment_;      // m_t: first moment estimate
  WaveFunctionT second_moment_;     // v_t: second moment estimate
  size_t adam_timestep_;            // t: iteration counter for bias correction
  bool adam_initialized_;

  // L-BFGS state (master rank ONLY)
  std::deque<LBFGSHistoryPair> lbfgs_history_;
  WaveFunctionT lbfgs_anchor_state_;
  WaveFunctionT lbfgs_anchor_gradient_;
  bool lbfgs_has_anchor_ = false;
  size_t lbfgs_skip_curvature_count_ = 0;
  size_t lbfgs_damping_count_ = 0;
  size_t lbfgs_descent_reset_count_ = 0;
  LBFGSSnapshot lbfgs_prev_snapshot_;

  // Spike detection state (master rank ONLY)
  EMATracker ema_energy_;
  EMATracker ema_error_;
  EMATracker ema_grad_norm_;
  EMATracker ema_ngrad_norm_;
  double prev_energy_ = 0.0;
  SpikeStatistics spike_stats_;

  // S4 rollback state (master rank ONLY, allocated only when enable_rollback)
  WaveFunctionT prev_accepted_state_;
  bool has_prev_state_ = false;

  // Spike detection and recovery methods (master rank evaluates, then broadcasts)
  SpikeSignal DetectS1S2_(double error, double grad_norm) const;
  SpikeSignal DetectS3_(double ngrad_norm, size_t sr_iters) const;
  SpikeSignal DetectS4_(double energy) const;
  SpikeAction DecideAction_(SpikeSignal signal, size_t attempts, size_t step) const;
  SpikeAction BroadcastAction_(SpikeAction action);
  void LogSpikeEvent_(size_t step, size_t attempt, SpikeSignal signal,
                      SpikeAction action, double value, double threshold);

  // Checkpoint
  void SaveCheckpoint_(const WaveFunctionT& state, size_t step,
                       const std::vector<TenElemT>& energy_traj,
                       const std::vector<double>& error_traj);

  // L-BFGS helpers
  double RealInnerProduct_(const WaveFunctionT& lhs, const WaveFunctionT& rhs) const;
  void ResetLBFGSState_();
  void SaveLBFGSSnapshot_();
  void RestoreLBFGSSnapshot_();
  void UpdateLBFGSHistoryFromAnchor_(const WaveFunctionT& current_state,
                                     const WaveFunctionT& current_gradient,
                                     const LBFGSParams& params);
  WaveFunctionT ComputeLBFGSSearchDirection_(const WaveFunctionT& gradient,
                                             const LBFGSParams& params);
  std::pair<WaveFunctionT, double> ApplyFixedLBFGSStep_(const WaveFunctionT& current_state,
                                                         const WaveFunctionT& search_direction,
                                                         double learning_rate);
  std::tuple<WolfeStepStatus, WaveFunctionT, double> StrongWolfeLBFGSStep_(
      const WaveFunctionT& current_state,
      const WaveFunctionT& search_direction,
      const std::function<std::tuple<TenElemT, WaveFunctionT, double>(const WaveFunctionT&)>& energy_evaluator,
      double learning_rate,
      const LBFGSParams& params,
      double phi0,
      double dphi0);

  // Helper methods
  void LogOptimizationStep(size_t iteration,
                          double energy,
                          double energy_error,
                          double gradient_norm,
                          double step_length = 0.0,
                          const std::vector<double>& accept_rates = {},
                          size_t sr_iterations = 0,
                          double sr_natural_grad_norm = 0.0,
                          double energy_eval_time = 0.0,
                          double update_time = 0.0);
  
  template<typename UpdateFunc>
  OptimizationResult PerformOptimization(
      const WaveFunctionT& initial_state,
      std::function<std::pair<TenElemT, WaveFunctionT>(const WaveFunctionT&)> energy_evaluator,
      UpdateFunc update_func,
      const OptimizationCallback& callback);

  // Algorithm-specific update methods
  
  /**
   * @brief SGD with momentum and Nesterov acceleration support
   * 
   * 🚫 MPI RESPONSIBILITY: This function does NOT broadcast the updated state!
   * Energy evaluator handles all state distribution for Monte Carlo sampling.
   * 
   * Unified implementation naturally handles all SGD variants:
   * - momentum = 0.0: reduces to vanilla SGD 
   * - momentum > 0.0 + nesterov = false: standard momentum SGD
   * - momentum > 0.0 + nesterov = true: Nesterov accelerated gradient
   */
  WaveFunctionT SGDUpdate(const WaveFunctionT& current_state,
                         const WaveFunctionT& gradient,
                         double step_length,
                         const SGDParams& params);
  
 };



} // namespace qlpeps

#include "qlpeps/optimizer/optimizer_impl.h"

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_H 
