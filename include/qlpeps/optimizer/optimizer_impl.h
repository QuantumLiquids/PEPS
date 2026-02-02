// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Implementation for optimizer.
*
* 🚫 CRITICAL MPI ARCHITECTURE: This implementation follows strict responsibility separation
*
 * 【CORE PRINCIPLE】: "Clear MPI responsibility: Optimizer handles algorithm MPI, Energy evaluator owns STATE distribution."
*
* This design eliminates triple redundant broadcasts (3→1 per iteration) and follows 
* Linus Torvalds' "good taste" philosophy: "Good taste is about eliminating special 
* cases and making the normal case work correctly."
*
 * 【RESPONSIBILITY BOUNDARIES】:
 * - Optimizer:        Algorithm computation (SR uses MPI for CG solving), NO state broadcasts
 * - Energy Evaluator: State distribution owner, broadcasts states for Monte Carlo sampling  
 * - VMC Executor:     Final synchronization guarantee at optimization completion
*
 * 【MPI FLOW PER ITERATION】:
 * 1. Optimizer updates state (master rank only)        ← NO state broadcast
 * 2. SR algorithm: Distributed CG solving              ← Optimizer internal MPI
 * 3. Energy evaluator receives state (master only)     ← NO state broadcast yet
 * 4. Energy evaluator broadcasts for MC sampling       ← SINGLE state broadcast
 * 5. Distributed Monte Carlo execution                 ← All ranks
 * 6. Energy evaluator gathers gradients to master      ← Standard gather
 * 7. Loop continues...
*
* PERFORMANCE IMPACT: 67% reduction in state broadcast overhead per optimization step
*
* TODO: Implement Stability Monitoring: Add checks in the code to automatically detect and reject unphysical energy values, reverting to the previous stable
     state.
   In cent:/home/gzcgu/haoxinwang/finite-size_PEPS_tJ/run_from_ipeps/InitDoping0.06D8_To12x12Mu1.5/iPEPS12x12D8SRMu1.5.log
   is quite unstable example.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H

#include <iomanip>
#include <algorithm>
#include <complex>
#include <filesystem>
#include <fstream>
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/utility/helpers.h"
#include "qlpeps/utility/filesystem_utils.h"
#include "qlten/utility/timer.h"

namespace qlpeps {
using qlten::Timer;

template<typename TenElemT, typename QNT>
Optimizer<TenElemT, QNT>::Optimizer(const OptimizerParams &params,
                                    const MPI_Comm &comm,
                                    int rank,
                                    int mpi_size)
    : params_(params), comm_(comm), rank_(rank), mpi_size_(mpi_size),
      adagrad_initialized_(false),
      sgd_momentum_initialized_(false),
      adam_timestep_(0),
      adam_initialized_(false),
      ema_energy_(params.spike_recovery_params.ema_window),
      ema_error_(params.spike_recovery_params.ema_window),
      ema_grad_norm_(params.spike_recovery_params.ema_window),
      ema_ngrad_norm_(params.spike_recovery_params.ema_window),
      prev_energy_(0.0),
      has_prev_state_(false) {
}

template<typename TenElemT, typename QNT>
double Optimizer<TenElemT, QNT>::GetCurrentLearningRate(size_t iteration, double current_energy) const {
  double learning_rate = params_.base_params.learning_rate;  // Default/base rate
  
  if (params_.base_params.lr_scheduler) {
    learning_rate = params_.base_params.lr_scheduler->GetLearningRate(iteration, current_energy);
  }
  
  return learning_rate;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::OptimizationResult
Optimizer<TenElemT, QNT>::LineSearchOptimize(
    const SITPST &initial_state,
    std::function<std::pair<TenElemT, SITPST>(const SITPST &)> energy_evaluator,
    const OptimizationCallback &callback) {

  OptimizationResult result;
  result.optimized_state = initial_state;
  result.converged = false;
  result.total_iterations = 0;

  // Evaluate initial energy and gradient
  auto [initial_energy, initial_gradient] = energy_evaluator(initial_state);
  result.energy_trajectory.push_back(initial_energy);
  result.min_energy = std::real(initial_energy);

  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    result.gradient_norms.push_back(std::sqrt(initial_gradient.NormSquare()));
  }

  // Determine search direction based on algorithm type
  SITPST search_direction;
  size_t cg_iterations = 0;
  double natural_grad_norm = 0.0;

  // Simple dispatch - currently only SGD and SR variants supported for line search
  if (params_.IsAlgorithm<SGDParams>()) {
    search_direction = initial_gradient;
  } else if (params_.IsAlgorithm<StochasticReconfigurationParams>()) {
    // For natural gradient, we need to calculate it
    // This is a simplified version - in practice, you'd need the gradient samples
    SITPST init_guess(initial_state.rows(), initial_state.cols(), initial_state.PhysicalDim());
    auto [natural_grad, cg_iters] = CalculateNaturalGradient(
        initial_gradient, {}, {}, init_guess);
    search_direction = natural_grad;
    cg_iterations = cg_iters;
    natural_grad_norm = std::sqrt(natural_grad.NormSquare());
  } else {
    throw std::runtime_error("Invalid algorithm type for line search");
  }

  // Perform line search along the search direction
  SITPST best_state = initial_state;
  double best_energy = std::real(initial_energy);
  double cumulative_step = 0.0;

  // For line search, use multiple steps with the current learning rate
  // Default to 3 line search steps if no specific configuration
  size_t max_line_search_steps = 3;
  
  for (size_t i = 0; i < max_line_search_steps; ++i) {
    double step_length = GetCurrentLearningRate(i, best_energy);
    cumulative_step += step_length;

    // Start timer for update step
    Timer update_timer("line_search_update");

    // Update state along search direction
    SITPST test_state = initial_state;
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      test_state += (-cumulative_step) * search_direction;
    }

    // Get elapsed time for update step
    double update_time = update_timer.Elapsed();

    // Start timer for energy evaluation (the dominant computational cost)
    Timer energy_eval_timer("line_search_energy_evaluation");

    // Evaluate energy at new point
    auto [test_energy, test_gradient] = energy_evaluator(test_state);
    result.energy_trajectory.push_back(test_energy);

    // Get elapsed time for energy evaluation
    double energy_eval_time = energy_eval_timer.Elapsed();

    // Extract energy error from the energy evaluator if available
    double energy_error = 0.0;
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      // The energy error should be calculated by the energy evaluator
      // For now, we'll use a placeholder
      result.energy_error_trajectory.push_back(energy_error);
      result.gradient_norms.push_back(std::sqrt(test_gradient.NormSquare()));
    }

    // Update best state if energy improved
    if (std::real(test_energy) < best_energy) {
      best_energy = std::real(test_energy);
      best_state = test_state;
      result.min_energy = best_energy;

      if (callback.on_best_state_found) {
        callback.on_best_state_found(best_state, best_energy);
      }
    }

    // Log progress
    LogOptimizationStep(i,
                        std::real(test_energy),
                        energy_error,
                        std::sqrt(test_gradient.NormSquare()),
                        cumulative_step,
                        {},
                        0,
                        0.0,
                        energy_eval_time,
                        update_time);

    if (callback.on_iteration) {
      callback.on_iteration(i, std::real(test_energy), energy_error, std::sqrt(test_gradient.NormSquare()));
    }

    if (callback.should_stop && callback.should_stop(i, std::real(test_energy), energy_error)) {
      break;
    }
  }

  result.optimized_state = best_state;
  result.final_energy = best_energy;
  result.total_iterations = max_line_search_steps;
  result.converged = true;

  // Clean up optimization state
  ClearUp();

  return result;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::OptimizationResult
Optimizer<TenElemT, QNT>::LineSearchOptimize(
    const SITPST &initial_state,
    std::function<std::tuple<TenElemT, SITPST, double>(const SITPST &)> energy_evaluator,
    const OptimizationCallback &callback) {

  OptimizationResult result;
  result.optimized_state = initial_state;
  result.converged = false;
  result.total_iterations = 0;

  // Evaluate initial energy and gradient
  auto [initial_energy, initial_gradient, initial_error] = energy_evaluator(initial_state);
  result.energy_trajectory.push_back(initial_energy);
  result.min_energy = std::real(initial_energy);

  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    result.energy_error_trajectory.push_back(initial_error);
    result.gradient_norms.push_back(std::sqrt(initial_gradient.NormSquare()));
  }

  // Determine search direction based on algorithm type
  SITPST search_direction;
  size_t cg_iterations = 0;
  double natural_grad_norm = 0.0;

  // Simple dispatch - currently only SGD and SR variants supported for line search
  if (params_.IsAlgorithm<SGDParams>()) {
    search_direction = initial_gradient;
  } else if (params_.IsAlgorithm<StochasticReconfigurationParams>()) {
    // For natural gradient, we need to calculate it
    // This is a simplified version - in practice, you'd need the gradient samples
    SITPST init_guess(initial_state.rows(), initial_state.cols(), initial_state.PhysicalDim());
    auto [natural_grad, cg_iters] = CalculateNaturalGradient(
        initial_gradient, {}, {}, init_guess);
    search_direction = natural_grad;
    cg_iterations = cg_iters;
    natural_grad_norm = std::sqrt(natural_grad.NormSquare());
  } else {
    throw std::runtime_error("Invalid algorithm type for line search");
  }

  // Perform line search along the search direction
  SITPST best_state = initial_state;
  double best_energy = std::real(initial_energy);
  double cumulative_step = 0.0;

  // For line search, use multiple steps with the current learning rate
  // Default to 3 line search steps if no specific configuration
  size_t max_line_search_steps = 3;
  
  for (size_t i = 0; i < max_line_search_steps; ++i) {
    double step_length = GetCurrentLearningRate(i, best_energy);
    cumulative_step += step_length;

    // Start timer for update step
    Timer update_timer("line_search_update");

    // Update state along search direction
    SITPST test_state = initial_state;
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      test_state += (-cumulative_step) * search_direction;
    }

    // Get elapsed time for update step
    double update_time = update_timer.Elapsed();

    // Start timer for energy evaluation (the dominant computational cost)
    Timer energy_eval_timer("line_search_energy_evaluation");

    // Evaluate energy at new point
    auto [test_energy, test_gradient, test_error] = energy_evaluator(test_state);
    result.energy_trajectory.push_back(test_energy);

    // Get elapsed time for energy evaluation
    double energy_eval_time = energy_eval_timer.Elapsed();

    // Store energy error and gradient norm only on master rank
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      result.energy_error_trajectory.push_back(test_error);
      result.gradient_norms.push_back(std::sqrt(test_gradient.NormSquare()));
    }

    // Update best state if energy improved
    if (std::real(test_energy) < best_energy) {
      best_energy = std::real(test_energy);
      best_state = test_state;
      result.min_energy = best_energy;

      if (callback.on_best_state_found) {
        callback.on_best_state_found(best_state, best_energy);
      }
    }

    // Log progress
    LogOptimizationStep(i,
                        std::real(test_energy),
                        test_error,
                        std::sqrt(test_gradient.NormSquare()),
                        cumulative_step,
                        {},
                        0,
                        0.0,
                        energy_eval_time,
                        update_time);

    if (callback.on_iteration) {
      callback.on_iteration(i, std::real(test_energy), test_error, std::sqrt(test_gradient.NormSquare()));
    }

    if (callback.should_stop && callback.should_stop(i, std::real(test_energy), test_error)) {
      break;
    }
  }

  result.optimized_state = best_state;
  result.final_energy = best_energy;
  result.total_iterations = max_line_search_steps;
  result.converged = true;

  // Clean up optimization state
  ClearUp();

  return result;
}

/**
 * @brief Iterative optimization with MPI support
 * 
 * MPI Behavior Documentation:
 * - INPUT: initial_state valid ONLY on master rank (optimizer manages distribution)
 * - energy_evaluator: Takes state from master rank, returns (energy, gradient, energy_error) where:
 *   * energy: Valid on ALL ranks (broadcast by energy_evaluator)
 *   * gradient: Valid ONLY on master rank (gathered by energy_evaluator)  
 *   * energy_error: Valid ONLY on master rank
 * - All algorithm updates are performed only on master rank
 * - Energy evaluator internally broadcasts state for Monte Carlo sampling
 * - Optimization statistics (energy_trajectory, etc.) are tracked only on master rank
 * 
 * @param initial_state Initial TPS state (valid ONLY on master rank)
 * @param energy_evaluator Function returning (energy, gradient, energy_error)
 * @param callback Optional callback for monitoring progress
 * @param Ostar_samples O^*(S) samples for stochastic reconfiguration (if used)
 * @param Ostar_mean Average O^* for stochastic reconfiguration (if used)
 * @return Optimization result with final state valid on all ranks
 */
template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::OptimizationResult
Optimizer<TenElemT, QNT>::IterativeOptimize(
    const SITPST &initial_state,
    std::function<std::tuple<TenElemT, SITPST, double>(const SITPST &)> energy_evaluator,
    const OptimizationCallback &callback,
    const std::vector<SITPST> *Ostar_samples,
    const SITPST *Ostar_mean) {

  OptimizationResult result;
  result.optimized_state = initial_state;
  result.converged = false;
  result.total_iterations = 0;

  SITPST current_state = initial_state;
  SITPST best_state = initial_state;
  double best_energy = std::numeric_limits<double>::max();

  // Initialize for stochastic reconfiguration if needed
  SITPST sr_init_guess;
  if (params_.IsAlgorithm<StochasticReconfigurationParams>()) {
    sr_init_guess = SITPST(initial_state.rows(), initial_state.cols(), initial_state.PhysicalDim());
  }

  // Advanced stopping criteria tracking
  double previous_energy = std::numeric_limits<double>::max();
  size_t iterations_without_improvement = 0;  // plateau counter

  // Spike recovery config aliases
  const auto &spike_cfg = params_.spike_recovery_params;
  const auto &ckpt_cfg = params_.checkpoint_params;
  const bool is_sr = params_.IsAlgorithm<StochasticReconfigurationParams>();

  size_t total_iterations_performed = 0;
  bool done = false;
  for (size_t iter = 0; iter < params_.base_params.max_iterations && !done; ++iter) {
    bool step_accepted = false;
    size_t attempts = 0;

    while (!step_accepted) {
      // === A: Evaluate energy and gradient ===
      Timer energy_eval_timer("energy_evaluation");
      auto [current_energy, current_gradient, current_error] = energy_evaluator(current_state);
      double energy_eval_time = energy_eval_timer.Elapsed();

      double grad_norm = 0.0;
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        grad_norm = std::sqrt(current_gradient.NormSquare());
      }

      // === B: S1/S2 detection (before any trajectory or state update) ===
      // Guard uses iter > 0 (not ema_.IsInitialized()) so ALL ranks enter together for MPI_Bcast.
      if (spike_cfg.enable_auto_recover && iter > 0) {
        SpikeSignal signal = SpikeSignal::kNone;
        SpikeAction action = SpikeAction::kAccept;
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          signal = DetectS1S2_(current_error, grad_norm);
          if (signal != SpikeSignal::kNone) {
            action = DecideAction_(signal, attempts, iter);
          }
        }
        action = BroadcastAction_(action);
        if (action == SpikeAction::kResample || action == SpikeAction::kRollback) {
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            double thresh = (signal == SpikeSignal::kS1_ErrorbarSpike)
                ? spike_cfg.factor_err * ema_error_.Mean()
                : spike_cfg.factor_grad * ema_grad_norm_.Mean();
            double val = (signal == SpikeSignal::kS1_ErrorbarSpike) ? current_error : grad_norm;
            LogSpikeEvent_(iter, attempts, signal, action, val, thresh);
            if (action == SpikeAction::kRollback && has_prev_state_) {
              // MPI invariant: only master rank holds the authoritative current_state.
              // Non-master ranks' current_state is stale but harmless — the energy_evaluator
              // broadcasts the master's state at the start of each evaluation call.
              current_state = prev_accepted_state_;
              has_prev_state_ = false;
            }
          }
          if (action == SpikeAction::kResample) { attempts++; }
          else { attempts = 0; }
          continue;
        }
        if (action == SpikeAction::kAcceptWithWarning && rank_ == qlten::hp_numeric::kMPIMasterRank) {
          double thresh = (signal == SpikeSignal::kS1_ErrorbarSpike)
              ? spike_cfg.factor_err * ema_error_.Mean()
              : spike_cfg.factor_grad * ema_grad_norm_.Mean();
          double val = (signal == SpikeSignal::kS1_ErrorbarSpike) ? current_error : grad_norm;
          LogSpikeEvent_(iter, attempts, signal, action, val, thresh);
        }
      }

      // === C: Compute learning rate and optimization update ===
      double learning_rate = GetCurrentLearningRate(iter, (iter == 0) ? std::real(current_energy) : previous_energy);

      Timer update_timer("optimization_update");

      // Gradient preprocessing (clipping) for first-order methods
      SITPST preprocessed_gradient = current_gradient;
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        if (params_.IsFirstOrder()) {
          if (params_.base_params.clip_value && *(params_.base_params.clip_value) > 0.0) {
            preprocessed_gradient.ElementWiseClipTo(*(params_.base_params.clip_value));
          }
          if (params_.base_params.clip_norm && *(params_.base_params.clip_norm) > 0.0) {
            preprocessed_gradient.ClipByGlobalNorm(*(params_.base_params.clip_norm));
          }
        }
      }

      SITPST updated_state;
      size_t sr_iterations = 0;
      double sr_natural_grad_norm = 0.0;

      // Single algorithm dispatch with correct learning rate
      std::visit([&](const auto& algo_params) {
        using T = std::decay_t<decltype(algo_params)>;

        if constexpr (std::is_same_v<T, SGDParams>) {
          updated_state = SGDUpdate(current_state, preprocessed_gradient, learning_rate, algo_params);
        }
        else if constexpr (std::is_same_v<T, StochasticReconfigurationParams>) {
          // Fix Intel icpx compiler crash: avoid structured binding in lambda
          auto sr_result = StochasticReconfigurationUpdate(
              current_state, current_gradient,
              Ostar_samples ? *Ostar_samples : std::vector<SITPST>{},
              Ostar_mean ? *Ostar_mean : SITPST{},
              learning_rate, sr_init_guess, algo_params.normalize_update);
          updated_state = std::get<0>(sr_result);
          sr_natural_grad_norm = std::get<1>(sr_result);
          sr_iterations = std::get<2>(sr_result);
          sr_init_guess = std::get<0>(sr_result);
        }
        else if constexpr (std::is_same_v<T, AdaGradParams>) {
          updated_state = AdaGradUpdate(current_state, preprocessed_gradient, learning_rate);
        }
        else if constexpr (std::is_same_v<T, AdamParams>) {
          updated_state = AdamUpdate(current_state, preprocessed_gradient, learning_rate, algo_params);
        }
        else if constexpr (std::is_same_v<T, LBFGSParams>) {
          throw std::runtime_error("L-BFGS algorithm not yet implemented");
        }
        else {
          throw std::runtime_error("Unsupported algorithm type for iterative optimization");
        }
      }, params_.algorithm_params);

      double update_time = update_timer.Elapsed();

      // === D: S3 detection (SR-only, after CG solve) ===
      // Known limitation: if this were extended to non-SR algorithms (AdaGrad/Adam),
      // a RESAMPLE here would cause the algorithm's accumulator state (e.g., AdaGrad's
      // sum-of-squared-gradients, Adam's moment estimates) to be double-updated, since
      // algorithm dispatch (section C) already ran. Currently safe because S3 is gated
      // on is_sr, and SR's CG solver has no persistent accumulator state.
      // See RFC 2025-09-01 "Known Limitations" section.
      // Guard uses iter > 0 (not ema_.IsInitialized()) so ALL ranks enter together for MPI_Bcast.
      if (is_sr && spike_cfg.enable_auto_recover && iter > 0) {
        SpikeSignal signal = SpikeSignal::kNone;
        SpikeAction action = SpikeAction::kAccept;
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          signal = DetectS3_(sr_natural_grad_norm, sr_iterations);
          if (signal != SpikeSignal::kNone) {
            action = DecideAction_(signal, attempts, iter);
          }
        }
        action = BroadcastAction_(action);
        if (action == SpikeAction::kResample || action == SpikeAction::kRollback) {
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            LogSpikeEvent_(iter, attempts, signal, action, sr_natural_grad_norm,
                           spike_cfg.factor_ngrad * ema_ngrad_norm_.Mean());
            if (action == SpikeAction::kRollback && has_prev_state_) {
              current_state = prev_accepted_state_;
              has_prev_state_ = false;
            }
          }
          if (action == SpikeAction::kResample) { attempts++; }
          else { attempts = 0; }
          continue;
        }
        if (action == SpikeAction::kAcceptWithWarning && rank_ == qlten::hp_numeric::kMPIMasterRank) {
          LogSpikeEvent_(iter, attempts, signal, action, sr_natural_grad_norm,
                         spike_cfg.factor_ngrad * ema_ngrad_norm_.Mean());
        }
      }

      // === D2: S4 detection (rollback, opt-in) ===
      // Guard uses iter > 0 (not ema_.IsInitialized()) so ALL ranks enter together for MPI_Bcast.
      if (spike_cfg.enable_rollback && iter > 0) {
        SpikeSignal signal = SpikeSignal::kNone;
        SpikeAction action = SpikeAction::kAccept;
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          signal = DetectS4_(std::real(current_energy));
          if (signal != SpikeSignal::kNone) {
            action = SpikeAction::kRollback;
          }
        }
        action = BroadcastAction_(action);
        if (action == SpikeAction::kRollback) {
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            LogSpikeEvent_(iter, attempts, signal, action,
                           std::real(current_energy),
                           ema_energy_.Mean() + spike_cfg.sigma_k * ema_energy_.Std());
            if (has_prev_state_) {
              // MPI invariant: only master rank holds the authoritative current_state.
              // Non-master ranks' current_state is stale but harmless — the energy_evaluator
              // broadcasts the master's state at the start of each evaluation call.
              current_state = prev_accepted_state_;
              has_prev_state_ = false;
            }
          }
          attempts = 0;
          continue;
        }
      }

      // === E: ACCEPT — update EMA trackers (only on accepted steps) ===
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        ema_energy_.Update(std::real(current_energy));
        ema_error_.Update(current_error);
        ema_grad_norm_.Update(grad_norm);
        if (is_sr) {
          ema_ngrad_norm_.Update(sr_natural_grad_norm);
        }
        prev_energy_ = std::real(current_energy);
      }

      // === F: Trajectories, best-state, stopping, state update, log, callback ===
      result.energy_trajectory.push_back(current_energy);
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        result.energy_error_trajectory.push_back(current_error);
        result.gradient_norms.push_back(grad_norm);
      }

      total_iterations_performed = iter + 1;

      // Update best state
      if (std::real(current_energy) < best_energy) {
        best_energy = std::real(current_energy);
        best_state = current_state;
        result.min_energy = best_energy;
        iterations_without_improvement = 0;
        if (callback.on_best_state_found) {
          callback.on_best_state_found(best_state, best_energy);
        }
      } else {
        iterations_without_improvement++;
      }

      // Advanced stopping criteria (skip first iteration)
      if (iter > 0) {
        double current_energy_real = std::real(current_energy);
        bool should_stop = ShouldStop(current_energy_real, previous_energy, grad_norm, iterations_without_improvement);
        if (should_stop) {
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            LogOptimizationStep(iter, current_energy_real, current_error, grad_norm,
                               learning_rate, {}, sr_iterations, sr_natural_grad_norm,
                               energy_eval_time, update_time);
            result.converged = true;
          }
          step_accepted = true;
          done = true;
          break;
        }
        previous_energy = current_energy_real;
      } else {
        previous_energy = std::real(current_energy);
      }

      // Save prev state for S4 rollback BEFORE applying update
      if (spike_cfg.enable_rollback && rank_ == qlten::hp_numeric::kMPIMasterRank) {
        prev_accepted_state_ = current_state;
        has_prev_state_ = true;
      }

      // Apply the state update.
      // Note: updated_state is only meaningful on master rank (algorithm dispatch
      // computes on master). Non-master ranks rely on energy_evaluator to receive
      // the updated state via MPI broadcast on the next evaluation call.
      current_state = updated_state;

      // Log progress
      LogOptimizationStep(iter, std::real(current_energy), current_error, grad_norm,
                          learning_rate, {}, sr_iterations, sr_natural_grad_norm,
                          energy_eval_time, update_time);

      if (callback.on_iteration) {
        callback.on_iteration(iter, std::real(current_energy), current_error, grad_norm);
      }

      if (callback.should_stop && callback.should_stop(iter, std::real(current_energy), current_error)) {
        step_accepted = true;
        done = true;
        break;
      }

      // === G: Checkpoint ===
      if (ckpt_cfg.IsEnabled() && iter > 0 && (iter % ckpt_cfg.every_n_steps == 0)) {
        SaveCheckpoint_(current_state, iter, result.energy_trajectory, result.energy_error_trajectory);
      }

      step_accepted = true;
    } // end while (!step_accepted)
  } // end for

  result.optimized_state = best_state;
  result.final_energy = best_energy;
  result.total_iterations = total_iterations_performed;
  result.spike_stats = spike_stats_;

  // Ensure the final state is valid
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // Check that the state has the correct dimensions
    if (result.optimized_state.rows() != initial_state.rows() ||
        result.optimized_state.cols() != initial_state.cols()) {
      throw std::runtime_error("Optimized state has incorrect dimensions");
    }
  }

  // Clean up optimization state
  ClearUp();

  return result;
}



/**
 * @brief Unified SGD implementation with momentum and Nesterov support
 * 
 * ELEGANT UNIFICATION: Single implementation handles all SGD variants:
 * - momentum = 0.0: reduces to vanilla SGD (direct inline update)
 * - momentum > 0.0 + nesterov = false: standard momentum SGD
 * - momentum > 0.0 + nesterov = true: Nesterov accelerated gradient
 * 
 * MPI RESPONSIBILITY: Follows the same "no state broadcast" contract as vanilla SGD
 * - gradient: Valid ONLY on master rank (gathered by energy_evaluator)
 * - velocity_: Maintained ONLY on master rank (algorithm state isolation)
 * - Update: Performed ONLY on master rank 
 * - Broadcast: Energy evaluator's sole responsibility
 * 
 * Algorithm (unified mathematical formulation):
 * v_{t+1} = μ * v_t + g_t
 * θ_{t+1} = (1 - αλ) θ_t - α * (μ * v_{t+1} + g_t)  [if Nesterov, decoupled L2 (λ)]
 * θ_{t+1} = (1 - αλ) θ_t - α * v_{t+1}              [if standard momentum, decoupled L2]
 * 
 * Special case (μ=0): v_{t+1} = g_t → both reduce to θ_{t+1} = θ_t - α * g_t
 * This case uses direct inline update for simplicity.
 */
template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::SGDUpdate(const SITPST &current_state,
                                   const SITPST &gradient,
                                   double step_length,
                                   const SGDParams &params) {
  
  SITPST updated_state = current_state;
  
  // MPI VERIFICATION: Only master rank processes gradients and algorithm state
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // ⚙️ LAZY INITIALIZATION: Initialize velocity on first use (master rank only)
    if (!sgd_momentum_initialized_) {
      velocity_ = SITPST(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      // Default construction gives zero tensors - perfect for initial velocity
      sgd_momentum_initialized_ = true;
      
      // 🔍 DEBUG: Velocity state exists only on master rank
      // Non-master ranks never initialize or access velocity_
    }
    
    // Apply decoupled L2 weight decay (AdamW-style) to parameters
    if (params.weight_decay > 0.0) {
      const double decay_factor = 1.0 - step_length * params.weight_decay;
      updated_state *= decay_factor;
    }

    if (params.momentum > 0.0) {
      // Momentum SGD: maintain velocity state
      // v_{t+1} = μ * v_t + g_t
      velocity_ = params.momentum * velocity_ + gradient;
      
      if (params.nesterov) {
        // Nesterov: θ_{t+1} = θ_t - α * (μ * v_{t+1} + g_t)
        SITPST nesterov_update = params.momentum * velocity_ + gradient;
        updated_state += (-step_length) * nesterov_update;
      } else {
        // Standard momentum: θ_{t+1} = θ_t - α * v_{t+1}
        updated_state += (-step_length) * velocity_;
      }
    } else {
      // Vanilla SGD: direct inline update (already inside master-only block)
      updated_state += (-step_length) * gradient;
    }
  }
  
  // CRITICAL MPI DESIGN: Do NOT broadcast here!
  // 
  // RESPONSIBILITY SEPARATION: Energy evaluator owns all state distribution for Monte Carlo sampling.
  return updated_state;
}

template<typename TenElemT, typename QNT>
std::pair<typename Optimizer<TenElemT, QNT>::SITPST, size_t>
Optimizer<TenElemT, QNT>::CalculateNaturalGradient(
    const SITPST &gradient,
    const std::vector<SITPST> &Ostar_samples,
    const SITPST &Ostar_mean,
    const SITPST &init_guess) {

  // Get CG parameters from StochasticReconfigurationParams
  const auto& sr_params = params_.GetAlgorithmParams<StochasticReconfigurationParams>();
  const ConjugateGradientParams &cg_params = sr_params.cg_params;

  // Create S-matrix for stochastic reconfiguration
  SITPST *pOstar_mean = nullptr;
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    pOstar_mean = const_cast<SITPST *>(&Ostar_mean);
  }

  SRSMatrix s_matrix(const_cast<std::vector<SITPST> *>(&Ostar_samples), pOstar_mean, mpi_size_);
  s_matrix.diag_shift = cg_params.diag_shift;

  SITPST natural_gradient;
  size_t cg_iterations;

  if constexpr (QLTensor<TenElemT, QNT>::IsFermionic()) {
    auto signed_gradient = gradient;
    signed_gradient.ActFermionPOps();
    natural_gradient = ConjugateGradientSolver(s_matrix, signed_gradient, init_guess,
                                               cg_params.max_iter, cg_params.tolerance,
                                               cg_params.residue_restart_step, cg_iterations, comm_);
    natural_gradient.ActFermionPOps();
  } else {
    natural_gradient = ConjugateGradientSolver(s_matrix, gradient, init_guess,
                                               cg_params.max_iter, cg_params.tolerance,
                                               cg_params.residue_restart_step, cg_iterations, comm_);
  }

  return {natural_gradient, cg_iterations};
}

template<typename TenElemT, typename QNT>
std::tuple<typename Optimizer<TenElemT, QNT>::SITPST, double, size_t>
Optimizer<TenElemT, QNT>::StochasticReconfigurationUpdate(
    const SITPST &current_state,
    const SITPST &gradient,
    const std::vector<SITPST> &Ostar_samples,
    const SITPST &Ostar_mean,
    double step_length,
    const SITPST &init_guess,
    bool normalize) {

  // Calculate natural gradient using stochastic reconfiguration
  // This involves solving the SR equation which should be done by all cores together
  auto [natural_gradient, cg_iterations] = CalculateNaturalGradient(
      gradient, Ostar_samples, Ostar_mean, init_guess);

  double natural_grad_norm = std::sqrt(natural_gradient.NormSquare());

  if (normalize) {
    step_length /= std::sqrt(natural_grad_norm);
  }

  // Apply the update using the natural gradient
  SITPST updated_state = current_state;
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    updated_state += (-step_length) * natural_gradient;
  }

  return {updated_state, natural_grad_norm, cg_iterations};
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::BoundedGradientUpdate(const SITPST &current_state,
                                                const SITPST &gradient,
                                                double step_length) {
  SITPST updated_state = current_state;

  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // Apply bounded gradient update only on master rank
    // This matches the behavior of the original VMCPEPSExecutor
    for (size_t row = 0; row < current_state.rows(); ++row) {
      for (size_t col = 0; col < current_state.cols(); ++col) {
        const size_t phy_dim = gradient({row, col}).size();
        for (size_t compt = 0; compt < phy_dim; ++compt) {
          Tensor &grad_ten = const_cast<SITPST &>(gradient)({row, col})[compt];
          grad_ten.ElementWiseClipTo(step_length);
          updated_state({row, col})[compt] += (-step_length) * grad_ten;
        }
      }
    }
  }

  // CRITICAL MPI DESIGN: Do NOT broadcast here!
  // Energy evaluator owns all state distribution for Monte Carlo sampling.
  return updated_state;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::AdaGradUpdate(const SITPST &current_state,
                                        const SITPST &gradient,
                                        double step_length) {
  // MPI VERIFICATION: Only master rank processes gradients and algorithm state
  SITPST updated_state = current_state;
  
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // Get AdaGrad parameters from the algorithm params
    const auto& adagrad_params = params_.GetAlgorithmParams<AdaGradParams>();
    
    // LAZY INITIALIZATION: Initialize AdaGrad state on first use (master rank only)
    if (!adagrad_initialized_) {
      accumulated_gradients_ = SITPST(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      // Initialize with small values to avoid division by zero
      for (size_t row = 0; row < accumulated_gradients_.rows(); ++row) {
        for (size_t col = 0; col < accumulated_gradients_.cols(); ++col) {
          for (size_t i = 0; i < accumulated_gradients_({row, col}).size(); ++i) {
            accumulated_gradients_({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
            QNT div = gradient({row, col})[i].Div();
            accumulated_gradients_({row, col})[i].Fill(div, adagrad_params.initial_accumulator_value);
            // Note here for fermionic tensor, positive and negative are relatively define, 
            // In the AdaGrad, we fix the order of indexes to make it make sense.
          }
        }
      }
      adagrad_initialized_ = true;
      
      // DEBUG: AdaGrad state exists only on master rank
      // Non-master ranks never initialize or access accumulated_gradients_
    }

    // Update accumulated gradients: G_k = G_{k-1} + |gradient|^2
    SITPST squared_gradient = ElementWiseSquaredNorm(gradient);
    accumulated_gradients_ += squared_gradient;

    // Compute adaptive learning rates: 1/sqrt(G_k) for |G_k| > epsilon
    SITPST adaptive_rates = ElementWiseInverse(ElementWiseSqrt(accumulated_gradients_), adagrad_params.epsilon);

    // Apply AdaGrad update: θ_{k+1} = θ_k - η * adaptive_rates * gradient
    for (size_t row = 0; row < current_state.rows(); ++row) {
      for (size_t col = 0; col < current_state.cols(); ++col) {
        for (size_t i = 0; i < current_state({row, col}).size(); ++i) {
          // Compute adaptive step: step_length * adaptive_rate * gradient
          Tensor adaptive_step = ElementWiseMultiply(adaptive_rates({row, col})[i], gradient({row, col})[i]) * step_length;

          // Update state: θ_{k+1} = θ_k - adaptive_step
          updated_state({row, col})[i] += (-adaptive_step);
        }
      }
    }
  }

  // CRITICAL MPI DESIGN: Do NOT broadcast here!
  // Energy evaluator owns all state distribution for Monte Carlo sampling.
  return updated_state;
}

/**
 * @brief Adam optimizer with bias-corrected moment estimates
 * 
 * MPI CONTRACT:
 * - gradient: Valid ONLY on master rank (gathered by energy_evaluator)
 * - first_moment_, second_moment_, adam_timestep_: Master rank ONLY
 * - updated_state: Valid ONLY on master rank
 * - State broadcast: Energy evaluator's sole responsibility (NOT here)
 * 
 * Algorithm (AdamW-style decoupled weight decay):
 *   m_t = β₁ m_{t-1} + (1-β₁) g_t
 *   v_t = β₂ v_{t-1} + (1-β₂) g_t²
 *   m̂_t = m_t / (1 - β₁^t)
 *   v̂_t = v_t / (1 - β₂^t)
 *   θ_{t+1} = (1 - αλ) θ_t - α m̂_t / (√v̂_t + ε)
 */
template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::AdamUpdate(const SITPST &current_state,
                                     const SITPST &gradient,
                                     double step_length,
                                     const AdamParams &params) {
  SITPST updated_state = current_state;
  
  // MPI VERIFICATION: Only master rank processes gradients and algorithm state
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // LAZY INITIALIZATION on first use (master rank only)
    if (!adam_initialized_) {
      first_moment_ = SITPST(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      second_moment_ = SITPST(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      // Initialize with zeros
      for (size_t row = 0; row < first_moment_.rows(); ++row) {
        for (size_t col = 0; col < first_moment_.cols(); ++col) {
          for (size_t i = 0; i < first_moment_({row, col}).size(); ++i) {
            first_moment_({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
            second_moment_({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
            QNT div = gradient({row, col})[i].Div();
            first_moment_({row, col})[i].Fill(div, 0.0);
            second_moment_({row, col})[i].Fill(div, 0.0);
          }
        }
      }
      adam_timestep_ = 0;
      adam_initialized_ = true;
    }
    
    // Increment timestep
    adam_timestep_++;
    
    // Apply decoupled L2 weight decay (AdamW-style) to parameters
    if (params.weight_decay > 0.0) {
      const double decay_factor = 1.0 - step_length * params.weight_decay;
      updated_state *= decay_factor;
    }
    
    // Update biased first moment estimate: m_t = β₁ m_{t-1} + (1-β₁) g_t
    first_moment_ = params.beta1 * first_moment_ + (1.0 - params.beta1) * gradient;
    
    // Update biased second moment estimate: v_t = β₂ v_{t-1} + (1-β₂) |g_t|²
    SITPST squared_gradient = ElementWiseSquaredNorm(gradient);
    second_moment_ = params.beta2 * second_moment_ + (1.0 - params.beta2) * squared_gradient;
    
    // Compute bias correction factors
    double bias_correction1 = 1.0 - std::pow(params.beta1, adam_timestep_);
    double bias_correction2 = 1.0 - std::pow(params.beta2, adam_timestep_);
    
    // Compute bias-corrected estimates
    SITPST m_hat = (1.0 / bias_correction1) * first_moment_;
    SITPST v_hat = (1.0 / bias_correction2) * second_moment_;
    
    // Compute update: -α * m̂_t / (√v̂_t + ε)
    SITPST sqrt_v_hat = ElementWiseSqrt(v_hat);
    SITPST denom = ElementWiseInverse(sqrt_v_hat, params.epsilon);  // 1/(√v̂ + ε)
    
    for (size_t row = 0; row < current_state.rows(); ++row) {
      for (size_t col = 0; col < current_state.cols(); ++col) {
        for (size_t i = 0; i < current_state({row, col}).size(); ++i) {
          Tensor update = ElementWiseMultiply(denom({row, col})[i], m_hat({row, col})[i]) * step_length;
          updated_state({row, col})[i] += (-update);
        }
      }
    }
  }
  
  // CRITICAL MPI DESIGN: Do NOT broadcast here!
  // Energy evaluator owns all state distribution for Monte Carlo sampling.
  return updated_state;
}

// Element-wise helpers are now defined on SplitIndexTPS; Optimizer no longer owns them.



template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::LogOptimizationStep(size_t iteration,
                                                   double energy,
                                                   double energy_error,
                                                   double gradient_norm,
                                                   double step_length,
                                                   const std::vector<double> &accept_rates,
                                                   size_t sr_iterations,
                                                   double sr_natural_grad_norm,
                                                   double energy_eval_time,
                                                   double update_time) {
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    std::cout << "Iter " << std::setw(4) << iteration;
    if (step_length > 0.0) {
      std::cout << "LR = " << std::setw(9) << std::scientific << std::setprecision(1) << step_length;
    }
    std::cout << "E0 = " << std::setw(14) << std::fixed << std::setprecision(6) << energy;
    if (energy_error > 0.0) {
      std::cout << " ± " << std::setw(10) << std::scientific << std::setprecision(2) << energy_error;
    }
    std::cout << "Grad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << gradient_norm;

    // Print acceptance rates if available
    if (!accept_rates.empty()) {
      std::cout << "Accept rate = [";
      for (double rate : accept_rates) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(2) << rate;
      }
      std::cout << "]";
    }

    // Print SR information if available
    if (params_.IsAlgorithm<StochasticReconfigurationParams>() && sr_iterations > 0) {
      std::cout << "SRSolver Iter = " << std::setw(4) << sr_iterations;
      std::cout << "NGrad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << sr_natural_grad_norm;
    }

    // Print timing information
    double total_time = energy_eval_time + update_time;
    std::cout << " EvalT = " << std::setw(8) << std::fixed << std::setprecision(2) << energy_eval_time << "s";
    std::cout << " UpdateT = " << std::setw(8) << std::fixed << std::setprecision(2) << update_time << "s";
    std::cout << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << total_time << "s";

    std::cout << std::endl;
  }
}

template<typename TenElemT, typename QNT>
bool Optimizer<TenElemT, QNT>::ShouldStop(double current_energy, double previous_energy,
                                          double gradient_norm, size_t iterations_without_improvement) {
  // EFFICIENT APPROACH: Only master rank evaluates stopping criteria, then broadcasts decision
  bool should_stop = false;
  
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // Check gradient norm convergence
    if (gradient_norm < params_.base_params.gradient_tolerance) {
      std::cout << "Stopping: Gradient norm converged (" << std::scientific << std::setprecision(1) << gradient_norm
                << " < " << std::scientific << std::setprecision(1) << params_.base_params.gradient_tolerance
                << ")" << std::endl;
      should_stop = true;
    }

    // Check energy convergence
    if (std::abs(current_energy - previous_energy) < params_.base_params.energy_tolerance) {
      std::cout << "Stopping: Energy converged (|ΔE| = " << std::scientific << std::setprecision(1)
                << std::abs(current_energy - previous_energy)
                << " < " << std::scientific << std::setprecision(1) << params_.base_params.energy_tolerance
                << ")" << std::endl;
      should_stop = true;
    }

    // Check plateau detection
    if (iterations_without_improvement >= params_.base_params.plateau_patience) {
      std::cout << "Stopping: No improvement for " << params_.base_params.plateau_patience
                << " iterations (plateau detected)" << std::endl;
      should_stop = true;
    }
  }

  // Single-rank fast path: avoid calling MPI collectives when mpi_size_ == 1.
  // This keeps Optimizer usable in non-MPI test binaries and simplifies debugging.
  if (mpi_size_ == 1) return should_stop;

  // Broadcast the stopping decision to all ranks
  int stop_flag = should_stop ? 1 : 0;
  HANDLE_MPI_ERROR(::MPI_Bcast(&stop_flag, 1, MPI_INT, qlten::hp_numeric::kMPIMasterRank, comm_));
  
  return stop_flag == 1;
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::ClearUp() {
  // Clear AdaGrad state
  if (adagrad_initialized_) {
    accumulated_gradients_ = SITPST();
    adagrad_initialized_ = false;
  }

  // Clear SGD momentum state
  if (sgd_momentum_initialized_) {
    velocity_ = SITPST();
    sgd_momentum_initialized_ = false;
  }

  // Clear Adam state
  if (adam_initialized_) {
    first_moment_ = SITPST();
    second_moment_ = SITPST();
    adam_timestep_ = 0;
    adam_initialized_ = false;
  }

  // Clear spike detection state
  ema_energy_.Reset();
  ema_error_.Reset();
  ema_grad_norm_.Reset();
  ema_ngrad_norm_.Reset();
  prev_energy_ = 0.0;

  // Write CSV trigger log if configured
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    const auto &csv_path = params_.spike_recovery_params.log_trigger_csv_path;
    if (!csv_path.empty() && !spike_stats_.event_log.empty()) {
      std::ofstream csv(csv_path);
      if (csv) {
        csv << "step,attempt,signal,action,value,threshold\n";
        for (const auto &ev : spike_stats_.event_log) {
          csv << ev.step << "," << ev.attempt << ","
              << SignalName(ev.signal) << "," << ActionName(ev.action) << ","
              << std::scientific << std::setprecision(6) << ev.value << ","
              << ev.threshold << "\n";
        }
      }
    }
  }

  // Clear spike statistics (after CSV has been written above)
  spike_stats_ = SpikeStatistics{};

  // Clear rollback state
  prev_accepted_state_ = WaveFunctionT();
  has_prev_state_ = false;
}

// =============================================================================
// Spike detection methods
// =============================================================================

template<typename TenElemT, typename QNT>
SpikeSignal Optimizer<TenElemT, QNT>::DetectS1S2_(double error, double grad_norm) const {
  const auto &cfg = params_.spike_recovery_params;
  // S1: error bar spike
  if (ema_error_.IsInitialized() && ema_error_.Mean() > 0.0) {
    if (error > cfg.factor_err * ema_error_.Mean()) {
      return SpikeSignal::kS1_ErrorbarSpike;
    }
  }
  // S2: gradient norm spike
  if (ema_grad_norm_.IsInitialized() && ema_grad_norm_.Mean() > 0.0) {
    if (grad_norm > cfg.factor_grad * ema_grad_norm_.Mean()) {
      return SpikeSignal::kS2_GradientNormSpike;
    }
  }
  return SpikeSignal::kNone;
}

template<typename TenElemT, typename QNT>
SpikeSignal Optimizer<TenElemT, QNT>::DetectS3_(double ngrad_norm, size_t sr_iters) const {
  const auto &cfg = params_.spike_recovery_params;
  // Check natural gradient norm anomaly against EMA
  if (ema_ngrad_norm_.IsInitialized() && ema_ngrad_norm_.Mean() > 0.0) {
    if (ngrad_norm > cfg.factor_ngrad * ema_ngrad_norm_.Mean()) {
      return SpikeSignal::kS3_NaturalGradientAnomaly;
    }
  }
  // Suspiciously few CG iterations — independent trigger (no ngrad threshold).
  // When CG converges in very few iterations the solver may have "succeeded"
  // trivially on a near-singular system, so this alone signals trouble.
  if (sr_iters <= cfg.sr_min_iters_suspicious) {
    return SpikeSignal::kS3_NaturalGradientAnomaly;
  }
  return SpikeSignal::kNone;
}

template<typename TenElemT, typename QNT>
SpikeSignal Optimizer<TenElemT, QNT>::DetectS4_(double energy) const {
  const auto &cfg = params_.spike_recovery_params;
  if (!ema_energy_.IsInitialized()) return SpikeSignal::kNone;
  // Only trigger on upward spikes: energy increased AND exceeds EMA + k*sigma
  double delta = energy - ema_energy_.Mean();
  if (delta > 0.0 && ema_energy_.Std() > 0.0) {
    if (delta > cfg.sigma_k * ema_energy_.Std()) {
      return SpikeSignal::kS4_EMAEnergySpikeUpward;
    }
  }
  return SpikeSignal::kNone;
}

template<typename TenElemT, typename QNT>
SpikeAction Optimizer<TenElemT, QNT>::DecideAction_(SpikeSignal signal, size_t attempts, size_t step) const {
  if (signal == SpikeSignal::kNone) return SpikeAction::kAccept;

  const auto &cfg = params_.spike_recovery_params;

  // For S4, always rollback (caller decides to invoke this)
  if (signal == SpikeSignal::kS4_EMAEnergySpikeUpward) {
    return SpikeAction::kRollback;
  }

  // S1/S2/S3: try resample if under retry limit
  if (attempts < cfg.redo_mc_max_retries) {
    return SpikeAction::kResample;
  }

  // Retries exhausted: try rollback if enabled and we have a prev state
  if (cfg.enable_rollback && has_prev_state_ && step > 0) {
    return SpikeAction::kRollback;
  }

  // Last resort: accept with warning
  return SpikeAction::kAcceptWithWarning;
}

template<typename TenElemT, typename QNT>
SpikeAction Optimizer<TenElemT, QNT>::BroadcastAction_(SpikeAction action) {
  if (mpi_size_ == 1) return action;

  int action_int = static_cast<int>(action);
  HANDLE_MPI_ERROR(::MPI_Bcast(&action_int, 1, MPI_INT, qlten::hp_numeric::kMPIMasterRank, comm_));
  return static_cast<SpikeAction>(action_int);
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::LogSpikeEvent_(
    size_t step, size_t attempt, SpikeSignal signal,
    SpikeAction action, double value, double threshold) {
  // Console log (master rank only — caller guarantees this)
  std::cout << "[SPIKE] Step " << step << " attempt " << attempt
            << " | " << SignalName(signal)
            << " | value=" << std::scientific << std::setprecision(3) << value
            << " threshold=" << std::scientific << std::setprecision(3) << threshold
            << " | -> " << ActionName(action) << std::endl;

  // Update statistics
  if (action == SpikeAction::kResample) spike_stats_.total_resamples++;
  else if (action == SpikeAction::kRollback) spike_stats_.total_rollbacks++;
  else if (action == SpikeAction::kAcceptWithWarning) spike_stats_.total_forced_accepts++;

  // Record event for optional CSV output
  spike_stats_.event_log.push_back({step, attempt, signal, action, value, threshold});
}

// =============================================================================
// Checkpoint
// =============================================================================

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::SaveCheckpoint_(
    const WaveFunctionT &state, size_t step,
    const std::vector<TenElemT> &energy_traj,
    const std::vector<double> &error_traj) {
  const auto &cfg = params_.checkpoint_params;
  if (!cfg.IsEnabled()) return;

  std::string dir = cfg.base_path + "/step_" + std::to_string(step);

  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    EnsureDirectoryExists(dir);
    // Save TPS (Dump is non-const due to optional release_mem; const_cast is safe here)
    const_cast<WaveFunctionT &>(state).Dump(dir);

    // Write metadata
    std::string meta_path = dir + "/checkpoint_meta.txt";
    std::ofstream meta(meta_path);
    if (meta) {
      meta << "step " << step << "\n";
      if (!energy_traj.empty()) {
        meta << "energy " << std::setprecision(17) << std::real(energy_traj.back()) << "\n";
      }
      if (!error_traj.empty()) {
        meta << "error " << std::setprecision(17) << error_traj.back() << "\n";
      }
    }

    // Snapshot trajectory CSV
    std::string csv_path = dir + "/trajectory_snapshot.csv";
    std::ofstream csv(csv_path);
    if (csv) {
      csv << "iteration,energy,energy_error\n";
      for (size_t i = 0; i < energy_traj.size(); ++i) {
        csv << i << "," << std::setprecision(17) << std::real(energy_traj[i]) << ","
            << (i < error_traj.size() ? error_traj[i] : 0.0) << "\n";
      }
    }

    std::cout << "[CHECKPOINT] Saved step " << step << " to " << dir << std::endl;
  }

  // All ranks wait for master I/O to complete
  if (mpi_size_ > 1) {
    HANDLE_MPI_ERROR(::MPI_Barrier(comm_));
  }
}

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H 