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
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/utility/helpers.h"
#include "qlten/utility/timer.h"

namespace qlpeps {
using qlten::Timer;

template<typename TenElemT, typename QNT>
Optimizer<TenElemT, QNT>::Optimizer(const OptimizerParams &params,
                                    const MPI_Comm &comm,
                                    int rank,
                                    int mpi_size)
    : params_(params), comm_(comm), rank_(rank), mpi_size_(mpi_size),
      random_engine_(std::random_device{}()),
      uniform_dist_(0.0, 1.0),
      adagrad_initialized_(false),
      sgd_momentum_initialized_(false) {
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
  result.min_energy = Real(initial_energy);

  if (rank_ == kMPIMasterRank) {
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
  double best_energy = Real(initial_energy);
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
    if (rank_ == kMPIMasterRank) {
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
    if (rank_ == kMPIMasterRank) {
      // The energy error should be calculated by the energy evaluator
      // For now, we'll use a placeholder
      result.energy_error_trajectory.push_back(energy_error);
      result.gradient_norms.push_back(std::sqrt(test_gradient.NormSquare()));
    }

    // Update best state if energy improved
    if (Real(test_energy) < best_energy) {
      best_energy = Real(test_energy);
      best_state = test_state;
      result.min_energy = best_energy;

      if (callback.on_best_state_found) {
        callback.on_best_state_found(best_state, best_energy);
      }
    }

    // Log progress
    LogOptimizationStep(i,
                        Real(test_energy),
                        energy_error,
                        std::sqrt(test_gradient.NormSquare()),
                        cumulative_step,
                        {},
                        0,
                        0.0,
                        energy_eval_time,
                        update_time);

    if (callback.on_iteration) {
      callback.on_iteration(i, Real(test_energy), energy_error, std::sqrt(test_gradient.NormSquare()));
    }

    if (callback.should_stop && callback.should_stop(i, Real(test_energy), energy_error)) {
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
  result.min_energy = Real(initial_energy);

  if (rank_ == kMPIMasterRank) {
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
  double best_energy = Real(initial_energy);
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
    if (rank_ == kMPIMasterRank) {
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
    if (rank_ == kMPIMasterRank) {
      result.energy_error_trajectory.push_back(test_error);
      result.gradient_norms.push_back(std::sqrt(test_gradient.NormSquare()));
    }

    // Update best state if energy improved
    if (Real(test_energy) < best_energy) {
      best_energy = Real(test_energy);
      best_state = test_state;
      result.min_energy = best_energy;

      if (callback.on_best_state_found) {
        callback.on_best_state_found(best_state, best_energy);
      }
    }

    // Log progress
    LogOptimizationStep(i,
                        Real(test_energy),
                        test_error,
                        std::sqrt(test_gradient.NormSquare()),
                        cumulative_step,
                        {},
                        0,
                        0.0,
                        energy_eval_time,
                        update_time);

    if (callback.on_iteration) {
      callback.on_iteration(i, Real(test_energy), test_error, std::sqrt(test_gradient.NormSquare()));
    }

    if (callback.should_stop && callback.should_stop(i, Real(test_energy), test_error)) {
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

  size_t total_iterations_performed = 0;
  for (size_t iter = 0; iter < params_.base_params.max_iterations; ++iter) {
    // Start timer for energy evaluation (the dominant computational cost)
    Timer energy_eval_timer("energy_evaluation");
    
    // Evaluate energy and gradient at current state
    auto [current_energy, current_gradient, current_error] = energy_evaluator(current_state);
    result.energy_trajectory.push_back(current_energy);

    // Get elapsed time for energy evaluation
    double energy_eval_time = energy_eval_timer.Elapsed();

    // Store energy error and gradient norm only on master rank
    if (rank_ == kMPIMasterRank) {
      result.energy_error_trajectory.push_back(current_error);
      result.gradient_norms.push_back(std::sqrt(current_gradient.NormSquare()));
    }

    total_iterations_performed = iter + 1;

    // Update best state if energy improved
    bool energy_improved = false;
    if (Real(current_energy) < best_energy) {
      best_energy = Real(current_energy);
      best_state = current_state;
      result.min_energy = best_energy;
      energy_improved = true;
      iterations_without_improvement = 0;

      if (callback.on_best_state_found) {
        callback.on_best_state_found(best_state, best_energy);
      }
    } else {
      iterations_without_improvement++;
    }

    // Determine learning rate after we have energy info
    // Use previous iteration energy for scheduler when iter>0; otherwise use current energy
    double learning_rate = GetCurrentLearningRate(iter, (iter == 0) ? Real(current_energy) : previous_energy);

    // Advanced stopping criteria checks (skip for first iteration)
    if (iter > 0) {
      double current_energy_real = Real(current_energy);
      double gradient_norm = std::sqrt(current_gradient.NormSquare());

      // EFFICIENT APPROACH: Only master rank evaluates stopping criteria, then broadcasts the decision.
      bool should_stop = ShouldStop(current_energy_real, previous_energy, gradient_norm, iterations_without_improvement);
      
      if (should_stop) {
        // Log the final iteration before stopping (only on master rank)
        if (rank_ == kMPIMasterRank) {
          LogOptimizationStep(iter,
                             current_energy_real,
                             current_error,
                             gradient_norm,
                             learning_rate,
                             {},
                             0, // sr_iterations
                             0.0, // sr_natural_grad_norm
                             energy_eval_time,
                             0.0); // update_time
        result.converged = true;
        }
        
        break;  // All ranks will break together now
      }

      previous_energy = current_energy_real;
    } else {
      // For first iteration, just record the energy for next comparison
      previous_energy = Real(current_energy);
    }

    // Start timer for optimization update step
    Timer update_timer("optimization_update");

    // Apply optimization update based on algorithm type
    SITPST updated_state;
    size_t sr_iterations = 0;
    double sr_natural_grad_norm = 0.0;

    // Use std::visit for type-safe algorithm dispatch
    std::visit([&](const auto& algo_params) {
      using T = std::decay_t<decltype(algo_params)>;
      
      if constexpr (std::is_same_v<T, SGDParams>) {
        // SGD with momentum and Nesterov acceleration support
        updated_state = SGDUpdate(current_state, current_gradient, learning_rate, algo_params);
      }
      else if constexpr (std::is_same_v<T, StochasticReconfigurationParams>) {
        // Stochastic Reconfiguration variants
        auto [new_state, ng_norm, sr_iters] = StochasticReconfigurationUpdate(
            current_state, current_gradient,
            Ostar_samples ? *Ostar_samples : std::vector<SITPST>{},
            Ostar_mean ? *Ostar_mean : SITPST{},
            learning_rate, sr_init_guess, algo_params.normalize_update);
        updated_state = new_state;
        sr_iterations = sr_iters;
        sr_natural_grad_norm = ng_norm;
        sr_init_guess = new_state; // Use as initial guess for next iteration
      }
      else if constexpr (std::is_same_v<T, AdaGradParams>) {
        updated_state = AdaGradUpdate(current_state, current_gradient, learning_rate);
      }
      else if constexpr (std::is_same_v<T, AdamParams>) {
        // TODO: Implement Adam when needed
        throw std::runtime_error("Adam algorithm not yet implemented");
      }
      else if constexpr (std::is_same_v<T, LBFGSParams>) {
        // TODO: Implement L-BFGS when needed
        throw std::runtime_error("L-BFGS algorithm not yet implemented");
      }
      else {
        throw std::runtime_error("Unsupported algorithm type for iterative optimization");
      }
    }, params_.algorithm_params);

    current_state = updated_state;

    // Get elapsed time for optimization update step
    double update_time = update_timer.Elapsed();

    // Calculate total iteration time
    double total_iteration_time = energy_eval_time + update_time;

    // Log progress
    LogOptimizationStep(iter,
                        Real(current_energy),
                        current_error,
                        std::sqrt(current_gradient.NormSquare()),
                        learning_rate,
                        {},
                        sr_iterations,
                        sr_natural_grad_norm,
                        energy_eval_time,
                        update_time);

    if (callback.on_iteration) {
      callback.on_iteration(iter, Real(current_energy), current_error, std::sqrt(current_gradient.NormSquare()));
    }

    if (callback.should_stop && callback.should_stop(iter, Real(current_energy), current_error)) {
      break;
    }
  }

  result.optimized_state = best_state;
  result.final_energy = best_energy;
  result.total_iterations = total_iterations_performed;

  // Ensure the final state is valid
  if (rank_ == kMPIMasterRank) {
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
  if (rank_ == kMPIMasterRank) {
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
  if (rank_ == kMPIMasterRank) {
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
  if (rank_ == kMPIMasterRank) {
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

  if (rank_ == kMPIMasterRank) {
    // Apply bounded gradient update only on master rank
    // This matches the behavior of the original VMCPEPSExecutor
    for (size_t row = 0; row < current_state.rows(); ++row) {
      for (size_t col = 0; col < current_state.cols(); ++col) {
        const size_t phy_dim = gradient({row, col}).size();
        for (size_t compt = 0; compt < phy_dim; ++compt) {
          Tensor &grad_ten = const_cast<SITPST &>(gradient)({row, col})[compt];
          grad_ten.ElementWiseBoundTo(step_length);
          updated_state({row, col})[compt] += (-step_length) * grad_ten;
        }
      }
    }
  }

  // 🚫 CRITICAL MPI DESIGN: Do NOT broadcast here!
  // Energy evaluator owns all state distribution for Monte Carlo sampling.
  return updated_state;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::RandomGradientUpdate(const SITPST &current_state,
                                               const SITPST &gradient,
                                               double step_length) {
  SITPST random_gradient = gradient;

  if (rank_ == kMPIMasterRank) {
    for (size_t row = 0; row < random_gradient.rows(); ++row) {
      for (size_t col = 0; col < random_gradient.cols(); ++col) {
        for (size_t i = 0; i < random_gradient({row, col}).size(); ++i) {
          if (uniform_dist_(random_engine_) < 0.5) {
            random_gradient({row, col})[i] *= -1.0;
          }
        }
      }
    }
  }

  // Apply random gradient update
  SITPST updated_state = current_state;
  if (rank_ == kMPIMasterRank) {
    updated_state += (-step_length) * random_gradient;
  }
  return updated_state;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::AdaGradUpdate(const SITPST &current_state,
                                        const SITPST &gradient,
                                        double step_length) {
  // MPI VERIFICATION: Only master rank processes gradients and algorithm state
  SITPST updated_state = current_state;
  
  if (rank_ == kMPIMasterRank) {
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

    // Update accumulated gradients: G_k = G_{k-1} + (gradient)^2
    SITPST squared_gradient = ElementWiseSquare(gradient);
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
  if (rank_ == kMPIMasterRank) {
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
  
  if (rank_ == kMPIMasterRank) {
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

  // Broadcast the stopping decision to all ranks
  int stop_flag = should_stop ? 1 : 0;
  HANDLE_MPI_ERROR(::MPI_Bcast(&stop_flag, 1, MPI_INT, kMPIMasterRank, comm_));
  
  return stop_flag == 1;
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::ClearUp() {
  // Clear AdaGrad state
  if (adagrad_initialized_) {
    accumulated_gradients_ = SITPST(); // Clear the tensor
    adagrad_initialized_ = false;
  }
  
  // Clear SGD momentum state
  if (sgd_momentum_initialized_) {
    velocity_ = SITPST(); // Clear the velocity tensor
    sgd_momentum_initialized_ = false;
  }

  // Clear any other algorithm-specific state here
  // For example, if we add Adam in the future:
  // if (adam_initialized_) {
  //   first_moment_ = SITPST();
  //   second_moment_ = SITPST();
  //   adam_initialized_ = false;
  // }

  // Reset random engine state if needed
  // random_engine_.seed(std::random_device{}());

  // Clear any cached data or temporary storage
  // This ensures memory is freed and the optimizer is ready for the next run
}

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H 