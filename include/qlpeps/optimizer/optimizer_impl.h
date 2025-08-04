// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Implementation for optimizer.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H

#include <iomanip>
#include <algorithm>
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/utility/helpers.h"
#include "qlpeps/monte_carlo_tools/statistics.h"
#include "qlten/utility/timer.h"

namespace qlpeps {
using namespace qlten;

template<typename TenElemT, typename QNT>
Optimizer<TenElemT, QNT>::Optimizer(const OptimizerParams &params,
                                    const MPI_Comm &comm,
                                    int rank,
                                    int mpi_size)
    : params_(params), comm_(comm), rank_(rank), mpi_size_(mpi_size),
      random_engine_(std::random_device{}()),
      uniform_dist_(0.0, 1.0),
      adagrad_initialized_(false) {
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
    result.gradient_norms.push_back(initial_gradient.NormSquare());
  }

  // Determine search direction based on optimization scheme
  SITPST search_direction;
  size_t cg_iterations = 0;
  double natural_grad_norm = 0.0;

  switch (params_.update_scheme) {
    case GradientLineSearch:search_direction = initial_gradient;
      break;

    case NaturalGradientLineSearch: {
      // For natural gradient, we need to calculate it
      // This is a simplified version - in practice, you'd need the gradient samples
      SITPST init_guess(initial_state.rows(), initial_state.cols(), initial_state.PhysicalDim());
      auto [natural_grad, cg_iters] = CalculateNaturalGradient(
          initial_gradient, {}, {}, init_guess);
      search_direction = natural_grad;
      cg_iterations = cg_iters;
      natural_grad_norm = natural_grad.NormSquare();
      break;
    }

    default:throw std::runtime_error("Invalid optimization scheme for line search");
  }

  // Perform line search along the search direction
  SITPST best_state = initial_state;
  double best_energy = Real(initial_energy);
  double cumulative_step = 0.0;

  for (size_t i = 0; i < params_.core_params.step_lengths.size(); ++i) {
    double step_length = params_.core_params.step_lengths[i];
    cumulative_step += step_length;

    // Start timer for update step
    Timer update_timer("line_search_update");

    // Update state along search direction
    SITPST test_state = UpdateTPSByGradient(initial_state, search_direction, cumulative_step);

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
      result.gradient_norms.push_back(test_gradient.NormSquare());
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
                        test_gradient.NormSquare(),
                        cumulative_step,
                        {},
                        0,
                        0.0,
                        energy_eval_time,
                        update_time);

    if (callback.on_iteration) {
      callback.on_iteration(i, Real(test_energy), energy_error, test_gradient.NormSquare());
    }

    if (callback.should_stop && callback.should_stop(i, Real(test_energy), energy_error)) {
      break;
    }
  }

  result.optimized_state = best_state;
  result.final_energy = best_energy;
  result.total_iterations = params_.core_params.step_lengths.size();
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
    result.gradient_norms.push_back(initial_gradient.NormSquare());
  }

  // Determine search direction based on optimization scheme
  SITPST search_direction;
  size_t cg_iterations = 0;
  double natural_grad_norm = 0.0;

  switch (params_.update_scheme) {
    case GradientLineSearch:search_direction = initial_gradient;
      break;

    case NaturalGradientLineSearch: {
      // For natural gradient, we need to calculate it
      // This is a simplified version - in practice, you'd need the gradient samples
      SITPST init_guess(initial_state.rows(), initial_state.cols(), initial_state.PhysicalDim());
      auto [natural_grad, cg_iters] = CalculateNaturalGradient(
          initial_gradient, {}, {}, init_guess);
      search_direction = natural_grad;
      cg_iterations = cg_iters;
      natural_grad_norm = natural_grad.NormSquare();
      break;
    }

    default:throw std::runtime_error("Invalid optimization scheme for line search");
  }

  // Perform line search along the search direction
  SITPST best_state = initial_state;
  double best_energy = Real(initial_energy);
  double cumulative_step = 0.0;

  for (size_t i = 0; i < params_.core_params.step_lengths.size(); ++i) {
    double step_length = params_.core_params.step_lengths[i];
    cumulative_step += step_length;

    // Start timer for update step
    Timer update_timer("line_search_update");

    // Update state along search direction
    SITPST test_state = UpdateTPSByGradient(initial_state, search_direction, cumulative_step);

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
      result.gradient_norms.push_back(test_gradient.NormSquare());
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
                        test_gradient.NormSquare(),
                        cumulative_step,
                        {},
                        0,
                        0.0,
                        energy_eval_time,
                        update_time);

    if (callback.on_iteration) {
      callback.on_iteration(i, Real(test_energy), test_error, test_gradient.NormSquare());
    }

    if (callback.should_stop && callback.should_stop(i, Real(test_energy), test_error)) {
      break;
    }
  }

  result.optimized_state = best_state;
  result.final_energy = best_energy;
  result.total_iterations = params_.core_params.step_lengths.size();
  result.converged = true;

  // Clean up optimization state
  ClearUp();

  return result;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::OptimizationResult
Optimizer<TenElemT, QNT>::IterativeOptimize(
    const SITPST &initial_state,
    std::function<std::tuple<TenElemT, SITPST, double>(const SITPST &)> energy_evaluator,
    const OptimizationCallback &callback,
    const std::vector<SITPST> *gten_samples,
    const SITPST *gten_average) {

  OptimizationResult result;
  result.optimized_state = initial_state;
  result.converged = false;
  result.total_iterations = 0;

  SITPST current_state = initial_state;
  SITPST best_state = initial_state;
  double best_energy = std::numeric_limits<double>::max();

  // Initialize for stochastic reconfiguration if needed
  SITPST sr_init_guess;
  if (UsesStochasticReconfiguration(params_.update_scheme)) {
    sr_init_guess = SITPST(initial_state.rows(), initial_state.cols(), initial_state.PhysicalDim());
  }

  // Advanced stopping criteria tracking
  double previous_energy = std::numeric_limits<double>::max();
  size_t iterations_without_improvement = 0;  // plateau counter

  size_t total_iterations_performed = 0;
  for (size_t iter = 0; iter < params_.core_params.max_iterations; ++iter) {
    // For AdaGrad, use uniform learning rate; for other methods use step_lengths with additional steps repeat use the last step length
    double step_length;
    if (params_.update_scheme == AdaGrad) {
      step_length = params_.core_params.step_lengths.front();
    } else {
      step_length =
          params_.core_params.step_lengths[iter < params_.core_params.step_lengths.size() ? iter :
                                           params_.core_params.step_lengths.size() - 1];
    }

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
      result.gradient_norms.push_back(current_gradient.NormSquare());
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

    // Advanced stopping criteria checks (skip for first iteration)
    if (iter > 0) {
      double current_energy_real = Real(current_energy);
      double gradient_norm = current_gradient.NormSquare();

      // EFFICIENT APPROACH: Only master rank evaluates stopping criteria, and broadcast the decision.
      bool should_stop = ShouldStop(current_energy_real, previous_energy, gradient_norm, iterations_without_improvement);
      
      if (should_stop) {
        // Log the final iteration before stopping (only on master rank)
        if (rank_ == kMPIMasterRank) {
          LogOptimizationStep(iter,
                             current_energy_real,
                             current_error,
                             gradient_norm,
                             step_length,
                             {},
                             0, // sr_iterations
                             0.0, // sr_natural_grad_norm
                             energy_eval_time,
                             0.0); // update_time
        result.optimized_state = best_state;
        result.final_energy = best_energy;
        result.total_iterations = total_iterations_performed;
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

    // Apply optimization update based on scheme
    SITPST updated_state;
    size_t sr_iterations = 0;
    double sr_natural_grad_norm = 0.0;

    switch (params_.update_scheme) {
      case StochasticGradient:updated_state = UpdateTPSByGradient(current_state, current_gradient, step_length);
        break;

      case RandomStepStochasticGradient:step_length *= uniform_dist_(random_engine_);
        updated_state = UpdateTPSByGradient(current_state, current_gradient, step_length);
        break;

      case StochasticReconfiguration: {
        auto [new_state, ng_norm, sr_iters] = StochasticReconfigurationUpdate(
            current_state, current_gradient,
            gten_samples ? *gten_samples : std::vector<SITPST>{},
            gten_average ? *gten_average : SITPST{},
            step_length, sr_init_guess, false);
        updated_state = new_state;
        sr_iterations = sr_iters;
        sr_natural_grad_norm = ng_norm;
        sr_init_guess = new_state; // Use as initial guess for next iteration
        break;
      }

      case RandomStepStochasticReconfiguration: {
        step_length *= uniform_dist_(random_engine_);
        auto [new_state, ng_norm, sr_iters] = StochasticReconfigurationUpdate(
            current_state, current_gradient,
            gten_samples ? *gten_samples : std::vector<SITPST>{},
            gten_average ? *gten_average : SITPST{},
            step_length, sr_init_guess, false);
        updated_state = new_state;
        sr_iterations = sr_iters;
        sr_natural_grad_norm = ng_norm;
        sr_init_guess = new_state;
        break;
      }

      case NormalizedStochasticReconfiguration: {
        auto [new_state, ng_norm, sr_iters] = StochasticReconfigurationUpdate(
            current_state, current_gradient,
            gten_samples ? *gten_samples : std::vector<SITPST>{},
            gten_average ? *gten_average : SITPST{},
            step_length, sr_init_guess, true);
        updated_state = new_state;
        sr_iterations = sr_iters;
        sr_natural_grad_norm = ng_norm;
        sr_init_guess = new_state;
        break;
      }

      case RandomGradientElement: {
        SITPST random_gradient = current_gradient;
        // Apply random signs to gradient elements
        for (size_t row = 0; row < random_gradient.rows(); ++row) {
          for (size_t col = 0; col < random_gradient.cols(); ++col) {
            for (size_t i = 0; i < random_gradient({row, col}).size(); ++i) {
              if (uniform_dist_(random_engine_) < 0.5) {
                random_gradient({row, col})[i] *= -1.0;
              }
            }
          }
        }
        updated_state = UpdateTPSByGradient(current_state, random_gradient, step_length);
        break;
      }

      case BoundGradientElement:updated_state = BoundedGradientUpdate(current_state, current_gradient, step_length);
        break;

      case AdaGrad:updated_state = AdaGradUpdate(current_state, current_gradient, step_length);
        break;

      default:throw std::runtime_error("Unsupported optimization scheme for iterative optimization");
    }

    current_state = updated_state;

    // Get elapsed time for optimization update step
    double update_time = update_timer.Elapsed();

    // Calculate total iteration time
    double total_iteration_time = energy_eval_time + update_time;

    // Log progress
    LogOptimizationStep(iter,
                        Real(current_energy),
                        current_error,
                        current_gradient.NormSquare(),
                        step_length,
                        {},
                        sr_iterations,
                        sr_natural_grad_norm,
                        energy_eval_time,
                        update_time);

    if (callback.on_iteration) {
      callback.on_iteration(iter, Real(current_energy), current_error, current_gradient.NormSquare());
    }

    if (callback.should_stop && callback.should_stop(iter, Real(current_energy), current_error)) {
      break;
    }
  }

  result.optimized_state = best_state;
  result.final_energy = best_energy;
  result.total_iterations = total_iterations_performed;
  result.converged = false;

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

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::UpdateTPSByGradient(const SITPST &current_state,
                                              const SITPST &gradient,
                                              double step_length) {
  SITPST updated_state = current_state;

  if (rank_ == kMPIMasterRank) {
    // Apply gradient update only on master rank
    // This matches the behavior of the original VMCPEPSExecutor
    updated_state += (-step_length) * gradient;
  }

  // Broadcast the updated state to all ranks
  // This ensures all ranks have the same state for the next iteration
  return BroadcastState(updated_state);
}

template<typename TenElemT, typename QNT>
std::pair<typename Optimizer<TenElemT, QNT>::SITPST, size_t>
Optimizer<TenElemT, QNT>::CalculateNaturalGradient(
    const SITPST &gradient,
    const std::vector<SITPST> &gten_samples,
    const SITPST &gten_average,
    const SITPST &init_guess) {

  const ConjugateGradientParams &cg_params = params_.cg_params;

  // Create S-matrix for stochastic reconfiguration
  SITPST *pgten_average = nullptr;
  if (rank_ == kMPIMasterRank) {
    pgten_average = const_cast<SITPST *>(&gten_average);
  }

  SRSMatrix s_matrix(const_cast<std::vector<SITPST> *>(&gten_samples), pgten_average, mpi_size_);
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
    const std::vector<SITPST> &gten_samples,
    const SITPST &gten_average,
    double step_length,
    const SITPST &init_guess,
    bool normalize) {

  // Calculate natural gradient using stochastic reconfiguration
  // This involves solving the SR equation which should be done by all cores together
  auto [natural_gradient, cg_iterations] = CalculateNaturalGradient(
      gradient, gten_samples, gten_average, init_guess);

  double natural_grad_norm = natural_gradient.NormSquare();

  if (normalize) {
    step_length /= std::sqrt(natural_grad_norm);
  }

  // Apply the update using the natural gradient
  // The update is done on master rank and broadcast to all ranks
  SITPST updated_state = UpdateTPSByGradient(current_state, natural_gradient, step_length);

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

  // Broadcast the updated state to all ranks
  return BroadcastState(updated_state);
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

  return UpdateTPSByGradient(current_state, random_gradient, step_length);
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::AdaGradUpdate(const SITPST &current_state,
                                        const SITPST &gradient,
                                        double step_length) {
  // Initialize AdaGrad state if not already done
  if (!adagrad_initialized_) {
    accumulated_gradients_ = SITPST(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
    // Initialize with small values to avoid division by zero
    for (size_t row = 0; row < accumulated_gradients_.rows(); ++row) {
      for (size_t col = 0; col < accumulated_gradients_.cols(); ++col) {
        for (size_t i = 0; i < accumulated_gradients_({row, col}).size(); ++i) {
          accumulated_gradients_({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
          QNT div = gradient({row, col})[i].Div();
          accumulated_gradients_({row, col})[i].Fill(div, params_.GetAdaGradParams().initial_accumulator_value);
          // Note here for fermionic tensor, positive and negative are relatively define, 
          // In the AdaGrad, we fix the order of indexes to make it make sense.
        }
      }
    }
    adagrad_initialized_ = true;
  }

  // Update accumulated gradients: G_k = G_{k-1} + (gradient)^2
  SITPST squared_gradient = ElementWiseSquare(gradient);
  accumulated_gradients_ += squared_gradient;

  // Compute adaptive learning rates: 1/sqrt(G_k) for |G_k| > epsilon
  SITPST
      adaptive_rates = ElementWiseInverse(ElementWiseSqrt(accumulated_gradients_), params_.GetAdaGradParams().epsilon);

  // Apply AdaGrad update: θ_{k+1} = θ_k - η * adaptive_rates * gradient
  SITPST updated_state = current_state;

  if (rank_ == kMPIMasterRank) {
    // Apply AdaGrad update only on master rank
    // This matches the behavior of the original VMCPEPSExecutor
    for (size_t row = 0; row < current_state.rows(); ++row) {
      for (size_t col = 0; col < current_state.cols(); ++col) {
        for (size_t i = 0; i < current_state({row, col}).size(); ++i) {
          // Compute adaptive step: step_length * adaptive_rate * gradient
          Tensor
              adaptive_step = ElementWiseMultiply(adaptive_rates({row, col})[i], gradient({row, col})[i]) * step_length;

          // Update state: θ_{k+1} = θ_k - adaptive_step
          updated_state({row, col})[i] += (-adaptive_step);
        }
      }
    }
  }

  // Broadcast the updated state to all ranks
  return BroadcastState(updated_state);
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::ElementWiseSquare(const SITPST &tps) {
  SITPST result = tps;

  for (size_t row = 0; row < tps.rows(); ++row) {
    for (size_t col = 0; col < tps.cols(); ++col) {
      for (size_t i = 0; i < tps({row, col}).size(); ++i) {
        if (!result({row, col})[i].IsDefault()) {
          result({row, col})[i].ElementWiseSquare();
        }
      }
    }
  }

  return result;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::ElementWiseSqrt(const SITPST &tps) {
  SITPST result = tps;

  for (size_t row = 0; row < result.rows(); ++row) {
    for (size_t col = 0; col < result.cols(); ++col) {
      for (size_t i = 0; i < result({row, col}).size(); ++i) {
        if (!result({row, col})[i].IsDefault()) {
          result({row, col})[i].ElementWiseSqrt();
        }
      }
    }
  }

  return result;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::ElementWiseInverse(const SITPST &tps, double epsilon) {
  SITPST result = tps;

  for (size_t row = 0; row < tps.rows(); ++row) {
    for (size_t col = 0; col < tps.cols(); ++col) {
      for (size_t i = 0; i < tps({row, col}).size(); ++i) {
        if (!tps({row, col})[i].IsDefault()) {
          result({row, col})[i].ElementWiseInv(epsilon);
        }
      }
    }
  }

  return result;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::SITPST
Optimizer<TenElemT, QNT>::BroadcastState(const SITPST &state) {
  SITPST broadcasted_state = state;
  BroadCast(broadcasted_state, comm_);
  return broadcasted_state;
}

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
      std::cout << "Alpha = " << std::setw(9) << std::scientific << std::setprecision(1) << step_length;
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
    if (UsesStochasticReconfiguration(params_.update_scheme) && sr_iterations > 0) {
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
    if (gradient_norm < params_.core_params.gradient_tolerance) {
      std::cout << "Stopping: Gradient norm converged (" << std::scientific << std::setprecision(1) << gradient_norm
                << " < " << std::scientific << std::setprecision(1) << params_.core_params.gradient_tolerance
                << ")" << std::endl;
      should_stop = true;
    }

    // Check energy convergence
    if (std::abs(current_energy - previous_energy) < params_.core_params.energy_tolerance) {
      std::cout << "Stopping: Energy converged (|ΔE| = " << std::scientific << std::setprecision(1)
                << std::abs(current_energy - previous_energy)
                << " < " << std::scientific << std::setprecision(1) << params_.core_params.energy_tolerance
                << ")" << std::endl;
      should_stop = true;
    }

    // Check plateau detection
    if (iterations_without_improvement >= params_.core_params.plateau_patience) {
      std::cout << "Stopping: No improvement for " << params_.core_params.plateau_patience
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