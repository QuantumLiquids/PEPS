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
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "qlpeps/optimizer/stochastic_reconfiguration_smatrix.h"

namespace qlpeps {
using namespace qlten;

/**
 * @brief Optimizer for VMC PEPS that handles different optimization strategies
 * 
 * This class provides a unified interface for different optimization strategies.
 * It supports line search and iterative optimization, with or without energy error support.
 * 
 * The optimization strategies are:
 * 1. Gradient line search
 * 2. Natural gradient line search
 * 3. Stochastic reconfiguration
 * 4. Bounded gradient element update
 * 5. Random gradient element update
 * 6. AdaGrad (Adaptive Gradient)
 * 
 * The optimization strategies are selected based on the update scheme parameter.
 * 
 * The optimization result is returned as an OptimizationResult object, which contains the optimized state,
 * the final energy, the minimum energy, the energy trajectory, the energy error trajectory, the gradient norm trajectory,
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
   * @param gten_samples Optional gradient samples for stochastic reconfiguration
   * @param gten_average Optional average gradient tensor for stochastic reconfiguration
   * @return Optimization result
   */
  OptimizationResult IterativeOptimize(
      const WaveFunctionT& initial_state,
      std::function<std::tuple<TenElemT, WaveFunctionT, double>(const WaveFunctionT&)> energy_evaluator,
      const OptimizationCallback& callback = OptimizationCallback{},
      const std::vector<WaveFunctionT>* gten_samples = nullptr,
      const WaveFunctionT* gten_average = nullptr);

  /**
   * @brief Update TPS using gradient descent
   * 
   * @param current_state Current TPS state
   * @param gradient Gradient direction
   * @param step_length Step length for update
   * @return Updated TPS state
   */
  WaveFunctionT UpdateTPSByGradient(const WaveFunctionT& current_state, 
                             const WaveFunctionT& gradient, 
                             double step_length);

  /**
   * @brief Calculate natural gradient using stochastic reconfiguration
   * 
   * @param gradient Standard gradient
   * @param gten_samples Gradient tensor samples
   * @param gten_average Average gradient tensor
   * @param init_guess Initial guess for conjugate gradient solver
   * @return Natural gradient and number of CG iterations
   */
  std::pair<WaveFunctionT, size_t> CalculateNaturalGradient(
      const WaveFunctionT& gradient,
      const std::vector<WaveFunctionT>& gten_samples,
      const WaveFunctionT& gten_average,
      const WaveFunctionT& init_guess);

  /**
   * @brief Apply stochastic reconfiguration update
   * 
   * @param current_state Current TPS state
   * @param gradient Standard gradient
   * @param gten_samples Gradient tensor samples
   * @param gten_average Average gradient tensor
   * @param step_length Step length
   * @param init_guess Initial guess for CG solver
   * @param normalize Whether to normalize the natural gradient
   * @return Updated state, natural gradient norm, and CG iterations
   */
  std::tuple<WaveFunctionT, double, size_t> StochasticReconfigurationUpdate(
      const WaveFunctionT& current_state,
      const WaveFunctionT& gradient,
      const std::vector<WaveFunctionT>& gten_samples,
      const WaveFunctionT& gten_average,
      double step_length,
      const WaveFunctionT& init_guess,
      bool normalize = false);

  /**
   * @brief Apply bounded gradient element update
   * 
   * @param current_state Current TPS state
   * @param gradient Gradient
   * @param step_length Step length
   * @return Updated TPS state
   */
  WaveFunctionT BoundedGradientUpdate(const WaveFunctionT& current_state,
                               const WaveFunctionT& gradient,
                               double step_length);

  /**
   * @brief Apply random gradient element update
   * 
   * @param current_state Current TPS state
   * @param gradient Gradient
   * @param step_length Step length
   * @return Updated TPS state
   */
  WaveFunctionT RandomGradientUpdate(const WaveFunctionT& current_state,
                              const WaveFunctionT& gradient,
                              double step_length);

  /**
   * @brief Apply AdaGrad update
   * 
   * @param current_state Current TPS state
   * @param gradient Gradient
   * @param step_length Step length (learning rate)
   * @param epsilon Small constant for numerical stability
   * @return Updated TPS state
   */
  WaveFunctionT AdaGradUpdate(const WaveFunctionT& current_state,
                       const WaveFunctionT& gradient,
                       double step_length);

  /**
   * @brief Get current learning rate based on iteration and scheduler
   * 
   * @param iteration Current optimization iteration
   * @param current_energy Current energy value (for energy-aware schedulers)
   * @return Learning rate for this iteration
   */
  double GetCurrentLearningRate(size_t iteration, double current_energy = 0.0) const;

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
  OptimizerParams params_;
  MPI_Comm comm_;
  int rank_;
  int mpi_size_;
  
  std::mt19937 random_engine_;
  std::uniform_real_distribution<double> uniform_dist_;

  // AdaGrad state
  WaveFunctionT accumulated_gradients_;
  bool adagrad_initialized_;

  // Helper methods
  WaveFunctionT BroadcastState(const WaveFunctionT& state);
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

  // Tensor operations for AdaGrad
  WaveFunctionT ElementWiseSquare(const WaveFunctionT& tensor);
  WaveFunctionT ElementWiseSqrt(const WaveFunctionT& tensor);
  WaveFunctionT ElementWiseInverse(const WaveFunctionT& tensor, double epsilon = 1e-8);
  
 };



} // namespace qlpeps

#include "qlpeps/optimizer/optimizer_impl.h"

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_H 