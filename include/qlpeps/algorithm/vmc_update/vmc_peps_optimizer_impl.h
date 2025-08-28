// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Implementation for VMC PEPS optimizer executor.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_IMPL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_IMPL_H

#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/consts.h"  // For kTpsPath constant
#include "qlpeps/utility/helpers.h"
#include "qlpeps/utility/filesystem_utils.h"
#include "qlpeps/vmc_basic/monte_carlo_tools/statistics.h"
#include "qlpeps/vmc_basic/statistics_tensor.h"
#include "qlten/utility/timer.h"
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "qlpeps/optimizer/stochastic_reconfiguration_smatrix.h"

namespace qlpeps {
using qlten::Timer;

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::VMCPEPSOptimizer(
    const VMCPEPSOptimizerParams &params,
    const SITPST &sitpst_init,
    const MPI_Comm &comm,
    const EnergySolver &solver)
    : qlten::Executor(),
      monte_carlo_engine_(sitpst_init, params.mc_params, params.peps_params, comm),
      params_(params),
      energy_solver_(solver),
      optimizer_(params.optimizer_params, comm, monte_carlo_engine_.Rank(), monte_carlo_engine_.MpiSize()),
      grad_(monte_carlo_engine_.Ly(), monte_carlo_engine_.Lx()),
      en_min_(std::numeric_limits<double>::max()),
      tps_lowest_(monte_carlo_engine_.State()),
      current_energy_error_(0.0) {

  // Check if using stochastic reconfiguration algorithm
  stochastic_reconfiguration_update_class_ = params.optimizer_params.IsAlgorithm<StochasticReconfigurationParams>();

  // Create persistent evaluator to reuse internal buffers; SR buffers toggled by algorithm type
  energy_grad_evaluator_ = std::make_unique<MCEnergyGradEvaluator<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>>(
      monte_carlo_engine_, energy_solver_, monte_carlo_engine_.Comm(), stochastic_reconfiguration_update_class_);

  // Ensure necessary directories exist for output
  if (!params_.tps_dump_base_name.empty()) {
    monte_carlo_engine_.EnsureDirectoryExists(params_.tps_dump_base_name + "final");
  }
  ReserveSamplesDataSpace_();
  // Hint evaluator for buffer reservation (coarse-grained)
  if (energy_grad_evaluator_) {
    energy_grad_evaluator_->ReserveBuffers(monte_carlo_engine_.Ly(), monte_carlo_engine_.Lx(), params_.mc_params.num_samples);
  }
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
std::unique_ptr<VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>>
VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::CreateByLoadingTPS(
    const VMCPEPSOptimizerParams &params,
    const std::string &tps_path,
    const MPI_Comm &comm,
    const EnergySolver &solver) {

  // Load TPS from file path with proper error handling
  SITPST loaded_tps(params.mc_params.initial_config.rows(), params.mc_params.initial_config.cols());
  if (!loaded_tps.Load(tps_path)) {
    throw std::runtime_error("Failed to load TPS from path: " + tps_path);
  }

  // Create executor using the primary constructor with loaded TPS
  return std::make_unique<VMCPEPSOptimizer>(params, loaded_tps, comm, solver);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::Execute(void) {
  this->SetStatus(ExecutorStatus::EXEING);
  monte_carlo_engine_.WarmUp();

  // Set up optimization callback to track progress
  optimization_callback_.on_iteration =
      [this](size_t iteration, double energy, double energy_error, double gradient_norm) {
        if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
          energy_trajectory_.push_back(energy);
          energy_error_traj_.push_back(energy_error);
          grad_norm_.push_back(gradient_norm);

          if (energy < en_min_) {
            en_min_ = energy;
            tps_lowest_ = monte_carlo_engine_.State();
          }
        }
      };

  optimization_callback_.on_best_state_found = [this](const SITPST &state, double energy) {
    if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
      tps_lowest_ = state;
      en_min_ = energy;
    }
  };

  // Create energy evaluator function that provides gradient samples for stochastic reconfiguration
  auto energy_evaluator = [this](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    if (custom_energy_evaluator_) {
      return custom_energy_evaluator_(state);
    } else {
      return DefaultEnergyEvaluator_(state);
    }
  };

  // Perform optimization based on scheme
  typename OptimizerT::OptimizationResult result;

  // Use iterative optimization for all algorithms in the current implementation
  // Line search support can be added as enhancement later
  if (stochastic_reconfiguration_update_class_) {
    result = optimizer_.IterativeOptimize(monte_carlo_engine_.State(), energy_evaluator, optimization_callback_,
                                          &Ostar_samples_, &Ostar_mean_);
  } else {
    result = optimizer_.IterativeOptimize(monte_carlo_engine_.State(), energy_evaluator, optimization_callback_);
  }

  // CRITICAL: Update final state and synchronize across all ranks
  if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    // Validate the final state to prevent segmentation faults
    ValidateState_(result.optimized_state);
    monte_carlo_engine_.AssignState(result.optimized_state);
  }

  // Broadcast the final optimized state to all ranks
  MPI_Bcast(monte_carlo_engine_.State(), monte_carlo_engine_.Comm());

  // Update wavefunction component and normalize after final state update
  UpdateWavefunctionComponent_();
  monte_carlo_engine_.NormalizeStateOrder1();

  DumpData();
  this->SetStatus(ExecutorStatus::FINISH);
}

/**
 * @brief Update wavefunction component after tensor state changes.
 *
 * This method is called after the split index TPS is updated to ensure
 * the Monte Carlo sampling uses the updated wavefunction. This is critical
 * for maintaining consistency between the tensor state and the sampling.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::UpdateWavefunctionComponent_(void) {
  // CRITICAL: Update wavefunction component once the wave function (split index TPS) updates.
  // This ensures the Monte Carlo sampling uses the updated wavefunction
  Configuration config = monte_carlo_engine_.WavefuncComp().config;
  monte_carlo_engine_.WavefuncComp() = WaveFunctionComponentT(monte_carlo_engine_.State(), config, monte_carlo_engine_.WavefuncComp().trun_para);
}

/**
 * @brief Default energy evaluator for VMC PEPS optimizer.
 *
 * MPI Behavior:
 * - Input state is assumed to be valid only on the master rank
 * - State is broadcast to all ranks for Monte Carlo sampling
 * - Monte Carlo sampling is performed on all ranks in parallel
 * - Gradient calculation is performed on all ranks and gathered to master
 * - Energy and gradient are returned to the optimizer
 *
 * The input state is assumed only valid on the master rank.
 * After the energy evaluation,
 * the normalized state is broadcasted to all ranks
 * and this->split_index_tps_ will be assigned by the normalized state.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
std::tuple<TenElemT,
           typename VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SITPST,
           double>
VMCPEPSOptimizer<TenElemT,
                         QNT,
                         MonteCarloSweepUpdater,
                         EnergySolver>::DefaultEnergyEvaluator_(const SITPST &state) {
  // Delegate to persistent evaluator which encapsulates state broadcast, sampling and MPI reductions
  auto result = energy_grad_evaluator_->Evaluate(state);

  // Wire SR buffers into optimizer-owned storage when SR is enabled
  if (stochastic_reconfiguration_update_class_) {
    if (result.Ostar_mean.has_value()) {
      Ostar_mean_ = std::move(result.Ostar_mean.value());
    }
    Ostar_samples_ = std::move(result.Ostar_samples);
  }

  // Track trajectories on master is handled inside Gather in evaluator; here we just return
  return std::make_tuple(result.energy, std::move(result.gradient), result.energy_error);
}

/**
 * @brief Pre-allocate storage for energy samples, accumulators, and SR buffers.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::ReserveSamplesDataSpace_(void) {
  const size_t mc_samples = params_.mc_params.num_samples;
  if (mc_samples == 0) {
    throw std::invalid_argument("Monte Carlo samples cannot be zero");
  }

  for (size_t row = 0; row < monte_carlo_engine_.Ly(); row++) {
    for (size_t col = 0; col < monte_carlo_engine_.Lx(); col++) {
      const size_t dim = monte_carlo_engine_.State()({row, col}).size();
      if (dim == 0) {
        throw std::runtime_error("Zero dimension tensor at position (" +
            std::to_string(row) + ", " + std::to_string(col) + ")");
      }

      // No longer pre-alloc accumulators here; evaluator manages its own buffers
    }
  }

  for (size_t row = 0; row < monte_carlo_engine_.Ly(); row++) {
    for (size_t col = 0; col < monte_carlo_engine_.Lx(); col++) {
      const size_t dim = monte_carlo_engine_.State()({row, col}).size();
      grad_({row, col}) = std::vector<Tensor>(dim);
    }
  }

  if (monte_carlo_engine_.Rank() == 0) {
    const size_t step_count = params_.optimizer_params.base_params.max_iterations;
    energy_trajectory_.reserve(step_count);
    energy_error_traj_.reserve(step_count);
  }

  if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    grad_norm_.reserve(params_.optimizer_params.base_params.max_iterations);
  }

  if (stochastic_reconfiguration_update_class_) {
    Ostar_samples_.reserve(mc_samples);
    // Note: Ostar_mean_ will be initialized in GatherStatisticEnergyAndGrad_ when needed
  }
}

/**
 * @brief Print human-readable configuration of the executor (algorithm name, SR params, tech info).
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::PrintExecutorInfo_(void) {
  monte_carlo_engine_.PrintCommonInfo("VMC PEPS OPTIMIZER EXECUTOR");
  if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    size_t indent = 40;
    std::cout << std::setw(indent) << "PEPS update times:" << params_.optimizer_params.base_params.max_iterations
              << "\n";
    std::cout << std::setw(indent) << "PEPS update strategy:";
    // Get algorithm name from variant
    std::visit([](const auto& algo_params) {
      using T = std::decay_t<decltype(algo_params)>;
      if constexpr (std::is_same_v<T, SGDParams>) {
        std::cout << "StochasticGradient";
      } else if constexpr (std::is_same_v<T, StochasticReconfigurationParams>) {
        std::cout << "StochasticReconfiguration";
      } else if constexpr (std::is_same_v<T, AdaGradParams>) {
        std::cout << "AdaGrad";
      } else if constexpr (std::is_same_v<T, AdamParams>) {
        std::cout << "Adam";
      } else if constexpr (std::is_same_v<T, LBFGSParams>) {
        std::cout << "LBFGS";
      } else {
        std::cout << "Unknown";
      }
    }, params_.optimizer_params.algorithm_params);
    std::cout << "\n";
    if (stochastic_reconfiguration_update_class_) {
      const auto& sr_params = params_.optimizer_params.GetAlgorithmParams<StochasticReconfigurationParams>();
      std::cout << std::setw(indent) << "Conjugate gradient diagonal shift:"
                << sr_params.cg_params.diag_shift
                << "\n";
    }
  }
  monte_carlo_engine_.PrintTechInfo();
}

/**
 * @brief Core per-sample path: compute E_loc and O^*; update accumulators and SR buffers.
 */
// [removed] Legacy per-sample accumulation moved to MCEnergyGradEvaluator

/**
 * @brief Energy-only sampling (no gradient accumulation).
 */
// [removed] Legacy energy-only sampling moved to MCEnergyGradEvaluator

/**
 * @brief Clear per-iteration energy samples and gradient accumulators.
 */
// [removed] Legacy clearing moved to evaluator ownership

/**
 * @brief Gather energy and gradient statistics across all MPI ranks.
 *
 * MPI Behavior:
 * - Energy samples are gathered from all ranks and averaged
 * - Gradient tensors are calculated on each rank and gathered to master
 * - Only master rank holds the final gradient after gathering
 * - Energy trajectory and error are tracked only on master rank
 */
// [removed] Legacy MPI gather is implemented inside MCEnergyGradEvaluator

/**
 * @brief Detect anomalously low acceptance rates relative to global maxima and report.
 */
// [removed] Acceptance check handled in evaluator


/**
 * @brief Dump energy samples/trajectory and TPS using configured base name.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpData(const bool release_mem) {
  DumpData(params_.tps_dump_base_name, release_mem);  // Use base name from parameters
}

/**
 * @brief Implementation of the dumping routine with explicit base path.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              EnergySolver>::DumpData(const std::string &tps_base_name,
                                                      const bool release_mem) {
  // Generate paths with consistent naming: tps_base_name + "final" and tps_base_name + "lowest"
  std::string final_tps_path = tps_base_name + "final";
  std::string lowest_tps_path = tps_base_name + "lowest";
  std::string energy_data_path = "./energy";

  if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    // Only dump TPS if base name is not empty
    if (!tps_base_name.empty()) {
      monte_carlo_engine_.State().Dump(final_tps_path, release_mem);
      tps_lowest_.Dump(lowest_tps_path, release_mem);
    }
    EnsureDirectoryExists(energy_data_path);
  }
  MPI_Barrier(monte_carlo_engine_.Comm()); // configurations dump will collapse when creating path if there is no barrier.
  // Dump configuration using path from MonteCarloParams (empty = no dump)
  if (!params_.mc_params.config_dump_path.empty()) {
    monte_carlo_engine_.WavefuncComp().config.Dump(params_.mc_params.config_dump_path, monte_carlo_engine_.Rank());
  }
  // Note: legacy energy_sample<rank> dump is temporarily disabled during refactor
  if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    DumpVecData_(energy_data_path + "/energy_trajectory", energy_trajectory_);
    DumpVecDataDouble_(energy_data_path + "/energy_err_trajectory", energy_error_traj_);
  }
}

/**
 * @brief Binary dump of a vector of tensor elements.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpVecData_(
    const std::string &path, const std::vector<TenElemT> &data) {
  std::ofstream ofs(path, std::ofstream::binary);
  if (ofs) {
    for (const auto &elem : data) {
      ofs.write(reinterpret_cast<const char *>(&elem), sizeof(TenElemT));
    }
    ofs.close();
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              EnergySolver>::ValidateState_(const SITPST &state) {
  // Check that all tensors in the state are valid
  for (size_t row = 0; row < state.rows(); ++row) {
    for (size_t col = 0; col < state.cols(); ++col) {
      if (state(row, col) == nullptr || state({row, col}).empty()) {
        throw std::runtime_error("Empty tensor data at position (" +
            std::to_string(row) + ", " + std::to_string(col) + ")");
      }
      const auto &tensors = state({row, col});

      for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = tensors[i];
        if (tensor.IsDefault()) {
          throw std::runtime_error("Tensor at position (" +
              std::to_string(row) + ", " + std::to_string(col) +
              ", " + std::to_string(i) + ") is default tensor");
        }

        // Check for NaN or infinite values
        if (std::isnan(tensor.GetMaxAbs()) || std::isinf(tensor.GetMaxAbs())) {
          throw std::runtime_error("Tensor at position (" +
              std::to_string(row) + ", " + std::to_string(col) +
              ", " + std::to_string(i) + ") contains NaN or infinite values");
        }
      }
    }
  }
}

/**
 * @brief Binary dump of a vector of doubles.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpVecDataDouble_(
    const std::string &path, const std::vector<double> &data) {
  std::ofstream ofs(path, std::ofstream::binary);
  if (ofs) {
    for (const auto &elem : data) {
      ofs.write(reinterpret_cast<const char *>(&elem), sizeof(double));
    }
    ofs.close();
  }
}

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_IMPL_H 