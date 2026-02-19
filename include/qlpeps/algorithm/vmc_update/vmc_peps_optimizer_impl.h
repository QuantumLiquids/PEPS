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

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::VMCPEPSOptimizer(
    const VMCPEPSOptimizerParams &params,
    const SITPST &sitpst_init,
    const MPI_Comm &comm,
    const EnergySolver &solver,
    MonteCarloSweepUpdater mc_updater)
    : qlten::Executor(),
      monte_carlo_engine_(sitpst_init, params.mc_params, params.peps_params, comm, std::move(mc_updater),
                          params.runtime_params.config_rescue),
      params_(params),
      energy_solver_(solver),
      optimizer_(params.optimizer_params, comm, monte_carlo_engine_.Rank(), monte_carlo_engine_.MpiSize()),
      grad_(monte_carlo_engine_.Ly(), monte_carlo_engine_.Lx()),
      en_min_(std::numeric_limits<double>::max()),
      tps_lowest_(monte_carlo_engine_.State()),
      current_energy_error_(0.0) {

  // EnergySolver contract: must support receiving runtime warning params.
  // Printing is handled at executor/evaluator level for consistent budgets and rank policies.
  static_assert(requires(EnergySolver &s, const PsiConsistencyWarningParams &p) {
    s.SetPsiConsistencyWarningParams(p);
  }, "EnergySolver must implement SetPsiConsistencyWarningParams(const PsiConsistencyWarningParams&).");
  energy_solver_.SetPsiConsistencyWarningParams(params_.runtime_params.psi_consistency);

  // Check if using stochastic reconfiguration algorithm
  stochastic_reconfiguration_update_class_ = params.optimizer_params.IsAlgorithm<StochasticReconfigurationParams>();

  // Create persistent evaluator to reuse internal buffers; SR buffers toggled by algorithm type
  energy_grad_evaluator_ = std::make_unique<
      MCEnergyGradEvaluator<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>>(
      monte_carlo_engine_, energy_solver_, monte_carlo_engine_.Comm(), stochastic_reconfiguration_update_class_,
      params_.runtime_params.psi_consistency);

  // Ensure necessary directories exist for output
  if (!params_.tps_dump_base_name.empty()) {
    monte_carlo_engine_.EnsureDirectoryExists(params_.tps_dump_base_name + "final");
  }
  ReserveSamplesDataSpace_();
  // Hint evaluator for buffer reservation (coarse-grained)
  if (energy_grad_evaluator_) {
    energy_grad_evaluator_->ReserveBuffers(monte_carlo_engine_.Ly(), monte_carlo_engine_.Lx(), monte_carlo_engine_.SamplesPerRank());
  }
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
std::unique_ptr<VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>>
VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::CreateByLoadingTPS(
    const VMCPEPSOptimizerParams &params,
    const std::string &tps_path,
    const MPI_Comm &comm,
    const EnergySolver &solver,
    MonteCarloSweepUpdater mc_updater) {

  // Load TPS from file path with proper error handling
  SITPST loaded_tps(params.mc_params.initial_config.rows(), params.mc_params.initial_config.cols());
  if (!loaded_tps.Load(tps_path)) {
    throw std::runtime_error("Failed to load TPS from path: " + tps_path);
  }

  // Create executor using the primary constructor with loaded TPS
  return std::make_unique<VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>>(
      params, loaded_tps, comm, solver, std::move(mc_updater));
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::Execute(void) {
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

  // Cache spike stats from the result before the optimizer's internal copy is cleared
  spike_stats_ = result.spike_stats;

  // CRITICAL: Update final state and synchronize across all ranks
  if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    // Validate the final state to prevent segmentation faults
    ValidateState_(result.optimized_state);
    monte_carlo_engine_.AssignState(result.optimized_state);
  }

  // Broadcast the final optimized state to all ranks
  MPI_Bcast(monte_carlo_engine_.State(), monte_carlo_engine_.Comm());

  // Refresh wavefunction component and normalize after final state update
  monte_carlo_engine_.RefreshWavefunctionComponent();
  monte_carlo_engine_.NormalizeStateOrder1();

  DumpData();
  this->SetStatus(ExecutorStatus::FINISH);
}

// Removed: UpdateWavefunctionComponent_ is obsolete; use MonteCarloEngine::RefreshWavefunctionComponent instead.

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
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
std::tuple<TenElemT,
           typename VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::SITPST,
           double>
VMCPEPSOptimizer<TenElemT,
                         QNT,
                         MonteCarloSweepUpdater,
                         EnergySolver,
                         ContractorT>::DefaultEnergyEvaluator_(const SITPST &state) {
  // Delegate to persistent evaluator which encapsulates state broadcast, sampling and MPI reductions
  auto result = energy_grad_evaluator_->Evaluate(state);
  optimizer_.SetCurrentAcceptRates(result.accept_rates_avg);

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
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::ReserveSamplesDataSpace_(void) {
  const size_t mc_samples = monte_carlo_engine_.SamplesPerRank();
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
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::PrintExecutorInfo_(void) {
  monte_carlo_engine_.PrintCommonInfo("VMC PEPS OPTIMIZER EXECUTOR");
  if (monte_carlo_engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    const size_t indent = 40;
    const auto &base = params_.optimizer_params.base_params;
    std::cout << std::setw(indent) << "PEPS update times:" << base.max_iterations << "\n";

    std::cout << std::setw(indent) << "PEPS update strategy:";
    std::visit([&](const auto &algo_params) {
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

    std::cout << std::setw(indent) << "Energy tolerance:" << base.energy_tolerance << "\n";
    std::cout << std::setw(indent) << "Gradient tolerance:" << base.gradient_tolerance << "\n";
    std::cout << std::setw(indent) << "Plateau patience:" << base.plateau_patience << "\n";
    std::cout << std::setw(indent) << "Learning rate:" << base.learning_rate << "\n";
    if (base.lr_scheduler) {
      std::cout << std::setw(indent) << "LR scheduler:"
                << base.lr_scheduler->Name() << " (" << base.lr_scheduler->Describe() << ")\n";
    } else {
      std::cout << std::setw(indent) << "LR scheduler:" << "None" << "\n";
    }
    if (base.clip_value.has_value()) {
      std::cout << std::setw(indent) << "Gradient clip (per-element |g|):" << *base.clip_value << "\n";
    }
    if (base.clip_norm.has_value()) {
      std::cout << std::setw(indent) << "Gradient clip (global L2 norm):" << *base.clip_norm << "\n";
    }

    // Algorithm-specific parameters
    std::visit([&](const auto &algo_params) {
      using T = std::decay_t<decltype(algo_params)>;
      if constexpr (std::is_same_v<T, SGDParams>) {
        std::cout << std::setw(indent) << "SGD momentum:" << algo_params.momentum << "\n";
        std::cout << std::setw(indent) << "SGD nesterov:" << (algo_params.nesterov ? "true" : "false") << "\n";
        std::cout << std::setw(indent) << "SGD weight decay:" << algo_params.weight_decay << "\n";
      } else if constexpr (std::is_same_v<T, AdamParams>) {
        std::cout << std::setw(indent) << "Adam beta1:" << algo_params.beta1 << "\n";
        std::cout << std::setw(indent) << "Adam beta2:" << algo_params.beta2 << "\n";
        std::cout << std::setw(indent) << "Adam epsilon:" << algo_params.epsilon << "\n";
        std::cout << std::setw(indent) << "Adam weight decay:" << algo_params.weight_decay << "\n";
      } else if constexpr (std::is_same_v<T, AdaGradParams>) {
        std::cout << std::setw(indent) << "AdaGrad epsilon:" << algo_params.epsilon << "\n";
        std::cout << std::setw(indent) << "AdaGrad init accumulator:" << algo_params.initial_accumulator_value << "\n";
      } else if constexpr (std::is_same_v<T, LBFGSParams>) {
        std::cout << std::setw(indent) << "LBFGS history size:" << algo_params.history_size << "\n";
        std::cout << std::setw(indent) << "LBFGS tol_grad:" << algo_params.tolerance_grad << "\n";
        std::cout << std::setw(indent) << "LBFGS tol_change:" << algo_params.tolerance_change << "\n";
        std::cout << std::setw(indent) << "LBFGS max_eval/step:" << algo_params.max_eval << "\n";
      } else if constexpr (std::is_same_v<T, StochasticReconfigurationParams>) {
        std::cout << std::setw(indent) << "SR normalize update:"
                  << (algo_params.normalize_update ? "true" : "false") << "\n";
        std::cout << std::setw(indent) << "SR adaptive diag shift:" << algo_params.adaptive_diagonal_shift << "\n";
        std::cout << std::setw(indent) << "CG max iter:" << algo_params.cg_params.max_iter << "\n";
        std::cout << std::setw(indent) << "CG tolerance:" << algo_params.cg_params.tolerance << "\n";
        std::cout << std::setw(indent) << "CG absolute tolerance:" << algo_params.cg_params.absolute_tolerance << "\n";
        std::cout << std::setw(indent) << "CG residue restart step:" << algo_params.cg_params.residue_restart_step
                  << "\n";
        std::cout << std::setw(indent) << "CG diagonal shift:" << algo_params.cg_params.diag_shift << "\n";
      }
    }, params_.optimizer_params.algorithm_params);

    // Checkpoint configuration
    const auto &ckpt = params_.optimizer_params.checkpoint_params;
    if (ckpt.IsEnabled()) {
      std::cout << std::setw(indent) << "Checkpoint every:" << ckpt.every_n_steps << " steps\n";
      std::cout << std::setw(indent) << "Checkpoint base path:" << ckpt.base_path << "\n";
    } else {
      std::cout << std::setw(indent) << "Checkpoint:" << "disabled\n";
    }

    // Spike recovery configuration
    const auto &spike = params_.optimizer_params.spike_recovery_params;
    std::cout << std::setw(indent) << "Spike auto-recover:"
              << (spike.enable_auto_recover ? "enabled" : "disabled") << "\n";
    if (spike.enable_auto_recover) {
      std::cout << std::setw(indent) << "Spike max MC retries:" << spike.redo_mc_max_retries << "\n";
      std::cout << std::setw(indent) << "Spike S1 factor:" << spike.factor_err << "\n";
      std::cout << std::setw(indent) << "Spike S2 factor:" << spike.factor_grad << "\n";
      std::cout << std::setw(indent) << "Spike S3 factor:" << spike.factor_ngrad << "\n";
      std::cout << std::setw(indent) << "Spike SR min iters:" << spike.sr_min_iters_suspicious << "\n";
    }
    std::cout << std::setw(indent) << "Spike S4 rollback:"
              << (spike.enable_rollback ? "enabled" : "disabled") << "\n";
    if (spike.enable_rollback) {
      std::cout << std::setw(indent) << "Spike EMA window:" << spike.ema_window << "\n";
      std::cout << std::setw(indent) << "Spike S4 sigma_k:" << spike.sigma_k << "\n";
    }
    if (!spike.log_trigger_csv_path.empty()) {
      std::cout << std::setw(indent) << "Spike CSV log:" << spike.log_trigger_csv_path << "\n";
    }

    // Runtime warning controls
    const auto &psi = params_.runtime_params.psi_consistency;
    std::cout << std::setw(indent) << "psi_consistency warnings:"
              << (psi.enabled ? "enabled" : "disabled") << "\n";
    std::cout << std::setw(indent) << "psi_consistency master_only:" << (psi.master_only ? "true" : "false") << "\n";
    std::cout << std::setw(indent) << "psi_consistency threshold:" << psi.threshold << "\n";
    std::cout << std::setw(indent) << "psi_consistency max warnings:" << psi.max_warnings << "\n";
    std::cout << std::setw(indent) << "psi_consistency max print elems:" << psi.max_print_elems << "\n";

    if (!params_.tps_dump_base_name.empty()) {
      std::cout << std::setw(indent) << "TPS dump base name:" << params_.tps_dump_base_name << "\n";
    }
    if (!params_.tps_dump_path.empty()) {
      std::cout << std::setw(indent) << "TPS dump path:" << params_.tps_dump_path << "\n";
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
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::DumpData(const bool release_mem) {
  DumpData(params_.tps_dump_base_name, release_mem);  // Use base name from parameters
}

/**
 * @brief Implementation of the dumping routine with explicit base path.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
void VMCPEPSOptimizer<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              EnergySolver,
                              ContractorT>::DumpData(const std::string &tps_base_name,
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
    // Temporary deprecation notice for legacy binary dumps; keep behavior unchanged
    static bool deprec_once = false;
    if (!deprec_once) {
      std::cout << "[VMCPEPSOptimizer] Notice: binary trajectory dumps (energy_trajectory,"
                   " energy_err_trajectory) are deprecated and will be removed in a future release."
                   " Use energy/energy_trajectory.csv instead.\n";
      deprec_once = true;
    }
    DumpVecData_(energy_data_path + "/energy_trajectory", energy_trajectory_);
    DumpVecDataDouble_(energy_data_path + "/energy_err_trajectory", energy_error_traj_);
    // Also dump a human-friendly CSV of the trajectory, in append mode for multi-run continuity
    const std::string csv_path = energy_data_path + "/energy_trajectory.csv";
    const bool csv_exists = std::filesystem::exists(csv_path);
    std::ofstream csv(csv_path, std::ios::app);
    if (csv) {
      // Write header if file newly created
      if (!csv_exists) {
        csv << "iteration,energy,energy_error,gradient_norm\n";
      }
      const size_t n = energy_trajectory_.size();
      for (size_t i = 0; i < n; ++i) {
        const double e = std::real(energy_trajectory_[i]);
        const double err = (i < energy_error_traj_.size() ? energy_error_traj_[i] : 0.0);
        const double gnorm = (i < grad_norm_.size() ? grad_norm_[i] : 0.0);
        csv << i << "," << std::setprecision(17) << e << "," << err << "," << gnorm << "\n";
      }
      csv.close();
    }
  }
}

/**
 * @brief Binary dump of a vector of tensor elements.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::DumpVecData_(
    const std::string &path, const std::vector<TenElemT> &data) {
  std::ofstream ofs(path, std::ofstream::binary);
  if (ofs) {
    for (const auto &elem : data) {
      ofs.write(reinterpret_cast<const char *>(&elem), sizeof(TenElemT));
    }
    ofs.close();
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
void VMCPEPSOptimizer<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              EnergySolver,
                              ContractorT>::ValidateState_(const SITPST &state) {
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
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT>
void VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>::DumpVecDataDouble_(
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
