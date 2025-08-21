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
VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::VMCPEPSOptimizerExecutor(
    const VMCPEPSOptimizerParams &params,
    const SITPST &sitpst_init,
    const MPI_Comm &comm,
    const EnergySolver &solver)
    : BaseExecutor(sitpst_init, params.mc_params, params.peps_params, comm),
      params_(params),
      energy_solver_(solver),
      optimizer_(params.optimizer_params, comm, this->rank_, this->mpi_size_),
      Ostar_sum_(this->ly_, this->lx_),
      ELocConj_Ostar_sum_(this->ly_, this->lx_),
      grad_(this->ly_, this->lx_),
      en_min_(std::numeric_limits<double>::max()),
      tps_lowest_(this->split_index_tps_),
      current_energy_error_(0.0) {

  // Check if using stochastic reconfiguration algorithm
  stochastic_reconfiguration_update_class_ = params.optimizer_params.IsAlgorithm<StochasticReconfigurationParams>();

  // Ensure necessary directories exist for output
  if (!params_.tps_dump_base_name.empty()) {
    this->EnsureDirectoryExists_(params_.tps_dump_base_name + "final");
  }
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
std::unique_ptr<VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>>
VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::CreateByLoadingTPS(
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
  return std::make_unique<VMCPEPSOptimizerExecutor>(params, loaded_tps, comm, solver);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::Execute(void) {
  this->SetStatus(ExecutorStatus::EXEING);
  this->WarmUp_();

  // Set up optimization callback to track progress
  optimization_callback_.on_iteration =
      [this](size_t iteration, double energy, double energy_error, double gradient_norm) {
        if (this->rank_ == kMPIMasterRank) {
          energy_trajectory_.push_back(energy);
          energy_error_traj_.push_back(energy_error);
          grad_norm_.push_back(gradient_norm);

          if (energy < en_min_) {
            en_min_ = energy;
            tps_lowest_ = this->split_index_tps_;
          }
        }
      };

  optimization_callback_.on_best_state_found = [this](const SITPST &state, double energy) {
    if (this->rank_ == kMPIMasterRank) {
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
    result = optimizer_.IterativeOptimize(this->split_index_tps_, energy_evaluator, optimization_callback_,
                                          &Ostar_samples_, &Ostar_mean_);
  } else {
    result = optimizer_.IterativeOptimize(this->split_index_tps_, energy_evaluator, optimization_callback_);
  }

  // CRITICAL: Update final state and synchronize across all ranks
  if (this->rank_ == kMPIMasterRank) {
    // Validate the final state to prevent segmentation faults
    ValidateState_(result.optimized_state);
    this->split_index_tps_ = result.optimized_state;
  }
  
  // Broadcast the final optimized state to all ranks
  MPI_Bcast(this->split_index_tps_, this->comm_);
  
  // Update wavefunction component and normalize after final state update
  UpdateWavefunctionComponent_();
  this->NormTPSForOrder1Amplitude_();

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
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::UpdateWavefunctionComponent_(void) {
  // CRITICAL: Update wavefunction component once the wave function (split index TPS) updates.
  // This ensures the Monte Carlo sampling uses the updated wavefunction
  Configuration config = this->tps_sample_.config;
  this->tps_sample_ = WaveFunctionComponentT(this->split_index_tps_, config, this->tps_sample_.trun_para);
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
           typename VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SITPST,
           double>
VMCPEPSOptimizerExecutor<TenElemT,
                         QNT,
                         MonteCarloSweepUpdater,
                         EnergySolver>::DefaultEnergyEvaluator_(const SITPST &state) {
  // Update internal state - avoid self-assignment
  if (this->rank_ == kMPIMasterRank) {
    // Check if this is a self-assignment to avoid issues
    if (&state != &this->split_index_tps_) {
      this->split_index_tps_ = state;
    }
  }
  
  // CRITICAL: Broadcast the updated state to all ranks
  // This ensures all ranks have the same state for Monte Carlo sampling
  MPI_Bcast(this->split_index_tps_, this->comm_);

  // Update wavefunction component after tensor update
  UpdateWavefunctionComponent_();

  // Normalize TPS tensor so that the max amplitude of the wave function across all ranks is order 1. 
  this->NormTPSForOrder1Amplitude_();

  // Clear previous samples
  ClearEnergyAndHoleSamples_();

  // Perform Monte Carlo sampling 
  std::vector<double> accept_rates_accum;
  for (size_t sweep = 0; sweep < params_.mc_params.num_samples; sweep++) {
    std::vector<double> accept_rates = this->MCSweep_();
    if (sweep == 0) {
      accept_rates_accum = accept_rates;
    } else {
      for (size_t i = 0; i < accept_rates_accum.size(); i++) {
        accept_rates_accum[i] += accept_rates[i];
      }
    }
    SampleEnergyAndHoles_();
  }

  // Calculate average acceptance rates
  std::vector<double> accept_rates_avg = accept_rates_accum;
  for (double &rate : accept_rates_avg) {
    rate /= double(params_.mc_params.num_samples);
  }

  // Check acceptance rates for anomalies
  AcceptanceRateCheck(accept_rates_avg);

  // Perform Monte Carlo sampling and energy evaluation
  auto [energy, gradient, energy_error] = GatherStatisticEnergyAndGrad_();

  return std::make_tuple(energy, gradient, energy_error);
}

/**
 * @brief Pre-allocate storage for energy samples, accumulators, and SR buffers.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::ReserveSamplesDataSpace_(void) {
  const size_t mc_samples = params_.mc_params.num_samples;
  if (mc_samples == 0) {
    throw std::invalid_argument("Monte Carlo samples cannot be zero");
  }

  energy_samples_.reserve(mc_samples);

  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      const size_t dim = this->split_index_tps_({row, col}).size();
      if (dim == 0) {
        throw std::runtime_error("Zero dimension tensor at position (" +
            std::to_string(row) + ", " + std::to_string(col) + ")");
      }

      Ostar_sum_({row, col}) = std::vector<Tensor>(dim);
      for (size_t compt = 0; compt < dim; compt++) {
        Ostar_sum_({row, col})[compt] = Tensor(this->split_index_tps_({row, col})[compt].GetIndexes());
      }

      ELocConj_Ostar_sum_({row, col}) = Ostar_sum_({row, col});
    }
  }

  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      const size_t dim = this->split_index_tps_({row, col}).size();
      grad_({row, col}) = std::vector<Tensor>(dim);
    }
  }

  if (this->rank_ == 0) {
    const size_t step_count = params_.optimizer_params.base_params.max_iterations;
    energy_trajectory_.reserve(step_count);
    energy_error_traj_.reserve(step_count);
  }

  if (this->rank_ == kMPIMasterRank) {
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
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::PrintExecutorInfo_(void) {
  this->PrintCommonInfo_("VMC PEPS OPTIMIZER EXECUTOR");
  if (this->rank_ == kMPIMasterRank) {
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
  this->PrintTechInfo_();
}

/**
 * @brief Core per-sample path: compute E_loc and O^*; update accumulators and SR buffers.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SampleEnergyAndHoles_(void) {
#ifdef QLPEPS_TIMING_MODE
  Timer cal_e_loc_and_holes_timer("cal_e_loc_and_holes (rank " + std::to_string(this->rank_) + ")");
#endif

  // Calculate local energy and holes for current configuration
  // E_loc = ∑_{S'} (Ψ*(S')/Ψ*(S)) <S'|H|S>
  TensorNetwork2D<TenElemT, QNT> holes(this->ly_, this->lx_);
  TenElemT local_energy = energy_solver_.template CalEnergyAndHoles<TenElemT, QNT, true>(
      &this->split_index_tps_, &this->tps_sample_, holes);

  // For complex gradient calculation: use complex conjugate of local energy
  // This implements Eq. (complex grad) from the research notes
  TenElemT local_energy_conjugate = ComplexConjugate(local_energy);
  TenElemT inverse_amplitude = ComplexConjugate(1.0 / this->tps_sample_.GetAmplitude());

  energy_samples_.push_back(local_energy);

  // Gradient accumulation uses:
  // E_loc = ∑_{S'} (Ψ*(S')/Ψ*(S)) <S'|H|S>
  // ∂E/∂θ* = <E_loc^* O*> − E^* <O*>, where O* = ∂ln(Ψ*)/∂θ*
  // SR: store per-sample O*(S) when enabled
  SITPST Ostar_sample(this->ly_, this->lx_, this->split_index_tps_.PhysicalDim());
  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      const size_t basis_index = this->tps_sample_.GetConfiguration({row, col});

      Tensor Ostar_tensor;
      if constexpr (Tensor::IsFermionic()) {
        Ostar_tensor = CalGTenForFermionicTensors(holes({row, col}), this->tps_sample_.tn({row, col}));
      } else {
        Ostar_tensor = inverse_amplitude * holes({row, col});
      }

      Ostar_sum_({row, col})[basis_index] += Ostar_tensor; // Σ O^*
      ELocConj_Ostar_sum_({row, col})[basis_index] += local_energy_conjugate * Ostar_tensor; // Σ E_loc^* O^*

      if (stochastic_reconfiguration_update_class_) {
        Ostar_sample({row, col})[basis_index] = Ostar_tensor; // O^*(S)
      }
    }
  }
  if (stochastic_reconfiguration_update_class_) {
    Ostar_samples_.emplace_back(Ostar_sample); // SR: push one O* sample for S-matrix
  }

#ifdef QLPEPS_TIMING_MODE
  cal_e_loc_and_holes_timer.PrintElapsed();
#endif
}

/**
 * @brief Energy-only sampling (no gradient accumulation).
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
TenElemT VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SampleEnergy_(void) {
  TenElemT energy_loc = energy_solver_.CalEnergy(&this->split_index_tps_, &this->tps_sample_);
  energy_samples_.push_back(energy_loc);
  return energy_loc;
}

/**
 * @brief Clear per-iteration energy samples and gradient accumulators.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::ClearEnergyAndHoleSamples_(void) {
  energy_samples_.clear();

  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      size_t dim = this->split_index_tps_.PhysicalDim({row, col});
      Ostar_sum_({row, col}) = std::vector<Tensor>(dim);
      for (size_t compt = 0; compt < dim; compt++) {
        Ostar_sum_({row, col})[compt] = Tensor(this->split_index_tps_({row, col})[compt].GetIndexes());
      }
      ELocConj_Ostar_sum_({row, col}) = Ostar_sum_({row, col});
    }
  }

  if (stochastic_reconfiguration_update_class_) {
    Ostar_samples_.clear();
  }
}

/**
 * @brief Gather energy and gradient statistics across all MPI ranks.
 * 
 * MPI Behavior:
 * - Energy samples are gathered from all ranks and averaged
 * - Gradient tensors are calculated on each rank and gathered to master
 * - Only master rank holds the final gradient after gathering
 * - Energy trajectory and error are tracked only on master rank
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
std::tuple<TenElemT,
           typename VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SITPST,
           double>
VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::GatherStatisticEnergyAndGrad_(void) {
  TenElemT en_self = Mean(energy_samples_); //energy value in each processor
  auto [energy, en_err] = GatherStatisticSingleData(en_self, MPI_Comm(this->comm_));
  qlten::hp_numeric::MPI_Bcast(&energy, 1, kMPIMasterRank, MPI_Comm(this->comm_));
  if (this->rank_ == kMPIMasterRank) {
    energy_trajectory_.push_back(energy);
    energy_error_traj_.push_back(en_err);
  }

  //calculate grad in each processor
  const size_t sample_num = params_.mc_params.num_samples;
  Ostar_mean_ = Ostar_sum_ * (1.0 / sample_num);
  grad_ = ELocConj_Ostar_sum_ * (1.0 / sample_num) + ComplexConjugate(-energy) * Ostar_mean_;

  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      const size_t phy_dim = grad_({row, col}).size();
      for (size_t compt = 0; compt < phy_dim; compt++) {
        // gather and estimate grad in master (and maybe the error bar of grad)
        grad_({row, col})[compt] = MPIMeanTensor(grad_({row, col})[compt], this->comm_);
        // note here the grad data except in master are clear
        if (stochastic_reconfiguration_update_class_) {
          Ostar_mean_({row, col})[compt] = MPIMeanTensor(Ostar_mean_({row, col})[compt], this->comm_);
        }
      }
    }
  }
  grad_.ActFermionPOps();
  if (this->rank_ == kMPIMasterRank) {
    grad_norm_.push_back(grad_.NormSquare());
  }
  //do not broadcast because only broadcast the updated TPS
  double energy_error = 0.0;
  if (this->rank_ == kMPIMasterRank && !energy_error_traj_.empty()) {
    energy_error = energy_error_traj_.back();
  }
  return std::make_tuple(energy, grad_, energy_error);
}

/**
 * @brief Detect anomalously low acceptance rates relative to global maxima and report.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
bool VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::AcceptanceRateCheck(
    const std::vector<double> &accept_rate) const {
  bool too_small = false;
  std::vector<double> global_max(accept_rate.size());
  HANDLE_MPI_ERROR(::MPI_Allreduce(accept_rate.data(),
                                   global_max.data(),
                                   accept_rate.size(),
                                   MPI_DOUBLE,
                                   MPI_MAX,
                                   this->comm_));
  for (size_t i = 0; i < accept_rate.size(); i++) {
    if (accept_rate[i] < 0.5 * global_max[i]) { //anomaly case
      too_small = true;
      std::cout << "Process " << this->rank_ << ": Acceptance rate[" << i
                << "] = " << accept_rate[i] << " is too small compared to global max "
                << global_max[i] << std::endl;
    }
  }
  return too_small;
}


/**
 * @brief Dump energy samples/trajectory and TPS using configured base name.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpData(const bool release_mem) {
  DumpData(params_.tps_dump_base_name, release_mem);  // Use base name from parameters
}

/**
 * @brief Implementation of the dumping routine with explicit base path.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              EnergySolver>::DumpData(const std::string &tps_base_name,
                                                      const bool release_mem) {
  // Generate paths with consistent naming: tps_base_name + "final" and tps_base_name + "lowest"
  std::string final_tps_path = tps_base_name + "final";
  std::string lowest_tps_path = tps_base_name + "lowest";
  std::string energy_data_path = "./energy";
  
  if (this->rank_ == kMPIMasterRank) {
    // Only dump TPS if base name is not empty
    if (!tps_base_name.empty()) {
      this->split_index_tps_.Dump(final_tps_path, release_mem);
      tps_lowest_.Dump(lowest_tps_path, release_mem);
    }
    EnsureDirectoryExists(energy_data_path);
  }
  MPI_Barrier(this->comm_); // configurations dump will collapse when creating path if there is no barrier.
  // Dump configuration using path from MonteCarloParams (empty = no dump)
  if (!params_.mc_params.config_dump_path.empty()) {
    this->tps_sample_.config.Dump(params_.mc_params.config_dump_path, this->rank_);
  }
  DumpVecData_(energy_data_path + "/energy_sample" + std::to_string(this->rank_), energy_samples_);
  if (this->rank_ == kMPIMasterRank) {
    DumpVecData_(energy_data_path + "/energy_trajectory", energy_trajectory_);
    DumpVecDataDouble_(energy_data_path + "/energy_err_trajectory", energy_error_traj_);
  }
}

/**
 * @brief Binary dump of a vector of tensor elements.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpVecData_(
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
void VMCPEPSOptimizerExecutor<TenElemT,
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
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpVecDataDouble_(
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