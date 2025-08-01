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
#include "qlpeps/algorithm/vmc_update/vmc_peps_impl.h"  // For CalGTenForFermionicTensors
#include "qlpeps/utility/helpers.h"
#include "qlpeps/monte_carlo_tools/statistics.h"
#include "qlten/utility/timer.h"
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "qlpeps/algorithm/vmc_update/stochastic_reconfiguration_smatrix.h"

namespace qlpeps {
using namespace qlten;

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::VMCPEPSOptimizerExecutor(
    const VMCPEPSOptimizerParams &params,
    const TPST &tps_init,
    const MPI_Comm &comm,
    const EnergySolver &solver)
    : VMCPEPSOptimizerExecutor(params, SITPST(tps_init), comm, solver) {}

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
      gten_sum_(this->ly_, this->lx_),
      g_times_energy_sum_(this->ly_, this->lx_),
      grad_(this->ly_, this->lx_),
      en_min_(std::numeric_limits<double>::max()),
      tps_lowest_(this->split_index_tps_),
      current_energy_error_(0.0) {

  if (std::find(stochastic_reconfiguration_methods.cbegin(),
                stochastic_reconfiguration_methods.cend(),
                params.optimizer_params.update_scheme) != stochastic_reconfiguration_methods.cend()) {
    stochastic_reconfiguration_update_class_ = true;
  } else {
    stochastic_reconfiguration_update_class_ = false;
  }

  CreateDirectoryIfNeeded_(params_.peps_params.wavefunction_path);
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::VMCPEPSOptimizerExecutor(
    const VMCPEPSOptimizerParams &params,
    const size_t ly, const size_t lx,
    const MPI_Comm &comm,
    const EnergySolver &solver)
    : BaseExecutor(ly, lx, params.mc_params, params.peps_params, comm),
      params_(params),
      energy_solver_(solver),
      optimizer_(params.optimizer_params, comm, this->rank_, this->mpi_size_),
      gten_sum_(this->ly_, this->lx_),
      g_times_energy_sum_(this->ly_, this->lx_),
      grad_(this->ly_, this->lx_),
      en_min_(std::numeric_limits<double>::max()),
      tps_lowest_(this->split_index_tps_),
      current_energy_error_(0.0) {

  if (std::find(stochastic_reconfiguration_methods.cbegin(),
                stochastic_reconfiguration_methods.cend(),
                params.optimizer_params.update_scheme) != stochastic_reconfiguration_methods.cend()) {
    stochastic_reconfiguration_update_class_ = true;
  } else {
    stochastic_reconfiguration_update_class_ = false;
  }

  CreateDirectoryIfNeeded_(params_.peps_params.wavefunction_path);
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
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

  if (OptimizerT::IsLineSearchScheme(params_.optimizer_params.update_scheme)) {
    result = optimizer_.LineSearchOptimize(this->split_index_tps_, energy_evaluator, optimization_callback_);
  } else {
    // For stochastic reconfiguration, pass the gradient samples
    if (stochastic_reconfiguration_update_class_) {
      result = optimizer_.IterativeOptimize(this->split_index_tps_, energy_evaluator, optimization_callback_,
                                            &gten_samples_, &gten_ave_);
    } else {
      result = optimizer_.IterativeOptimize(this->split_index_tps_, energy_evaluator, optimization_callback_);
    }
  }

  // CRITICAL: Update final state and synchronize across all ranks
  if (this->rank_ == kMPIMasterRank) {
    // Validate the final state to prevent segmentation faults
    ValidateState_(result.optimized_state);
    this->split_index_tps_ = result.optimized_state;
  }
  
  // Broadcast the final optimized state to all ranks
  BroadCast(this->split_index_tps_, this->comm_);
  
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
  BroadCast(this->split_index_tps_, this->comm_);

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

      gten_sum_({row, col}) = std::vector<Tensor>(dim);
      for (size_t compt = 0; compt < dim; compt++) {
        gten_sum_({row, col})[compt] = Tensor(this->split_index_tps_({row, col})[compt].GetIndexes());
      }

      g_times_energy_sum_({row, col}) = gten_sum_({row, col});
    }
  }

  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      const size_t dim = this->split_index_tps_({row, col}).size();
      grad_({row, col}) = std::vector<Tensor>(dim);
    }
  }

  if (this->rank_ == 0) {
    const size_t step_count = params_.optimizer_params.core_params.step_lengths.size();
    energy_trajectory_.reserve(step_count);
    energy_error_traj_.reserve(step_count);
  }

  if (this->rank_ == kMPIMasterRank) {
    grad_norm_.reserve(params_.optimizer_params.core_params.step_lengths.size());
  }

  if (stochastic_reconfiguration_update_class_) {
    gten_samples_.reserve(mc_samples);
    // Note: gten_ave_ will be initialized in GatherStatisticEnergyAndGrad_ when needed
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::PrintExecutorInfo_(void) {
  this->PrintCommonInfo_("VMC PEPS OPTIMIZER EXECUTOR");
  if (this->rank_ == kMPIMasterRank) {
    size_t indent = 40;
    std::cout << std::setw(indent) << "PEPS update times:" << params_.optimizer_params.core_params.step_lengths.size()
              << "\n";
    std::cout << std::setw(indent) << "PEPS update strategy:"
              << WavefunctionUpdateSchemeString(params_.optimizer_params.update_scheme) << "\n";
    if (stochastic_reconfiguration_update_class_) {
      std::cout << std::setw(indent) << "Conjugate gradient diagonal shift:"
                << params_.optimizer_params.cg_params.diag_shift
                << "\n";
    }
  }
  this->PrintTechInfo_();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SampleEnergyAndHoles_(void) {
#ifdef QLPEPS_TIMING_MODE
  Timer cal_e_loc_and_holes_timer("cal_e_loc_and_holes (rank " + std::to_string(this->rank_) + ")");
#endif

  // Calculate local energy and holes for current configuration
  TensorNetwork2D<TenElemT, QNT> holes(this->ly_, this->lx_);
  TenElemT local_energy = energy_solver_.template CalEnergyAndHoles<TenElemT, QNT, true>(
      &this->split_index_tps_, &this->tps_sample_, holes);

  TenElemT local_energy_conjugate = ComplexConjugate(local_energy);
  TenElemT inverse_amplitude = ComplexConjugate(1.0 / this->tps_sample_.GetAmplitude());

  energy_samples_.push_back(local_energy);

  // Prepare gradient tensor sample for stochastic reconfiguration
  SITPST gradient_tensor_sample(this->ly_, this->lx_, this->split_index_tps_.PhysicalDim());

  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      const size_t basis_index = this->tps_sample_.GetConfiguration({row, col});

      Tensor gradient_tensor;
      if constexpr (Tensor::IsFermionic()) {
        // For fermionic systems, use specialized gradient calculation
        gradient_tensor = CalGTenForFermionicTensors(holes({row, col}), this->tps_sample_.tn({row, col}));
      } else {
        // For bosonic systems, use standard gradient calculation
        gradient_tensor = inverse_amplitude * holes({row, col});
      }

      // Accumulate gradient tensors
      gten_sum_({row, col})[basis_index] += gradient_tensor;
      g_times_energy_sum_({row, col})[basis_index] += local_energy_conjugate * gradient_tensor;

      // Store for stochastic reconfiguration if needed
      if (stochastic_reconfiguration_update_class_) {
        gradient_tensor_sample({row, col})[basis_index] = gradient_tensor;
      }
    }
  }

  if (stochastic_reconfiguration_update_class_) {
    gten_samples_.emplace_back(gradient_tensor_sample);
  }

#ifdef QLPEPS_TIMING_MODE
  cal_e_loc_and_holes_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
TenElemT VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::SampleEnergy_(void) {
  TenElemT energy_loc = energy_solver_.CalEnergy(&this->split_index_tps_, &this->tps_sample_);
  energy_samples_.push_back(energy_loc);
  return energy_loc;
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::ClearEnergyAndHoleSamples_(void) {
  energy_samples_.clear();

  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      size_t dim = this->split_index_tps_.PhysicalDim({row, col});
      gten_sum_({row, col}) = std::vector<Tensor>(dim);
      for (size_t compt = 0; compt < dim; compt++) {
        gten_sum_({row, col})[compt] = Tensor(this->split_index_tps_({row, col})[compt].GetIndexes());
      }
      g_times_energy_sum_({row, col}) = gten_sum_({row, col});
    }
  }

  if (stochastic_reconfiguration_update_class_) {
    gten_samples_.clear();
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
  gten_ave_ = gten_sum_ * (1.0 / sample_num);
  grad_ = g_times_energy_sum_ * (1.0 / sample_num) + ComplexConjugate(-energy) * gten_ave_;

  for (size_t row = 0; row < this->ly_; row++) {
    for (size_t col = 0; col < this->lx_; col++) {
      const size_t phy_dim = grad_({row, col}).size();
      for (size_t compt = 0; compt < phy_dim; compt++) {
        // gather and estimate grad in master (and maybe the error bar of grad)
        grad_({row, col})[compt] = MPIMeanTensor(grad_({row, col})[compt], this->comm_);
        // note here the grad data except in master are clear
        if (stochastic_reconfiguration_update_class_) {
          gten_ave_({row, col})[compt] = MPIMeanTensor(gten_ave_({row, col})[compt], this->comm_);
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

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              EnergySolver>::CreateDirectoryIfNeeded_(const std::string &path) {
  if (path.empty()) {
    if (this->rank_ == kMPIMasterRank) {
      std::cerr << "Warning: Wavefunction path is empty. No directory will be created." << std::endl;
    }
    return;
  }

  // Extract directory from path
  std::filesystem::path fs_path(path);
  std::filesystem::path dir_path = fs_path.parent_path();

  if (!dir_path.empty()) {
    try {
      if (!std::filesystem::exists(dir_path)) {
        if (this->rank_ == kMPIMasterRank) {
          std::cout << "Creating directory: " << dir_path.string() << std::endl;
          std::filesystem::create_directories(dir_path);
        }
        // Ensure all processes wait for directory creation
        MPI_Barrier(this->comm_);
      }
    } catch (const std::exception &e) {
      if (this->rank_ == kMPIMasterRank) {
        std::cerr << "Error creating directory " << dir_path.string() << ": " << e.what() << std::endl;
      }
    }
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpData(const bool release_mem) {
  DumpData(params_.peps_params.wavefunction_path, release_mem);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              EnergySolver>::DumpData(const std::string &tps_path,
                                                      const bool release_mem) {
  std::string energy_data_path = "./energy";
  if (this->rank_ == kMPIMasterRank) {
    this->split_index_tps_.Dump(tps_path, release_mem);
    tps_lowest_.Dump(tps_path + "lowest", release_mem);
    if (!qlmps::IsPathExist(energy_data_path)) {
      qlmps::CreatPath(energy_data_path);
    }
  }
  MPI_Barrier(this->comm_); // configurations dump will collapse when creating path if there is no barrier.
  this->tps_sample_.config.Dump(tps_path, this->rank_);
  DumpVecData(energy_data_path + "/energy_sample" + std::to_string(this->rank_), energy_samples_);
  if (this->rank_ == kMPIMasterRank) {
    DumpVecData(energy_data_path + "/energy_trajectory", energy_trajectory_);
    DumpVecDataDouble(energy_data_path + "/energy_err_trajectory", energy_error_traj_);
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpVecData(
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

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
void VMCPEPSOptimizerExecutor<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>::DumpVecDataDouble(
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