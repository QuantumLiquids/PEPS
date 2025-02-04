/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Base class for monte-carlo based measurement and variational update on PEPS
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_PEPS_BASE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_PEPS_BASE_H

#include "qlpeps/two_dim_tn/tps/tps.h"                            // TPS
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"                // SplitIndexTPS
#include "wave_function_component.h"                              // CheckWaveFunctionAmplitudeValidity
#include "monte_carlo_peps_params.h"                              // MonteCarloUpdateParams

namespace qlpeps {

/**
 *
 *  SetUp for Configuration
 *
 */
template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
class MonteCarloPEPSBaseExecutor : public Executor {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;

  MonteCarloPEPSBaseExecutor(const SITPST &sitps,
                             const MonteCarloParams &monte_carlo_params,
                             const PEPSParams &peps_params,
                             const MPI_Comm &comm) :
      split_index_tps_(sitps),
      lx_(sitps.cols()),
      ly_(sitps.rows()),
      tps_sample_(sitps.rows(), sitps.cols()),
      monte_carlo_params_(monte_carlo_params),
      u_double_(0, 1),
      warm_up_(false),
      comm_(comm) {
    MPI_SetUp_();
    WaveFunctionComponentType::trun_para = peps_params.truncate_para;
    random_engine.seed(std::random_device{}() + rank_ * 10086); // global random engineer
    InitConfigs_(monte_carlo_params.config_path, monte_carlo_params.alternative_init_config);
  }

  MonteCarloPEPSBaseExecutor(const size_t ly, const size_t lx,
                             const MonteCarloParams &monte_carlo_params,
                             const PEPSParams &peps_params,
                             const MPI_Comm &comm) :
      split_index_tps_(ly, lx),
      lx_(lx),
      ly_(ly),
      tps_sample_(ly, lx),
      monte_carlo_params_(monte_carlo_params),
      u_double_(0, 1),
      warm_up_(false),
      comm_(comm) {
    MPI_SetUp_();
    WaveFunctionComponentType::trun_para = peps_params.truncate_para;
    random_engine.seed(std::random_device{}() + rank_ * 10086); // global random engineer
    LoadTenData_(peps_params.wavefunction_path);
    InitConfigs_(monte_carlo_params.config_path, monte_carlo_params.alternative_init_config);
  }

  void Execute(void) override {}

 protected:
  void PrintCommonInfo_(const std::string &) const;
  void PrintTechInfo_() const;

  void LoadTenData_(const std::string &tps_path);

  ///< @return Success info
  int WarmUp_(void);

  ///< @return : accepted ratios
  std::vector<double> MCSweep_(const size_t);
  std::vector<double> MCSweep_();

  SITPST split_index_tps_;
  size_t lx_; //cols
  size_t ly_; //rows

  WaveFunctionComponentType tps_sample_;

  const MPI_Comm &comm_;
  int rank_;
  int mpi_size_;

  std::uniform_real_distribution<double> u_double_;

  MonteCarloParams monte_carlo_params_;

  bool warm_up_; // if has warmed up

 private:
  void MPI_SetUp_() {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);
  }

  void SyncValidConfiguration_();

  ///< @return Success info
  int InitConfigs_(const std::string &config_path, const Configuration &alternative_configs);
  int InitConfigs_(const Configuration &init_configs, bool warm_up = false);

};//MonteCarloPEPSBaseExecutor


template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                WaveFunctionComponentType>::PrintCommonInfo_(const std::string &header) const {
  if (rank_ == kMPIMasterRank) {
    const size_t indent = 40;
    std::cout << std::left;  // Set left alignment for the output
    std::cout << "\n";
    std::cout << "=====> " << header << " <=====" << "\n";

    std::cout << std::setw(indent) << "System size (lx, ly):" << "(" << lx_ << ", " << ly_ << ")\n";
    std::cout << std::setw(indent) << "PEPS bond dimension:" << split_index_tps_.GetMaxBondDimension() << "\n";
    std::cout << std::setw(indent) << "BMPS bond dimension:" << WaveFunctionComponentType::trun_para.value().D_min << "/"
              << WaveFunctionComponentType::trun_para.value().D_max << "\n";
    std::cout << std::setw(indent) << "BMPS Truncate Scheme:"
              << static_cast<int>(WaveFunctionComponentType::trun_para.value().compress_scheme) << "\n";
    std::cout << std::setw(indent) << "Sampling numbers:" << monte_carlo_params_.num_samples << "\n";
    std::cout << std::setw(indent) << "Monte Carlo sweep repeat times:" << monte_carlo_params_.sweeps_between_samples
              << "\n";
  }
}

template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                WaveFunctionComponentType>::PrintTechInfo_() const {
  if (rank_ == kMPIMasterRank) {
    const size_t indent = 40;
    std::cout << std::left;  // Set left alignment for the output
    std::cout << "\n";
    std::cout << "=====> TECHNICAL PARAMETERS <=====" << "\n";
    std::cout << std::setw(indent) << "The number of processors (including master):" << mpi_size_ << "\n";
    std::cout << std::setw(indent) << "The number of threads per processor:"
              << hp_numeric::GetTensorManipulationThreads()
              << "\n";
  }
}

template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
int MonteCarloPEPSBaseExecutor<TenElemT,
                               QNT,
                               WaveFunctionComponentType>::InitConfigs_(const std::string &config_path,
                                                                        const Configuration &alternative_configs) {
  Configuration config(ly_, lx_);
  bool load_success = config.Load(config_path, rank_);
  if (load_success) {
    tps_sample_ = WaveFunctionComponentType(split_index_tps_, config);
    warm_up_ = true;
  } else {
    // Fallback to default configuration from parameters
    tps_sample_ = WaveFunctionComponentType(split_index_tps_, alternative_configs);
    warm_up_ = false;
  }
  SyncValidConfiguration_();
  return 0;
}

///< Directly initialize with the given configuration
template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
int MonteCarloPEPSBaseExecutor<TenElemT,
                               QNT,
                               WaveFunctionComponentType>::InitConfigs_(const qlpeps::Configuration &init_configs,
                                                                        bool warm_up) {
  tps_sample_ = WaveFunctionComponentType(split_index_tps_, init_configs);
  warm_up_ = warm_up;
  SyncValidConfiguration_();
  return 0;
}

template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
int MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType>::WarmUp_() {
  if (!warm_up_) {
    Timer warm_up_timer("proc " + std::to_string(rank_) + " warm up");
    for (size_t sweep = 0; sweep < monte_carlo_params_.num_warmup_sweeps; sweep++) {
      auto accept_rates = MCSweep_(1);
    }
    warm_up_timer.PrintElapsed();
    warm_up_ = true;
  }
  bool psi_legal = CheckWaveFunctionAmplitudeValidity(tps_sample_);
  if (!psi_legal) {
    std::cout << "Proc " << std::setw(4) << rank_
              << ", psi : " << tps_sample_.amplitude
              << " Amplitude is still not legal after warm up. "
              << " Terminate the program"
              << std::endl;
    return 1;
  }
  return 0;
}

template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                WaveFunctionComponentType>::LoadTenData_(const std::string &tps_path) {
  if (!split_index_tps_.Load(tps_path)) {
    std::cout << "Loading TPS files fails." << std::endl;
    exit(-1);
  }
}

template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
std::vector<double> MonteCarloPEPSBaseExecutor<TenElemT,
                                               QNT,
                                               WaveFunctionComponentType>::MCSweep_(const size_t sweeps_between_samples) {
  std::vector<double> accept_rates;
  for (size_t i = 0; i < sweeps_between_samples; i++) {
    tps_sample_.MonteCarloSweepUpdate(split_index_tps_, unit_even_distribution, accept_rates);
  }
  return accept_rates;
}

template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
std::vector<double> MonteCarloPEPSBaseExecutor<TenElemT,
                                               QNT,
                                               WaveFunctionComponentType>::MCSweep_() {
  return MCSweep_(monte_carlo_params_.sweeps_between_samples);
}

template<typename TenElemT, typename QNT, typename WaveFunctionComponentType>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                WaveFunctionComponentType>::SyncValidConfiguration_() {
  int local_valid = CheckWaveFunctionAmplitudeValidity(tps_sample_) ? 1 : 0;
  std::vector<int> global_valid(mpi_size_);
  HANDLE_MPI_ERROR(MPI_Allgather(&local_valid, 1, MPI_INT,
                                 global_valid.data(), 1, MPI_INT,
                                 comm_));
  int all_valid = 1;
  for (int r = 0; r < mpi_size_; ++r) {
    if (!global_valid[r]) {
      all_valid = 0;
      break;
    }
  }
  if (all_valid) {
    std::cout << "All configurations are valid. " << std::endl;
    return;
  }

  // Find first valid rank (lowest priority)
  int any_valid = 0;
  int source_rank = MPI_UNDEFINED;
  for (int r = 0; r < mpi_size_; ++r) {
    if (global_valid[r]) {
      any_valid = 1;
      source_rank = r;
      break;
    }
  }

  // Broadcast valid configuration if exists
  if (any_valid) {
    Configuration config_valid;
    if (rank_ == source_rank) {
      config_valid = tps_sample_.config;
    }
    MPI_BCast(config_valid, source_rank, comm_);
    if (!local_valid) {
      tps_sample_ = WaveFunctionComponentType(split_index_tps_, config_valid);
      std::cout << "Rank" << rank_ << "replace configuration with valid configuration in Rank" << source_rank
                << std::endl;
      warm_up_ = false;
    }
  } else {
    // Handle all invalid case
    std::ostringstream oss;
    oss << "Rank " << rank_ << " invalid configuration:\n";
    oss << tps_sample_.config;
    oss << "\n";

    // Gather all error messages to rank 0 for clean output
    std::string local_msg = oss.str();
    std::vector<char> global_msgs;
    if (rank_ == kMPIMasterRank) {
      global_msgs.resize(local_msg.size() * mpi_size_);
    }

    MPI_Gather(local_msg.data(), local_msg.size(), MPI_CHAR,
               global_msgs.data(), local_msg.size(), MPI_CHAR,
               kMPIMasterRank, comm_);

    if (rank_ == kMPIMasterRank) {
      std::cerr << "All configurations invalid:\n";
      for (int r = 0; r < mpi_size_; ++r) {
        std::cerr << &global_msgs[r * local_msg.size()];
      }
    }

    MPI_Abort(comm_, EXIT_FAILURE);
  }
}
}//qlpeps
#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_PEPS_BASE_H
