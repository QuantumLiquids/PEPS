/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Base class for monte-carlo based measurement and variational update on PEPS
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_PEPS_BASE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_PEPS_BASE_H

#include <iomanip>                                                // std::setprecision
#include <filesystem>                                             // std::filesystem
#include "qlpeps/two_dim_tn/tps/tps.h"                            // TPS
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"                // SplitIndexTPS
#include "qlpeps/vmc_basic/wave_function_component.h"                              // CheckWaveFunctionAmplitudeValidity
#include "monte_carlo_peps_params.h"                              // MonteCarloUpdateParams

namespace qlpeps {

template<typename T, typename TenElemT, typename QNT>
concept MonteCarloSweepUpdaterConcept = requires(
    T updater,
    const SplitIndexTPS<TenElemT, QNT> &sitps,
    TPSWaveFunctionComponent<TenElemT, QNT> &component,
    std::vector<double> &accept_ratios
) {
  { updater(sitps, component, accept_ratios) } -> std::same_as<void>;
};

/**
 * @brief Base class for Monte Carlo based measurement and variational update on PEPS
 * 
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 * @tparam MonteCarloSweepUpdater Functor defining Monte Carlo sweep update
 * 
 * @details The MonteCarloSweepUpdater functor should have default constructor and operator() signature:
 * void operator()(const SplitIndexTPS<TenElemT, QNT>&,
 *                 TPSWaveFunctionComponent<TenElemT, QNT>&,
 *                 std::vector<double>&)
 * 
 * It updates the TPS-based wavefunction component and stores accept ratios in a vector.
 * 
 * Built-in updaters in configuration_update_strategies/:
 * - MCUpdateSquareNNExchange
 * - MCUpdateSquareNNFullSpaceUpdate  
 * - MCUpdateSquareTNN3SiteExchange
 * 
 * @todo Configuration setup needs to be declared/redesigned
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater> requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
class MonteCarloPEPSBaseExecutor : public Executor {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using WaveFunctionComponentT = TPSWaveFunctionComponent<TenElemT, QNT>;
  
  /**
   * @brief Constructor with explicit TPS provided by user.
   * 
   * User provides all data explicitly - no hidden file loading.
   * This constructor gives users complete control over the input data.
   * 
   * @param sitps Split-index TPS provided by user
   * @param monte_carlo_params Monte Carlo sampling parameters
   * @param peps_params PEPS calculation parameters  
   * @param comm MPI communicator
   */
  MonteCarloPEPSBaseExecutor(const SITPST &sitps,
                             const MonteCarloParams &monte_carlo_params,
                             const PEPSParams &peps_params,
                             const MPI_Comm &comm);
  
  /**
   * @brief Static factory function to create executor by loading TPS from file path.
   * 
   * Convenience factory for users who have TPS data stored on disk.
   * This follows the single-responsibility principle: factory handles loading,
   * constructor only handles initialization.
   * 
   * @param tps_path Path to TPS data files on disk
   * @param monte_carlo_params Monte Carlo sampling parameters (must contain valid initial_config)
   * @param peps_params PEPS calculation parameters
   * @param comm MPI communicator
   * @return Unique pointer to the created executor
   * 
   * @note The initial_config in monte_carlo_params must be properly sized to determine lattice dimensions
   * @note This factory automatically loads TPS from disk and initializes the system
   * 
   * Usage:
   *   auto executor = MonteCarloPEPSBaseExecutor::CreateByLoadingTPS(tps_path, mc_params, peps_params, comm);
   */
  static std::unique_ptr<MonteCarloPEPSBaseExecutor> 
  CreateByLoadingTPS(const std::string& tps_path,
                     const MonteCarloParams& monte_carlo_params,
                     const PEPSParams& peps_params,
                     const MPI_Comm& comm);

  void Execute(void) override {}

  // State access methods - clean interface for users
  const Configuration& GetCurrentConfiguration() const { return tps_sample_.config; }
  const SITPST& GetCurrentTPS() const { return split_index_tps_; }
  
  // Optional persistence methods - user controls I/O
  void DumpConfiguration(const std::string& path) const {
    tps_sample_.config.Dump(path, rank_);
  }
  
  void DumpTPS(const std::string& path) const {
    split_index_tps_.Dump(path);
  }

 protected:
  
  /**
   * @brief Ensure directory exists for the given file path (MPI-safe)
   * @param file_path Path to file (directory will be extracted and created if needed)
   * 
   * This method handles MPI coordination automatically:
   * - Only rank 0 creates directories
   * - All processes wait at MPI barrier for completion
   * - Safe to call from any derived class
   */
  void EnsureDirectoryExists_(const std::string& file_path);
  void PrintCommonInfo_(const std::string &) const;
  void PrintTechInfo_() const;

  ///< @return Success info
  int WarmUp_(void);

  ///< @return : accepted ratios
  std::vector<double> MCSweep_(const size_t);
  std::vector<double> MCSweep_();

  void NormTPSForOrder1Amplitude_(void);

  SITPST split_index_tps_;
  size_t lx_; //cols
  size_t ly_; //rows

  WaveFunctionComponentT tps_sample_;

  const MPI_Comm &comm_;
  int rank_;
  int mpi_size_;

  MonteCarloSweepUpdater mc_sweep_updater_;
  std::uniform_real_distribution<double> u_double_;

  MonteCarloParams monte_carlo_params_;

  bool warm_up_; // if has warmed up

 private:
  void MPI_SetUp_() {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);
  }

  /**
   * @brief Synchronize valid configurations across all MPI processes.
   * 
   * @details This function is crucial for numerical stability in large-scale 
   * Monte Carlo simulations. When initial configurations lead to wave function
   * amplitudes that are too small (approaching floating-point underflow), 
   * this function ensures system reliability by:
   * 
   * 1. **Validity Check**: Each MPI process checks if its current configuration
   *    produces a numerically valid wave function amplitude
   * 2. **Global Communication**: All validity statuses are gathered via MPI_Allgather
   * 3. **Configuration Rescue**: If some processes have invalid configurations,
   *    they adopt valid configurations from other processes
   * 4. **Warm-up Reset**: Processes that receive rescued configurations are marked
   *    as not warmed up, requiring re-thermalization
   * 
   * This mechanism is particularly important for:
   * - Large system sizes where some random initial configurations may be pathological
   * - Complex many-body states where amplitude can vary by many orders of magnitude
   * - Ensuring all MPI processes start from numerically stable states
   * 
   * @warning If ALL processes have invalid configurations, the program terminates
   * with detailed error information to help diagnose the underlying issue.
   * 
   * @note This function modifies tps_sample_ and warm_up_ status for processes
   * that receive rescued configurations.
   * 
   * @todo Enhanced diagnostic output: Add detailed amplitude information, 
   * per-process diagnostics, and more comprehensive error reporting. 
   * Requires careful MPI output coordination to avoid race conditions.
   * Current implementation uses minimal but safe output.
   */
  void EnsureConfigurationValidity_();



  /**
   * @brief Initialize Monte Carlo sampling system with user-provided configuration.
   * 
   * Performs complete initialization sequence:
   * 1. Creates wave function sample from user configuration and TPS
   * 2. Validates and synchronizes configurations across MPI processes (with fallback)
   * 3. Normalizes wave function for numerical stability
   * 
   * @note This is the main initialization entry point called by constructors.
   * Includes fallback logic: if user configuration is invalid on some MPI processes,
   * valid configurations from other processes will be broadcast as rescue.
   * 
   * @warning If ALL processes have invalid configurations, program terminates.
   */
  void Initialize();

private:

};//MonteCarloPEPSBaseExecutor


template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                MonteCarloSweepUpdater>::PrintCommonInfo_(const std::string &header) const {
  if (rank_ == kMPIMasterRank) {
    const size_t indent = 40;
    std::cout << std::left;  // Set left alignment for the output
    std::cout << "\n";
    std::cout << "=====> " << header << " <=====" << "\n";

    std::cout << std::setw(indent) << "System size (lx, ly):" << "(" << lx_ << ", " << ly_ << ")\n";
    std::cout << std::setw(indent) << "PEPS bond dimension:" << split_index_tps_.GetMaxBondDimension() << "\n";
    std::cout << std::setw(indent) << "BMPS bond dimension:" << tps_sample_.trun_para.D_min
              << "/"
              << tps_sample_.trun_para.D_max << "\n";
    std::cout << std::setw(indent) << "BMPS Truncate Scheme:"
              << static_cast<int>(tps_sample_.trun_para.compress_scheme) << "\n";
    std::cout << std::setw(indent) << "Sampling numbers:" << monte_carlo_params_.num_samples << "\n";
    std::cout << std::setw(indent) << "Monte Carlo sweep repeat times:" << monte_carlo_params_.sweeps_between_samples
              << "\n";
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                MonteCarloSweepUpdater>::PrintTechInfo_() const {
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

/**
 * @brief Initialize Monte Carlo sampling system with user-provided configuration.
 * 
 * Entry point that orchestrates the complete initialization sequence:
 * construction, validation/fallback, and normalization.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                MonteCarloSweepUpdater>::Initialize() {
  // Create wave function sample from user configuration and TPS
  tps_sample_ = WaveFunctionComponentT(split_index_tps_, 
                                      monte_carlo_params_.initial_config, 
                                      tps_sample_.trun_para);
  
  // Validate configurations and rescue invalid ones from other processes
  EnsureConfigurationValidity_();
  
  // Normalize for numerical stability
  NormTPSForOrder1Amplitude_();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
int MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::WarmUp_() {
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
    std::cout << "Proc " << rank_
              << ", psi : " << std::scientific << tps_sample_.amplitude
              << " Amplitude is still not legal after warm up. "
              << " Terminate the program"
              << std::endl;
    MPI_Abort(comm_, EXIT_FAILURE);
    return 1;
  }
  NormTPSForOrder1Amplitude_();
  return 0;
}



template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
std::vector<double> MonteCarloPEPSBaseExecutor<TenElemT,
                                               QNT,
                                               MonteCarloSweepUpdater>::MCSweep_(const size_t sweeps_between_samples) {
#ifdef QLPEPS_TIMING_MODE
  Timer mc_sweep_timer("monte_carlo_sweep (rank " + std::to_string(rank_) + ")");
#endif
  std::vector<double> accept_rates;
  for (size_t i = 0; i < sweeps_between_samples; i++) {
    mc_sweep_updater_(split_index_tps_, tps_sample_, accept_rates);
  }
#ifdef QLPEPS_TIMING_MODE
  mc_sweep_timer.PrintElapsed();
#endif
  return accept_rates;
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
std::vector<double> MonteCarloPEPSBaseExecutor<TenElemT,
                                               QNT,
                                               MonteCarloSweepUpdater>::MCSweep_() {
  return MCSweep_(monte_carlo_params_.sweeps_between_samples);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                MonteCarloSweepUpdater>::EnsureConfigurationValidity_() {
  // 1. Collect validity status from all ranks
  int local_valid = CheckWaveFunctionAmplitudeValidity(tps_sample_) ? 1 : 0;
  std::vector<int> global_valid(mpi_size_);
  HANDLE_MPI_ERROR(MPI_Allgather(&local_valid, 1, MPI_INT,
                                 global_valid.data(), 1, MPI_INT, comm_));
  
  // 2. Count valid processes
  int num_valid = std::accumulate(global_valid.begin(), global_valid.end(), 0);
  
  // 3. All valid - just log success
  if (num_valid == mpi_size_) {
    if (rank_ == kMPIMasterRank) {
      std::cout << "✓ Configuration validation: All " << mpi_size_ 
                << " processes have valid configurations." << std::endl;
    }
    return;
  }
  
  // 4. Find first valid rank for rescue source
  auto valid_iter = std::find(global_valid.begin(), global_valid.end(), 1);
  if (valid_iter == global_valid.end()) {
    // Complete failure - log diagnostics and abort
    if (rank_ == kMPIMasterRank) {
      std::cerr << "\n=== CRITICAL CONFIGURATION FAILURE ===\n"
                << "All " << mpi_size_ << " processes have invalid configurations!\n"
                << "Check: bond dimension, truncation cutoff, initial configuration\n";
    }
    std::ostringstream oss;
    oss << "Rank " << rank_ << ": amplitude=" << tps_sample_.amplitude
        << ", magnitude=" << std::abs(tps_sample_.amplitude) << std::endl;
    hp_numeric::GatherAndPrintErrorMessages(oss.str(), comm_);
    MPI_Abort(comm_, EXIT_FAILURE);
  }
  
  // 5. Execute rescue operation
  int source_rank = static_cast<int>(valid_iter - global_valid.begin());
  Configuration config_valid(ly_, lx_);
  
  if (rank_ == source_rank) {
    config_valid = tps_sample_.config;
  }
  MPI_BCast(config_valid, source_rank, comm_);
  
  if (local_valid == 0) {  
    tps_sample_ = WaveFunctionComponentT(split_index_tps_, config_valid, tps_sample_.trun_para);
    warm_up_ = false;  // Reset warm-up for rescued processes
    std::cout << "Rank " << rank_ << ": rescued from rank " << source_rank 
              << " (new amplitude: " << std::abs(tps_sample_.amplitude) << ")" << std::endl;
  }
  
  // 6. Log rescue completion summary
  if (rank_ == kMPIMasterRank) {
    std::cout << " Configuration rescue completed: " << (mpi_size_ - num_valid) << "/" << mpi_size_ 
              << " processes rescued from rank " << source_rank << std::endl;
  }
}

/**
 * @brief Normalize TPS tensor so that the amplitude of the wave function is order 1. 
 * This normalization is safer for Monte-Carlo based calculations.
 * 
 * The TPSs across all ranks are gathered and the maximum absolute value is found.
 * All the TPSs across all ranks are then scaled by the scale factor 1 / max_abs uniformly.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires
MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
void MonteCarloPEPSBaseExecutor<TenElemT,
                                QNT,
                                MonteCarloSweepUpdater>::NormTPSForOrder1Amplitude_() {
  std::unique_ptr<TenElemT[]> gather_amplitude;
  if (rank_ == kMPIMasterRank) {
    gather_amplitude = std::make_unique<TenElemT[]>(mpi_size_);
  }
  HANDLE_MPI_ERROR(::MPI_Gather(&tps_sample_.amplitude,
                                1,
                                hp_numeric::GetMPIDataType<TenElemT>(),
                                gather_amplitude.get(),
                                1,
                                hp_numeric::GetMPIDataType<TenElemT>(),
                                kMPIMasterRank,
                                comm_));
  double scale_factor;
  if (rank_ == kMPIMasterRank) {
    std::cout << std::endl;
    auto max_it = std::max_element(gather_amplitude.get(), gather_amplitude.get() + mpi_size_,
                                   [](const TenElemT &a, const TenElemT &b) {
                                     return std::abs(a) < std::abs(b);
                                   });
    double max_abs = std::abs(*max_it);
    scale_factor = 1.0 / max_abs;
  }
  HANDLE_MPI_ERROR(::MPI_Bcast(&scale_factor, 1, MPI_DOUBLE, kMPIMasterRank, comm_));
  double scale_factor_on_site = std::pow(scale_factor, 1.0 / double(lx_ * ly_));
  split_index_tps_ *= scale_factor_on_site;
  Configuration config = tps_sample_.config;
  tps_sample_ = WaveFunctionComponentT(split_index_tps_, config, tps_sample_.trun_para);
  //print the normalization info
  if (rank_ == kMPIMasterRank) {
    std::cout << "Normalization TPS For Order 1 Amplitude info: " << std::endl;
    std::cout << "Overall scale factor: " << scale_factor << std::endl;
    std::cout << "Scale factor on site: " << scale_factor_on_site << std::endl;
    std::cout << "TPS sample amplitude (rank " << rank_ << "): " << tps_sample_.amplitude << std::endl;
  }
  }

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
void MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::EnsureDirectoryExists_(const std::string& file_path) {
  if (file_path.empty()) {
    if (rank_ == kMPIMasterRank) {
      std::cerr << "Warning: File path is empty. No directory will be created." << std::endl;
    }
    return;
  }
  
  // Extract directory from path
  std::filesystem::path parent_dir = std::filesystem::path(file_path).parent_path();
  if (parent_dir.empty()) return;
  
  if (rank_ == kMPIMasterRank) {
    try {
      if (!std::filesystem::exists(parent_dir)) {
        std::cout << "Creating directory: " << parent_dir.string() << std::endl;
        std::filesystem::create_directories(parent_dir);
      }
    } catch (const std::exception& e) {
      std::cerr << "Error creating directory " << parent_dir.string() << ": " << e.what() << std::endl;
    }
  }
  // Always barrier for safety - all processes must wait
  MPI_Barrier(comm_);
}

// Constructor implementations
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::MonteCarloPEPSBaseExecutor(
    const SITPST &sitps,
    const MonteCarloParams &monte_carlo_params,
    const PEPSParams &peps_params,
    const MPI_Comm &comm) :
    split_index_tps_(sitps),
    lx_(sitps.cols()),
    ly_(sitps.rows()),
    tps_sample_(sitps.rows(), sitps.cols(), peps_params.truncate_para),
    monte_carlo_params_(monte_carlo_params),
    u_double_(0, 1),
    warm_up_(monte_carlo_params.is_warmed_up),
    comm_(comm) {
  MPI_SetUp_();
  Initialize();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater>
requires MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT>
std::unique_ptr<MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>>
MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::CreateByLoadingTPS(
    const std::string& tps_path,
    const MonteCarloParams& monte_carlo_params,
    const PEPSParams& peps_params,
    const MPI_Comm& comm) {
  
  // Load TPS from file path with proper error handling
  SITPST loaded_tps(monte_carlo_params.initial_config.rows(), monte_carlo_params.initial_config.cols());
  if (!loaded_tps.Load(tps_path)) {
    throw std::runtime_error("Failed to load TPS from path: " + tps_path);
  }
  
  // Create executor using the primary constructor with loaded TPS
  return std::make_unique<MonteCarloPEPSBaseExecutor>(
      loaded_tps, monte_carlo_params, peps_params, comm);
}

}//qlpeps
#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_PEPS_BASE_H
