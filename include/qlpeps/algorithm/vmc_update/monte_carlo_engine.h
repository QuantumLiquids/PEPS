/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-23
*
* Description: QuantumLiquids/PEPS project. MonteCarloEngine - compositional core for MC sampling on PEPS
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_ENGINE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_ENGINE_H

#include <iomanip>
#include <filesystem>
#include <random>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include "qlpeps/two_dim_tn/tps/tps.h"                            // TPS
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"                // SplitIndexTPS
#include "qlpeps/two_dim_tn/common/boundary_condition.h"          // BoundaryCondition
#include "qlpeps/vmc_basic/wave_function_component.h"             // TPSWaveFunctionComponent
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"  // MonteCarloParams, PEPSParams
#include "qlpeps/two_dim_tn/tensor_network_2d/trg/trg_contractor.h" // TRGContractor
#include "qlten/utility/timer.h"
#include "qlpeps/utility/helpers.h"

namespace qlpeps {

template<typename T, typename TenElemT, typename QNT, template<typename, typename> class ContractorT>
concept MonteCarloSweepUpdaterConcept = requires(
    T updater,
    const SplitIndexTPS<TenElemT, QNT> &sitps,
    TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, ContractorT> &component,
    std::vector<double> &accept_ratios
) {
  { updater(sitps, component, accept_ratios) } -> std::same_as<void>;
};

/**
 * @class MonteCarloEngine
 * @brief Compositional Monte Carlo sampling engine for TPS（in form of SplitIndexTPS) without model logic.
 *
 * Responsibilities:
 * - Own the current tensor state `SplitIndexTPS` and its `TPSWaveFunctionComponent` sample.
 * - Provide sampling primitives `WarmUp()` and `StepSweep()` via an injected MonteCarloSweepUpdater.
 * - Provide numeric stability utilities such as `NormalizeStateOrder1()` to scale global wavefunction and amplitude.
 * - Provide MPI-safe helpers for directory creation and informative runtime prints.
 *
 * Non-responsibilities:
 * - It does NOT broadcast states across MPI ranks. The evaluator (energy/measurement) is the single owner
 *   of broadcasting semantics.
 * - It does NOT contain model Hamiltonian logic (energy/observables). Those belong to model solvers.
 *
 * MPI semantics:
 * - Stores `MPI_Comm` and rank/size metadata.
 * - Uses collective ops internally only for configuration validation and normalization routines.
 *
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 * @tparam MonteCarloSweepUpdater Functor that performs one MC sweep update with signature:
 *   void operator()(const SplitIndexTPS<TenElemT,QNT>&, TPSWaveFunctionComponent<TenElemT,QNT>&, std::vector<double>&)
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater,
         template<typename, typename> class ContractorT = BMPSContractor>
requires MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT, ContractorT>
class MonteCarloEngine {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using WaveFunctionComponentT = TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, ContractorT>;
  using TruncateParamsT = typename WaveFunctionComponentT::TruncateParams;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;

  MonteCarloEngine(const SITPST &sitps,
                   const MonteCarloParams &monte_carlo_params,
                   const PEPSParams &peps_params,
                   const MPI_Comm &comm,
                   MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater(),
                   const ConfigurationRescueParams &config_rescue = ConfigurationRescueParams())
      : split_index_tps_(sitps),
        lx_(sitps.cols()),
        ly_(sitps.rows()),
        // Select truncate params in a BC/contractor-aware way so we throw a stable
        // std::invalid_argument for mismatched OBC/PBC backends before any variant access throws.
        tps_sample_(sitps.rows(), sitps.cols(), SelectTruncateParams_(sitps, peps_params)),
        comm_(comm),
        mc_sweep_updater_(std::move(mc_updater)),
        u_double_(0, 1),
        monte_carlo_params_(monte_carlo_params),
        warm_up_(monte_carlo_params.is_warmed_up),
        config_rescue_(config_rescue) {
    MPI_SetUp_();
    samples_per_rank_ = (monte_carlo_params_.total_samples + static_cast<size_t>(mpi_size_) - 1)
                        / static_cast<size_t>(mpi_size_);
    Initialize_();
  }

  /**
   * @name Accessors
   * @{ */
  const SITPST &State() const { return split_index_tps_; }
  SITPST &State() { return split_index_tps_; }
  const WaveFunctionComponentT &WavefuncComp() const { return tps_sample_; }
  WaveFunctionComponentT &WavefuncComp() { return tps_sample_; }
  size_t Lx() const { return lx_; }
  size_t Ly() const { return ly_; }
  const MPI_Comm &Comm() const { return comm_; }
  int Rank() const { return rank_; }
  int MpiSize() const { return mpi_size_; }
  const MonteCarloParams &MCParams() const { return monte_carlo_params_; }
  /// Samples per MPI rank: ceil(total_samples / mpi_size)
  size_t SamplesPerRank() const { return samples_per_rank_; }
  /// Total samples across all MPI ranks (from params)
  size_t TotalSamples() const { return monte_carlo_params_.total_samples; }
  /** @} */

  /**
   * @brief Assign a new split-index TPS state.
   * @param state New state to take ownership.
   */
  void AssignState(const SITPST &state) {
    split_index_tps_ = state;
  }

  /**
   * @brief Refresh `TPSWaveFunctionComponent` after the state or configuration changes.
   *        Keeps truncation parameters unchanged and preserves the current configuration.
   */
  void RefreshWavefunctionComponent() {
    Configuration config = tps_sample_.config;
    tps_sample_ = WaveFunctionComponentT(split_index_tps_, config, tps_sample_.trun_para);
  }

  /**
   * @brief Warm up the Markov chain by performing the configured number of sweeps.
   *        Ensures the amplitude sanity and normalizes the state to O(1) amplitude across MPI after warm up.
   * @return 0 on success; aborts MPI on failure.
   */
  int WarmUp() {
    if (!warm_up_) {
      Timer warm_up_timer("proc " + std::to_string(rank_) + " warm up");
      for (size_t sweep = 0; sweep < monte_carlo_params_.num_warmup_sweeps; sweep++) {
        auto accept_rates = StepSweep(1);
        (void)accept_rates;
      }
      warm_up_timer.PrintElapsed();
      warm_up_ = true;
    }
    bool psi_legal = CheckWaveFunctionAmplitudeValidity(
        tps_sample_,
        config_rescue_.amplitude_min_threshold,
        config_rescue_.amplitude_max_threshold);
    if (!psi_legal) {
      std::cout << "Proc " << rank_
                << ", psi : " << std::scientific << tps_sample_.amplitude
                << " Amplitude is still not legal after warm up.  Terminate the program" << std::endl;
      MPI_Abort(comm_, EXIT_FAILURE);
      return 1;
    }
    NormalizeStateOrder1();
    return 0;
  }

  /**
   * @brief Perform a number of MC sweeps and return per-sweep acceptance statistics.
   * @param sweeps_between_samples Number of consecutive sweeps to perform.
   * @return Vector of acceptance ratios collected during updates.
   */
  std::vector<double> StepSweep(const size_t sweeps_between_samples) {
    std::vector<double> accept_rates;
#ifdef QLPEPS_TIMING_MODE
    Timer mc_sweep_timer("monte_carlo_sweep (rank " + std::to_string(rank_) + ")");
#endif
    for (size_t i = 0; i < sweeps_between_samples; i++) {
      mc_sweep_updater_(split_index_tps_, tps_sample_, accept_rates);
    }
#ifdef QLPEPS_TIMING_MODE
    mc_sweep_timer.PrintElapsed();
#endif
    return accept_rates;
  }

  /**
   * @brief Perform default-count sweeps as specified by MonteCarloParams::sweeps_between_samples.
   */
  std::vector<double> StepSweep() {
    return StepSweep(monte_carlo_params_.sweeps_between_samples);
  }

  /**
   * @brief Normalize the global amplitude to order 1 across all MPI ranks.
   *        Scales each site tensor uniformly so that the maximum amplitude over ranks becomes 1.
   *        This setup is better for numeric stability.
   */
  void NormalizeStateOrder1() {
    std::unique_ptr<TenElemT[]> gather_amplitude;
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      gather_amplitude = std::make_unique<TenElemT[]>(mpi_size_);
    }
    HANDLE_MPI_ERROR(::MPI_Gather(&tps_sample_.amplitude,
                                  1,
                                  hp_numeric::GetMPIDataType<TenElemT>(),
                                  gather_amplitude.get(),
                                  1,
                                  hp_numeric::GetMPIDataType<TenElemT>(),
                                  qlten::hp_numeric::kMPIMasterRank,
                                  comm_));
    double scale_factor;
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      std::cout << std::endl;
      auto max_it = std::max_element(gather_amplitude.get(), gather_amplitude.get() + mpi_size_,
                                     [](const TenElemT &a, const TenElemT &b) {
                                       return std::abs(a) < std::abs(b);
                                     });
      double max_abs = std::abs(*max_it);
      scale_factor = 1.0 / max_abs;
    }
    HANDLE_MPI_ERROR(::MPI_Bcast(&scale_factor, 1, MPI_DOUBLE, qlten::hp_numeric::kMPIMasterRank, comm_));
    double scale_factor_on_site = std::pow(scale_factor, 1.0 / double(lx_ * ly_));
    split_index_tps_ *= scale_factor_on_site;
    Configuration config = tps_sample_.config;
    tps_sample_ = WaveFunctionComponentT(split_index_tps_, config, tps_sample_.trun_para);
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      std::cout << "Normalization TPS For Order 1 Amplitude info: " << std::endl;
      std::cout << "Overall scale factor: " << scale_factor << std::endl;
      std::cout << "Scale factor on site: " << scale_factor_on_site << std::endl;
      std::cout << "TPS sample amplitude (rank " << rank_ << "): " << tps_sample_.amplitude << std::endl;
    }
  }

  /**
   * @brief Ensure parent directory of the given file path exists using rank-0 creation and MPI barrier.
   * @param file_path Intended file path whose parent directory will be created if missing.
   */
  void EnsureDirectoryExists(const std::string &file_path) const {
    if (file_path.empty()) {
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        std::cerr << "Warning: File path is empty. No directory will be created." << std::endl;
      }
      return;
    }
    std::filesystem::path parent_dir = std::filesystem::path(file_path).parent_path();
    if (parent_dir.empty()) return;
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      try {
        if (!std::filesystem::exists(parent_dir)) {
          std::cout << "Creating directory: " << parent_dir.string() << std::endl;
          std::filesystem::create_directories(parent_dir);
        }
      } catch (const std::exception& e) {
        std::cerr << "Error creating directory " << parent_dir.string() << ": " << e.what() << std::endl;
      }
    }
    MPI_Barrier(comm_);
  }

  /**
   * @brief Print high-level run information (system size, bond dimensions, MC parameters) on master rank.
   * @param header Title header.
   */
  void PrintCommonInfo(const std::string &header) const {
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      const size_t indent = 40;
      std::cout << std::left;
      std::cout << "\n";
      std::cout << "=====> " << header << " <=====" << "\n";
      std::cout << std::setw(indent) << "System size (lx, ly):" << "(" << lx_ << ", " << ly_ << ")\n";
      const std::string bc_str =
          (split_index_tps_.GetBoundaryCondition() == BoundaryCondition::Periodic) ? "Periodic" : "Open";
      std::cout << std::setw(indent) << "Boundary condition:" << bc_str << "\n";
      std::cout << std::setw(indent) << "PEPS bond dimension:" << split_index_tps_.GetMaxBondDimension() << "\n";
      if constexpr (std::is_same_v<ContractorT<TenElemT, QNT>, TRGContractor<TenElemT, QNT>>) {
        std::cout << std::setw(indent) << "TRG bond dimension:"
                  << tps_sample_.trun_para.D_min << "/" << tps_sample_.trun_para.D_max << "\n";
        std::cout << std::setw(indent) << "TRG trunc_err:" << tps_sample_.trun_para.trunc_err << "\n";
        std::cout << std::setw(indent) << "TRG inv_relative_eps:" << tps_sample_.trun_para.inv_relative_eps << "\n";
      } else {
        std::cout << std::setw(indent) << "BMPS bond dimension:" << tps_sample_.trun_para.D_min
                  << "/" << tps_sample_.trun_para.D_max << "\n";
        std::cout << std::setw(indent) << "BMPS Truncate Scheme:"
                  << static_cast<int>(tps_sample_.trun_para.compress_scheme) << "\n";
      }
      std::cout << std::setw(indent) << "Total samples:" << monte_carlo_params_.total_samples << "\n";
      std::cout << std::setw(indent) << "Samples per rank (ceil "
                << monte_carlo_params_.total_samples << " / " << mpi_size_ << " ranks):"
                << SamplesPerRank() << "\n";
      std::cout << std::setw(indent) << "Warm-up sweeps:" << monte_carlo_params_.num_warmup_sweeps << "\n";
      std::cout << std::setw(indent) << "Initial config warmed up:" << (monte_carlo_params_.is_warmed_up ? "yes" : "no")
                << "\n";
      std::cout << std::setw(indent) << "Monte Carlo sweep repeat times:" << monte_carlo_params_.sweeps_between_samples
                << "\n";
      if (!monte_carlo_params_.config_dump_path.empty()) {
        std::cout << std::setw(indent) << "Config dump path:" << monte_carlo_params_.config_dump_path << "\n";
      }
      std::cout << std::setw(indent) << "Configuration rescue:"
                << (config_rescue_.enabled ? "enabled" : "disabled") << "\n";
      if (config_rescue_.enabled) {
        std::cout << std::setw(indent) << "Rescue amplitude min/max:"
                  << config_rescue_.amplitude_min_threshold << " / " << config_rescue_.amplitude_max_threshold << "\n";
      }
    }
  }

  /**
   * @brief Print technical runtime info (MPI size, thread count) on master rank.
   */
  void PrintTechInfo() const {
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      const size_t indent = 40;
      std::cout << std::left;
      std::cout << "\n";
      std::cout << "=====> TECHNICAL PARAMETERS <=====" << "\n";
      std::cout << std::setw(indent) << "The number of processors (including master):" << mpi_size_ << "\n";
      std::cout << std::setw(indent) << "The number of threads per processor:"
                << hp_numeric::GetTensorManipulationThreads() << "\n";
    }
  }

  /**
   * @brief Validate and rescue configurations across MPI ranks.
   *
   * Each rank checks both construction success (init_valid_) and amplitude legality.
   * If some ranks are invalid, a valid configuration is broadcast from a healthy rank
   * and replaced locally; those ranks are marked as not warmed up.
   *
   * Rescue can be disabled via runtime_params_.config_rescue.enabled.
   * When disabled, invalid configurations cause immediate abort.
   */
  void EnsureConfigurationValidity() {
    // Validity requires both successful construction and legal amplitude
    int local_valid = (init_valid_ && CheckWaveFunctionAmplitudeValidity(
        tps_sample_,
        config_rescue_.amplitude_min_threshold,
        config_rescue_.amplitude_max_threshold)) ? 1 : 0;
    std::vector<int> global_valid(mpi_size_);
    HANDLE_MPI_ERROR(MPI_Allgather(&local_valid, 1, MPI_INT,
                                   global_valid.data(), 1, MPI_INT, comm_));
    int num_valid = std::accumulate(global_valid.begin(), global_valid.end(), 0);
    if (num_valid == mpi_size_) {
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        std::cout << "\u2713 Configuration validation: All " << mpi_size_
                  << " processes have valid configurations." << std::endl;
      }
      return;
    }

    // Check if rescue is enabled
    if (!config_rescue_.enabled) {
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        std::cerr << "\n=== CONFIGURATION FAILURE ===\n"
                  << (mpi_size_ - num_valid) << "/" << mpi_size_ << " processes have invalid configurations.\n"
                  << "Configuration rescue is DISABLED. Aborting.\n";
      }
      std::ostringstream oss;
      oss << "Rank " << rank_ << ": valid=" << local_valid
          << ", init_valid=" << init_valid_
          << ", amplitude=" << (init_valid_ ? std::abs(tps_sample_.amplitude) : 0.0) << std::endl;
      hp_numeric::GatherAndPrintErrorMessages(oss.str(), comm_);
      MPI_Abort(comm_, EXIT_FAILURE);
    }

    auto valid_iter = std::find(global_valid.begin(), global_valid.end(), 1);
    if (valid_iter == global_valid.end()) {
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        std::cerr << "\n=== CRITICAL CONFIGURATION FAILURE ===\n"
                  << "All " << mpi_size_ << " processes have invalid configurations!\n"
                  << "Check: bond dimension, truncation cutoff, initial configuration\n";
      }
      std::ostringstream oss;
      oss << "Rank " << rank_ << ": init_valid=" << init_valid_
          << ", amplitude=" << (init_valid_ ? tps_sample_.amplitude : TenElemT(0))
          << ", magnitude=" << (init_valid_ ? std::abs(tps_sample_.amplitude) : 0.0) << std::endl;
      hp_numeric::GatherAndPrintErrorMessages(oss.str(), comm_);
      MPI_Abort(comm_, EXIT_FAILURE);
    }

    int source_rank = static_cast<int>(valid_iter - global_valid.begin());
    Configuration config_valid(ly_, lx_);
    if (rank_ == source_rank) {
      config_valid = tps_sample_.config;
    }
    MPI_BCast(config_valid, source_rank, comm_);

    if (local_valid == 0) {
      bool rescue_success = TryConstructWavefunction_(config_valid);
      if (!rescue_success) {
        // Even the valid configuration from another rank failed here
        // This indicates a fundamental problem (e.g., TPS incompatible with local truncation params)
        std::cerr << "Rank " << rank_ << ": rescue FAILED - construction threw exception even with valid config from rank "
                  << source_rank << ". This may indicate TPS or truncation parameter issues." << std::endl;
        MPI_Abort(comm_, EXIT_FAILURE);
      }
      init_valid_ = true;
      warm_up_ = false;
      std::cout << "Rank " << rank_ << ": rescued from rank " << source_rank
                << " (new amplitude: " << std::abs(tps_sample_.amplitude) << ")" << std::endl;
    }

    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      std::cout << "\u2713 Configuration rescue completed: " << (mpi_size_ - num_valid) << "/" << mpi_size_
                << " processes rescued from rank " << source_rank << std::endl;
    }
  }

 private:
  void MPI_SetUp_() {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &mpi_size_);
  }

  void Initialize_() {
    ValidateBoundaryCondition_();
    init_valid_ = TryConstructWavefunction_(monte_carlo_params_.initial_config);
    EnsureConfigurationValidity();
    NormalizeStateOrder1();
  }

  /**
   * @brief Attempt to construct WaveFunctionComponent with given configuration.
   * @param config Configuration to use.
   * @return true if construction succeeded; false if an exception was caught.
   */
  bool TryConstructWavefunction_(const Configuration &config) {
    try {
      tps_sample_ = WaveFunctionComponentT(split_index_tps_, config, tps_sample_.trun_para);
      return true;
    } catch (const std::runtime_error &e) {
      // Catch Empty tensor exceptions from BMPS contraction
      std::cerr << "Rank " << rank_ << ": WaveFunctionComponent construction failed: " << e.what() << std::endl;
      return false;
    }
  }

  static bool IsTRGSupportedLinearSize_(size_t n) {
    if (n == 0) return false;
    while ((n % 2) == 0) {
      n /= 2;
    }
    return (n == 1) || (n == 3);
  }

  static TruncateParamsT SelectTruncateParams_(const SITPST &sitps, const PEPSParams &peps_params) {
    const BoundaryCondition bc = sitps.GetBoundaryCondition();
    if constexpr (std::is_same_v<ContractorT<TenElemT, QNT>, TRGContractor<TenElemT, QNT>>) {
      if (bc != BoundaryCondition::Periodic) {
        throw std::invalid_argument("MonteCarloEngine: TRGContractor requires BoundaryCondition::Periodic.");
      }
      if (!peps_params.IsPBC()) {
        throw std::invalid_argument("MonteCarloEngine: TRGContractor requires TRGTruncateParams in PEPSParams.");
      }
      return peps_params.GetTRGParams();
    } else {
      if (bc == BoundaryCondition::Periodic) {
        throw std::invalid_argument("MonteCarloEngine: PBC requires TRGContractor in the current implementation.");
      }
      if (!peps_params.IsOBC()) {
        throw std::invalid_argument("MonteCarloEngine: BMPSContractor requires BMPSTruncateParams in PEPSParams.");
      }
      return peps_params.GetBMPSParams();
    }
  }

  void ValidateBoundaryCondition_() const {
    const BoundaryCondition bc = split_index_tps_.GetBoundaryCondition();
    if constexpr (std::is_same_v<ContractorT<TenElemT, QNT>, TRGContractor<TenElemT, QNT>>) {
      if (bc != BoundaryCondition::Periodic) {
        throw std::invalid_argument("MonteCarloEngine: TRGContractor requires BoundaryCondition::Periodic.");
      }
      if (split_index_tps_.rows() != split_index_tps_.cols()) {
        throw std::invalid_argument("MonteCarloEngine: TRGContractor requires a square lattice for PBC.");
      }
      if (!IsTRGSupportedLinearSize_(split_index_tps_.rows())) {
        throw std::invalid_argument("MonteCarloEngine: TRGContractor requires L=2^k or L=3*2^k.");
      }
      if constexpr (Tensor::IsFermionic()) {
        throw std::invalid_argument("MonteCarloEngine: TRGContractor does not support fermionic tensors.");
      }
    } else {
      if (bc == BoundaryCondition::Periodic) {
        throw std::invalid_argument("MonteCarloEngine: PBC requires TRGContractor in the current implementation.");
      }
    }
  }

 private:
  SITPST split_index_tps_;
  size_t lx_;
  size_t ly_;
  WaveFunctionComponentT tps_sample_;
  const MPI_Comm &comm_;
  int rank_;
  int mpi_size_;
  MonteCarloSweepUpdater mc_sweep_updater_;
  std::uniform_real_distribution<double> u_double_;
  MonteCarloParams monte_carlo_params_;
  size_t samples_per_rank_;  ///< ceil(total_samples / mpi_size)
  bool warm_up_;
  ConfigurationRescueParams config_rescue_;
  bool init_valid_ = true;  ///< Whether initial WaveFunctionComponent construction succeeded
};

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_ENGINE_H
