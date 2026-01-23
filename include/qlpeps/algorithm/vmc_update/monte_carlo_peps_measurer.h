/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-11-02
*
* Description: QuantumLiquids/PEPS project.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_H

#include "qlpeps/two_dim_tn/tps/tps.h"                            // TPS data structure
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"                // SplitIndexTPS state
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"  // MCMeasurementParams
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // Observable registry API
#include "qlpeps/vmc_basic/monte_carlo_tools/statistics.h"        // Mean, Variance, DumpVecData, ...
#include "qlpeps/utility/helpers.h"                               // Real helpers
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_engine.h"
#include "qlpeps/base/mpi_signal_guard.h"
#include <complex>
#include <unordered_map>
#include <optional>
#include <string>


namespace qlpeps {

// Default cap for the number of lags returned by autocorrelation helpers
inline constexpr size_t kAutocorrMaxLagCap = 20;

// Conjugation helper that preserves the input domain type.
// For real numbers, returns the value itself. For complex numbers, returns std::conj.
template<typename T>
inline T Conj(const T &v) { return v; }
template<typename T>
inline std::complex<T> Conj(const std::complex<T> &v) { return std::conj(v); }

///< Sum over element-wise product of two spin configurations
template<typename ElemT>
ElemT SpinConfigurationOverlap(
    const std::vector<ElemT> &sz1,
    const std::vector<ElemT> &sz2
) {
  ElemT overlap(0);
  for (size_t i = 0; i < sz1.size(); i++) {
    overlap += Conj(sz1[i]) * sz2[i];
  }
  return overlap;
}

template<typename T>
std::vector<T> ComputeAutocorrelation(
    const std::vector<T> &data,
    const T mean
) {
  const size_t num_samples = data.size();
  const size_t num_lags = kAutocorrMaxLagCap > num_samples / 2 ? num_samples / 2 : kAutocorrMaxLagCap;
  std::vector<T> result(num_lags, T(0));
  for (size_t tau = 0; tau < num_lags; ++tau) {
    T sum(0);
    for (size_t j = 0; j < num_samples - tau; ++j) {
      sum += Conj(data[j]) * data[j + tau];
    }
    result[tau] = sum / static_cast<double>(num_samples - tau) - mean * Conj(mean);
  }
  return result;
}

/**
 * Calculate the site-averaged auto-correlation for a site-local observable (e.g., spin or charge)
 * from per-sample configurations. The observable at site i and MC step t is denoted as O_i(t).
 *
 * Definition (with per-site mean removal):
 *   C(tau) = (1/N) * Sum_i [ < (O_i(t) - mu_i) * conj(O_i(t+tau) - mu_i) >_t ]
 * where mu_i = < O_i(t) >_t is the time average at site i over the local samples.
 *
 * This function makes no assumption such as <O_i> = 0.
 */
template<typename ElemT>
std::vector<ElemT> ComputeSiteAveragedAutocorrelation(
    const std::vector<std::vector<ElemT>> &site_local_samples
) {
  const size_t num_samples = site_local_samples.size();
  const size_t num_lags = kAutocorrMaxLagCap > num_samples / 2 ? num_samples / 2 : kAutocorrMaxLagCap;
  const size_t num_sites = site_local_samples[0].size(); // lattice size
  std::vector<ElemT> result(num_lags, ElemT(0));

  // Compute per-site time mean mu_i
  std::vector<ElemT> per_site_mean(num_sites, ElemT(0));
  for (size_t j = 0; j < num_samples; ++j) {
    const auto &frame = site_local_samples[j];
    for (size_t i = 0; i < num_sites; ++i) {
      per_site_mean[i] += frame[i];
    }
  }
  for (size_t i = 0; i < num_sites; ++i) {
    per_site_mean[i] = per_site_mean[i] / static_cast<double>(num_samples);
  }

  for (size_t tau = 0; tau < num_lags; ++tau) {
    ElemT overlap_sum(0);
    for (size_t j = 0; j < num_samples - tau; ++j) {
      const auto &t0 = site_local_samples[j];
      const auto &tt = site_local_samples[j + tau];
      for (size_t i = 0; i < num_sites; ++i) {
        overlap_sum += Conj(t0[i] - per_site_mean[i]) * (tt[i] - per_site_mean[i]);
      }
    }
    result[tau] = overlap_sum / static_cast<double>(num_samples - tau) / static_cast<double>(num_sites);
  }
  return result;
}

void PrintProgressBar(size_t progress, size_t total) {
  size_t bar_width = 70; // width of the progress bar

  std::cout << "[";
  size_t pos = bar_width * progress / total;
  for (size_t i = 0; i < bar_width; ++i) {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "] " << int((double) progress * 100.0 / (double) total) << " %" << std::endl;
}

/**
      * @brief Dumps sample_data to CSV (utility for simple vectors).
      *
      * Each row corresponds to a single sample; columns are elements of the inner vector.
      * Use registry-aware DumpStats* helpers for observable statistics instead.
      *
      * @throws std::ios_base::failure If the file cannot be opened for writing.
      */
template<typename TenElemT>
void DumpSampleData(const std::vector<std::vector<TenElemT>> sample_data, const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::ios_base::failure("Failed to open file: " + filename);
  }

  // Write the sample_data to the file
  for (const auto &sample : sample_data) {
    for (size_t i = 0; i < sample.size(); ++i) {
      file << sample[i];
      if (i + 1 < sample.size()) {
        file << ","; // Add a comma between elements
      }
    }
    file << "\n"; // Newline for the next sample
  }

  file.close();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT = BMPSContractor>
class MCPEPSMeasurer : public qlten::Executor {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;

  /**
   * @brief Constructor with explicit TPS provided by user.
   * 
   * User provides all data explicitly - no hidden file loading.
   * This constructor gives users complete control over the input data.
   * 
   * @param sitps Split-index TPS provided by user
   * @param measurement_params Unified measurement parameters (MC + PEPS + dump path)
   * @param comm MPI communicator
   * @param solver Model measurement solver for observables
   */
  MCPEPSMeasurer(const SITPST &sitps,
                 const MCMeasurementParams &measurement_params,
                 const MPI_Comm &comm,
                 const MeasurementSolver &solver = MeasurementSolver(),
                 MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater());

  /**
   * @brief Static factory function to create measurement executor by loading TPS from file path.
   * 
   * Convenience factory for users who have TPS data stored on disk.
   * This is the recommended approach when TPS data is saved from previous calculations.
   * The lattice dimensions (ly, lx) are automatically inferred from the
   * initial configuration size: ly = initial_config.rows(), lx = initial_config.cols().
   * 
   * @param tps_path Path to TPS data files on disk
   * @param measurement_params Unified measurement parameters (must contain valid initial_config)
   * @param comm MPI communicator
   * @param solver Model measurement solver for observables
   * @return Unique pointer to the created measurement executor
   * 
   * @note The initial_config in measurement_params.mc_params must be properly sized to determine lattice dimensions
   * @note This factory automatically loads TPS from disk and initializes the measurement system
   * 
   * Usage:
   *   auto executor = MCPEPSMeasurer::CreateByLoadingTPS(tps_path, measurement_params, comm, solver);
   */
  static std::unique_ptr<MCPEPSMeasurer>
  CreateByLoadingTPS(const std::string& tps_path,
                     const MCMeasurementParams& measurement_params,
                     const MPI_Comm& comm,
                     const MeasurementSolver& solver = MeasurementSolver(),
                     MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater());

  void Execute(void) override;

  void ReplicaTest(std::function<double(const Configuration &, const Configuration &)>); // for check the ergodicity

  /**
   * @brief Dump aggregated statistics for registered observables.
   *
   * Writes per-key CSV files under the configured measurement dump path (stats/<key>.csv),
   * with columns: index, mean, stderr. Shapes and index semantics are provided by
   * DescribeObservables(). For pair observables using upper-triangular packing, an auxiliary
   * index map file is emitted.
   */
  void DumpData();

  /**
   * @brief Dump aggregated statistics to the specified directory.
   */
  void DumpData(const std::string &measurement_data_path);

  MCMeasurementParams mc_measure_params;

  struct EnergyEstimate {
    TenElemT energy;
    double stderr;
  };

  std::pair<TenElemT, double> OutputEnergy() const;

  std::optional<EnergyEstimate> GetEnergyEstimate() const;

  /**
   * @brief Get the current configuration of the TPS sample
   * @return Current configuration
   */
  const Configuration& GetCurrentConfiguration() const {
    return engine_.WavefuncComp().config;
  }

  const auto &ObservableRegistry() const { return registry_stats_; }
 private:
  void ReserveSamplesData_();

  void PrintExecutorInfo_(void);

  void Measure_(void);

  void MeasureSample_(void);

  void GatherStatistic_(void);

  void SynchronizeConfiguration_(const size_t root = 0); //for the replica test

  // Friendly stats dump helpers (text files)
  void DumpStatsMatrix_(const std::string &dir,
                        const std::string &key,
                        const std::vector<TenElemT> &vals,
                        const std::vector<double> &errs,
                        size_t rows,
                        size_t cols) const;
  void DumpStatsFlat_(const std::string &dir,
                      const std::string &key,
                      const std::vector<TenElemT> &vals,
                      const std::vector<double> &errs) const;
  void DumpStatsFlatReal_(const std::string &dir,
                          const std::string &key,
                          const std::vector<double> &vals,
                          const std::vector<double> &errs) const;
  void DumpPackedUpperTriIndexMap_(const std::string &dir,
                                   const std::string &key,
                                   size_t packed_len) const;

  // Compute psi(S) consistency: return pair (psi_mean, psi_rel_err)
  std::pair<TenElemT, double> ComputePsiConsistencyRelErr_(const std::vector<TenElemT> &psi_list) const;

  // registry aggregated stats keyed by observable name
  std::unordered_map<std::string, std::pair<std::vector<TenElemT>, std::vector<double>>> registry_stats_;

  MeasurementSolver measurement_solver_;
  MonteCarloEngine<TenElemT, QNT, MonteCarloSweepUpdater, ContractorT> engine_;
  std::vector<ObservableMeta> observables_meta_;

  /**
   * Record the statistic inside `bin_size`
   */
  // observable
  struct SampleData {
    void Reserve(const size_t sample_num) { (void)sample_num; }

    void Clear() {}

    

    // registry samples: key -> list of flat values (per-sample)
    std::unordered_map<std::string, std::vector<std::vector<TenElemT>>> registry_samples;

    // New path: push values from registry map (keyed)
    void PushBackRegistry(TenElemT /*wave_function_amplitude*/, const ObservableMap<TenElemT>& obs_map) {
      for (const auto &kv : obs_map) {
        registry_samples[kv.first].push_back(kv.second);
        
      }
    }

    // Compute element-wise mean and naive standard error (no autocorr) for registry keys within one rank
    std::unordered_map<std::string, std::pair<std::vector<TenElemT>, std::vector<double>>> StatisticRegistry() const {
      std::unordered_map<std::string, std::pair<std::vector<TenElemT>, std::vector<double>>> out;
      for (const auto &kv : registry_samples) {
        const auto &samples = kv.second; // vector<flat>
        if (samples.empty()) { continue; }
        const size_t vec_len = samples[0].size();
        std::vector<TenElemT> mean(vec_len, TenElemT(0));
        std::vector<double> stderr(vec_len, 0.0);
        const size_t S = samples.size();
        // mean
        for (size_t s = 0; s < S; ++s) {
          for (size_t i = 0; i < vec_len; ++i) { mean[i] += samples[s][i]; }
        }
        for (size_t i = 0; i < vec_len; ++i) { mean[i] = mean[i] / static_cast<double>(S); }
        // stderr (naive)
        for (size_t i = 0; i < vec_len; ++i) {
          double var_acc = 0.0;
          for (size_t s = 0; s < S; ++s) {
            auto diff = samples[s][i] - mean[i];
            var_acc += std::norm(diff);
          }
          double var = (S > 0) ? (var_acc / static_cast<double>(S)) : 0.0;
          stderr[i] = (S > 1) ? std::sqrt(var / static_cast<double>(S - 1)) : std::numeric_limits<double>::infinity();
        }
        out.emplace(kv.first, std::make_pair(std::move(mean), std::move(stderr)));
      }
      return out;
    }
  } sample_data_;

  // Psi sample tuple: (psi_mean, psi_rel_err)
  std::vector<std::pair<TenElemT, double>> psi_samples_;

  std::optional<EnergyEstimate> QueryEnergyEstimate_() const;

  // Dump per-sample psi summary to samples/psi.csv on master rank
  void DumpPsiSamples_(const std::string &dir) const {
    const bool is_master = engine_.Rank() == qlten::hp_numeric::kMPIMasterRank;
    const std::string samples_dir = dir + "samples/";
    engine_.EnsureDirectoryExists(samples_dir + "dummy");
    if (!is_master) return;
    std::ofstream ofs(samples_dir + "psi.csv");
    ofs << "sample_id,psi_mean_re,psi_mean_im,psi_rel_err\n";
    for (size_t i = 0; i < psi_samples_.size(); ++i) {
      ofs << i << "," << std::real(psi_samples_[i].first) << "," << std::imag(psi_samples_[i].first)
          << "," << psi_samples_[i].second << "\n";
    }
  }
};//MCPEPSMeasurer

// Explicit PBC/TRG alias for clarity at call sites.
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
using MCPEPSMeasurerPBC = MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, TRGContractor>;


}//qlpeps

// Include implementation
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer_impl.h"

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_H