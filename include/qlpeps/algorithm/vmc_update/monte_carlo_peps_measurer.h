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
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ObservablesLocal solver
#include "qlpeps/vmc_basic/monte_carlo_tools/statistics.h"        // Mean, Variance, DumpVecData, ...
#include "qlpeps/utility/helpers.h"                               // Real helpers
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_engine.h"
#include "qlpeps/base/mpi_signal_guard.h"


namespace qlpeps {



///< Sum over element-wise product of two spin configurations
template<typename ElemT>
ElemT SpinConfigurationOverlap(
    const std::vector<ElemT> &sz1,
    const std::vector<ElemT> &sz2
) {
  ElemT overlap(0);
  for (size_t i = 0; i < sz1.size(); i++) {
    overlap += sz1[i] * sz2[i];
  }
  return overlap;
}

template<typename T>
std::vector<T> CalAutoCorrelation(
    const std::vector<T> &data,
    const T mean
) {
  const size_t sample_num = data.size();
  const size_t res_len = 20 > sample_num / 2 ? sample_num / 2 : 20;
  std::vector<T> res(res_len, T(0));
  for (size_t t = 0; t < res_len; t++) {
    T sum(0);
    for (size_t j = 0; j < data.size() - t; j++) {
      sum += data[j] * data[j + t];
    }
    res[t] = sum / double(data.size() - t) - mean * mean;
  }
  return res;
}

/**
 * Calculate the spin auto-correlation from the local_sz_samples
 * The auto-correlation is defined as
 *
 * 1/N Sum_i <S_i(t)*S_i(t+delta t)> - 1/N Sum_i <S_i>^2
 *
 * Where we define <S_i> = 0. As an example, S_i = \pm 0.5 (distinct with 0/1).
 * Note to use the consistent definition as the input.
 * @param local_sz_samples
 * @return
 */
template<typename ElemT>
std::vector<ElemT> CalSpinAutoCorrelation(
    const std::vector<std::vector<ElemT>> &local_sz_samples
) {
  const size_t sample_num = local_sz_samples.size();
  const size_t res_len = 20 > sample_num / 2 ? sample_num / 2 : 20;
  const size_t N = local_sz_samples[0].size();// lattice size
  std::vector<ElemT> res(res_len, 0.0);
  for (size_t t = 0; t < res_len; t++) {
    ElemT overlap_sum(0);
    for (size_t j = 0; j < local_sz_samples.size() - t; j++) {
      overlap_sum += SpinConfigurationOverlap(local_sz_samples[j], local_sz_samples[j + t]);
    }
    res[t] = overlap_sum / (double) (local_sz_samples.size() - t) / (double) N;
  }
  return res;
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
      * @brief Dumps the sample_data to a CSV file.
      *
      * This function writes the `sample_data` data to a file in CSV format.
      * Each row in the output file corresponds to a single sample (outer vector),
      * and each column within a row corresponds to an element of the inner vector  (e.g. for one point function, corresponding to site index).
      *
      * @param filename The name of the output file. Should follow the `.csv` naming convention.
      *
      * @details
      * The output file format is designed for easy reading in MATLAB or Python.
      *
      * Example of MATLAB usage:
      * ```
      * data = csvread('output.csv');
      * ```
      *
      * Example of Python usage:
      * ```python
      * import numpy as np
      * data = np.loadtxt('output.csv', delimiter=',')
      * ```
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

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
class MCPEPSMeasurer : public qlten::Executor {
  struct Result;
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

  void DumpData();

  void DumpData(const std::string &measurement_data_path);

  MCMeasurementParams mc_measure_params;

  std::pair<TenElemT, double> OutputEnergy() const {
    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
      if (this->GetStatus() == ExecutorStatus::FINISH) {
        std::cout << "Measured energy : "
                  << std::setw(8) << res.energy
                  << pm_sign << " "
                  << std::scientific << res.en_err
                  << std::endl;
      } else {
        std::cout << "The program didn't complete the measurements. " << std::endl;
      }
    }
    return {res.energy, res.en_err};
  }

  const Result &GetMeasureResult() const {
    if (this->GetStatus() != ExecutorStatus::FINISH) {
      std::cout << "The program didn't complete the measurements. " << std::endl;
    }
    return res;
  }

  /**
   * @brief Get the current configuration of the TPS sample
   * @return Current configuration
   */
  const Configuration& GetCurrentConfiguration() const {
    return engine_.WavefuncComp().config;
  }
 private:
  void ReserveSamplesDataSpace_();

  void PrintExecutorInfo_(void);

  void Measure_(void);

  void MeasureSample_(void);

  void GatherStatistic_(void);

  void SynchronizeConfiguration_(const size_t root = 0); //for the replica test

  MeasurementSolver measurement_solver_;
  MonteCarloEngine<TenElemT, QNT, MonteCarloSweepUpdater> engine_;
  struct Result {
    TenElemT energy;
    double en_err;

    std::vector<TenElemT> bond_energys;
    std::vector<double> bond_energy_errs;
    std::vector<TenElemT> one_point_functions;
    std::vector<double> one_point_function_errs;
    std::vector<TenElemT> two_point_functions;
    std::vector<double> two_point_function_errs;

    std::vector<TenElemT> energy_auto_corr;
    std::vector<double> energy_auto_corr_err;
    std::vector<TenElemT> one_point_functions_auto_corr;
    std::vector<double> one_point_functions_auto_corr_err;

    Result(void) = default;

    ~Result() {
      return;
    }

    void Dump() const {
      std::string filename = "energy_statistics";
      std::ofstream ofs(filename, std::ofstream::binary);
      if (!ofs.is_open()) {
        throw std::ios_base::failure("Failed to open file: " + filename);
      }
      ofs.write((const char *) &energy, 1 * sizeof(TenElemT));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write energy to " + filename);
      ofs.write((const char *) &en_err, 1 * sizeof(double));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write en_err to " + filename);
      ofs.write((const char *) bond_energys.data(), bond_energys.size() * sizeof(TenElemT));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write bond_energys to " + filename);
      ofs.write((const char *) energy_auto_corr.data(), energy_auto_corr.size() * sizeof(TenElemT));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write energy_auto_corr to " + filename);
      ofs << std::endl;
      if (ofs.fail()) throw std::ios_base::failure("Failed to write endl to " + filename);
      ofs.close();
      if (ofs.fail()) throw std::ios_base::failure("Failed to close " + filename);

      filename = "one_point_functions";
      ofs.open(filename, std::ofstream::binary);
      if (!ofs.is_open()) {
        throw std::ios_base::failure("Failed to open file: " + filename);
      }
      ofs.write((const char *) one_point_functions.data(), one_point_functions.size() * sizeof(TenElemT));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write one_point_functions to " + filename);
      ofs.write((const char *) one_point_function_errs.data(), one_point_function_errs.size() * sizeof(double));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write one_point_function_errs to " + filename);
      ofs.write((const char *) one_point_functions_auto_corr.data(),
                one_point_functions_auto_corr.size() * sizeof(TenElemT));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write one_point_functions_auto_corr to " + filename);
      ofs << std::endl;
      if (ofs.fail()) throw std::ios_base::failure("Failed to write endl to " + filename);
      ofs.close();
      if (ofs.fail()) throw std::ios_base::failure("Failed to close " + filename);

      filename = "two_point_functions";
      ofs.open(filename, std::ofstream::binary);
      if (!ofs.is_open()) {
        throw std::ios_base::failure("Failed to open file: " + filename);
      }
      ofs.write((const char *) two_point_functions.data(), two_point_functions.size() * sizeof(TenElemT));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write two_point_functions to " + filename);
      ofs.write((const char *) two_point_function_errs.data(), two_point_function_errs.size() * sizeof(double));
      if (ofs.fail()) throw std::ios_base::failure("Failed to write two_point_function_errs to " + filename);
      ofs << std::endl;
      if (ofs.fail()) throw std::ios_base::failure("Failed to write endl to " + filename);
      ofs.close();
      if (ofs.fail()) throw std::ios_base::failure("Failed to close " + filename);
    }
    void DumpCSV() const {
      // Dump energy and error
      {
        std::ofstream ofs("energy_statistics.csv");
        if (!ofs.is_open()) throw std::ios_base::failure("Failed to open energy_statistics.csv");
        ofs << "energy,en_err\n";
        ofs << energy << "," << en_err << "\n";
        ofs.close();
      }

      // Dump bond energies and errors
      {
        std::ofstream ofs("bond_energys.csv");
        if (!ofs.is_open()) throw std::ios_base::failure("Failed to open bond_energys.csv");
        ofs << "bond_energy,bond_energy_err\n";
        for (size_t i = 0; i < bond_energys.size(); ++i) {
          ofs << bond_energys[i] << "," << (i < bond_energy_errs.size() ? bond_energy_errs[i] : 0.0) << "\n";
        }
        ofs.close();
      }

      // Dump one point functions and errors
      {
        std::ofstream ofs("one_point_functions.csv");
        if (!ofs.is_open()) throw std::ios_base::failure("Failed to open one_point_functions.csv");
        ofs << "one_point_function,one_point_function_err\n";
        for (size_t i = 0; i < one_point_functions.size(); ++i) {
          ofs << one_point_functions[i] << ","
              << (i < one_point_function_errs.size() ? one_point_function_errs[i] : 0.0) << "\n";
        }
        ofs.close();
      }

      // Dump two point functions and errors
      {
        std::ofstream ofs("two_point_functions.csv");
        if (!ofs.is_open()) throw std::ios_base::failure("Failed to open two_point_functions.csv");
        ofs << "two_point_function,two_point_function_err\n";
        for (size_t i = 0; i < two_point_functions.size(); ++i) {
          ofs << two_point_functions[i] << ","
              << (i < two_point_function_errs.size() ? two_point_function_errs[i] : 0.0) << "\n";
        }
        ofs.close();
      }

      // Dump energy auto correlation
      {
        std::ofstream ofs("energy_auto_corr.csv");
        if (!ofs.is_open()) throw std::ios_base::failure("Failed to open energy_auto_corr.csv");
        ofs << "energy_auto_corr,energy_auto_corr_err\n";
        for (size_t i = 0; i < energy_auto_corr.size(); ++i) {
          ofs << energy_auto_corr[i] << "," << (i < energy_auto_corr_err.size() ? energy_auto_corr_err[i] : 0.0)
              << "\n";
        }
        ofs.close();
      }

      // Dump one point function auto correlation
      {
        std::ofstream ofs("one_point_functions_auto_corr.csv");
        if (!ofs.is_open()) throw std::ios_base::failure("Failed to open one_point_functions_auto_corr.csv");
        ofs << "one_point_functions_auto_corr,one_point_functions_auto_corr_err\n";
        for (size_t i = 0; i < one_point_functions_auto_corr.size(); ++i) {
          ofs << one_point_functions_auto_corr[i] << ","
              << (i < one_point_functions_auto_corr_err.size() ? one_point_functions_auto_corr_err[i] : 0.0) << "\n";
        }
        ofs.close();
      }
    }
  } res;

  /**
   * Record the statistic inside `bin_size`
   */
  struct BinStatistics {
    double energy_mean;
    double energy_square_mean;  // used to restore the global variance
    std::vector<double> bond_energy_mean;
    std::vector<double> bond_square_mean;

    std::vector<double> one_point_function_mean;
    std::vector<double> two_point_function_mean;

  };
  // observable
  struct SampleData {
    std::vector<TenElemT> wave_function_amplitude_samples;
    std::vector<TenElemT> energy_samples;

    std::vector<std::vector<TenElemT>> bond_energy_samples;
    std::vector<std::vector<TenElemT>>
        one_point_function_samples; // outside is the sample index, inner side is the lattice index.
    std::vector<std::vector<TenElemT>>
        two_point_function_samples;

    void Reserve(const size_t sample_num) {
      wave_function_amplitude_samples.reserve(sample_num);
      energy_samples.reserve(sample_num);
      bond_energy_samples.reserve(sample_num);
      one_point_function_samples.reserve(sample_num);
      two_point_function_samples.reserve(sample_num);
    }

    void Clear() {
      wave_function_amplitude_samples.clear();
      energy_samples.clear();
      bond_energy_samples.clear();
      one_point_function_samples.clear();
      two_point_function_samples.clear();
    }

    void PushBack(TenElemT wave_function_amplitude, ObservablesLocal<TenElemT> &&observables_sample) {
      wave_function_amplitude_samples.push_back(wave_function_amplitude);
      energy_samples.push_back(observables_sample.energy_loc);
      bond_energy_samples.push_back(std::move(observables_sample.bond_energys_loc));
      one_point_function_samples.push_back(std::move(observables_sample.one_point_functions_loc));
      two_point_function_samples.push_back(std::move(observables_sample.two_point_functions_loc));
    }

    /**
     * Average, Standard error, auto correlation inside one MPI process
     * @return
     */
    Result Statistic(void) const {
      Result res_thread;
      res_thread.energy = Mean(energy_samples);
      res_thread.en_err = 0.0;
      res_thread.bond_energys = AveListOfData(bond_energy_samples);
      res_thread.energy_auto_corr = CalAutoCorrelation(energy_samples, res_thread.energy);
      res_thread.one_point_functions = AveListOfData(one_point_function_samples);
      res_thread.two_point_functions = AveListOfData(two_point_function_samples);
      // Here we assume one_point_functions is something like sz configuration
      res_thread.one_point_functions_auto_corr = CalSpinAutoCorrelation(one_point_function_samples);
      return res_thread;
    }

    void DumpOnePointFunctionSamples(const std::string &filename) const {
      DumpSampleData(one_point_function_samples, filename);
    }
    /**
      * @brief Dumps the two_point_function_samples to a CSV file.
      *
      * This function writes the `two_point_function_samples` data to a file in CSV format.
      * Each row in the output file corresponds to a single sample (outer vector),
      * and each column within a row corresponds to an element of the inner vector.
      *
      * @param filename The name of the output file. Should follow the `.csv` naming convention.
      *
      * @details
      * The output file format is designed for easy reading in MATLAB or Python.
      *
      * Example of MATLAB usage:
      * ```
      * data = csvread('output.csv');
      * ```
      *
      * Example of Python usage:
      * ```python
      * import numpy as np
      * data = np.loadtxt('output.csv', delimiter=',')
      * ```
      *
      * @throws std::ios_base::failure If the file cannot be opened for writing.
      */
    void DumpTwoPointFunctionSamples(const std::string &filename) const {
      DumpSampleData(two_point_function_samples, filename);
    }
  } sample_data_;
};//MCPEPSMeasurer


}//qlpeps

// Include implementation
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer_impl.h"

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_H