/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-11-02
*
* Description: QuantumLiquids/PEPS project.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_H

#include "qlpeps/two_dim_tn/tps/tps.h"                            // TPS
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"                //SplitIndexTPS
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"  //MCMeasurementPara
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" //ObservablesLocal
#include "qlpeps/monte_carlo_tools/statistics.h"                  // Mean, Variance, DumpVecData, ...
#include "monte_carlo_peps_base.h"

namespace qlpeps {
using namespace qlten;

using qlmps::IsPathExist;
using qlmps::CreatPath;

///< sum (config1 * config2)
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

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
class MonteCarloMeasurementExecutor : public MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater> {
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::comm_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::mpi_size_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::rank_;

  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::split_index_tps_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::tps_sample_;
  struct Result;
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;

  //Load Data from path
  MonteCarloMeasurementExecutor(const MCMeasurementPara &measurement_para,
                                const size_t ly, const size_t lx,
                                const MPI_Comm &comm,
                                const MeasurementSolver &solver = MeasurementSolver());

  MonteCarloMeasurementExecutor(const MCMeasurementPara &measurement_para,
                                const SITPST &sitps,
                                const MPI_Comm &comm,
                                const MeasurementSolver &solver = MeasurementSolver());

  void Execute(void) override;

  void ReplicaTest(std::function<double(const Configuration &, const Configuration &)>); // for check the ergodicity

  void DumpData();

  void DumpData(const std::string &tps_path);

  MCMeasurementPara mc_measure_para;

  std::pair<TenElemT, double> OutputEnergy() const {
    if (rank_ == kMPIMasterRank) {
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
 private:
  void ReserveSamplesDataSpace_();

  void PrintExecutorInfo_(void);

  void Measure_(void);

  void MeasureSample_(void);

  void GatherStatistic_(void);

  void SynchronizeConfiguration_(const size_t root = 0); //for the replica test

  MeasurementSolver measurement_solver_;
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
      ofs.write((const char *) &energy, 1 * sizeof(TenElemT));
      ofs.write((const char *) &en_err, 1 * sizeof(double));
      ofs.write((const char *) bond_energys.data(), bond_energys.size() * sizeof(TenElemT));
      ofs.write((const char *) energy_auto_corr.data(), energy_auto_corr.size() * sizeof(TenElemT));
      ofs << std::endl;
      ofs.close();

      filename = "one_point_functions";
      ofs.open(filename, std::ofstream::binary);
      ofs.write((const char *) one_point_functions.data(), one_point_functions.size() * sizeof(TenElemT));
      ofs.write((const char *) one_point_function_errs.data(), one_point_function_errs.size() * sizeof(double));
      ofs.write((const char *) one_point_functions_auto_corr.data(),
                one_point_functions_auto_corr.size() * sizeof(TenElemT));
      ofs << std::endl;
      ofs.close();

      filename = "two_point_functions";
      ofs.open(filename, std::ofstream::binary);
      ofs.write((const char *) two_point_functions.data(), two_point_functions.size() * sizeof(TenElemT));
      ofs.write((const char *) two_point_function_errs.data(), two_point_function_errs.size() * sizeof(double));
      ofs << std::endl;
      ofs.close();
    }
  } res;

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

    void PushBack(TenElemT wave_function_amplitude, ObservablesLocal<TenElemT> &&observables_sample) {
      wave_function_amplitude_samples.push_back(wave_function_amplitude);
      energy_samples.push_back(observables_sample.energy_loc);
      bond_energy_samples.push_back(std::move(observables_sample.bond_energys_loc));
      one_point_function_samples.push_back(std::move(observables_sample.one_point_functions_loc));
      two_point_function_samples.push_back(std::move(observables_sample.two_point_functions_loc));
    }

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
      std::ofstream file(filename);
      if (!file.is_open()) {
        throw std::ios_base::failure("Failed to open file: " + filename);
      }

      // Write the two_point_function_samples to the file
      for (const auto &sample : two_point_function_samples) {
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
  } sample_data_;
};//MonteCarloMeasurementExecutor

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
MonteCarloMeasurementExecutor<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              MeasurementSolver>::MonteCarloMeasurementExecutor(
    const MCMeasurementPara &measurement_para,
    const size_t ly, const size_t lx,
    const MPI_Comm &comm,
    const MeasurementSolver &solver):
    MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>(ly, lx,
                                                                      MonteCarloParams(measurement_para),
                                                                      PEPSParams(measurement_para),
                                                                      comm),
    mc_measure_para(measurement_para),
    measurement_solver_(solver) {
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
MonteCarloMeasurementExecutor<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              MeasurementSolver>::MonteCarloMeasurementExecutor(
    const MCMeasurementPara &measurement_para,
    const SITPST &sitpst,
    const MPI_Comm &comm,
    const MeasurementSolver &solver):
    MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>(sitpst,
                                                                      MonteCarloParams(measurement_para),
                                                                      PEPSParams(measurement_para),
                                                                      comm),
    mc_measure_para(measurement_para),
    measurement_solver_(solver) {
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::ReplicaTest(
    std::function<double(const Configuration &,
                         const Configuration &)> overlap_func // calculate overlap like, 1/N * sum (sz1 * sz2)
) {
  SynchronizeConfiguration_();
  std::vector<double> overlaps;
  overlaps.reserve(mc_measure_para.mc_samples);
//  std::cout << "Random number from worker " << rank_ << " : " << u_double_(random_engine) << std::endl;
  std::vector<double> accept_rates_accum;
  for (size_t sweep = 0; sweep < mc_measure_para.mc_samples; sweep++) {
    std::vector<double> accept_rates = this->MCSweep_();
    if (sweep == 0) {
      accept_rates_accum = accept_rates;
    } else {
      for (size_t i = 0; i < accept_rates_accum.size(); i++) {
        accept_rates_accum[i] += accept_rates[i];
      }
    }
    // send-recv configuration
    Configuration config2(this->ly_, this->lx_);
    size_t dest = (rank_ + 1) % mpi_size_;
    size_t source = (rank_ + mpi_size_ - 1) % mpi_size_;
    MPI_Status status;
    int err_msg = MPI_Sendrecv(tps_sample_.config, dest, dest, config2, source, rank_, MPI_Comm(comm_),
                               &status);

    // calculate overlap
    overlaps.push_back(overlap_func(tps_sample_.config, config2));
    if (rank_ == kMPIMasterRank && (sweep + 1) % (mc_measure_para.mc_samples / 10) == 0) {
      PrintProgressBar((sweep + 1), mc_measure_para.mc_samples);

      auto accept_rates_avg = accept_rates_accum;
      for (double &rates : accept_rates_avg) {
        rates /= double(sweep + 1);
      }
      std::cout << "Accept rate = [";
      for (double &rate : accept_rates_avg) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(2) << rate;
      }
      std::cout << "]";
    }
  }
  //DumpData
  std::string replica_overlap_path = "replica_overlap/";
  if (rank_ == kMPIMasterRank)
    if (!IsPathExist(replica_overlap_path)) {
      CreatPath(replica_overlap_path);
    }
  MPI_Barrier(comm_);
  DumpVecData(replica_overlap_path + "/replica_overlap" + std::to_string(rank_), overlaps);
  tps_sample_.config.Dump(mc_measure_para.wavefunction_path, rank_);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::Execute(void) {
  this->SetStatus(ExecutorStatus::EXEING);
  this->WarmUp_();
  Measure_();
  DumpData();
  this->SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT,
                                   QNT,
                                   MonteCarloSweepUpdater,
                                   MeasurementSolver>::ReserveSamplesDataSpace_(
    void) {
  sample_data_.Reserve(mc_measure_para.mc_samples);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::MeasureSample_() {
#ifdef QLPEPS_TIMING_MODE
  Timer evaluate_sample_obsrvb_timer("evaluate_sample_observable (rank " + std::to_string(rank_) + ")");
#endif
  ObservablesLocal<TenElemT> observables_local = measurement_solver_(&split_index_tps_, &tps_sample_);
  sample_data_.PushBack(tps_sample_.amplitude, std::move(observables_local));
#ifdef QLPEPS_TIMING_MODE
  evaluate_sample_obsrvb_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::GatherStatistic_() {
  Result res_thread = sample_data_.Statistic();
  std::cout << "Rank " << rank_ << ": statistic data finished." << std::endl;

  auto [energy, en_err] = GatherStatisticSingleData(res_thread.energy, MPI_Comm(comm_));
  res.energy = energy;
  res.en_err = en_err;
  GatherStatisticListOfData(res_thread.bond_energys,
                            comm_,
                            res.bond_energys,
                            res.bond_energy_errs);
  GatherStatisticListOfData(res_thread.one_point_functions,
                            comm_,
                            res.one_point_functions,
                            res.one_point_function_errs);
  GatherStatisticListOfData(res_thread.two_point_functions,
                            comm_,
                            res.two_point_functions,
                            res.two_point_function_errs);
  GatherStatisticListOfData(res_thread.energy_auto_corr,
                            comm_,
                            res.energy_auto_corr,
                            res.energy_auto_corr_err);
  GatherStatisticListOfData(res_thread.one_point_functions_auto_corr,
                            comm_,
                            res.one_point_functions_auto_corr,
                            res.one_point_functions_auto_corr_err);

}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void
MonteCarloMeasurementExecutor<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              MeasurementSolver>::DumpData(const std::string &tps_path) {
  tps_sample_.config.Dump(tps_path, rank_);

  if (rank_ == kMPIMasterRank) {
    res.Dump();
  }
  //dump sample data
  std::string energy_raw_path = "energy_sample_data/";
  std::string wf_amplitude_path = "wave_function_amplitudes/";
  std::string two_point_function_raw_data_path = "two_point_function_samples/";
  if (rank_ == kMPIMasterRank && !IsPathExist(energy_raw_path))
    CreatPath(energy_raw_path);
  if (rank_ == kMPIMasterRank && !IsPathExist(wf_amplitude_path))
    CreatPath(wf_amplitude_path);
  if (rank_ == kMPIMasterRank && !IsPathExist(two_point_function_raw_data_path))
    CreatPath(two_point_function_raw_data_path);
  MPI_Barrier(comm_);
  DumpVecData(energy_raw_path + "/energy" + std::to_string(rank_), sample_data_.energy_samples);
  DumpVecData(wf_amplitude_path + "/psi" + std::to_string(rank_), sample_data_.wave_function_amplitude_samples);
  sample_data_.DumpTwoPointFunctionSamples(
      two_point_function_raw_data_path + "/sample" + std::to_string(rank_) + ".csv");
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT,
                                   QNT,
                                   MonteCarloSweepUpdater,
                                   MeasurementSolver>::PrintExecutorInfo_(void) {
  this->PrintCommonInfo_("MONTE-CARLO MEASUREMENT PROGRAM FOR PEPS");
  this->PrintTechInfo_();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT,
                                   QNT,
                                   MonteCarloSweepUpdater,
                                   MeasurementSolver>::SynchronizeConfiguration_(
    const size_t root) {
  Configuration config(tps_sample_.config);
  MPI_BCast(config, root, MPI_Comm(comm_));
  if (rank_ != root) {
    tps_sample_ = MonteCarloSweepUpdater(split_index_tps_, config);
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::DumpData(void) {
  DumpData(mc_measure_para.wavefunction_path);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MonteCarloMeasurementExecutor<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::Measure_(void) {
  std::vector<double> accept_rates_accum;
  const size_t print_bar_length = (mc_measure_para.mc_samples / 10) > 0 ? (mc_measure_para.mc_samples / 10) : 1;
  for (size_t sweep = 0; sweep < mc_measure_para.mc_samples; sweep++) {
    std::vector<double> accept_rates = this->MCSweep_();
    if (sweep == 0) {
      accept_rates_accum = accept_rates;
    } else {
      for (size_t i = 0; i < accept_rates_accum.size(); i++) {
        accept_rates_accum[i] += accept_rates[i];
      }
    }
    MeasureSample_();
    if (rank_ == kMPIMasterRank && (sweep + 1) % print_bar_length == 0) {
      PrintProgressBar((sweep + 1), mc_measure_para.mc_samples);
    }
  }
  std::vector<double> accept_rates_avg = accept_rates_accum;
  for (double &rates : accept_rates_avg) {
    rates /= double(mc_measure_para.mc_samples);
  }
  std::cout << "Accept rate = [";
  for (double &rate : accept_rates_avg) {
    std::cout << std::setw(5) << std::fixed << std::setprecision(2) << rate;
  }
  std::cout << "]";
  GatherStatistic_();
}
}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_H
