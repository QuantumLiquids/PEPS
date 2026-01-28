/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-11-02
*
* Description: QuantumLiquids/PEPS project. Implementation file for MCPEPSMeasurer template class.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_IMPL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_IMPL_H

#include <complex>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace qlpeps {
inline void ConfigureStreamForHighPrecision(std::ofstream &ofs) {
  ofs.setf(std::ios::scientific, std::ios::floatfield);
  ofs << std::setprecision(std::numeric_limits<double>::max_digits10);
}

inline std::string ToCsvString(double value) {
  std::ostringstream oss;
  oss.setf(std::ios::scientific, std::ios::floatfield);
  oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return oss.str();
}

inline std::string ToCsvString(const std::complex<double> &value) {
  std::ostringstream oss;
  oss.setf(std::ios::scientific, std::ios::floatfield);
  oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return oss.str();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
MCPEPSMeasurer<TenElemT,
               QNT,
               MonteCarloSweepUpdater,
               MeasurementSolver,
               ContractorT>::MCPEPSMeasurer(
  const SITPST &sitpst,
  const MCMeasurementParams &measurement_params,
  const MPI_Comm &comm,
  const MeasurementSolver &solver,
  MonteCarloSweepUpdater mc_updater) : qlten::Executor(),
                                       mc_measure_params(measurement_params),
                                       measurement_solver_(solver),
                                       engine_(sitpst,
                                               measurement_params.mc_params,
                                               measurement_params.peps_params,
                                               comm,
                                               std::move(mc_updater),
                                               measurement_params.runtime_params.config_rescue) {
  ReserveSamplesData_();
  qlpeps::MPISignalGuard::Register();
  // Load observable metadata from solver
  observables_meta_ = measurement_solver_.DescribeObservables(engine_.Ly(), engine_.Lx());
  PrintExecutorInfo_();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
std::unique_ptr<MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT> >
MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::CreateByLoadingTPS(
  const std::string &tps_path,
  const MCMeasurementParams &measurement_params,
  const MPI_Comm &comm,
  const MeasurementSolver &solver,
  MonteCarloSweepUpdater mc_updater) {
  // Load TPS from file path with proper error handling
  SITPST loaded_tps(measurement_params.mc_params.initial_config.rows(),
                    measurement_params.mc_params.initial_config.cols());
  if (!loaded_tps.Load(tps_path)) {
    throw std::runtime_error("Failed to load TPS from path: " + tps_path);
  }

  // Create executor using the primary constructor with loaded TPS
  return std::make_unique<MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>>(
    loaded_tps,
    measurement_params,
    comm,
    solver,
    std::move(mc_updater));
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::ReplicaTest(
  std::function<double(const Configuration &,
                       const Configuration &)> overlap_func // calculate overlap like, 1/N * sum (sz1 * sz2)
) {
  SynchronizeConfiguration_();
  std::vector<double> overlaps;
  overlaps.reserve(mc_measure_params.mc_params.num_samples);
  //  std::cout << "Random number from worker " << rank_ << " : " << u_double_(random_engine) << std::endl;
  std::vector<double> accept_rates_accum;
  for (size_t sweep = 0; sweep < mc_measure_params.mc_params.num_samples; sweep++) {
    std::vector<double> accept_rates = engine_.StepSweep();
    if (sweep == 0) {
      accept_rates_accum = accept_rates;
    } else {
      for (size_t i = 0; i < accept_rates_accum.size(); i++) {
        accept_rates_accum[i] += accept_rates[i];
      }
    }
    // send-recv configuration
    Configuration config2(engine_.Ly(), engine_.Lx());
    size_t dest = (engine_.Rank() + 1) % engine_.MpiSize();
    size_t source = (engine_.Rank() + engine_.MpiSize() - 1) % engine_.MpiSize();
    MPI_Status status;
    int err_msg = MPI_Sendrecv(engine_.WavefuncComp().config,
                               dest,
                               dest,
                               config2,
                               source,
                               engine_.Rank(),
                               MPI_Comm(engine_.Comm()),
                               &status);

    // calculate overlap
    overlaps.push_back(overlap_func(engine_.WavefuncComp().config, config2));
    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank && (sweep + 1) % (mc_measure_params.mc_params.num_samples /
      10) == 0) {
      PrintProgressBar((sweep + 1), mc_measure_params.mc_params.num_samples);

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
  // Create replica overlap directory using EnsureDirectoryExists_ with dummy filename
  // Note: EnsureDirectoryExists_ expects a file path and creates its parent directory.
  // We append "dummy" as a placeholder filename to make it create the target directory itself.
  engine_.EnsureDirectoryExists(replica_overlap_path + "dummy"); // Creates replica_overlap/
  DumpVecData(replica_overlap_path + "/replica_overlap" + std::to_string(engine_.Rank()), overlaps);
  // Dump configuration using path from MonteCarloParams (empty = no dump)
  if (!mc_measure_params.mc_params.config_dump_path.empty()) {
    engine_.WavefuncComp().config.Dump(mc_measure_params.mc_params.config_dump_path, engine_.Rank());
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::Execute(void) {
  this->SetStatus(ExecutorStatus::EXEING);
  engine_.WarmUp();
  Measure_();
  DumpData();
  this->SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT,
                    QNT,
                    MonteCarloSweepUpdater,
                    MeasurementSolver,
                    ContractorT>::ReserveSamplesData_(
  void) {
  sample_data_.Reserve(mc_measure_params.mc_params.num_samples);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::MeasureSample_() {
#ifdef QLPEPS_TIMING_MODE
  Timer evaluate_sample_obsrvb_timer("evaluate_sample_observable (rank " + std::to_string(engine_.Rank()) + ")");
#endif
  // Registry-based path only
  auto registry_map = measurement_solver_.template EvaluateObservables<TenElemT, QNT>(
    &engine_.State(),
    &engine_.WavefuncComp());
  sample_data_.PushBackRegistry(engine_.WavefuncComp().amplitude, registry_map);
  // Psi summary via dedicated API (no registry involvement)
  {
    static size_t warn_count = 0;
    auto psi_summary = measurement_solver_.template EvaluatePsiSummary<TenElemT, QNT>(
      &engine_.State(),
      &engine_.WavefuncComp());

    const auto &p = mc_measure_params.runtime_params.psi_consistency;

    const bool should_print_rank =
        (!p.master_only) || (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank);
    if (p.enabled && should_print_rank && psi_summary.psi_rel_err > p.threshold && warn_count < p.max_warnings) {
      ++warn_count;
      std::cerr << "[psi_consistency] rel_err=" << std::scientific << psi_summary.psi_rel_err
          << " > threshold=" << p.threshold
          << ". Consider relaxing truncation parameters."
          << " psi_mean=" << psi_summary.psi_mean
          << "\n";
      if (warn_count == p.max_warnings) {
        std::cerr << "[psi_consistency] reached max warnings (" << p.max_warnings <<
            ") on this rank, suppressing further messages.\n";
      }
    }
    // Append to per-sample psi list for later dump
    psi_samples_.push_back({psi_summary.psi_mean, psi_summary.psi_rel_err});
  }
#ifdef QLPEPS_TIMING_MODE
  evaluate_sample_obsrvb_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::GatherStatistic_() {
  // No legacy per-rank stats; we aggregate registry directly
  std::cout << "Rank " << engine_.Rank() << ": statistic data finished." << std::endl;

  // Aggregate registry-based observables
  auto local_registry = sample_data_.StatisticRegistry();
  // Then gather all keys into registry_stats_
  for (const auto &kv : local_registry) {
    const std::string &key = kv.first;
    const std::vector<TenElemT> &local_mean = kv.second.first;
    std::vector<TenElemT> global_mean;
    std::vector<double> global_stderr;
    GatherStatisticListOfData(local_mean, engine_.Comm(), global_mean, global_stderr);
    registry_stats_[key] = std::make_pair(std::move(global_mean), std::move(global_stderr));
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void
MCPEPSMeasurer<TenElemT,
               QNT,
               MonteCarloSweepUpdater,
               MeasurementSolver,
               ContractorT>::DumpData(const std::string &measurement_data_path) {
  // Dump configuration if path is specified in MonteCarloParams
  if (!mc_measure_params.mc_params.config_dump_path.empty()) {
    // Create directory for configuration dump with informative output
    engine_.EnsureDirectoryExists(mc_measure_params.mc_params.config_dump_path);
    engine_.WavefuncComp().config.Dump(mc_measure_params.mc_params.config_dump_path, engine_.Rank());
  }

  const std::string base_dir = (measurement_data_path.empty() ? std::string("./") : (measurement_data_path + "/"));
  const std::string stats_dir = base_dir + "stats/";
  engine_.EnsureDirectoryExists(stats_dir + "dummy");

  if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    if (!registry_stats_.empty()) {
      // Dump only registry-based results keyed by observable name
      for (const auto &kv : registry_stats_) {
        const std::string &key = kv.first;
        const auto &pair = kv.second;
        const auto &vals = pair.first;
        const auto &errs = pair.second;

        bool dumped = false;

        const ObservableMeta *meta_ptr = nullptr;
        for (const auto &meta : observables_meta_) {
          if (meta.key == key) {
            meta_ptr = &meta;
            break;
          }
        }

        if (meta_ptr != nullptr && !meta_ptr->shape.empty()) {
          size_t expected = 1;
          for (size_t dim : meta_ptr->shape) {
            if (dim == 0) {
              throw std::runtime_error("Observable '" + key + "' declares zero-sized dimension.");
            }
            expected *= dim;
          }
          if (expected != vals.size()) {
            throw std::runtime_error("Observable '" + key + "' metadata shape mismatch: expected "
                                     + std::to_string(expected) + " entries, got " + std::to_string(vals.size()));
          }

          if (meta_ptr->shape.size() == 2) {
            DumpStatsMatrix_(stats_dir, key, vals, errs, meta_ptr->shape[0], meta_ptr->shape[1]);
            dumped = true;
          }
        }

        if (!dumped) {
          if (key == "psi_rel_err") {
            std::vector<double> vals_real;
            vals_real.reserve(vals.size());
            for (const auto &v : vals) { vals_real.push_back(static_cast<double>(std::real(v))); }
            DumpStatsFlatReal_(stats_dir, key, vals_real, errs);
          } else {
            DumpStatsFlat_(stats_dir, key, vals, errs);
          }
        }

        // If meta hints triangular packing, emit index_map
        for (const auto &m : observables_meta_) {
          if (m.key == key) {
            if (!m.index_labels.empty() && m.index_labels[0] == "pair_packed_upper_tri") {
              DumpPackedUpperTriIndexMap_(stats_dir, key, vals.size());
            }
            break;
          }
        }
      }

      // Generate coordinate mapping files for observables with coord_generator
      for (const auto &meta : observables_meta_) {
        if (meta.coord_generator) {
          std::string content = meta.coord_generator(engine_.Ly(), engine_.Lx());
          std::ofstream ofs(stats_dir + meta.key + "_coords.txt");
          if (ofs.is_open()) {
            ofs << content;
          }
        }
      }
    }
  }
  // Dump psi samples separately (samples/psi.csv)
  DumpPsiSamples_(base_dir);
  // raw samples dump removed in registry-only mode
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT,
                    QNT,
                    MonteCarloSweepUpdater,
                    MeasurementSolver,
                    ContractorT>::PrintExecutorInfo_(void) {
  engine_.PrintCommonInfo("MONTE-CARLO MEASUREMENT PROGRAM FOR PEPS");
  if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    const size_t indent = 40;
    std::cout << std::left;
    std::cout << std::setw(indent) << "Measurement dump path:" << mc_measure_params.measurement_data_dump_path << "\n";
    std::cout << std::setw(indent) << "Registered observables:" << observables_meta_.size() << "\n";

    const auto &psi = mc_measure_params.runtime_params.psi_consistency;
    std::cout << std::setw(indent) << "psi_consistency warnings:" << (psi.enabled ? "enabled" : "disabled") << "\n";
    std::cout << std::setw(indent) << "psi_consistency master_only:" << (psi.master_only ? "true" : "false") << "\n";
    std::cout << std::setw(indent) << "psi_consistency threshold:" << psi.threshold << "\n";
    std::cout << std::setw(indent) << "psi_consistency max warnings:" << psi.max_warnings << "\n";
    std::cout << std::setw(indent) << "psi_consistency max print elems:" << psi.max_print_elems << "\n";
  }
  engine_.PrintTechInfo();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT,
                    QNT,
                    MonteCarloSweepUpdater,
                    MeasurementSolver,
                    ContractorT>::SynchronizeConfiguration_(
  const size_t root) {
  Configuration config(engine_.WavefuncComp().config);
  MPI_BCast(config, root, MPI_Comm(engine_.Comm()));
  if (engine_.Rank() != root) {
    engine_.WavefuncComp() =
        typename MonteCarloEngine<TenElemT, QNT, MonteCarloSweepUpdater, ContractorT>::WaveFunctionComponentT(
      engine_.State(),
      config,
      engine_.WavefuncComp().trun_para);
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::DumpData(void) {
  DumpData(mc_measure_params.measurement_data_dump_path); // Use measurement data dump path
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::Measure_(void) {
  std::vector<double> accept_rates_accum;
  const size_t print_bar_length = (mc_measure_params.mc_params.num_samples / 10) > 0
                                    ? (mc_measure_params.mc_params.num_samples / 10)
                                    : 1;
  for (size_t sweep = 0; sweep < mc_measure_params.mc_params.num_samples; sweep++) {
    // Emergency stop check (MPI-aware)
    if (qlpeps::MPISignalGuard::EmergencyStopRequested(engine_.Comm())) {
      if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
        std::cout << "\n[Emergency Stop] Signal received. Dumping current results and exiting gracefully.\n";
      }
      break;
    }

    std::vector<double> accept_rates = engine_.StepSweep();
    // Accept rates accumulation
    if (sweep == 0) {
      accept_rates_accum = accept_rates;
    } else {
      for (size_t i = 0; i < accept_rates_accum.size(); i++) {
        accept_rates_accum[i] += accept_rates[i];
      }
    }
    MeasureSample_();
    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank && (sweep + 1) % print_bar_length == 0) {
      PrintProgressBar((sweep + 1), mc_measure_params.mc_params.num_samples);
    }
  }
  std::vector<double> accept_rates_avg = accept_rates_accum;
  for (double &rates : accept_rates_avg) {
    rates /= double(mc_measure_params.mc_params.num_samples);
  }
  std::cout << "Accept rate = [";
  for (double &rate : accept_rates_avg) {
    std::cout << std::setw(5) << std::fixed << std::setprecision(2) << rate;
  }
  std::cout << "]";
  GatherStatistic_();
}

  /**
   * Layout contract:
   *  - `vals` MUST be Row-Major packed: idx = row * cols + col.
   *  - ObservableMatrix::Flatten/Extract already enforces this upstream.
   *  - DescribeObservables.shape = {rows, cols}, so CSV rows/cols align with lattice.
   * Any caller producing column-major buffers must transpose before invoking this helper.
   */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::DumpStatsMatrix_(
  const std::string &dir,
  const std::string &key,
  const std::vector<TenElemT> &vals,
  const std::vector<double> &errs,
  size_t rows,
  size_t cols) const {
  const std::string mean_path = dir + key + "_mean.csv";
  const std::string stderr_path = dir + key + "_stderr.csv";
  {
    std::ofstream ofs(mean_path);
    if (!ofs.is_open()) {
      throw std::runtime_error("Cannot open file: " + mean_path);
    }
    ConfigureStreamForHighPrecision(ofs);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        const size_t idx = r * cols + c;
        ofs << ToCsvString(vals[idx]);
        if (c + 1 < cols) {
          ofs << ",";
        }
      }
      ofs << "\n";
    }
  }
  {
    std::ofstream ofs(stderr_path);
    if (!ofs.is_open()) {
      throw std::runtime_error("Cannot open file: " + stderr_path);
    }
    ConfigureStreamForHighPrecision(ofs);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        const size_t idx = r * cols + c;
        const double err = (idx < errs.size()) ? errs[idx] : 0.0;
        ofs << ToCsvString(err);
        if (c + 1 < cols) {
          ofs << ",";
        }
      }
      ofs << "\n";
    }
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::DumpStatsFlat_(
  const std::string &dir,
  const std::string &key,
  const std::vector<TenElemT> &vals,
  const std::vector<double> &errs) const {
  std::ofstream ofs(dir + key + ".csv");
  ofs << "index,mean,stderr\n";
  ConfigureStreamForHighPrecision(ofs);
  for (size_t i = 0; i < vals.size(); ++i) {
    const double err = (i < errs.size()) ? errs[i] : 0.0;
    ofs << i << "," << ToCsvString(vals[i]) << "," << ToCsvString(err) << "\n";
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::DumpStatsFlatReal_(
  const std::string &dir,
  const std::string &key,
  const std::vector<double> &vals,
  const std::vector<double> &errs) const {
  std::ofstream ofs(dir + key + ".csv");
  ofs << "index,mean,stderr\n";
  ConfigureStreamForHighPrecision(ofs);
  for (size_t i = 0; i < vals.size(); ++i) {
    const double err = (i < errs.size()) ? errs[i] : 0.0;
    ofs << i << "," << ToCsvString(vals[i]) << "," << ToCsvString(err) << "\n";
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::DumpPackedUpperTriIndexMap_(
  const std::string &dir,
  const std::string &key,
  size_t packed_len) const {
  // Emit a text mapping describing how to reconstruct (i,j) from linear index k for upper-triangular packing
  std::ofstream ofs(dir + key + "_index_map.txt");
  ofs << "# index -> (i,j) mapping for upper-triangular packed pairs (i<=j)\n";
  ofs << "# k = i*L - i*(i-1)/2 + (j - i), with L being linear size along flattening;\n";
  ofs << "# This file only documents the convention; concrete (i,j) require model-specific L.\n";
  ofs << "# length=" << packed_len << "\n";
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
std::pair<TenElemT, double>
MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::ComputePsiConsistencyRelErr_(
  const std::vector<TenElemT> &psi_list) const {
  TenElemT mean(0);
  if (psi_list.empty()) { return {mean, 0.0}; }
  for (const auto &v : psi_list) mean += v;
  mean = mean / static_cast<double>(psi_list.size());
  auto abs_val = [](const TenElemT &v) -> double { return static_cast<double>(std::abs(v)); };
  const double denom = std::max(abs_val(mean), 1e-300);
  double max_dev = 0.0;
  for (const auto &v : psi_list) {
    max_dev = std::max(max_dev, abs_val(v - mean));
  }
  const double rel = max_dev / denom; // radius_rel = max_i |psi_i - mean| / |mean|
  return {mean, rel};
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
std::optional<typename MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::EnergyEstimate>
MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::QueryEnergyEstimate_() const {
  auto it = registry_stats_.find("energy");
  if (it == registry_stats_.end()) {
    return std::nullopt;
  }
  const auto &vals = it->second.first;
  const auto &errs = it->second.second;
  if (vals.empty()) {
    return std::nullopt;
  }
  TenElemT energy = vals.front();
  double stderr = errs.empty() ? 0.0 : errs.front();
  return EnergyEstimate{energy, stderr};
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
std::pair<TenElemT, double>
MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::OutputEnergy() const {
  auto energy_opt = QueryEnergyEstimate_();
  if (!energy_opt.has_value()) {
    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
      std::cout << "Energy observable is unavailable." << std::endl;
    }
    return {TenElemT(0), 0.0};
  }
  const auto &energy_est = energy_opt.value();
  if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    if (this->GetStatus() == ExecutorStatus::FINISH) {
      std::cout << "Measured energy : "
          << std::setw(8) << energy_est.energy
          << " +/- "
          << std::scientific << energy_est.stderr
          << std::endl;
    } else {
      std::cout << "The program didn't complete the measurements. " << std::endl;
    }
  }
  return {energy_est.energy, energy_est.stderr};
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver,
         template<typename, typename> class ContractorT>
std::optional<typename MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::EnergyEstimate>
MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, ContractorT>::GetEnergyEstimate() const {
  if (this->GetStatus() != ExecutorStatus::FINISH) {
    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
      std::cout << "The program didn't complete the measurements. " << std::endl;
    }
    return std::nullopt;
  }
  return QueryEnergyEstimate_();
}
} // namespace qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_IMPL_H
