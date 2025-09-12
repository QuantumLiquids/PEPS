/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-11-02
*
* Description: QuantumLiquids/PEPS project. Implementation file for MCPEPSMeasurer template class.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_IMPL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_IMPL_H

namespace qlpeps {

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
MCPEPSMeasurer<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              MeasurementSolver>::MCPEPSMeasurer(
    const SITPST &sitpst,
    const MCMeasurementParams &measurement_params,
    const MPI_Comm &comm,
    const MeasurementSolver &solver,
    MonteCarloSweepUpdater mc_updater):
    qlten::Executor(),
    mc_measure_params(measurement_params),
    measurement_solver_(solver),
    engine_(sitpst, measurement_params.mc_params, measurement_params.peps_params, comm, std::move(mc_updater)) {
  ReserveSamplesDataSpace_();
  PrintExecutorInfo_();
  qlpeps::MPISignalGuard::Register();
  // Load observable metadata from solver
  observables_meta_ = measurement_solver_.DescribeObservables();
  this->SetStatus(ExecutorStatus::INITED);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
std::unique_ptr<MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>>
MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::CreateByLoadingTPS(
    const std::string& tps_path,
    const MCMeasurementParams& measurement_params,
    const MPI_Comm& comm,
    const MeasurementSolver& solver,
    MonteCarloSweepUpdater mc_updater) {
  
  // Load TPS from file path with proper error handling
  SITPST loaded_tps(measurement_params.mc_params.initial_config.rows(), 
                    measurement_params.mc_params.initial_config.cols());
  if (!loaded_tps.Load(tps_path)) {
    throw std::runtime_error("Failed to load TPS from path: " + tps_path);
  }
  
  // Create executor using the primary constructor with loaded TPS
  return std::make_unique<MCPEPSMeasurer>(
      loaded_tps, measurement_params, comm, solver, std::move(mc_updater));
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::ReplicaTest(
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
    int err_msg = MPI_Sendrecv(engine_.WavefuncComp().config, dest, dest, config2, source, engine_.Rank(), MPI_Comm(engine_.Comm()),
                               &status);

    // calculate overlap
    overlaps.push_back(overlap_func(engine_.WavefuncComp().config, config2));
    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank && (sweep + 1) % (mc_measure_params.mc_params.num_samples / 10) == 0) {
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
  engine_.EnsureDirectoryExists(replica_overlap_path + "dummy");  // Creates replica_overlap/
  DumpVecData(replica_overlap_path + "/replica_overlap" + std::to_string(engine_.Rank()), overlaps);
  // Dump configuration using path from MonteCarloParams (empty = no dump)
  if (!mc_measure_params.mc_params.config_dump_path.empty()) {
    engine_.WavefuncComp().config.Dump(mc_measure_params.mc_params.config_dump_path, engine_.Rank());
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::Execute(void) {
  this->SetStatus(ExecutorStatus::EXEING);
  engine_.WarmUp();
  Measure_();
  DumpData();
  this->SetStatus(ExecutorStatus::FINISH);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT,
                                   QNT,
                                   MonteCarloSweepUpdater,
                                   MeasurementSolver>::ReserveSamplesDataSpace_(
    void) {
  sample_data_.Reserve(mc_measure_params.mc_params.num_samples);
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::MeasureSample_() {
#ifdef QLPEPS_TIMING_MODE
  Timer evaluate_sample_obsrvb_timer("evaluate_sample_observable (rank " + std::to_string(engine_.Rank()) + ")");
#endif
  // Registry-based path only
  auto registry_map = measurement_solver_.template EvaluateObservables<TenElemT, QNT>(&engine_.State(), &engine_.WavefuncComp());
  sample_data_.PushBackRegistry(engine_.WavefuncComp().amplitude, registry_map);
  // Psi summary via dedicated API (no registry involvement)
  {
    static size_t warn_count = 0;
    auto psi_summary = measurement_solver_.template EvaluatePsiSummary<TenElemT, QNT>(&engine_.State(), &engine_.WavefuncComp());
    const bool enabled = mc_measure_params.psi_consistency_warning_enabled;
    const double th = mc_measure_params.psi_consistency_warn_threshold;
    const size_t maxw = mc_measure_params.psi_consistency_max_warnings;
    if (enabled && psi_summary.psi_rel_err > th && warn_count < maxw) {
      ++warn_count;
      std::cerr << "[psi_consistency] rel_err=" << std::scientific << psi_summary.psi_rel_err
                << " > threshold=" << th << ". Consider relaxing truncation parameters.\n";
      if (warn_count == maxw) {
        std::cerr << "[psi_consistency] reached max warnings (" << maxw << ") on this rank, suppressing further messages.\n";
      }
    }
    // Append to per-sample psi list for later dump
    psi_samples_.push_back({psi_summary.psi_mean, psi_summary.psi_rel_err});
  }
#ifdef QLPEPS_TIMING_MODE
  evaluate_sample_obsrvb_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::GatherStatistic_() {
  // No legacy per-rank stats; we aggregate registry directly
  std::cout << "Rank " << engine_.Rank() << ": statistic data finished." << std::endl;

  // Aggregate registry-based observables
  auto local_registry = sample_data_.StatisticRegistry();
  // First process energy (scalar) to populate Result
  if (auto it = local_registry.find("energy"); it != local_registry.end()) {
    const auto &local_mean = it->second.first; // length 1
    std::vector<TenElemT> global_mean;
    std::vector<double> global_stderr;
    GatherStatisticListOfData(local_mean, engine_.Comm(), global_mean, global_stderr);
    if (!global_mean.empty()) res.energy = global_mean[0];
    if (!global_stderr.empty()) res.en_err = global_stderr[0];
  }
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

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void
MCPEPSMeasurer<TenElemT,
                              QNT,
                              MonteCarloSweepUpdater,
                              MeasurementSolver>::DumpData(const std::string &measurement_data_path) {
  // Dump configuration if path is specified in MonteCarloParams
  if (!mc_measure_params.mc_params.config_dump_path.empty()) {
    // Create directory for configuration dump with informative output
    engine_.EnsureDirectoryExists(mc_measure_params.mc_params.config_dump_path);
    engine_.WavefuncComp().config.Dump(mc_measure_params.mc_params.config_dump_path, engine_.Rank());
  }

  if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
    const std::string base_dir = (measurement_data_path.empty() ? std::string("./") : (measurement_data_path + "/"));
    const std::string stats_dir = base_dir + "stats/";
    engine_.EnsureDirectoryExists(stats_dir + "dummy");
    if (!registry_stats_.empty()) {
      // Dump only registry-based results keyed by observable name
      for (const auto &kv : registry_stats_) {
        const std::string &key = kv.first;
        const auto &pair = kv.second;
        const auto &vals = pair.first;
        const auto &errs = pair.second;

        // Determine lattice size once
        const size_t ly = engine_.Ly();
        const size_t lx = engine_.Lx();

        // Decide dump style based on observable meta
        bool dumped = false;
        for (const auto &m : observables_meta_) {
          if (m.key != key) continue;
          const bool shape_is_matrix = (m.shape.size() == 2 && m.shape[0] == ly && m.shape[1] == lx);
          const bool labels_imply_matrix = (m.shape.empty() && m.index_labels.size() == 2 && m.index_labels[0] == "y" && m.index_labels[1] == "x" && vals.size() == ly * lx);
          if (shape_is_matrix || labels_imply_matrix) {
            DumpStatsMatrix_(stats_dir, key, vals, ly, lx);
            dumped = true;
          }
          break;
        }

        if (!dumped) {
          // Default: flat dump; special-case conceptually real quantities
          if (key == "psi_rel_err") {
            std::vector<double> vals_real;
            vals_real.reserve(vals.size());
            for (const auto &v : vals) { vals_real.push_back(static_cast<double>(std::real(v))); }
            DumpStatsFlat_(stats_dir, key, vals_real, errs);
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
    }
    // Dump psi samples separately (samples/psi.csv)
    DumpPsiSamples_(base_dir);
  }
  // raw samples dump removed in registry-only mode
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT,
                                   QNT,
                                   MonteCarloSweepUpdater,
                                   MeasurementSolver>::PrintExecutorInfo_(void) {
  engine_.PrintCommonInfo("MONTE-CARLO MEASUREMENT PROGRAM FOR PEPS");
  engine_.PrintTechInfo();
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT,
                                   QNT,
                                   MonteCarloSweepUpdater,
                                   MeasurementSolver>::SynchronizeConfiguration_(
    const size_t root) {
  Configuration config(engine_.WavefuncComp().config);
  MPI_BCast(config, root, MPI_Comm(engine_.Comm()));
  if (engine_.Rank() != root) {
    engine_.WavefuncComp() = typename MonteCarloEngine<TenElemT, QNT, MonteCarloSweepUpdater>::WaveFunctionComponentT(
        engine_.State(), config, engine_.WavefuncComp().trun_para);
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::DumpData(void) {
  DumpData(mc_measure_params.measurement_data_dump_path);  // Use measurement data dump path
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::Measure_(void) {
  std::vector<double> accept_rates_accum;
  const size_t print_bar_length = (mc_measure_params.mc_params.num_samples / 10) > 0 ? (mc_measure_params.mc_params.num_samples / 10) : 1;
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

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::DumpStatsMatrix_(
    const std::string &dir,
    const std::string &key,
    const std::vector<TenElemT> &vals,
    size_t ly,
    size_t lx) const {
  std::ofstream ofs(dir + key + ".csv");
  ofs << "index,mean,stderr\n";
  for (size_t y = 0; y < ly; ++y) {
    for (size_t x = 0; x < lx; ++x) {
      const size_t idx = y * lx + x;
      ofs << idx << "," << vals[idx] << "," << 0.0 << "\n";
    }
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::DumpStatsFlat_(
    const std::string &dir,
    const std::string &key,
    const std::vector<TenElemT> &vals,
    const std::vector<double> &errs) const {
  std::ofstream ofs(dir + key + ".csv");
  ofs << "index,mean,stderr\n";
  for (size_t i = 0; i < vals.size(); ++i) {
    const double err = (i < errs.size()) ? errs[i] : 0.0;
    ofs << i << "," << vals[i] << "," << err << "\n";
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::DumpStatsFlat_(
    const std::string &dir,
    const std::string &key,
    const std::vector<double> &vals,
    const std::vector<double> &errs) const {
  std::ofstream ofs(dir + key + ".csv");
  ofs << "index,mean,stderr\n";
  for (size_t i = 0; i < vals.size(); ++i) {
    const double err = (i < errs.size()) ? errs[i] : 0.0;
    ofs << i << "," << vals[i] << "," << err << "\n";
  }
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::DumpPackedUpperTriIndexMap_(
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

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
std::pair<TenElemT, double>
MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::ComputePsiConsistencyRelErr_(
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

} // namespace qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_IMPL_H
