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
    Configuration config2(engine_.Ly(), engine_.Lx_);
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
  ObservablesLocal<TenElemT> observables_local = measurement_solver_(&engine_.State(), &engine_.WavefuncComp());
  sample_data_.PushBack(engine_.WavefuncComp().amplitude, std::move(observables_local));
#ifdef QLPEPS_TIMING_MODE
  evaluate_sample_obsrvb_timer.PrintElapsed();
#endif
}

template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
void MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>::GatherStatistic_() {
  Result res_thread = sample_data_.Statistic();
  std::cout << "Rank " << engine_.Rank() << ": statistic data finished." << std::endl;

  auto [energy, en_err] = GatherStatisticSingleData(res_thread.energy, MPI_Comm(engine_.Comm()));
  res.energy = energy;
  res.en_err = en_err;
  GatherStatisticListOfData(res_thread.bond_energys,
                            engine_.Comm(),
                            res.bond_energys,
                            res.bond_energy_errs);
  GatherStatisticListOfData(res_thread.one_point_functions,
                            engine_.Comm(),
                            res.one_point_functions,
                            res.one_point_function_errs);
  GatherStatisticListOfData(res_thread.two_point_functions,
                            engine_.Comm(),
                            res.two_point_functions,
                            res.two_point_function_errs);
  GatherStatisticListOfData(res_thread.energy_auto_corr,
                            engine_.Comm(),
                            res.energy_auto_corr,
                            res.energy_auto_corr_err);
  GatherStatisticListOfData(res_thread.one_point_functions_auto_corr,
                            engine_.Comm(),
                            res.one_point_functions_auto_corr,
                            res.one_point_functions_auto_corr_err);

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
    res.Dump();
    res.DumpCSV(); // Dump data in two forms
  }
  //dump sample data relative to measurement_data_path
  const std::string base_path = measurement_data_path.empty() ? "./" : measurement_data_path + "/";
  const std::string energy_raw_path = base_path + "energy_sample_data/";
  const std::string wf_amplitude_path = base_path + "wave_function_amplitudes/";
  const std::string one_point_function_raw_data_path = base_path + "one_point_function_samples/";
  const std::string two_point_function_raw_data_path = base_path + "two_point_function_samples/";
  
  // Create measurement data directories using EnsureDirectoryExists_ with dummy filenames
  // Note: EnsureDirectoryExists_ expects a file path and creates its parent directory.
  // We append "dummy" as a placeholder filename to make it create the target directory itself.
  // This is a common workaround pattern when using file-path-based directory creation functions.
  engine_.EnsureDirectoryExists(energy_raw_path + "dummy");                    // Creates energy_sample_data/
  engine_.EnsureDirectoryExists(wf_amplitude_path + "dummy");                  // Creates wave_function_amplitudes/  
  engine_.EnsureDirectoryExists(one_point_function_raw_data_path + "dummy");   // Creates one_point_function_samples/
  engine_.EnsureDirectoryExists(two_point_function_raw_data_path + "dummy");   // Creates two_point_function_samples/
  DumpVecData(energy_raw_path + "/energy" + std::to_string(engine_.Rank()), sample_data_.energy_samples);
  DumpVecData(wf_amplitude_path + "/psi" + std::to_string(engine_.Rank()), sample_data_.wave_function_amplitude_samples);
  sample_data_.DumpOnePointFunctionSamples(
      one_point_function_raw_data_path + "/sample" + std::to_string(engine_.Rank()) + ".csv");
  sample_data_.DumpTwoPointFunctionSamples(
      two_point_function_raw_data_path + "/sample" + std::to_string(engine_.Rank()) + ".csv");
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

} // namespace qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MONTE_CARLO_MEASUREMENT_IMPL_H
