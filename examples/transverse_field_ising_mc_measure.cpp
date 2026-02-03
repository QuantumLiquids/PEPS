// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Description: Monte Carlo measurement example for transverse-field Ising on 4x4.
*              Workflow: load optimized SplitIndexTPS (from VMC tutorial), then measure observables.
*/

#include <iostream>
#include <string>

#include "qlten/qlten.h"
#include "qlpeps/api/vmc_api.h" // MonteCarloMeasure
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h" // MCUpdateSquareNNFullSpaceUpdate
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square_obc.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"

using namespace qlten;
using namespace qlpeps;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // As a tutorial/example: force single thread per MPI rank for BLAS/HPTT/etc.
  hp_numeric::SetTensorManipulationThreads(1);

  try {
    using QNT = qlten::special_qn::TrivialRepQN;
    using TenElemT = QLTEN_Double;
    using SITPST = SplitIndexTPS<TenElemT, QNT>;

    const double h = 0.5;
    const std::string sitps_path = (argc >= 2) ? std::string(argv[1]) : std::string("./optimized_tps");

    // 1) Load optimized state dumped by the VMC example.
    SITPST sitps;
    if (!sitps.Load(sitps_path)) {
      if (rank == 0) {
        std::cerr << "[TFI-MEAS] Failed to load SplitIndexTPS from: " << sitps_path << "\n"
                  << "Please run examples/transverse_field_ising_vmc_optimize.cpp first.\n";
      }
      MPI_Finalize();
      return 2;
    }

    const size_t Ly = sitps.rows();
    const size_t Lx = sitps.cols();

    // 2) Build measurement parameters.
    Configuration init_config(Ly, Lx);
    init_config.Random(OccupancyNum{Ly * Lx / 2, Ly * Lx / 2}); // TFIM: local dim=2

    MonteCarloParams mc_params(
        /*total_samples=*/2000,
        /*num_warmup_sweeps=*/200,
        /*sweeps_between_samples=*/2,
        /*initial_config=*/init_config,
        /*is_warmed_up=*/false,
        /*config_dump_path=*/""); // set a path if you want to dump final configs

    BMPSTruncateParams<double> trunc_para = BMPSTruncateParams<double>::SVD(
        /*D_min=*/2,
        /*D_max=*/8,
        /*trunc_err=*/1e-14);
    PEPSParams peps_params(trunc_para);

    MCMeasurementParams meas_params(mc_params, peps_params, "./mc_measure_output");

    // 3) Measurement solver (model)
    TransverseFieldIsingSquareOBC model(h);

    if (rank == 0) {
      std::cout << "[TFI-MEAS] Start Monte Carlo measurement: Ly=" << Ly << ", Lx=" << Lx
                << ", h=" << h
                << ", total_samples=" << mc_params.total_samples
                << ", warmup=" << mc_params.num_warmup_sweeps
                << ", sweeps_between=" << mc_params.sweeps_between_samples
                << ", dump=" << meas_params.measurement_data_dump_path
                << std::endl;
    }

    // 4) One-call measurement wrapper
    auto result = MonteCarloMeasure(sitps, meas_params, MPI_COMM_WORLD, model, MCUpdateSquareNNFullSpaceUpdate{});

    if (rank == 0) {
      std::cout << "[TFI-MEAS] Done.\n"
                << "  Energy = " << result.energy.first << " ± " << result.energy.second << "\n"
                << "  Dump path = " << result.dump_path << "\n"
                << "  See: " << result.dump_path << "/stats/\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "[TFI-MEAS] Error: " << e.what() << std::endl;
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}
