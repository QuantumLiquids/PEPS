// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Description: VMC optimization example for transverse-field Ising on 4x4.
*              Workflow: load PEPS (from SU dump), convert to SplitIndexTPS, run SR.
*              All parameters are hard-coded for tutorial simplicity.
*/

#include <iostream>
#include <string>

#include "qlten/qlten.h"
#include "qlpeps/api/conversions.h"                              // ToSplitIndexTPS
#include "qlpeps/api/vmc_api.h"                                  // VmcOptimize
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h" // MonteCarloParams, PEPSParams
#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h" // MCUpdateSquareNNExchange
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square_obc.h" // TransverseFieldIsingSquare
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"  // SquareLatticePEPS

using namespace qlten;
using namespace qlpeps;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // IMPORTANT: as a tutorial, force one thread per process for BLAS/HPTT/etc.
  hp_numeric::SetTensorManipulationThreads(1);

  try {
    using QNT = qlten::special_qn::TrivialRepQN;
    using TenElemT = QLTEN_Double;
    using SITPST = SplitIndexTPS<TenElemT, QNT>;

    const size_t Lx = 4;
    const size_t Ly = 4;
    const size_t phy_dim = 2;  // spin-1/2
    const double h = 0.5;      // transverse field

    // 1) Load PEPS dumped by Simple Update example
    //    Expect files under this directory produced by executor.DumpResult("PEPS", ...)
    const std::string peps_dump_path = "peps"; // unified with SU example

    // Build a dummy PEPS to load
    Index<QNT> pb_out({QNSector(QNT(), phy_dim)}, TenIndexDirType::OUT);
    SquareLatticePEPS<TenElemT, QNT> peps(pb_out, Ly, Lx);
    bool ok = peps.Load(peps_dump_path);
    if (!ok) {
      if (rank == 0) {
        std::cerr << "[TFI-VMC] Failed to load PEPS from path: " << peps_dump_path
                  << "\nPlease run the simple update example first and dump results." << std::endl;
      }
      MPI_Finalize();
      return 2;
    }

    // 2) Convert to SplitIndexTPS (PEPS -> TPS -> SplitIndexTPS)
    SITPST sitps = ToSplitIndexTPS<TenElemT, QNT>(peps);

    // 3) Prepare VMC parameters (SR)
    // Monte Carlo params
    Configuration init_config(Ly, Lx);
    // half-up/half-down random Ising configuration
    std::vector<size_t> occupancy = {Ly * Lx / 2, Ly * Lx / 2};
    init_config.Random(occupancy);
    MonteCarloParams mc_params(
        /*num_samples=*/500,
        /*num_warmup_sweeps=*/200,
        /*sweeps_between_samples=*/2,
        /*initial_config=*/init_config,
        /*is_warmed_up=*/false,
        /*config_dump_path=*/"./vmc_configs");

    // PEPS contraction params (BMPS truncation)
    // SVD compression: variational parameters are unused
    BMPSTruncateParams<qlten::QLTEN_Double> trunc_para = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(
        /*D_min=*/2,
        /*D_max=*/8,
        /*trunc_err=*/1e-14);
    PEPSParams peps_params(trunc_para);

    // Optimizer params: Stochastic Reconfiguration with CG
    ConjugateGradientParams cg_params(/*max_iter=*/100, /*tolerance=*/1e-5, /*restart=*/20, /*diag_shift=*/1e-3);
    auto opt_params = OptimizerFactory::CreateStochasticReconfiguration(
        /*max_iterations=*/40, cg_params, /*learning_rate=*/0.1);

    VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params, /*tps_dump_path=*/"./optimized_tps");

    // 4) Energy solver (model): transverse-field Ising on square lattice
    TransverseFieldIsingSquareOBC model(h);

    if (rank == 0) {
      std::cout << "[TFI-VMC] Start VMC optimize: 4x4, SR, h=" << h
                << ", samples=" << params.mc_params.num_samples
                << ", warmup=" << params.mc_params.num_warmup_sweeps
                << ", sweeps_between=" << params.mc_params.sweeps_between_samples
                << ", max_iter=" << params.optimizer_params.base_params.max_iterations
                << std::endl;
    }

    // 5) One-call optimization (type-deduced): returns executor pointer, already executed
    auto executor = VmcOptimize(
        params, sitps, MPI_COMM_WORLD, model, MCUpdateSquareNNFullSpaceUpdate{});

    if (rank == 0) {
      std::cout << "[TFI-VMC] Finished. Dumping optimized TPS..." << std::endl;
    }
    executor->DumpData(/*release_mem=*/false);

    if (rank == 0) {
      std::cout << "[TFI-VMC] Done." << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "[TFI-VMC] Error: " << e.what() << std::endl;
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}
