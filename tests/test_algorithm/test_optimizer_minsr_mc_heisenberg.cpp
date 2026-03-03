// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-03-02
*
* Description: MinSR integration test for 2x2 Heisenberg OBC model.
*   Exercises the full VmcOptimize API -> VMCPEPSOptimizer -> MinSR pipeline.
*   Run with mpirun -np 4.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/api/vmc_api.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

#include "../test_mpi_env.h"
#include "../utilities.h"

using namespace qlten;
using namespace qlpeps;

using TenElemT = TEN_ELEM_TYPE;
using QNT = qlten::special_qn::TrivialRepQN;
using SITPST = SplitIndexTPS<TenElemT, QNT>;

struct MinSRHeisenbergTest : MPITest {
  static constexpr size_t kLy = 2;
  static constexpr size_t kLx = 2;

  SITPST LoadPreGeneratedTPS() const {
    std::string type_suffix;
    if constexpr (std::is_same_v<TenElemT, QLTEN_Double>) {
      type_suffix = "_double_from_simple_update";
    } else {
      type_suffix = "_complex_from_simple_update";
    }
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/heisenberg_tps" + type_suffix;

    SITPST sitps(kLy, kLx);
    int load_ok = 0;
    if (rank == 0) {
      load_ok = sitps.Load(data_path) ? 1 : 0;
    }
    MPI_Bcast(&load_ok, 1, MPI_INT, 0, comm);
    if (!load_ok) {
      throw std::runtime_error("Failed to load Heisenberg TPS from: " + data_path);
    }
    if (mpi_size > 1) {
      qlpeps::MPI_Bcast(sitps, comm, 0);
    }
    return sitps;
  }
};

TEST_F(MinSRHeisenbergTest, EnergyConvergence) {
  constexpr double kExactEnergy = -2.0;
  constexpr size_t kTotalSamples = 200;
  constexpr size_t kWarmupSweeps = 20;
  constexpr size_t kSweepsBetween = 1;
  constexpr size_t kMaxIter = 100;
  constexpr double kLearningRate = 0.02;

  SITPST sitps = LoadPreGeneratedTPS();

  // Initial configuration: half-filling (2 up, 2 down)
  Configuration init_config(kLy, kLx);
  init_config({0, 0}) = 1;
  init_config({0, 1}) = 0;
  init_config({1, 0}) = 0;
  init_config({1, 1}) = 1;

  // MinSR optimizer params via builder
  MinSRParams minsr_alg(/*r_pinv=*/1e-4, /*a_pinv=*/0.0, /*soft_cutoff=*/true,
                        MinSRSolverMode::kReplicated);
  auto opt_params = OptimizerParamsBuilder()
      .SetMaxIterations(kMaxIter)
      .SetLearningRate(kLearningRate)
      .WithMinSR(minsr_alg)
      .Build();
  opt_params.base_params.jsonl_log_path = "minsr_heisenberg_2x2_log.jsonl";

  auto trun_para = BMPSTruncateParams<QLTEN_Double>::SVD(/*D_min=*/1, /*D_max=*/8, /*trunc_error=*/1e-16);
  MonteCarloParams mc_params(kTotalSamples, kWarmupSweeps, kSweepsBetween, init_config, false);
  PEPSParams peps_params(trun_para);

  // Use empty tps_dump_path to suppress TPS file I/O
  VMCPEPSOptimizerParams vmc_params(opt_params, mc_params, peps_params, /*tps_dump_path=*/"");

  SquareSpinOneHalfXXZModelOBC model(/*jz=*/1.0, /*jxy=*/1.0, /*pinning00=*/0.0);

  auto result = VmcOptimize<TenElemT, QNT,
                            MCUpdateSquareNNExchange,
                            SquareSpinOneHalfXXZModelOBC>(
      vmc_params, sitps, comm, model);

  if (rank == hp_numeric::kMPIMasterRank) {
    double min_energy = result.min_energy;
    const auto &traj = result.energy_trajectory;

    ASSERT_FALSE(traj.empty());

    std::cout << "MinSR final energy: " << std::real(traj.back())
              << ", min energy: " << min_energy
              << ", exact: " << kExactEnergy << std::endl;

    // The energy should be close to the exact value (within MC error)
    EXPECT_GT(min_energy, kExactEnergy - 0.1)
        << "Energy went below exact ground state (numerical issue)";
    EXPECT_LT(min_energy, kExactEnergy + 0.5)
        << "MinSR failed to converge to near ground state energy";
  }
}

int main(int argc, char *argv[]) {
  testing::AddGlobalTestEnvironment(new MPIEnvironment);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  return RUN_ALL_TESTS();
}
