// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-27
*
* Description: QuantumLiquids/PEPS project. Smoke test for independent MC
*              energy+gradient evaluator on a tiny 2x2 system.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/mc_energy_grad_evaluator.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include <filesystem>

#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

#if TEN_ELEM_TYPE_NUM == 1
std::string data_type_in_file_name = "Double";
#elif TEN_ELEM_TYPE_NUM == 2
std::string data_type_in_file_name = "Complex";
#else
#error "Unexpected TEN_ELEM_TYPE_NUM"
#endif

struct SmokeEvaluator2x2 : MPITest {
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::TrivialRepQN;
  using IndexT = Index<QNT>;

  size_t Lx = 2;
  size_t Ly = 2;
  size_t Dpeps = 4;

  void SetUp(void) {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);
  }
};

// Use Heisenberg XXZ model as a quick running Hamiltonian; any model works for smoke
TEST_F(SmokeEvaluator2x2, EvaluateEnergyAndGradient) {
  using Model = SquareSpinOneHalfXXZModel;

  // Prepare TPS from test data (already small and compatible)
  std::string type_tag = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ? std::string("double") : std::string("complex");
  std::filesystem::path tps_path_p = std::filesystem::path(TEST_SOURCE_DIR) /
                                     ("test_data/heisenberg_tps_" + type_tag + "lowest");
  std::string tps_path = tps_path_p.string();

  SplitIndexTPS<TenElemT, QNT> sitps(Ly, Lx);
  ASSERT_TRUE(sitps.Load(tps_path));

  // MC params: small and fast
  Configuration config(Ly, Lx);
  config.Random(std::vector<size_t>(2, Lx * Ly / 2));
  MonteCarloParams mc_params(50, 20, 1, config, false);
  PEPSParams peps_params(BMPSTruncatePara(Dpeps, 2 * Dpeps, 1e-15,
                                          CompressMPSScheme::SVD_COMPRESS,
                                          std::make_optional<double>(1e-14),
                                          std::make_optional<size_t>(10)));

  MonteCarloEngine<TenElemT, QNT, MCUpdateSquareNNExchange> engine(sitps, mc_params, peps_params, comm);

  // Simple isotropic Heisenberg J=1
  Model heisenberg_solver(1.0, 1.0, 0.0);

  // Evaluator with SR buffers disabled
  MCEnergyGradEvaluator<TenElemT, QNT, MCUpdateSquareNNExchange, Model> evaluator(engine, heisenberg_solver, comm, false);

  auto result = evaluator.Evaluate(sitps);

  // Basic sanity checks
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    EXPECT_TRUE(std::isfinite(Real(result.energy)));
    EXPECT_TRUE(std::isfinite(result.energy_error) || result.energy_error == 0.0);
    EXPECT_GE(result.gradient.NormSquare(), 0.0);
    EXPECT_FALSE(result.accept_rates_avg.empty());
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}


