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
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square_pbc.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
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

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> BuildTFIMSimpleUpdatePBC(
    size_t n, double J, double h, size_t dmax, size_t steps, double tau) {
  using Tensor = qlten::QLTensor<TenElemT, QNT>;
  using IndexT = qlten::Index<QNT>;
  using QNSctT = qlten::QNSector<QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;

  IndexT pb_out({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  Tensor ham_nn({pb_in, pb_out, pb_in, pb_out});
  Tensor ham_onsite({pb_in, pb_out});

  const double z_vals[2] = {+1.0, -1.0};
  for (size_t s1 = 0; s1 < 2; ++s1) {
    for (size_t s2 = 0; s2 < 2; ++s2) {
      ham_nn({s1, s1, s2, s2}) = TenElemT(-J * z_vals[s1] * z_vals[s2]);
    }
  }
  ham_onsite({0, 1}) = TenElemT(-h);
  ham_onsite({1, 0}) = TenElemT(-h);

  PEPST peps0(pb_out, n, n, BoundaryCondition::Periodic);
  std::vector<std::vector<size_t>> activates(n, std::vector<size_t>(n, 0));
  for (size_t y = 0; y < n; ++y) {
    for (size_t x = 0; x < n; ++x) {
      activates[y][x] = (x + y) % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(steps, tau, /*Dmin=*/1, /*Dmax=*/dmax, /*Trunc_err=*/1e-6);
  SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT> su_exe(update_para, peps0, ham_nn, ham_onsite);
  su_exe.Execute();

  auto tps = qlpeps::ToTPS<TenElemT, QNT>(su_exe.GetPEPS());
  return SplitIndexTPS<TenElemT, QNT>::FromTPS(tps);
}

// Use Heisenberg XXZ model as a quick running Hamiltonian; any model works for smoke
TEST_F(SmokeEvaluator2x2, EvaluateEnergyAndGradient) {
  using Model = SquareSpinOneHalfXXZModelOBC;

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
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>(Dpeps, 2 * Dpeps, 1e-15,
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
    EXPECT_TRUE(std::isfinite(std::real(result.energy)));
    EXPECT_TRUE(std::isfinite(result.energy_error) || result.energy_error == 0.0);
    EXPECT_GE(result.gradient.NormSquare(), 0.0);
    EXPECT_FALSE(result.accept_rates_avg.empty());
  }
}

TEST_F(SmokeEvaluator2x2, EvaluateTFIMPBC_TRG_8x8_Smoke) {
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::TrivialRepQN;

  constexpr size_t n = 8;
  SplitIndexTPS<TenElemT, QNT> sitps;
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    sitps = BuildTFIMSimpleUpdatePBC<TenElemT, QNT>(
        n, /*J=*/1.0, /*h=*/1.0, /*dmax=*/2, /*steps=*/4, /*tau=*/0.5);
  }
  qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);

  Configuration config(n, n);
  config.Random(std::vector<size_t>(2, n * n / 2));
  MonteCarloParams mc_params(/*samples=*/5, /*warmup_sweeps=*/2, /*sweeps_between=*/1, config, false);

  PEPSParams peps_params(TRGTruncateParams<qlten::QLTEN_Double>::SVD(/*d_min=*/1, /*d_max=*/4, /*trunc_error=*/0.0));

  MonteCarloEngine<TenElemT, QNT, MCUpdateSquareNNFullSpaceUpdatePBC, TRGContractor> engine(
      sitps, mc_params, peps_params, comm);

  TransverseFieldIsingSquarePBC model(/*h=*/1.0);
  MCEnergyGradEvaluator<TenElemT, QNT, MCUpdateSquareNNFullSpaceUpdatePBC, TransverseFieldIsingSquarePBC, TRGContractor>
      evaluator(engine, model, comm, false);

  auto result = evaluator.Evaluate(sitps);
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    EXPECT_TRUE(std::isfinite(std::real(result.energy)));
    EXPECT_TRUE(std::isfinite(result.energy_error) || result.energy_error == 0.0);
    EXPECT_FALSE(result.accept_rates_avg.empty());
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
