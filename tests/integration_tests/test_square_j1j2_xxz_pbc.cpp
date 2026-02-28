// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2026-01-25
 *
 * Description: Integration test for J1-J2 XXZ PBC using TRG + VMC optimization.
 */

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_j1j2_xxz_pbc.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nnn_simple_update.h"
#include "qlpeps/api/conversions.h"
#include "../test_mpi_env.h"
#include "../utilities.h"
#include <cmath>
#include <filesystem>
#include <optional>

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;
using PEPST = SquareLatticePEPS<TenElemT, QNT>;
using SITPST = SplitIndexTPS<TenElemT, QNT>;

struct SquareJ1J2XXZPBCSystem : public MPITest {
  size_t Lx = 4;
  size_t Ly = 4;
  size_t Dpeps = 6;

  double jz1 = 0.5;
  double jxy1 = 1.0;
  double jz2 = -0.2;
  double jxy2 = -0.3;

  // ED ground state energy for 4x4 PBC J1-J2 XXZ
  // From: python tests/tools/pbc_benchmarks.py --model j1j2_xxz --lx 4 --ly 4 --jz1 0.5 --jxy1 1.0 --jz2 -0.2 --jxy2 -0.3
  double energy_ed = -12.066009559762076;

  std::string model_name = "square_j1j2_xxz_pbc_trg";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);

  IndexT pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  Tensor ham_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  Tensor ham_nnn = Tensor({pb_in, pb_out, pb_in, pb_out});

  TRGTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type> trg_trunc_para =
      TRGTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type>(
          /*d_min=*/1, /*d_max=*/16, /*trunc_error=*/1e-10);

  VMCPEPSOptimizerParams vmc_peps_para = VMCPEPSOptimizerParams(
      OptimizerFactory::CreateStochasticReconfiguration(
          100,
          StochasticReconfigurationParams{.cg_params = ConjugateGradientParams{.max_iter = 100, .relative_tolerance = 3e-3,
                                                                                .residual_recompute_interval = 20},
                                          .diag_shift = 0.001}, 0.1),
      MonteCarloParams(5000, 100, 1,
                       Configuration(Ly, Lx,
                                     OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})),  // Sz = 0
                       false),
      PEPSParams(trg_trunc_para));
  std::optional<MCMeasurementParams> measure_para;

  void SetUp() override {
    MPITest::SetUp();

    ham_nn({0, 0, 0, 0}) = TenElemT(jz1 * 0.25);
    ham_nn({1, 1, 1, 1}) = TenElemT(jz1 * 0.25);
    ham_nn({0, 0, 1, 1}) = TenElemT(-jz1 * 0.25);
    ham_nn({1, 1, 0, 0}) = TenElemT(-jz1 * 0.25);
    ham_nn({0, 1, 1, 0}) = TenElemT(jxy1 * 0.5);
    ham_nn({1, 0, 0, 1}) = TenElemT(jxy1 * 0.5);

    ham_nnn({0, 0, 0, 0}) = TenElemT(jz2 * 0.25);
    ham_nnn({1, 1, 1, 1}) = TenElemT(jz2 * 0.25);
    ham_nnn({0, 0, 1, 1}) = TenElemT(-jz2 * 0.25);
    ham_nnn({1, 1, 0, 0}) = TenElemT(-jz2 * 0.25);
    ham_nnn({0, 1, 1, 0}) = TenElemT(jxy2 * 0.5);
    ham_nnn({1, 0, 0, 1}) = TenElemT(jxy2 * 0.5);

    auto measure_mc = MonteCarloParams(
        2000, 40, 1,
        Configuration(Ly, Lx, OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})),
        false);
    measure_para.emplace(
        measure_mc, PEPSParams(trg_trunc_para),
        GetTestOutputPath("integration_j1j2_xxz_pbc_trg", "measurement"));
  }
};

TEST_F(SquareJ1J2XXZPBCSystem, SimpleUpdate) {
  if (rank != hp_numeric::kMPIMasterRank) {
    return;
  }
  PEPST peps0(pb_out, Ly, Lx, BoundaryCondition::Periodic);

  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; ++y) {
    for (size_t x = 0; x < Lx; ++x) {
      activates[y][x] = (x + y) % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(100, 0.1, 1, 4, 1e-15);
  auto su_exe = new SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>(
      update_para, peps0, ham_nn, ham_nnn);
  su_exe->Execute();

  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-15;
  su_exe->ResetStepLenth(0.01);
  su_exe->Execute();

  su_exe->update_para.Dmax = Dpeps;
  su_exe->update_para.Trunc_err = 1e-15;
  su_exe->ResetStepLenth(0.001);
  su_exe->Execute();

  auto tps = ToTPS<TenElemT, QNT>(su_exe->GetPEPS());
  for (auto &ten : tps) {
    ten *= (1.0 / ten.GetMaxAbs());
  }
  auto sitps = ToSplitIndexTPS<TenElemT, QNT>(tps);
  sitps.Dump(tps_path);
  delete su_exe;
}

TEST_F(SquareJ1J2XXZPBCSystem, StochasticReconfigurationOpt) {
  MPI_Barrier(comm);

  SITPST tps(Ly, Lx);
  if (rank == hp_numeric::kMPIMasterRank) {
    ASSERT_TRUE(tps.Load(tps_path));
  }
  qlpeps::MPI_Bcast(tps, comm);

  SquareSpinOneHalfJ1J2XXZModelPBC model(jz1, jxy1, jz2, jxy2, 0.0);

  auto executor = new VMCPEPSOptimizer<TenElemT, QNT,
                                          MCUpdateSquareNNExchangePBC,
                                          SquareSpinOneHalfJ1J2XXZModelPBC,
                                          TRGContractor>(
      vmc_peps_para, tps, comm, model);

  size_t start_flop = flop;
  Timer vmc_timer("vmc");

  executor->Execute();

  size_t end_flop = flop;
  double elapsed_time = vmc_timer.PrintElapsed();
  double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "VMC Gflops = " << Gflops << std::endl;

  double best_energy = executor->GetMinEnergy();
  std::cout << "Best energy during VMC: " << best_energy << std::endl;
  std::cout << "ED energy: " << energy_ed << std::endl;
  EXPECT_NEAR(best_energy, energy_ed, 2.0);

  tps = executor->GetState();
  delete executor;

  tps.Dump(tps_path);
}

TEST_F(SquareJ1J2XXZPBCSystem, MeasurementPBC) {
  MPI_Barrier(comm);

  SITPST tps(Ly, Lx);
  if (rank == hp_numeric::kMPIMasterRank) {
    ASSERT_TRUE(tps.Load(tps_path));
  }
  qlpeps::MPI_Bcast(tps, comm);

  SquareSpinOneHalfJ1J2XXZModelPBC model(jz1, jxy1, jz2, jxy2, 0.0);
  auto executor = new MCPEPSMeasurer<TenElemT, QNT,
                                        MCUpdateSquareNNExchangePBC,
                                        SquareSpinOneHalfJ1J2XXZModelPBC,
                                        TRGContractor>(
      tps, *measure_para, comm, model);

  executor->Execute();

  if (rank == hp_numeric::kMPIMasterRank) {
    auto [energy, en_err] = executor->OutputEnergy();
    (void)en_err;
    EXPECT_NEAR(std::real(energy), energy_ed, 3.0);

    const auto &registry = executor->ObservableRegistry();
    EXPECT_NE(registry.find("energy"), registry.end());
    EXPECT_NE(registry.find("spin_z"), registry.end());
    EXPECT_NE(registry.find("bond_energy_h"), registry.end());
    EXPECT_NE(registry.find("bond_energy_v"), registry.end());
    EXPECT_NE(registry.find("bond_energy_dr"), registry.end());
    EXPECT_NE(registry.find("bond_energy_ur"), registry.end());
    if (registry.find("spin_z") != registry.end()) {
      EXPECT_EQ(registry.at("spin_z").first.size(), Ly * Lx);
    }
    if (registry.find("bond_energy_h") != registry.end()) {
      EXPECT_EQ(registry.at("bond_energy_h").first.size(), Ly * Lx);
    }
    if (registry.find("bond_energy_v") != registry.end()) {
      EXPECT_EQ(registry.at("bond_energy_v").first.size(), Ly * Lx);
    }
    if (registry.find("bond_energy_dr") != registry.end()) {
      EXPECT_EQ(registry.at("bond_energy_dr").first.size(), Ly * Lx);
    }
    if (registry.find("bond_energy_ur") != registry.end()) {
      EXPECT_EQ(registry.at("bond_energy_ur").first.size(), Ly * Lx);
    }
    if (registry.find("energy") != registry.end() && !registry.at("energy").first.empty()) {
      EXPECT_TRUE(std::isfinite(std::real(registry.at("energy").first.front())));
    }

    const std::filesystem::path stats_dir =
        std::filesystem::path(measure_para->measurement_data_dump_path) / "stats";
    ASSERT_TRUE(std::filesystem::exists(stats_dir));
    ASSERT_TRUE(std::filesystem::is_directory(stats_dir));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "energy.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "spin_z_mean.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "spin_z_stderr.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "bond_energy_h_mean.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "bond_energy_h_stderr.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "bond_energy_v_mean.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "bond_energy_v_stderr.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "bond_energy_dr_mean.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "bond_energy_dr_stderr.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "bond_energy_ur_mean.csv"));
    ASSERT_TRUE(std::filesystem::exists(stats_dir / "bond_energy_ur_stderr.csv"));
  }

  delete executor;
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
