// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2026-01-23
 *
 * Description: Integration test for Heisenberg PBC (4x4) using TRG + VMC optimization.
 * Workflow matches other integration tests: simple update (master) -> VMC optimization.
 */

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_j1j2_xxz_pbc.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
#include "qlpeps/api/conversions.h"
#include <optional>
#include "../test_mpi_env.h"
#include "../utilities.h"
#include <cmath>
#include <filesystem>

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;
using PEPST = SquareLatticePEPS<TenElemT, QNT>;
using SITPST = SplitIndexTPS<TenElemT, QNT>;

struct SquareHeisenbergPBCSystem : public MPITest {
  size_t Lx = 4;
  size_t Ly = 4;
  size_t Dpeps = 6;
  double J = 1.0;  // Isotropic Heisenberg coupling

  // ED ground state energy for 4x4 PBC Heisenberg
  // From: python tests/tools/pbc_benchmarks.py --model heisenberg --lx 4 --ly 4
  double energy_ed = -11.228483208428866;

  std::string model_name = "square_heisenberg_pbc_trg";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);

  IndexT pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  // Heisenberg Hamiltonian: H = J * sum_{<i,j>} S_i . S_j
  // Matrix elements in |↑↑>, |↑↓>, |↓↑>, |↓↓> basis:
  // S^z S^z: diag(1/4, -1/4, -1/4, 1/4)
  // S^+ S^- + S^- S^+: off-diag with 1/2 on |↑↓> <-> |↓↑>
  Tensor ham_nn = Tensor({pb_in, pb_out, pb_in, pb_out});

  // TRG truncation parameters for PBC
  TRGTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type> trg_trunc_para =
      TRGTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type>(
          /*d_min=*/1, /*d_max=*/16, /*trunc_error=*/1e-10);

  // VMC optimization parameters
  // Use exchange update to preserve total Sz
  VMCPEPSOptimizerParams vmc_peps_para = VMCPEPSOptimizerParams(
      OptimizerFactory::CreateStochasticReconfiguration(
          100, ConjugateGradientParams(100, 3e-3, 20, 0.001), 0.1),
      MonteCarloParams(5000, 100, 1,
                       Configuration(Ly, Lx,
                                     OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})),  // Sz = 0
                       false),
      PEPSParams(trg_trunc_para));
  std::optional<MCMeasurementParams> measure_para;

  void SetUp() override {
    MPITest::SetUp();

    // Heisenberg Hamiltonian: H = J * (S^z_i S^z_j + (S^+_i S^-_j + S^-_i S^+_j)/2)
    // In computational basis |0>=|↓>, |1>=|↑>:
    // S^z|0> = -1/2 |0>, S^z|1> = +1/2 |1>
    // S^+|0> = |1>, S^+|1> = 0
    // S^-|1> = |0>, S^-|0> = 0

    // Diagonal elements: J * S^z_i * S^z_j
    ham_nn({0, 0, 0, 0}) = TenElemT(J * 0.25);   // |↓↓> : (-1/2)*(-1/2) = 1/4
    ham_nn({1, 1, 1, 1}) = TenElemT(J * 0.25);   // |↑↑> : (+1/2)*(+1/2) = 1/4
    ham_nn({0, 0, 1, 1}) = TenElemT(-J * 0.25);  // |↓↑> : (-1/2)*(+1/2) = -1/4
    ham_nn({1, 1, 0, 0}) = TenElemT(-J * 0.25);  // |↑↓> : (+1/2)*(-1/2) = -1/4

    // Off-diagonal elements: J/2 * (S^+_i S^-_j + S^-_i S^+_j)
    // S^+_i S^-_j |↓↑> = |↑↓>, S^-_i S^+_j |↑↓> = |↓↑>
    ham_nn({0, 1, 1, 0}) = TenElemT(J * 0.5);  // |↓↑> -> |↑↓>
    ham_nn({1, 0, 0, 1}) = TenElemT(J * 0.5);  // |↑↓> -> |↓↑>

    // Measurement parameters (TRG + PBC)
    auto measure_mc = MonteCarloParams(
        2000, 40, 1,
        Configuration(Ly, Lx, OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})),
        false);
    measure_para.emplace(
        measure_mc, PEPSParams(trg_trunc_para),
        GetTestOutputPath("integration_heisenberg_pbc_trg", "measurement"));
  }
};

TEST_F(SquareHeisenbergPBCSystem, SimpleUpdate) {
  if (rank != hp_numeric::kMPIMasterRank) {
    return;
  }
  PEPST peps0(pb_out, Ly, Lx, BoundaryCondition::Periodic);

  // Initialize with Neel order (antiferromagnetic pattern)
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; ++y) {
    for (size_t x = 0; x < Lx; ++x) {
      activates[y][x] = (x + y) % 2;  // Checkerboard pattern
    }
  }
  peps0.Initial(activates);

  // Phase 1: Large step length warmup
  SimpleUpdatePara update_para(100, 0.1, 1, 4, 1e-15);
  auto su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>(
      update_para, peps0, ham_nn);
  su_exe->Execute();

  // Phase 2: Reduce step length
  su_exe->update_para.Dmax = 4;
  su_exe->update_para.Trunc_err = 1e-15;
  su_exe->ResetStepLenth(0.01);
  su_exe->Execute();

  // Phase 3: Fine optimization
  su_exe->update_para.Dmax = Dpeps;
  su_exe->update_para.Trunc_err = 1e-15;
  su_exe->ResetStepLenth(0.001);
  su_exe->Execute();

  // Save TPS
  auto tps = ToTPS<TenElemT, QNT>(su_exe->GetPEPS());
  for (auto &ten : tps) {
    ten *= (1.0 / ten.GetMaxAbs());
  }
  auto sitps = ToSplitIndexTPS<TenElemT, QNT>(tps);
  sitps.Dump(tps_path);
  delete su_exe;
}

TEST_F(SquareHeisenbergPBCSystem, StochasticReconfigurationOpt) {
  MPI_Barrier(comm);

  SITPST tps(Ly, Lx);
  if (rank == hp_numeric::kMPIMasterRank) {
    ASSERT_TRUE(tps.Load(tps_path));
  }
  qlpeps::MPI_Bcast(tps, comm);

  SquareSpinOneHalfJ1J2XXZModelPBC model(J, J, 0.0, 0.0, 0.0);

  // VMC optimization using PBC optimizer (TRGContractor)
  // Use exchange update to preserve total Sz symmetry
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

  // Get the best energy achieved during optimization
  double best_energy = executor->GetMinEnergy();
  std::cout << "Best energy during VMC: " << best_energy << std::endl;
  std::cout << "ED energy: " << energy_ed << std::endl;

  // Validate energy is approaching ED result
  // Using a tolerance since VMC may not fully converge to ED
  EXPECT_NEAR(best_energy, energy_ed, 1.0);

  tps = executor->GetState();
  delete executor;

  // Save optimized TPS
  tps.Dump(tps_path);
}

TEST_F(SquareHeisenbergPBCSystem, MeasurementPBC) {
  MPI_Barrier(comm);

  SITPST tps(Ly, Lx);
  if (rank == hp_numeric::kMPIMasterRank) {
    ASSERT_TRUE(tps.Load(tps_path));
  }
  qlpeps::MPI_Bcast(tps, comm);

  SquareSpinOneHalfJ1J2XXZModelPBC model(J, J, 0.0, 0.0, 0.0);
  auto executor = new MCPEPSMeasurer<TenElemT, QNT,
                                        MCUpdateSquareNNExchangePBC,
                                        SquareSpinOneHalfJ1J2XXZModelPBC,
                                        TRGContractor>(
      tps, *measure_para, comm, model);

  executor->Execute();

  if (rank == hp_numeric::kMPIMasterRank) {
    auto [energy, en_err] = executor->OutputEnergy();
    (void)en_err;
    EXPECT_NEAR(std::real(energy), energy_ed, 2.0);

    const auto &registry = executor->ObservableRegistry();
    EXPECT_NE(registry.find("energy"), registry.end());
    EXPECT_NE(registry.find("spin_z"), registry.end());
    EXPECT_NE(registry.find("bond_energy_h"), registry.end());
    EXPECT_NE(registry.find("bond_energy_v"), registry.end());
    if (registry.find("spin_z") != registry.end()) {
      EXPECT_EQ(registry.at("spin_z").first.size(), Ly * Lx);
    }
    if (registry.find("bond_energy_h") != registry.end()) {
      EXPECT_EQ(registry.at("bond_energy_h").first.size(), Ly * Lx);
    }
    if (registry.find("bond_energy_v") != registry.end()) {
      EXPECT_EQ(registry.at("bond_energy_v").first.size(), Ly * Lx);
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
