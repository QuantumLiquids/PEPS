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
#include "qlpeps/algorithm/vmc_update/model_solvers/heisenberg_square_pbc.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
#include "qlpeps/api/conversions.h"
#include "../test_mpi_env.h"
#include "../utilities.h"

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
          100, ConjugateGradientParams(100, 1e-5, 20, 0.001), 0.1),
      MonteCarloParams(100, 100, 1,
                       Configuration(Ly, Lx,
                                     OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})),  // Sz = 0
                       false),
      PEPSParams(trg_trunc_para));

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
  su_exe->update_para.Dmax = 3;
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

  HeisenbergSquarePBC model(J);

  // VMC optimization using PBC optimizer (TRGContractor)
  // Use exchange update to preserve total Sz symmetry
  auto executor = new VMCPEPSOptimizerPBC<TenElemT, QNT,
                                          MCUpdateSquareNNExchangePBC,
                                          HeisenbergSquarePBC>(
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

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
