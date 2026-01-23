// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Description: Integration test for TFIM PBC (4x4) using TRG + VMC optimization.
 * Workflow matches other integration tests: simple update (master) -> VMC optimization.
 * 
 * Note: MCPEPSMeasurerPBC cannot be used because TransverseFieldIsingSquarePBC
 * does not implement EvaluateObservables for TRGContractor. Energy validation
 * is done via the VMC optimizer's output.
 */

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square_pbc.h"
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

struct SquareTFIMPBCSystem : public MPITest {
  size_t Lx = 4;
  size_t Ly = 4;
  size_t Dpeps = 6;
  double J = 1.0;
  double h = 1.0;

  // ED ground state energy for 4x4 PBC TFIM with h=1.0
  // From: python tests/tools/pbc_benchmarks.py --model ising --lx 4 --ly 4 --param 1.0
  double energy_ed = -34.01059755084629;

  std::string model_name = "square_tfim_pbc_trg";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);

  IndexT pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);
  Tensor ham_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  Tensor ham_onsite = Tensor({pb_in, pb_out});

  // TRG truncation parameters for PBC
  TRGTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type> trg_trunc_para =
      TRGTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type>(
          /*d_min=*/1, /*d_max=*/8, /*trunc_error=*/1e-10);

  // VMC optimization parameters
  // For TFIM, physical dimension is 2 (spin up/down), no U(1) symmetry constraint
  // Use smaller learning rate (0.1) since Simple Update already gives good initial state
  VMCPEPSOptimizerParams vmc_peps_para = VMCPEPSOptimizerParams(
      OptimizerFactory::CreateStochasticReconfiguration(
          40, ConjugateGradientParams(100, 1e-5, 20, 0.001), 0.1),
      MonteCarloParams(100, 100, 1,
                       Configuration(Ly, Lx, 2),  // Physical dimension 2, random init
                       false),
      PEPSParams());

  void SetUp() override {
    MPITest::SetUp();
    // -J sz sz
    const double z_vals[2] = {+1.0, -1.0};
    for (size_t s1 = 0; s1 < 2; ++s1) {
      for (size_t s2 = 0; s2 < 2; ++s2) {
        ham_nn({s1, s1, s2, s2}) = TenElemT(-J * z_vals[s1] * z_vals[s2]);
      }
    }
    // -h sx (off-diagonal)
    ham_onsite({0, 1}) = TenElemT(-h);
    ham_onsite({1, 0}) = TenElemT(-h);

    // Set TRG params for VMC
    vmc_peps_para.peps_params.trg_truncate_para = trg_trunc_para;
  }
};

TEST_F(SquareTFIMPBCSystem, SimpleUpdate) {
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

  // Phase 1: Large step length warmup
  SimpleUpdatePara update_para(100, 0.1, 1, 4, 1e-15);
  auto su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>(
      update_para, peps0, ham_nn, ham_onsite);
  su_exe->Execute();

  // Phase 2: Reduce step length
  su_exe->update_para.Dmax = 6;
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

TEST_F(SquareTFIMPBCSystem, StochasticReconfigurationOpt) {
  MPI_Barrier(comm);

  SITPST tps(Ly, Lx);
  if (rank == hp_numeric::kMPIMasterRank) {
    ASSERT_TRUE(tps.Load(tps_path));
  }
  qlpeps::MPI_Bcast(tps, comm);

  TransverseFieldIsingSquarePBC model(h);

  // VMC optimization using PBC optimizer (TRGContractor)
  auto executor = new VMCPEPSOptimizerPBC<TenElemT, QNT,
                                          MCUpdateSquareNNFullSpaceUpdatePBC,
                                          TransverseFieldIsingSquarePBC>(
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
