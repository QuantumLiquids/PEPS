// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-21
*
* Description: QuantumLiquids/PEPS project. Integration testing for Square Heisenberg model.
*/

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/api/vmc_api.h"
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

// Test spin systems
struct HeisenbergSystem : public MPITest {
  size_t Lx = 3;
  size_t Ly = 4;

  size_t Dpeps = 6;
  QNT qn0 = QNT();
  double energy_ed = -6.691680193514947;
  IndexT pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  Tensor ham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  Tensor ham_hei_tri;  // three-site hamiltonian in triangle lattice

  std::string model_name = "square_afm_heisenberg";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);

  VMCPEPSOptimizerParams vmc_peps_para = VMCPEPSOptimizerParams(
      OptimizerFactory::CreateStochasticReconfiguration(40, ConjugateGradientParams(100, 1e-5, 20, 0.001), 0.3),
      MonteCarloParams(100, 100, 1,
                       Configuration(Ly, Lx,
                                     OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})),
                       false), // Sz = 0, not warmed up initially
      PEPSParams(BMPSTruncatePara(6, 12, 1e-15, CompressMPSScheme::SVD_COMPRESS,
                                  std::make_optional<double>(1e-14), std::make_optional<size_t>(10))));

  MonteCarloParams measure_mc_params{1000, 1000, 1,
                                      Configuration(Ly, Lx,
                                                    OccupancyNum({Lx * Ly / 2, Lx * Ly / 2})), // Random generate configuration with Sz = 0
                                      false}; // not warmed up initially
  PEPSParams measure_peps_params{BMPSTruncatePara(Dpeps, 2 * Dpeps, 1e-15,
                                                  CompressMPSScheme::SVD_COMPRESS,
                                                  std::make_optional<double>(1e-14),
                                                  std::make_optional<size_t>(10))};
  MCMeasurementParams measure_para{measure_mc_params, measure_peps_params};

  void SetUp() {
    MPITest::SetUp();
    ham_hei_nn({0, 0, 0, 0}) = 0.25;
    ham_hei_nn({1, 1, 1, 1}) = 0.25;
    ham_hei_nn({1, 1, 0, 0}) = -0.25;
    ham_hei_nn({0, 0, 1, 1}) = -0.25;
    ham_hei_nn({0, 1, 1, 0}) = 0.5;
    ham_hei_nn({1, 0, 0, 1}) = 0.5;
  }
};

TEST_F(HeisenbergSystem, SimpleUpdate) {
  if (rank == hp_numeric::kMPIMasterRank) {
    SquareLatticePEPS<TenElemT, QNT> peps0(pb_out, Ly, Lx);
    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        size_t sz_int = x + y;
        activates[y][x] = sz_int % 2;
      }
    }
    peps0.Initial(activates);

    SimpleUpdatePara update_para(1000, 0.1, 1, 4, 1e-15);
    SimpleUpdateExecutor<TenElemT, QNT>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>(update_para, peps0,
                                                                         ham_hei_nn);
    su_exe->Execute();

    su_exe->update_para.Dmax = 6;
    su_exe->update_para.Trunc_err = 1e-15;
    su_exe->ResetStepLenth(0.01);
    su_exe->Execute();

    su_exe->update_para.Dmax = 6;
    su_exe->update_para.Trunc_err = 1e-15;
    su_exe->ResetStepLenth(0.001);
    su_exe->Execute();

    auto tps = ToTPS<TenElemT, QNT>(su_exe->GetPEPS());
//  std::string peps_path = "Hei_PEPS" + std::to_string(Ly) + "x"
//      + std::to_string(Lx) + "D" + std::to_string(su_exe->update_para.Dmax);
//  su_exe->DumpResult(peps_path, true);
    for (auto &ten : tps) {
      ten *= (1.0 / ten.GetMaxAbs());
    }
    auto sitps = ToSplitIndexTPS<TenElemT, QNT>(tps);
    sitps.Dump(tps_path);
    delete su_exe;
  }
}

// Check if the TPS doesn't change by setting step length = 0
TEST_F(HeisenbergSystem, SGDWithZeroLR) {
  MPI_Barrier(comm);
  vmc_peps_para.optimizer_params.base_params.learning_rate = 0.0;
  vmc_peps_para.optimizer_params.base_params.max_iterations = 3;
  // Switch to SGD (zero LR) to exclude SR/CG MPI path for debugging
  vmc_peps_para.optimizer_params = OptimizerFactory::CreateSGDWithDecay(
      3 /*max_iterations*/, 0.0 /*initial_lr*/, 1.0 /*decay_rate*/, 1 /*decay_steps*/);
  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "[ZeroUpdate] Using SGD with zero learning rate to disable SR/CG path" << std::endl;
  }
  SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
  if (rank == hp_numeric::kMPIMasterRank) {
    tps.Load(tps_path);
  }
  qlpeps::MPI_Bcast(tps, comm);
  auto init_tps = tps;
  auto executor_ptr = VmcOptimize<TenElemT, QNT, MCUpdateSquareNNExchange, SquareSpinOneHalfXXZModel>(
      vmc_peps_para, tps, comm, SquareSpinOneHalfXXZModel());
  size_t start_flop = flop;
  Timer vmc_timer("vmc");
  // already executed inside VmcOptimize
  size_t end_flop = flop;
  double elapsed_time = vmc_timer.Elapsed();
  double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "flop = " << end_flop - start_flop << std::endl;
  std::cout << "Gflops = " << Gflops << std::endl;
  SplitIndexTPS<TenElemT, QNT> result_sitps = executor_ptr->GetState();
  init_tps.NormalizeAllSite();
  result_sitps.NormalizeAllSite();
  auto diff = init_tps + (-result_sitps);
  EXPECT_NE(diff.NormSquare(), 1e-14);
  // unique_ptr auto cleanup
}

TEST_F(HeisenbergSystem, StochasticReconfigurationOpt) {
  MPI_Barrier(comm);

  SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
  if (rank == hp_numeric::kMPIMasterRank) {
    tps.Load(tps_path);
  }
  qlpeps::MPI_Bcast(tps, comm);

  //VMC
  auto executor_ptr = VmcOptimize<TenElemT, QNT, MCUpdateSquareNNExchange, SquareSpinOneHalfXXZModel>(
      vmc_peps_para, tps, comm, SquareSpinOneHalfXXZModel());
  size_t start_flop = flop;
  Timer vmc_timer("vmc");

  // already executed inside VmcOptimize

  size_t end_flop = flop;
  double elapsed_time = vmc_timer.PrintElapsed();
  double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "Gflops = " << Gflops / elapsed_time << std::endl;

  tps = executor_ptr->GetState();
  // unique_ptr auto cleanup

  //Measure
  auto measure_exe_ptr = MonteCarloMeasure<TenElemT, QNT, MCUpdateSquareNNExchange, SquareSpinOneHalfXXZModel>(
      tps, measure_para, comm, SquareSpinOneHalfXXZModel());
  start_flop = flop;

  // already executed inside MonteCarloMeasure

  end_flop = flop;
  elapsed_time = vmc_timer.PrintElapsed();
  Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "Gflops = " << Gflops / elapsed_time << std::endl;

  auto [energy, en_err] = measure_exe_ptr->OutputEnergy();
  EXPECT_NEAR(std::real(energy), energy_ed, 0.001);
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
