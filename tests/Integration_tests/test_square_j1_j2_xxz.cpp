// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-06-11
*
* Description: QuantumLiquids/PEPS project. Integration testing for J1-J2 XXZ Model.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/qlpeps.h"
#include "../test_mpi_env.h"
#include "../utilities.h"

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;

struct SpinOneHalfSystem : public MPITest {
  size_t Lx = 3;
  size_t Ly = 4;
  double jz1 = 0.5;
  double jxy1 = 1;
  double jz2 = -0.2; // no frustration
  double jxy2 = -0.3;

  double energy_ed = -6.523925897312232;

  size_t Dpeps = 6;
  QNT qn0 = QNT();

  IndexT pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  IndexT pb_in = InverseIndex(pb_out);

  Tensor ham_hei_nn = Tensor({pb_in, pb_out, pb_in, pb_out});
  Tensor ham_hei_nnn = Tensor({pb_in, pb_out, pb_in, pb_out});
  VMCOptimizePara optimize_para =
      VMCOptimizePara(BMPSTruncatePara(6, 12, 1e-15,
                                       CompressMPSScheme::SVD_COMPRESS,
                                       std::make_optional<double>(1e-14),
                                       std::make_optional<size_t>(10)),
                      100, 100, 1,
                      std::vector<size_t>(2, Lx * Ly / 2),
                      Ly, Lx,
                      std::vector<double>(40, 0.3),
                      StochasticReconfiguration,
                      ConjugateGradientParams(100, 1e-5, 20, 0.001));

  MCMeasurementPara measure_para = MCMeasurementPara(
      BMPSTruncatePara(Dpeps, 2 * Dpeps, 1e-15,
                       CompressMPSScheme::SVD_COMPRESS,
                       std::make_optional<double>(1e-14),
                       std::make_optional<size_t>(10)),
      1000, 1000, 1,
      std::vector<size_t>(2, Lx * Ly / 2),
      Ly, Lx);

  std::string model_name = "square_j1j2_xxz";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);

  void SetUp() {
    MPITest::SetUp();
    ham_hei_nn({0, 0, 0, 0}) = 0.25 * jz1;
    ham_hei_nn({1, 1, 1, 1}) = 0.25 * jz1;
    ham_hei_nn({1, 1, 0, 0}) = -0.25 * jz1;
    ham_hei_nn({0, 0, 1, 1}) = -0.25 * jz1;
    ham_hei_nn({0, 1, 1, 0}) = 0.5 * jxy1;
    ham_hei_nn({1, 0, 0, 1}) = 0.5 * jxy1;

    ham_hei_nnn({0, 0, 0, 0}) = 0.25 * jz2;
    ham_hei_nnn({1, 1, 1, 1}) = 0.25 * jz2;
    ham_hei_nnn({1, 1, 0, 0}) = -0.25 * jz2;
    ham_hei_nnn({0, 0, 1, 1}) = -0.25 * jz2;
    ham_hei_nnn({0, 1, 1, 0}) = 0.5 * jxy2;
    ham_hei_nnn({1, 0, 0, 1}) = 0.5 * jxy2;

    optimize_para.wavefunction_path = tps_path;
    measure_para.wavefunction_path = tps_path;
  }

};

TEST_F(SpinOneHalfSystem, SimpleUpdate) {
  if (rank == kMPIMasterRank) {
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
        *su_exe = new SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>(update_para, peps0,
                                                                          ham_hei_nn,
                                                                          ham_hei_nnn);
    su_exe->Execute();

    su_exe->update_para.Dmax = 6;
    su_exe->update_para.Trunc_err = 1e-15;
    su_exe->ResetStepLenth(0.01);
    su_exe->Execute();

    su_exe->update_para.Dmax = Dpeps;
    su_exe->update_para.Trunc_err = 1e-15;
    su_exe->ResetStepLenth(0.001);
    su_exe->Execute();

    double estimated_energy = su_exe->GetEstimatedEnergy();
    EXPECT_NEAR(estimated_energy / energy_ed, 1, 0.1);
    auto tps = TPS<TenElemT, QNT>(su_exe->GetPEPS());
//  std::string peps_path = "Hei_PEPS" + std::to_string(Ly) + "x"
//      + std::to_string(Lx) + "D" + std::to_string(su_exe->update_para.Dmax);
//  su_exe->DumpResult(peps_path, true);
    for (auto &ten : tps) {
      ten *= (1.0 / ten.GetMaxAbs());
    }
    SplitIndexTPS<TenElemT, QNT> sitps = tps;
    sitps.Dump(tps_path);
    delete su_exe;
  }
}

TEST_F(SpinOneHalfSystem, StochasticReconfigurationOpt) {
  MPI_Barrier(comm);

  SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
  tps.Load(tps_path);

  using Model = SquareSpinOneHalfJ1J2XXZModel;
  Model j1j2_model(jz1, jxy1, jz2, jxy2, 0);
  //VMC
  auto executor =
      new VMCPEPSExecutor<TenElemT, QNT, MCUpdateSquareTNN3SiteExchange, Model>(optimize_para, tps,
                                                                                 comm,
                                                                                 j1j2_model);
  size_t start_flop = flop;
  Timer vmc_timer("vmc");

  executor->Execute();

  size_t end_flop = flop;
  double elapsed_time = vmc_timer.PrintElapsed();
  double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "Gflops = " << Gflops / elapsed_time << std::endl;

  tps = executor->GetState();
  delete executor;

  //Measure
  auto measure_exe =
      new MonteCarloMeasurementExecutor<TenElemT, QNT, MCUpdateSquareTNN3SiteExchange, Model>(
          measure_para,
          tps,
          comm,
          j1j2_model);
  start_flop = flop;

  measure_exe->Execute();

  end_flop = flop;
  elapsed_time = vmc_timer.PrintElapsed();
  Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "Gflops = " << Gflops / elapsed_time << std::endl;

  auto [energy, en_err] = measure_exe->OutputEnergy();
  EXPECT_NEAR(Real(energy), energy_ed, 0.001);
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
