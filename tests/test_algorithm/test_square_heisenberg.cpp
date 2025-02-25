// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-21
*
* Description: QuantumLiquids/PEPS project. Unittests for Square Heisenberg model.
*/

#define QLTEN_COUNT_FLOPS 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/qlpeps.h"
#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

std::string GenTPSPath(std::string model_name, size_t Dmax, size_t Lx, size_t Ly) {
#if TEN_ELEM_TYPE == QLTEN_Double
  return "dtps_" + model_name + "_D" + std::to_string(Dmax) + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#elif TEN_ELEM_TYPE == QLTEN_Double
  return "ztps_" + model_name + "_D" + std::to_string(Dmax)  + "_L" + std::to_string(Lx) + "x" + std::to_string(Ly);
#else
#error "Unexpected TEN_ELEM_TYPE"
#endif
}

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

  std::string model_name = "square_afm_heisenberg";
  std::string tps_path = GenTPSPath(model_name, Dpeps, Lx, Ly);
  void SetUp() {
    MPITest::SetUp();
    ham_hei_nn({0, 0, 0, 0}) = 0.25;
    ham_hei_nn({1, 1, 1, 1}) = 0.25;
    ham_hei_nn({1, 1, 0, 0}) = -0.25;
    ham_hei_nn({0, 0, 1, 1}) = -0.25;
    ham_hei_nn({0, 1, 1, 0}) = 0.5;
    ham_hei_nn({1, 0, 0, 1}) = 0.5;

    Tensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      ham_hei_tri_terms[i] = Tensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
    }

    for (size_t i = 0; i < 2; i++) {
      ham_hei_tri_terms[0]({0, 0, 0, 0, i, i}) = 0.25;
      ham_hei_tri_terms[0]({1, 1, 1, 1, i, i}) = 0.25;
      ham_hei_tri_terms[0]({1, 1, 0, 0, i, i}) = -0.25;
      ham_hei_tri_terms[0]({0, 0, 1, 1, i, i}) = -0.25;
      ham_hei_tri_terms[0]({0, 1, 1, 0, i, i}) = 0.5;
      ham_hei_tri_terms[0]({1, 0, 0, 1, i, i}) = 0.5;
    }

    for (size_t i = 0; i < 2; i++) {
      ham_hei_tri_terms[1]({0, 0, i, i, 0, 0}) = 0.25;
      ham_hei_tri_terms[1]({1, 1, i, i, 1, 1}) = 0.25;
      ham_hei_tri_terms[1]({1, 1, i, i, 0, 0}) = -0.25;
      ham_hei_tri_terms[1]({0, 0, i, i, 1, 1}) = -0.25;
      ham_hei_tri_terms[1]({0, 1, i, i, 1, 0}) = 0.5;
      ham_hei_tri_terms[1]({1, 0, i, i, 0, 1}) = 0.5;
    }

    for (size_t i = 0; i < 2; i++) {
      ham_hei_tri_terms[2]({i, i, 0, 0, 0, 0}) = 0.25;
      ham_hei_tri_terms[2]({i, i, 1, 1, 1, 1}) = 0.25;
      ham_hei_tri_terms[2]({i, i, 1, 1, 0, 0}) = -0.25;
      ham_hei_tri_terms[2]({i, i, 0, 0, 1, 1}) = -0.25;
      ham_hei_tri_terms[2]({i, i, 0, 1, 1, 0}) = 0.5;
      ham_hei_tri_terms[2]({i, i, 1, 0, 0, 1}) = 0.5;
    }
    ham_hei_tri = ham_hei_tri_terms[0] + ham_hei_tri_terms[1] + ham_hei_tri_terms[2];

    optimize_para.wavefunction_path = tps_path;
    measure_para.wavefunction_path = tps_path;
  }
};

TEST_F(HeisenbergSystem, SimpleUpdate) {
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

// Check if the TPS doesn't change by setting step length = 0
TEST_F(HeisenbergSystem, ZeroUpdate) {

  optimize_para.step_lens = {0.0};
  SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
  tps.Load(tps_path);
  auto init_tps = tps;
  auto executor =
      new VMCPEPSExecutor<TenElemT, QNT, MCUpdateSquareNNExchange, SpinOneHalfHeisenbergSquare>(optimize_para, tps,
                                                                                                comm);
  size_t start_flop = flop;
  Timer vmc_timer("vmc");
  executor->Execute();
  size_t end_flop = flop;
  double elapsed_time = vmc_timer.Elapsed();
  double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "flop = " << end_flop - start_flop << std::endl;
  std::cout << "Gflops = " << Gflops << std::endl;
  SplitIndexTPS<TenElemT, QNT> result_sitps = executor->GetState();
  auto diff = init_tps + (-result_sitps);
  EXPECT_NE(diff.NormSquare(), 1e-14);
  delete executor;
}

TEST_F(HeisenbergSystem, StochasticReconfigurationOpt) {
  size_t Dpeps = 6;

  SplitIndexTPS<TenElemT, QNT> tps(Ly, Lx);
  tps.Load(tps_path);

  //VMC
  auto executor =
      new VMCPEPSExecutor<TenElemT, QNT, MCUpdateSquareNNExchange, SpinOneHalfHeisenbergSquare>(optimize_para, tps,
                                                                                                comm);
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
      new MonteCarloMeasurementExecutor<TenElemT, QNT, MCUpdateSquareNNExchange, SpinOneHalfHeisenbergSquare>(
          measure_para,
          tps,
          comm);
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
