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

size_t L; // system size

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

// Test spin systems
struct SpinSystemSimpleUpdate : public testing::Test {
  size_t Lx; //cols
  size_t Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
#ifdef U1SYM
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
#endif
  IndexT pb_in = InverseIndex(pb_out);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});

  DQLTensor dham_hei_nn = DQLTensor({pb_in, pb_out, pb_in, pb_out});
  DQLTensor dham_hei_tri;  // three-site hamiltonian in triangle lattice

  void SetUp(size_t L) {
    Lx = L;
    Ly = L;

    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    dham_hei_nn({0, 0, 0, 0}) = 0.25;
    dham_hei_nn({1, 1, 1, 1}) = 0.25;
    dham_hei_nn({1, 1, 0, 0}) = -0.25;
    dham_hei_nn({0, 0, 1, 1}) = -0.25;
    dham_hei_nn({0, 1, 1, 0}) = 0.5;
    dham_hei_nn({1, 0, 0, 1}) = 0.5;

    DQLTensor ham_hei_tri_terms[3];
    for (size_t i = 0; i < 3; i++) {
      ham_hei_tri_terms[i] = DQLTensor({pb_in, pb_out, pb_in, pb_out, pb_in, pb_out});
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
    dham_hei_tri = ham_hei_tri_terms[0] + ham_hei_tri_terms[1] + ham_hei_tri_terms[2];

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;
  }
};

TEST_F(SpinSystemSimpleUpdate, SquareHeisenberg) {
  SetUp(L);
  SquareLatticePEPS<QLTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(100, 0.2, 1, 4, 1e-15);
  SimpleUpdateExecutor<QLTEN_Double, U1QN>
      *su_exe = new SquareLatticeNNSimpleUpdateExecutor<QLTEN_Double, U1QN>(update_para, peps0,
                                                                            dham_hei_nn);
  su_exe->Execute();

  su_exe->update_para.Dmax = 6;
  su_exe->update_para.Trunc_err = 1e-15;
  su_exe->SetStepLenth(0.05);
  su_exe->Execute();

  su_exe->update_para.Dmax = 6;
  su_exe->update_para.Trunc_err = 1e-15;
  su_exe->SetStepLenth(0.01);
  su_exe->Execute();

  auto tps = TPS<QLTEN_Double, U1QN>(su_exe->GetPEPS());
  std::string peps_path = "Hei_PEPS" + std::to_string(Ly) + "x"
      + std::to_string(Lx) + "D" + std::to_string(su_exe->update_para.Dmax);
  su_exe->DumpResult(peps_path, true);
  std::string tps_path = "Hei_TPS" + std::to_string(Ly) + "x"
      + std::to_string(Lx) + "D" + std::to_string(su_exe->update_para.Dmax);
  for (auto &ten : tps) {
    ten *= (1.0 / ten.GetMaxAbs());
  }
  tps.Dump(tps_path);
  delete su_exe;
}

struct SpinSystemVMCPEPS : public testing::Test {
  size_t Lx; //cols
  size_t Ly;
  size_t N;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
#ifdef U1SYM
  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                          QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );
#else
  IndexT pb_out = IndexT({QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
                         TenIndexDirType::OUT
  );
#endif
  IndexT pb_in = InverseIndex(pb_out);

  VMCOptimizePara optimize_para;

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});

  boost::mpi::communicator world;

  void SetUp(size_t L) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

    Lx = L;
    Ly = L;
    N = Lx * Ly;
    optimize_para =
        VMCOptimizePara(BMPSTruncatePara(8, 16, 1e-15,
                                         CompressMPSScheme::VARIATION2Site,
                                         std::make_optional<double>(1e-14),
                                         std::make_optional<size_t>(10)),
                        10, 10, 2,
                        std::vector<size_t>(2, N / 2),
                        Ly, Lx,
                        std::vector<double>(5, 0.1),
                        NaturalGradientLineSearch,
                        ConjugateGradientParams(100, 1e-5, 20, 0.01));
    qlten::hp_numeric::SetTensorManipulationThreads(1);

    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;
  }
};


// Check if the TPS doesn't change by setting step length = 0
TEST_F(SpinSystemVMCPEPS, ZeroUpdate) {
  SpinSystemVMCPEPS::SetUp(L);
  size_t Dpeps = 6;
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(Dpeps);
  optimize_para.step_lens = {0.0};
  optimize_para.update_scheme = StochasticReconfiguration;
  using WaveFunctionT = SquareTPSSampleNNExchange<QLTEN_Double, U1QN>;
  using Model = SpinOneHalfHeisenbergSquare<QLTEN_Double, U1QN>;
  VMCPEPSExecutor<QLTEN_Double, U1QN, WaveFunctionT, Model> *executor(nullptr);
  TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
  std::string tps_path = "Hei_TPS" + std::to_string(Ly) + "x"
      + std::to_string(Lx) + "D" + std::to_string(Dpeps);
  if (!tps.Load(tps_path)) {
    std::cout << "Loading simple updated TPS files is broken." << std::endl;
    exit(-2);
  };
  for (auto &tensor : tps) {
    tensor *= (1.0 / tensor.GetMaxAbs());
  }
  SplitIndexTPS<QLTEN_Double, U1QN> init_sitps(tps);
  size_t start_flop = flop;
  Timer vmc_timer("vmc");
  executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, WaveFunctionT, Model>(optimize_para, tps,
                                                                           world);
  executor->Execute();
  size_t end_flop = flop;
  double elapsed_time = vmc_timer.Elapsed();
  double Gflops = (end_flop - start_flop) * 1.e-9 / elapsed_time;
  std::cout << "flop = " << end_flop - start_flop << std::endl;
  std::cout << "Gflops = " << Gflops << std::endl;
  SplitIndexTPS<QLTEN_Double, U1QN> result_sitps = executor->GetState();
  for (size_t row = 0; row < L; row++) {
    for (size_t col = 0; col < L; col++) {
      for (size_t i = 0; i < 2; i++) {
        auto init_ten = init_sitps({row, col})[i];
        auto res_ten = result_sitps({row, col})[i];
        DQLTensor diff_ten = init_ten + (-res_ten);
        EXPECT_NEAR(diff_ten.Get2Norm(), 0.0, 1e-13);
      }
    }
  }
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD6StochasticReconfiguration) {
  SpinSystemVMCPEPS::SetUp(L);
  size_t Dpeps = 6;
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(Dpeps);
  using WaveFunctionT = SquareTPSSampleNNExchange<QLTEN_Double, U1QN>;
  using Model = SpinOneHalfHeisenbergSquare<QLTEN_Double, U1QN>;
  VMCPEPSExecutor<QLTEN_Double, U1QN, WaveFunctionT, Model> *executor(nullptr);

  if (qlmps::IsPathExist(optimize_para.wavefunction_path)) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, WaveFunctionT, Model>(optimize_para,
                                                                             Ly, Lx,
                                                                             world);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    std::string tps_path = "Hei_TPS" + std::to_string(Ly) + "x"
        + std::to_string(Lx) + "D" + std::to_string(Dpeps);
    if (!tps.Load(tps_path)) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, WaveFunctionT, Model>(optimize_para, tps,
                                                                             world);
  }
  executor->Execute();
  delete executor;
}

int main(int argc, char *argv[]) {
  boost::mpi::environment env;
  testing::InitGoogleTest(&argc, argv);
  L = atoi(argv[1]);
  hp_numeric::SetTensorManipulationThreads(1);
  return RUN_ALL_TESTS();
}
