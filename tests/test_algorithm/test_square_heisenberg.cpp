// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-21
*
* Description: QuantumLiquids/PEPS project. Unittests for Square Heisenberg model.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlmps/case_params_parser.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nn_simple_update.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_heisenberg_square.h"    // SpinOneHalfHeisenbergSquare
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_squareJ1J2.h"           // SpinOneHalfJ1J2HeisenbergSquare
#include "qlpeps/algorithm/vmc_update/wave_function_component_classes/square_tps_sample_3site_exchange.h"
#include "qlpeps/algorithm/vmc_update/wave_function_component_classes/square_tps_sample_full_space_nn_flip.h"

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

  DQLTensor ham_hei_nn = DQLTensor({pb_in, pb_out, pb_in, pb_out});
  DQLTensor ham_hei_tri;  // three-site hamiltonian in triangle lattice

  void SetUp(size_t L) {
    Lx = L;
    Ly = L;

    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    ham_hei_nn({0, 0, 0, 0}) = 0.25;
    ham_hei_nn({1, 1, 1, 1}) = 0.25;
    ham_hei_nn({1, 1, 0, 0}) = -0.25;
    ham_hei_nn({0, 0, 1, 1}) = -0.25;
    ham_hei_nn({0, 1, 1, 0}) = 0.5;
    ham_hei_nn({1, 0, 0, 1}) = 0.5;

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
    ham_hei_tri = ham_hei_tri_terms[0] + ham_hei_tri_terms[1] + ham_hei_tri_terms[2];

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
                                                                            ham_hei_nn);
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
  su_exe->DumpResult("Hei_PEPS_D6", true);
  tps.Dump("tps_heisenberg_D6");
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
        VMCOptimizePara(BMPSTruncatePara(8, 16, 1e-15, CompressMPSScheme::VARIATION2Site),
                        100, 10, 2,
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

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD6StochasticReconfiguration) {
  SpinSystemVMCPEPS::SetUp(L);
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(6);
  using WaveFunctionT = SquareTPSSampleNNExchange<QLTEN_Double, U1QN>;
  using Model = SpinOneHalfHeisenbergSquare<QLTEN_Double, U1QN>;
  VMCPEPSExecutor<QLTEN_Double, U1QN, WaveFunctionT, Model> *executor(nullptr);

  if (qlmps::IsPathExist(optimize_para.wavefunction_path)) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, WaveFunctionT, Model>(optimize_para,
                                                                             Ly, Lx,
                                                                             world);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_heisenberg_D" + std::to_string(6))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    for (auto tensor : tps) {
      tensor.Normalize();
    }
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
  return RUN_ALL_TESTS();
}
