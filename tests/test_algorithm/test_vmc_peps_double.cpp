// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: QuantumLiquids/PEPS project. Unittests for VMC Optimization in PEPS.
*/


//#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlmps/case_params_parser.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

using TPSSampleNNFlipT = SquareTPSSampleNNExchange<QLTEN_Double, U1QN>;
using SquareTPSSample3SiteExchangeT = SquareTPSSample3SiteExchange<QLTEN_Double, U1QN>;
using SquareTPSSampleFullSpaceNNFlipT = SquareTPSSampleFullSpaceNNFlip<QLTEN_Double, U1QN>;

using qlmps::CaseParamsParserBasic;

char *params_file;

struct VMCUpdateParams : public CaseParamsParserBasic {
  VMCUpdateParams(const char *f) : CaseParamsParserBasic(f) {
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
    D = ParseInt("D");
    Db_min = ParseInt("Dbmps_min");
    Db_max = ParseInt("Dbmps_max");
    MC_samples = ParseInt("MC_samples");
    WarmUp = ParseInt("WarmUp");
    Continue_from_VMC = ParseBool("Continue_from_VMC");
    size_t update_times = ParseInt("UpdateNum");
    step_len = std::vector<double>(update_times);
    if (update_times > 0) {
      step_len[0] = ParseDouble("StepLengthFirst");
      double step_len_change = ParseDouble("StepLengthDecrease");
      for (size_t i = 1; i < update_times; i++) {
        step_len[i] = step_len[0] - i * step_len_change;
      }
    }
  }

  size_t Ly;
  size_t Lx;
  size_t D;
  size_t Db_min;
  size_t Db_max;
  size_t MC_samples;
  size_t WarmUp;
  bool Continue_from_VMC;
  std::vector<double> step_len;
};

// Test spin systems
struct SpinSystemVMCPEPS : public testing::Test {
  VMCUpdateParams params = VMCUpdateParams(params_file);
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;
  size_t N = Lx * Ly;

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

  VMCOptimizePara optimize_para =
      VMCOptimizePara(BMPSTruncatePara(params.Db_min, params.Db_max,
                                       1e-15, CompressMPSScheme::VARIATION2Site,
                                       std::make_optional<double>(1e-14),
                                       std::make_optional<size_t>(10)),
                      params.MC_samples, params.WarmUp, 1,
                      std::vector<size_t>(2, N / 2),
                      Ly, Lx,
                      params.step_len,
                      StochasticGradient);

  boost::mpi::communicator world;

  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

    qlten::hp_numeric::SetTensorManipulationThreads(1);
  }
};

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD4StochasticGradient) {
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(params.D);

  using Model = SpinOneHalfHeisenbergSquare<QLTEN_Double, U1QN>;
  VMCPEPSExecutor<QLTEN_Double, U1QN, SquareTPSSampleFullSpaceNNFlipT, Model> *executor(nullptr);

  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, SquareTPSSampleFullSpaceNNFlipT, Model>(optimize_para,
                                                                                               Ly, Lx,
                                                                                               world);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, SquareTPSSampleFullSpaceNNFlipT, Model>(optimize_para, tps,
                                                                                               world);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD4BMPSSingleSiteVariational) {
  optimize_para.bmps_trunc_para.compress_scheme = CompressMPSScheme::VARIATION1Site;
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(params.D);
  using Model = SpinOneHalfHeisenbergSquare<QLTEN_Double, U1QN>;
  VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);

  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                Ly, Lx,
                                                                                world);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                                world);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD4StochasticReconfigration) {
  using Model = SpinOneHalfHeisenbergSquare<QLTEN_Double, U1QN>;
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(params.D);
  optimize_para.cg_params = ConjugateGradientParams(100, 1e-4, 20, 0.01);
  VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);

  optimize_para.update_scheme = StochasticReconfiguration;
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                Ly, Lx,
                                                                                world);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                                world);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, HeisenbergD4GradientLineSearch) {
  using Model = SpinOneHalfHeisenbergSquare<QLTEN_Double, U1QN>;
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(params.D);
  VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);

  optimize_para.update_scheme = GradientLineSearch;
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                Ly, Lx,
                                                                                world);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                                world);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD4NaturalGradientLineSearch) {
  using Model = SpinOneHalfHeisenbergSquare<QLTEN_Double, U1QN>;
  VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(params.D);
  optimize_para.cg_params = ConjugateGradientParams(100, 1e-4, 20, 0.01);
  optimize_para.update_scheme = NaturalGradientLineSearch;
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                Ly, Lx,
                                                                                world);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                                world);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareJ1J2D4) {
  using Model = SpinOneHalfJ1J2HeisenbergSquare<QLTEN_Double, U1QN>;
  optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(params.D);
  VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  double j2 = 0.2;
  Model j1j2solver(j2);
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                Ly, Lx,
                                                                                world, j1j2solver);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                                world, j1j2solver);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, TriHeisenbergD4) {
  using Model = SpinOneHalfTriHeisenbergSqrPEPS<QLTEN_Double, U1QN>;
  VMCPEPSExecutor<QLTEN_Double, U1QN, SquareTPSSample3SiteExchangeT, Model> *executor(nullptr);
  Model triangle_hei_solver;
  optimize_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, SquareTPSSample3SiteExchangeT, Model>(optimize_para,
                                                                                             Ly,
                                                                                             Lx,
                                                                                             world,
                                                                                             triangle_hei_solver);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_tri_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, SquareTPSSample3SiteExchangeT, Model>(optimize_para,
                                                                                             tps,
                                                                                             world,
                                                                                             triangle_hei_solver);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, TriJ1J2HeisenbergD4) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS<QLTEN_Double, U1QN>;
  VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  Model trianglej1j2_hei_solver(0.2);
  optimize_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                Ly, Lx,
                                                                                world, trianglej1j2_hei_solver);
  } else {
    TPS<QLTEN_Double, U1QN> tps = TPS<QLTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_tri_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                                world, trianglej1j2_hei_solver);
  }

  executor->Execute();
  delete executor;
}

int main(int argc, char *argv[]) {
  boost::mpi::environment env;
  testing::InitGoogleTest(&argc, argv);
  std::cout << argc << std::endl;
  params_file = argv[1];
  return RUN_ALL_TESTS();
}
