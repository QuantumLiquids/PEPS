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
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlmps/case_params_parser.h"
#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;

std::string GenTPSPath(std::string model_name, size_t Dmax) {
#if TEN_ELEM_TYPE == QLTEN_Double
  return "dtps_" + model_name + "_D" + std::to_string(Dmax);
#elif TEN_ELEM_TYPE == QLTEN_Complex
  return "ztps_" + model_name + "_D" + std::to_string(Dmax);
#else
#error "Unexpected TEN_ELEM_TYPE"
#endif
}

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
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<U1QN>;
  using QNSctVecT = QNSectorVec<U1QN>;

  using TenElemT = TEN_ELEM_TYPE;
  using Tensor = QLTensor<TenElemT, U1QN>;
  using TPSSampleNNFlipT = MCUpdateSquareNNExchange;

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

  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

    qlten::hp_numeric::SetTensorManipulationThreads(1);
  }
};

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD4StochasticGradient) {
  std::string model_name = "square_nn_hei";
  optimize_para.wavefunction_path = "vmc_tps_" + model_name + "D" + std::to_string(params.D);

  using Model = SquareSpinOneHalfXXZModel;
  VMCPEPSExecutor<TenElemT, U1QN, MCUpdateSquareNNFullSpaceUpdate, Model> *executor(nullptr);

  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<TenElemT, U1QN, MCUpdateSquareNNFullSpaceUpdate, Model>(optimize_para,
                                                                                           Ly, Lx,
                                                                                           comm);
  } else {
    TPS<TenElemT, U1QN> tps = TPS<TenElemT, U1QN>(Ly, Lx);
    if (!tps.Load(GenTPSPath(model_name, params.D))) {
      std::cout << "Loading simple updated TPS files "
                << GenTPSPath(model_name, params.D)
                << "failed." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<TenElemT, U1QN, MCUpdateSquareNNFullSpaceUpdate, Model>(optimize_para, tps,
                                                                                           comm);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD4BMPSSingleSiteVariational) {
  std::string model_name = "square_nn_hei";
  optimize_para.bmps_trunc_para.compress_scheme = CompressMPSScheme::VARIATION1Site;
  optimize_para.wavefunction_path = "vmc_tps_" + model_name + "D" + std::to_string(params.D);
  using Model = SquareSpinOneHalfXXZModel;
  VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);

  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                            Ly, Lx,
                                                                            comm);
  } else {
    TPS<TenElemT, U1QN> tps = TPS<TenElemT, U1QN>(Ly, Lx);
    if (!tps.Load(GenTPSPath(model_name, params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                            comm);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD4StochasticReconfigration) {
  using Model = SquareSpinOneHalfXXZModel;
  std::string model_name = "square_nn_hei";
  optimize_para.wavefunction_path = "vmc_tps_" + model_name + "D" + std::to_string(params.D);
  optimize_para.cg_params = ConjugateGradientParams(100, 1e-4, 20, 0.01);
  VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);

  optimize_para.update_scheme = StochasticReconfiguration;
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                            Ly, Lx,
                                                                            comm);
  } else {
    TPS<TenElemT, U1QN> tps = TPS<TenElemT, U1QN>(Ly, Lx);
    if (!tps.Load(GenTPSPath(model_name, params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                            comm);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, HeisenbergD4GradientLineSearch) {
  using Model = SquareSpinOneHalfXXZModel;
  std::string model_name = "square_nn_hei";
  optimize_para.wavefunction_path = "vmc_tps_" + model_name + "D" + std::to_string(params.D);
  VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);

  optimize_para.update_scheme = GradientLineSearch;
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                            Ly, Lx,
                                                                            comm);
  } else {
    TPS<TenElemT, U1QN> tps = TPS<TenElemT, U1QN>(Ly, Lx);
    if (!tps.Load(GenTPSPath(model_name, params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                            comm);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareHeisenbergD4NaturalGradientLineSearch) {
  using Model = SquareSpinOneHalfXXZModel;
  VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  std::string model_name = "square_nn_hei";
  optimize_para.wavefunction_path = "vmc_tps_" + model_name + "D" + std::to_string(params.D);
  optimize_para.cg_params = ConjugateGradientParams(100, 1e-4, 20, 0.01);
  optimize_para.update_scheme = NaturalGradientLineSearch;
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                            Ly, Lx,
                                                                            comm);
  } else {
    TPS<TenElemT, U1QN> tps = TPS<TenElemT, U1QN>(Ly, Lx);
    if (!tps.Load(GenTPSPath(model_name, params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                            comm);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, SquareJ1J2D4) {
  using Model = SpinOneHalfJ1J2HeisenbergSquare;
  std::string model_name = "square_j1j2_hei";
  optimize_para.wavefunction_path = "vmc_tps_" + model_name + "D" + std::to_string(params.D);
  VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  double j2 = 0.2;
  Model j1j2solver(j2);
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                            Ly, Lx,
                                                                            comm, j1j2solver);
  } else {
    TPS<TenElemT, U1QN> tps = TPS<TenElemT, U1QN>(Ly, Lx);
    if (!tps.Load(GenTPSPath(model_name, params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                            comm, j1j2solver);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, TriHeisenbergD4) {
  using Model = SpinOneHalfTriHeisenbergSqrPEPS;
  VMCPEPSExecutor<TenElemT, U1QN, MCUpdateSquareTNN3SiteExchange, Model> *executor(nullptr);
  Model triangle_hei_solver;
  optimize_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<TenElemT, U1QN, MCUpdateSquareTNN3SiteExchange, Model>(optimize_para,
                                                                                         Ly,
                                                                                         Lx,
                                                                                         comm,
                                                                                         triangle_hei_solver);
  } else {
    TPS<TenElemT, U1QN> tps = TPS<TenElemT, U1QN>(Ly, Lx);
    if (!tps.Load("tps_tri_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<TenElemT, U1QN, MCUpdateSquareTNN3SiteExchange, Model>(optimize_para,
                                                                                         tps,
                                                                                         comm,
                                                                                         triangle_hei_solver);
  }

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, TriJ1J2HeisenbergD4) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS;
  VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  Model trianglej1j2_hei_solver(0.2);
  optimize_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);
  if (params.Continue_from_VMC) {
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                            Ly, Lx,
                                                                            comm, trianglej1j2_hei_solver);
  } else {
    TPS<TenElemT, U1QN> tps = TPS<TenElemT, U1QN>(Ly, Lx);
    if (!tps.Load("tps_tri_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<TenElemT, U1QN, TPSSampleNNFlipT, Model>(optimize_para, tps,
                                                                            comm, trianglej1j2_hei_solver);
  }

  executor->Execute();
  delete executor;
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  if (argc == 1) {
    std::cout << "No parameter file input." << std::endl;
    MPI_Finalize();
    return 1;
  }
  params_file = argv[1];
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
