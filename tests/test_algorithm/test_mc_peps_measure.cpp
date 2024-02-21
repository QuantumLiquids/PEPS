// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: QuantumLiquids/PEPS project. Unittests for Monte-Carlo Measurement for finite-size PEPS.
*/



#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_measurement.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_heisenberg_square.h"    // SpinOneHalfHeisenbergSquare
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_squareJ1J2.h"           // SpinOneHalfJ1J2HeisenbergSquare
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_triangle_heisenberg_sqrpeps.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/spin_onehalf_triangle_heisenbergJ1J2_sqrpeps.h"

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

boost::mpi::environment env;

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

  VMCOptimizePara optimize_para = VMCOptimizePara(
      BMPSTruncatePara(params.Db_min, params.Db_max, 1e-10, CompressMPSScheme::VARIATION2Site),
      params.MC_samples, params.WarmUp, 1,
      std::vector<size_t>(2, N / 2),
      Ly, Lx,
      {0.1},
      StochasticGradient);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});

  boost::mpi::communicator world;

  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

    qlten::hp_numeric::SetTensorManipulationThreads(1);

    optimize_para.step_lens = params.step_len;
    optimize_para.wavefunction_path = "vmc_tps_heisenbergD" + std::to_string(params.D);

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

TEST_F(SpinSystemVMCPEPS, TriHeisenbergD4) {
  using Model = SpinOneHalfTriHeisenbergSqrPEPS<QLTEN_Double, U1QN>;
  MonteCarloMeasurementExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  Model triangle_hei_solver;
  optimize_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);

  executor = new MonteCarloMeasurementExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                            Ly, Lx,
                                                                                            world, triangle_hei_solver);

  executor->Execute();
  delete executor;
}

TEST_F(SpinSystemVMCPEPS, TriJ1J2HeisenbergD4) {
  using Model = SpinOneHalfTriJ1J2HeisenbergSqrPEPS<QLTEN_Double, U1QN>;
  MonteCarloMeasurementExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model> *executor(nullptr);
  Model trianglej1j2_hei_solver(0.2);
  optimize_para.wavefunction_path = "vmc_tps_tri_heisenbergD" + std::to_string(params.D);
  executor = new MonteCarloMeasurementExecutor<QLTEN_Double, U1QN, TPSSampleNNFlipT, Model>(optimize_para,
                                                                                            Ly,
                                                                                            Lx,
                                                                                            world,
                                                                                            trianglej1j2_hei_solver);
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

