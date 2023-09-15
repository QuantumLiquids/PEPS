// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: GraceQ/VMC-PEPS project. Unittests for VMC Optimization in PEPS.
*/


#define PLAIN_TRANSPOSE 1

#include "gqten/gqten.h"
#include "gtest/gtest.h"
#include "gqpeps/algorithm/vmc_update/vmc_peps.h"
#include "gqpeps/algorithm/vmc_update/model_energy_solvers/spin_onehalf_heisenberg_square.h"    // SpinOneHalfHeisenbergSquare
#include "gqmps2/case_params_parser.h"

using namespace gqten;
using namespace gqpeps;

using gqten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

boost::mpi::environment env;

using gqmps2::CaseParamsParserBasic;

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
struct TestSpinSystemVMCPEPS : public testing::Test {
  VMCUpdateParams params = VMCUpdateParams("../../tests/test_algorithm/test_params.json");;
  size_t Lx = params.Lx; //cols
  size_t Ly = params.Ly;
  size_t N = Lx * Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         GQTenIndexDirType::OUT
  );
//  IndexT pb_out = IndexT({
//                             QNSctT(U1QN({QNCard("Sz", U1QNVal(0))}), 2)},
//                         GQTenIndexDirType::OUT
//  );
  IndexT pb_in = InverseIndex(pb_out);

  VMCOptimizePara optimize_para = VMCOptimizePara(1e-15, params.Db_min, params.Db_max,
                                                  params.MC_samples, params.WarmUp,
                                                  {N / 2, N / 2}, {0.1});

  DGQTensor did = DGQTensor({pb_in, pb_out});
  DGQTensor dsz = DGQTensor({pb_in, pb_out});
  DGQTensor dsp = DGQTensor({pb_in, pb_out});
  DGQTensor dsm = DGQTensor({pb_in, pb_out});

  ZGQTensor zid = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsz = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsp = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsm = ZGQTensor({pb_in, pb_out});

  boost::mpi::communicator world;


  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }

    gqten::hp_numeric::SetTensorManipulationThreads(1);
    gqten::hp_numeric::SetTensorTransposeNumThreads(1);

    optimize_para.step_lens = params.step_len;


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

TEST_F(TestSpinSystemVMCPEPS, HeisenbergD4) {
  using Model = SpinOneHalfHeisenbergSquare<GQTEN_Double, U1QN>;
  VMCPEPSExecutor<GQTEN_Double, U1QN, Model> *executor(nullptr);

  if (params.Continue_from_VMC) {
    SplitIndexTPS<GQTEN_Double, U1QN> sitps(Ly, Lx);
    if (!sitps.Load("vmc_tps_heisenbergD" + std::to_string(params.D))) {
      std::cout << "Loading last TPS files is broken." << std::endl;
      exit(-1);
    }
    executor = new VMCPEPSExecutor<GQTEN_Double, U1QN, Model>(optimize_para, sitps,
                                                              world);
  } else {
    TPS<GQTEN_Double, U1QN> tps = TPS<GQTEN_Double, U1QN>(Ly, Lx);
    if (!tps.Load("tps_heisenberg_D" + std::to_string(params.D))) {
      std::cout << "Loading simple updated TPS files is broken." << std::endl;
      exit(-2);
    };
    executor = new VMCPEPSExecutor<GQTEN_Double, U1QN, Model>(optimize_para, tps,
                                                              world);
  }

  executor->Execute();
  executor->DumpTenData("vmc_tps_heisenbergD" + std::to_string(params.D));
  delete executor;
}