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

using namespace gqten;
using namespace gqpeps;

using gqten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

boost::mpi::environment env;

// Test spin systems
struct TestSpinSystemVMCPEPS : public testing::Test {
  size_t Lx = 6; //cols
  size_t Ly = 4;
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

  VMCOptimizePara optimize_para = VMCOptimizePara(1e-15, 5, 10, 100, 10, {N / 2, N / 2}, {0.1});

  TPS<GQTEN_Double, U1QN> tps = TPS<GQTEN_Double, U1QN>(Ly, Lx);

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

    optimize_para.step_lens = {0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02};

    if (!tps.Load("tps_heisenberg_D4")) {
      std::cout << "Loading TPS files is broken." << std::endl;
      exit(-1);
    };

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
  VMCPEPSExecutor<GQTEN_Double, U1QN, Model> executor = VMCPEPSExecutor<GQTEN_Double, U1QN, Model>(optimize_para, tps,
                                                                                                   world);
  executor.Execute();
  executor.DumpTenData("vmc_tps_heisenbergD4");
}