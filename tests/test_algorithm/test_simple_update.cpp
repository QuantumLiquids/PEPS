// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: GraceQ/VMC-PEPS project. Unittests for Simple Update in PEPS optimization.
*/

#define PLAIN_TRANSPOSE 1

#include "gqten/gqten.h"
#include "gtest/gtest.h"
#include "gqpeps/algorithm/simple_update/simple_update.h"

using namespace gqten;
using namespace gqpeps;

using gqten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

// Test spin systems
struct TestSimpleUpdateSpinSystem : public testing::Test {
  size_t Lx = 4; //cols
  size_t Ly = 4;

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

  DGQTensor did = DGQTensor({pb_in, pb_out});
  DGQTensor dsz = DGQTensor({pb_in, pb_out});
  DGQTensor dsp = DGQTensor({pb_in, pb_out});
  DGQTensor dsm = DGQTensor({pb_in, pb_out});

  ZGQTensor zid = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsz = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsp = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsm = ZGQTensor({pb_in, pb_out});

  void SetUp(void) {
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

TEST_F(TestSimpleUpdateSpinSystem, NNIsing) {
  PEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(5, 0.01, 1, 4, 1e-5);
//  DGQTensor ham_nn({pb_in, pb_out, pb_in, pb_out});
//  ham_nn({0, 0, 0, 0}) = 0.25;
//  ham_nn({1, 1, 1, 1}) = 0.25;
//  ham_nn({1, 1, 0, 0}) = -0.25;
//  ham_nn({0, 0, 1, 1}) = -0.25;
  DGQTensor ham_nn;
  Contract(&dsz, {}, &dsz, {}, &ham_nn);

  auto su_exe = SimpleUpdateExecutor(update_para, ham_nn, peps0);
  su_exe.Execute();
}

TEST_F(TestSimpleUpdateSpinSystem, NNHeisenberg) {
  gqten::hp_numeric::SetTensorManipulationThreads(1);

  PEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
  size_t sz_int = 0;
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = x + y;
      activates[y][x] = sz_int % 2;
    }
  }
  peps0.Initial(activates);

  SimpleUpdatePara update_para(10, 0.1, 1, 2, 1e-5);
  DGQTensor ham_nn({pb_in, pb_out, pb_in, pb_out});
  ham_nn({0, 0, 0, 0}) = 0.25;
  ham_nn({1, 1, 1, 1}) = 0.25;
  ham_nn({1, 1, 0, 0}) = -0.25;
  ham_nn({0, 0, 1, 1}) = -0.25;
  ham_nn({0, 1, 1, 0}) = 0.5;
  ham_nn({1, 0, 0, 1}) = 0.5;

  auto su_exe = SimpleUpdateExecutor(update_para, ham_nn, peps0);
  su_exe.Execute();

  su_exe.update_para.Dmax = 4;
  su_exe.Execute();

  su_exe.update_para.Dmax = 8;
  su_exe.update_para.Trunc_err = 1e-6;
  su_exe.Execute();

  su_exe.SetStepLenth(0.01);
  su_exe.update_para.Trunc_err = 1e-8;
  su_exe.Execute();

  su_exe.SetStepLenth(0.001);
  su_exe.update_para.Trunc_err = 1e-10;
  su_exe.Execute();

  su_exe.DumpResult("su_update_result", true);

}


TEST_F(TestSimpleUpdateSpinSystem, NNHeisenbergLargeD) {
  PEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  peps0.Load("su_update_result");

  SimpleUpdatePara update_para(10, 0.1, 1, 8, 1e-10);
  DGQTensor ham_nn({pb_in, pb_out, pb_in, pb_out});
  ham_nn({0, 0, 0, 0}) = 0.25;
  ham_nn({1, 1, 1, 1}) = 0.25;
  ham_nn({1, 1, 0, 0}) = -0.25;
  ham_nn({0, 0, 1, 1}) = -0.25;
  ham_nn({0, 1, 1, 0}) = 0.5;
  ham_nn({1, 0, 0, 1}) = 0.5;

  auto su_exe = SimpleUpdateExecutor(update_para, ham_nn, peps0);
  su_exe.Execute();

  su_exe.DumpResult("su_update_resultD8", true);
}


TEST_F(TestSimpleUpdateSpinSystem, NNHeisenbergLargeLargeD) {
  PEPS<GQTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  peps0.Load("su_update_resultD8");

  SimpleUpdatePara update_para(10, 0.05, 1, 16, 1e-10);
  DGQTensor ham_nn({pb_in, pb_out, pb_in, pb_out});
  ham_nn({0, 0, 0, 0}) = 0.25;
  ham_nn({1, 1, 1, 1}) = 0.25;
  ham_nn({1, 1, 0, 0}) = -0.25;
  ham_nn({0, 0, 1, 1}) = -0.25;
  ham_nn({0, 1, 1, 0}) = 0.5;
  ham_nn({1, 0, 0, 1}) = 0.5;

  auto su_exe = SimpleUpdateExecutor(update_para, ham_nn, peps0);
  su_exe.Execute();

  su_exe.DumpResult("su_update_resultD16", true);
}