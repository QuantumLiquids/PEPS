// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-21
*
* Description: QuantumLiquids/PEPS project. Unittests for PEPS class.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DQLTensor = QLTensor<QLTEN_Double, U1QN>;
using ZQLTensor = QLTensor<QLTEN_Complex, U1QN>;

// Test spin systems
struct TestSpinSystem : public testing::Test {
  size_t Lx = 5; //cols
  size_t Ly = 4;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_out = IndexT({
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(1))}), 1),
                             QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                         TenIndexDirType::OUT
  );
  IndexT pb_in = InverseIndex(pb_out);

  DQLTensor did = DQLTensor({pb_in, pb_out});
  DQLTensor dsz = DQLTensor({pb_in, pb_out});
  DQLTensor dsp = DQLTensor({pb_in, pb_out});
  DQLTensor dsm = DQLTensor({pb_in, pb_out});

  ZQLTensor zid = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsz = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsp = ZQLTensor({pb_in, pb_out});
  ZQLTensor zsm = ZQLTensor({pb_in, pb_out});

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

TEST_F(TestSpinSystem, InitialCopyAndIO) {
  SquareLatticePEPS<QLTEN_Double, U1QN> peps0(pb_out, Ly, Lx);
  std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));

  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      size_t sz_int = (x + y) % 2;
      activates[y][x] = sz_int;
    }
  }
  peps0.Initial(activates);

  EXPECT_EQ(peps0.Rows(), Ly);
  EXPECT_EQ(peps0.Cols(), Lx);

  EXPECT_EQ(peps0.Gamma.rows(), Ly);
  EXPECT_EQ(peps0.Gamma.cols(), Lx);

  EXPECT_EQ(peps0.lambda_horiz.rows(), Ly);
  EXPECT_EQ(peps0.lambda_horiz.cols(), Lx + 1);

  EXPECT_EQ(peps0.lambda_vert.rows(), Ly + 1);
  EXPECT_EQ(peps0.lambda_vert.cols(), Lx);

  auto peps1 = peps0;
  EXPECT_EQ(peps1, peps0);

  peps0.Dump("peps_for_test_io", true);
  SquareLatticePEPS<QLTEN_Double, U1QN> peps2(pb_out, Ly, Lx);
  peps2.Load("peps_for_test_io");
  EXPECT_EQ(peps1, peps2);
}