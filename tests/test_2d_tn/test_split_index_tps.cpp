// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-22
*
* Description: QuantumLiquids/PEPS project. Unittests for SplitIndexTPS
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using DTPS = TPS<QLTEN_Double, U1QN>;
using DSITPS = SplitIndexTPS<QLTEN_Double, U1QN>;
using DTensor = QLTensor<QLTEN_Double, U1QN>;
struct SplitIdxTPSData : public testing::Test {
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
  size_t Lx = 4;
  size_t Ly = 4;
  size_t N = Lx * Ly;

  std::string tps_path = "tps_heisenberg_D8";
  DTPS dtps = DTPS(Ly, Lx);
  DSITPS dsitps = DSITPS(Ly, Lx);

  void SetUp() {
    dtps.Load(tps_path);
    dsitps = SplitIndexTPS(dtps);
  }
};

//TEST_F(SplitIdxTPSData, TestConstructorAndCopy) {
//
//}

TEST_F(SplitIdxTPSData, TestNormalization) {
  dsitps.NormalizeAllSite();
  for (std::vector<DTensor> &split_ten : dsitps) {
    double norm = split_ten[0].Get2Norm() * split_ten[0].Get2Norm();
    norm += split_ten[1].Get2Norm() * split_ten[1].Get2Norm();
    EXPECT_DOUBLE_EQ(norm, 1.0);
  }

  for (auto &ten : dtps) {
    ten.Normalize();
  }
  DSITPS dsitps2(dtps);
  for (std::vector<DTensor> &split_ten : dsitps2) {
    double norm = split_ten[0].Get2Norm() * split_ten[0].Get2Norm();
    norm += split_ten[1].Get2Norm() * split_ten[1].Get2Norm();
    EXPECT_NEAR(norm, 1.0, 1e-13);
  }
}
