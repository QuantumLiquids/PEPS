// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Unittests for TenMatrix
*/
#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/framework/ten_matrix.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;
using Tensor = QLTensor<QLTEN_Double, U1QN>;

TEST(TestTenVec, TestIO) {
  QNT qn0 = QNT({QNCard("N", U1QNVal(0))});
  QNT qn1 = QNT({QNCard("N", U1QNVal(1))});
  QNT qnm1 = QNT({QNCard("N", U1QNVal(-1))});
  IndexT idx_out = IndexT(
      {QNSctT(qn0, 2), QNSctT(qn1, 2)},
      TenIndexDirType::OUT
  );
  auto idx_in = InverseIndex(idx_out);
  Tensor ten0({idx_in, idx_out});
  Tensor ten1({idx_in, idx_out});
  Tensor ten2({idx_in, idx_out});
  ten0.Random(qn0);
  ten1.Random(qn1);
  ten2.Random(qnm1);

  TenMatrix<Tensor> tenmat(3, 4);
  tenmat({0, 0}) = ten0;
  tenmat({0, 1}) = ten1;
  tenmat({2, 1}) = ten2;
  tenmat.DumpTen(0, 0, "ten00." + kQLTenFileSuffix);
  tenmat.DumpTen(0, 1, "ten01." + kQLTenFileSuffix, true);
  tenmat.DumpTen(2, 1, "ten21." + kQLTenFileSuffix, false);
  tenmat.dealloc(0, 0);
  tenmat.dealloc(2, 1);
  EXPECT_TRUE(tenmat.empty());

  tenmat.LoadTen(2, 3, "ten00." + kQLTenFileSuffix);
  tenmat.LoadTen(1, 2, "ten01." + kQLTenFileSuffix);
  tenmat.LoadTen(2, 1, "ten21." + kQLTenFileSuffix);
  EXPECT_EQ(tenmat({2, 1}), ten2);
  EXPECT_EQ(tenmat({1, 2}), ten1);
  EXPECT_EQ(tenmat({2, 3}), ten0);
}
