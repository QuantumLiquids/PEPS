// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-03
*
* Description: QuantumLiquids/PEPS project. Unittests for TensorNetwork2D as a data container.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::U1QN;

struct TensorNetwork2DContainerTest : public testing::Test {
  using QNT = U1QN;
  using IndexT = Index<U1QN>;
  using QNSctT = QNSector<QNT>;
  using DQLTensor = QLTensor<QLTEN_Double, QNT>;

  const size_t Lx = 4;
  const size_t Ly = 4;

  void SetUp() {
  }
};

TEST_F(TensorNetwork2DContainerTest, ConstructionAndAccess) {
  TensorNetwork2D<QLTEN_Double, U1QN> tn(Ly, Lx);
  EXPECT_EQ(tn.rows(), Ly);
  EXPECT_EQ(tn.cols(), Lx);

  // Test setting and getting a tensor
  IndexT idx = IndexT({QNSctT(U1QN(0), 1)}, TenIndexDirType::OUT);
  DQLTensor tensor({idx, InverseIndex(idx)});
  tensor({0, 0}) = 1.23;

  tn({1, 1}) = tensor;
  EXPECT_DOUBLE_EQ(tn({1, 1})({0, 0}), 1.23);
}

TEST_F(TensorNetwork2DContainerTest, CopyConstruction) {
  TensorNetwork2D<QLTEN_Double, U1QN> tn1(Ly, Lx);
  IndexT idx = IndexT({QNSctT(U1QN(0), 1)}, TenIndexDirType::OUT);
  DQLTensor tensor({idx, InverseIndex(idx)});
  tensor({0, 0}) = 4.56;
  tn1({2, 2}) = tensor;

  TensorNetwork2D<QLTEN_Double, U1QN> tn2 = tn1;
  EXPECT_EQ(tn2.rows(), Ly);
  EXPECT_EQ(tn2.cols(), Lx);
  EXPECT_DOUBLE_EQ(tn2({2, 2})({0, 0}), 4.56);

  // Verify independence
  tn2({2, 2})({0, 0}) = 7.89;
  EXPECT_DOUBLE_EQ(tn1({2, 2})({0, 0}), 4.56);
  EXPECT_DOUBLE_EQ(tn2({2, 2})({0, 0}), 7.89);
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
