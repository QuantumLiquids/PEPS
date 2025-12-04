// SPDX-License-Identifier: LGPL-3.0-only

#include "gtest/gtest.h"

#include "qlpeps/utility/observable_matrix.h"

using qlpeps::ObservableMatrix;
using qlpeps::SiteIdx;

TEST(ObservableMatrixTest, BasicAccessAndFlatten) {
  ObservableMatrix<double> mat(2, 3, 0.0);
  mat(0, 0) = 1.0;
  mat(0, 1) = 2.0;
  mat(0, 2) = 3.0;
  mat(1, 0) = 4.0;
  mat(1, 1) = 5.0;
  mat(1, 2) = 6.0;

  const auto &flat = mat.Flatten();
  ASSERT_EQ(flat.size(), 6);
  EXPECT_DOUBLE_EQ(flat[0], 1.0);
  EXPECT_DOUBLE_EQ(flat[5], 6.0);
}

TEST(ObservableMatrixTest, SiteIdxOperatorAndAdd) {
  ObservableMatrix<int> mat(3, 2, 0);
  mat(SiteIdx{0, 1}) = 5;
  mat.Add(SiteIdx{0, 1}, 3);
  EXPECT_EQ(mat(SiteIdx{0, 1}), 8);

  mat.Add(2, 0, 7);
  EXPECT_EQ(mat(2, 0), 7);

  auto data = mat.Extract();
  ASSERT_EQ(data.size(), 6);
  EXPECT_EQ(data[1], 8);
  EXPECT_EQ(data[4], 7);
  EXPECT_EQ(data[5], 0);  // ensure remaining entries untouched
}

