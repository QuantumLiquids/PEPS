// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Unittests for DuoMatrix
*/

#include "gtest/gtest.h"
#include "qlpeps/two_dim_tn/framework/duomatrix.h"

using namespace qlpeps;

template<typename ElemT>
void RunTestDuoMatrixConstructorsCase(const size_t rows, const size_t cols) {
  DuoMatrix<ElemT> duomat(rows, cols);
  EXPECT_EQ(duomat.rows(), rows);
  EXPECT_EQ(duomat.cols(), cols);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      duomat({i, j}) = i * cols + j + 5;
    }
  }
  DuoMatrix<ElemT> duomat_copy(duomat);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      EXPECT_EQ(duomat_copy({i, j}), duomat({i, j}));
      EXPECT_NE(duomat_copy(i, j), duomat(i, j));
    }
  }

  auto craw_data_copy = duomat_copy.cdata();
  DuoMatrix<ElemT> duomat_moved(std::move(duomat_copy));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      EXPECT_EQ(duomat_moved(i, j), craw_data_copy[i][j]);
      EXPECT_EQ(duomat_moved({i, j}), duomat({i, j}));
    }
  }

  DuoMatrix<ElemT> duomat_copy2;
  duomat_copy2 = duomat;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      EXPECT_EQ(duomat_copy2({i, j}), duomat({i, j}));
      EXPECT_NE(duomat_copy2(i, j), duomat(i, j));
    }
  }

  auto craw_data_copy2 = duomat_copy2.cdata();
  DuoMatrix<ElemT> duomat_moved2;
  duomat_moved2 = std::move(duomat_copy2);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      EXPECT_EQ(duomat_moved2(i, j), craw_data_copy2[i][j]);
      EXPECT_EQ(duomat_moved2({i, j}), duomat({i, j}));

    }
  }
}

TEST(TestDuoMatrix, TestConstructors) {
  DuoMatrix<int> default_duomat;
  EXPECT_EQ(default_duomat.rows(), 0);
  EXPECT_EQ(default_duomat.cols(), 0);

  RunTestDuoMatrixConstructorsCase<int>(1, 1);
  RunTestDuoMatrixConstructorsCase<int>(2, 3);

  RunTestDuoMatrixConstructorsCase<double>(1, 1);
  RunTestDuoMatrixConstructorsCase<double>(3, 2);
}

TEST(TestDuoMatrix, TestElemAccess) {
  DuoMatrix<int> intduomat(1, 1);
  intduomat({0, 0}) = 3;
  EXPECT_EQ(intduomat({0, 0}), 3);

  auto pelem = intduomat.cdata()[0][0];
  intduomat({0, 0}) = 5;
  EXPECT_EQ(intduomat.cdata()[0][0], pelem);
  EXPECT_EQ(intduomat({0, 0}), 5);

  int *pelem2 = new int(4);
  delete intduomat(0, 0);
  intduomat(0, 0) = pelem2;
  EXPECT_EQ(intduomat({0, 0}), 4);
  EXPECT_NE(intduomat.cdata()[0][0], pelem);
  EXPECT_EQ(intduomat.cdata()[0][0], pelem2);
}

TEST(TestDuoMatrix, TestElemAllocDealloc) {
  DuoMatrix<int> intduomat(2, 2);

  intduomat.alloc(0, 0);
  EXPECT_NE(intduomat.cdata()[0][0], nullptr);
  EXPECT_EQ(intduomat.cdata()[1][0], nullptr);

  intduomat({0, 0}) = 3;
  EXPECT_EQ(intduomat({0, 0}), 3);

  intduomat.dealloc(0, 0);
  EXPECT_EQ(intduomat.cdata()[0][0], nullptr);

  intduomat.dealloc(1, 1);
  EXPECT_EQ(intduomat.cdata()[1][1], nullptr);
}

TEST(TestDuoMatrix, TestIterator) {
  DuoMatrix<int> intduomat(2, 2);
  for (size_t r = 0; r < intduomat.rows(); r++) {
    for (size_t c = 0; c < intduomat.cols(); c++) {
      intduomat({r, c}) = r + c;
    }
  }
  for (auto &elem : intduomat) {
    std::cout << "elem :" << elem << std::endl;
  }

  std::for_each(intduomat.begin(), intduomat.end(), [](int &element) {
    // Do something with the element
    std::cout << "Element: " << element << std::endl;
  });

  const DuoMatrix<int> &intduomat2 = intduomat;
  for (const auto &elem : intduomat2) {
    std::cout << "elem2 :" << elem << std::endl;
  }
  std::for_each(intduomat2.cbegin(), intduomat2.cend(), [](const int &element) {
    // Do something with the element
    std::cout << "Element2: " << element << std::endl;
  });
}
