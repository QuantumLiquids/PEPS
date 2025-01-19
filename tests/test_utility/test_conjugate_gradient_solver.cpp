// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-17
*
* Description: QuantumLiquids/PEPS project. Unittests for conjugate gradient solver
*/

#include "gtest/gtest.h"
#include "qlpeps/utility/conjugate_gradient_solver.h"
#include "my_vector_matrix.h"

using namespace qlten;
using namespace qlpeps;

template<typename ElemT>
void RunTestPlainCGSolverNoParallelCase(
    const MySquareMatrix<ElemT> &mat,
    const MyVector<ElemT> &b,
    const MyVector<ElemT> &x0,
    const MyVector<ElemT> x_res
) {
  size_t iter;
  auto x = ConjugateGradientSolver(mat, b, x0, 100, 1e-16, iter);
  auto diff_vec = x - x_res;
  EXPECT_NEAR(diff_vec.NormSquare(), 0.0, 1e-13);
}

TEST(TestPlainCGSolver, NoParallel) {
  MySquareMatrix<double> dmat1({{1.0, 2.0, 3.0},
                                {2.0, 5.0, 7.0},
                                {3.0, 7.0, 15.0}});
  MyVector<double> db1({11.0, 12.0, 13.0});
  MyVector<double> dx01({-1.0, 1.0, 0.0});
  MyVector<double> dx_res1({33.0, -8.0, -2.0});
  RunTestPlainCGSolverNoParallelCase(dmat1, db1, dx01, dx_res1);

  MySquareMatrix<QLTEN_Complex> zmat1({{1.0, 2.0, 3.0},
                                       {2.0, 5.0, 7.0},
                                       {3.0, 7.0, 15.0}});
  MyVector<QLTEN_Complex> zb1({11.0, 12.0, 13.0});
  MyVector<QLTEN_Complex> zx01({-1.0, 1.0, 0.0});
  MyVector<QLTEN_Complex> zx_res1({33.0, -8.0, -2.0});
  RunTestPlainCGSolverNoParallelCase(zmat1, zb1, zx01, zx_res1);

  MySquareMatrix<QLTEN_Complex> zmat2({
                                          {QLTEN_Complex(4.3, 0.0), QLTEN_Complex(1.0, 2.0), QLTEN_Complex(0.0, -3.0)},
                                          {QLTEN_Complex(1.0, -2.0), QLTEN_Complex(5.0, 0.0), QLTEN_Complex(2.0, 1.0)},
                                          {QLTEN_Complex(0.0, 3.0), QLTEN_Complex(2.0, -1.0), QLTEN_Complex(6.0, 0.0)}
                                      });

  MyVector<QLTEN_Complex> zx_res2({QLTEN_Complex(1.0, 0.5), QLTEN_Complex(-1.0, -1.5), QLTEN_Complex(2.0, 1.0)});
  MyVector<QLTEN_Complex> zb2({
                                  QLTEN_Complex(9.3, -7.35),
                                  QLTEN_Complex(0.0, -5.0),
                                  QLTEN_Complex(7.0, 7.0)
                              });
  MyVector<QLTEN_Complex> zx02({QLTEN_Complex(0.0, 0.0), QLTEN_Complex(0.0, 0.0), QLTEN_Complex(0.0, 0.0)});
  RunTestPlainCGSolverNoParallelCase(zmat2, zb2, zx02, zx_res2);
}

