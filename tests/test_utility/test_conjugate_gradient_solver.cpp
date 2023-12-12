// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-17
*
* Description: GraceQ/VMC-PEPS project. Unittests for conjugate gradient solver
*/

#include "gqpeps/utility/conjugate_gradient_solver.h"
#include "gtest/gtest.h"
#include "my_vector_matrix.h"

using namespace gqten;
using namespace gqpeps;


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

  MySquareMatrix<GQTEN_Complex> zmat1({{1.0, 2.0, 3.0},
                                       {2.0, 5.0, 7.0},
                                       {3.0, 7.0, 15.0}});
  MyVector<GQTEN_Complex> zb1({11.0, 12.0, 13.0});
  MyVector<GQTEN_Complex> zx01({-1.0, 1.0, 0.0});
  MyVector<GQTEN_Complex> zx_res1({33.0, -8.0, -2.0});
  RunTestPlainCGSolverNoParallelCase(zmat1, zb1, zx01, zx_res1);
}

