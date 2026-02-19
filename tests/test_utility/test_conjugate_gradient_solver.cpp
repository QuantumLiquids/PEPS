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
#include <type_traits>

// Verify CGResult struct exists and has expected fields
static_assert(std::is_same_v<decltype(qlpeps::CGResult<MyVector<double>>::x), MyVector<double>>);
static_assert(std::is_same_v<decltype(qlpeps::CGResult<MyVector<double>>::residual_norm), double>);
static_assert(std::is_same_v<decltype(qlpeps::CGResult<MyVector<double>>::iterations), size_t>);
static_assert(std::is_same_v<decltype(qlpeps::CGResult<MyVector<double>>::converged), bool>);

using namespace qlten;
using namespace qlpeps;

template<typename ElemT>
void RunTestPlainCGSolverNoParallelCase(
    const MySquareMatrix<ElemT> &mat,
    const MyVector<ElemT> &b,
    const MyVector<ElemT> &x0,
    const MyVector<ElemT> x_res
) {
  auto result = ConjugateGradientSolver(mat, b, x0, 100, 1e-16, 0);
  EXPECT_TRUE(result.converged);
  EXPECT_LT(result.iterations, 100u);
  EXPECT_LT(result.residual_norm, 1e-6);
  auto diff_vec = result.x - x_res;
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

TEST(TestPlainCGSolver, RelativeToleranceScaleIndependence) {
  // Same system at two different scales should converge with the same tolerance
  MySquareMatrix<double> mat({{4.0, 1.0}, {1.0, 3.0}});

  MyVector<double> b_small({1.0, 1.0});
  MyVector<double> b_large({1e6, 1e6});
  MyVector<double> x0({0.0, 0.0});

  double tol = 1e-10;

  auto result_small = ConjugateGradientSolver(mat, b_small, x0, 100, tol, 0);
  auto result_large = ConjugateGradientSolver(mat, b_large, x0, 100, tol, 0);

  // Both should converge (not hit max_iter)
  EXPECT_TRUE(result_small.converged);
  EXPECT_TRUE(result_large.converged);
  EXPECT_LT(result_small.iterations, 100u);
  EXPECT_LT(result_large.iterations, 100u);

  // Verify solutions: A*x - b should be small relative to ||b||
  auto diff_small = mat * result_small.x - b_small;
  auto diff_large = mat * result_large.x - b_large;
  EXPECT_LT(diff_small.NormSquare() / b_small.NormSquare(), tol);
  EXPECT_LT(diff_large.NormSquare() / b_large.NormSquare(), tol);
}

TEST(TestPlainCGSolver, SerialResidueRestart) {
  // Test that residue restart parameter is accepted and solver still converges
  MySquareMatrix<double> mat({{4.0, 1.0}, {1.0, 3.0}});
  MyVector<double> b({5.0, 4.0});
  MyVector<double> x0({0.0, 0.0});
  MyVector<double> x_expected({1.0, 1.0});

  // Call with residue_restart_step = 5
  auto result = ConjugateGradientSolver(mat, b, x0, 100, 1e-16, 5);
  EXPECT_TRUE(result.converged);
  auto diff = result.x - x_expected;
  EXPECT_NEAR(diff.NormSquare(), 0.0, 1e-13);
}

TEST(TestPlainCGSolver, ZeroRhsRelativeOnlyPreservesLegacyBehavior) {
  MySquareMatrix<double> mat({
      {1.7014087728892546, -1.9258430571407281},
      {-1.9258430571407281, 2.7386516770021552}
  });
  MyVector<double> b({0.0, 0.0});
  MyVector<double> x0({-1.3252085986071422, 0.84568824604762072});

  auto result = ConjugateGradientSolver(mat, b, x0, 200, 1e-10, 0);
  EXPECT_FALSE(result.converged);
}

TEST(TestPlainCGSolver, TinyRhsRelativeOnlyPreservesLegacyBehavior) {
  MySquareMatrix<double> mat({
      {1.7014087728892546, -1.9258430571407281},
      {-1.9258430571407281, 2.7386516770021552}
  });
  MyVector<double> b({1e-300, -1e-300});
  MyVector<double> x0({-1.3252085986071422, 0.84568824604762072});

  auto result = ConjugateGradientSolver(mat, b, x0, 200, 1e-10, 0);
  EXPECT_FALSE(result.converged);
}

TEST(TestPlainCGSolver, ZeroRhsConvergesWithExplicitAbsoluteTolerance) {
  MySquareMatrix<double> mat({
      {1.7014087728892546, -1.9258430571407281},
      {-1.9258430571407281, 2.7386516770021552}
  });
  MyVector<double> b({0.0, 0.0});
  MyVector<double> x0({-1.3252085986071422, 0.84568824604762072});

  auto result = ConjugateGradientSolver(mat, b, x0, 200, 1e-10, 0, 1e-150);
  EXPECT_TRUE(result.converged);
}

TEST(TestPlainCGSolver, TinyRhsConvergesWithExplicitAbsoluteTolerance) {
  MySquareMatrix<double> mat({
      {1.7014087728892546, -1.9258430571407281},
      {-1.9258430571407281, 2.7386516770021552}
  });
  MyVector<double> b({1e-300, -1e-300});
  MyVector<double> x0({-1.3252085986071422, 0.84568824604762072});

  auto result = ConjugateGradientSolver(mat, b, x0, 200, 1e-10, 0, 1e-150);
  EXPECT_TRUE(result.converged);
}

TEST(TestPlainCGSolver, BreakdownDetection) {
  // Singular matrix — CG should report non-convergence, not crash
  MySquareMatrix<double> mat({{1.0, 0.0}, {0.0, 0.0}});  // singular
  MyVector<double> b({1.0, 1.0});
  MyVector<double> x0({0.0, 0.0});

  auto result = ConjugateGradientSolver(mat, b, x0, 100, 1e-10, 0);
  // Should not crash; should report non-convergence
  EXPECT_FALSE(result.converged);
}

TEST(TestPlainCGSolver, NonConvergenceReported) {
  MySquareMatrix<double> mat({{4.0, 1.0}, {1.0, 3.0}});
  MyVector<double> b({5.0, 4.0});
  MyVector<double> x0({0.0, 0.0});

  // max_iter=1 with tight tolerance: one iteration is not enough to converge
  auto result = ConjugateGradientSolver(mat, b, x0, 1, 1e-30, 0);
  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.iterations, 1u);
  EXPECT_GT(result.residual_norm, 0.0);
}
