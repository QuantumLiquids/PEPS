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
// converged() is now a method, not a field — no static_assert needed

using namespace qlten;
using namespace qlpeps;

template<typename ElemT>
void RunTestPlainCGSolverNoParallelCase(
    const MySquareMatrix<ElemT> &mat,
    const MyVector<ElemT> &b,
    const MyVector<ElemT> &x0,
    const MyVector<ElemT> x_res
) {
  ConjugateGradientParams params{.max_iter = 100, .relative_tolerance = 1e-16};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  EXPECT_TRUE(result.converged());
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

  ConjugateGradientParams params{.max_iter = 100, .relative_tolerance = tol};
  auto result_small = ConjugateGradientSolver(mat, b_small, x0, params);
  auto result_large = ConjugateGradientSolver(mat, b_large, x0, params);

  // Both should converge (not hit max_iter)
  EXPECT_TRUE(result_small.converged());
  EXPECT_TRUE(result_large.converged());
  EXPECT_LT(result_small.iterations, 100u);
  EXPECT_LT(result_large.iterations, 100u);

  // Exact solutions: A = [[4,1],[1,3]], det = 11
  //   b = [1,1]   -> x = [2/11, 3/11]
  //   b = [1e6,1e6] -> x = [2e6/11, 3e6/11]
  MyVector<double> x_expected_small({2.0 / 11.0, 3.0 / 11.0});
  MyVector<double> x_expected_large({2.0e6 / 11.0, 3.0e6 / 11.0});
  auto diff_small = result_small.x - x_expected_small;
  auto diff_large = result_large.x - x_expected_large;
  EXPECT_NEAR(diff_small.NormSquare(), 0.0, 1e-13);
  EXPECT_NEAR(diff_large.NormSquare() / x_expected_large.NormSquare(), 0.0, 1e-13);
}

TEST(TestPlainCGSolver, SerialResidueRestart) {
  // Test that residue restart parameter is accepted and solver still converges
  MySquareMatrix<double> mat({{4.0, 1.0}, {1.0, 3.0}});
  MyVector<double> b({5.0, 4.0});
  MyVector<double> x0({0.0, 0.0});
  MyVector<double> x_expected({1.0, 1.0});

  // Call with residual_recompute_interval = 5
  ConjugateGradientParams params{.max_iter = 100, .relative_tolerance = 1e-16,
                                 .residual_recompute_interval = 5};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  EXPECT_TRUE(result.converged());
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

  ConjugateGradientParams params{.max_iter = 200, .relative_tolerance = 1e-10};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  EXPECT_FALSE(result.converged());
}

TEST(TestPlainCGSolver, TinyRhsRelativeOnlyPreservesLegacyBehavior) {
  MySquareMatrix<double> mat({
      {1.7014087728892546, -1.9258430571407281},
      {-1.9258430571407281, 2.7386516770021552}
  });
  MyVector<double> b({1e-300, -1e-300});
  MyVector<double> x0({-1.3252085986071422, 0.84568824604762072});

  ConjugateGradientParams params{.max_iter = 200, .relative_tolerance = 1e-10};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  EXPECT_FALSE(result.converged());
}

TEST(TestPlainCGSolver, ZeroRhsConvergesWithExplicitAbsoluteTolerance) {
  MySquareMatrix<double> mat({
      {1.7014087728892546, -1.9258430571407281},
      {-1.9258430571407281, 2.7386516770021552}
  });
  MyVector<double> b({0.0, 0.0});
  MyVector<double> x0({-1.3252085986071422, 0.84568824604762072});

  // absolute_tolerance must be achievable: CG on a 2x2 SPD system
  // reaches ||r|| ~ 1e-14 in 2 iterations; 1e-10 is comfortably above that.
  ConjugateGradientParams params{.max_iter = 200, .relative_tolerance = 1e-10,
                                 .absolute_tolerance = 1e-10};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  EXPECT_TRUE(result.converged());
  // Exact solution: x = {0, 0}
  MyVector<double> x_expected({0.0, 0.0});
  auto diff = result.x - x_expected;
  EXPECT_NEAR(diff.NormSquare(), 0.0, 1e-20);
}

TEST(TestPlainCGSolver, TinyRhsConvergesWithExplicitAbsoluteTolerance) {
  MySquareMatrix<double> mat({
      {1.7014087728892546, -1.9258430571407281},
      {-1.9258430571407281, 2.7386516770021552}
  });
  MyVector<double> b({1e-300, -1e-300});
  MyVector<double> x0({-1.3252085986071422, 0.84568824604762072});

  // With b ~ 1e-300 and x0 ~ O(1), CG's floating-point residual bottoms
  // out at ~1e-14 (dominated by roundoff in A*x, not by b).
  ConjugateGradientParams params{.max_iter = 200, .relative_tolerance = 1e-10,
                                 .absolute_tolerance = 1e-10};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  EXPECT_TRUE(result.converged());
  // Exact solution ~ O(1e-300), indistinguishable from zero in double precision
  MyVector<double> x_expected({0.0, 0.0});
  auto diff = result.x - x_expected;
  EXPECT_NEAR(diff.NormSquare(), 0.0, 1e-20);
}

TEST(TestPlainCGSolver, BreakdownDetection) {
  // Singular matrix — CG should report non-convergence, not crash
  MySquareMatrix<double> mat({{1.0, 0.0}, {0.0, 0.0}});  // singular
  MyVector<double> b({1.0, 1.0});
  MyVector<double> x0({0.0, 0.0});

  ConjugateGradientParams params{.max_iter = 100, .relative_tolerance = 1e-10};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  // Should not crash; should report non-convergence
  EXPECT_FALSE(result.converged());
}

TEST(TestPlainCGSolver, NonConvergenceReported) {
  MySquareMatrix<double> mat({{4.0, 1.0}, {1.0, 3.0}});
  MyVector<double> b({5.0, 4.0});
  MyVector<double> x0({0.0, 0.0});

  // max_iter=1 with tight tolerance: one iteration is not enough to converge
  ConjugateGradientParams params{.max_iter = 1, .relative_tolerance = 1e-30};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  EXPECT_FALSE(result.converged());
  EXPECT_EQ(result.iterations, 1u);
  EXPECT_GT(result.residual_norm, 0.0);
}

// ---------------------------------------------------------------------------
// Termination reason tests
// ---------------------------------------------------------------------------

TEST(TestPlainCGSolver, IndefiniteMatrixDetection) {
  // Diagonal matrix with a negative entry: diag(1, -1, 1).
  // With b directed along the negative-eigenvalue axis, the very first
  // search direction p = r = b gives p^T A p < 0, triggering kIndefiniteMatrix.
  MySquareMatrix<double> neg_mat({{1.0, 0.0, 0.0},
                                  {0.0, -1.0, 0.0},
                                  {0.0, 0.0, 1.0}});
  MyVector<double> b({0.0, 1.0, 0.0});
  MyVector<double> x0({0.0, 0.0, 0.0});
  ConjugateGradientParams params{.max_iter = 100, .relative_tolerance = 1e-10};
  auto result = ConjugateGradientSolver(neg_mat, b, x0, params);
  EXPECT_EQ(result.reason, CGTerminationReason::kIndefiniteMatrix);
  EXPECT_FALSE(result.converged());
}

// A matrix wrapper that injects Inf into the result after a given number of
// matrix-vector multiplications.  Inf passes the pap > 0 check but causes
// 0 * Inf = NaN in the residual update, triggering kNumericalBreakdown.
class InfInjectingMatrix {
 public:
  InfInjectingMatrix(const MySquareMatrix<double> &inner, size_t inf_after)
      : inner_(inner), inf_after_(inf_after) {}

  MyVector<double> operator*(const MyVector<double> &v) const {
    if (++call_count_ > inf_after_) {
      MyVector<double> result = inner_ * v;
      result.GetElements()[0] = std::numeric_limits<double>::infinity();
      return result;
    }
    return inner_ * v;
  }

 private:
  MySquareMatrix<double> inner_;
  mutable size_t call_count_ = 0;
  size_t inf_after_;
};

TEST(TestPlainCGSolver, NumericalBreakdownDetection) {
  // The first mat-vec (initial residual r = b - A*x0) is clean.
  // The second mat-vec (ap = A*p in iteration k=0) returns Inf in the
  // first element.  pap = p * ap = Inf > 0, so the indefinite check
  // passes.  Then alpha = rk_norm / Inf = 0, and the residual update
  // r = r - alpha * ap involves 0 * Inf = NaN, making the residual
  // norm non-finite and triggering kNumericalBreakdown.
  InfInjectingMatrix inf_mat(
      MySquareMatrix<double>({{2.0, 0.0, 0.0},
                              {0.0, 3.0, 0.0},
                              {0.0, 0.0, 5.0}}),
      1);  // Inf starting from the 2nd mat-vec
  MyVector<double> b({1.0, 2.0, 3.0});
  MyVector<double> x0({0.0, 0.0, 0.0});
  ConjugateGradientParams params{.max_iter = 100, .relative_tolerance = 1e-10};
  auto result = ConjugateGradientSolver(inf_mat, b, x0, params);
  EXPECT_EQ(result.reason, CGTerminationReason::kNumericalBreakdown);
  EXPECT_FALSE(result.converged());
}

TEST(TestPlainCGSolver, StagnationDetection) {
  // 10x10 diagonal system with eigenvalues spanning 15 orders of magnitude.
  // CG reaches machine precision quickly but cannot satisfy the impossibly
  // tight relative tolerance (1e-30).  The steps become smaller than
  // eps * ||x||, triggering kStagnated after 3 consecutive tiny-step
  // iterations.
  std::vector<std::vector<double>> diag_data(10, std::vector<double>(10, 0.0));
  std::vector<double> b_data(10);
  std::vector<double> x0_data(10, 0.0);
  for (size_t i = 0; i < 10; ++i) {
    double eigenvalue = std::pow(10.0, -static_cast<double>(i) * 15.0 / 9.0);
    diag_data[i][i] = eigenvalue;
    b_data[i] = eigenvalue;  // exact solution is (1, 1, ..., 1)
  }
  MySquareMatrix<double> ill_mat(diag_data);
  MyVector<double> b(b_data);
  MyVector<double> x0(x0_data);
  ConjugateGradientParams params{.max_iter = 10000, .relative_tolerance = 1e-30};
  auto result = ConjugateGradientSolver(ill_mat, b, x0, params);
  EXPECT_EQ(result.reason, CGTerminationReason::kStagnated);
  EXPECT_FALSE(result.converged());
}

TEST(TestPlainCGSolver, OrthogonalityRestartConverges) {
  // With an aggressively low orthogonality threshold (0.01), the solver
  // restarts the search direction much more frequently than the default.
  // Verify that CG still converges to the correct answer despite the
  // frequent restarts.
  MySquareMatrix<double> mat({{2.0, 0.0, 0.0, 0.0},
                              {0.0, 3.0, 0.0, 0.0},
                              {0.0, 0.0, 5.0, 0.0},
                              {0.0, 0.0, 0.0, 7.0}});
  MyVector<double> b({1.0, 2.0, 3.0, 4.0});
  MyVector<double> x0({0.0, 0.0, 0.0, 0.0});
  ConjugateGradientParams params{.max_iter = 100, .relative_tolerance = 1e-10,
                                 .orthogonality_threshold = 0.01};
  auto result = ConjugateGradientSolver(mat, b, x0, params);
  EXPECT_TRUE(result.converged());
  EXPECT_EQ(result.reason, CGTerminationReason::kConverged);
  // Exact solution: x = {1/2, 2/3, 3/5, 4/7}
  MyVector<double> x_expected({0.5, 2.0 / 3.0, 0.6, 4.0 / 7.0});
  auto diff = result.x - x_expected;
  EXPECT_NEAR(diff.NormSquare(), 0.0, 1e-13);
}
