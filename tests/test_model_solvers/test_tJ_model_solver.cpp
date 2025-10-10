/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-03-28.
*
* Description: QuantumLiquids/PEPS project. Unittests for t-J Model solvers.
*/


#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"

using namespace qlten;
using namespace qlpeps;

// Smoke test: ensure diagonal observables exist for an arbitrary configuration.
TEST(tJModelSolverTest, HandlesMeasureDiagonalOrder) {
  const double t = 1.0, J = 0.3;
  const size_t Lx = 4, Ly = 6;
  const size_t N = Lx * Ly;
  const size_t num_hole = Lx * Ly / 8;
  const size_t num_up = (Lx * Ly - num_hole) / 2;
  const size_t num_down = (Lx * Ly - num_hole - num_up);

  // random setup configuration
  Configuration configuration(Ly, Lx);
  configuration.Random(std::map < size_t, size_t > ({
    { size_t(tJSingleSiteState::Empty), num_hole },
    { size_t(tJSingleSiteState::SpinUp), num_up },
    { size_t(tJSingleSiteState::SpinDown), num_down }
  }));

  SquaretJNNModel model_solver(t, J, 0);

  using Ten = QLTEN_Double;
  SplitIndexTPS<Ten, qlten::special_qn::fZ2QN> sitps(Ly, Lx);
  BMPSTruncatePara trun_para; // default truncation for unit test
  TPSWaveFunctionComponent<Ten, qlten::special_qn::fZ2QN> tps_sample(sitps, configuration, trun_para);

  auto obs = model_solver.template EvaluateObservables<Ten, qlten::special_qn::fZ2QN>(&sitps, &tps_sample);
  ASSERT_TRUE(obs.find("spin_z") != obs.end());
  ASSERT_TRUE(obs.find("charge") != obs.end());
  EXPECT_EQ(obs["spin_z"].size(), N);
  EXPECT_EQ(obs["charge"].size(), N);
}