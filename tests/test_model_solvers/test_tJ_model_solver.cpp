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

const bool  SquaretJModelMixIn::enable_sc_measurement = false;
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
  ObservablesLocal<QLTEN_Double> d_obs_loc;
  model_solver.MeasureDiagonalOneAndTwoPointFunctions(configuration, d_obs_loc);
  EXPECT_EQ(d_obs_loc.one_point_functions_loc.size(), 2 * N);
  EXPECT_EQ(d_obs_loc.two_point_functions_loc.size(), 0);
  // TODO: test the content of d_obs_loc

  ObservablesLocal<QLTEN_Complex> z_obs_loc;
  model_solver.MeasureDiagonalOneAndTwoPointFunctions(configuration, z_obs_loc);
  EXPECT_EQ(z_obs_loc.one_point_functions_loc.size(), 2 * N);
  EXPECT_EQ(z_obs_loc.two_point_functions_loc.size(), 0);
}