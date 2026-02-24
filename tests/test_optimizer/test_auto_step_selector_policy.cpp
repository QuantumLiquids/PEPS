// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-12
*
* Description: Unit tests for auto step selector policy behavior.
*/

#include "gtest/gtest.h"
#include "mpi.h"
#include "qlten/qlten.h"
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"

namespace qlpeps {
namespace {

using TenElemT = qlten::QLTEN_Double;
using QNT = qlten::special_qn::TrivialRepQN;
using SITPST = SplitIndexTPS<TenElemT, QNT>;

SITPST CreateScalarState(double value) {
  SITPST state(1, 1, 1);
  qlten::Index<QNT> v_out({qlten::QNSector<QNT>(QNT(), 1)}, qlten::TenIndexDirType::OUT);
  qlten::Index<QNT> v_in = qlten::InverseIndex(v_out);
  state({0, 0})[0] = qlten::QLTensor<TenElemT, QNT>({v_in, v_out, v_out, v_in});
  state({0, 0})[0].Fill(QNT(), value);
  return state;
}

TEST(AutoStepSelectorPolicyTest, EarlyPhaseChoosesLowerMeanEnergyIgnoringErrorbar) {
  OptimizerParams::BaseParams base_params(/*max_iter=*/4, /*energy_tol=*/0.0, /*grad_tol=*/0.0,
                                          /*patience=*/4, /*learning_rate=*/3.0);
  base_params.periodic_step_selector = PeriodicStepSelectorParams{/*enabled=*/true, /*every_n_steps=*/1,
                                                          /*phase_switch_ratio=*/1.0,
                                                          /*enable_in_deterministic=*/true};
  OptimizerParams params(base_params, SGDParams());
  Optimizer<TenElemT, QNT> optimizer(params, MPI_COMM_SELF, /*rank=*/0, /*mpi_size=*/1);

  auto evaluator = [](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    SITPST grad = state;
    const double energy = state.NormSquare();
    return {energy, std::move(grad), /*error=*/10.0};
  };

  auto result = optimizer.IterativeOptimize(CreateScalarState(1.0), evaluator);
  ASSERT_FALSE(result.learning_rate_trajectory.empty());
  EXPECT_DOUBLE_EQ(result.learning_rate_trajectory.front(), 1.5);
}

TEST(AutoStepSelectorPolicyTest, LatePhaseRequiresSignificantImprovementToHalve) {
  OptimizerParams::BaseParams base_params(/*max_iter=*/1, /*energy_tol=*/0.0, /*grad_tol=*/0.0,
                                          /*patience=*/1, /*learning_rate=*/0.2);
  base_params.periodic_step_selector = PeriodicStepSelectorParams{/*enabled=*/true, /*every_n_steps=*/1,
                                                          /*phase_switch_ratio=*/0.0,
                                                          /*enable_in_deterministic=*/true};
  OptimizerParams params(base_params, SGDParams());
  Optimizer<TenElemT, QNT> optimizer(params, MPI_COMM_SELF, /*rank=*/0, /*mpi_size=*/1);

  auto evaluator = [](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    SITPST grad = state;
    const double x = state({0, 0})[0].GetMaxAbs();
    const double target = 0.88;
    const double energy = (x - target) * (x - target);
    return {energy, std::move(grad), /*error=*/0.01};
  };

  auto result = optimizer.IterativeOptimize(CreateScalarState(1.0), evaluator);
  ASSERT_FALSE(result.learning_rate_trajectory.empty());
  EXPECT_DOUBLE_EQ(result.learning_rate_trajectory.front(), 0.2);
}

} // namespace
} // namespace qlpeps
