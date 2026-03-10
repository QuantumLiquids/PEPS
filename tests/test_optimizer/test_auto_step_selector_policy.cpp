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

#include <vector>

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

void EnsureMPIInitialized() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    int argc = 0;
    char **argv = nullptr;
    MPI_Init(&argc, &argv);
  }
}

struct DeterministicMinSRFixture {
  std::vector<SITPST> Ostar_samples;
  SITPST Ostar_mean;
  std::vector<TenElemT> energy_samples;
  MinSRParams minsr_params;
};

DeterministicMinSRFixture CreateDeterministicMinSRFixture() {
  EnsureMPIInitialized();
  return DeterministicMinSRFixture{
      /*Ostar_samples=*/{CreateScalarState(0.0), CreateScalarState(2.0)},
      /*Ostar_mean=*/CreateScalarState(1.0),
      /*energy_samples=*/{0.0, 2.0},
      /*minsr_params=*/MinSRParams(/*r_pinv=*/0.0, /*a_pinv=*/0.0,
                                   /*soft_cutoff=*/false,
                                   MinSRSolverMode::kReplicated)};
}

TEST(AutoStepSelectorPolicyTest, PeriodicSelectorSkipsIter0) {
  // Verify that the periodic step selector does NOT trigger at iter 0.
  // At iter 0 there is no prior energy baseline, so triggering is wasteful.
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
  ASSERT_GE(result.learning_rate_trajectory.size(), 2u);
  // Iter 0: no selector → base lr
  EXPECT_DOUBLE_EQ(result.learning_rate_trajectory[0], 3.0);
  // Iter 1: selector fires (early phase) → halves to 1.5
  EXPECT_DOUBLE_EQ(result.learning_rate_trajectory[1], 1.5);
}

TEST(AutoStepSelectorPolicyTest, LatePhaseRequiresSignificantImprovementToHalve) {
  // Verify late-phase selector does NOT halve when energy improvement < error bar.
  // max_iter=2 so the selector triggers at iter 1 (iter 0 is always skipped).
  OptimizerParams::BaseParams base_params(/*max_iter=*/2, /*energy_tol=*/0.0, /*grad_tol=*/0.0,
                                          /*patience=*/2, /*learning_rate=*/0.2);
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
    // Large error bar ensures improvement < threshold so selector keeps lr unchanged
    return {energy, std::move(grad), /*error=*/0.05};
  };

  auto result = optimizer.IterativeOptimize(CreateScalarState(1.0), evaluator);
  ASSERT_GE(result.learning_rate_trajectory.size(), 2u);
  // Iter 1: late phase selector fires but improvement (0.032) < error threshold (0.05) → no halving
  EXPECT_DOUBLE_EQ(result.learning_rate_trajectory[1], 0.2);
}

TEST(AutoStepSelectorPolicyTest, EnergyOnlyEvaluatorDispatchedBySelector) {
  // Verify that when energy_only_evaluator is provided, selector trials use it
  // instead of the full energy_evaluator.
  OptimizerParams::BaseParams base_params(/*max_iter=*/3, /*energy_tol=*/0.0, /*grad_tol=*/0.0,
                                          /*patience=*/3, /*learning_rate=*/0.1);
  base_params.periodic_step_selector = PeriodicStepSelectorParams{/*enabled=*/true, /*every_n_steps=*/1,
                                                          /*phase_switch_ratio=*/1.0,
                                                          /*enable_in_deterministic=*/true};
  OptimizerParams params(base_params, SGDParams());
  Optimizer<TenElemT, QNT> optimizer(params, MPI_COMM_SELF, /*rank=*/0, /*mpi_size=*/1);

  size_t full_eval_count = 0;
  size_t energy_only_count = 0;

  auto full_evaluator = [&](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    ++full_eval_count;
    SITPST grad = state;
    const double energy = state.NormSquare();
    return {energy, std::move(grad), /*error=*/10.0};
  };

  auto energy_only_eval = [&](const SITPST &state) -> std::pair<double, double> {
    ++energy_only_count;
    const double energy = state.NormSquare();
    return {energy, 10.0};
  };

  auto result = optimizer.IterativeOptimize(
      CreateScalarState(1.0), full_evaluator, {}, nullptr, nullptr, nullptr, energy_only_eval);

  // Main path: 3 iterations → 3 full evaluations
  EXPECT_EQ(full_eval_count, 3u);
  // Selector fires at iter 1 and 2 (skipped at iter 0), each evaluates 2 candidates
  EXPECT_GE(energy_only_count, 2u);
}

TEST(AutoStepSelectorPolicyTest, FallbackWhenEnergyOnlyEvaluatorIsNull) {
  // Verify backward compatibility: when energy_only_evaluator is null,
  // selector trials fall back to the full energy_evaluator.
  OptimizerParams::BaseParams base_params(/*max_iter=*/3, /*energy_tol=*/0.0, /*grad_tol=*/0.0,
                                          /*patience=*/3, /*learning_rate=*/0.1);
  base_params.periodic_step_selector = PeriodicStepSelectorParams{/*enabled=*/true, /*every_n_steps=*/1,
                                                          /*phase_switch_ratio=*/1.0,
                                                          /*enable_in_deterministic=*/true};
  OptimizerParams params(base_params, SGDParams());
  Optimizer<TenElemT, QNT> optimizer(params, MPI_COMM_SELF, /*rank=*/0, /*mpi_size=*/1);

  size_t full_eval_count = 0;

  auto full_evaluator = [&](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    ++full_eval_count;
    SITPST grad = state;
    const double energy = state.NormSquare();
    return {energy, std::move(grad), /*error=*/10.0};
  };

  // Pass nullptr explicitly for energy_only_evaluator
  auto result = optimizer.IterativeOptimize(
      CreateScalarState(1.0), full_evaluator, {}, nullptr, nullptr, nullptr, nullptr);

  // 3 main-path + at least 2 selector trials (iter 1 and 2) × 2 candidates each = 7+
  EXPECT_GE(full_eval_count, 5u);
}

TEST(AutoStepSelectorPolicyTest, InitialSelectorSupportsMinSR) {
  OptimizerParams::BaseParams base_params(/*max_iter=*/1, /*energy_tol=*/0.0, /*grad_tol=*/0.0,
                                          /*patience=*/1, /*learning_rate=*/0.3);
  base_params.initial_step_selector = InitialStepSelectorParams{/*enabled=*/true,
                                                                /*max_line_search_steps=*/3,
                                                                /*enable_in_deterministic=*/true};
  const auto fixture = CreateDeterministicMinSRFixture();
  OptimizerParams params(base_params, fixture.minsr_params);
  Optimizer<TenElemT, QNT> optimizer(params, MPI_COMM_SELF, /*rank=*/0, /*mpi_size=*/1);

  auto evaluator = [](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    SITPST grad = state;
    const double x = state({0, 0})[0].GetMaxAbs();
    return {x * x, std::move(grad), /*error=*/0.0};
  };

  auto result = optimizer.IterativeOptimize(
      CreateScalarState(1.0), evaluator, {},
      &fixture.Ostar_samples, &fixture.Ostar_mean, &fixture.energy_samples, nullptr);
  ASSERT_EQ(result.total_iterations, 1u);
  ASSERT_EQ(result.learning_rate_trajectory.size(), 1u);
  EXPECT_DOUBLE_EQ(result.learning_rate_trajectory.front(), 0.9);
}

TEST(AutoStepSelectorPolicyTest, PeriodicSelectorSupportsMinSR) {
  OptimizerParams::BaseParams base_params(/*max_iter=*/3, /*energy_tol=*/0.0, /*grad_tol=*/0.0,
                                          /*patience=*/3, /*learning_rate=*/3.0);
  base_params.periodic_step_selector = PeriodicStepSelectorParams{/*enabled=*/true, /*every_n_steps=*/1,
                                                          /*phase_switch_ratio=*/1.0,
                                                          /*enable_in_deterministic=*/true};
  const auto fixture = CreateDeterministicMinSRFixture();
  OptimizerParams params(base_params, fixture.minsr_params);
  Optimizer<TenElemT, QNT> optimizer(params, MPI_COMM_SELF, /*rank=*/0, /*mpi_size=*/1);

  auto evaluator = [](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    SITPST grad = state;
    const double x = state({0, 0})[0].GetMaxAbs();
    return {x * x, std::move(grad), /*error=*/0.0};
  };

  auto result = optimizer.IterativeOptimize(
      CreateScalarState(1.0), evaluator, {},
      &fixture.Ostar_samples, &fixture.Ostar_mean, &fixture.energy_samples, nullptr);
  ASSERT_GE(result.learning_rate_trajectory.size(), 2u);
  EXPECT_DOUBLE_EQ(result.learning_rate_trajectory[0], 3.0);
  EXPECT_DOUBLE_EQ(result.learning_rate_trajectory[1], 1.5);
}

} // namespace
} // namespace qlpeps
