// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-12
*
* Description: Unit tests for LBFGS recovery policy:
*              - RESAMPLE keeps LBFGS history
*              - ROLLBACK restores one-step LBFGS snapshot
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"

using namespace qlten;
using namespace qlpeps;

namespace {

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> CreateScalarState(double value) {
  SplitIndexTPS<TenElemT, QNT> s(1, 1, 1);
  Index<QNT> v_out({QNSector<QNT>(QNT(), 1)}, OUT);
  Index<QNT> v_in = InverseIndex(v_out);
  s({0, 0})[0] = QLTensor<TenElemT, QNT>({v_in, v_out, v_out, v_in});
  s({0, 0})[0].Fill(QNT(), value);
  return s;
}

template<typename TenElemT, typename QNT>
double ExtractScalarAbs(const SplitIndexTPS<TenElemT, QNT>& s) {
  return s({0, 0})[0].GetMaxAbs();
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> BuildGradientLike(const SplitIndexTPS<TenElemT, QNT>& state, double grad_value) {
  SplitIndexTPS<TenElemT, QNT> grad = state;
  grad({0, 0})[0].Fill(QNT(), grad_value);
  return grad;
}

}  // namespace

TEST(OptimizerLBFGSRecoveryPolicy, ResampleKeepsHistory) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  OptimizerParams::BaseParams base(/*max_iterations=*/3,
                                   /*energy_tolerance=*/0.0,
                                   /*gradient_tolerance=*/0.0,
                                   /*plateau_patience=*/20,
                                   /*learning_rate=*/0.5);
  LBFGSParams lbfgs(/*hist=*/5,
                    /*tol_grad=*/1e-8,
                    /*tol_change=*/1e-12,
                    /*max_eval=*/20,
                    /*step_mode=*/LBFGSStepMode::kFixed);
  SpikeRecoveryParams spike;
  spike.enable_auto_recover = true;
  spike.redo_mc_max_retries = 1;   // first spike => RESAMPLE
  spike.factor_err = 1e100;
  spike.factor_grad = 1.05;
  spike.enable_rollback = false;
  spike.ema_window = 8;
  OptimizerParams params(base, lbfgs, CheckpointParams{}, spike);

  Optimizer<TenElemT, QNT> opt(params, MPI_COMM_SELF, 0, 1);
  SITPST init = CreateScalarState<TenElemT, QNT>(0.0);

  size_t call_count = 0;
  std::vector<double> states_seen;
  auto evaluator = [&](const SITPST& state) -> std::tuple<TenElemT, SITPST, double> {
    ++call_count;
    const double x = ExtractScalarAbs<TenElemT, QNT>(state);
    states_seen.push_back(x);

    double grad = 3.0 * (x - 2.0);
    double energy = 1.5 * (x - 2.0) * (x - 2.0);
    if (call_count == 2) {
      // Force a one-time gradient spike on iter-1 first attempt -> RESAMPLE.
      grad = 120.0;
      energy = 1e4;
    }

    SITPST g = BuildGradientLike<TenElemT, QNT>(state, grad);
    return {TenElemT(energy), std::move(g), 0.0};
  };

  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
  (void)opt.IterativeOptimize(init, evaluator, cb);

  ASSERT_GE(states_seen.size(), 4u);
  // Call sequence:
  //   c1: iter0 at x=0
  //   c2: iter1 attempt1 at x=3 (spike -> resample)
  //   c3: iter1 attempt2 at x=3 (accepted)
  //   c4: iter2 at updated x
  // With history preserved, x should move 3 -> 2.5 (not steepest 1.5).
  EXPECT_NEAR(states_seen[1], 3.0, 1e-9);
  EXPECT_NEAR(states_seen[2], 3.0, 1e-9);
  EXPECT_NEAR(states_seen[3], 2.5, 1e-6);
}

TEST(OptimizerLBFGSRecoveryPolicy, RollbackRestoresOneStepSnapshot) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  OptimizerParams::BaseParams base(/*max_iterations=*/3,
                                   /*energy_tolerance=*/0.0,
                                   /*gradient_tolerance=*/0.0,
                                   /*plateau_patience=*/20,
                                   /*learning_rate=*/0.5);
  LBFGSParams lbfgs(/*hist=*/5,
                    /*tol_grad=*/1e-8,
                    /*tol_change=*/1e-12,
                    /*max_eval=*/20,
                    /*step_mode=*/LBFGSStepMode::kFixed);
  SpikeRecoveryParams spike;
  spike.enable_auto_recover = true;
  spike.redo_mc_max_retries = 0;   // first spike => no resample path
  spike.factor_err = 1e100;
  spike.factor_grad = 1.05;
  spike.enable_rollback = true;    // -> ROLLBACK
  spike.ema_window = 8;
  OptimizerParams params(base, lbfgs, CheckpointParams{}, spike);

  Optimizer<TenElemT, QNT> opt(params, MPI_COMM_SELF, 0, 1);
  SITPST init = CreateScalarState<TenElemT, QNT>(0.0);

  size_t call_count = 0;
  std::vector<double> states_seen;
  auto evaluator = [&](const SITPST& state) -> std::tuple<TenElemT, SITPST, double> {
    ++call_count;
    const double x = ExtractScalarAbs<TenElemT, QNT>(state);
    states_seen.push_back(x);

    double grad = 3.0 * (x - 2.0);
    double energy = 1.5 * (x - 2.0) * (x - 2.0);
    if (call_count == 2) {
      // Force one-time spike on iter-1 first attempt -> ROLLBACK.
      grad = 120.0;
      energy = 1e4;
    }

    SITPST g = BuildGradientLike<TenElemT, QNT>(state, grad);
    return {TenElemT(energy), std::move(g), 0.0};
  };

  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
  (void)opt.IterativeOptimize(init, evaluator, cb);

  ASSERT_GE(states_seen.size(), 4u);
  // Call sequence:
  //   c1: iter0 at x=0
  //   c2: iter1 attempt1 at x=3 (spike -> rollback to x=0)
  //   c3: iter1 attempt2 at x=0 (accepted)
  //   c4: iter2 at updated x
  // With one-step snapshot restore, history is rewound and update is steepest again: x=3.
  EXPECT_NEAR(states_seen[1], 3.0, 1e-9);
  EXPECT_NEAR(states_seen[2], 0.0, 1e-9);
  EXPECT_NEAR(states_seen[3], 3.0, 1e-6);
}
