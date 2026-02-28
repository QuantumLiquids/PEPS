// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-24
*
* Description: Unit tests for OptimizerParams utilities (IsFirstOrder, Builder clip setters).
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"

using namespace qlpeps;

namespace {

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> CreateScalarState(double value) {
  SplitIndexTPS<TenElemT, QNT> s(1, 1, 1);
  qlten::Index<QNT> v_out({qlten::QNSector<QNT>(QNT(), 1)}, qlten::TenIndexDirType::OUT);
  qlten::Index<QNT> v_in = qlten::InverseIndex(v_out);
  s({0, 0})[0] = qlten::QLTensor<TenElemT, QNT>({v_in, v_out, v_out, v_in});
  s({0, 0})[0].Fill(QNT(), value);
  return s;
}

template<typename TenElemT, typename QNT>
std::tuple<TenElemT, SplitIndexTPS<TenElemT, QNT>, double> QuadraticEval(
    const SplitIndexTPS<TenElemT, QNT> &state) {
  SplitIndexTPS<TenElemT, QNT> grad = state;
  grad({0, 0})[0].Fill(QNT(), 1.0);
  return {TenElemT(1.0), std::move(grad), 0.0};
}

}  // namespace

TEST(OptimizerParamsTest, IsFirstOrderDetectsAlgorithms) {
  OptimizerParams::BaseParams base_params(10, 0.0, 0.0, 5, 0.1);

  // SGD
  OptimizerParams p_sgd(base_params, SGDParams());
  EXPECT_TRUE(p_sgd.IsFirstOrder());

  // AdaGrad
  OptimizerParams p_adagrad(base_params, AdaGradParams(1e-8, 0.0));
  EXPECT_TRUE(p_adagrad.IsFirstOrder());

  // Adam (even if unimplemented in optimizer, still first-order family)
  OptimizerParams p_adam(base_params, AdamParams());
  EXPECT_TRUE(p_adam.IsFirstOrder());

  // SR is not first-order
  ConjugateGradientParams cg{};
  StochasticReconfigurationParams sr{.cg_params = cg};
  OptimizerParams p_sr(base_params, sr);
  EXPECT_FALSE(p_sr.IsFirstOrder());

  // LBFGS is not first-order
  OptimizerParams p_lbfgs(base_params, LBFGSParams());
  EXPECT_FALSE(p_lbfgs.IsFirstOrder());
}

TEST(OptimizerParamsTest, BuilderClipSettersRequireBaseParams) {
  OptimizerParamsBuilder builder;
  // Calling clip setters without BaseParams should throw
  EXPECT_THROW(builder.SetClipValue(1.0), std::invalid_argument);
  EXPECT_THROW(builder.SetClipNorm(2.0), std::invalid_argument);

  // After setting BaseParams, setters should work and persist
  builder.SetMaxIterations(5).SetLearningRate(0.1);
  EXPECT_NO_THROW(builder.SetClipValue(1.0));
  EXPECT_NO_THROW(builder.SetClipNorm(2.0));

  // Need an algorithm to build
  builder.WithSGD();
  OptimizerParams params = builder.Build();
  ASSERT_TRUE(params.base_params.clip_value.has_value());
  ASSERT_TRUE(params.base_params.clip_norm.has_value());
  EXPECT_DOUBLE_EQ(*params.base_params.clip_value, 1.0);
  EXPECT_DOUBLE_EQ(*params.base_params.clip_norm, 2.0);
}

// --- CheckpointParams tests ---
TEST(CheckpointParamsTest, DefaultIsDisabled) {
  CheckpointParams ckpt;
  EXPECT_EQ(ckpt.every_n_steps, 0u);
  EXPECT_TRUE(ckpt.base_path.empty());
  EXPECT_FALSE(ckpt.IsEnabled());
}

TEST(CheckpointParamsTest, EnabledWhenConfigured) {
  CheckpointParams ckpt{10, "/tmp/ckpt"};
  EXPECT_TRUE(ckpt.IsEnabled());
}

TEST(CheckpointParamsTest, DisabledWithZeroSteps) {
  CheckpointParams ckpt{0, "/tmp/ckpt"};
  EXPECT_FALSE(ckpt.IsEnabled());
}

TEST(CheckpointParamsTest, DisabledWithEmptyPath) {
  CheckpointParams ckpt{10, ""};
  EXPECT_FALSE(ckpt.IsEnabled());
}

// --- SpikeRecoveryParams tests ---
TEST(SpikeRecoveryParamsTest, DefaultValues) {
  SpikeRecoveryParams spike;
  EXPECT_TRUE(spike.enable_auto_recover);
  EXPECT_FALSE(spike.enable_rollback);
  EXPECT_EQ(spike.redo_mc_max_retries, 2u);
  EXPECT_DOUBLE_EQ(spike.factor_err, 100.0);
  EXPECT_DOUBLE_EQ(spike.factor_grad, 1e10);
  EXPECT_DOUBLE_EQ(spike.factor_ngrad, 10.0);
  EXPECT_EQ(spike.sr_min_iters_suspicious, 1u);
  EXPECT_EQ(spike.ema_window, 50u);
  EXPECT_DOUBLE_EQ(spike.sigma_k, 10.0);
  EXPECT_TRUE(spike.log_trigger_csv_path.empty());
}

// --- Builder checkpoint/spike tests ---
TEST(OptimizerParamsBuilderTest, SetCheckpointPersists) {
  OptimizerParamsBuilder builder;
  builder.SetMaxIterations(5).SetLearningRate(0.1).WithSGD();
  builder.SetCheckpoint(20, "/tmp/test_ckpt");
  OptimizerParams params = builder.Build();
  EXPECT_TRUE(params.checkpoint_params.IsEnabled());
  EXPECT_EQ(params.checkpoint_params.every_n_steps, 20u);
  EXPECT_EQ(params.checkpoint_params.base_path, "/tmp/test_ckpt");
}

TEST(OptimizerParamsBuilderTest, SetSpikeRecoveryPersists) {
  SpikeRecoveryParams custom;
  custom.enable_auto_recover = false;
  custom.factor_err = 50.0;

  OptimizerParamsBuilder builder;
  builder.SetMaxIterations(5).SetLearningRate(0.1).WithSGD();
  builder.SetSpikeRecovery(custom);
  OptimizerParams params = builder.Build();
  EXPECT_FALSE(params.spike_recovery_params.enable_auto_recover);
  EXPECT_DOUBLE_EQ(params.spike_recovery_params.factor_err, 50.0);
}

TEST(OptimizerParamsBuilderTest, DisableSpikeRecoveryWorks) {
  OptimizerParamsBuilder builder;
  builder.SetMaxIterations(5).SetLearningRate(0.1).WithSGD();
  builder.DisableSpikeRecovery();
  OptimizerParams params = builder.Build();
  EXPECT_FALSE(params.spike_recovery_params.enable_auto_recover);
  EXPECT_FALSE(params.spike_recovery_params.enable_rollback);
}

TEST(PeriodicStepSelectorParamsTest, DefaultConfigIsDisabled) {
  PeriodicStepSelectorParams selector;
  EXPECT_FALSE(selector.enabled);
  EXPECT_EQ(selector.every_n_steps, 10u);
  EXPECT_DOUBLE_EQ(selector.phase_switch_ratio, 0.3);
  EXPECT_FALSE(selector.enable_in_deterministic);
}

TEST(InitialStepSelectorParamsTest, DefaultConfigIsDisabled) {
  InitialStepSelectorParams selector;
  EXPECT_FALSE(selector.enabled);
  EXPECT_EQ(selector.max_line_search_steps, 3u);
  EXPECT_FALSE(selector.enable_in_deterministic);
}

TEST(OptimizerParamsBuilderTest, SetPeriodicStepSelectorPersists) {
  OptimizerParamsBuilder builder;
  builder.SetMaxIterations(20).SetLearningRate(0.2).WithSGD();
  builder.SetPeriodicStepSelector(true, /*every_n_steps=*/7, /*phase_switch_ratio=*/0.4,
                              /*enable_in_deterministic=*/true);
  OptimizerParams params = builder.Build();
  EXPECT_TRUE(params.base_params.periodic_step_selector.enabled);
  EXPECT_EQ(params.base_params.periodic_step_selector.every_n_steps, 7u);
  EXPECT_DOUBLE_EQ(params.base_params.periodic_step_selector.phase_switch_ratio, 0.4);
  EXPECT_TRUE(params.base_params.periodic_step_selector.enable_in_deterministic);
}

TEST(OptimizerParamsBuilderTest, SetInitialStepSelectorPersists) {
  OptimizerParamsBuilder builder;
  builder.SetMaxIterations(20).SetLearningRate(0.2).WithSGD();
  builder.SetInitialStepSelector(true, /*max_line_search_steps=*/5,
                                 /*enable_in_deterministic=*/true);
  OptimizerParams params = builder.Build();
  EXPECT_TRUE(params.base_params.initial_step_selector.enabled);
  EXPECT_EQ(params.base_params.initial_step_selector.max_line_search_steps, 5u);
  EXPECT_TRUE(params.base_params.initial_step_selector.enable_in_deterministic);
}

// --- Factory default spike/checkpoint tests ---
TEST(OptimizerFactoryTest, FactoryCreatedParamsHaveSpikeDefaults) {
  auto params = OptimizerFactory::CreateAdaGrad(100, 0.01, 1e-8, 0.0);
  // Spike recovery defaults: S1-S3 on, S4 off
  EXPECT_TRUE(params.spike_recovery_params.enable_auto_recover);
  EXPECT_FALSE(params.spike_recovery_params.enable_rollback);
  // Checkpoint defaults: disabled
  EXPECT_FALSE(params.checkpoint_params.IsEnabled());
}

TEST(OptimizerFactoryTest, CreateLBFGSDefaultsToFixedStep) {
  auto params = OptimizerFactory::CreateLBFGS(100, 0.5, 7);
  ASSERT_TRUE(params.IsAlgorithm<LBFGSParams>());
  const auto &lbfgs = params.GetAlgorithmParams<LBFGSParams>();
  EXPECT_EQ(lbfgs.history_size, 7u);
  EXPECT_EQ(lbfgs.step_mode, LBFGSStepMode::kFixed);
  EXPECT_FALSE(lbfgs.allow_fallback_to_fixed_step);
}

TEST(OptimizerFactoryTest, CreateLBFGSAdvancedAcceptsExplicitLBFGSOptions) {
  LBFGSParams lbfgs(/*hist=*/11,
                    /*tol_grad=*/1e-6,
                    /*tol_change=*/1e-10,
                    /*max_eval=*/31,
                    /*step_mode=*/LBFGSStepMode::kStrongWolfe,
                    /*wolfe_c1=*/1e-4,
                    /*wolfe_c2=*/0.8,
                    /*min_step=*/1e-7,
                    /*max_step=*/2.0,
                    /*min_curvature=*/1e-10,
                    /*use_damping=*/true,
                    /*max_direction_norm=*/55.0,
                    /*allow_fallback_to_fixed_step=*/true,
                    /*fallback_fixed_step_scale=*/0.35);
  auto params = OptimizerFactory::CreateLBFGSAdvanced(
      /*max_iterations=*/200,
      /*energy_tolerance=*/1e-8,
      /*gradient_tolerance=*/1e-8,
      /*plateau_patience=*/20,
      /*learning_rate=*/0.7,
      lbfgs);
  const auto &got = params.GetAlgorithmParams<LBFGSParams>();
  EXPECT_EQ(got.history_size, 11u);
  EXPECT_EQ(got.max_eval, 31u);
  EXPECT_EQ(got.step_mode, LBFGSStepMode::kStrongWolfe);
  EXPECT_DOUBLE_EQ(got.max_step, 2.0);
  EXPECT_TRUE(got.allow_fallback_to_fixed_step);
  EXPECT_DOUBLE_EQ(got.fallback_fixed_step_scale, 0.35);
}

TEST(OptimizerFactoryTest, LBFGSRejectsZeroHistorySizeFailFast) {
  using TenElemT = qlten::QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  LBFGSParams lbfgs(/*hist=*/0, /*tol_grad=*/1e-8, /*tol_change=*/1e-12, /*max_eval=*/20,
                    /*step_mode=*/LBFGSStepMode::kFixed);
  auto params = OptimizerFactory::CreateLBFGSAdvanced(
      /*max_iterations=*/1, /*energy_tolerance=*/0.0, /*gradient_tolerance=*/0.0,
      /*plateau_patience=*/1, /*learning_rate=*/0.1, lbfgs);
  Optimizer<TenElemT, QNT> opt(params, MPI_COMM_SELF, 0, 1);

  SITPST init = CreateScalarState<TenElemT, QNT>(1.0);
  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
  EXPECT_THROW((void)opt.IterativeOptimize(init, QuadraticEval<TenElemT, QNT>, cb), std::invalid_argument);
}

TEST(OptimizerFactoryTest, LBFGSRejectsInvalidStrongWolfeConstantsFailFast) {
  using TenElemT = qlten::QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  LBFGSParams lbfgs(/*hist=*/5,
                    /*tol_grad=*/1e-8,
                    /*tol_change=*/1e-12,
                    /*max_eval=*/20,
                    /*step_mode=*/LBFGSStepMode::kStrongWolfe,
                    /*wolfe_c1=*/0.9,
                    /*wolfe_c2=*/0.8);  // invalid: c2 <= c1
  auto params = OptimizerFactory::CreateLBFGSAdvanced(
      /*max_iterations=*/1, /*energy_tolerance=*/0.0, /*gradient_tolerance=*/0.0,
      /*plateau_patience=*/1, /*learning_rate=*/0.1, lbfgs);
  Optimizer<TenElemT, QNT> opt(params, MPI_COMM_SELF, 0, 1);

  SITPST init = CreateScalarState<TenElemT, QNT>(1.0);
  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
  EXPECT_THROW((void)opt.IterativeOptimize(init, QuadraticEval<TenElemT, QNT>, cb), std::invalid_argument);
}

TEST(OptimizerFactoryTest, LBFGSRejectsNegativeToleranceGradFailFast) {
  using TenElemT = qlten::QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  LBFGSParams lbfgs(/*hist=*/5,
                    /*tol_grad=*/-1e-8,
                    /*tol_change=*/1e-12,
                    /*max_eval=*/20,
                    /*step_mode=*/LBFGSStepMode::kStrongWolfe,
                    /*wolfe_c1=*/1e-4,
                    /*wolfe_c2=*/0.9);
  auto params = OptimizerFactory::CreateLBFGSAdvanced(
      /*max_iterations=*/1, /*energy_tolerance=*/0.0, /*gradient_tolerance=*/0.0,
      /*plateau_patience=*/1, /*learning_rate=*/0.1, lbfgs);
  Optimizer<TenElemT, QNT> opt(params, MPI_COMM_SELF, 0, 1);

  SITPST init = CreateScalarState<TenElemT, QNT>(1.0);
  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
  EXPECT_THROW((void)opt.IterativeOptimize(init, QuadraticEval<TenElemT, QNT>, cb), std::invalid_argument);
}
