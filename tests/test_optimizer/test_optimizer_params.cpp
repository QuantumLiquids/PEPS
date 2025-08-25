// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-24
*
* Description: Unit tests for OptimizerParams utilities (IsFirstOrder, Builder clip setters).
*/

#include "gtest/gtest.h"
#include "qlpeps/optimizer/optimizer_params.h"

using namespace qlpeps;

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
  ConjugateGradientParams cg;
  OptimizerParams p_sr(base_params, StochasticReconfigurationParams(cg));
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


