// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-24
*
* Description: Unit tests for learning rate schedulers.
*/

#include "gtest/gtest.h"
#include "qlpeps/optimizer/lr_schedulers.h"

using namespace qlpeps;

TEST(LRSchedulers, ExponentialDecayBasic) {
  ExponentialDecayLR lr(0.01, 0.95, 100);
  // t=0
  EXPECT_NEAR(lr.GetLearningRate(0, 0.0), 0.01, 1e-12);
  // t=100 -> 0.01 * 0.95^(1)
  EXPECT_NEAR(lr.GetLearningRate(100, 0.0), 0.01 * std::pow(0.95, 1.0), 1e-12);
  // t=250 -> 0.01 * 0.95^(2.5)
  EXPECT_NEAR(lr.GetLearningRate(250, 0.0), 0.01 * std::pow(0.95, 2.5), 1e-12);
}

TEST(LRSchedulers, StepLRBasic) {
  StepLR lr(0.02, 200, 0.5);
  EXPECT_NEAR(lr.GetLearningRate(0, 0.0), 0.02, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(199, 0.0), 0.02, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(200, 0.0), 0.02 * 0.5, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(400, 0.0), 0.02 * 0.25, 1e-12);
}

TEST(LRSchedulers, CosineAnnealingEndpoints) {
  CosineAnnealingLR lr(0.01, 1000, 0.0);
  EXPECT_NEAR(lr.GetLearningRate(0, 0.0), 0.01, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(1000, 0.0), 0.0, 1e-12);
  double mid = lr.GetLearningRate(500, 0.0);
  // midpoint should be around (eta_min + eta_max)/2 = 0.005
  EXPECT_NEAR(mid, 0.005, 5e-3); // cosine midpoint tolerance
}

TEST(LRSchedulers, WarmupLinear) {
  WarmupLR lr(0.02, 4, 0.0);
  EXPECT_NEAR(lr.GetLearningRate(0, 0.0), 0.02 * 1.0 / 4.0, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(1, 0.0), 0.02 * 2.0 / 4.0, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(3, 0.0), 0.02 * 4.0 / 4.0, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(10, 0.0), 0.02, 1e-12);
}

TEST(LRSchedulers, MultiStepDecay) {
  MultiStepLR lr(0.01, std::vector<size_t>{10, 20}, 0.1);
  EXPECT_NEAR(lr.GetLearningRate(0, 0.0), 0.01, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(10, 0.0), 0.001, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(15, 0.0), 0.001, 1e-12);
  EXPECT_NEAR(lr.GetLearningRate(20, 0.0), 0.0001, 1e-12);
}

TEST(LRSchedulers, PlateauLRHalving) {
  PlateauLR lr(0.02, 0.5, 3, 1e-6);
  // Energy improves slightly then plateaus
  double e = 10.0;
  double lr0 = lr.GetLearningRate(0, e);
  e -= 1e-3; // improvement
  double lr1 = lr.GetLearningRate(1, e);
  // now simulate no improvement for 3 steps -> lr halves
  double lr2 = lr.GetLearningRate(2, e);
  double lr3 = lr.GetLearningRate(3, e);
  double lr4 = lr.GetLearningRate(4, e);
  EXPECT_NEAR(lr0, 0.02, 1e-12);
  EXPECT_NEAR(lr1, 0.02, 1e-12);
  EXPECT_NEAR(lr4, 0.01, 1e-12);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


