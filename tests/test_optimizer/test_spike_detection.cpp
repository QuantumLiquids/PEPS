// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-09-01
*
* Description: Unit tests for EMATracker and spike detection enums.
*/

#include "gtest/gtest.h"
#include "qlpeps/optimizer/spike_detection.h"
using namespace qlpeps;

// --- EMATracker tests ---
TEST(EMATrackerTest, InitializesOnFirstUpdate) {
  EMATracker t(10);
  EXPECT_FALSE(t.IsInitialized());
  t.Update(5.0);
  EXPECT_TRUE(t.IsInitialized());
  EXPECT_DOUBLE_EQ(t.Mean(), 5.0);
  EXPECT_DOUBLE_EQ(t.Var(), 0.0);
}

TEST(EMATrackerTest, ConvergesToConstant) {
  EMATracker t(10);
  for (int i = 0; i < 200; ++i) t.Update(3.0);
  EXPECT_NEAR(t.Mean(), 3.0, 1e-10);
  EXPECT_NEAR(t.Std(), 0.0, 1e-10);
}

TEST(EMATrackerTest, TracksStepChange) {
  EMATracker t(10);
  for (int i = 0; i < 100; ++i) t.Update(1.0);
  EXPECT_NEAR(t.Mean(), 1.0, 1e-10);
  for (int i = 0; i < 100; ++i) t.Update(5.0);
  EXPECT_NEAR(t.Mean(), 5.0, 0.01);
}

TEST(EMATrackerTest, NonzeroStdForVariation) {
  EMATracker t(10);
  for (int i = 0; i < 100; ++i) t.Update(i % 2 == 0 ? 1.0 : 3.0);
  EXPECT_GT(t.Std(), 0.0);
}

TEST(EMATrackerTest, ResetClearsState) {
  EMATracker t(10);
  t.Update(5.0);
  t.Reset();
  EXPECT_FALSE(t.IsInitialized());
  EXPECT_DOUBLE_EQ(t.Mean(), 0.0);
}

// --- SignalName / ActionName tests ---
TEST(SpikeEnumTest, SignalNames) {
  EXPECT_STREQ(SignalName(SpikeSignal::kNone), "NONE");
  EXPECT_STREQ(SignalName(SpikeSignal::kS1_ErrorbarSpike), "S1_ERRORBAR");
  EXPECT_STREQ(SignalName(SpikeSignal::kS2_GradientNormSpike), "S2_GRAD_NORM");
  EXPECT_STREQ(SignalName(SpikeSignal::kS3_NaturalGradientAnomaly), "S3_NGRAD_ANOMALY");
  EXPECT_STREQ(SignalName(SpikeSignal::kS4_EMAEnergySpikeUpward), "S4_ENERGY_SPIKE");
}

TEST(SpikeEnumTest, ActionNames) {
  EXPECT_STREQ(ActionName(SpikeAction::kAccept), "ACCEPT");
  EXPECT_STREQ(ActionName(SpikeAction::kResample), "RESAMPLE");
  EXPECT_STREQ(ActionName(SpikeAction::kRollback), "ROLLBACK");
  EXPECT_STREQ(ActionName(SpikeAction::kAcceptWithWarning), "ACCEPT_WARN");
}
