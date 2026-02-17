// SPDX-License-Identifier: LGPL-3.0-only

#include "gtest/gtest.h"

#include <vector>
#include <functional>
#include <sstream>

#include "qlten/qlten.h"
#include "qlpeps/algorithm/simple_update/simple_update.h"

using namespace qlten;
using namespace qlpeps;

namespace {

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using LambdaTensor = QLTensor<RealT, QNT>;
using PEPST = SquareLatticePEPS<TenElemT, QNT>;
using ExecutorT = SimpleUpdateExecutor<TenElemT, QNT>;
using StepMetrics = typename ExecutorT::StepMetrics;

// --- Helpers (same as test_simple_update_advanced_stop.cpp) ---

LambdaTensor MakeDiagLambda(size_t dim, RealT scale) {
  IndexT idx_out({QNSctT(QNT(), dim)}, TenIndexDirType::OUT);
  IndexT idx_in = InverseIndex(idx_out);
  LambdaTensor lambda({idx_in, idx_out});
  for (size_t i = 0; i < dim; ++i) {
    lambda({i, i}) = scale * static_cast<RealT>(i + 1);
  }
  return lambda;
}

PEPST BuildInitialPEPS(size_t ly, size_t lx) {
  IndexT pb_out({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
  PEPST peps0(pb_out, ly, lx, BoundaryCondition::Open);
  std::vector<std::vector<size_t>> activates(ly, std::vector<size_t>(lx, 0));
  peps0.Initial(activates);
  return peps0;
}

class MockSimpleUpdateExecutor : public SimpleUpdateExecutor<TenElemT, QNT> {
 public:
  using SweepResult = typename SimpleUpdateExecutor<TenElemT, QNT>::SweepResult;

  struct SweepPattern {
    RealT energy;
    RealT en;                            // estimated En (norm-based energy)
    std::optional<RealT> trunc_err;      // nullopt = executor doesn't report it
    size_t lambda_dim;
    RealT lambda_scale;
  };

  MockSimpleUpdateExecutor(const SimpleUpdatePara &para,
                           const PEPST &peps0,
                           std::vector<SweepPattern> patterns)
      : SimpleUpdateExecutor<TenElemT, QNT>(para, peps0),
        patterns_(std::move(patterns)) {
    if (patterns_.empty()) {
      throw std::invalid_argument("patterns must not be empty");
    }
  }

  void SetPatterns(std::vector<SweepPattern> patterns) {
    if (patterns.empty()) {
      throw std::invalid_argument("patterns must not be empty");
    }
    patterns_ = std::move(patterns);
    sweep_idx_ = 0;
  }

 protected:
  void SetEvolveGate_(void) override {}

  SweepResult SimpleUpdateSweep_(void) override {
    const size_t idx = std::min(sweep_idx_, patterns_.size() - 1);
    ApplyLambdaPattern(patterns_[idx]);
    ++sweep_idx_;
    auto [dmin, dmax] = this->peps_.GetMinMaxBondDim();
    return SweepResult{
        patterns_[idx].energy,
        patterns_[idx].en,
        patterns_[idx].trunc_err,
        0.001,  // elapsed_sec (fake)
        dmin,
        dmax
    };
  }

 private:
  void ApplyLambdaPattern(const SweepPattern &pattern) {
    const LambdaTensor lambda = MakeDiagLambda(pattern.lambda_dim, pattern.lambda_scale);
    for (size_t row = 0; row < this->peps_.lambda_vert.rows(); ++row) {
      for (size_t col = 0; col < this->peps_.lambda_vert.cols(); ++col) {
        this->peps_.lambda_vert({row, col}) = lambda;
      }
    }
    for (size_t row = 0; row < this->peps_.lambda_horiz.rows(); ++row) {
      for (size_t col = 0; col < this->peps_.lambda_horiz.cols(); ++col) {
        this->peps_.lambda_horiz({row, col}) = lambda;
      }
    }
  }

  std::vector<SweepPattern> patterns_;
  size_t sweep_idx_ = 0;
};

} // namespace

// --- Test: StepMetrics vector is populated with correct count ---

TEST(SimpleUpdateObservability, StepMetricsCountEqualsExecutedSteps) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/5, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-10.0, -9.8, 1e-5, 2, 1.0},
      {-10.1, -9.9, 1e-6, 2, 1.0},
      {-10.2, -10.0, 1e-7, 2, 1.0},
      {-10.3, -10.1, 1e-8, 2, 1.0},
      {-10.4, -10.2, 1e-9, 2, 1.0},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  const auto &metrics = exe.GetStepMetrics();
  ASSERT_EQ(metrics.size(), 5u);
  EXPECT_EQ(exe.GetLastRunSummary().executed_steps, 5u);
}

// --- Test: StepMetrics fields are populated correctly ---

TEST(SimpleUpdateObservability, StepMetricsFieldsAreCorrect) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/2, /*tau=*/0.05, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-5.0, -4.8, 1e-4, 2, 1.0},
      {-5.1, -4.9, 1e-5, 2, 1.0},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  const auto &metrics = exe.GetStepMetrics();
  ASSERT_EQ(metrics.size(), 2u);

  // Step 0
  EXPECT_EQ(metrics[0].step_index, 0u);
  EXPECT_DOUBLE_EQ(metrics[0].tau, 0.05);
  EXPECT_DOUBLE_EQ(metrics[0].estimated_e0, -5.0);
  EXPECT_DOUBLE_EQ(metrics[0].estimated_en, -4.8);
  ASSERT_TRUE(metrics[0].trunc_err.has_value());
  EXPECT_NEAR(metrics[0].trunc_err.value(), 1e-4, 1e-10);
  EXPECT_GT(metrics[0].elapsed_sec, 0.0);
  EXPECT_FALSE(metrics[0].bond_dim_changed);

  // Step 1
  EXPECT_EQ(metrics[1].step_index, 1u);
  EXPECT_DOUBLE_EQ(metrics[1].estimated_e0, -5.1);
}

// --- Test: RunSummary extended fields ---

TEST(SimpleUpdateObservability, RunSummaryHasExtendedFields) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/3, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-10.0, -9.8, 1e-5, 2, 1.0},
      {-10.1, -9.9, 1e-6, 2, 1.0},
      {-10.2, -10.0, 1e-7, 2, 1.0},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  const auto &summary = exe.GetLastRunSummary();
  EXPECT_EQ(summary.executed_steps, 3u);
  EXPECT_FALSE(summary.converged);
  EXPECT_EQ(summary.stop_reason, ExecutorT::StopReason::kMaxSteps);
  ASSERT_TRUE(summary.final_estimated_e0.has_value());
  EXPECT_DOUBLE_EQ(summary.final_estimated_e0.value(), -10.2);
  ASSERT_TRUE(summary.final_estimated_en.has_value());
  EXPECT_DOUBLE_EQ(summary.final_estimated_en.value(), -10.0);
}

// --- Test: Observer callback receives correct step sequence ---

TEST(SimpleUpdateObservability, ObserverCallbackReceivesAllSteps) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/4, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);

  using CallbackMetrics = SimpleUpdateStepMetrics<double>;
  std::vector<CallbackMetrics> observed;
  para.step_observer = [&observed](const CallbackMetrics &m) {
    observed.push_back(m);
  };

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-1.0, -0.9, 1e-3, 2, 1.0},
      {-1.1, -1.0, 1e-4, 2, 1.0},
      {-1.2, -1.1, 1e-5, 2, 1.0},
      {-1.3, -1.2, 1e-6, 2, 1.0},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  ASSERT_EQ(observed.size(), 4u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(observed[i].step_index, i);
  }
  EXPECT_DOUBLE_EQ(observed[0].estimated_e0, -1.0);
  EXPECT_DOUBLE_EQ(observed[3].estimated_e0, -1.3);
}

// --- Test: No observer => no crash, zero behavior change ---

TEST(SimpleUpdateObservability, NoObserverNoCrash) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/2, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);
  // para.step_observer is not set (nullopt by default)

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-2.0, -1.9, 1e-5, 2, 1.0},
      {-2.1, -2.0, 1e-6, 2, 1.0},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();  // should not crash

  EXPECT_EQ(exe.GetStepMetrics().size(), 2u);
}

// --- Test: StepMetrics cleared on re-Execute ---

TEST(SimpleUpdateObservability, StepMetricsClearedOnReExecute) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/3, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-3.0, -2.9, 1e-5, 2, 1.0},
      {-3.1, -3.0, 1e-6, 2, 1.0},
      {-3.2, -3.1, 1e-7, 2, 1.0},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();
  EXPECT_EQ(exe.GetStepMetrics().size(), 3u);

  exe.update_para.steps = 2;
  exe.SetPatterns({
      {-4.0, -3.9, 1e-5, 2, 1.0},
      {-4.1, -4.0, 1e-6, 2, 1.0},
  });
  exe.Execute();
  EXPECT_EQ(exe.GetStepMetrics().size(), 2u);
  EXPECT_DOUBLE_EQ(exe.GetStepMetrics()[0].estimated_e0, -4.0);
}

// --- Test: nullopt trunc_err propagated correctly ---

TEST(SimpleUpdateObservability, NulloptTruncErrPropagated) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/2, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-1.0, -0.9, std::nullopt, 2, 1.0},   // trunc_err not reported
      {-1.1, -1.0, std::optional<RealT>(1e-5), 2, 1.0},  // trunc_err reported
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  const auto &metrics = exe.GetStepMetrics();
  ASSERT_EQ(metrics.size(), 2u);
  EXPECT_FALSE(metrics[0].trunc_err.has_value());  // nullopt
  ASSERT_TRUE(metrics[1].trunc_err.has_value());
  EXPECT_NEAR(metrics[1].trunc_err.value(), 1e-5, 1e-12);
}

// --- Test: bond_dim_changed detection ---

TEST(SimpleUpdateObservability, BondDimChangedDetection) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/3, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/3, /*Trunc_err=*/1e-8);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-1.0, -0.9, 1e-3, 2, 1.0},   // initial bond dim = 2
      {-1.1, -1.0, 1e-4, 3, 1.0},   // bond dim grows to 3
      {-1.2, -1.1, 1e-5, 3, 1.0},   // bond dim stays at 3
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  const auto &metrics = exe.GetStepMetrics();
  ASSERT_EQ(metrics.size(), 3u);
  EXPECT_FALSE(metrics[0].bond_dim_changed);  // no previous step to compare
  EXPECT_TRUE(metrics[1].bond_dim_changed);   // D changed 2->3
  EXPECT_FALSE(metrics[2].bond_dim_changed);  // D stayed at 3
}

// --- Test: Machine-readable metric line emitted when enabled ---

TEST(SimpleUpdateObservability, MachineReadableMetricsEmitted) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/2, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);
  para.emit_machine_readable_metrics = true;

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-1.0, -0.9, std::optional<RealT>(1e-3), 2, 1.0},
      {-1.1, -1.0, std::optional<RealT>(1e-4), 2, 1.0},
  };

  // Capture stdout
  std::ostringstream captured;
  std::streambuf *old = std::cout.rdbuf(captured.rdbuf());

  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  std::cout.rdbuf(old);  // Restore

  std::string output = captured.str();
  // Check that SU_METRIC lines appear
  EXPECT_NE(output.find("SU_METRIC step=0"), std::string::npos);
  EXPECT_NE(output.find("SU_METRIC step=1"), std::string::npos);
  // Check that key fields are present
  EXPECT_NE(output.find("tau="), std::string::npos);
  EXPECT_NE(output.find("e0="), std::string::npos);
  EXPECT_NE(output.find("en="), std::string::npos);
  EXPECT_NE(output.find("trunc_err="), std::string::npos);
  EXPECT_NE(output.find("elapsed_sec="), std::string::npos);
}

// --- Test: Machine-readable metrics shows N/A for nullopt trunc_err ---

TEST(SimpleUpdateObservability, MachineReadableMetricsNulloptTruncErr) {
  const auto peps0 = BuildInitialPEPS(2, 2);
  SimpleUpdatePara para(/*steps=*/1, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);
  para.emit_machine_readable_metrics = true;

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {-1.0, -0.9, std::nullopt, 2, 1.0},  // trunc_err not reported
  };

  std::ostringstream captured;
  std::streambuf *old = std::cout.rdbuf(captured.rdbuf());

  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  std::cout.rdbuf(old);

  std::string output = captured.str();
  EXPECT_NE(output.find("trunc_err=N/A"), std::string::npos);
}
