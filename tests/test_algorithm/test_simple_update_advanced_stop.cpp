// SPDX-License-Identifier: LGPL-3.0-only

#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

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
        patterns_[idx].energy,  // en = e0 for mock
        std::nullopt,           // no trunc_err for mock
        0.001,                  // fake elapsed_sec
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

TEST(SimpleUpdateAdvancedStop, AdvancedStopDisabled_RunsToMaxSteps) {
  const auto peps0 = BuildInitialPEPS(/*ly=*/2, /*lx=*/2);
  SimpleUpdatePara para(/*steps=*/4, /*tau=*/0.1, /*Dmin=*/1, /*Dmax=*/2, /*Trunc_err=*/1e-8);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {10.0, 2, 1.0},
      {10.0, 2, 1.0},
      {10.0, 2, 1.0},
      {10.0, 2, 1.0},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  EXPECT_FALSE(exe.LastRunConverged());
  EXPECT_EQ(exe.LastRunExecutedSteps(), 4u);
  EXPECT_EQ(exe.GetLastRunSummary().stop_reason,
            ExecutorT::StopReason::kMaxSteps);
}

TEST(SimpleUpdateAdvancedStop, AdvancedStop_StopsWhenEnergyAndLambdaPassWithPatience) {
  const auto peps0 = BuildInitialPEPS(/*ly=*/2, /*lx=*/2);
  auto para = SimpleUpdatePara::Advanced(
      /*steps=*/10,
      /*tau=*/0.1,
      /*Dmin=*/1,
      /*Dmax=*/2,
      /*Trunc_err=*/1e-8,
      /*energy_abs_tol=*/1e-7,
      /*energy_rel_tol=*/1e-9,
      /*lambda_rel_tol=*/1e-7,
      /*patience=*/2,
      /*min_steps=*/1);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {10.0, 2, 1.00},
      {9.0, 2, 1.20},
      {9.0 + 5e-9, 2, 1.20 + 1e-9},
      {9.0 + 6e-9, 2, 1.20 + 2e-9},
      {9.0 + 7e-9, 2, 1.20 + 3e-9},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  EXPECT_TRUE(exe.LastRunConverged());
  EXPECT_LT(exe.LastRunExecutedSteps(), para.steps);
  EXPECT_EQ(exe.LastRunExecutedSteps(), 4u);
  EXPECT_EQ(exe.GetLastRunSummary().stop_reason,
            ExecutorT::StopReason::kAdvancedConverged);
}

TEST(SimpleUpdateAdvancedStop, AdvancedStop_SkipsLambdaUntilBondDimsStable) {
  const auto peps0 = BuildInitialPEPS(/*ly=*/2, /*lx=*/2);
  auto para = SimpleUpdatePara::Advanced(
      /*steps=*/10,
      /*tau=*/0.1,
      /*Dmin=*/1,
      /*Dmax=*/3,
      /*Trunc_err=*/1e-8,
      /*energy_abs_tol=*/1e-7,
      /*energy_rel_tol=*/1e-9,
      /*lambda_rel_tol=*/1e-7,
      /*patience=*/2,
      /*min_steps=*/1);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {8.0, 2, 1.00},
      {8.0 + 5e-9, 3, 1.00 + 1e-9},
      {8.0 + 6e-9, 3, 1.00 + 2e-9},
      {8.0 + 7e-9, 3, 1.00 + 3e-9},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  EXPECT_TRUE(exe.LastRunConverged());
  EXPECT_EQ(exe.LastRunExecutedSteps(), 4u);
}

TEST(SimpleUpdateAdvancedStop, AdvancedStop_StreakResetsOnGateFailure) {
  const auto peps0 = BuildInitialPEPS(/*ly=*/2, /*lx=*/2);
  auto para = SimpleUpdatePara::Advanced(
      /*steps=*/12,
      /*tau=*/0.1,
      /*Dmin=*/1,
      /*Dmax=*/2,
      /*Trunc_err=*/1e-8,
      /*energy_abs_tol=*/1e-7,
      /*energy_rel_tol=*/1e-9,
      /*lambda_rel_tol=*/1e-7,
      /*patience=*/2,
      /*min_steps=*/1);

  std::vector<MockSimpleUpdateExecutor::SweepPattern> patterns = {
      {7.0, 2, 1.0},
      {6.0, 2, 1.1},
      {6.0 + 5e-9, 2, 1.1 + 1e-9},
      {5.5, 2, 1.1 + 1e-9},
      {5.5 + 5e-9, 2, 1.1 + 2e-9},
      {5.5 + 6e-9, 2, 1.1 + 3e-9},
  };
  MockSimpleUpdateExecutor exe(para, peps0, patterns);
  exe.Execute();

  EXPECT_TRUE(exe.LastRunConverged());
  EXPECT_EQ(exe.LastRunExecutedSteps(), 6u);
}

TEST(SimpleUpdateAdvancedStop, RunSummary_ResetsEachExecuteCall) {
  const auto peps0 = BuildInitialPEPS(/*ly=*/2, /*lx=*/2);
  auto para = SimpleUpdatePara::Advanced(
      /*steps=*/8,
      /*tau=*/0.1,
      /*Dmin=*/1,
      /*Dmax=*/2,
      /*Trunc_err=*/1e-8,
      /*energy_abs_tol=*/1e-7,
      /*energy_rel_tol=*/1e-9,
      /*lambda_rel_tol=*/1e-7,
      /*patience=*/2,
      /*min_steps=*/1);

  MockSimpleUpdateExecutor exe(para, peps0,
      {
          {4.0, 2, 1.0},
          {4.0 + 5e-9, 2, 1.0 + 1e-9},
          {4.0 + 6e-9, 2, 1.0 + 2e-9},
      });
  exe.Execute();
  EXPECT_TRUE(exe.LastRunConverged());
  EXPECT_EQ(exe.LastRunExecutedSteps(), 3u);

  exe.update_para.steps = 2;
  exe.SetPatterns({
      {3.0, 2, 1.0},
      {2.0, 2, 1.5},
  });
  exe.Execute();

  EXPECT_FALSE(exe.LastRunConverged());
  EXPECT_EQ(exe.LastRunExecutedSteps(), 2u);
  EXPECT_EQ(exe.GetLastRunSummary().stop_reason,
            ExecutorT::StopReason::kMaxSteps);
}
