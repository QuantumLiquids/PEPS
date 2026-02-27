// SPDX-License-Identifier: LGPL-3.0-only

#define PLAIN_TRANSPOSE 1

#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <sstream>
#include <vector>

#include "qlten/qlten.h"
#include "qlpeps/algorithm/loop_update/loop_update.h"

using namespace qlten;
using namespace qlpeps;
using qlten::special_qn::U1QN;

namespace {

using QNT = qlten::special_qn::U1QN;
using TenElemT = TEN_ELEM_TYPE;
using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;
using PEPST = SquareLatticePEPS<TenElemT, QNT>;
using ExecutorT = LoopUpdateExecutor<TenElemT, QNT>;
using LoopGateT = LoopGates<Tensor>;
using CallbackMetrics = LoopUpdateStepMetrics<double>;

IndexT MakePhysIndex(void) {
  return IndexT({QNSctT(U1QN({QNCard("Sz", ::U1QNVal(0))}), 2)}, TenIndexDirType::OUT);
}

PEPST BuildInitialPEPS(const IndexT &pb_out) {
  PEPST peps0(pb_out, 2, 2, BoundaryCondition::Open);
  std::vector<std::vector<size_t>> activates(2, std::vector<size_t>(2, 0));
  for (size_t y = 0; y < 2; ++y) {
    for (size_t x = 0; x < 2; ++x) {
      activates[y][x] = (x + y) % 2;
    }
  }
  peps0.Initial(activates);
  return peps0;
}

LoopUpdateTruncatePara BuildTruncatePara(void) {
  ArnoldiParams arnoldi_params(1e-10, 40);
  ConjugateGradientParams cg_params(50, 1e-5, 10, 0.0);
  FullEnvironmentTruncateParams fet_params(1, 2, 1e-10, 1e-12, 12, cg_params);
  return LoopUpdateTruncatePara(arnoldi_params, 1e-7, fet_params);
}

LoopGateT BuildHeisenbergLoopGates(const IndexT &pb_out, const double tau) {
  IndexT pb_in = InverseIndex(pb_out);
  IndexT vb_out({QNSctT(U1QN({QNCard("Sz", ::U1QNVal(0))}), 5)}, TenIndexDirType::OUT);
  IndexT vb_in = InverseIndex(vb_out);

  const std::array<size_t, 4> ns = {1, 1, 1, 1};
  LoopGateT gates;
  for (size_t i = 0; i < 4; ++i) {
    auto &gate = gates[i];
    gate = Tensor({vb_in, pb_in, pb_out, vb_out});
    // D=5 exact cyclic MPO for 2x2 Heisenberg loop: numerically stable real-valued fixture.
    gate({0, 0, 0, 0}) = 1.0;
    gate({0, 1, 1, 0}) = 1.0;
    gate({0, 0, 0, 1}) = 0.5;
    gate({0, 1, 1, 1}) = -0.5;
    gate({0, 1, 0, 2}) = 1.0;
    gate({0, 0, 1, 3}) = 1.0;
    gate({1, 0, 0, 4}) = -0.5 * tau / double(ns[i]);
    gate({1, 1, 1, 4}) = 0.5 * tau / double(ns[i]);
    gate({2, 0, 1, 4}) = -tau / (2.0 * double(ns[i]));
    gate({3, 1, 0, 4}) = -tau / (2.0 * double(ns[i]));
    gate({4, 0, 0, 0}) = 1.0;
    gate({4, 1, 1, 0}) = 1.0;
  }
  return gates;
}

DuoMatrix<LoopGateT> BuildHeisenbergEvolveGates(const IndexT &pb_out, const double tau) {
  DuoMatrix<LoopGateT> evolve_gates(1, 1);
  evolve_gates({0, 0}) = BuildHeisenbergLoopGates(pb_out, tau);
  return evolve_gates;
}

std::unique_ptr<ExecutorT> BuildExecutor(const LoopUpdatePara &para) {
  const IndexT pb_out = MakePhysIndex();
  const auto peps0 = BuildInitialPEPS(pb_out);
  const auto evolve_gates = BuildHeisenbergEvolveGates(pb_out, para.tau);
  return std::make_unique<ExecutorT>(para, evolve_gates, peps0);
}

}  // namespace

TEST(LoopUpdateObservability, AdvancedStopDisabledRunsToMaxSteps) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  LoopUpdatePara para(BuildTruncatePara(), /*steps=*/2, /*tau=*/0.02);
  auto exe = BuildExecutor(para);
  exe->Execute();

  EXPECT_FALSE(exe->LastRunConverged());
  EXPECT_EQ(exe->LastRunExecutedSteps(), 2u);
  EXPECT_EQ(exe->GetLastRunSummary().stop_reason, ExecutorT::StopReason::kMaxSteps);
}

TEST(LoopUpdateObservability, AdvancedStopEnabledStopsEarly) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  auto para = LoopUpdatePara::Advanced(
      BuildTruncatePara(),
      /*steps=*/3,
      /*tau=*/0.02,
      /*energy_abs_tol=*/1e6,
      /*energy_rel_tol=*/1e6,
      /*lambda_rel_tol=*/1e6,
      /*patience=*/1,
      /*min_steps=*/1);
  auto exe = BuildExecutor(para);
  exe->Execute();

  EXPECT_TRUE(exe->LastRunConverged());
  EXPECT_LT(exe->LastRunExecutedSteps(), para.steps);
  EXPECT_EQ(exe->GetLastRunSummary().stop_reason, ExecutorT::StopReason::kAdvancedConverged);
}

TEST(LoopUpdateObservability, RunSummaryAndMetricsResetEachExecuteCall) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  LoopUpdatePara para(BuildTruncatePara(), /*steps=*/1, /*tau=*/0.02);
  auto exe = BuildExecutor(para);

  exe->Execute();
  EXPECT_EQ(exe->LastRunExecutedSteps(), 1u);
  EXPECT_EQ(exe->GetStepMetrics().size(), 1u);

  exe->Execute();
  EXPECT_EQ(exe->LastRunExecutedSteps(), 1u);
  EXPECT_EQ(exe->GetStepMetrics().size(), 1u);
  EXPECT_EQ(exe->GetLastRunSummary().stop_reason, ExecutorT::StopReason::kMaxSteps);
}

TEST(LoopUpdateObservability, StepMetricsFieldsArePopulated) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  LoopUpdatePara para(BuildTruncatePara(), /*steps=*/1, /*tau=*/0.02);
  auto exe = BuildExecutor(para);
  exe->Execute();

  const auto &metrics = exe->GetStepMetrics();
  ASSERT_EQ(metrics.size(), exe->LastRunExecutedSteps());
  ASSERT_EQ(metrics.size(), 1u);

  EXPECT_EQ(metrics[0].step_index, 0u);
  EXPECT_DOUBLE_EQ(metrics[0].tau, 0.02);
  EXPECT_TRUE(std::isfinite(static_cast<double>(metrics[0].estimated_e0)));
  EXPECT_TRUE(std::isfinite(static_cast<double>(metrics[0].estimated_en)));
  EXPECT_FALSE(metrics[0].trunc_err.has_value());
  EXPECT_GT(metrics[0].elapsed_sec, 0.0);
  EXPECT_FALSE(metrics[0].bond_dim_changed);
}

TEST(LoopUpdateObservability, ObserverCallbackReceivesAllStepsInOrder) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  LoopUpdatePara para(BuildTruncatePara(), /*steps=*/2, /*tau=*/0.02);
  std::vector<CallbackMetrics> observed;
  para.step_observer = [&observed](const CallbackMetrics &m) {
    observed.push_back(m);
  };

  auto exe = BuildExecutor(para);
  exe->Execute();

  ASSERT_EQ(observed.size(), 2u);
  EXPECT_EQ(observed[0].step_index, 0u);
  EXPECT_EQ(observed[1].step_index, 1u);
}

TEST(LoopUpdateObservability, MachineReadableMetricsEmitted) {
  qlten::hp_numeric::SetTensorManipulationThreads(1);

  LoopUpdatePara para(BuildTruncatePara(), /*steps=*/1, /*tau=*/0.02);
  para.emit_machine_readable_metrics = true;

  std::ostringstream captured;
  std::streambuf *old = std::cout.rdbuf(captured.rdbuf());

  auto exe = BuildExecutor(para);
  exe->Execute();

  std::cout.rdbuf(old);
  const std::string output = captured.str();

  EXPECT_NE(output.find("LU_METRIC step=0"), std::string::npos);
  EXPECT_NE(output.find("tau="), std::string::npos);
  EXPECT_NE(output.find("e0="), std::string::npos);
  EXPECT_NE(output.find("en="), std::string::npos);
  EXPECT_NE(output.find("trunc_err=N/A"), std::string::npos);
  EXPECT_NE(output.find("elapsed_sec="), std::string::npos);
}
