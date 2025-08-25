// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-24
*
* Description: Integration-style unit test to verify gradient clipping is applied
*              only for first-order optimizers in Optimizer::IterativeOptimize.
 *
 * NOTE:
 * - This test ONLY covers first-order optimizers (SGD/AdaGrad/Adam) with gradient
 *   preprocessing (clip_value, clip_norm).
 * - It does NOT verify that second-order methods (SR/L-BFGS) skip clipping.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"

using namespace qlten;
using namespace qlpeps;

// Minimal fake energy evaluator: returns fixed gradient with a single large value
template<typename TenElemT, typename QNT>
static std::tuple<TenElemT, SplitIndexTPS<TenElemT, QNT>, double>
FakeEnergyEvaluator(const SplitIndexTPS<TenElemT, QNT>& state) {
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  SITPST grad(state.rows(), state.cols(), state.PhysicalDim());
  for (size_t r=0;r<state.rows();++r) {
    for (size_t c=0;c<state.cols();++c) {
      for (size_t i=0;i<state.PhysicalDim();++i) {
        // Copy structure to ensure valid blocks exist
        grad({r,c})[i] = state({r,c})[i];
        grad({r,c})[i].Fill(QNT(), 100.0); // large gradient
      }
    }
  }
  TenElemT energy = TenElemT(0.0);
  double err = 0.0;
  return {energy, grad, err};
}

template<typename TenElemT, typename QNT>
static SplitIndexTPS<TenElemT, QNT> CreateZeroState(size_t Ly=1, size_t Lx=1, size_t phy=1) {
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  SITPST s(Ly, Lx, phy);
  // Create minimal 4-leg tensor with 1-dim indices to avoid scalar tensors
  Index<QNT> v_out({QNSector<QNT>(QNT(), 1)}, OUT);
  Index<QNT> v_in  = InverseIndex(v_out);
  for (size_t r=0;r<Ly;++r) for (size_t c=0;c<Lx;++c) for (size_t i=0;i<phy;++i) {
    s({r,c})[i] = QLTensor<TenElemT, QNT>({v_in, v_out, v_out, v_in});
    s({r,c})[i].Fill(QNT(), 0.0);
  }
  return s;
}

TEST(OptimizerGradientPreprocessing, FirstOrderClipsGradient) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  // Build optimizer: SGD with clip_value = 1.0
  OptimizerParams::BaseParams base(1, 0.0, 0.0, 1, 0.1);
  base.clip_value = 1.0;
  OptimizerParams params(base, SGDParams());

  // Dummy single-rank MPI
  MPI_Comm comm = MPI_COMM_SELF;
  Optimizer<TenElemT, QNT> opt(params, comm, 0, 1);

  SITPST init = CreateZeroState<TenElemT, QNT>(1,1,1);

  auto evaluator = [&](const SITPST& s){ return FakeEnergyEvaluator<TenElemT, QNT>(s); };
  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;

  // Run one iteration; ensure the applied update uses clipped gradient (<= 1.0 max abs)
  auto res = opt.IterativeOptimize(init, evaluator, cb);
  // We can't access preprocessed_gradient directly; instead, check that the state update magnitude
  // is bounded by step_length*clip_value due to in-place clipping.
  // Because initial state is zeros and learning rate is 0.1, max element should be <= 0.1.
  for (auto &vec : res.optimized_state) {
    for (auto &ten : vec) {
      if (ten.IsDefault()) continue;
      EXPECT_LE(ten.GetMaxAbs(), 0.1 + 1e-12);
    }
  }
}

TEST(OptimizerGradientPreprocessing, ClipNormClipsGlobalNorm) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  // Build optimizer: SGD with clip_norm = 1.0 (global norm clip)
  OptimizerParams::BaseParams base(1, 0.0, 0.0, 1, 0.1);
  base.clip_norm = 1.0;
  OptimizerParams params(base, SGDParams());

  MPI_Comm comm = MPI_COMM_SELF;
  Optimizer<TenElemT, QNT> opt(params, comm, 0, 1);

  SITPST init = CreateZeroState<TenElemT, QNT>(1,1,1);

  auto evaluator = [&](const SITPST& s){ return FakeEnergyEvaluator<TenElemT, QNT>(s); };
  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;

  // With gradient element 100 and clip_norm=1, preprocessed grad becomes 1.
  // Update magnitude per element should be <= 0.1 (lr * 1).
  auto res = opt.IterativeOptimize(init, evaluator, cb);
  for (auto &vec : res.optimized_state) {
    for (auto &ten : vec) {
      if (ten.IsDefault()) continue;
      EXPECT_LE(ten.GetMaxAbs(), 0.1 + 1e-12);
    }
  }
}
