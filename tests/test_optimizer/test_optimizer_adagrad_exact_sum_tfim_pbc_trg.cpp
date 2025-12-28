// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2025-12-15
 *
 * Description: Regression test: exact-sum + optimizer for 2x2 TFIM PBC using TRG.
 */

#include "gtest/gtest.h"

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"

#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/transverse_field_ising_square_pbc.h"
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/trg_contractor.h"

using namespace qlten;
using namespace qlpeps;

namespace {

double Calculate2x2OBCTransverseIsingEnergy(double J, double h) {
  // Same analytic formula used in tests/test_optimizer/test_optimizer_adagrad_exact_sum.cpp.
  // For N=4 sites in the even-parity (ground-state) sector:
  // k = π/4, 3π/4 (k>0 modes).
  const double e1 = 2.0 * std::sqrt(J * J + h * h - 2.0 * J * h * std::cos(M_PI / 4.0));
  const double e2 = 2.0 * std::sqrt(J * J + h * h - 2.0 * J * h * std::cos(3.0 * M_PI / 4.0));
  return -(e1 + e2);
}

std::vector<Configuration> GenerateAllConfigs2x2Binary() {
  std::vector<Configuration> out;
  out.reserve(16);
  for (size_t mask = 0; mask < 16; ++mask) {
    std::vector<size_t> v(4);
    v[0] = (mask >> 0) & 1;
    v[1] = (mask >> 1) & 1;
    v[2] = (mask >> 2) & 1;
    v[3] = (mask >> 3) & 1;
    out.push_back(Vec2Config(v, /*Lx=*/2, /*Ly=*/2));
  }
  return out;
}

std::string TypeSuffixFromTenElem() {
  if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) return "_double";
  if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Complex>) return "_complex";
  return "_unknown";
}

}  // namespace

TEST(TFIMPBC_TRG_ExactSum, AdaGradConverges2x2) {
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::TrivialRepQN;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using SITPS = SplitIndexTPS<TenElemT, QNT>;

  constexpr size_t Ly = 2;
  constexpr size_t Lx = 2;

  // Model parameters.
  // NOTE: For 2x2 PBC, the "right+down per site" convention double-counts bonds.
  // We follow the user's decision and use the reference:
  //   E_exact(PBC, 2x2, J, h) = E_exact(OBC, 2x2, 2J, h).
  const double J = 1.0;
  const double h = 1.0;
  const double energy_exact = Calculate2x2OBCTransverseIsingEnergy(/*J=*/2.0 * J, /*h=*/h);

  // Load initial TPS from pre-generated PBC simple-update data.
  const std::string data_path =
      std::string(TEST_SOURCE_DIR) + "/test_data/" +
      "transverse_ising_tps_pbc" + TypeSuffixFromTenElem() + "_from_simple_update";
  SITPS sitps(Ly, Lx, /*phy_dim=*/2, BoundaryCondition::Periodic);
  ASSERT_TRUE(sitps.Load(data_path)) << "Failed to load TPS data from: " << data_path;

  // Exact summation configurations (2^4).
  const auto all_configs = GenerateAllConfigs2x2Binary();

  // TRG truncation params (required by TPSWaveFunctionComponent; 2x2 is exact anyway).
  const TRGTruncateParams<RealT> trunc_para(/*d_min=*/1, /*d_max=*/4, /*trunc_error=*/0.0);

  TransverseFieldIsingSquarePBC model(h);

  // Energy evaluator for Optimizer: returns (E, grad, err).
  auto energy_evaluator = [&](const SITPS& state) -> std::tuple<TenElemT, SITPS, double> {
    return ExactSumEnergyEvaluatorMPI<TransverseFieldIsingSquarePBC, TenElemT, QNT, TRGContractor>(
        state, all_configs, trunc_para, model, Ly, Lx, MPI_COMM_WORLD, /*rank=*/0, /*mpi_size=*/1);
  };

  // Evaluate initial energy.
  const auto [e0, g0, err0] = energy_evaluator(sitps);
  (void)g0;
  EXPECT_NEAR(err0, 0.0, 0.0);

  // Optimizer params (first-order, no SR).
  // Conservative settings; exact gradients => stable convergence.
  OptimizerParams::BaseParams base_params(
      /*max_iter=*/250,
      /*energy_tol=*/1e-14,
      /*grad_tol=*/1e-12,
      /*patience=*/80,
      /*learning_rate=*/0.05);
  AdaGradParams adagrad_params(/*epsilon=*/1e-10, /*initial_accumulator_value=*/0.0);
  OptimizerParams opt_params(base_params, adagrad_params);

  Optimizer<TenElemT, QNT> optimizer(opt_params, MPI_COMM_WORLD, /*rank=*/0, /*mpi_size=*/1);
  auto result = optimizer.IterativeOptimize(sitps, energy_evaluator);

  const double e_init = std::real(e0);
  const double e_final = std::real(result.final_energy);

  // Must strictly improve vs the SU initialization (regression guard).
  EXPECT_LT(e_final, e_init);

  // Should converge tightly to the chosen exact reference (2J convention).
  EXPECT_NEAR(e_final, energy_exact, 1e-6);
}


