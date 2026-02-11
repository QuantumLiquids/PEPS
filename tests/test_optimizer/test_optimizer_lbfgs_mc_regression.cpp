// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-12
*
* Description: MC regression tests for L-BFGS fixed-step mode.
*              Covers one bosonic and one fermionic 2x2 OBC system.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/api/vmc_api.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

namespace {

template<typename TenElemT>
std::string LowestSuffix() {
  if constexpr (std::is_same_v<TenElemT, QLTEN_Double>) {
    return "_doublelowest";
  } else if constexpr (std::is_same_v<TenElemT, QLTEN_Complex>) {
    return "_complexlowest";
  }
  throw std::runtime_error("Unsupported TEN_ELEM_TYPE for LowestSuffix.");
}

template<typename TenElemT>
std::string FromSimpleUpdateSuffix() {
  if constexpr (std::is_same_v<TenElemT, QLTEN_Double>) {
    return "_double_from_simple_update";
  } else if constexpr (std::is_same_v<TenElemT, QLTEN_Complex>) {
    return "_complex_from_simple_update";
  }
  throw std::runtime_error("Unsupported TEN_ELEM_TYPE for FromSimpleUpdateSuffix.");
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> LoadTPSBroadcast(const std::string &path, const MPI_Comm &comm, int rank, int mpi_size) {
  SplitIndexTPS<TenElemT, QNT> sitps(2, 2);
  if (rank == hp_numeric::kMPIMasterRank) {
    const bool ok = sitps.Load(path);
    if (!ok) {
      throw std::runtime_error("Failed to load TPS from: " + path);
    }
  }
  if (mpi_size > 1) {
    qlpeps::MPI_Bcast(sitps, comm, hp_numeric::kMPIMasterRank);
  }
  return sitps;
}

inline OptimizerParams CreateFixedStepLBFGSParams(size_t max_iterations, double learning_rate) {
  LBFGSParams lbfgs(/*hist=*/8,
                    /*tol_grad=*/1e-8,
                    /*tol_change=*/1e-12,
                    /*max_eval=*/20,
                    /*step_mode=*/LBFGSStepMode::kFixed,
                    /*wolfe_c1=*/1e-4,
                    /*wolfe_c2=*/0.9,
                    /*min_step=*/1e-8,
                    /*max_step=*/1.0,
                    /*min_curvature=*/1e-12,
                    /*use_damping=*/true,
                    /*max_direction_norm=*/1e3,
                    /*allow_fallback_to_fixed_step=*/false,
                    /*fallback_fixed_step_scale=*/0.2);
  return OptimizerFactory::CreateLBFGSAdvanced(
      max_iterations, /*energy_tolerance=*/0.0, /*gradient_tolerance=*/0.0,
      /*plateau_patience=*/max_iterations, learning_rate, lbfgs);
}

template<typename TenElemT>
void AssertMCRegressionSanity(const VmcOptimizeResult<TenElemT, qlten::special_qn::TrivialRepQN> &res) {
  ASSERT_FALSE(res.energy_trajectory.empty());
  ASSERT_FALSE(res.gradient_norms.empty());
  const auto it_min = std::min_element(res.energy_trajectory.begin(), res.energy_trajectory.end(),
                                       [](const TenElemT &a, const TenElemT &b) {
                                         return std::real(a) < std::real(b);
                                       });
  const double traj_min = std::real(*it_min);
  EXPECT_TRUE(std::isfinite(res.min_energy));
  EXPECT_TRUE(std::isfinite(traj_min));
  EXPECT_NEAR(res.min_energy, traj_min, 1e-12);
  for (const auto &e : res.energy_trajectory) {
    EXPECT_TRUE(std::isfinite(std::real(e)));
  }
  for (double gnorm : res.gradient_norms) {
    EXPECT_TRUE(std::isfinite(gnorm));
  }
}

template<typename TenElemT>
void AssertMCRegressionSanity(const VmcOptimizeResult<TenElemT, qlten::special_qn::fZ2QN> &res) {
  ASSERT_FALSE(res.energy_trajectory.empty());
  ASSERT_FALSE(res.gradient_norms.empty());
  const auto it_min = std::min_element(res.energy_trajectory.begin(), res.energy_trajectory.end(),
                                       [](const TenElemT &a, const TenElemT &b) {
                                         return std::real(a) < std::real(b);
                                       });
  const double traj_min = std::real(*it_min);
  EXPECT_TRUE(std::isfinite(res.min_energy));
  EXPECT_TRUE(std::isfinite(traj_min));
  EXPECT_NEAR(res.min_energy, traj_min, 1e-12);
  for (const auto &e : res.energy_trajectory) {
    EXPECT_TRUE(std::isfinite(std::real(e)));
  }
  for (double gnorm : res.gradient_norms) {
    EXPECT_TRUE(std::isfinite(gnorm));
  }
}

}  // namespace

struct LBFGSMCBosonRegression : MPITest {
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  void SetUp() override {
    MPITest::SetUp();
    hp_numeric::SetTensorManipulationThreads(1);
  }
};

TEST_F(LBFGSMCBosonRegression, Heisenberg2x2FixedStepStable) {
  using Model = SquareSpinOneHalfXXZModelOBC;
  using Updater = MCUpdateSquareNNExchange;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::TrivialRepQN;

  const std::string tps_path =
      std::string(TEST_SOURCE_DIR) + "/test_data/heisenberg_tps" + LowestSuffix<TenElemT>();
  auto sitps = LoadTPSBroadcast<TenElemT, QNT>(tps_path, comm, rank, mpi_size);

  Configuration init_config(2, 2);
  init_config({0, 0}) = 0;
  init_config({0, 1}) = 1;
  init_config({1, 0}) = 1;
  init_config({1, 1}) = 0;

  OptimizerParams opt_params = CreateFixedStepLBFGSParams(/*max_iterations=*/5, /*learning_rate=*/0.03);
  MonteCarloParams mc_params(/*samples=*/300, /*warmup_sweeps=*/30, /*sweeps_between=*/1, init_config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>(
      /*D_min=*/4, /*D_max=*/8, /*trunc_err=*/1e-15, CompressMPSScheme::SVD_COMPRESS,
      std::make_optional<double>(1e-14), std::make_optional<size_t>(10)));
  VMCPEPSOptimizerParams vmc_params(opt_params, mc_params, peps_params);

  Model model(/*Jx=*/1.0, /*Jz=*/1.0, /*h=*/0.0);
  auto result = VmcOptimize<TenElemT, QNT, Updater, Model>(vmc_params, sitps, comm, model, Updater{});

  if (rank == hp_numeric::kMPIMasterRank) {
    AssertMCRegressionSanity(result);
  }
}

struct LBFGSMCFermionRegression : MPITest {
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::fZ2QN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  void SetUp() override {
    MPITest::SetUp();
    hp_numeric::SetTensorManipulationThreads(1);
  }
};

TEST_F(LBFGSMCFermionRegression, SpinlessFermion2x2FixedStepStable) {
  using Model = SquareSpinlessFermion;
  using Updater = MCUpdateSquareNNExchange;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::fZ2QN;

  const std::string tps_path = std::string(TEST_SOURCE_DIR) +
      "/test_data/spinless_fermion_tps_t2_0.000000" + FromSimpleUpdateSuffix<TenElemT>();
  auto sitps = LoadTPSBroadcast<TenElemT, QNT>(tps_path, comm, rank, mpi_size);

  Configuration init_config(2, 2);
  init_config({0, 0}) = 1;
  init_config({0, 1}) = 0;
  init_config({1, 0}) = 0;
  init_config({1, 1}) = 1;

  OptimizerParams opt_params = CreateFixedStepLBFGSParams(/*max_iterations=*/5, /*learning_rate=*/0.02);
  MonteCarloParams mc_params(/*samples=*/300, /*warmup_sweeps=*/30, /*sweeps_between=*/1, init_config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>(
      /*D_min=*/4, /*D_max=*/8, /*trunc_err=*/1e-15, CompressMPSScheme::SVD_COMPRESS,
      std::make_optional<double>(1e-14), std::make_optional<size_t>(10)));
  VMCPEPSOptimizerParams vmc_params(opt_params, mc_params, peps_params);

  Model model(/*t=*/1.0, /*t2=*/0.0, /*V=*/0.0);
  auto result = VmcOptimize<TenElemT, QNT, Updater, Model>(vmc_params, sitps, comm, model, Updater{});

  if (rank == hp_numeric::kMPIMasterRank) {
    AssertMCRegressionSanity(result);
  }
}

int main(int argc, char *argv[]) {
  testing::AddGlobalTestEnvironment(new MPIEnvironment);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  return RUN_ALL_TESTS();
}
