// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-11
*
* Description: Regression golden test for fermionic exact-summation + optimizer path.
*              This test intentionally uses a tiny 2x2 system and fixed input TPS.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"

#include <filesystem>
#include <complex>

#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

namespace {

template<typename SITPST, typename TenElemT>
std::complex<double> WeightedProbeInnerProduct(const SITPST &x) {
  SITPST probe = x;
  for (size_t row = 0; row < probe.rows(); ++row) {
    for (size_t col = 0; col < probe.cols(); ++col) {
      const size_t phy_dim = probe({row, col}).size();
      for (size_t i = 0; i < phy_dim; ++i) {
        auto &ten = probe({row, col})[i];
        if (ten.IsDefault()) { continue; }
        const double base = 0.01 * static_cast<double>((row + 1) * 13 + (col + 1) * 7 + (i + 1) * 3);
        if constexpr (std::is_same_v<TenElemT, QLTEN_Complex>) {
          const double imag = 0.003 * static_cast<double>((row + 1) + 2 * (i + 1));
          ten *= TenElemT(base, imag);
        } else {
          ten *= TenElemT(base);
        }
      }
    }
  }
  TenElemT ip = x * probe;
  return {std::real(ip), std::imag(ip)};
}

template<typename SITPST, typename TenElemT, typename QNT>
std::complex<double> RandomProbeInnerProduct(const SITPST &x, unsigned int seed) {
  SITPST probe = x;
  qlten::SetRandomSeed(seed);
  for (size_t row = 0; row < probe.rows(); ++row) {
    for (size_t col = 0; col < probe.cols(); ++col) {
      const size_t phy_dim = probe({row, col}).size();
      for (size_t i = 0; i < phy_dim; ++i) {
        auto &ten = probe({row, col})[i];
        if (ten.IsDefault()) { continue; }
        ten.Random(QNT(0));
      }
    }
  }
  TenElemT ip = x * probe;
  return {std::real(ip), std::imag(ip)};
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> LoadSpinlessFermionTPS2x2() {
  constexpr size_t kLy = 2;
  constexpr size_t kLx = 2;
  std::string type_suffix;
  if constexpr (std::is_same_v<TenElemT, QLTEN_Double>) {
    type_suffix = "_double_from_simple_update";
  } else {
    type_suffix = "_complex_from_simple_update";
  }
  const std::filesystem::path tps_path =
      std::filesystem::path(TEST_SOURCE_DIR) /
      ("test_data/spinless_fermion_tps_t2_0.000000" + type_suffix);
  SplitIndexTPS<TenElemT, QNT> sitps(kLy, kLx);
  EXPECT_TRUE(sitps.Load(tps_path.string()));
  return sitps;
}

template<typename TenElemT>
void PrintCurrentValues(const char *label,
                        const TenElemT &energy,
                        double grad_norm,
                        const std::complex<double> &grad_probe,
                        const std::complex<double> &grad_probe_rand1,
                        const std::complex<double> &grad_probe_rand2) {
  std::cout << "[GOLDEN-CANDIDATE][" << label << "] "
            << "energy=(" << std::real(energy) << "," << std::imag(energy) << ") "
            << "grad_norm=" << grad_norm << " "
            << "grad_probe=(" << grad_probe.real() << "," << grad_probe.imag() << ") "
            << "grad_probe_rand1=(" << grad_probe_rand1.real() << "," << grad_probe_rand1.imag() << ") "
            << "grad_probe_rand2=(" << grad_probe_rand2.real() << "," << grad_probe_rand2.imag() << ")"
            << std::endl;
}

} // namespace

struct FermionExactOptimizerIntegrationGoldenTest : MPITest {
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::fZ2QN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  static constexpr size_t kLy = 2;
  static constexpr size_t kLx = 2;
  static constexpr double kT = 1.0;
  static constexpr double kT2 = 0.0;
  static constexpr double kV = 0.0;
};

TEST_F(FermionExactOptimizerIntegrationGoldenTest, ExactSumOptimizerOneStepGolden) {
  SITPST sitps = LoadSpinlessFermionTPS2x2<TenElemT, QNT>();
  SquareSpinlessFermion model(kT, kT2, kV);

  auto trun_para = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(/*D_min=*/4, /*D_max=*/16, /*trunc_error=*/0.0);
  const std::vector<Configuration> all_configs = GenerateAllPermutationConfigs({2, 2}, kLx, kLy);

  OptimizerParams::BaseParams base_params(/*max_iterations=*/1,
                                          /*energy_tolerance=*/1e-30,
                                          /*gradient_tolerance=*/1e-30,
                                          /*plateau_patience=*/1,
                                          /*learning_rate=*/0.0);
  SGDParams sgd_params(/*momentum=*/0.0, /*nesterov=*/false);
  OptimizerParams opt_params(base_params, sgd_params);
  Optimizer<TenElemT, QNT> optimizer(opt_params, comm, rank, mpi_size);

  TenElemT captured_energy = TenElemT(0.0);
  SITPST captured_grad;
  bool captured_once = false;
  auto energy_evaluator = [&](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    auto [energy, gradient, error] = ExactSumEnergyEvaluatorMPI<SquareSpinlessFermion, TenElemT, QNT>(
        state, all_configs, trun_para, model, kLy, kLx, comm, rank, mpi_size);
    captured_energy = energy;
    captured_grad = gradient;
    captured_once = true;
    return {energy, gradient, error};
  };

  auto result = optimizer.IterativeOptimize(sitps, energy_evaluator);
  ASSERT_TRUE(captured_once);
  if (rank != qlten::hp_numeric::kMPIMasterRank) { return; }

  const double grad_norm = captured_grad.NormSquare();
  const std::complex<double> grad_probe = WeightedProbeInnerProduct<SITPST, TenElemT>(captured_grad);
  const std::complex<double> grad_probe_rand1 =
      RandomProbeInnerProduct<SITPST, TenElemT, QNT>(captured_grad, /*seed=*/1337U);
  const std::complex<double> grad_probe_rand2 =
      RandomProbeInnerProduct<SITPST, TenElemT, QNT>(captured_grad, /*seed=*/424242U);
  constexpr bool kPrintGolden = false;
  if (kPrintGolden) {
    PrintCurrentValues("exact+optimizer",
                       captured_energy,
                       grad_norm,
                       grad_probe,
                       grad_probe_rand1,
                       grad_probe_rand2);
  }

  // Golden values are captured from the current implementation before refactor.
  if constexpr (std::is_same_v<TenElemT, QLTEN_Double>) {
    constexpr double kEnergyReal = -1.9821805385446198;
    constexpr double kEnergyImag = 0.0;
    constexpr double kGradNorm = 0.064616411237561233;
    constexpr double kProbeReal = 0.021610150065783196;
    constexpr double kProbeImag = 0.0;
    constexpr double kProbeRand1Real = 0.15760675624240009;
    constexpr double kProbeRand1Imag = 0.0;
    constexpr double kProbeRand2Real = 0.23724316019053188;
    constexpr double kProbeRand2Imag = 0.0;
    EXPECT_NEAR(std::real(captured_energy), kEnergyReal, 1e-10);
    EXPECT_NEAR(std::imag(captured_energy), kEnergyImag, 1e-12);
    EXPECT_NEAR(grad_norm, kGradNorm, 1e-10);
    EXPECT_NEAR(grad_probe.real(), kProbeReal, 1e-10);
    EXPECT_NEAR(grad_probe.imag(), kProbeImag, 1e-12);
    EXPECT_NEAR(grad_probe_rand1.real(), kProbeRand1Real, 1e-10);
    EXPECT_NEAR(grad_probe_rand1.imag(), kProbeRand1Imag, 1e-10);
    EXPECT_NEAR(grad_probe_rand2.real(), kProbeRand2Real, 1e-10);
    EXPECT_NEAR(grad_probe_rand2.imag(), kProbeRand2Imag, 1e-10);
  } else {
    constexpr double kEnergyReal = -1.9821805385446198;
    constexpr double kEnergyImag = 0.0;
    constexpr double kGradNorm = 0.064616411237560373;
    constexpr double kProbeReal = 0.021610150065782957;
    constexpr double kProbeImag = 0.0008416615629266691;
    constexpr double kProbeRand1Real = 0.37887758822255185;
    constexpr double kProbeRand1Imag = -0.12079103432562188;
    constexpr double kProbeRand2Real = 0.33949123826991584;
    constexpr double kProbeRand2Imag = -0.11642875645925056;
    EXPECT_NEAR(std::real(captured_energy), kEnergyReal, 1e-10);
    EXPECT_NEAR(std::imag(captured_energy), kEnergyImag, 1e-12);
    EXPECT_NEAR(grad_norm, kGradNorm, 1e-10);
    EXPECT_NEAR(grad_probe.real(), kProbeReal, 1e-10);
    EXPECT_NEAR(grad_probe.imag(), kProbeImag, 1e-10);
    EXPECT_NEAR(grad_probe_rand1.real(), kProbeRand1Real, 1e-10);
    EXPECT_NEAR(grad_probe_rand1.imag(), kProbeRand1Imag, 1e-10);
    EXPECT_NEAR(grad_probe_rand2.real(), kProbeRand2Real, 1e-10);
    EXPECT_NEAR(grad_probe_rand2.imag(), kProbeRand2Imag, 1e-10);
  }

  // Keep optimizer path in assertion scope.
  EXPECT_TRUE(std::isfinite(std::real(result.final_energy)));
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
