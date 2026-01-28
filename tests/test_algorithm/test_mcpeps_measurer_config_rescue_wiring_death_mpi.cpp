// SPDX-License-Identifier: LGPL-3.0-only

/*
 * Purpose: MPI “death test” at the ctest level.
 *
 * This test verifies that MCPEPSMeasurer forwards MCMeasurementParams::runtime_params.config_rescue
 * into MonteCarloEngine. When rescue is disabled and one rank has an invalid initial configuration,
 * MonteCarloEngine must call MPI_Abort during initialization.
 *
 * The test is registered with ctest property WILL_FAIL, so a non-zero exit code is considered PASS.
 */

#include "gtest/gtest.h"

#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"

#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"

#include "../test_mpi_env.h"

namespace {

using TenElemT = qlten::QLTEN_Double;
using QNT = qlten::special_qn::TrivialRepQN;
using IndexT = qlten::Index<QNT>;
using QNSctT = qlten::QNSector<QNT>;
using SITPST = qlpeps::SplitIndexTPS<TenElemT, QNT>;
using PEPST = qlpeps::SquareLatticePEPS<TenElemT, QNT>;

constexpr size_t kLx = 2;
constexpr size_t kLy = 2;
constexpr size_t kDpeps = 3;  // after direct-sum superposition

struct NoOpSweepUpdater {
  template<typename TElem, typename Qn, template<typename, typename> class ContractorT>
  void operator()(const qlpeps::SplitIndexTPS<TElem, Qn> & /*sitps*/,
                  qlpeps::TPSWaveFunctionComponent<TElem, Qn, qlpeps::NoDress, ContractorT> & /*component*/,
                  std::vector<double> &accept_ratios) {
    accept_ratios = {0.0};
  }
};

struct DummyMeasurementSolver {
  template<typename ElemT>
  struct PsiSummary {
    ElemT psi_mean{};
    double psi_rel_err = 0.0;
  };

  template<typename ElemT, typename Qn>
  qlpeps::ObservableMap<ElemT> EvaluateObservables(
      const qlpeps::SplitIndexTPS<ElemT, Qn> * /*sitps*/,
      qlpeps::TPSWaveFunctionComponent<ElemT, Qn> * /*tps_sample*/) {
    return {};
  }

  template<typename ElemT, typename Qn, typename ComponentT>
  PsiSummary<ElemT> EvaluatePsiSummary(
      const qlpeps::SplitIndexTPS<ElemT, Qn> * /*sitps*/,
      ComponentT * /*tps_sample*/) const {
    return PsiSummary<ElemT>{ElemT(0), 0.0};
  }

  std::vector<qlpeps::ObservableMeta> DescribeObservables(size_t, size_t) const { return {}; }
};

PEPST CreateProductStatePEPS(const IndexT &phy_idx,
                             const std::vector<std::vector<size_t>> &config) {
  PEPST peps(phy_idx, kLy, kLx, qlpeps::BoundaryCondition::Open);
  auto cfg = config;  // Initial() takes non-const reference
  peps.Initial(cfg);
  return peps;
}

SITPST CreateSuperpositionTPS(TenElemT beta = 1e-300) {
  IndexT phy_idx = IndexT({QNSctT(QNT(), 2)}, qlten::OUT);

  std::vector<std::vector<size_t>> neel_config = {{0, 1}, {1, 0}};
  std::vector<std::vector<size_t>> anti_neel_config = {{1, 0}, {0, 1}};
  std::vector<std::vector<size_t>> third_config = {{0, 0}, {1, 1}};

  auto peps_neel = CreateProductStatePEPS(phy_idx, neel_config);
  auto peps_anti_neel = CreateProductStatePEPS(phy_idx, anti_neel_config);
  auto peps_third = CreateProductStatePEPS(phy_idx, third_config);

  auto tps_neel = peps_neel.ToTPS();
  auto tps_anti_neel = peps_anti_neel.ToTPS();
  auto tps_third = peps_third.ToTPS();

  std::vector<decltype(tps_neel)> tps_list = {tps_neel, tps_anti_neel, tps_third};
  std::vector<TenElemT> coeffs = {TenElemT(1.0), beta, TenElemT(0.0)};
  auto tps_sum = qlpeps::WaveFunctionSum(tps_list, coeffs);
  return SITPST::FromTPS(tps_sum);
}

qlpeps::Configuration CreateNeelConfig() {
  qlpeps::Configuration config(kLy, kLx);
  config({0, 0}) = 0;
  config({0, 1}) = 1;
  config({1, 0}) = 1;
  config({1, 1}) = 0;
  return config;
}

qlpeps::Configuration CreateFourthSz0Config() {
  qlpeps::Configuration config(kLy, kLx);
  config({0, 0}) = 1;
  config({0, 1}) = 1;
  config({1, 0}) = 0;
  config({1, 1}) = 0;
  return config;
}

}  // namespace

TEST_F(MPITest, MCPEPSMeasurerRescueDisabledAbortsOnInvalidRank) {
  if (mpi_size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI ranks";
  }

  auto sitps = CreateSuperpositionTPS();

  // Rank 0: valid config; Rank 1: invalid config (outside wavefunction support).
  auto init_cfg = (rank == 0) ? CreateNeelConfig() : CreateFourthSz0Config();

  qlpeps::MonteCarloParams mc_params(/*samples=*/1,
                                    /*warmup_sweeps=*/0,
                                    /*sweeps_between=*/1,
                                    init_cfg,
                                    /*is_warmed_up=*/true);
  qlpeps::PEPSParams peps_params(qlpeps::BMPSTruncateParams<qlten::QLTEN_Double>(
      kDpeps, 2 * kDpeps, 1e-15,
      qlpeps::CompressMPSScheme::SVD_COMPRESS,
      std::make_optional<double>(1e-14),
      std::make_optional<size_t>(10)));

  qlpeps::MCMeasurementParams params(mc_params, peps_params);
  params.runtime_params.config_rescue.enabled = false;  // should force MPI_Abort on invalid rank

  DummyMeasurementSolver solver;

  // Construction should MPI_Abort during MonteCarloEngine initialization.
  // ctest marks this test as WILL_FAIL, so abort => PASS.
  qlpeps::MCPEPSMeasurer<TenElemT, QNT, NoOpSweepUpdater, DummyMeasurementSolver> measurer(
      sitps, params, comm, solver, NoOpSweepUpdater());

  (void)measurer;
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
