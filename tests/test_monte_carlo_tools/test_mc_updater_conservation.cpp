// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Description: MC updater conservation and smoke tests.
 * Verifies that MC configuration updaters maintain conserved quantum numbers
 * (particle number for exchange updater) and produce valid configurations.
 */

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/vmc_basic/wave_function_component.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

using namespace qlten;
using namespace qlpeps;

using TenElemT = TEN_ELEM_TYPE;
using QNT = qlten::special_qn::TrivialRepQN;

namespace {

/// Build a synthetic OBC SplitIndexTPS for a 2x2 lattice with phy_dim=2.
/// Deterministic tensor values; no file I/O.
SplitIndexTPS<TenElemT, QNT> BuildSyntheticOBCSitps() {
  using IndexT = qlten::Index<QNT>;
  using QNSctT = qlten::QNSector<QNT>;

  constexpr size_t n = 2;
  constexpr size_t D = 2;

  SplitIndexTPS<TenElemT, QNT> sitps(n, n, /*phy_dim=*/2, BoundaryCondition::Open);

  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      // OBC: boundary edges have dim 1, interior edges have dim D
      const size_t dim_l = (c == 0)     ? 1 : D;
      const size_t dim_r = (c == n - 1) ? 1 : D;
      const size_t dim_u = (r == 0)     ? 1 : D;
      const size_t dim_d = (r == n - 1) ? 1 : D;

      const IndexT idx_l({QNSctT(QNT(), dim_l)}, TenIndexDirType::IN);
      const IndexT idx_r({QNSctT(QNT(), dim_r)}, TenIndexDirType::OUT);
      const IndexT idx_u({QNSctT(QNT(), dim_u)}, TenIndexDirType::IN);
      const IndexT idx_d({QNSctT(QNT(), dim_d)}, TenIndexDirType::OUT);

      for (size_t phys = 0; phys < 2; ++phys) {
        qlten::QLTensor<TenElemT, QNT> T({idx_l, idx_d, idx_r, idx_u});
        // Dense fill with deterministic, distinct values per site/phys
        for (size_t l = 0; l < dim_l; ++l) {
          for (size_t d = 0; d < dim_d; ++d) {
            for (size_t rr = 0; rr < dim_r; ++rr) {
              for (size_t u = 0; u < dim_u; ++u) {
                const double base = 0.1 + 0.02 * double((r + 1) * 7 + (c + 1) * 3 + (phys + 1) * 5);
                const double val = base * (1.0 + 0.1 * double(l) + 0.2 * double(d)
                                               + 0.3 * double(rr) + 0.4 * double(u));
                T({l, d, rr, u}) = TenElemT(val);
              }
            }
          }
        }
        sitps({r, c})[phys] = std::move(T);
      }
    }
  }
  return sitps;
}

} // namespace

// Tests that MCUpdateSquareNNExchangeOBC preserves state counts (particle number).
TEST(MCUpdaterConservation, NNExchangeOBC_PreservesStateCounts) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;

  auto sitps = BuildSyntheticOBCSitps();

  // Half-filling initial configuration: 2 zeros, 2 ones
  Configuration config({{0, 1}, {1, 0}});
  auto initial_counts = config.CountOccupancy();

  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 2, 0.0);
  TPSWaveFunctionComponent<TenElemT, QNT> comp(sitps, config, trun_para);
  ASSERT_TRUE(comp.IsAmplitudeSquareLegal());

  MCUpdateSquareNNExchangeOBC updater(/*seed=*/42U);
  std::vector<double> accept_rates;

  // Run 20 sweeps and check conservation after each
  for (int sweep = 0; sweep < 20; ++sweep) {
    updater(sitps, comp, accept_rates);
    auto current_counts = comp.config.CountOccupancy();
    EXPECT_EQ(current_counts, initial_counts)
        << "State counts changed after sweep " << sweep;
    EXPECT_TRUE(comp.IsAmplitudeSquareLegal())
        << "Amplitude invalid after sweep " << sweep;
  }
}

// Tests that MCUpdateSquareNNFullSpaceUpdateOBC keeps amplitude finite.
TEST(MCUpdaterConservation, NNFullSpaceOBC_AmplitudeStaysFinite) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;

  auto sitps = BuildSyntheticOBCSitps();

  // Fixed initial configuration
  Configuration config({{0, 0}, {1, 1}});

  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 2, 0.0);
  TPSWaveFunctionComponent<TenElemT, QNT> comp(sitps, config, trun_para);
  ASSERT_TRUE(comp.IsAmplitudeSquareLegal());

  MCUpdateSquareNNFullSpaceUpdateOBC updater(/*seed=*/42U);
  std::vector<double> accept_rates;

  for (int sweep = 0; sweep < 10; ++sweep) {
    updater(sitps, comp, accept_rates);
    ASSERT_EQ(accept_rates.size(), 1U);
    EXPECT_GE(accept_rates[0], 0.0);
    EXPECT_LE(accept_rates[0], 1.0);
    EXPECT_TRUE(comp.IsAmplitudeSquareLegal())
        << "Amplitude invalid after sweep " << sweep;
    EXPECT_TRUE(std::isfinite(double(std::abs(comp.GetAmplitude()))))
        << "Amplitude not finite after sweep " << sweep;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
