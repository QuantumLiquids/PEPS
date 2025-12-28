// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Description: Smoke test for VMC NN updater + TRGContractor (PBC).
 *
 * This is intentionally NOT a physics validation test. It only checks:
 * - the trial/accept/reject workflow does not throw;
 * - amplitude stays finite;
 * - acceptance rate is within [0, 1].
 */

#include "gtest/gtest.h"

#include <cmath>
#include <vector>

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/trg_contractor.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

using namespace qlten;
using namespace qlpeps;

using TenElemT = TEN_ELEM_TYPE;
using QNT = qlten::special_qn::TrivialRepQN;

TEST(MCUpdaterTRGSmoke, SquareNNExchangePBC_2x2) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using IndexT = qlten::Index<QNT>;
  using QNSctT = qlten::QNSector<QNT>;

  constexpr size_t n = 2;  // keep tiny: this is a smoke test; TRGContractor requires power-of-two square

  // Build a synthetic, fully-initialized SplitIndexTPS with PBC-consistent virtual indices.
  // This avoids depending on PEPS initialization details and focuses this test on:
  // updater + TPSWaveFunctionComponent + TRGContractor integration.
  SplitIndexTPS<TenElemT, QNT> sitps(n, n, /*phy_dim=*/2, BoundaryCondition::Periodic);

  // TrivialRep: single sector of dimension D.
  constexpr size_t D = 2;
  const IndexT virt_out({QNSctT(QNT(), D)}, TenIndexDirType::OUT);

  // Edge indices (directed "out" on one side, inverse on the other).
  std::vector<std::vector<IndexT>> hx_out(n, std::vector<IndexT>(n, virt_out));  // (r,c) -> (r,c+1)
  std::vector<std::vector<IndexT>> vy_out(n, std::vector<IndexT>(n, virt_out));  // (r,c) -> (r+1,c)

  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const IndexT idx_l = InverseIndex(hx_out[r][(c + n - 1) % n]);
      const IndexT idx_r = hx_out[r][c];
      const IndexT idx_u = InverseIndex(vy_out[(r + n - 1) % n][c]);
      const IndexT idx_d = vy_out[r][c];

      // Two physical components; both non-default.
      for (size_t phys = 0; phys < 2; ++phys) {
        qlten::QLTensor<TenElemT, QNT> T({idx_l, idx_d, idx_r, idx_u});
        // Dense fill with deterministic values.
        for (size_t l = 0; l < D; ++l) {
          for (size_t d = 0; d < D; ++d) {
            for (size_t rr = 0; rr < D; ++rr) {
              for (size_t u = 0; u < D; ++u) {
                const double base = 0.1 + 0.01 * double((r + 1) * 17 + (c + 1) * 31 + (phys + 1) * 13);
                const double val = base * (1.0 + 0.1 * double(l) + 0.2 * double(d) + 0.3 * double(rr) + 0.4 * double(u));
                T({l, d, rr, u}) = TenElemT(val);
              }
            }
          }
        }
        sitps({r, c})[phys] = std::move(T);
      }
    }
  }

  // Random initial configuration (dim=2).
  Configuration config(n, n, /*dim=*/2);

  // TRG-specific truncation params.
  const TRGTruncateParams<RealT> trunc(/*d_min=*/1, /*d_max=*/16, /*trunc_error=*/0.0);

  // TRG contractor component.
  TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, qlpeps::TRGContractor> comp(sitps, config, trunc);
  ASSERT_TRUE(comp.IsAmplitudeSquareLegal());

  MCUpdateSquareNNExchangePBC updater;

  std::vector<double> accept_rates;
  for (int sweep = 0; sweep < 1; ++sweep) {
    EXPECT_NO_THROW(updater(sitps, comp, accept_rates));
    ASSERT_EQ(accept_rates.size(), 1U);
    EXPECT_GE(accept_rates[0], 0.0);
    EXPECT_LE(accept_rates[0], 1.0);
    EXPECT_TRUE(comp.IsAmplitudeSquareLegal());
    EXPECT_TRUE(std::isfinite(double(std::abs(comp.GetAmplitude()))));
  }
}


