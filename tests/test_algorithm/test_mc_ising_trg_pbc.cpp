// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Deterministic Monte Carlo regression test (classical Ising via TRG amplitude):
 *
 * Key identity:
 *   Build an "Ising TPS" where the physical spin is NOT summed, using the standard
 *   Z2 bond decomposition with coupling Kpsi. Then contracting virtual bonds gives
 *     psi(s) ∝ exp(Kpsi * Σ_{<ij>} s_i s_j).
 *   Therefore |psi(s)|^2 samples the classical Ising Boltzmann weight at coupling
 *     K = 2*Kpsi.
 *
 * This test avoids long Monte Carlo runs:
 * - Use a fixed proposal sequence (bond list) and a fixed u-sequence for Metropolis.
 * - Verify *pathwise* agreement between:
 *     (A) classical acceptance from Δ(Σ s_i s_j) at coupling K
 *     (B) TRG amplitude ratio from psi(s) built at Kpsi=K/2.
 * - After each move, configurations must match exactly.
 */

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/trg_contractor.h"
#include "qlpeps/vmc_basic/wave_function_component.h"

using namespace qlten;
using namespace qlpeps;

using TenElemT = TEN_ELEM_TYPE;
using QNT = qlten::special_qn::TrivialRepQN;

namespace {

inline int SpinFromConfig(size_t cfg) { return (cfg == 0) ? +1 : -1; }

int SumNN_PBC(const Configuration& cfg) {
  const size_t n = cfg.rows();
  const size_t m = cfg.cols();
  int sum = 0;
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < m; ++c) {
      const int s = SpinFromConfig(cfg({r, c}));
      const int sr = SpinFromConfig(cfg({r, (c + 1) % m}));
      const int sd = SpinFromConfig(cfg({(r + 1) % n, c}));
      sum += s * sr;
      sum += s * sd;
    }
  }
  return sum;
}

SplitIndexTPS<TenElemT, QNT> BuildIsingSplitTPS_PBC(size_t n, double Kpsi) {
  // Bond variable p in {0,1} with dim=2.
  // Decomposition:
  //   exp(K s s') = cosh(K) + s s' sinh(K) = sum_{p=0,1} lambda_p v_p(s) v_p(s')
  // where lambda_0=cosh(K), lambda_1=sinh(K), v_0(s)=1, v_1(s)=s.
  // We set w(s,p)=sqrt(lambda_p)*v_p(s).
  //
  // Ising TPS (physical spin not summed):
  //   A_s(l,d,r,u) = prod_dir w_dir(s, p_dir)
  //
  // Then contracting virtual bonds gives:
  //   psi(s) = prod_{<ij>} exp(Kpsi * s_i s_j).
  // Therefore |psi|^2 corresponds to classical Ising at coupling K=2*Kpsi.

  using IndexT = qlten::Index<QNT>;
  using QNSctT = qlten::QNSector<QNT>;

  constexpr size_t D = 2;
  const IndexT virt_out({QNSctT(QNT(), D)}, TenIndexDirType::OUT);

  SplitIndexTPS<TenElemT, QNT> sitps(n, n, /*phy_dim=*/2, BoundaryCondition::Periodic);

  const double l0 = std::cosh(Kpsi);
  const double l1 = std::sinh(Kpsi);
  const double sl0 = std::sqrt(l0);
  const double sl1 = std::sqrt(std::max(0.0, l1));

  auto w = [&](int s, size_t p) -> double {
    if (p == 0) return sl0;
    // p==1
    return sl1 * double(s);
  };

  // Define directed edge indices (PBC-consistent).
  std::vector<std::vector<IndexT>> hx_out(n, std::vector<IndexT>(n, virt_out));  // (r,c)->(r,c+1)
  std::vector<std::vector<IndexT>> vy_out(n, std::vector<IndexT>(n, virt_out));  // (r,c)->(r+1,c)

  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const IndexT idx_l = InverseIndex(hx_out[r][(c + n - 1) % n]);
      const IndexT idx_r = hx_out[r][c];
      const IndexT idx_u = InverseIndex(vy_out[(r + n - 1) % n][c]);
      const IndexT idx_d = vy_out[r][c];

      for (size_t phys = 0; phys < 2; ++phys) {
        const int s = (phys == 0) ? +1 : -1;
        qlten::QLTensor<TenElemT, QNT> T({idx_l, idx_d, idx_r, idx_u});
        for (size_t l = 0; l < D; ++l) {
          for (size_t d = 0; d < D; ++d) {
            for (size_t rr = 0; rr < D; ++rr) {
              for (size_t u = 0; u < D; ++u) {
                const double val = w(s, l) * w(s, d) * w(s, rr) * w(s, u);
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

}  // namespace

TEST(MCIsingTRG, PathwiseAgreementWithClassicalMetropolis_4x4_PBC) {
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;

  constexpr size_t n = 4;
  const double K = 0.3;         // classical coupling for |psi|^2
  const double Kpsi = 0.5 * K;  // build psi so that |psi|^2 samples classical at K

  // TPS + TRG wavefunction component.
  const auto sitps = BuildIsingSplitTPS_PBC(n, Kpsi);

  // Deterministic initial configuration (checkerboard).
  std::vector<std::vector<size_t>> init(n, std::vector<size_t>(n, 0));
  for (size_t r = 0; r < n; ++r) for (size_t c = 0; c < n; ++c) init[r][c] = (r + c) & 1U;
  Configuration cfg_classical(init);

  // TRG truncation: keep exact for D=2.
  const auto trunc = BMPSTruncateParams<RealT>::SVD(/*d_min=*/1, /*d_max=*/4, /*trunc_error=*/0.0);
  TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, qlpeps::TRGContractor> comp(sitps, cfg_classical, trunc);

  // Fixed proposal set: all PBC NN bonds (each site contributes right + down).
  std::vector<std::pair<SiteIdx, SiteIdx>> bonds;
  bonds.reserve(n * n * 2);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      bonds.push_back({SiteIdx{r, c}, SiteIdx{r, (c + 1) % n}});
      bonds.push_back({SiteIdx{r, c}, SiteIdx{(r + 1) % n, c}});
    }
  }

  // Deterministic RNG streams: one for picking bonds, one for acceptance u.
  std::mt19937 gen_bond(20251214);
  std::mt19937 gen_u(20251215);
  std::uniform_int_distribution<size_t> pick(0, bonds.size() - 1);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  constexpr int steps = 20;
  for (int step = 0; step < steps; ++step) {
    const auto& b = bonds[pick(gen_bond)];
    const SiteIdx s1 = b.first;
    const SiteIdx s2 = b.second;
    const double u = uni(gen_u);

    const size_t c1 = cfg_classical(s1);
    const size_t c2 = cfg_classical(s2);

    // If same spin, exchange does nothing; both chains should no-op.
    if (c1 == c2) {
      EXPECT_TRUE(static_cast<const qlpeps::DuoMatrix<size_t>&>(comp.config) ==
                  static_cast<const qlpeps::DuoMatrix<size_t>&>(cfg_classical));
      continue;
    }

    const int sum_old = SumNN_PBC(cfg_classical);
    Configuration cfg_new = cfg_classical;
    std::swap(cfg_new(s1), cfg_new(s2));
    const int sum_new = SumNN_PBC(cfg_new);
    const double ratio_sq_classical = std::exp(K * double(sum_new - sum_old));
    const bool accept_classical = (ratio_sq_classical >= 1.0) || (u < ratio_sq_classical);

    // TRG trial: swap two physical states => replace two site tensors.
    using TensorT = typename TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, qlpeps::TRGContractor>::Tensor;
    std::vector<std::pair<SiteIdx, TensorT>> repl{
        {s1, sitps(s1)[c2]},
        {s2, sitps(s2)[c1]},
    };
    std::vector<std::pair<SiteIdx, size_t>> new_cfgs{
        {s1, c2},
        {s2, c1},
    };
    const auto psi_a = comp.GetAmplitude();
    const auto psi_b = comp.BeginTrial(repl, new_cfgs);
    const double ratio_sq_trg = std::pow(std::abs(psi_b) / std::max(1e-300, std::abs(psi_a)), 2);

    EXPECT_NEAR(ratio_sq_trg, ratio_sq_classical, 5e-10 * std::max(1.0, std::abs(ratio_sq_classical)));
    const bool accept_trg = (ratio_sq_trg >= 1.0) || (u < ratio_sq_trg);
    EXPECT_EQ(accept_trg, accept_classical);

    if (accept_trg) {
      comp.AcceptTrial(sitps);
      cfg_classical = cfg_new;
    } else {
      comp.RejectTrial();
    }
    EXPECT_TRUE(static_cast<const qlpeps::DuoMatrix<size_t>&>(comp.config) ==
                static_cast<const qlpeps::DuoMatrix<size_t>&>(cfg_classical));
  }
}


