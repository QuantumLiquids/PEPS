// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2025-12-12
 *
 * Description: Unit tests for TRGContractor (finite-size PBC TRG).
 */

#include "gtest/gtest.h"

#include <cmath>
#include <array>
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/trg_contractor.h"

using namespace qlten;
using namespace qlpeps;

namespace {
// ---- Shared helpers for Z2-symmetric classical Ising TN on a torus ----

struct Z2IsingTNBuilder {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;
  using IndexT = qlten::Index<QNT>;
  using QNSctT = qlten::QNSector<QNT>;
  using TensorT = qlten::QLTensor<TenElemT, QNT>;

  // Z2 sectors: even(0), odd(1), each dim=1.
  const QNT q_even{0};
  const QNT q_odd{1};
  const QNSctT s_even{q_even, 1};
  const QNSctT s_odd{q_odd, 1};

  // IMPORTANT: match TRGContractor single-layer convention:
  // leg order (l, d, r, u) with directions: l IN, d OUT, r OUT, u IN.
  const IndexT idx_l{{s_even, s_odd}, TenIndexDirType::IN};
  const IndexT idx_d{{s_even, s_odd}, TenIndexDirType::OUT};
  const IndexT idx_r{{s_even, s_odd}, TenIndexDirType::OUT};
  const IndexT idx_u{{s_even, s_odd}, TenIndexDirType::IN};

  static std::array<double, 2> SqrtLambda(double K) {
    // Z2 decomposition for edge weight exp(K s s'):
    // lambda_even = cosh(K), lambda_odd = sinh(K)
    const double l0 = std::cosh(K);
    const double l1 = std::sinh(K);
    return {std::sqrt(l0), std::sqrt(std::max(0.0, l1))};
  }

  TensorT MakeSiteTensorFromSqrtLambda(const std::array<double, 2> &sl_l,
                                       const std::array<double, 2> &sl_d,
                                       const std::array<double, 2> &sl_r,
                                       const std::array<double, 2> &sl_u) const {
    // Z2-symmetric local vertex tensor:
    // T(l,d,r,u) = 2 * sl_l[l] * sl_d[d] * sl_r[r] * sl_u[u] if total parity is even, else 0.
    TensorT T({idx_l, idx_d, idx_r, idx_u});
    // Allocate allowed block(s) for Z2-even divergence.
    T.Fill(q_even, TenElemT(0));
    for (size_t l = 0; l < 2; ++l) {
      for (size_t d = 0; d < 2; ++d) {
        for (size_t r = 0; r < 2; ++r) {
          for (size_t u = 0; u < 2; ++u) {
            if (((l + d + r + u) & 1U) != 0U) continue;
            const double amp = 2.0 * sl_l[l] * sl_d[d] * sl_r[r] * sl_u[u];
            T({l, d, r, u}) = TenElemT(amp);
          }
        }
      }
    }
    return T;
  }
};

template<class KxFn, class KyFn>
qlpeps::TensorNetwork2D<QLTEN_Double, qlten::special_qn::Z2QN>
BuildZ2IsingTorusTN(size_t n, const KxFn &Kx, const KyFn &Ky) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;

  const Z2IsingTNBuilder b;
  qlpeps::TensorNetwork2D<TenElemT, QNT> tn(n, n, qlpeps::BoundaryCondition::Periodic);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const size_t cm1 = (c + n - 1) % n;
      const size_t rm1 = (r + n - 1) % n;
      // Per-edge couplings incident to site (r,c):
      // left uses Kx(r,c-1), right uses Kx(r,c), down uses Ky(r,c), up uses Ky(r-1,c).
      const auto sl_l = Z2IsingTNBuilder::SqrtLambda(Kx(r, cm1));
      const auto sl_r = Z2IsingTNBuilder::SqrtLambda(Kx(r, c));
      const auto sl_d = Z2IsingTNBuilder::SqrtLambda(Ky(r, c));
      const auto sl_u = Z2IsingTNBuilder::SqrtLambda(Ky(rm1, c));
      tn({r, c}) = b.MakeSiteTensorFromSqrtLambda(sl_l, sl_d, sl_r, sl_u);
    }
  }
  return tn;
}

template<class KxFn, class KyFn>
Z2IsingTNBuilder::TensorT MakeZ2IsingSiteTensorAt(size_t n,
                                                  size_t r,
                                                  size_t c,
                                                  const KxFn &Kx,
                                                  const KyFn &Ky) {
  const Z2IsingTNBuilder b;
  const size_t cm1 = (c + n - 1) % n;
  const size_t rm1 = (r + n - 1) % n;
  const auto sl_l = Z2IsingTNBuilder::SqrtLambda(Kx(r, cm1));
  const auto sl_r = Z2IsingTNBuilder::SqrtLambda(Kx(r, c));
  const auto sl_d = Z2IsingTNBuilder::SqrtLambda(Ky(r, c));
  const auto sl_u = Z2IsingTNBuilder::SqrtLambda(Ky(rm1, c));
  return b.MakeSiteTensorFromSqrtLambda(sl_l, sl_d, sl_r, sl_u);
}

template<typename TenElemT, typename QNT>
double TraceTRGZ(const qlpeps::TensorNetwork2D<TenElemT, QNT> &tn, size_t dmax) {
  qlpeps::TRGContractor<TenElemT, QNT> trg(tn.rows(), tn.cols());
  trg.SetTruncateParams(qlpeps::TRGContractor<TenElemT, QNT>::TruncateParams::SVD(
    /*d_min=*/2,
              /*d_max=*/
              dmax,
              /*trunc_error=*/
              0.0));
  trg.Init(tn);
  const double Z = trg.Trace(tn);
  if (!std::isfinite(Z) || !(Z > 0.0)) {
    throw std::runtime_error("TraceTRGZ: TRG returned non-finite or non-positive Z.");
  }
  return Z;
}

// Helper for testing PunchHole correctness on Z2 Ising torus
void RunZ2IsingPunchHoleTest(size_t n, bool sample_diagonal_only) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;
  using TensorT = qlten::QLTensor<TenElemT, QNT>;

  const double K = 0.3;
  auto Kx = [&](size_t /*r*/, size_t /*c*/) { return K; };
  auto Ky = [&](size_t /*r*/, size_t /*c*/) { return K; };
  const auto tn = BuildZ2IsingTorusTN(n, Kx, Ky);

  qlpeps::TRGContractor<TenElemT, QNT> trg(n, n);
  // Dmax=8 is sufficient for topology validation.
  trg.SetTruncateParams(qlpeps::TRGContractor<TenElemT, QNT>::TruncateParams::SVD(/*d_min=*/2, /*d_max=*/8, /*trunc_error=*/0.0));
  trg.Init(tn);

  std::cerr << "[PunchHole" << n << "x" << n << "Z2Ising] begin Trace()" << std::endl;
  const TenElemT Z = trg.Trace(tn);
  std::cerr << "[PunchHole" << n << "x" << n << "Z2Ising] Trace() done, Z=" << Z << std::endl;
  ASSERT_TRUE(std::isfinite(Z));

  using qlten::Contract;
  double ratio_sum = 0.0;
  size_t ratio_cnt = 0;

  auto check_site = [&](size_t r, size_t c) {
      const SiteIdx site{r, c};
      TensorT hole;
      ASSERT_NO_THROW(hole = trg.PunchHole(tn, site));

      TensorT out;
      const TensorT& Ts = tn({r, c});
      ASSERT_NO_THROW(Contract(&hole, {0, 1, 2, 3}, &Ts, {0, 1, 2, 3}, &out));
      const TenElemT z_rec = out();

      const double ratio = z_rec / Z;
      ratio_sum += ratio;
      ++ratio_cnt;

      // Loose per-site check
      EXPECT_NEAR(ratio, 1.0, 5e-2) << "site=(" << r << "," << c << ") ratio=" << ratio;
  };

  if (sample_diagonal_only) {
      for (size_t i = 0; i < n; ++i) {
          check_site(i, i);
      }
  } else {
      for (size_t r = 0; r < n; ++r) {
          for (size_t c = 0; c < n; ++c) {
              check_site(r, c);
          }
      }
  }

  const double ratio_mean = ratio_sum / double(ratio_cnt);
  EXPECT_NEAR(ratio_mean, 1.0, 3e-3) << "mean ratio=" << ratio_mean;
}

// Exact Ising torus references (computed externally via Python transfer-matrix).
// Keep tests self-contained and fast; do NOT re-run transfer matrix in C++.
constexpr double kZ2IsingPbcZ_3x3_K0p3 = 1551.4044599910355;
constexpr double kZ2IsingPbcZ_6x6_K0p3 = 2394192514840.829;
} // namespace

TEST(TRGContractorPBC, Uniform4x4) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;

  const size_t n = 4;
  const double K = 0.3;
  constexpr double Z_tm_ref = 3.57011518773655815e+05; // Python transfer-matrix (M=N=4, K=0.3)

  auto Kx = [&](size_t /*r*/, size_t /*c*/) { return K; };
  auto Ky = [&](size_t /*r*/, size_t /*c*/) { return K; };
  const auto tn = BuildZ2IsingTorusTN(n, Kx, Ky);

  const double Z_trg = TraceTRGZ<TenElemT, QNT>(tn, /*dmax=*/16);
  EXPECT_NEAR(Z_trg, Z_tm_ref, 1e-10 * std::max(1.0, std::abs(Z_tm_ref)));
}

TEST(TRGContractorPBC, Uniform3x3Z2Ising) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;

  const size_t n = 3;
  const double K = 0.3;

  auto Kx = [&](size_t /*r*/, size_t /*c*/) { return K; };
  auto Ky = [&](size_t /*r*/, size_t /*c*/) { return K; };
  const auto tn = BuildZ2IsingTorusTN(n, Kx, Ky);

  qlpeps::TRGContractor<TenElemT, QNT> trg(n, n);
  trg.SetTruncateParams(decltype(trg)::TruncateParams::SVD(/*d_min=*/2, /*d_max=*/16, /*trunc_error=*/0.0));
  trg.Init(tn);

  const double Z_trg = trg.Trace(tn);
  EXPECT_NEAR(Z_trg, kZ2IsingPbcZ_3x3_K0p3, 1e-12 * std::max(1.0, std::abs(kZ2IsingPbcZ_3x3_K0p3)));
}

TEST(TRGContractorPBC, Uniform6x6Z2Ising) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;

  const size_t n = 6;
  const double K = 0.3;

  auto Kx = [&](size_t /*r*/, size_t /*c*/) { return K; };
  auto Ky = [&](size_t /*r*/, size_t /*c*/) { return K; };
  const auto tn = BuildZ2IsingTorusTN(n, Kx, Ky);

  qlpeps::TRGContractor<TenElemT, QNT> trg(n, n);
  // Use a generous Dmax; for 6x6 this should be enough to match the exact transfer-matrix result.
  trg.SetTruncateParams(decltype(trg)::TruncateParams::SVD(/*d_min=*/2, /*d_max=*/8, /*trunc_error=*/0.0));
  trg.Init(tn);

  const double Z_trg = trg.Trace(tn);
  EXPECT_NEAR(Z_trg, kZ2IsingPbcZ_6x6_K0p3, 1e-10 * std::max(1.0, std::abs(kZ2IsingPbcZ_6x6_K0p3)));
}

TEST(TRGContractorPBC, NonUniform4x4) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;

  const size_t n = 4;
  const double K0 = 0.3;
  // Deterministic non-uniform couplings (no pinning field):
  // - Horizontal Kx depends on (r+c) parity: K0*(1 ± 0.2)
  // - Vertical   Ky depends on row parity:  K0*(1 ± 0.1)
  //
  // Python TM reference used:
  //   Z_tm_nonuniform_4x4 = 3.65044896079656901e+05
  constexpr double Z_tm_ref = 3.65044896079656901e+05;

  auto Kx = [&](size_t r, size_t c) -> double {
    return K0 * (1.0 + 0.2 * (((r + c) & 1U) ? -1.0 : +1.0));
  };
  auto Ky = [&](size_t r, size_t /*c*/) -> double {
    return K0 * (1.0 + 0.1 * ((r & 1U) ? -1.0 : +1.0));
  };
  const auto tn = BuildZ2IsingTorusTN(n, Kx, Ky);

  const double Z_trg = TraceTRGZ<TenElemT, QNT>(tn, /*dmax=*/16);
  EXPECT_NEAR(Z_trg, Z_tm_ref, 1e-10 * std::max(1.0, std::abs(Z_tm_ref)));
}

TEST(TRGContractorPBC, NonUniform8x8) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;

  const size_t n = 8; // supported by TRGContractor (n = 2^m or 3*2^k)
  const double K0 = 0.5;

  // Deterministic, strictly-positive, non-uniform couplings (no pinning field, Z2 symmetric).
  // These match the Python transfer-matrix reference used to generate logZ_tm_ref below:
  //   Kx[r,c] = K0*(1 + 0.1*(r+1) + 0.2*(c+1))
  //   Ky[r,c] = K0*(0.8 + 0.3*(c+1) + 0.5*(r+1))
  //
  // Python TM (M=N=8, 256x256) gives:
  //   logZ_tm = 2.16695339878650913e+02
  //   Kx_sum  = 7.52000000000000028e+01, Ky_sum = 1.40800000000000011e+02
  //   Kx_min  = 6.50000000000000022e-01, Kx_max = 1.70000000000000018e+00
  //   Ky_min  = 8.00000000000000044e-01, Ky_max = 3.60000000000000009e+00
  constexpr double logZ_tm_ref = 2.16695339878650913e+02;

  auto Kx = [&](size_t r, size_t c) -> double {
    return K0 * (1.0 + 0.1 * double(r + 1) + 0.2 * double(c + 1));
  };
  auto Ky = [&](size_t r, size_t c) -> double {
    return K0 * (0.8 + 0.3 * double(c + 1) + 0.5 * double(r + 1));
  };
  const auto tn = BuildZ2IsingTorusTN(n, Kx, Ky);
  const double Z_trg = TraceTRGZ<TenElemT, QNT>(tn, /*dmax=*/16);
  const double logZ_trg = std::log(Z_trg);

  // For larger systems TRG may introduce approximation once bond dimension saturates.
  // We compare logZ, which is numerically stable and is the quantity used in many checks.
  EXPECT_NEAR(logZ_trg, logZ_tm_ref, 1e-10 * std::max(1.0, std::abs(logZ_tm_ref)));
}

TEST(TRGContractorPBC, TrialReplacementSingleBond4x4) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;
  using Contractor = qlpeps::TRGContractor<TenElemT, QNT>;
  using TensorT = typename Contractor::Tensor;

  const size_t n = 4;
  const double K0 = 0.3;
  const size_t r0 = 1;
  const size_t c0 = 2;
  const double K1 = 0.37; // modify one horizontal bond Kx(r0,c0)

  // Pure-Python transfer-matrix reference (4x4 torus, Ky=K0 uniform, Kx has one modified bond).
  constexpr double Z_tm_ref = 3.68441965476735204e+05;
  constexpr double Z0_tm_ref = 3.57011518773655815e+05; // uniform K=0.3 (same as Uniform4x4)

  auto Kx0 = [&](size_t /*r*/, size_t /*c*/) { return K0; };
  auto Ky0 = [&](size_t /*r*/, size_t /*c*/) { return K0; };
  const auto tn0 = BuildZ2IsingTorusTN(n, Kx0, Ky0);

  auto Kx1 = [&](size_t r, size_t c) -> double { return (r == r0 && c == c0) ? K1 : K0; };
  auto Ky1 = [&](size_t /*r*/, size_t /*c*/) -> double { return K0; };

  Contractor trg(n, n);
  trg.SetTruncateParams(Contractor::TruncateParams::SVD(/*d_min=*/2, /*d_max=*/16, /*trunc_error=*/0.0));
  trg.Init(tn0);

  // Initialize cache.
  const double Z0 = trg.Trace(tn0);
  EXPECT_NEAR(Z0, Z0_tm_ref, 1e-10 * std::max(1.0, std::abs(Z0_tm_ref)));

  // Only two site tensors change when one horizontal bond Kx(r0,c0) is modified:
  // - site (r0, c0)     (right leg uses Kx(r0,c0))
  // - site (r0, c0+1)   (left  leg uses Kx(r0,c0))
  const SiteIdx sL{r0, c0};
  const SiteIdx sR{r0, (c0 + 1) % n};
  const TensorT tL = MakeZ2IsingSiteTensorAt(n, sL.row(), sL.col(), Kx1, Ky1);
  const TensorT tR = MakeZ2IsingSiteTensorAt(n, sR.row(), sR.col(), Kx1, Ky1);

  const std::vector<std::pair<SiteIdx, TensorT> > repl{{sL, tL}, {sR, tR}};
  auto trial = trg.BeginTrialWithReplacement(repl);
  const double Z_trial = trial.amplitude;
  EXPECT_NEAR(Z_trial, Z_tm_ref, 1e-10 * std::max(1.0, std::abs(Z_tm_ref)));

  // Commit trial into cache, then `Trace()` should return the modified Z without needing to reload tn.
  trg.CommitTrial(std::move(trial));
  const auto tn1 = BuildZ2IsingTorusTN(n, Kx1, Ky1);
  const double Z_after = trg.Trace(tn1);
  EXPECT_NEAR(Z_after, Z_tm_ref, 1e-10 * std::max(1.0, std::abs(Z_tm_ref)));
}

TEST(TRGContractorPBC, PunchHole2x2U1Random) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::U1QN;
  using IndexT = qlten::Index<QNT>;
  using QNSctT = qlten::QNSector<QNT>;
  using TensorT = qlten::QLTensor<TenElemT, QNT>;

  auto q = [](int sz) { return QNT({QNCard("Sz", U1QNVal(sz))}); };

  // Build 8 distinct bond indices (each has at least charge-0 sector, so divergence-0 blocks exist).
  const IndexT x01_out({QNSctT(q(0), 1), QNSctT(q(+1), 2)}, TenIndexDirType::OUT); // (0,0).R -> (0,1).L
  const IndexT x10_out({QNSctT(q(0), 2), QNSctT(q(-1), 1)}, TenIndexDirType::OUT); // (0,1).R -> (0,0).L
  const IndexT x23_out({QNSctT(q(0), 1), QNSctT(q(+2), 1)}, TenIndexDirType::OUT); // (1,0).R -> (1,1).L
  const IndexT x32_out({QNSctT(q(0), 3)}, TenIndexDirType::OUT); // (1,1).R -> (1,0).L

  const IndexT y02_out({QNSctT(q(0), 1), QNSctT(q(+1), 1), QNSctT(q(-1), 1)}, TenIndexDirType::OUT);
  // (0,0).D -> (1,0).U
  const IndexT y20_out({QNSctT(q(0), 2), QNSctT(q(+1), 1)}, TenIndexDirType::OUT); // (1,0).D -> (0,0).U
  const IndexT y13_out({QNSctT(q(0), 1), QNSctT(q(-2), 1)}, TenIndexDirType::OUT); // (0,1).D -> (1,1).U
  const IndexT y31_out({QNSctT(q(0), 2), QNSctT(q(-1), 2)}, TenIndexDirType::OUT); // (1,1).D -> (0,1).U

  // Site tensors follow TRGContractor convention: (l IN, d OUT, r OUT, u IN).
  auto make_site = [&](const IndexT &l_in, const IndexT &d_out, const IndexT &r_out, const IndexT &u_in) {
    TensorT T({l_in, d_out, r_out, u_in});
    T.Random(QNT(0)); // divergence 0
    return T;
  };

  qlpeps::TensorNetwork2D<TenElemT, QNT> tn(2, 2, qlpeps::BoundaryCondition::Periodic);

  // Layout:
  //   0:(0,0)  1:(0,1)
  //   2:(1,0)  3:(1,1)
  //
  // Horizontal bonds (two per row):
  //   0.R <-> 1.L uses x01_out
  //   1.R <-> 0.L uses x10_out
  //   2.R <-> 3.L uses x23_out
  //   3.R <-> 2.L uses x32_out
  //
  // Vertical bonds (two per column):
  //   0.D <-> 2.U uses y02_out
  //   2.D <-> 0.U uses y20_out
  //   1.D <-> 3.U uses y13_out
  //   3.D <-> 1.U uses y31_out
  tn({0, 0}) = make_site(InverseIndex(x10_out), y02_out, x01_out, InverseIndex(y20_out));
  tn({0, 1}) = make_site(InverseIndex(x01_out), y13_out, x10_out, InverseIndex(y31_out));
  tn({1, 0}) = make_site(InverseIndex(x32_out), y20_out, x23_out, InverseIndex(y02_out));
  tn({1, 1}) = make_site(InverseIndex(x23_out), y31_out, x32_out, InverseIndex(y13_out));

  qlpeps::TRGContractor<TenElemT, QNT> trg(2, 2);
  trg.SetTruncateParams(decltype(trg)::TruncateParams::SVD(/*d_min=*/2, /*d_max=*/8, /*trunc_error=*/0.0));
  trg.Init(tn);

  const TenElemT Z = trg.Trace(tn);
  ASSERT_TRUE(std::isfinite(Z));

  using qlten::Contract;
  for (size_t r = 0; r < 2; ++r) {
    for (size_t c = 0; c < 2; ++c) {
      const SiteIdx site{r, c};
      const TensorT hole = trg.PunchHole(tn, site);

      TensorT out;
      const TensorT &Ts = tn({r, c});
      Contract(&hole, {0, 1, 2, 3}, &Ts, {0, 1, 2, 3}, &out);
      const TenElemT z_reconstructed = out();
      EXPECT_NEAR(z_reconstructed, Z, 1e-10 * std::max(1.0, std::abs(Z)));
    }
  }
}

TEST(TRGContractorPBC, PunchHole3x3Z2Ising) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;
  using TensorT = qlten::QLTensor<TenElemT, QNT>;

  const size_t n = 3;
  const double K = 0.3;
  auto Kx = [&](size_t /*r*/, size_t /*c*/) { return K; };
  auto Ky = [&](size_t /*r*/, size_t /*c*/) { return K; };
  const auto tn = BuildZ2IsingTorusTN(n, Kx, Ky);

  qlpeps::TRGContractor<TenElemT, QNT> trg(n, n);
  trg.SetTruncateParams(decltype(trg)::TruncateParams::SVD(/*d_min=*/2, /*d_max=*/8, /*trunc_error=*/0.0));
  trg.Init(tn);

  const TenElemT Z = trg.Trace(tn);
  ASSERT_TRUE(std::isfinite(Z));

  using qlten::Contract;
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const SiteIdx site{r, c};
      TensorT hole;
      ASSERT_NO_THROW(hole = trg.PunchHole(tn, site));

      TensorT out;
      const TensorT& Ts = tn({r, c});
      ASSERT_NO_THROW(Contract(&hole, {0, 1, 2, 3}, &Ts, {0, 1, 2, 3}, &out));
      const TenElemT z_reconstructed = out();
      EXPECT_NEAR(z_reconstructed, Z, 1e-12 * std::max(1.0, std::abs(Z)));
    }
  }
}

TEST(TRGContractorPBC, PunchHole4x4U1RandomDistinctLegs) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::U1QN;
  using IndexT = qlten::Index<QNT>;
  using QNSctT = qlten::QNSector<QNT>;
  using TensorT = qlten::QLTensor<TenElemT, QNT>;

  auto q = [](int sz) { return QNT({QNCard("Sz", U1QNVal(sz))}); };

  constexpr size_t n = 4;

  // Build per-directed-bond indices:
  // - x_out[r,c] is (r,c).R -> (r,c+1).L
  // - y_out[r,c] is (r,c).D -> (r+1,c).U
  //
  // Each index includes at least charge-0 sector so divergence-0 blocks exist.
  std::array<IndexT, n * n> x_out;
  std::array<IndexT, n * n> y_out;
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const size_t id = r * n + c;
      const int d0 = 1 + int(id % 3); // 1..3
      const int d1 = 1 + int((id / 3) % 2); // 1..2
      const int d2 = 1 + int((id / 5) % 2); // 1..2
      x_out[id] = IndexT({QNSctT(q(0), d0), QNSctT(q(+1), d1), QNSctT(q(-1), d2)}, TenIndexDirType::OUT);

      const int e0 = 1 + int((id + 1) % 3); // 1..3
      const int e2 = 1 + int(((id + 2) / 4) % 2); // 1..2
      y_out[id] = IndexT({QNSctT(q(0), e0), QNSctT(q(+2), e2)}, TenIndexDirType::OUT);
    }
  }

  // Site tensors follow TRGContractor convention: (l IN, d OUT, r OUT, u IN).
  auto make_site = [&](const IndexT &l_in, const IndexT &d_out, const IndexT &r_out, const IndexT &u_in) {
    TensorT T({l_in, d_out, r_out, u_in});
    T.Random(QNT(0)); // divergence 0
    return T;
  };

  qlpeps::TensorNetwork2D<TenElemT, QNT> tn(n, n, qlpeps::BoundaryCondition::Periodic);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const size_t id = r * n + c;
      const size_t cm1 = (c + n - 1) % n;
      const size_t rm1 = (r + n - 1) % n;
      const size_t id_l = r * n + cm1; // left neighbor contributes x_out[r,cm1] to our l_in
      const size_t id_u = rm1 * n + c; // up   neighbor contributes y_out[rm1,c] to our u_in

      const IndexT l_in = InverseIndex(x_out[id_l]);
      const IndexT d_out = y_out[id];
      const IndexT r_out = x_out[id];
      const IndexT u_in = InverseIndex(y_out[id_u]);
      tn({r, c}) = make_site(l_in, d_out, r_out, u_in);
    }
  }

  qlpeps::TRGContractor<TenElemT, QNT> trg(n, n);
  // Use large enough Dmax to avoid truncation being the dominant error source in this wiring test.
  trg.SetTruncateParams(decltype(trg)::TruncateParams::SVD(/*d_min=*/2, /*d_max=*/8, /*trunc_error=*/0.0));
  trg.Init(tn);

  const TenElemT Z = trg.Trace(tn);
  ASSERT_TRUE(std::isfinite(Z));

  using qlten::Contract;
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const SiteIdx site{r, c};
      TensorT hole;
      ASSERT_NO_THROW(hole = trg.PunchHole(tn, site));

// Optional (expensive) probe baseline comparisons for local debugging.
// This should stay OFF in normal CI runs.
#if defined(QLPEPS_TRG_PUNCHHOLE_COMPARE_BASELINE)
      if (r == 0 && c == 0) {
        const TensorT hole_ref = trg.PunchHoleBaselineByProbingForTest(tn, site);
        TensorT out_ref;
        const TensorT& Ts0 = tn({r, c});
        ASSERT_NO_THROW(Contract(&hole_ref, {0, 1, 2, 3}, &Ts0, {0, 1, 2, 3}, &out_ref));
        const TenElemT z_ref = out_ref();
        EXPECT_NEAR(z_ref, Z, 1e-10 * std::max(1.0, std::abs(Z)));
      }
#endif

      TensorT out;
      const TensorT &Ts = tn({r, c});
      ASSERT_NO_THROW(Contract(&hole, {0, 1, 2, 3}, &Ts, {0, 1, 2, 3}, &out));
      const TenElemT z_reconstructed = out();
      EXPECT_NEAR(z_reconstructed, Z, 1e-10 * std::max(1.0, std::abs(Z)));
    }
  }
}

TEST(TRGContractorPBC, PunchHole4x4Z2Ising) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::Z2QN;

  const size_t n = 4;
  const double K = 0.3;
  auto Kx = [&](size_t /*r*/, size_t /*c*/) { return K; };
  auto Ky = [&](size_t /*r*/, size_t /*c*/) { return K; };
  const auto tn = BuildZ2IsingTorusTN(n, Kx, Ky);

  qlpeps::TRGContractor<TenElemT, QNT> trg(n, n);
  // Use a small Dmax first to make topology/indexing bugs easier to spot (less room to hide behind large subspaces).
  trg.SetTruncateParams(decltype(trg)::TruncateParams::SVD(/*d_min=*/2, /*d_max=*/16, /*trunc_error=*/0.0));
  trg.Init(tn);

  // Initialize cache so the baseline probing uses a clean, initialized TRG cache.
  double Z = 0.0;
  ASSERT_NO_THROW(Z = trg.Trace(tn));

  // Correctness check (option-1 semantics):
  // For each site i, contract hole_i with the original site tensor and recover the global trace.
  using qlten::Contract;
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const SiteIdx site{r, c};
      decltype(trg)::Tensor hole;
      // Now use the production PunchHole which supports 4x4
      ASSERT_NO_THROW(hole = trg.PunchHole(tn, site));

      const auto &Ts = tn({site.row(), site.col()});

      // Optional: Compare with baseline for one site to ensure exact match of the hole tensor itself
      if (r == 0 && c == 0) {
#if defined(QLPEPS_TRG_PUNCHHOLE_COMPARE_BASELINE)
        decltype(trg)::Tensor hole_ref = trg.PunchHoleBaselineByProbingForTest(tn, site);
        // Expect very close match (same TRG environment logic)
        decltype(trg)::Tensor diff;
        diff = hole + (-hole_ref);
        EXPECT_NEAR(diff.GetQuasi2Norm(), 0.0, 2e-4);
        // Debug code: we sometimes observe maybe hole = 4 * hole_ref
        // auto diff2 = hole + (-4.0 * hole_ref);
        // if (diff2.Get2Norm() < 1e-6) {
        //   std::cout << "[diff2]" << diff2 <<", Yes, hole = 4 * hole_ref"<< std::endl;
        // }

        // Baseline sanity: contracting the baseline hole with the original site tensor
        // should reconstruct the same global trace (approximately under truncation).
        {
          decltype(trg)::Tensor out_ref;
          ASSERT_NO_THROW(Contract(&hole_ref, {0, 1, 2, 3}, &Ts, {0, 1, 2, 3}, &out_ref));
          const double z_ref = out_ref();
          EXPECT_NEAR(z_ref, Z, 1e-6 * std::max(1.0, std::abs(Z)));
        }
#endif
      }

      typename decltype(trg)::Tensor out;
      ASSERT_NO_THROW(Contract(&hole, {0, 1, 2, 3}, &Ts, {0, 1, 2, 3}, &out));
      const double z_reconstructed = out();

      // With truncation this is approximate; allow a looser relative tolerance for small Dmax.
      EXPECT_NEAR(z_reconstructed, Z, 1e-6 * std::max(1.0, std::abs(Z)));
    }
  }
}

TEST(TRGContractorPBC, PunchHole6x6Z2Ising) {
  RunZ2IsingPunchHoleTest(6, /*sample_diagonal_only=*/false);
}

TEST(TRGContractorPBC, PunchHole8x8Z2Ising) {
  // Use diagonal sampling to speed up large system check
  RunZ2IsingPunchHoleTest(8, /*sample_diagonal_only=*/true);
}

TEST(TRGContractorPBC, PunchHole12x12Z2Ising) {
  // Use diagonal sampling to speed up large system check
  RunZ2IsingPunchHoleTest(12, /*sample_diagonal_only=*/true);
}
