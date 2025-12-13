// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2025-12-12
 *
 * Description: QuantumLiquids/PEPS project.
 * Implementation of TRGContractor (header-only).
 */

#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_IMPL_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_IMPL_H

#include <cassert>
#include <string>

#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "trg_contractor.h"

namespace qlpeps {

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::ResetGeometry_(size_t rows, size_t cols) {
  rows_ = rows;
  cols_ = cols;
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::ClearCache() {
  scales_.clear();
  dirty_scale0_.clear();
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::Init(const TensorNetwork2D<TenElemT, QNT>& tn) {
  bc_ = tn.GetBoundaryCondition();
  ResetGeometry_(tn.rows(), tn.cols());

  if (bc_ != BoundaryCondition::Periodic) {
    throw std::invalid_argument("TRGContractor::Init: requires BoundaryCondition::Periodic.");
  }
  if (rows_ != cols_) {
    throw std::invalid_argument("TRGContractor::Init: requires square lattice (rows == cols).");
  }
  if (!IsPowerOfTwo_(rows_)) {
    throw std::invalid_argument("TRGContractor::Init: requires n = 2^m (power-of-two linear size).");
  }

  ClearCache();
  scales_.resize(1);
  scales_[0].tens.resize(rows_ * cols_);
  // Note: we intentionally do not deep-copy tensors here. TRG pipeline will read from `tn`
  // when building coarse-grained tensors. The scale-0 tensor vector is kept for future
  // implementations that may want to store projected/renormalized tensors explicitly.
  BuildScale0GraphPBCSquare_();
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::BuildScale0GraphPBCSquare_() {
  auto& g = scales_[0].graph;
  const size_t n = rows_;
  const size_t N = n * n;
  g.nbr.assign(N, {});
  g.sublattice.resize(N);
  g.split_dir.resize(N);

  auto mod = [n](size_t x) { return (x + n) % n; };

  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const uint32_t id = NodeId_(r, c);
      const size_t left_c = mod(c - 1);
      const size_t right_c = mod(c + 1);
      const size_t down_r = mod(r + 1);
      const size_t up_r = mod(r - 1);

      const uint32_t left_id = NodeId_(r, left_c);
      const uint32_t right_id = NodeId_(r, right_c);
      const uint32_t down_id = NodeId_(down_r, c);
      const uint32_t up_id = NodeId_(up_r, c);

      // Leg order: 0=left, 1=down, 2=right, 3=up.
      // PBC adjacency:
      //  - my 0 <-> left 2
      //  - my 2 <-> right 0
      //  - my 1 <-> down 3
      //  - my 3 <-> up 1
      g.nbr[id][0] = Neighbor{left_id, 2};
      g.nbr[id][2] = Neighbor{right_id, 0};
      g.nbr[id][1] = Neighbor{down_id, 3};
      g.nbr[id][3] = Neighbor{up_id, 1};

      const bool is_a = ((r + c) & 1U) == 0U;
      g.sublattice[id] = is_a ? SubLattice::A : SubLattice::B;

      // Navy–Levin TRG convention: A/B use different split directions.
      // We intentionally keep this explicit instead of sprinkling if-else across the code.
      g.split_dir[id] = is_a ? SplitDir::Horizontal : SplitDir::Vertical;
    }
  }

#ifndef NDEBUG
  // Sanity check: nbr symmetry.
  for (uint32_t u = 0; u < static_cast<uint32_t>(N); ++u) {
    for (uint8_t leg = 0; leg < 4; ++leg) {
      const Neighbor v = g.nbr[u][leg];
      assert(v.node < N);
      assert(v.leg < 4);
      const Neighbor back = g.nbr[v.node][v.leg];
      assert(back.node == u);
      assert(back.leg == leg);
    }
  }
#endif
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::MarkDirtySeed_(uint32_t node) {
  dirty_scale0_.insert(node);
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::InvalidateEnvs(const SiteIdx& site) {
  if (bc_ != BoundaryCondition::Periodic) {
    // If user calls this before Init(), keep behavior strict and explicit.
    throw std::logic_error("TRGContractor::InvalidateEnvs: call Init(tn) first.");
  }
  if (site.row() >= rows_ || site.col() >= cols_) {
    throw std::out_of_range("TRGContractor::InvalidateEnvs: site out of range.");
  }
  MarkDirtySeed_(NodeId_(site.row(), site.col()));
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::Trace(const TensorNetwork2D<TenElemT, QNT>& tn) const {
  if (bc_ != BoundaryCondition::Periodic) {
    throw std::logic_error("TRGContractor::Trace: call Init(tn) first.");
  }
  if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) {
    throw std::invalid_argument("TRGContractor::Trace: tn must be periodic.");
  }
  if (tn.rows() != rows_ || tn.cols() != cols_) {
    throw std::invalid_argument("TRGContractor::Trace: tn geometry differs from Init(tn).");
  }
  if constexpr (Tensor::IsFermionic()) {
    throw std::runtime_error("TRGContractor::Trace: fermionic TRG is not implemented yet.");
  }
  if (!trunc_params_.has_value()) {
    throw std::logic_error("TRGContractor::Trace: truncation params are not set. Call SetTruncateParams().");
  }
  const auto& trunc_params = *trunc_params_;
  if (trunc_params.compress_scheme != CompressMPSScheme::SVD_COMPRESS) {
    throw std::invalid_argument("TRGContractor::Trace: TRG currently supports only SVD truncation (no variational).");
  }

  // ------------------------------------------------------------
  // Finite-size TRG pipeline (checkerboard plaquette coarse-graining)
  //
  // We alternate two kinds of scales:
  // - Even scales: axis-aligned n x n vertex tensors (count = n^2).
  // - Odd scales:  diagonal/rotated lattice embedded in n x n but only parity-even sites exist
  //               (count = n^2/2). We index those nodes by their embedding coordinate (r,c) with (r+c)%2==0.
  //
  // The (even -> odd) step maps n^2 -> n^2/2 (64->32) using disjoint 2x2 plaquettes with (r+c)%2==0.
  // The (odd  -> even) step maps n^2/2 -> (n/2)^2 (32->16) using disjoint "diamond plaquettes" on the
  // rotated lattice selected by a simple checkerboard rule.
  // ------------------------------------------------------------

  using qlten::Contract;
  using qlten::ElementWiseSqrt;
  using qlten::Eye;
  using qlten::SVD;

  struct SplitARes {
    Tensor P;  // "NW" piece: (leg0, leg3, alpha)
    Tensor Q;  // "SE" piece: (alpha, leg1, leg2)
  };
  struct SplitBRes {
    Tensor Q;  // "SW/N" piece depending on grouping: (leg0, leg1, alpha) for type1
    Tensor P;  // "NE/S" piece depending on grouping: (alpha, leg2, leg3) for type1
  };

  const auto split_type0 = [&](const Tensor& T_in) -> SplitARes {
    if (T_in.IsDefault()) {
      throw std::runtime_error("TRG split_type0: input tensor is default.");
    }
    // Group legs (0,3) | (1,2) by transposing to {0,3,1,2}.
    //
    // ASCII diagram (type0 split, A-sublattice):
    //
    //   up(3)                       up(3)
    //     |                          |
    // l(0)-T-r(2)   ==SVD==>    l(0)-P----alpha----Q-r(2)
    //     |                                        |
    //   d(1)                                     d(1)
    //
    // Grouping convention:
    //   left group  = (l, up)  = (0,3)
    //   right group = (d, r)   = (1,2)
    //
    // Output tensors:
    //   P(l, up, alpha)   with alpha as OUT (new bond leaving P)
    //   Q(alpha, d, r)    with alpha as IN  (new bond entering Q)
    Tensor T = T_in;
    T.Transpose({0, 3, 1, 2});
    Tensor u, s, vt;
    RealT trunc_err_actual = RealT(0);
    size_t bond_dim_actual = 0;
    SVD(&T,
        /*left_dims=*/2,
        T.Div(),
        trunc_params.trunc_err,
        trunc_params.D_min,
        trunc_params.D_max,
        &u,
        &s,
        &vt,
        &trunc_err_actual,
        &bond_dim_actual);
    auto s_sqrt = ElementWiseSqrt(s);
    SplitARes out;
    // P: u * sqrt(s)  (legs: 0,1,alpha) == (orig 0,3,alpha)
    Contract(&u, &s_sqrt, {{2}, {0}}, &out.P);
    // Q: sqrt(s) * vt (legs: alpha,2,3) == (alpha, orig 1,2)
    Contract(&s_sqrt, &vt, {{1}, {0}}, &out.Q);
    if (out.P.IsDefault() || out.Q.IsDefault()) {
      throw std::runtime_error("TRG split_type0 produced default tensor (likely index-direction/QN mismatch).");
    }
    return out;
  };

  const auto split_type1 = [&](const Tensor& T_in) -> SplitBRes {
    if (T_in.IsDefault()) {
      throw std::runtime_error("TRG split_type1: input tensor is default.");
    }
    // Group legs (0,1) | (2,3) in the default order.
    //
    // ASCII diagram (type1 split, B-sublattice):
    //
    //      up(3)                                 up(3)
    //      |                                     |
    // l(0)-T-r(2)   ==SVD==>  l(0)-Q----alpha----P-r(2)
    //      |                       |
    //    d(1)                      d(1)
    //
    // Grouping convention:
    //   left group  = (l, d)   = (0,1)
    //   right group = (r, up)  = (2,3)
    //
    // Output tensors:
    //   Q(l, d, alpha)    with alpha as OUT
    //   P(alpha, r, up)   with alpha as IN
    Tensor T = T_in;
    Tensor u, s, vt;
    RealT trunc_err_actual = RealT(0);
    size_t bond_dim_actual = 0;
    SVD(&T,
        /*left_dims=*/2,
        T.Div(),
        trunc_params.trunc_err,
        trunc_params.D_min,
        trunc_params.D_max,
        &u,
        &s,
        &vt,
        &trunc_err_actual,
        &bond_dim_actual);
    auto s_sqrt = ElementWiseSqrt(s);
    SplitBRes out;
    // Q: u * sqrt(s)  (legs: 0,1,alpha) == (orig 0,1,alpha)
    Contract(&u, &s_sqrt, {{2}, {0}}, &out.Q);
    // P: sqrt(s) * vt (legs: alpha,2,3) == (alpha, orig 2,3)
    Contract(&s_sqrt, &vt, {{1}, {0}}, &out.P);
    if (out.P.IsDefault() || out.Q.IsDefault()) {
      throw std::runtime_error("TRG split_type1 produced default tensor (likely index-direction/QN mismatch).");
    }
    return out;
  };

  auto even_to_odd = [&](const std::vector<Tensor>& even_tens, size_t n) -> std::vector<Tensor> {
    // even_tens indexed by (r,c) -> r*n + c, legs are [L(0),D(1),R(2),U(3)]
    // output odd_tens indexed by (r,c) with (r+c)%2==0, stored as id = r*(n/2) + (c/2).
    //
    // ASCII diagram (even -> odd, 2x2 plaquette coarse-graining)
    //
    // We pick plaquettes with (r+c)%2==0 (checkerboard). For each 2x2 plaquette:
    //
    //   TL = (r, c)      TR = (r, c+1)
    //   BL = (r+1, c)    BR = (r+1, c+1)
    //
    // After splitting (TL,BR use type0; TR,BL use type1), we contract:
    //
    //          (external alphas become legs of the odd tensor)
    //
    //              alpha_NW (from TL.Q0)   alpha_NE (from TR.Q1)
    //                        \             /
    //                         \           /
    //                         [   odd tensor   ]   (stored with leg order [NW, NE, SE, SW])
    //                         /           \
    //                        /             \
    //              alpha_SE (from BR.P0)   alpha_SW (from BL.P1)
    //
    // Internal bonds (contracted):
    // - TL.down  <-> BL.up
    // - TR.down  <-> BR.up
    // - TR.left  <-> TL.right
    // - BL.right <-> BR.left
    std::vector<Tensor> odd_tens;
    odd_tens.resize(n * (n / 2));

    auto id_even = [n](size_t r, size_t c) { return r * n + c; };
    auto mod = [n](size_t x) { return (x + n) % n; };
    auto id_odd = [n](size_t r, size_t c) {
      // Requires (r+c)%2==0; r and c have the same parity so (c/2) is well-defined in [0, n/2).
      return r * (n / 2) + (c / 2);
    };

    for (size_t r = 0; r < n; ++r) {
      for (size_t c = 0; c < n; ++c) {
        if (((r + c) & 1U) != 0U) continue;  // only black plaquettes

        // 2x2 plaquette corners (with PBC)
        const size_t r0 = r;
        const size_t c0 = c;
        const size_t r1 = mod(r + 1);
        const size_t c1 = mod(c + 1);

        const Tensor& T_tl = even_tens[id_even(r0, c0)];
        const Tensor& T_tr = even_tens[id_even(r0, c1)];
        const Tensor& T_bl = even_tens[id_even(r1, c0)];
        const Tensor& T_br = even_tens[id_even(r1, c1)];

        // A/B assignment on vertex lattice: A at (r+c) even.
        // For this plaquette: TL and BR are A; TR and BL are B.
        const auto tl = split_type0(T_tl);  // A: P0/Q0
        const auto br = split_type0(T_br);  // A: P0/Q0
        const auto tr = split_type1(T_tr);  // B: Q1/P1
        const auto bl = split_type1(T_bl);  // B: Q1/P1

        // Select the four half-tensors that sit on the plaquette edges:
        // TL(A): use Q0(alpha, D, R) -> internal legs D,R
        // TR(B): use Q1(L, D, alpha) -> internal legs L,D
        // BL(B): use P1(alpha, R, U) -> internal legs R,U
        // BR(A): use P0(L, U, alpha) -> internal legs L,U
        const Tensor& Q0 = tl.Q;
        const Tensor& Q1 = tr.Q;
        const Tensor& P1 = bl.P;
        const Tensor& P0 = br.P;

        // Contract internal bonds.
        // 1) BL.R (P1 idx=1) with BR.L (P0 idx=0)
        Tensor tmp0;
        Contract(&P1, {1}, &P0, {0}, &tmp0);
        // tmp0 indices order: [P1(alpha), P1(U), P0(U), P0(alpha)]

        // 2) TR.L (Q1 idx=0) with TL.R (Q0 idx=2)
        Tensor tmp1;
        Contract(&Q1, {0}, &Q0, {2}, &tmp1);
        // tmp1 indices order: [Q1(D), Q1(alpha), Q0(alpha), Q0(D)]

        // 3) connect TL.D with BL.U and TR.D with BR.U
        // TL.D is in tmp1 last index (from Q0(D) at idx=3),
        // BL.U is in tmp0 index 1 (from P1(U)).
        // TR.D is in tmp1 index 0 (from Q1(D)),
        // BR.U is in tmp0 index 2 (from P0(U)).
        Tensor tmp2;
        Contract(&tmp0, {1, 2}, &tmp1, {3, 0}, &tmp2);
        // Remaining indices in tmp2 correspond to:
        // - tmp0: P1(alpha) [0], P0(alpha) [3]
        // - tmp1: Q1(alpha) [1], Q0(alpha) [2]
        // So tmp2 is rank-4 in some order; reorder to [TL, TR, BR, BL] = [Q0a, Q1a, P0a, P1a].
        // Current order is [P1a, P0a, Q1a, Q0a] (by construction above).
        tmp2.Transpose({3, 2, 1, 0});
        if (tmp2.IsDefault()) {
          throw std::runtime_error("TRG even_to_odd produced default coarse tensor.");
        }

        odd_tens[id_odd(r0, c0)] = tmp2;
      }
    }
    return odd_tens;
  };

  auto odd_to_even = [&](const std::vector<Tensor>& odd_tens, size_t n_embed) -> std::vector<Tensor> {
    // odd_tens live on parity-even coordinates (r,c) in Z_n^2, stored by id = r*(n/2) + (c/2)
    // leg order on each odd tensor: [0=NW, 1=NE, 2=SE, 3=SW] in embedding coordinates.
    //
    // We coarse-grain using disjoint "diamond plaquettes" centered at (i,j) with:
    // - i even, j odd
    // This gives exactly n^2/4 plaquettes, mapping n^2/2 -> (n/2)^2.
    //
    // ASCII diagram (odd -> even, diamond coarse-graining on rotated lattice)
    //
    // Odd-scale tensors live on parity-even embedding sites (r+c)%2==0 and have leg order:
    //   [0=NW, 1=NE, 2=SE, 3=SW].
    //
    // For each diamond center (i even, j odd), take four odd tensors:
    //
    //              N = (i-1, j)
    //                  /\
    //                 /  \
    //     W = (i, j-1)    E = (i, j+1)
    //                 \  /
    //                  \/
    //              S = (i+1, j)
    //
    // After splitting (E,W use type0; N,S use type1) we contract the diamond internal bonds,
    // producing one *even-scale* coarse tensor.
    //
    // IMPORTANT: all even-scale tensors must use a single global leg order:
    //   [L(0), D(1), R(2), U(3)].
    const size_t n = n_embed;
    if ((n % 2) != 0) {
      throw std::logic_error("TRG odd_to_even: embedding n must be even.");
    }
    const size_t n2 = n / 2;
    std::vector<Tensor> even_tens;
    even_tens.resize(n2 * n2);

    auto mod = [n](size_t x) { return (x + n) % n; };
    auto id_odd = [n](size_t r, size_t c) { return r * (n / 2) + (c / 2); };
    auto id_even = [n2](size_t r, size_t c) { return r * n2 + c; };

    auto is_A = [](size_t r, size_t c) {
      // On parity-even subset, (r,c) are either both even or both odd.
      // We use that to define A/B consistently across odd scales.
      return ((r & 1U) == 0U) && ((c & 1U) == 0U);
    };

    for (size_t i = 0; i < n; ++i) {
      if ((i & 1U) != 0U) continue;  // i even
      for (size_t j = 0; j < n; ++j) {
        if ((j & 1U) == 0U) continue;  // j odd
        const size_t I = i / 2;
        const size_t J = (j - 1) / 2;
        // Diamond nodes around center (i,j):
        // N=(i-1,j), E=(i,j+1), S=(i+1,j), W=(i,j-1)
        const size_t rn = mod(i - 1);
        const size_t cn = j;
        const size_t re = i;
        const size_t ce = mod(j + 1);
        const size_t rs = mod(i + 1);
        const size_t cs = j;
        const size_t rw = i;
        const size_t cw = mod(j - 1);

        const Tensor& Tn = odd_tens[id_odd(rn, cn)];
        const Tensor& Te = odd_tens[id_odd(re, ce)];
        const Tensor& Ts = odd_tens[id_odd(rs, cs)];
        const Tensor& Tw = odd_tens[id_odd(rw, cw)];

        // A/B around the diamond:
        // N and S are (odd,odd) -> B, E and W are (even,even) -> A (under our is_A).
        (void)is_A;  // keep in case we add asserts later.

        // Split convention on odd lattice:
        // - A nodes: use type0 split (group 0&3 | 1&2) i.e. west/east
        // - B nodes: use type1 split (group 0&1 | 2&3) i.e. north/south
        const auto splitA = [&](const Tensor& T_in) -> SplitARes { return split_type0(T_in); };
        const auto splitB = [&](const Tensor& T_in) -> SplitBRes { return split_type1(T_in); };

        const auto nB = splitB(Tn);
        const auto eA = splitA(Te);
        const auto sB = splitB(Ts);
        const auto wA = splitA(Tw);

        // Choose half-tensors containing the internal edges:
        // Internal bonds:
        // - N.leg2 <-> E.leg0
        // - N.leg3 <-> W.leg1
        // - S.leg1 <-> E.leg3
        // - S.leg0 <-> W.leg2
        //
        // Selected pieces:
        // - N (B): need legs2&3 -> use P (alpha,2,3)
        // - E (A): need legs0&3 -> use P (0,1,alpha) but note after split_type0 P corresponds to legs0&3.
        // - S (B): need legs0&1 -> use Q (0,1,alpha)
        // - W (A): need legs1&2 -> use Q (alpha,1,2) (east piece)
        const Tensor& Np = nB.P;  // (alpha,2,3)
        const Tensor& Ep = eA.P;  // (0,3,alpha) in embedded meaning, stored as (leg0,leg3,alpha)
        const Tensor& Sq = sB.Q;  // (0,1,alpha)
        const Tensor& Wq = wA.Q;  // (alpha,1,2)

        // Direct contraction path (cleaner):
        //
        // Step A: connect S.leg0 <-> W.leg2 using Sq & Wq into SW block.
        Tensor SW;
        Contract(&Sq, {0}, &Wq, {2}, &SW);
        // SW indices order: [leg1S, alphaS, alphaW, leg1W]
        //
        // Step B: connect N.leg2 <-> E.leg0 using Np & Ep into NE block.
        Tensor NE;
        Contract(&Np, {1}, &Ep, {0}, &NE);
        // NE indices order: [alphaN, leg3N, leg3E, alphaE]
        //
        // Step C: connect N.leg3 <-> W.leg1 and S.leg1 <-> E.leg3 by contracting SW and NE.
        //
        // - N.leg3 is in NE index 1 (leg3N)
        // - W.leg1 is in SW index 3 (leg1W)
        // - S.leg1 is in SW index 0 (leg1S)
        // - E.leg3 is in NE index 2 (leg3E)
        Tensor out;
        Contract(&SW, {0, 3}, &NE, {2, 1}, &out);
        // Remaining indices in out correspond to [alphaS, alphaW, alphaN, alphaE] in that order.
        //
        // IMPORTANT: We enforce a single global convention for *all even-scale tensors*:
        //   leg order = [L(0), D(1), R(2), U(3)].
        //
        // In this diamond coarse-graining, the new coarse tensor legs correspond to:
        //   L <- alphaW, D <- alphaS, R <- alphaE, U <- alphaN.
        // Therefore we reorder [alphaS, alphaW, alphaN, alphaE] -> [alphaW, alphaS, alphaE, alphaN].
        out.Transpose({1, 0, 3, 2});
        if (out.IsDefault()) {
          throw std::runtime_error("TRG odd_to_even produced default coarse tensor.");
        }

        even_tens[id_even(I, J)] = out;
      }
    }
    return even_tens;
  };

  auto contract_even_1x1 = [&](const Tensor& T) -> TenElemT {
    // PBC 1x1: left<->right, down<->up. Use identity tensors to trace paired legs.
    //
    // ASCII diagram (final 1x1 trace)
    //
    //            up(IN,3)
    //            ^
    //            |
    // L(IN,0) <- T -> R(OUT,2)
    //            |
    //            v
    //            D(OUT,1)
    //
    // PBC identifications:
    //   L <-> R,   D <-> U
    //
    // Implemented via contracting with identity tensors Eye():
    //   trace(L,R) then trace(D,U).
    const auto& idx_r = T.GetIndex(2);  // right (OUT) is the safe choice to build Eye
    const auto& idx_d = T.GetIndex(1);  // down (OUT)
    const auto eye_lr = Eye<TenElemT, QNT>(idx_r);  // (OUT, IN)
    const auto eye_du = Eye<TenElemT, QNT>(idx_d);  // (OUT, IN)
    Tensor tmp;
    Contract(&T, {0, 2}, &eye_lr, {0, 1}, &tmp);  // trace left(IN)-right(OUT)
    Tensor tmp2;
    // tmp carries the remaining legs (down, up). To trace down(OUT) with up(IN),
    // we must connect down to the IN leg of eye_du (axis=1) and up to the OUT leg (axis=0).
    Contract(&tmp, {0, 1}, &eye_du, {1, 0}, &tmp2);
    return tmp2();
  };

  // Build scale-0 tensor list from tn.
  const size_t n0 = rows_;
  std::vector<Tensor> cur_even;
  cur_even.resize(n0 * n0);
  for (size_t r = 0; r < n0; ++r) {
    for (size_t c = 0; c < n0; ++c) {
      cur_even[r * n0 + c] = tn({r, c});
      if (cur_even[r * n0 + c].IsDefault()) {
        throw std::runtime_error("TRGContractor::Trace: scale-0 tensor is default. "
                                 "This indicates an uninitialized TensorNetwork2D entry.");
      }
    }
  }

  size_t n = n0;
  while (true) {
    if (n == 1) {
      return contract_even_1x1(cur_even[0]);
    }
    // even n x n -> odd embedded n (count n^2/2)
    auto cur_odd = even_to_odd(cur_even, n);
    // odd embedded n -> even (n/2) x (n/2)
    cur_even = odd_to_even(cur_odd, n);
    n = n / 2;
  }
}

}  // namespace qlpeps

#endif  // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_IMPL_H


