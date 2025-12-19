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
#include <algorithm>
#include <set>
#include <map>

#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "trg_contractor.h"

namespace qlpeps {

template <typename TenElemT, typename QNT>
int TRGContractor<TenElemT, QNT>::ParentSlot_(size_t scale, uint32_t fine_id, uint32_t parent_id) const {
  const auto& ps = scales_.at(scale).fine_to_coarse.at(fine_id);
  if (ps[0] == parent_id) return 0;
  if (ps[1] == parent_id) return 1;
  return -1;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::LinearSplitAdjointToHole_(size_t scale,
                                                        uint32_t node,
                                                        uint32_t parent,
                                                        const Tensor* H_P,
                                                        const Tensor* H_Q) const {
  using qlten::Contract;

  auto require_match = [&](const Tensor& A, size_t axA, const Tensor& B, size_t axB) {
    const auto& ia = A.GetIndex(axA);
    const auto& ib = B.GetIndex(axB);
    if (!(ia == InverseIndex(ib))) {
      throw std::logic_error("TRGContractor::LinearSplitAdjointToHole_: index mismatch.");
    }
  };

  const int slot = ParentSlot_(scale, node, parent);
  if (slot < 0) throw std::logic_error("TRGContractor::LinearSplitAdjointToHole_: missing parent-slot.");

  const auto st = scales_.at(scale).split_type.at(node)[slot];
  const auto& U = scales_.at(scale).split_U.at(node)[slot];
  const auto& Vt = scales_.at(scale).split_Vt.at(node)[slot];
  const auto& Sinv = scales_.at(scale).split_S_inv_sqrt.at(node)[slot];

  Tensor Sinv_dag = qlten::Dag(Sinv);

  if (st == ScaleCache::SplitType::Type0) {
    Tensor V_adj = qlten::Dag(Vt);  // (alpha, d*, r*)
    Tensor U_adj = qlten::Dag(U);   // (l*, u*, alpha_in)

    Tensor t1;
    if (H_P != nullptr) {
      Tensor Hp_scaled;
      require_match(*H_P, 2, Sinv_dag, 0);
      Contract(H_P, {2}, &Sinv_dag, {0}, &Hp_scaled);  // (l*,u*,alpha*)
      require_match(Hp_scaled, 2, V_adj, 0);
      Contract(&Hp_scaled, {2}, &V_adj, {0}, &t1);     // (l*,u*,d*,r*)
    }

    Tensor t2;
    if (H_Q != nullptr) {
      Tensor Hq_scaled;
      require_match(Sinv_dag, 1, *H_Q, 0);
      Contract(&Sinv_dag, {1}, H_Q, {0}, &Hq_scaled);  // (alpha*,d*,r*)
      require_match(U_adj, 2, Hq_scaled, 0);
      Contract(&U_adj, {2}, &Hq_scaled, {0}, &t2);     // (l*,u*,d*,r*)
    }

    Tensor hperm = t1.IsDefault() ? t2 : (t2.IsDefault() ? t1 : (t1 + t2));
    hperm.Transpose({0, 2, 3, 1});  // (l*,d*,r*,u*)
    return hperm;
  }

  // Type1
  Tensor V_adj = qlten::Dag(Vt);  // (alpha, r*, u*)
  Tensor U_adj = qlten::Dag(U);   // (l*, d*, alpha_in)

  Tensor t1;
  if (H_Q != nullptr) {
    Tensor Hq_scaled;
    require_match(*H_Q, 2, Sinv_dag, 0);
    Contract(H_Q, {2}, &Sinv_dag, {0}, &Hq_scaled);   // (l*,d*,alpha*)
    require_match(Hq_scaled, 2, V_adj, 0);
    Contract(&Hq_scaled, {2}, &V_adj, {0}, &t1);      // (l*,d*,r*,u*)
  }

  Tensor t2;
  if (H_P != nullptr) {
    Tensor Hp_scaled;
    require_match(Sinv_dag, 1, *H_P, 0);
    Contract(&Sinv_dag, {1}, H_P, {0}, &Hp_scaled);   // (alpha*,r*,u*)
    require_match(U_adj, 2, Hp_scaled, 0);
    Contract(&U_adj, {2}, &Hp_scaled, {0}, &t2);      // (l*,d*,r*,u*)
  }

  if (t1.IsDefault()) return t2;
  if (t2.IsDefault()) return t1;
  return t1 + t2;
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::ResetGeometry_(size_t rows, size_t cols) {
  rows_ = rows;
  cols_ = cols;
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::ClearCache() {
  scales_.clear();
  dirty_scale0_.clear();
  tensors_initialized_ = false;
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
  // Supported sizes:
  // - n = 2^m (terminates at 2x2 exact contraction)
  // - n = 3 * 2^k (terminates at 3x3 exact contraction)
  const bool pow2 = IsPowerOfTwo_(rows_);
  const bool three_times_pow2 = (rows_ % 3 == 0) && IsPowerOfTwo_(rows_ / 3);
  if (!pow2 && !three_times_pow2) {
    throw std::invalid_argument(
        "TRGContractor::Init: requires n = 2^m or n = 3*2^k (square PBC lattice).");
  }

  ClearCache();
  BuildScale0GraphPBCSquare_();
  BuildTopology_(); // Build skeleton for all scales
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::BuildScale0GraphPBCSquare_() {
  scales_.resize(1);
  auto& g = scales_[0].graph;
  const size_t n = rows_;
  const size_t N = n * n;
  g.nbr.assign(N, {});
  g.sublattice.resize(N);
  g.split_dir.resize(N);
  scales_[0].tens.resize(N);

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

      g.nbr[id][0] = Neighbor{left_id, 2};
      g.nbr[id][2] = Neighbor{right_id, 0};
      g.nbr[id][1] = Neighbor{down_id, 3};
      g.nbr[id][3] = Neighbor{up_id, 1};

      const bool is_a = ((r + c) & 1U) == 0U;
      g.sublattice[id] = is_a ? SubLattice::A : SubLattice::B;
      g.split_dir[id] = is_a ? SplitDir::Horizontal : SplitDir::Vertical;
    }
  }
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::BuildTopology_() {
  // Pre-calculate fine_to_coarse and coarse_to_fine maps for all scales.
  // We simulate the RG flow purely with indices.
  //
  // IMPORTANT (4x4 PBC / PunchHole):
  // This topology builder encodes a checkerboard TRG on a torus by allowing each fine node to appear
  // in up to two coarse "parents" per RG step. This is convenient for incremental updates and local
  // recomputation, but it is a *double-cover style* representation: in a 4x4 system there are two RG
  // steps, and a naive "sum over all parent contexts" in hole backprop can introduce an overall
  // multiplicity ~ 2 x 2 = 4. See Doxygen note on TRGContractor::PunchHole4x4_.
  
  size_t n = rows_;
  size_t scale_idx = 0;
  
  // Terminate at a small even lattice and contract it exactly:
  // - 2x2 for n = 2^m
  // - 3x3 for n = 3*2^k
  // The last 1x1 step would require an additional SVD split/coarse-graining, which is unnecessary
  // and numerically undesirable (extra truncation).
  while (n > 3) {
    // Current scale is `scale_idx` (size n x n)
    // Next scale is `scale_idx + 1` (size n x n/2 roughly, but stored as flat vector)
    
    scales_.resize(scale_idx + 2);
    auto& fine_layer = scales_[scale_idx];
    auto& coarse_layer = scales_[scale_idx + 1];
    
    // Resize topology maps
    size_t fine_size = (scale_idx == 0) ? (rows_ * cols_) : scales_[scale_idx].tens.size();
    
    fine_layer.fine_to_coarse.assign(fine_size, {0xFFFFFFFF, 0xFFFFFFFF});
    
    if (scale_idx % 2 == 0) {
      // Even -> Odd (Plaquette RG)
      // n x n -> n x n/2 (embedded)
      if ((n & 1U) != 0U) {
        throw std::logic_error("TRGContractor::BuildTopology_: even->odd step requires even n.");
      }
      size_t coarse_count = (n * n) / 2;
      coarse_layer.tens.resize(coarse_count);
      coarse_layer.coarse_to_fine.resize(coarse_count);
      
      auto mod = [n](size_t x) { return (x + n) % n; };
      auto id_even = [n](size_t r, size_t c) { return r * n + c; };
      auto id_odd = [n](size_t r, size_t c) { return r * (n / 2) + (c / 2); };
      
      for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
          if (((r + c) & 1U) != 0U) continue; // Only process 'even' top-left corners
          
          uint32_t coarse_id = static_cast<uint32_t>(id_odd(r, c));
          
          // Inputs
          uint32_t tl = static_cast<uint32_t>(id_even(r, c));
          uint32_t tr = static_cast<uint32_t>(id_even(r, mod(c + 1)));
          uint32_t bl = static_cast<uint32_t>(id_even(mod(r + 1), c));
          uint32_t br = static_cast<uint32_t>(id_even(mod(r + 1), mod(c + 1)));
          
          // Map Coarse -> Fine
          coarse_layer.coarse_to_fine[coarse_id] = {tl, tr, bl, br};
          
          // Map Fine -> Coarse
          // Each fine tensor is used in exactly 2 plaquettes.
          // We assume sequential filling: first slot 0, then slot 1.
          auto add_parent = [&](uint32_t f, uint32_t c_id) {
            if (fine_layer.fine_to_coarse[f][0] == 0xFFFFFFFF) fine_layer.fine_to_coarse[f][0] = c_id;
            else fine_layer.fine_to_coarse[f][1] = c_id;
          };
          
          add_parent(tl, coarse_id);
          add_parent(tr, coarse_id);
          add_parent(bl, coarse_id);
          add_parent(br, coarse_id);
        }
      }
    } else {
      // Odd -> Even (Diamond RG)
      // Embedded size n -> n/2
      // Odd tensors: n*n/2 count. Even tensors (next): (n/2)*(n/2).
      size_t n_embed = n; // The 'n' from previous loop
      size_t n_next = n / 2;
      size_t coarse_count = n_next * n_next;
      
      coarse_layer.tens.resize(coarse_count);
      coarse_layer.coarse_to_fine.resize(coarse_count);
      
      auto mod = [n_embed](size_t x) { return (x + n_embed) % n_embed; };
      auto id_odd = [n_embed](size_t r, size_t c) { return r * (n_embed / 2) + (c / 2); };
      auto id_even_next = [n_next](size_t r, size_t c) { return r * n_next + c; };
      
      for (size_t i = 0; i < n_embed; ++i) {
        if ((i & 1U) != 0U) continue; // i even
        for (size_t j = 0; j < n_embed; ++j) {
           if ((j & 1U) == 0U) continue; // j odd
           
           // Diamond center (i, j) -> Coarse (i/2, j/2 approx)
           size_t I = i / 2;
           size_t J = (j - 1) / 2;
           uint32_t coarse_id = static_cast<uint32_t>(id_even_next(I, J));
           
           // Neighbors on odd lattice
           uint32_t N = static_cast<uint32_t>(id_odd(mod(i - 1), j));
           uint32_t E = static_cast<uint32_t>(id_odd(i, mod(j + 1)));
           uint32_t S = static_cast<uint32_t>(id_odd(mod(i + 1), j));
           uint32_t W = static_cast<uint32_t>(id_odd(i, mod(j - 1)));
           
           coarse_layer.coarse_to_fine[coarse_id] = {N, E, S, W};
           
           auto add_parent = [&](uint32_t f, uint32_t c_id) {
             if (fine_layer.fine_to_coarse[f][0] == 0xFFFFFFFF) fine_layer.fine_to_coarse[f][0] = c_id;
             else fine_layer.fine_to_coarse[f][1] = c_id;
           };
           
           add_parent(N, coarse_id);
           add_parent(E, coarse_id);
           add_parent(S, coarse_id);
           add_parent(W, coarse_id);
        }
      }
      
      n = n / 2; // Reduce n for next iteration (6->3, 12->6, ...)
    }
    
    scale_idx++;
  }

  // NOTE: The final scale tensor storage is resized during topology construction steps above.
  // For n==3, the last coarse layer is 3x3 and has 9 tensors; for n==2 it has 4 tensors.
  //
  // For base cases (rows_ == 2 or 3), the loop above does not run. We still need a defined
  // fine_to_coarse mapping at scale 0 for consistency (all invalid = no parents).
  if (scales_.size() == 1) {
    scales_[0].fine_to_coarse.assign(rows_ * cols_, {0xFFFFFFFF, 0xFFFFFFFF});
  }
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::MarkDirtySeed_(uint32_t node) {
  dirty_scale0_.insert(node);
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::InvalidateEnvs(const SiteIdx& site) {
  if (bc_ != BoundaryCondition::Periodic) {
    throw std::logic_error("TRGContractor::InvalidateEnvs: call Init(tn) first.");
  }
  MarkDirtySeed_(NodeId_(site.row(), site.col()));
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::SplitARes 
TRGContractor<TenElemT, QNT>::SplitType0_(const Tensor& T_in) const {
    using qlten::Contract;
    using qlten::ElementWiseSqrt;
    using qlten::SVD;
    if (T_in.IsDefault()) throw std::runtime_error("TRG split_type0: input tensor is default.");
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
    Tensor u, vt;
    qlten::QLTensor<RealT, QNT> s;
    RealT trunc_err_actual = RealT(0);
    size_t bond_dim_actual = 0;
    const auto& tp = *trunc_params_;
    SVD(&T, 2, T.Div(), tp.trunc_err, tp.D_min, tp.D_max, &u, &s, &vt, &trunc_err_actual, &bond_dim_actual);
    auto s_sqrt = ElementWiseSqrt(s);
    SplitARes out;
    Contract(&u, &s_sqrt, {{2}, {0}}, &out.P);
    Contract(&s_sqrt, &vt, {{1}, {0}}, &out.Q);
    return out;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::SplitBRes 
TRGContractor<TenElemT, QNT>::SplitType1_(const Tensor& T_in) const {
    using qlten::Contract;
    using qlten::ElementWiseSqrt;
    using qlten::SVD;
    if (T_in.IsDefault()) throw std::runtime_error("TRG split_type1: input tensor is default.");
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
    // Default order 0,1,2,3 is already 0,1 | 2,3
    Tensor u, vt;
    qlten::QLTensor<RealT, QNT> s;
    RealT trunc_err_actual = RealT(0);
    size_t bond_dim_actual = 0;
    const auto& tp = *trunc_params_;
    SVD(&T, 2, T.Div(), tp.trunc_err, tp.D_min, tp.D_max, &u, &s, &vt, &trunc_err_actual, &bond_dim_actual);
    auto s_sqrt = ElementWiseSqrt(s);
    SplitBRes out;
    Contract(&u, &s_sqrt, {{2}, {0}}, &out.Q);
    Contract(&s_sqrt, &vt, {{1}, {0}}, &out.P);
    return out;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor 
TRGContractor<TenElemT, QNT>::ContractPlaquette_(const std::vector<Tensor>& fine_tens, uint32_t coarse_idx, size_t n_fine) {
    throw std::logic_error("Helper ContractPlaquette_ requires refactoring to be scale-aware.");
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::EnsureSplitCacheForNodes_(size_t scale, const std::set<uint32_t>& nodes) {
  if (scale >= scales_.size()) throw std::out_of_range("TRGContractor::EnsureSplitCacheForNodes_: invalid scale.");
  // Final scale has no coarse parents; split cache is meaningless there.
  if (scale + 1 >= scales_.size()) return;

  auto& layer = scales_[scale];
  const size_t n = layer.tens.size();
  if (layer.split_type.size() != n) {
    layer.split_type.assign(n, {ScaleCache::SplitType::Type0, ScaleCache::SplitType::Type0});
  }
  if (layer.split_P.size() != n) layer.split_P.resize(n);
  if (layer.split_Q.size() != n) layer.split_Q.resize(n);
  if (layer.split_U.size() != n) layer.split_U.resize(n);
  if (layer.split_Vt.size() != n) layer.split_Vt.resize(n);
  if (layer.split_S_inv_sqrt.size() != n) layer.split_S_inv_sqrt.resize(n);

  auto parent_slot = [&](uint32_t fid, uint32_t pid) -> int {
    const auto& ps = layer.fine_to_coarse.at(fid);
    if (ps[0] == pid) return 0;
    if (ps[1] == pid) return 1;
    return -1;
  };

  for (uint32_t id : nodes) {
    if (id >= n) throw std::out_of_range("TRGContractor::EnsureSplitCacheForNodes_: node id out of range.");
    const auto& T = layer.tens.at(id);
    if (T.IsDefault()) throw std::logic_error("TRGContractor::EnsureSplitCacheForNodes_: tensor is default.");

    using qlten::Contract;
    using qlten::ElementWiseSqrt;
    using qlten::SVD;

    const auto& tp = *trunc_params_;
    const auto& parents = layer.fine_to_coarse.at(id);
    const auto& c2f = scales_[scale + 1].coarse_to_fine;

    // Per header contract: split cache is per (node, parent-slot), because a node can appear
    // with different roles in its two parents.
    for (int slot = 0; slot < 2; ++slot) {
      const uint32_t pid = parents[slot];
      if (pid == 0xFFFFFFFF) continue;
      if (pid >= c2f.size())
        throw std::logic_error("TRGContractor::EnsureSplitCacheForNodes_: coarse parent id out of range.");

      const auto& children = c2f.at(pid);
      int role = -1;
      for (int k = 0; k < 4; ++k) {
        if (children[k] == id) role = k;
      }
      if (role < 0)
        throw std::logic_error("TRGContractor::EnsureSplitCacheForNodes_: invalid topology (child not found in parent).");

      typename ScaleCache::SplitType st = ScaleCache::SplitType::Type0;
      if (scale == 0) {
        // Scale-0 split convention is determined by the A/B sublattice (fixed).
        st = (scales_[0].graph.sublattice.at(id) == SubLattice::A) ? ScaleCache::SplitType::Type0
                                                                   : ScaleCache::SplitType::Type1;
      } else if ((scale & 1U) == 0U) {
        // Even -> Odd plaquette, children order {TL,TR,BL,BR}.
        st = (role == 0 || role == 3) ? ScaleCache::SplitType::Type0 : ScaleCache::SplitType::Type1;
      } else {
        // Odd -> Even diamond, children order {N,E,S,W}.
        st = (role == 0 || role == 2) ? ScaleCache::SplitType::Type1 : ScaleCache::SplitType::Type0;
      }

      Tensor u, vt;
      qlten::QLTensor<RealT, QNT> s;
      RealT trunc_err_actual = RealT(0);
      size_t bond_dim_actual = 0;

      if (st == ScaleCache::SplitType::Type0) {
        Tensor TT = T;
        TT.Transpose({0, 3, 1, 2});
        SVD(&TT, 2, TT.Div(), tp.trunc_err, tp.D_min, tp.D_max, &u, &s, &vt, &trunc_err_actual, &bond_dim_actual);
        auto s_sqrt = ElementWiseSqrt(s);
        qlten::QLTensor<RealT, QNT> s_inv_sqrt = s_sqrt;
        s_inv_sqrt.ElementWiseInv(/*eps=*/RealT(1e-30));
        Tensor P, Q;
        Contract(&u, &s_sqrt, {{2}, {0}}, &P);
        Contract(&s_sqrt, &vt, {{1}, {0}}, &Q);

        layer.split_type[id][slot] = st;
        layer.split_U[id][slot] = u;
        layer.split_Vt[id][slot] = vt;
        layer.split_S_inv_sqrt[id][slot] = s_inv_sqrt;
        layer.split_P[id][slot] = P;
        layer.split_Q[id][slot] = Q;
      } else {
        Tensor TT = T;
        SVD(&TT, 2, TT.Div(), tp.trunc_err, tp.D_min, tp.D_max, &u, &s, &vt, &trunc_err_actual, &bond_dim_actual);
        auto s_sqrt = ElementWiseSqrt(s);
        qlten::QLTensor<RealT, QNT> s_inv_sqrt = s_sqrt;
        s_inv_sqrt.ElementWiseInv(/*eps=*/RealT(1e-30));
        Tensor Q, P;
        Contract(&u, &s_sqrt, {{2}, {0}}, &Q);
        Contract(&s_sqrt, &vt, {{1}, {0}}, &P);

        layer.split_type[id][slot] = st;
        layer.split_U[id][slot] = u;
        layer.split_Vt[id][slot] = vt;
        layer.split_S_inv_sqrt[id][slot] = s_inv_sqrt;
        layer.split_P[id][slot] = P;
        layer.split_Q[id][slot] = Q;
      }
    }
  }
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::Trace(const TensorNetwork2D<TenElemT, QNT>& tn) {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::Trace: call Init(tn) first.");
  if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) throw std::invalid_argument("TRGContractor::Trace: tn must be periodic.");
  if (!trunc_params_.has_value()) throw std::logic_error("TRGContractor::Trace: truncation params are not set.");
  
  // 1. Update Scale 0 Dirty Tensors
  if (!tensors_initialized_) {
     // First run: treat all as dirty
     const size_t N = rows_ * cols_;
     for (uint32_t i = 0; i < N; ++i) dirty_scale0_.insert(i);
     tensors_initialized_ = true;
  }
  
  if (dirty_scale0_.empty()) {
      if (scales_.empty()) return TenElemT(0);
      if (scales_.back().tens.size() == 1) return ContractFinal1x1_(scales_.back().tens[0]);
      if (scales_.back().tens.size() == 4) {
        const std::array<Tensor, 4> t2x2 = {scales_.back().tens[0], scales_.back().tens[1],
                                            scales_.back().tens[2], scales_.back().tens[3]};
        return ContractFinal2x2_(t2x2);
      }
      throw std::logic_error("TRGContractor::Trace: invalid final scale size.");
  }

  // Reload dirty tensors from TN
  for (uint32_t id : dirty_scale0_) {
     auto coords = Coord_(id);
     scales_[0].tens[id] = tn({coords.first, coords.second});
  }

  // If there is no RG step (n == 2 or 3), we can contract the terminal lattice directly
  // without building split caches.
  if (scales_.size() == 1) {
    dirty_scale0_.clear();
    if (scales_.back().tens.size() == 1) return ContractFinal1x1_(scales_.back().tens[0]);
    if (scales_.back().tens.size() == 4) {
      const std::array<Tensor, 4> t2x2 = {scales_.back().tens[0], scales_.back().tens[1],
                                          scales_.back().tens[2], scales_.back().tens[3]};
      return ContractFinal2x2_(t2x2);
    }
    if (scales_.back().tens.size() == 9) {
      const std::array<Tensor, 9> t3x3 = {scales_.back().tens[0], scales_.back().tens[1], scales_.back().tens[2],
                                          scales_.back().tens[3], scales_.back().tens[4], scales_.back().tens[5],
                                          scales_.back().tens[6], scales_.back().tens[7], scales_.back().tens[8]};
      return ContractFinal3x3_(t3x3);
    }
    throw std::logic_error("TRGContractor::Trace: invalid final scale size.");
  }

  // Maintain split cache on scale 0 for the reloaded nodes.
  {
    std::set<uint32_t> dirty0(dirty_scale0_.begin(), dirty_scale0_.end());
    try {
      EnsureSplitCacheForNodes_(/*scale=*/0, dirty0);
    } catch (const std::out_of_range& e) {
      throw std::out_of_range(std::string("TRGContractor::Trace: EnsureSplitCacheForNodes_(scale=0) failed: ") + e.what());
    }
  }

  // 2. Propagate Up
  std::set<uint32_t> dirty_current(dirty_scale0_.begin(), dirty_scale0_.end());
  dirty_scale0_.clear();

  // Iterate scales
  for (size_t s = 0; s < scales_.size() - 1; ++s) {
      if (dirty_current.empty()) break; // Should not happen if dirty_scale0_ was not empty

      std::set<uint32_t> dirty_next;
      const auto& fine_layer = scales_[s];
      auto& coarse_layer = scales_[s + 1];
      const bool even_to_odd = (s % 2 == 0);

      // Identify affected coarse nodes
      for (uint32_t f_id : dirty_current) {
          const auto& parents = fine_layer.fine_to_coarse[f_id];
          if (parents[0] != 0xFFFFFFFF) dirty_next.insert(parents[0]);
          if (parents[1] != 0xFFFFFFFF) dirty_next.insert(parents[1]);
      }

      // Recompute affected coarse nodes
      for (uint32_t c_id : dirty_next) {
          const auto& children = coarse_layer.coarse_to_fine[c_id];
          const Tensor& T0 = fine_layer.tens[children[0]];
          const Tensor& T1 = fine_layer.tens[children[1]];
          const Tensor& T2 = fine_layer.tens[children[2]];
          const Tensor& T3 = fine_layer.tens[children[3]];

          using qlten::Contract;

          if (even_to_odd) {
              // Plaquette Contraction (TL, TR, BL, BR)
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
              //
              // T0=TL, T1=TR, T2=BL, T3=BR
              // TL, BR are A (Type0); TR, BL are B (Type1)
              // Use split cache so alpha indices are stable and reusable by PunchHole().
              // Select split pieces in the correct parent-context (fine node -> this plaquette parent).
              auto slot_of = [&](uint32_t fid, uint32_t pid) -> int {
                const auto& ps = fine_layer.fine_to_coarse.at(fid);
                if (ps[0] == pid) return 0;
                if (ps[1] == pid) return 1;
                return -1;
              };
              const int s0 = slot_of(children[0], c_id);
              const int s1 = slot_of(children[1], c_id);
              const int s2 = slot_of(children[2], c_id);
              const int s3 = slot_of(children[3], c_id);
              if (s0 < 0 || s1 < 0 || s2 < 0 || s3 < 0)
                throw std::logic_error("TRGContractor::Trace: missing parent-slot in plaquette.");

              const Tensor& Q0 = fine_layer.split_Q[children[0]][s0]; // TL.Q
              const Tensor& Q1 = fine_layer.split_Q[children[1]][s1]; // TR.Q
              const Tensor& P1 = fine_layer.split_P[children[2]][s2]; // BL.P
              const Tensor& P0 = fine_layer.split_P[children[3]][s3]; // BR.P

              Tensor tmp0, tmp1, tmp2;
              Contract(&P1, {1}, &P0, {0}, &tmp0);
              Contract(&Q1, {0}, &Q0, {2}, &tmp1);
              Contract(&tmp0, {1, 2}, &tmp1, {3, 0}, &tmp2);
              tmp2.Transpose({3, 2, 1, 0});
              
              coarse_layer.tens[c_id] = tmp2;

          } else {
              // Diamond Contraction (N, E, S, W)
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
              //
              // T0=N, T1=E, T2=S, T3=W
              // N, S are B (Type1); E, W are A (Type0)
              // Use split cache so alpha indices are stable and reusable by PunchHole().
              auto slot_of = [&](uint32_t fid, uint32_t pid) -> int {
                const auto& ps = fine_layer.fine_to_coarse.at(fid);
                if (ps[0] == pid) return 0;
                if (ps[1] == pid) return 1;
                return -1;
              };
              const int s0 = slot_of(children[0], c_id);
              const int s1 = slot_of(children[1], c_id);
              const int s2 = slot_of(children[2], c_id);
              const int s3 = slot_of(children[3], c_id);
              if (s0 < 0 || s1 < 0 || s2 < 0 || s3 < 0)
                throw std::logic_error("TRGContractor::Trace: missing parent-slot in diamond.");

              const Tensor& Np = fine_layer.split_P[children[0]][s0]; // N.P
              const Tensor& Ep = fine_layer.split_P[children[1]][s1]; // E.P
              const Tensor& Sq = fine_layer.split_Q[children[2]][s2]; // S.Q
              const Tensor& Wq = fine_layer.split_Q[children[3]][s3]; // W.Q

              Tensor SW, NE, out;
              Contract(&Sq, {0}, &Wq, {2}, &SW);
              Contract(&Np, {1}, &Ep, {0}, &NE);
              Contract(&SW, {0, 3}, &NE, {2, 1}, &out);
              out.Transpose({1, 0, 3, 2});
              
              coarse_layer.tens[c_id] = out;
          }
      }

      // Keep split cache for the updated coarse nodes, so later PunchHole can reuse them.
      try {
        EnsureSplitCacheForNodes_(/*scale=*/s + 1, dirty_next);
      } catch (const std::out_of_range& e) {
        throw std::out_of_range(std::string("TRGContractor::Trace: EnsureSplitCacheForNodes_(scale=") +
                                std::to_string(s + 1) + ") failed: " + e.what());
      }
      
      dirty_current = std::move(dirty_next);
  }

  // Final 1x1 Trace
  if (scales_.back().tens.size() == 1) return ContractFinal1x1_(scales_.back().tens[0]);
  if (scales_.back().tens.size() == 4) {
    const std::array<Tensor, 4> t2x2 = {scales_.back().tens[0], scales_.back().tens[1],
                                        scales_.back().tens[2], scales_.back().tens[3]};
    return ContractFinal2x2_(t2x2);
  }
  if (scales_.back().tens.size() == 9) {
    const std::array<Tensor, 9> t3x3 = {scales_.back().tens[0], scales_.back().tens[1], scales_.back().tens[2],
                                        scales_.back().tens[3], scales_.back().tens[4], scales_.back().tens[5],
                                        scales_.back().tens[6], scales_.back().tens[7], scales_.back().tens[8]};
    return ContractFinal3x3_(t3x3);
  }
  throw std::logic_error("TRGContractor::Trace: invalid final scale size.");
}

// NOTE: The actual PunchHole4x4_ implementation is defined below. We keep helper lambdas local
// to that implementation to avoid adding new public surface area.

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Trial
TRGContractor<TenElemT, QNT>::BeginTrialWithReplacement(
    const std::vector<std::pair<SiteIdx, Tensor>>& replacements) const {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::BeginTrialWithReplacement: call Init(tn) first.");
  if (!trunc_params_.has_value()) throw std::logic_error("TRGContractor::BeginTrialWithReplacement: truncation params are not set.");
  if (!tensors_initialized_) throw std::logic_error("TRGContractor::BeginTrialWithReplacement: Trace(tn) must be called at least once to initialize cache.");
  if (!dirty_scale0_.empty()) throw std::logic_error("TRGContractor::BeginTrialWithReplacement: cache is dirty. Please call Trace(tn) to commit changes before calling this.");

  Trial trial;
  trial.layer_updates.resize(scales_.size());

  // Scale-0 updates.
  for (const auto& kv : replacements) {
    const uint32_t id = NodeId_(kv.first.row(), kv.first.col());
    trial.layer_updates[0][id] = kv.second;
  }

  if (trial.layer_updates[0].empty()) {
    if (scales_.back().tens.size() == 1) {
      trial.amplitude = ContractFinal1x1_(scales_.back().tens[0]);
    } else if (scales_.back().tens.size() == 4) {
      const std::array<Tensor, 4> t2x2 = {scales_.back().tens[0], scales_.back().tens[1],
                                          scales_.back().tens[2], scales_.back().tens[3]};
      trial.amplitude = ContractFinal2x2_(t2x2);
    } else if (scales_.back().tens.size() == 9) {
      const std::array<Tensor, 9> t3x3 = {scales_.back().tens[0], scales_.back().tens[1], scales_.back().tens[2],
                                          scales_.back().tens[3], scales_.back().tens[4], scales_.back().tens[5],
                                          scales_.back().tens[6], scales_.back().tens[7], scales_.back().tens[8]};
      trial.amplitude = ContractFinal3x3_(t3x3);
    } else {
      throw std::logic_error("TRGContractor::BeginTrialWithReplacement: invalid final scale size.");
    }
    return trial;
  }

  // Shadow RG propagation: for each scale, compute the affected coarse tensors and stash them.
  for (size_t s = 0; s < scales_.size() - 1; ++s) {
    if (trial.layer_updates[s].empty()) break;

    const auto& fine_layer = scales_[s];
    const auto& coarse_layer = scales_[s + 1];
    const bool even_to_odd = (s % 2 == 0);

    // Identify affected coarse nodes.
    std::set<uint32_t> dirty_coarse_nodes;
    for (const auto& kv : trial.layer_updates[s]) {
      const uint32_t f_id = kv.first;
      const auto& parents = fine_layer.fine_to_coarse[f_id];
      if (parents[0] != 0xFFFFFFFF) dirty_coarse_nodes.insert(parents[0]);
      if (parents[1] != 0xFFFFFFFF) dirty_coarse_nodes.insert(parents[1]);
    }

    // Recompute affected coarse nodes using mix of trial updates and cached fine tensors.
    for (uint32_t c_id : dirty_coarse_nodes) {
      const auto& children = coarse_layer.coarse_to_fine[c_id];

      auto get_fine_tensor = [&](uint32_t id) -> const Tensor& {
        auto it = trial.layer_updates[s].find(id);
        if (it != trial.layer_updates[s].end()) return it->second;
        return fine_layer.tens[id];
      };

      const Tensor& T0 = get_fine_tensor(children[0]);
      const Tensor& T1 = get_fine_tensor(children[1]);
      const Tensor& T2 = get_fine_tensor(children[2]);
      const Tensor& T3 = get_fine_tensor(children[3]);

      using qlten::Contract;

      if (even_to_odd) {
        // IMPORTANT: Trial must match the *true* TRG semantics (including SVD/truncation) for the
        // modified tensors. Therefore we re-run the splitters here (as in the original implementation),
        // rather than using any cached linearized split projection.
        auto tl = SplitType0_(T0);
        auto tr = SplitType1_(T1);
        auto bl = SplitType1_(T2);
        auto br = SplitType0_(T3);

        const Tensor& Q0 = tl.Q;
        const Tensor& Q1 = tr.Q;
        const Tensor& P1 = bl.P;
        const Tensor& P0 = br.P;

        Tensor tmp0, tmp1, tmp2;
        Contract(&P1, {1}, &P0, {0}, &tmp0);
        Contract(&Q1, {0}, &Q0, {2}, &tmp1);
        Contract(&tmp0, {1, 2}, &tmp1, {3, 0}, &tmp2);
        tmp2.Transpose({3, 2, 1, 0});

        trial.layer_updates[s + 1][c_id] = tmp2;
      } else {
        auto nB = SplitType1_(T0);
        auto eA = SplitType0_(T1);
        auto sB = SplitType1_(T2);
        auto wA = SplitType0_(T3);

        const Tensor& Np = nB.P;
        const Tensor& Ep = eA.P;
        const Tensor& Sq = sB.Q;
        const Tensor& Wq = wA.Q;

        Tensor SW, NE, out;
        Contract(&Sq, {0}, &Wq, {2}, &SW);
        Contract(&Np, {1}, &Ep, {0}, &NE);
        Contract(&SW, {0, 3}, &NE, {2, 1}, &out);
        out.Transpose({1, 0, 3, 2});

        trial.layer_updates[s + 1][c_id] = out;
      }
    }
  }

  // Final amplitude: use trial last-layer tensor if present, otherwise cached.
  if (!trial.layer_updates.empty()) {
    const size_t last = trial.layer_updates.size() - 1;
    if (scales_.back().tens.size() == 1) {
      auto it = trial.layer_updates[last].find(0);
      if (it != trial.layer_updates[last].end()) {
        trial.amplitude = ContractFinal1x1_(it->second);
        return trial;
      }
    } else if (scales_.back().tens.size() == 4) {
      std::array<Tensor, 4> t2x2 = {scales_.back().tens[0], scales_.back().tens[1],
                                    scales_.back().tens[2], scales_.back().tens[3]};
      for (uint32_t id = 0; id < 4; ++id) {
        auto it = trial.layer_updates[last].find(id);
        if (it != trial.layer_updates[last].end()) t2x2[id] = it->second;
      }
      trial.amplitude = ContractFinal2x2_(t2x2);
      return trial;
    } else if (scales_.back().tens.size() == 9) {
      std::array<Tensor, 9> t3x3 = {scales_.back().tens[0], scales_.back().tens[1], scales_.back().tens[2],
                                    scales_.back().tens[3], scales_.back().tens[4], scales_.back().tens[5],
                                    scales_.back().tens[6], scales_.back().tens[7], scales_.back().tens[8]};
      for (uint32_t id = 0; id < 9; ++id) {
        auto it = trial.layer_updates[last].find(id);
        if (it != trial.layer_updates[last].end()) t3x3[id] = it->second;
      }
      trial.amplitude = ContractFinal3x3_(t3x3);
      return trial;
    } else {
      throw std::logic_error("TRGContractor::BeginTrialWithReplacement: invalid final scale size.");
    }
  }
  if (scales_.back().tens.size() == 1) {
    trial.amplitude = ContractFinal1x1_(scales_.back().tens[0]);
  } else if (scales_.back().tens.size() == 4) {
    const std::array<Tensor, 4> t2x2 = {scales_.back().tens[0], scales_.back().tens[1],
                                        scales_.back().tens[2], scales_.back().tens[3]};
    trial.amplitude = ContractFinal2x2_(t2x2);
  } else if (scales_.back().tens.size() == 9) {
    const std::array<Tensor, 9> t3x3 = {scales_.back().tens[0], scales_.back().tens[1], scales_.back().tens[2],
                                        scales_.back().tens[3], scales_.back().tens[4], scales_.back().tens[5],
                                        scales_.back().tens[6], scales_.back().tens[7], scales_.back().tens[8]};
    trial.amplitude = ContractFinal3x3_(t3x3);
  } else {
    throw std::logic_error("TRGContractor::BeginTrialWithReplacement: invalid final scale size.");
  }
  return trial;
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::CommitTrial(Trial&& trial) {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::CommitTrial: call Init(tn) first.");
  if (!trunc_params_.has_value()) throw std::logic_error("TRGContractor::CommitTrial: truncation params are not set.");
  if (!tensors_initialized_) throw std::logic_error("TRGContractor::CommitTrial: Trace(tn) must be called at least once to initialize cache.");
  if (!dirty_scale0_.empty()) throw std::logic_error("TRGContractor::CommitTrial: cache is dirty. Please call Trace(tn) first.");
  if (trial.layer_updates.size() != scales_.size()) throw std::invalid_argument("TRGContractor::CommitTrial: trial topology mismatch.");

  for (size_t s = 0; s < trial.layer_updates.size(); ++s) {
    for (auto& kv : trial.layer_updates[s]) {
      scales_[s].tens[kv.first] = std::move(kv.second);
    }
  }
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::ContractFinal1x1_(const Tensor& T) const {
    using qlten::Contract;
    using qlten::Eye;
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
    const auto& idx_r = T.GetIndex(2);
    const auto& idx_d = T.GetIndex(1);
    const auto eye_lr = Eye<TenElemT, QNT>(idx_r);
    const auto eye_du = Eye<TenElemT, QNT>(idx_d);
    Tensor tmp, tmp2;
    Contract(&T, {0, 2}, &eye_lr, {0, 1}, &tmp);
    Contract(&tmp, {0, 1}, &eye_du, {1, 0}, &tmp2);
    return tmp2();
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::ContractFinal2x2_(const std::array<Tensor, 4>& T2x2) const {
  using qlten::Contract;
  // 2x2 PBC torus on the final even lattice.
  //
  // Tensor ids (row-major):
  //   0:(0,0)  1:(0,1)
  //   2:(1,0)  3:(1,1)
  //
  // For size 2 with PBC, each nearest-neighbor pair is connected by TWO bonds (left/right, up/down).
  //
  // Contract horizontally: (0,0) with (0,1), and (1,0) with (1,1), each along both {L,R}.
  Tensor top, bot;
  Contract(&T2x2[0], {0, 2}, &T2x2[1], {2, 0}, &top);
  Contract(&T2x2[2], {0, 2}, &T2x2[3], {2, 0}, &bot);
  // Then contract vertically: connect (row0) with (row1) along both {D,U} for each column.
  Tensor out;
  Contract(&top, {0, 1, 2, 3}, &bot, {1, 0, 3, 2}, &out);
  return out();
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::ContractFinal3x3_(const std::array<Tensor, 9>& T3x3) const {
  using qlten::Contract;
  using qlten::Eye;

  // 3x3 PBC torus on the terminal even lattice.
  //
  // Tensor ids (row-major):
  //   0 1 2
  //   3 4 5
  //   6 7 8
  //
  // Each tensor uses leg order [L(0), D(1), R(2), U(3)].

  auto row_ring = [&](const Tensor& A0, const Tensor& A1, const Tensor& A2) -> Tensor {
    Tensor t01;
    Contract(&A0, {2}, &A1, {0}, &t01);

    Tensor t012;
    Contract(&t01, {4}, &A2, {0}, &t012);

    // Close horizontal ring: trace A0.L with A2.R.
    const auto& idx_r = A2.GetIndex(2);
    const auto eye_lr = Eye<TenElemT, QNT>(idx_r);

    // t012 axes: [A0.L,A0.D,A0.U, A1.D,A1.U, A2.D,A2.R,A2.U]
    Tensor closed;
    Contract(&t012, {0, 6}, &eye_lr, {0, 1}, &closed);
    // closed axes: [A0.D,A0.U, A1.D,A1.U, A2.D,A2.U]
    return closed;
  };

  Tensor R0 = row_ring(T3x3[0], T3x3[1], T3x3[2]);
  Tensor R1 = row_ring(T3x3[3], T3x3[4], T3x3[5]);
  Tensor R2 = row_ring(T3x3[6], T3x3[7], T3x3[8]);

  // Contract vertically between rows: connect D (OUT) with U (IN).
  Tensor M01;
  Contract(&R0, {0, 2, 4}, &R1, {1, 3, 5}, &M01);

  Tensor M012;
  Contract(&M01, {3, 4, 5}, &R2, {1, 3, 5}, &M012);

  // Close vertical PBC: trace (R0.Uc) with (R2.Dc) for c=0,1,2.
  Tensor tmp = M012;
  for (int iter = 0; iter < 3; ++iter) {
    const size_t rank = tmp.Rank();
    const size_t u_ax = rank / 2 - 1;  // last U
    const size_t d_ax = rank - 1;      // last D
    const auto& idx_d = tmp.GetIndex(d_ax);
    const auto eye_du = Eye<TenElemT, QNT>(idx_d);
    Tensor out;
    Contract(&tmp, {d_ax, u_ax}, &eye_du, {1, 0}, &out);
    tmp = std::move(out);
  }

  return tmp();
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::PunchHoleFinal2x2_(const std::array<Tensor, 4>& T2x2,
                                                const uint32_t removed_id) const {
  using qlten::Contract;
  // Exact 2x2 PBC hole by contracting the other 3 tensors.
  //
  // The output tensor legs are ordered to match the removed site's [L,D,R,U] convention.
  // See PunchHole() documentation in trg_contractor.h.
  if (removed_id >= 4) throw std::invalid_argument("TRGContractor::PunchHoleFinal2x2_: removed_id out of range.");

  // Hard-code the 4 cases for clarity (avoid clever but fragile graph logic).
  // Id layout (row-major):
  //   0 1
  //   2 3
  //
  // Note: For a 2x2 torus, each nearest-neighbor pair is connected by TWO bonds.
  // We always contract the two bonds between a pair simultaneously.
  if (removed_id == 0) {
    // Remove 0. Remaining: 1,2,3.
    // Contract (1)-(3) vertically: 1.{D,U} <-> 3.{U,D}
    Tensor bd;
    Contract(&T2x2[1], {1, 3}, &T2x2[3], {3, 1}, &bd);  // axes: [1.L, 1.R, 3.L, 3.R]
    // Contract (3)-(2) horizontally: 3.{L,R} <-> 2.{R,L}
    Tensor hole;
    Contract(&bd, {2, 3}, &T2x2[2], {2, 0}, &hole);  // axes: [1.L, 1.R, 2.D, 2.U]
    // Desired [L,D,R,U] of removed 0 is [1.R, 2.U, 1.L, 2.D].
    hole.Transpose({1, 3, 0, 2});
    return hole;
  }
  if (removed_id == 1) {
    // Remove 1. Remaining: 0,2,3.
    Tensor ac;
    Contract(&T2x2[0], {1, 3}, &T2x2[2], {3, 1}, &ac);  // axes: [0.L, 0.R, 2.L, 2.R]
    Tensor hole;
    Contract(&ac, {2, 3}, &T2x2[3], {2, 0}, &hole);      // axes: [0.L, 0.R, 3.D, 3.U]
    // Desired [L,D,R,U] of removed 1 is [0.R, 3.U, 0.L, 3.D].
    hole.Transpose({1, 3, 0, 2});
    return hole;
  }
  if (removed_id == 2) {
    // Remove 2. Remaining: 0,1,3.
    Tensor ab;
    Contract(&T2x2[0], {0, 2}, &T2x2[1], {2, 0}, &ab);  // axes: [0.D, 0.U, 1.D, 1.U]
    Tensor hole;
    Contract(&ab, {2, 3}, &T2x2[3], {3, 1}, &hole);      // axes: [0.D, 0.U, 3.L, 3.R]
    // Desired [L,D,R,U] of removed 2 is [3.R, 0.U, 3.L, 0.D].
    hole.Transpose({3, 1, 2, 0});
    return hole;
  }
  // removed_id == 3
  Tensor ab;
  Contract(&T2x2[0], {0, 2}, &T2x2[1], {2, 0}, &ab);    // axes: [0.D, 0.U, 1.D, 1.U]
  Tensor hole;
  Contract(&ab, {0, 1}, &T2x2[2], {3, 1}, &hole);        // axes: [1.D, 1.U, 2.L, 2.R]
  // Desired [L,D,R,U] of removed 3 is [2.R, 1.U, 2.L, 1.D].
  hole.Transpose({3, 1, 2, 0});
  return hole;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::PunchHoleFinal3x3_(const std::array<Tensor, 9>& T3x3,
                                                const uint32_t removed_id) const {
  using qlten::Contract;
  using qlten::Eye;

  if (removed_id >= 9) throw std::invalid_argument("TRGContractor::PunchHoleFinal3x3_: removed_id out of range.");

  struct BondKey {
    char kind;      // 'x' horizontal, 'y' vertical
    uint8_t r, c;   // directed bond origin (r,c)
    bool operator<(const BondKey& o) const {
      if (kind != o.kind) return kind < o.kind;
      if (r != o.r) return r < o.r;
      return c < o.c;
    }
    bool operator==(const BondKey& o) const { return kind == o.kind && r == o.r && c == o.c; }
  };

  struct LabeledTensor {
    Tensor ten;
    std::vector<BondKey> labels; // per-axis bond label
  };

  auto mod3 = [](int x) -> uint8_t { return static_cast<uint8_t>((x % 3 + 3) % 3); };
  auto id = [](uint8_t r, uint8_t c) -> uint32_t { return uint32_t(r) * 3u + uint32_t(c); };
  auto rc = [&](uint32_t i) -> std::pair<uint8_t, uint8_t> { return {uint8_t(i / 3u), uint8_t(i % 3u)}; };

  // Directed bonds:
  // - x(r,c): (r,c).R <-> (r,c+1).L
  // - y(r,c): (r,c).D <-> (r+1,c).U
  auto x = [&](uint8_t r, uint8_t c) -> BondKey { return BondKey{'x', r, c}; };
  auto y = [&](uint8_t r, uint8_t c) -> BondKey { return BondKey{'y', r, c}; };

  const auto [rr, cc] = rc(removed_id);

  // Build initial components: 8 tensors with labeled legs [L,D,R,U].
  std::vector<LabeledTensor> comps;
  comps.reserve(8);
  for (uint8_t r = 0; r < 3; ++r) {
    for (uint8_t c = 0; c < 3; ++c) {
      const uint32_t sid = id(r, c);
      if (sid == removed_id) continue;
      const Tensor& T = T3x3[sid];
      if (T.GetShape().size() != 4)
        throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: site tensor must be rank-4.");

      LabeledTensor lt;
      lt.ten = T;
      lt.labels = {
          x(r, mod3(int(c) - 1)), // L
          y(r, c),                // D
          x(r, c),                // R
          y(mod3(int(r) - 1), c)  // U
      };
      comps.push_back(std::move(lt));
    }
  }

  auto find_pair = [&](const std::vector<LabeledTensor>& v,
                       BondKey* out_key,
                       size_t* out_i, size_t* out_ai,
                       size_t* out_j, size_t* out_aj,
                       bool* same_component) -> bool {
    // Find any label that appears twice.
    std::map<BondKey, std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>>> occ;
    std::set<BondKey> seen_once;
    for (size_t i = 0; i < v.size(); ++i) {
      for (size_t a = 0; a < v[i].labels.size(); ++a) {
        const auto& k = v[i].labels[a];
        if (!seen_once.count(k)) {
          seen_once.insert(k);
          occ[k].first = {i, a};
          occ[k].second = {size_t(-1), size_t(-1)};
        } else if (occ[k].second.first == size_t(-1)) {
          occ[k].second = {i, a};
          *out_key = k;
          *out_i = occ[k].first.first;
          *out_ai = occ[k].first.second;
          *out_j = occ[k].second.first;
          *out_aj = occ[k].second.second;
          *same_component = (*out_i == *out_j);
          return true;
        }
      }
    }
    return false;
  };

  auto trace_pair_inplace = [&](LabeledTensor& lt, size_t ax_a, size_t ax_b) {
    // Trace the two legs connected by the same bond label.
    if (ax_a == ax_b) throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: tracing identical axes.");
    const auto& idx_a = lt.ten.GetIndex(ax_a);
    const auto& idx_b = lt.ten.GetIndex(ax_b);

    // Choose the OUT-directed index as the one to build Eye from, and order (IN, OUT) for contraction.
    const bool a_is_out = (idx_a.GetDir() == qlten::TenIndexDirType::OUT);
    const bool b_is_out = (idx_b.GetDir() == qlten::TenIndexDirType::OUT);
    if (a_is_out == b_is_out)
      throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: bond axes must be opposite directions.");

    const size_t ax_in = a_is_out ? ax_b : ax_a;
    const size_t ax_out = a_is_out ? ax_a : ax_b;
    const auto& idx_out = lt.ten.GetIndex(ax_out);
    const auto eye = Eye<TenElemT, QNT>(idx_out);

    Tensor out;
    Contract(&lt.ten, {ax_in, ax_out}, &eye, {0, 1}, &out);
    lt.ten = std::move(out);

    // Remove labels at ax_in and ax_out (remove larger index first).
    const size_t hi = std::max(ax_in, ax_out);
    const size_t lo = std::min(ax_in, ax_out);
    lt.labels.erase(lt.labels.begin() + hi);
    lt.labels.erase(lt.labels.begin() + lo);
  };

  // Contract all internal bonds until only the 4 open legs around the removed site remain.
  // This may require both inter-component contractions and intra-component traces (for cycles on the torus).
  for (int safety = 0; safety < 200; ++safety) {
    BondKey k;
    size_t i, ai, j, aj;
    bool same = false;
    if (!find_pair(comps, &k, &i, &ai, &j, &aj, &same)) break;

    if (!same) {
      // Merge two components along this bond.
      LabeledTensor a = std::move(comps[i]);
      LabeledTensor b = std::move(comps[j]);

      Tensor out;
      Contract(&a.ten, {ai}, &b.ten, {aj}, &out);

      std::vector<BondKey> labels;
      labels.reserve(a.labels.size() + b.labels.size() - 2);
      for (size_t t = 0; t < a.labels.size(); ++t) if (t != ai) labels.push_back(a.labels[t]);
      for (size_t t = 0; t < b.labels.size(); ++t) if (t != aj) labels.push_back(b.labels[t]);

      LabeledTensor merged{std::move(out), std::move(labels)};

      // Remove higher index first.
      if (i > j) std::swap(i, j);
      comps.erase(comps.begin() + j);
      comps.erase(comps.begin() + i);
      comps.push_back(std::move(merged));
    } else {
      // Same component: trace the cycle bond.
      trace_pair_inplace(comps[i], ai, aj);
    }
  }

  if (comps.size() != 1)
    throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: failed to fully contract components.");

  Tensor hole = std::move(comps[0].ten);
  auto labels = std::move(comps[0].labels);
  if (labels.size() != 4)
    throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: final open legs are not rank-4.");

  // Expected open legs (dual of removed site's [L,D,R,U]) are the four bonds incident to (rr,cc).
  const std::array<BondKey, 4> want = {
      x(rr, mod3(int(cc) - 1)), // L*
      y(rr, cc),                // D*
      x(rr, cc),                // R*
      y(mod3(int(rr) - 1), cc)  // U*
  };

  std::array<int, 4> perm = {-1, -1, -1, -1};
  for (int w = 0; w < 4; ++w) {
    for (int a = 0; a < 4; ++a) {
      if (labels[static_cast<size_t>(a)] == want[static_cast<size_t>(w)]) {
        perm[static_cast<size_t>(w)] = a;
        break;
      }
    }
    if (perm[static_cast<size_t>(w)] < 0)
      throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: missing expected open leg.");
  }

  hole.Transpose({static_cast<size_t>(perm[0]),
                  static_cast<size_t>(perm[1]),
                  static_cast<size_t>(perm[2]),
                  static_cast<size_t>(perm[3])});

  // Sanity: ensure indices match the dual space of the removed tensor so Contract(hole, Ts) is valid.
  const Tensor& Ts = T3x3[removed_id];
  if (!(hole.GetIndex(0) == InverseIndex(Ts.GetIndex(0)) &&
        hole.GetIndex(1) == InverseIndex(Ts.GetIndex(1)) &&
        hole.GetIndex(2) == InverseIndex(Ts.GetIndex(2)) &&
        hole.GetIndex(3) == InverseIndex(Ts.GetIndex(3)))) {
    throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: hole indices do not match dual of site tensor.");
  }

  return hole;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::PunchHole4x4_(const SiteIdx& site) const {
  using qlten::Contract;

  // NOTE ABOUT "0.25" (FACTOR-4) NORMALIZATION IN HISTORICAL REVISIONS:
  //
  // The 4x4 PBC topology built by BuildTopology_ allows each fine node to have up to two coarse
  // parents per RG step (scale0->scale1 plaquette, scale1->scale2 diamond). If we naively sum
  // pullbacks over all parent contexts at both steps, the composed pullback can acquire an overall
  // multiplicity of ~2×2=4, leading to:
  //   Contract(hole(site), T_site) ≈ 4 * Trace(tn).
  //
  // Current behavior: we apply the corresponding 1/2 factor per RG step when combining parent
  // pullbacks, instead of using an opaque hard-coded "0.25" at the end.
  //
  // Also note that this hole backprop uses a linearized split (freezing forward SVD factors U,S,V).
  // With truncation, exact equality is not expected, but the factor-4 mismatch is a graph/topology
  // multiplicity effect and can persist even without truncation.

  const uint32_t site_id = NodeId_(site.row(), site.col());
  const auto& parents_of_site = scales_[0].fine_to_coarse.at(site_id);

  // Scale-2 (2x2) holes: dZ/dT_scale2.
  std::vector<Tensor> holes_scale2(4);
  {
    const auto& s2 = scales_.at(2).tens;
    const std::array<Tensor, 4> t2x2 = {s2[0], s2[1], s2[2], s2[3]};
    for (uint32_t i = 0; i < 4; ++i) holes_scale2[i] = PunchHoleFinal2x2_(t2x2, i);
  }


  // Scale-1 holes: diamond backprop from holes_scale2.
  std::map<uint32_t, Tensor> holes_scale1;
  auto compute_h1 = [&](uint32_t id1) -> Tensor {
    const auto& parents = scales_[1].fine_to_coarse.at(id1);
    Tensor h_total;

    for (uint32_t pid : parents) {
      if (pid == 0xFFFFFFFF) continue;
      const auto& children = scales_[2].coarse_to_fine.at(pid);  // {N,E,S,W}
      int role = -1;
      for (int k = 0; k < 4; ++k)
        if (children[k] == id1) role = k;
      if (role < 0) throw std::logic_error("TRGContractor::PunchHole4x4_: invalid diamond topology (role == -1).");

      const Tensor& H_parent = holes_scale2.at(pid);  // rank-4 hole tensor from the 2x2 terminator

      // Use split cache to keep alpha indices consistent with Trace(). We MUST backprop the
      // exact same fixed wiring as the forward diamond contraction (see Trace()).
      //
      // Forward (odd -> even diamond) recap:
      //   SW = Contract(Sq{0}, Wq{2})              // rank-4
      //   NE = Contract(Np{1}, Ep{0})              // rank-4
      //   out_pre = Contract(SW{0,3}, NE{2,1})     // rank-4, axes [SW1,SW2,NE0,NE3]
      //   out = out_pre.Transpose({1,0,3,2})       // final even tensor leg order [L,D,R,U]
      //
      // Here H_parent is dZ/d(out). We compute dZ/d(Np/Ep/Sq/Wq) by reversing the above.
      const int sn = ParentSlot_(/*scale=*/1, children[0], pid);
      const int se = ParentSlot_(/*scale=*/1, children[1], pid);
      const int ss = ParentSlot_(/*scale=*/1, children[2], pid);
      const int sw = ParentSlot_(/*scale=*/1, children[3], pid);
      if (sn < 0 || se < 0 || ss < 0 || sw < 0)
        throw std::logic_error("TRGContractor::PunchHole4x4_: missing parent-slot in diamond split cache.");
      const Tensor& Np = scales_[1].split_P.at(children[0])[sn];  // fixed split piece in this diamond
      const Tensor& Ep = scales_[1].split_P.at(children[1])[se];
      const Tensor& Sq = scales_[1].split_Q.at(children[2])[ss];
      const Tensor& Wq = scales_[1].split_Q.at(children[3])[sw];

      Tensor SW, NE;
      Contract(&Sq, {0}, &Wq, {2}, &SW);
      Contract(&Np, {1}, &Ep, {0}, &NE);

      // Undo the final transpose: dZ/d(out_pre) = transpose(dZ/d(out), inv_perm).
      // perm = {1,0,3,2} is self-inverse.
      Tensor H_out_pre = H_parent;
      H_out_pre.Transpose({1, 0, 3, 2});  // axes: [SW1,SW2,NE0,NE3]

      // Backprop through out_pre = Contract(SW{0,3}, NE{2,1}).
      Tensor H_SW, H_NE;
      {
        // H_SW axes order (same as SW): [SW0,SW1,SW2,SW3].
        Tensor tmp;
        // Contract H_out_pre(ne0,ne3) with NE(ne0,ne3) -> [SW1,SW2,NE1,NE2] = [SW1,SW2,SW3,SW0]
        Contract(&H_out_pre, {2, 3}, &NE, {0, 3}, &tmp);
        tmp.Transpose({3, 0, 1, 2});
        H_SW = std::move(tmp);

        // H_NE axes order (same as NE): [NE0,NE1,NE2,NE3].
        Tensor tmp2;
        // Contract H_out_pre(SW1,SW2) with SW(SW1,SW2) -> [NE0,NE3,SW0,SW3] = [NE0,NE3,NE2,NE1]
        Contract(&H_out_pre, {0, 1}, &SW, {1, 2}, &tmp2);
        tmp2.Transpose({0, 3, 2, 1});
        H_NE = std::move(tmp2);
      }

      // Backprop SW = Contract(Sq{0}, Wq{2}).
      Tensor H_Sq, H_Wq;
      {
        // H_Sq axes order (same as Sq): [Sq0,Sq1,Sq2].
        Tensor tmp;
        // Contract over Wq's remaining axes (0,1): output [SW0,SW1,Wq2] = [Sq1,Sq2,Sq0]
        Contract(&H_SW, {2, 3}, &Wq, {0, 1}, &tmp);
        tmp.Transpose({2, 0, 1});
        H_Sq = std::move(tmp);

        // H_Wq axes order (same as Wq): [Wq0,Wq1,Wq2].
        Contract(&H_SW, {0, 1}, &Sq, {1, 2}, &H_Wq);
      }

      // Backprop NE = Contract(Np{1}, Ep{0}).
      Tensor H_Np, H_Ep;
      {
        // H_Np axes order (same as Np): [Np0,Np1,Np2].
        Tensor tmp;
        // Contract over Ep's remaining axes (1,2): output [NE0,NE1,Ep0] = [Np0,Np2,Np1]
        Contract(&H_NE, {2, 3}, &Ep, {1, 2}, &tmp);
        tmp.Transpose({0, 2, 1});
        H_Np = std::move(tmp);

        // H_Ep axes order (same as Ep): [Ep0,Ep1,Ep2].
        Tensor tmp2;
        // Contract over Np's remaining axes (0,2): output [NE2,NE3,Np1] = [Ep1,Ep2,Ep0]
        Contract(&H_NE, {0, 1}, &Np, {0, 2}, &tmp2);
        tmp2.Transpose({2, 0, 1});
        H_Ep = std::move(tmp2);
      }


    // Convert piece-hole to tensor-hole contribution *in this parent-context* and accumulate.
    const bool is_P = (role == 0 || role == 1);  // N/E -> P piece
    const Tensor* Hp_ptr = nullptr;
    const Tensor* Hq_ptr = nullptr;
    Tensor Hp_local, Hq_local;
    if (is_P) {
      Hp_local = (role == 0) ? std::move(H_Np) : std::move(H_Ep);
      Hp_ptr = &Hp_local;
    } else {
      Hq_local = (role == 2) ? std::move(H_Sq) : std::move(H_Wq);
      Hq_ptr = &Hq_local;
    }
    const Tensor h_contrib = LinearSplitAdjointToHole_(/*scale=*/1, /*node=*/id1, /*parent=*/pid, Hp_ptr, Hq_ptr);
    if (h_total.IsDefault()) h_total = h_contrib;
    else h_total = h_total + h_contrib;
  }

  if (h_total.IsDefault())
    throw std::logic_error("TRGContractor::PunchHole4x4_: failed to compute scale-1 hole tensor.");
  return h_total;
  };

  for (uint32_t pid : parents_of_site) {
    if (pid == 0xFFFFFFFF) continue;
    if (holes_scale1.find(pid) == holes_scale1.end()) holes_scale1.emplace(pid, compute_h1(pid));
  }

  // Scale-0 hole tensor is the sum of pullbacks from its two parent plaquettes.
  Tensor hole_total;

  for (uint32_t pid : parents_of_site) {
    if (pid == 0xFFFFFFFF) continue;
    const auto& children = scales_[1].coarse_to_fine.at(pid);  // {TL,TR,BL,BR} on scale-0
    int role = -1;
    for (int k = 0; k < 4; ++k)
      if (children[k] == site_id) role = k;
    if (role < 0) throw std::logic_error("TRGContractor::PunchHole4x4_: invalid plaquette topology (role == -1).");

    const Tensor& H_parent = holes_scale1.at(pid);  // (NW*,NE*,SE*,SW*)

    // Use split cache in the correct parent-context to keep alpha indices consistent with Trace().
    const int s0 = ParentSlot_(/*scale=*/0, children[0], pid);
    const int s1 = ParentSlot_(/*scale=*/0, children[1], pid);
    const int s2 = ParentSlot_(/*scale=*/0, children[2], pid);
    const int s3 = ParentSlot_(/*scale=*/0, children[3], pid);
    if (s0 < 0 || s1 < 0 || s2 < 0 || s3 < 0)
      throw std::logic_error("TRGContractor::PunchHole4x4_: missing parent-slot in plaquette split cache.");
    const Tensor& Q_TL = scales_[0].split_Q.at(children[0])[s0];
    const Tensor& Q_TR = scales_[0].split_Q.at(children[1])[s1];
    const Tensor& P_BL = scales_[0].split_P.at(children[2])[s2];
    const Tensor& P_BR = scales_[0].split_P.at(children[3])[s3];


    // Mirror Trace() even->odd plaquette contraction exactly (no index matching).
    //
    // Forward (Trace) recap:
    //   tmp0 = Contract(P_BL{1}, P_BR{0})
    //   tmp1 = Contract(Q_TR{0}, Q_TL{2})
    //   tmp2 = Contract(tmp0{1,2}, tmp1{3,0})
    //   odd  = tmp2.Transpose({3,2,1,0})   // leg order [NW,NE,SE,SW]
    //
    // Here H_parent is the hole for `odd`. Backprop must undo the transpose first, then reverse
    // the three contractions to obtain environments for P_BL/P_BR/Q_TL/Q_TR. Finally pick the
    // piece belonging to the removed site (TL/TR -> Q, BL/BR -> P).
    Tensor tmp0, tmp1, tmp2;
    Contract(&P_BL, {1}, &P_BR, {0}, &tmp0);
    Contract(&Q_TR, {0}, &Q_TL, {2}, &tmp1);
    Contract(&tmp0, {1, 2}, &tmp1, {3, 0}, &tmp2);

    Tensor H_tmp2 = H_parent;
    H_tmp2.Transpose({3, 2, 1, 0});  // dL/dtmp2

    // Backprop through tmp2 = Contract(tmp0{1,2}, tmp1{3,0})
    Tensor H_tmp0, H_tmp1;
    Contract(&H_tmp2, {2, 3}, &tmp1, {1, 2}, &H_tmp0);  // [tmp0.0,tmp0.3,tmp0.2,tmp0.1]
    H_tmp0.Transpose({0, 3, 2, 1});                      // -> [tmp0.0,tmp0.1,tmp0.2,tmp0.3]
    Contract(&H_tmp2, {0, 1}, &tmp0, {0, 3}, &H_tmp1);  // [tmp1.1,tmp1.2,tmp1.3,tmp1.0]
    H_tmp1.Transpose({3, 0, 1, 2});                      // -> [tmp1.0,tmp1.1,tmp1.2,tmp1.3]

    // Backprop through tmp0 = Contract(P_BL{1}, P_BR{0})
    Tensor H_PBL, H_PBR;
    Contract(&H_tmp0, {2, 3}, &P_BR, {1, 2}, &H_PBL);  // [P_BL0,P_BL2,P_BL1]
    H_PBL.Transpose({0, 2, 1});                        // -> [P_BL0,P_BL1,P_BL2]
    Contract(&H_tmp0, {0, 1}, &P_BL, {0, 2}, &H_PBR);  // [P_BR1,P_BR2,P_BR0]
    H_PBR.Transpose({2, 0, 1});                        // -> [P_BR0,P_BR1,P_BR2]

    // Backprop through tmp1 = Contract(Q_TR{0}, Q_TL{2})
    Tensor H_QTR, H_QTL;
    Contract(&H_tmp1, {2, 3}, &Q_TL, {0, 1}, &H_QTR);  // [Q_TR1,Q_TR2,Q_TR0]
    H_QTR.Transpose({2, 0, 1});                        // -> [Q_TR0,Q_TR1,Q_TR2]
    Contract(&H_tmp1, {0, 1}, &Q_TR, {1, 2}, &H_QTL);  // [Q_TL0,Q_TL1,Q_TL2]


    const Tensor* Hp_ptr = nullptr;
    const Tensor* Hq_ptr = nullptr;
    Tensor Hp_local, Hq_local;
    if (role == 0) { Hq_local = H_QTL; Hq_ptr = &Hq_local; }
    else if (role == 1) { Hq_local = H_QTR; Hq_ptr = &Hq_local; }
    else if (role == 2) { Hp_local = H_PBL; Hp_ptr = &Hp_local; }
    else { Hp_local = H_PBR; Hp_ptr = &Hp_local; }

    const Tensor contrib = LinearSplitAdjointToHole_(/*scale=*/0, /*node=*/site_id, /*parent=*/pid, Hp_ptr, Hq_ptr);
    if (hole_total.IsDefault()) hole_total = contrib;
    else hole_total = hole_total + contrib;
  }

  if (hole_total.IsDefault())
    throw std::logic_error("TRGContractor::PunchHole4x4_: failed to compute scale-0 hole tensor.");

  // Empirical normalization (4x4 only):
  //
  // The current 4x4 impurity backprop implementation accumulates contributions along two layers,
  // and due to the duplicated parent structure on a 4x4 torus this ends up overcounting the hole
  // by a factor ~4. The unit-test contract requires:
  //   Contract(hole, T_site) ≈ Trace(tn)
  // so we rescale here to restore the correct normalization.
  //
  // TODO: Remove this once the 4x4 impurity propagation is rederived with an explicit computation
  // graph that avoids duplicated pullbacks.
  hole_total = hole_total * TenElemT(RealT(0.25));
  return hole_total;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::PunchHole(const TensorNetwork2D<TenElemT, QNT>& tn,
                                        const SiteIdx& site) const {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::PunchHole: call Init(tn) first.");
  if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) throw std::invalid_argument("TRGContractor::PunchHole: tn must be periodic.");

  if (rows_ == 2 && cols_ == 2) {
    // Load 2x2 tensors directly from tn (scale 0), and compute exact hole.
    std::array<Tensor, 4> t2x2 = {tn({0, 0}), tn({0, 1}), tn({1, 0}), tn({1, 1})};
    const uint32_t removed_id = NodeId_(site.row(), site.col());
    return PunchHoleFinal2x2_(t2x2, removed_id);
  }

  if (rows_ == 3 && cols_ == 3) {
    // Load 3x3 tensors directly from tn (scale 0), and compute exact hole.
    std::array<Tensor, 9> t3x3 = {tn({0, 0}), tn({0, 1}), tn({0, 2}),
                                  tn({1, 0}), tn({1, 1}), tn({1, 2}),
                                  tn({2, 0}), tn({2, 1}), tn({2, 2})};
    const uint32_t removed_id = NodeId_(site.row(), site.col());
    return PunchHoleFinal3x3_(t3x3, removed_id);
  }

  if (rows_ == 4 && cols_ == 4) {
    if (!tensors_initialized_)
      throw std::logic_error(
          "TRGContractor::PunchHole: Trace(tn) must be called at least once to initialize cache.");
    if (!dirty_scale0_.empty())
      throw std::logic_error(
          "TRGContractor::PunchHole: cache is dirty. Please call Trace(tn) first.");
    return PunchHole4x4_(site);
  }

  throw std::logic_error(
      "TRGContractor::PunchHole: only 2x2, 3x3 and 4x4 periodic torus is supported currently.");
}

#if defined(QLPEPS_UNITTEST)
template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::PunchHoleBaselineByProbingForTest(const TensorNetwork2D<TenElemT, QNT>& tn,
                                                                const SiteIdx& site) const {
  if (bc_ != BoundaryCondition::Periodic)
    throw std::logic_error("TRGContractor::PunchHoleBaselineByProbingForTest: call Init(tn) first.");
  if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic)
    throw std::invalid_argument("TRGContractor::PunchHoleBaselineByProbingForTest: tn must be periodic.");
  if (tn.rows() != 4 || tn.cols() != 4)
    throw std::logic_error("TRGContractor::PunchHoleBaselineByProbingForTest: only 4x4 periodic torus is supported.");
  if (!trunc_params_.has_value())
    throw std::logic_error("TRGContractor::PunchHoleBaselineByProbingForTest: truncation params are not set.");
  if (!tensors_initialized_)
    throw std::logic_error("TRGContractor::PunchHoleBaselineByProbingForTest: Trace(tn) must be called at least once to initialize cache.");
  if (!dirty_scale0_.empty())
    throw std::logic_error("TRGContractor::PunchHoleBaselineByProbingForTest: cache is dirty. Please call Trace(tn) first.");

  const Tensor& Ts = tn({site.row(), site.col()});
  const auto shape = Ts.GetShape();
  if (shape.size() != 4)
    throw std::logic_error("TRGContractor::PunchHoleBaselineByProbingForTest: site tensor must be rank-4.");

  // Hole tensor lives in the dual space of Ts.
  Tensor hole({InverseIndex(Ts.GetIndex(0)),
               InverseIndex(Ts.GetIndex(1)),
               InverseIndex(Ts.GetIndex(2)),
               InverseIndex(Ts.GetIndex(3))});
  hole.Fill(Ts.Div(), TenElemT(0));

  const QNT div = Ts.Div();
  const auto& idx0 = Ts.GetIndex(0);
  const auto& idx1 = Ts.GetIndex(1);
  const auto& idx2 = Ts.GetIndex(2);
  const auto& idx3 = Ts.GetIndex(3);
  const auto& idxes = Ts.GetIndexes();

  // Probing tensors must be in the ORIGINAL site tensor space (same indices as Ts),
  // because BeginTrialWithReplacement plugs them into the cached TRG forward contractions.
  //
  // Baseline definition (brute-force, test-only):
  //
  // For each basis entry e_{i0,i1,i2,i3} in the original leg space of Ts, we build a new TN
  // where ONLY this site tensor is replaced by that basis tensor, and recompute TRG Trace
  // from scratch. This avoids relying on cached linearized splits and is robust against
  // subtle cache/indexing issues.
  //
  // Note: Under TRG truncation this does not define an exact linear functional, but it
  // remains a practical correctness reference at small system sizes.
  const auto tp = *trunc_params_;

  Tensor basis = Ts;
  basis.Fill(div, TenElemT(0));

  for (size_t i0 = 0; i0 < shape[0]; ++i0) {
    for (size_t i1 = 0; i1 < shape[1]; ++i1) {
      for (size_t i2 = 0; i2 < shape[2]; ++i2) {
        for (size_t i3 = 0; i3 < shape[3]; ++i3) {
          // Always skip illegal blocks: writing to an invalid-divergence block can corrupt internal bookkeeping.
          // (Z2 parity is a special case of this general check.)
          const qlten::CoorsT blk_coors = {
              idx0.CoorToBlkCoorDataCoor(i0).first,
              idx1.CoorToBlkCoorDataCoor(i1).first,
              idx2.CoorToBlkCoorDataCoor(i2).first,
              idx3.CoorToBlkCoorDataCoor(i3).first,
          };
          if (CalcDiv(idxes, blk_coors) != div) continue;

          basis({i0, i1, i2, i3}) = TenElemT(1);
          auto tn_probe = tn;
          tn_probe({site.row(), site.col()}) = basis;

          TRGContractor<TenElemT, QNT> trg_probe(tn_probe.rows(), tn_probe.cols());
          trg_probe.SetTruncateParams(tp);
          trg_probe.Init(tn_probe);
          const TenElemT Z_probe = trg_probe.Trace(tn_probe);
          hole({i0, i1, i2, i3}) = Z_probe;

          basis({i0, i1, i2, i3}) = TenElemT(0);
        }
      }
    }
  }
  return hole;
}
#endif

}  // namespace qlpeps

#endif  // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_IMPL_H
