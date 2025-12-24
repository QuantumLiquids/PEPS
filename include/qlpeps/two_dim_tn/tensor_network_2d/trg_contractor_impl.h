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

  // `Sinv` is stored as a *real* tensor. Convert on demand.
  Tensor Sinv_dag;
  {
    const auto Sinv_dag_r = qlten::Dag(Sinv);  // QLTensor<RealT,QNT>
    if constexpr (std::is_same<TenElemT, RealT>::value) {
      Sinv_dag = Sinv_dag_r;
    } else {
      Sinv_dag = ToComplex(Sinv_dag_r);  // ADL: qlten::ToComplex
    }
  }

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
  // Build the fine<->coarse mapping for all scales.
  //
  // IMPORTANT (PBC checkerboard / PunchHole):
  // This topology encodes a checkerboard TRG on a torus by allowing a fine node to have up to
  // two coarse parents per RG step. This is convenient for (future) local updates and for
  // backpropagating holes, but it is a double-cover style representation.
  //
  // On a 4x4 torus there are exactly two RG steps (scale0->scale1 and scale1->scale2).
  // If a hole backprop naively "sums over both parent contexts" at each step, it can introduce
  // an overall multiplicity ~ 2 x 2 = 4. (See the long Doxygen note in trg_contractor.h.)
  size_t n = rows_;
  size_t scale_idx = 0;
  
  while (n > 3) {
    scales_.resize(scale_idx + 2);
    auto& fine_layer = scales_[scale_idx];
    auto& coarse_layer = scales_[scale_idx + 1];
    
    size_t fine_size = (scale_idx == 0) ? (rows_ * cols_) : scales_[scale_idx].tens.size();
    fine_layer.fine_to_coarse.assign(fine_size, {0xFFFFFFFF, 0xFFFFFFFF});
    
    if (scale_idx % 2 == 0) {
      // Even -> Odd (Plaquette RG): n x n -> (n*n)/2 coarse nodes (embedded).
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
          // Only process "even" plaquette anchors.
          if (((r + c) & 1U) != 0U) continue;
          
          uint32_t coarse_id = static_cast<uint32_t>(id_odd(r, c));
          uint32_t tl = static_cast<uint32_t>(id_even(r, c));
          uint32_t tr = static_cast<uint32_t>(id_even(r, mod(c + 1)));
          uint32_t bl = static_cast<uint32_t>(id_even(mod(r + 1), c));
          uint32_t br = static_cast<uint32_t>(id_even(mod(r + 1), mod(c + 1)));
          
          coarse_layer.coarse_to_fine[coarse_id] = {tl, tr, bl, br};
          // Each fine tensor participates in exactly 2 plaquettes on the PBC checkerboard.
          // Fill parent slots sequentially: first slot 0, then slot 1.
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
      // Odd -> Even (Diamond RG): embedded size n -> n/2.
      // Odd tensors count = (n*n)/2; even (next) count = (n/2)*(n/2).
      size_t n_embed = n;
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
           
           // Diamond center (i, j) -> coarse (i/2, (j-1)/2).
           size_t I = i / 2;
           size_t J = (j - 1) / 2;
           uint32_t coarse_id = static_cast<uint32_t>(id_even_next(I, J));
           
           // Neighbors on odd lattice: N/E/S/W around the diamond center.
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

  if (scales_.size() == 1) {
    scales_[0].fine_to_coarse.assign(rows_ * cols_, {0xFFFFFFFF, 0xFFFFFFFF});
  }
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
TRGContractor<TenElemT, QNT>::ContractPlaquetteCore_(const Tensor& Q0, const Tensor& Q1, const Tensor& P1, const Tensor& P0) {
    using qlten::Contract;
    Tensor tmp0, tmp1, tmp2;
    Contract(&P1, {1}, &P0, {0}, &tmp0);
    Contract(&Q1, {0}, &Q0, {2}, &tmp1);
    Contract(&tmp0, {1, 2}, &tmp1, {3, 0}, &tmp2);
    tmp2.Transpose({3, 2, 1, 0});
    return tmp2;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::ContractDiamondCore_(const Tensor& Np, const Tensor& Ep, const Tensor& Sq, const Tensor& Wq) {
    using qlten::Contract;
    Tensor SW, NE, out;
    Contract(&Sq, {0}, &Wq, {2}, &SW);
    Contract(&Np, {1}, &Ep, {0}, &NE);
    Contract(&SW, {0, 3}, &NE, {2, 1}, &out);
    out.Transpose({1, 0, 3, 2});
    return out;
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::EnsureSplitCacheForNodes_(size_t scale, const std::set<uint32_t>& nodes) {
  if (scale >= scales_.size()) throw std::out_of_range("TRGContractor::EnsureSplitCacheForNodes_: invalid scale.");
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

    for (int slot = 0; slot < 2; ++slot) {
      const uint32_t pid = parents[slot];
      if (pid == 0xFFFFFFFF) continue;
      if (pid >= c2f.size()) throw std::logic_error("TRGContractor::EnsureSplitCacheForNodes_: coarse parent id out of range.");

      const auto& children = c2f.at(pid);
      int role = -1;
      for (int k = 0; k < 4; ++k) {
        if (children[k] == id) role = k;
      }
      if (role < 0) throw std::logic_error("TRGContractor::EnsureSplitCacheForNodes_: invalid topology (child not found in parent).");

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
TenElemT TRGContractor<TenElemT, QNT>::GetCachedAmplitude_() const {
    if (scales_.empty()) return TenElemT(0);
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
    throw std::logic_error("TRGContractor::GetCachedAmplitude_: invalid final scale size.");
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::Trace(const TensorNetwork2D<TenElemT, QNT>& tn) {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::Trace: call Init(tn) first.");
  if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) throw std::invalid_argument("TRGContractor::Trace: tn must be periodic.");
  if (!trunc_params_.has_value()) throw std::logic_error("TRGContractor::Trace: truncation params are not set.");
  
  if (tensors_initialized_) {
    return GetCachedAmplitude_();
  }

  // Full initialization
  const size_t N = rows_ * cols_;
  for (uint32_t i = 0; i < N; ++i) {
     auto coords = Coord_(i);
     scales_[0].tens[i] = tn({coords.first, coords.second});
  }

  // Handle small systems with no RG steps
  if (scales_.size() == 1) {
    tensors_initialized_ = true;
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

  // Split scale 0 (all nodes)
  std::set<uint32_t> all_nodes_0;
  for(uint32_t i=0; i<N; ++i) all_nodes_0.insert(i);
  EnsureSplitCacheForNodes_(0, all_nodes_0);

  // Propagate Up
  for (size_t s = 0; s < scales_.size() - 1; ++s) {
      const auto& fine_layer = scales_[s];
      auto& coarse_layer = scales_[s + 1];
      const bool even_to_odd = (s % 2 == 0);
      
      const size_t n_coarse = coarse_layer.tens.size();
      for (uint32_t c_id = 0; c_id < n_coarse; ++c_id) {
          const auto& children = coarse_layer.coarse_to_fine[c_id];
          
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
             throw std::logic_error("TRGContractor::Trace: missing parent-slot.");

          if (even_to_odd) {
              // Plaquette contraction (even -> odd), children order is {TL, TR, BL, BR}.
              //
              // We pick plaquettes with (r+c)%2==0 (checkerboard). For each 2x2 plaquette:
              //   TL = (r, c)      TR = (r, c+1)
              //   BL = (r+1, c)    BR = (r+1, c+1)
              //
              // Split convention (fixed by sublattice):
              //   TL, BR are A (Type0); TR, BL are B (Type1).
              //
              // We must select split pieces in the correct parent-context (fine node -> this plaquette parent)
              // so that alpha indices are stable and reusable by PunchHole().
              //
              // External alphas become legs of the coarse tensor; internal bonds are contracted:
              //
              //   alpha_NW (from TL.Q)    alpha_NE (from TR.Q)
              //            \\             /
              //             \\           /
              //             [  odd tensor  ]   (returned with leg order [NW, NE, SE, SW])
              //             /           \\
              //            /             \\
              //   alpha_SW (from BL.P)    alpha_SE (from BR.P)
              //
              const Tensor& Q0 = fine_layer.split_Q[children[0]][s0];
              const Tensor& Q1 = fine_layer.split_Q[children[1]][s1];
              const Tensor& P1 = fine_layer.split_P[children[2]][s2];
              const Tensor& P0 = fine_layer.split_P[children[3]][s3];
              coarse_layer.tens[c_id] = ContractPlaquetteCore_(Q0, Q1, P1, P0);
          } else {
              // Diamond contraction (odd -> even), children order is {N, E, S, W}.
              const Tensor& Np = fine_layer.split_P[children[0]][s0];
              const Tensor& Ep = fine_layer.split_P[children[1]][s1];
              const Tensor& Sq = fine_layer.split_Q[children[2]][s2];
              const Tensor& Wq = fine_layer.split_Q[children[3]][s3];
              coarse_layer.tens[c_id] = ContractDiamondCore_(Np, Ep, Sq, Wq);
          }
      }

      // Populate split cache for next scale
      std::set<uint32_t> all_coarse_nodes;
      for(uint32_t i=0; i<n_coarse; ++i) all_coarse_nodes.insert(i);
      EnsureSplitCacheForNodes_(s + 1, all_coarse_nodes);
  }

  tensors_initialized_ = true;
  
  // Return result
  return GetCachedAmplitude_();
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Trial
TRGContractor<TenElemT, QNT>::BeginTrialWithReplacement(
    const std::vector<std::pair<SiteIdx, Tensor>>& replacements) const {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::BeginTrialWithReplacement: call Init(tn) first.");
  if (!trunc_params_.has_value()) throw std::logic_error("TRGContractor::BeginTrialWithReplacement: truncation params are not set.");
  if (!tensors_initialized_) throw std::logic_error("TRGContractor::BeginTrialWithReplacement: Trace(tn) must be called at least once to initialize cache.");

  Trial trial;
  trial.layer_updates.resize(scales_.size());

  // Scale-0 updates.
  for (const auto& kv : replacements) {
    const uint32_t id = NodeId_(kv.first.row(), kv.first.col());
    trial.layer_updates[0][id] = kv.second;
  }

  if (trial.layer_updates[0].empty()) {
     // No changes
     trial.amplitude = GetCachedAmplitude_();
     return trial; 
  }

  // Shadow RG propagation
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

    // Recompute affected coarse nodes
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

      if (even_to_odd) {
        auto tl = SplitType0_(T0);
        auto tr = SplitType1_(T1);
        auto bl = SplitType1_(T2);
        auto br = SplitType0_(T3);
        trial.layer_updates[s + 1][c_id] = ContractPlaquetteCore_(tl.Q, tr.Q, bl.P, br.P);
      } else {
        auto nB = SplitType1_(T0);
        auto eA = SplitType0_(T1);
        auto sB = SplitType1_(T2);
        auto wA = SplitType0_(T3);
        trial.layer_updates[s + 1][c_id] = ContractDiamondCore_(nB.P, eA.P, sB.Q, wA.Q);
      }
    }
  }

  // Final amplitude
  if (!trial.layer_updates.empty()) {
    const size_t last = trial.layer_updates.size() - 1;
    if (scales_.back().tens.size() == 1) {
      auto it = trial.layer_updates[last].find(0);
      trial.amplitude = ContractFinal1x1_(it != trial.layer_updates[last].end() ? it->second : scales_.back().tens[0]);
    } else if (scales_.back().tens.size() == 4) {
      std::array<Tensor, 4> t2x2 = {scales_.back().tens[0], scales_.back().tens[1],
                                    scales_.back().tens[2], scales_.back().tens[3]};
      for (uint32_t id = 0; id < 4; ++id) {
        auto it = trial.layer_updates[last].find(id);
        if (it != trial.layer_updates[last].end()) t2x2[id] = it->second;
      }
      trial.amplitude = ContractFinal2x2_(t2x2);
    } else if (scales_.back().tens.size() == 9) {
      std::array<Tensor, 9> t3x3 = {scales_.back().tens[0], scales_.back().tens[1], scales_.back().tens[2],
                                    scales_.back().tens[3], scales_.back().tens[4], scales_.back().tens[5],
                                    scales_.back().tens[6], scales_.back().tens[7], scales_.back().tens[8]};
      for (uint32_t id = 0; id < 9; ++id) {
        auto it = trial.layer_updates[last].find(id);
        if (it != trial.layer_updates[last].end()) t3x3[id] = it->second;
      }
      trial.amplitude = ContractFinal3x3_(t3x3);
    } else {
      throw std::logic_error("TRGContractor::BeginTrialWithReplacement: invalid final scale size.");
    }
  }
  return trial;
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::CommitTrial(Trial&& trial) {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::CommitTrial: call Init(tn) first.");
  if (!trunc_params_.has_value()) throw std::logic_error("TRGContractor::CommitTrial: truncation params are not set.");
  if (!tensors_initialized_) throw std::logic_error("TRGContractor::CommitTrial: Trace(tn) must be called at least once to initialize cache.");
  if (trial.layer_updates.size() != scales_.size()) throw std::invalid_argument("TRGContractor::CommitTrial: trial topology mismatch.");

  for (size_t s = 0; s < trial.layer_updates.size(); ++s) {
    if (trial.layer_updates[s].empty()) continue;
    
    // Update tensors
    for (auto& kv : trial.layer_updates[s]) {
      scales_[s].tens[kv.first] = std::move(kv.second);
    }
    
    // Update split cache for the modified tensors (needed for future PunchHole)
    std::set<uint32_t> modified_nodes;
    for (auto& kv : trial.layer_updates[s]) {
        // kv.second is now moved-from (invalid), but the key is the node ID.
        // wait, we moved it in the loop above. We need the keys.
        // Actually, since we iterate `kv` in `trial.layer_updates[s]`, and map iteration gives valid keys:
        modified_nodes.insert(kv.first);
    }
    // Re-iterate is safer or copy keys first. Map node order is sorted, so:
    // We iterate trial.layer_updates[s] twice is fine, but we moved the value.
    // The key is const.
    // However, EnsureSplitCacheForNodes_ uses scales_[s].tens[id], which is now valid (updated).
    EnsureSplitCacheForNodes_(s, modified_nodes);
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
    Contract(&T2x2[1], {1, 3}, &T2x2[3], {3, 1}, &bd);  
    Tensor hole;
    Contract(&bd, {2, 3}, &T2x2[2], {2, 0}, &hole);  
    hole.Transpose({1, 3, 0, 2});
    return hole;
  }
  if (removed_id == 1) {
    Tensor ac;
    Contract(&T2x2[0], {1, 3}, &T2x2[2], {3, 1}, &ac);  
    Tensor hole;
    Contract(&ac, {2, 3}, &T2x2[3], {2, 0}, &hole);      
    hole.Transpose({1, 3, 0, 2});
    return hole;
  }
  if (removed_id == 2) {
    Tensor ab;
    Contract(&T2x2[0], {0, 2}, &T2x2[1], {2, 0}, &ab);  
    Tensor hole;
    Contract(&ab, {2, 3}, &T2x2[3], {3, 1}, &hole);      
    hole.Transpose({3, 1, 2, 0});
    return hole;
  }
  // removed_id == 3
  Tensor ab;
  Contract(&T2x2[0], {0, 2}, &T2x2[1], {2, 0}, &ab);    
  Tensor hole;
  Contract(&ab, {0, 1}, &T2x2[2], {3, 1}, &hole);        
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

  auto require_inv = [&](const Tensor& A, size_t axA, const Tensor& B, size_t axB, const char* where) {
    if (A.IsDefault() || B.IsDefault())
      throw std::logic_error(std::string("TRGContractor::PunchHoleFinal3x3_: default tensor at ") + where);
    const auto& ia = A.GetIndex(axA);
    const auto& ib = B.GetIndex(axB);
    if (!(ia == InverseIndex(ib)))
      throw std::logic_error(std::string("TRGContractor::PunchHoleFinal3x3_: index mismatch at ") + where);
  };

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

  std::vector<LabeledTensor> comps;
  comps.reserve(8);

  for (uint8_t r = 0; r < 3; ++r) {
    for (uint8_t c = 0; c < 3; ++c) {
      const uint32_t sid = id(r, c);
      if (sid == removed_id) continue;
      const Tensor& T = T3x3[sid];
      if (T.IsDefault())
        throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: site tensor is default.");
      
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

  auto trace_pair_inplace = [&](LabeledTensor& lt, size_t ax_a, size_t ax_b) {
    if (ax_a == ax_b) throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: tracing identical axes.");
    const auto& idx_a = lt.ten.GetIndex(ax_a);
    const auto& idx_b = lt.ten.GetIndex(ax_b);

    const bool a_is_out = (idx_a.GetDir() == qlten::TenIndexDirType::OUT);
    const bool b_is_out = (idx_b.GetDir() == qlten::TenIndexDirType::OUT);
    if (a_is_out == b_is_out)
      throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: bond axes must be opposite directions.");

    const size_t ax_in = a_is_out ? ax_b : ax_a;
    const size_t ax_out = a_is_out ? ax_a : ax_b;
    const auto& idx_out = lt.ten.GetIndex(ax_out);

    if (!(lt.ten.GetIndex(ax_in) == InverseIndex(idx_out))) {
      throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: trace legs are not inverse indices.");
    }
    const auto eye = Eye<TenElemT, QNT>(idx_out);

    Tensor out;
    Contract(&lt.ten, {ax_in, ax_out}, &eye, {0, 1}, &out);
    lt.ten = std::move(out);

    const size_t hi = std::max(ax_in, ax_out);
    const size_t lo = std::min(ax_in, ax_out);
    lt.labels.erase(lt.labels.begin() + hi);
    lt.labels.erase(lt.labels.begin() + lo);
  };

  for (int safety = 0; safety < 200; ++safety) {
    if (comps.size() == 1) {
      std::map<BondKey, size_t> kmap;
      std::vector<std::pair<size_t, size_t>> self_pairs;
      for (size_t a = 0; a < comps[0].labels.size(); ++a) {
        const auto& k = comps[0].labels[a];
        if (kmap.count(k)) {
          self_pairs.push_back({kmap[k], a});
          kmap.erase(k); 
        } else {
          kmap[k] = a;
        }
      }
      if (self_pairs.empty()) break; 
      trace_pair_inplace(comps[0], self_pairs[0].first, self_pairs[0].second);
      continue;
    }

    std::map<BondKey, std::pair<size_t, size_t>> open_bonds;
    std::map<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>> adj;

    for (size_t i = 0; i < comps.size(); ++i) {
      for (size_t a = 0; a < comps[i].labels.size(); ++a) {
        const auto& k = comps[i].labels[a];
        auto it = open_bonds.find(k);
        if (it != open_bonds.end()) {
          size_t j = it->second.first;
          size_t b = it->second.second;
          size_t u = std::min(i, j);
          size_t v = std::max(i, j);
          size_t ax_u = (i == u) ? a : b;
          size_t ax_v = (i == u) ? b : a;
          adj[{u, v}].push_back({ax_u, ax_v});
        } else {
          open_bonds[k] = {i, a};
        }
      }
    }

    if (adj.empty()) throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: components disconnected.");

    struct Move {
      size_t u, v;
      long cost; 
    };
    Move best_move = {0, 0, 999999};
    bool found_move = false;

    for (const auto& kv : adj) {
      size_t u = kv.first.first;
      size_t v = kv.first.second;
      size_t n_bonds = kv.second.size();
      
      long rank_u = static_cast<long>(comps[u].ten.Rank());
      long rank_v = static_cast<long>(comps[v].ten.Rank());
      
      long cost = 0;
      if (u == v) {
        cost = rank_u - 2 * static_cast<long>(n_bonds);
        cost -= 100000; 
      } else {
        cost = rank_u + rank_v - 2 * static_cast<long>(n_bonds);
      }

      if (!found_move || cost < best_move.cost) {
        best_move = {u, v, cost};
        found_move = true;
      }
    }

    size_t u = best_move.u;
    size_t v = best_move.v;
    const auto& bonds = adj[{u, v}];

    if (u == v) {
      trace_pair_inplace(comps[u], bonds[0].first, bonds[0].second);
    } else {
      LabeledTensor A = std::move(comps[u]);
      LabeledTensor B = std::move(comps[v]);
      
      std::vector<size_t> idx_A, idx_B;
      std::set<size_t> set_A, set_B; 
      for (const auto& p : bonds) {
        idx_A.push_back(p.first);
        idx_B.push_back(p.second);
        set_A.insert(p.first);
        set_B.insert(p.second);
        require_inv(A.ten, p.first, B.ten, p.second, "multi-bond merge");
      }

      Tensor out;
      Contract(&A.ten, idx_A, &B.ten, idx_B, &out);
      
      std::vector<BondKey> new_labels;
      new_labels.reserve(A.labels.size() + B.labels.size() - 2 * bonds.size());
      for (size_t k = 0; k < A.labels.size(); ++k) {
        if (set_A.find(k) == set_A.end()) new_labels.push_back(A.labels[k]);
      }
      for (size_t k = 0; k < B.labels.size(); ++k) {
        if (set_B.find(k) == set_B.end()) new_labels.push_back(B.labels[k]);
      }
      
      LabeledTensor merged{std::move(out), std::move(new_labels)};
      
      comps.erase(comps.begin() + v);
      comps.erase(comps.begin() + u);
      comps.push_back(std::move(merged));
    }
  }

  if (comps.size() != 1)
    throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: failed to fully contract components.");

  Tensor hole = std::move(comps[0].ten);
  auto labels = std::move(comps[0].labels);
  if (labels.size() != 4)
    throw std::logic_error("TRGContractor::PunchHoleFinal3x3_: final open legs are not rank-4.");

  const std::array<BondKey, 4> want = {
      x(rr, mod3(int(cc) - 1)), 
      y(rr, cc),                
      x(rr, cc),                
      y(mod3(int(rr) - 1), cc)  
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
  return hole;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::PunchHoleBackpropGeneric_(const SiteIdx& site) const {
  using qlten::Contract;

  auto require_inv = [&](const Tensor& A, size_t axA, const Tensor& B, size_t axB, const char* where) {
    if (A.IsDefault() || B.IsDefault()) {
      throw std::logic_error(
          std::string("TRGContractor::PunchHoleBackpropGeneric_: default tensor in ") + where);
    }
    const auto& ia = A.GetIndex(axA);
    const auto& ib = B.GetIndex(axB);
    if (!(ia == InverseIndex(ib))) {
      throw std::logic_error(std::string("TRGContractor::PunchHoleBackpropGeneric_: index mismatch at ") + where);
    }
  };

  const uint32_t site_id = NodeId_(site.row(), site.col());
  const size_t last = scales_.size() - 1;
  const size_t top_size = scales_.at(last).tens.size();
  
  if (top_size != 9 && top_size != 4)
    throw std::logic_error("TRGContractor::PunchHoleBackpropGeneric_: last scale must be 3x3 (size 9) or 2x2 (size 4).");

  std::vector<std::set<uint32_t>> anc(scales_.size());
  anc[0].insert(site_id);
  for (size_t s = 0; s < last; ++s) {
    for (uint32_t id : anc[s]) {
      const auto& parents = scales_.at(s).fine_to_coarse.at(id);
      if (parents[0] != 0xFFFFFFFF) anc[s + 1].insert(parents[0]);
      if (parents[1] != 0xFFFFFFFF) anc[s + 1].insert(parents[1]);
    }
  }

  std::map<uint32_t, Tensor> holes_next;
  {
    const auto& top = scales_.at(last).tens;
    if (top_size == 9) {
    const std::array<Tensor, 9> t3x3 = {top[0], top[1], top[2],
                                        top[3], top[4], top[5],
                                        top[6], top[7], top[8]};
    for (uint32_t id = 0; id < 9; ++id) {
      if (anc[last].count(id) == 0) continue;
      holes_next.emplace(id, PunchHoleFinal3x3_(t3x3, id));
        }
    } else {
        const std::array<Tensor, 4> t2x2 = {top[0], top[1], top[2], top[3]};
        for (uint32_t id = 0; id < 4; ++id) {
            if (anc[last].count(id) == 0) continue;
            holes_next.emplace(id, PunchHoleFinal2x2_(t2x2, id));
        }
    }
  }

  auto backprop_diamond_parent = [&](size_t s, uint32_t child_id, uint32_t pid, const Tensor& H_parent) -> Tensor {
    const auto& children = scales_.at(s + 1).coarse_to_fine.at(pid); 
    int role = -1;
    for (int k = 0; k < 4; ++k) if (children[k] == child_id) role = k;
    if (role < 0) throw std::logic_error("TRGContractor::PunchHoleBackpropGeneric_: invalid diamond topology.");

    const int sn = ParentSlot_(s, children[0], pid);
    const int se = ParentSlot_(s, children[1], pid);
    const int ss = ParentSlot_(s, children[2], pid);
    const int sw = ParentSlot_(s, children[3], pid);
    if (sn < 0 || se < 0 || ss < 0 || sw < 0)
      throw std::logic_error("TRGContractor::PunchHoleBackpropGeneric_: missing parent-slot in diamond split cache.");

    const Tensor& Np = scales_.at(s).split_P.at(children[0])[sn];
    const Tensor& Ep = scales_.at(s).split_P.at(children[1])[se];
    const Tensor& Sq = scales_.at(s).split_Q.at(children[2])[ss];
    const Tensor& Wq = scales_.at(s).split_Q.at(children[3])[sw];

    Tensor SW, NE;
    require_inv(Sq, 0, Wq, 2, "diamond: Sq{0} vs Wq{2}");
    Contract(&Sq, {0}, &Wq, {2}, &SW);
    require_inv(Np, 1, Ep, 0, "diamond: Np{1} vs Ep{0}");
    Contract(&Np, {1}, &Ep, {0}, &NE);

    Tensor H_out_pre = H_parent;
    H_out_pre.Transpose({1, 0, 3, 2}); 

    Tensor H_SW, H_NE;
    {
      Tensor tmp;
      require_inv(H_out_pre, 2, NE, 0, "diamond backprop: H_out_pre{2} vs NE{0}");
      require_inv(H_out_pre, 3, NE, 3, "diamond backprop: H_out_pre{3} vs NE{3}");
      Contract(&H_out_pre, {2, 3}, &NE, {0, 3}, &tmp);
      tmp.Transpose({3, 0, 1, 2});
      H_SW = std::move(tmp);

      Tensor tmp2;
      require_inv(H_out_pre, 0, SW, 1, "diamond backprop: H_out_pre{0} vs SW{1}");
      require_inv(H_out_pre, 1, SW, 2, "diamond backprop: H_out_pre{1} vs SW{2}");
      Contract(&H_out_pre, {0, 1}, &SW, {1, 2}, &tmp2);
      tmp2.Transpose({0, 3, 2, 1});
      H_NE = std::move(tmp2);
    }

    Tensor H_Sq, H_Wq;
    {
      Tensor tmp;
      require_inv(H_SW, 2, Wq, 0, "diamond backprop SW->Sq: H_SW{2} vs Wq{0}");
      require_inv(H_SW, 3, Wq, 1, "diamond backprop SW->Sq: H_SW{3} vs Wq{1}");
      Contract(&H_SW, {2, 3}, &Wq, {0, 1}, &tmp);
      tmp.Transpose({2, 0, 1});
      H_Sq = std::move(tmp);

      require_inv(H_SW, 0, Sq, 1, "diamond backprop SW->Wq: H_SW{0} vs Sq{1}");
      require_inv(H_SW, 1, Sq, 2, "diamond backprop SW->Wq: H_SW{1} vs Sq{2}");
      Contract(&H_SW, {0, 1}, &Sq, {1, 2}, &H_Wq);
    }

    Tensor H_Np, H_Ep;
    {
      Tensor tmp;
      require_inv(H_NE, 2, Ep, 1, "diamond backprop NE->Np: H_NE{2} vs Ep{1}");
      require_inv(H_NE, 3, Ep, 2, "diamond backprop NE->Np: H_NE{3} vs Ep{2}");
      Contract(&H_NE, {2, 3}, &Ep, {1, 2}, &tmp);
      tmp.Transpose({0, 2, 1});
      H_Np = std::move(tmp);

      Tensor tmp2;
      require_inv(H_NE, 0, Np, 0, "diamond backprop NE->Ep: H_NE{0} vs Np{0}");
      require_inv(H_NE, 1, Np, 2, "diamond backprop NE->Ep: H_NE{1} vs Np{2}");
      Contract(&H_NE, {0, 1}, &Np, {0, 2}, &tmp2);
      tmp2.Transpose({2, 0, 1});
      H_Ep = std::move(tmp2);
    }

    const bool is_P = (role == 0 || role == 1);  
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
    return LinearSplitAdjointToHole_(s, child_id, pid, Hp_ptr, Hq_ptr);
  };

  auto backprop_plaquette_parent = [&](size_t s, uint32_t child_id, uint32_t pid, const Tensor& H_parent) -> Tensor {
    const auto& children = scales_.at(s + 1).coarse_to_fine.at(pid);  
    int role = -1;
    for (int k = 0; k < 4; ++k) if (children[k] == child_id) role = k;
    if (role < 0) throw std::logic_error("TRGContractor::PunchHoleBackpropGeneric_: invalid plaquette topology.");

    const int s0 = ParentSlot_(s, children[0], pid);
    const int s1 = ParentSlot_(s, children[1], pid);
    const int s2 = ParentSlot_(s, children[2], pid);
    const int s3 = ParentSlot_(s, children[3], pid);
    if (s0 < 0 || s1 < 0 || s2 < 0 || s3 < 0)
      throw std::logic_error("TRGContractor::PunchHoleBackpropGeneric_: missing parent-slot in plaquette split cache.");

    const Tensor& Q_TL = scales_.at(s).split_Q.at(children[0])[s0];
    const Tensor& Q_TR = scales_.at(s).split_Q.at(children[1])[s1];
    const Tensor& P_BL = scales_.at(s).split_P.at(children[2])[s2];
    const Tensor& P_BR = scales_.at(s).split_P.at(children[3])[s3];

    Tensor tmp0, tmp1, tmp2;
    require_inv(P_BL, 1, P_BR, 0, "plaquette: P_BL{1} vs P_BR{0}");
    Contract(&P_BL, {1}, &P_BR, {0}, &tmp0);
    require_inv(Q_TR, 0, Q_TL, 2, "plaquette: Q_TR{0} vs Q_TL{2}");
    Contract(&Q_TR, {0}, &Q_TL, {2}, &tmp1);
    require_inv(tmp0, 1, tmp1, 3, "plaquette: tmp0{1} vs tmp1{3}");
    require_inv(tmp0, 2, tmp1, 0, "plaquette: tmp0{2} vs tmp1{0}");
    Contract(&tmp0, {1, 2}, &tmp1, {3, 0}, &tmp2);

    Tensor H_tmp2 = H_parent;
    H_tmp2.Transpose({3, 2, 1, 0}); 

    Tensor H_tmp0, H_tmp1;
    {
      Tensor t;
      require_inv(H_tmp2, 2, tmp1, 1, "plaquette backprop tmp2->tmp0: H_tmp2{2} vs tmp1{1}");
      require_inv(H_tmp2, 3, tmp1, 2, "plaquette backprop tmp2->tmp0: H_tmp2{3} vs tmp1{2}");
      Contract(&H_tmp2, {2, 3}, &tmp1, {1, 2}, &t);
      t.Transpose({0, 3, 2, 1});
      H_tmp0 = std::move(t);

      Tensor t2;
      require_inv(H_tmp2, 0, tmp0, 0, "plaquette backprop tmp2->tmp1: H_tmp2{0} vs tmp0{0}");
      require_inv(H_tmp2, 1, tmp0, 3, "plaquette backprop tmp2->tmp1: H_tmp2{1} vs tmp0{3}");
      Contract(&H_tmp2, {0, 1}, &tmp0, {0, 3}, &t2);
      t2.Transpose({3, 0, 1, 2});
      H_tmp1 = std::move(t2);
    }

    Tensor H_QTR, H_QTL;
    {
      Tensor t;
      require_inv(H_tmp1, 2, Q_TL, 0, "plaquette backprop tmp1->Q_TR: H_tmp1{2} vs Q_TL{0}");
      require_inv(H_tmp1, 3, Q_TL, 1, "plaquette backprop tmp1->Q_TR: H_tmp1{3} vs Q_TL{1}");
      Contract(&H_tmp1, {2, 3}, &Q_TL, {0, 1}, &t);
      t.Transpose({2, 0, 1});
      H_QTR = std::move(t);

      require_inv(H_tmp1, 0, Q_TR, 1, "plaquette backprop tmp1->Q_TL: H_tmp1{0} vs Q_TR{1}");
      require_inv(H_tmp1, 1, Q_TR, 2, "plaquette backprop tmp1->Q_TL: H_tmp1{1} vs Q_TR{2}");
      Contract(&H_tmp1, {0, 1}, &Q_TR, {1, 2}, &H_QTL);
    }

    Tensor H_PBL, H_PBR;
    {
      Tensor t;
      require_inv(H_tmp0, 2, P_BR, 1, "plaquette backprop tmp0->P_BL: H_tmp0{2} vs P_BR{1}");
      require_inv(H_tmp0, 3, P_BR, 2, "plaquette backprop tmp0->P_BL: H_tmp0{3} vs P_BR{2}");
      Contract(&H_tmp0, {2, 3}, &P_BR, {1, 2}, &t);
      t.Transpose({0, 2, 1});
      H_PBL = std::move(t);

      Tensor t2;
      require_inv(H_tmp0, 0, P_BL, 0, "plaquette backprop tmp0->P_BR: H_tmp0{0} vs P_BL{0}");
      require_inv(H_tmp0, 1, P_BL, 2, "plaquette backprop tmp0->P_BR: H_tmp0{1} vs P_BL{2}");
      Contract(&H_tmp0, {0, 1}, &P_BL, {0, 2}, &t2);
      t2.Transpose({2, 0, 1});
      H_PBR = std::move(t2);
    }

    const Tensor* Hp_ptr = nullptr;
    const Tensor* Hq_ptr = nullptr;
    Tensor Hp_local, Hq_local;
    if (role == 0) { Hq_local = std::move(H_QTL); Hq_ptr = &Hq_local; }
    else if (role == 1) { Hq_local = std::move(H_QTR); Hq_ptr = &Hq_local; }
    else if (role == 2) { Hp_local = std::move(H_PBL); Hp_ptr = &Hp_local; }
    else { Hp_local = std::move(H_PBR); Hp_ptr = &Hp_local; }

    return LinearSplitAdjointToHole_(s, child_id, pid, Hp_ptr, Hq_ptr);
  };

  for (size_t s = last; s-- > 0;) {
    std::map<uint32_t, Tensor> holes_cur;
    for (uint32_t id : anc[s]) {
      const auto& parents = scales_.at(s).fine_to_coarse.at(id);
      Tensor sum;
      int cnt = 0;
      for (uint32_t pid : parents) {
        if (pid == 0xFFFFFFFF) continue;
        auto it = holes_next.find(pid);
        if (it == holes_next.end()) continue;
        const Tensor& H_parent = it->second;
        const Tensor contrib = (s % 2 == 0)
                                   ? backprop_plaquette_parent(s, id, pid, H_parent)
                                   : backprop_diamond_parent(s, id, pid, H_parent);
        if (sum.IsDefault()) sum = contrib;
        else sum = sum + contrib;
        ++cnt;
      }
      if (cnt == 0)
        throw std::logic_error("TRGContractor::PunchHoleBackpropGeneric_: failed to compute hole (no valid parent contributions).");

      if (cnt > 1) sum = sum * TenElemT(RealT(1.0 / double(cnt)));
      holes_cur.emplace(id, std::move(sum));
    }
    holes_next = std::move(holes_cur);
  }

  auto it0 = holes_next.find(site_id);
  if (it0 == holes_next.end())
    throw std::logic_error("TRGContractor::PunchHoleBackpropGeneric_: missing final site hole.");
  return it0->second;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::PunchHole(const TensorNetwork2D<TenElemT, QNT>& tn,
                                        const SiteIdx& site) const {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::PunchHole: call Init(tn) first.");
  if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) throw std::invalid_argument("TRGContractor::PunchHole: tn must be periodic.");

  if (rows_ == 2 && cols_ == 2) {
    std::array<Tensor, 4> t2x2 = {tn({0, 0}), tn({0, 1}), tn({1, 0}), tn({1, 1})};
    const uint32_t removed_id = NodeId_(site.row(), site.col());
    return PunchHoleFinal2x2_(t2x2, removed_id);
  }

  if (rows_ == 3 && cols_ == 3) {
    std::array<Tensor, 9> t3x3 = {tn({0, 0}), tn({0, 1}), tn({0, 2}),
                                  tn({1, 0}), tn({1, 1}), tn({1, 2}),
                                  tn({2, 0}), tn({2, 1}), tn({2, 2})};
    const uint32_t removed_id = NodeId_(site.row(), site.col());
    return PunchHoleFinal3x3_(t3x3, removed_id);
  }

    if (!tensors_initialized_)
      throw std::logic_error(
          "TRGContractor::PunchHole: Trace(tn) must be called at least once to initialize cache.");

  const bool is_pow2 = IsPowerOfTwo_(rows_);
  const bool is_3pow2 = (rows_ % 3 == 0) && IsPowerOfTwo_(rows_ / 3);

  if ((rows_ == cols_) && (is_pow2 || is_3pow2)) {
    return PunchHoleBackpropGeneric_(site);
  }

  throw std::logic_error(
      "TRGContractor::PunchHole: only 2x2, 3x3 and N=2^k or 3*2^k periodic torus is supported currently.");
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

  const Tensor& Ts = tn({site.row(), site.col()});
  const auto shape = Ts.GetShape();
  if (shape.size() != 4)
    throw std::logic_error("TRGContractor::PunchHoleBaselineByProbingForTest: site tensor must be rank-4.");

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

  const auto tp = *trunc_params_;

  Tensor basis = Ts;
  basis.Fill(div, TenElemT(0));

  for (size_t i0 = 0; i0 < shape[0]; ++i0) {
    for (size_t i1 = 0; i1 < shape[1]; ++i1) {
      for (size_t i2 = 0; i2 < shape[2]; ++i2) {
        for (size_t i3 = 0; i3 < shape[3]; ++i3) {
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
