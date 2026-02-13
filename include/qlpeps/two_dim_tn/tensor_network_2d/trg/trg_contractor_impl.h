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
#include <numeric>
#include <set>
#include <map>

#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/trg/trg_contractor.h"

namespace qlpeps {

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::Tensor
TRGContractor<TenElemT, QNT>::LinearSplitAdjointToHole_(size_t scale,
                                                        uint32_t node,
                                                        const Tensor* H_P,
                                                        const Tensor* H_Q) const {
  using qlten::Contract;

#ifndef NDEBUG
  auto require_match = [&](const Tensor& A, size_t axA, const Tensor& B, size_t axB) {
    const auto& ia = A.GetIndex(axA);
    const auto& ib = B.GetIndex(axB);
    if (!(ia == InverseIndex(ib))) {
      throw std::logic_error("TRGContractor::LinearSplitAdjointToHole_: index mismatch.");
    }
  };
#endif

  const auto st = scales_.at(scale).split_type.at(node);
  const auto& U = scales_.at(scale).split_U.at(node);
  const auto& Vt = scales_.at(scale).split_Vt.at(node);
  const auto& Sinv = scales_.at(scale).split_S_inv_sqrt.at(node);

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
#ifndef NDEBUG
      require_match(*H_P, 2, Sinv_dag, 0);
#endif
      Contract(H_P, {2}, &Sinv_dag, {0}, &Hp_scaled);  // (l*,u*,alpha*)
#ifndef NDEBUG
      require_match(Hp_scaled, 2, V_adj, 0);
#endif
      Contract(&Hp_scaled, {2}, &V_adj, {0}, &t1);     // (l*,u*,d*,r*)
    }

    Tensor t2;
    if (H_Q != nullptr) {
      Tensor Hq_scaled;
#ifndef NDEBUG
      require_match(Sinv_dag, 1, *H_Q, 0);
#endif
      Contract(&Sinv_dag, {1}, H_Q, {0}, &Hq_scaled);  // (alpha*,d*,r*)
#ifndef NDEBUG
      require_match(U_adj, 2, Hq_scaled, 0);
#endif
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
#ifndef NDEBUG
    require_match(*H_Q, 2, Sinv_dag, 0);
#endif
    Contract(H_Q, {2}, &Sinv_dag, {0}, &Hq_scaled);   // (l*,d*,alpha*)
#ifndef NDEBUG
    require_match(Hq_scaled, 2, V_adj, 0);
#endif
    Contract(&Hq_scaled, {2}, &V_adj, {0}, &t1);      // (l*,d*,r*,u*)
  }

  Tensor t2;
  if (H_P != nullptr) {
    Tensor Hp_scaled;
#ifndef NDEBUG
    require_match(Sinv_dag, 1, *H_P, 0);
#endif
    Contract(&Sinv_dag, {1}, H_P, {0}, &Hp_scaled);   // (alpha*,r*,u*)
#ifndef NDEBUG
    require_match(U_adj, 2, Hp_scaled, 0);
#endif
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
  ValidateTopologyConsistency_();
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::ValidateTopologyConsistency_() const {
  for (size_t s = 0; s + 1 < scales_.size(); ++s) {
    const auto& fine = scales_.at(s);
    const auto& coarse = scales_.at(s + 1);

    for (uint32_t fid = 0; fid < fine.fine_to_coarse.size(); ++fid) {
      const auto& parents = fine.fine_to_coarse[fid];
      bool found_any_parent = false;
      std::optional<typename ScaleCache::SplitType> st0;
      for (int slot = 0; slot < 2; ++slot) {
        const uint32_t pid = parents[slot];
        if (pid == 0xFFFFFFFF) continue;
        found_any_parent = true;
        if (pid >= coarse.coarse_to_fine.size()) {
          throw std::logic_error("TRGContractor::ValidateTopologyConsistency_: parent id out of range.");
        }
        const auto& children = coarse.coarse_to_fine[pid];
        int role = -1;
        for (int k = 0; k < 4; ++k) {
          if (children[k] == fid) role = k;
        }
        if (role < 0) {
          throw std::logic_error("TRGContractor::ValidateTopologyConsistency_: child missing in parent.");
        }
        const auto st = ChildSplitType_(s, fid, role);
        if (!st0.has_value()) st0 = st;
        else if (*st0 != st) {
          throw std::logic_error("TRGContractor::ValidateTopologyConsistency_: split type mismatch across parent contexts.");
        }
      }
      if (!found_any_parent) {
        throw std::logic_error("TRGContractor::ValidateTopologyConsistency_: fine node has no valid parent.");
      }
    }
  }
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
typename TRGContractor<TenElemT, QNT>::SplitResult
TRGContractor<TenElemT, QNT>::SplitNode_(
    const Tensor& T_in,
    typename ScaleCache::SplitType split_type,
    bool compute_isometry) const {
  using qlten::Contract;
  using qlten::ElementWiseSqrt;
  using qlten::SVD;

  if (T_in.IsDefault()) {
    throw std::runtime_error("TRG SplitNode_: input tensor is default.");
  }

  // Type0 (A-sublattice): group legs (l,up)|(d,r) via transpose {0,3,1,2}.
  //
  //   up(3)                       up(3)
  //     |                          |
  // l(0)-T-r(2)   ==SVD==>    l(0)-P----alpha----Q-r(2)
  //     |                                        |
  //   d(1)                                     d(1)
  //
  // Type1 (B-sublattice): group legs (l,d)|(r,up) — already in order {0,1,2,3}.
  //
  //      up(3)                                 up(3)
  //      |                                     |
  // l(0)-T-r(2)   ==SVD==>  l(0)-Q----alpha----P-r(2)
  //      |                       |
  //    d(1)                      d(1)
  Tensor T = T_in;
  if (split_type == ScaleCache::SplitType::Type0) {
    T.Transpose({0, 3, 1, 2});
  }

  Tensor u, vt;
  qlten::QLTensor<RealT, QNT> s;
  RealT trunc_err_actual = RealT(0);
  size_t bond_dim_actual = 0;
  const auto& tp = *trunc_params_;
  SVD(&T, 2, T.Div(), tp.trunc_err, tp.D_min, tp.D_max,
      &u, &s, &vt, &trunc_err_actual, &bond_dim_actual);

  auto s_sqrt = ElementWiseSqrt(s);

  SplitResult out;
  out.split_type = split_type;

  // Type0: P = U*sqrt(S), Q = sqrt(S)*Vt
  // Type1: Q = U*sqrt(S), P = sqrt(S)*Vt  (note the swap)
  Tensor left_piece, right_piece;
  Contract(&u, &s_sqrt, {{2}, {0}}, &left_piece);
  Contract(&s_sqrt, &vt, {{1}, {0}}, &right_piece);

  if (split_type == ScaleCache::SplitType::Type0) {
    out.P = std::move(left_piece);
    out.Q = std::move(right_piece);
  } else {
    out.Q = std::move(left_piece);
    out.P = std::move(right_piece);
  }

  if (compute_isometry) {
    out.U = std::move(u);
    out.Vt = std::move(vt);
    const RealT max_s = s_sqrt.GetMaxAbs();
    const RealT effective_eps = tp.ComputeEffectiveInvEps(max_s);
    auto s_inv_sqrt = s_sqrt;
    s_inv_sqrt.ElementWiseInv(effective_eps);
    out.S_inv_sqrt = std::move(s_inv_sqrt);
  }

  return out;
}

template <typename TenElemT, typename QNT>
typename TRGContractor<TenElemT, QNT>::ScaleCache::SplitType
TRGContractor<TenElemT, QNT>::ChildSplitType_(size_t scale, uint32_t child_id, int role) const {
    if (scale == 0) {
      return (scales_[0].graph.sublattice.at(child_id) == SubLattice::A)
                 ? ScaleCache::SplitType::Type0
                 : ScaleCache::SplitType::Type1;
    }
    if ((scale & 1U) == 0U) {
      // Even -> Odd plaquette, children order {TL,TR,BL,BR}.
      return (role == 0 || role == 3) ? ScaleCache::SplitType::Type0 : ScaleCache::SplitType::Type1;
    }
    // Odd -> Even diamond, children order {N,E,S,W}.
    return (role == 0 || role == 2) ? ScaleCache::SplitType::Type1 : ScaleCache::SplitType::Type0;
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
void TRGContractor<TenElemT, QNT>::EnsureSplitCacheForNodes_(size_t scale, const std::vector<uint32_t>& nodes) {
  if (scale >= scales_.size()) throw std::out_of_range("TRGContractor::EnsureSplitCacheForNodes_: invalid scale.");
  if (scale + 1 >= scales_.size()) return;

  auto& layer = scales_[scale];
  const size_t n = layer.tens.size();
  if (layer.split_type.size() != n) {
    layer.split_type.assign(n, ScaleCache::SplitType::Type0);
  }
  if (layer.split_P.size() != n) layer.split_P.resize(n);
  if (layer.split_Q.size() != n) layer.split_Q.resize(n);
  if (layer.split_U.size() != n) layer.split_U.resize(n);
  if (layer.split_Vt.size() != n) layer.split_Vt.resize(n);
  if (layer.split_S_inv_sqrt.size() != n) layer.split_S_inv_sqrt.resize(n);

  const auto& c2f = scales_[scale + 1].coarse_to_fine;

  for (uint32_t id : nodes) {
    if (id >= n) throw std::out_of_range("TRGContractor::EnsureSplitCacheForNodes_: node id out of range.");
    const auto& T = layer.tens.at(id);
    if (T.IsDefault()) throw std::logic_error("TRGContractor::EnsureSplitCacheForNodes_: tensor is default.");

    // Determine split type from any active parent (both give the same result).
    const auto& parents = layer.fine_to_coarse.at(id);
    typename ScaleCache::SplitType st = ScaleCache::SplitType::Type0;
    bool found = false;
    for (int slot = 0; slot < 2 && !found; ++slot) {
      const uint32_t pid = parents[slot];
      if (pid == 0xFFFFFFFF) continue;
      const auto& children = c2f.at(pid);
      for (int k = 0; k < 4; ++k) {
        if (children[k] == id) {
          st = ChildSplitType_(scale, id, k);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      throw std::logic_error("TRGContractor::EnsureSplitCacheForNodes_: no active coarse parent found for node.");
    }

    auto result = SplitNode_(T, st, /*compute_isometry=*/true);
    layer.split_type[id] = result.split_type;
    layer.split_P[id] = std::move(result.P);
    layer.split_Q[id] = std::move(result.Q);
    layer.split_U[id] = std::move(result.U);
    layer.split_Vt[id] = std::move(result.Vt);
    layer.split_S_inv_sqrt[id] = std::move(result.S_inv_sqrt);
  }
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::ComputeFinalAmplitude_(
    const std::map<uint32_t, Tensor>* top_updates) const {
    if (scales_.empty()) return TenElemT(0);
    const auto& top = scales_.back().tens;
    const size_t n = top.size();

    if (n == 1) {
      if (top_updates) {
        auto it = top_updates->find(0);
        if (it != top_updates->end()) return ContractFinal1x1_(it->second);
      }
      return ContractFinal1x1_(top[0]);
    }
    if (n == 4) {
      std::array<Tensor, 4> t2x2 = {top[0], top[1], top[2], top[3]};
      if (top_updates) {
        for (uint32_t id = 0; id < 4; ++id) {
          auto it = top_updates->find(id);
          if (it != top_updates->end()) t2x2[id] = it->second;
        }
      }
      return ContractFinal2x2_(t2x2);
    }
    if (n == 9) {
      std::array<const Tensor*, 9> t3x3;
      // For overlaid tensors, we need stable storage for the pointer array.
      std::array<Tensor, 9> overlay_buf;
      for (uint32_t id = 0; id < 9; ++id) {
        if (top_updates) {
          auto it = top_updates->find(id);
          if (it != top_updates->end()) {
            overlay_buf[id] = it->second;
            t3x3[id] = &overlay_buf[id];
            continue;
          }
        }
        t3x3[id] = &top[id];
      }
      return ContractFinal3x3_(t3x3);
    }
    throw std::logic_error("TRGContractor::ComputeFinalAmplitude_: invalid final scale size.");
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::Trace(const TensorNetwork2D<TenElemT, QNT>& tn) {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::Trace: call Init(tn) first.");
  if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) throw std::invalid_argument("TRGContractor::Trace: tn must be periodic.");
  if (!trunc_params_.has_value()) throw std::logic_error("TRGContractor::Trace: truncation params are not set.");
  
  if (tensors_initialized_) {
    return ComputeFinalAmplitude_();
  }

  // Full initialization
  const size_t N = rows_ * cols_;
  for (uint32_t i = 0; i < N; ++i) {
     auto [r, c] = Coord_(i);
     scales_[0].tens[i] = tn({r, c});
  }

  // Handle small systems with no RG steps
  if (scales_.size() == 1) {
    tensors_initialized_ = true;
    return ComputeFinalAmplitude_();
  }

  // Split scale 0 (all nodes)
  std::vector<uint32_t> all_nodes_0(N);
  std::iota(all_nodes_0.begin(), all_nodes_0.end(), 0);
  EnsureSplitCacheForNodes_(0, all_nodes_0);

  // Propagate Up
  for (size_t s = 0; s < scales_.size() - 1; ++s) {
      const auto& fine_layer = scales_[s];
      auto& coarse_layer = scales_[s + 1];
      const bool even_to_odd = (s % 2 == 0);
      
      const size_t n_coarse = coarse_layer.tens.size();
      for (uint32_t c_id = 0; c_id < n_coarse; ++c_id) {
          const auto& children = coarse_layer.coarse_to_fine[c_id];
          
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
              // Split data is per-node (identical for both parent contexts).
              const Tensor& Q0 = fine_layer.split_Q[children[0]];
              const Tensor& Q1 = fine_layer.split_Q[children[1]];
              const Tensor& P1 = fine_layer.split_P[children[2]];
              const Tensor& P0 = fine_layer.split_P[children[3]];
              coarse_layer.tens[c_id] = ContractPlaquetteCore_(Q0, Q1, P1, P0);
          } else {
              // Diamond contraction (odd -> even), children order is {N, E, S, W}.
              const Tensor& Np = fine_layer.split_P[children[0]];
              const Tensor& Ep = fine_layer.split_P[children[1]];
              const Tensor& Sq = fine_layer.split_Q[children[2]];
              const Tensor& Wq = fine_layer.split_Q[children[3]];
              coarse_layer.tens[c_id] = ContractDiamondCore_(Np, Ep, Sq, Wq);
          }
      }

      // Populate split cache for next scale
      std::vector<uint32_t> all_coarse_nodes(n_coarse);
      std::iota(all_coarse_nodes.begin(), all_coarse_nodes.end(), 0);
      EnsureSplitCacheForNodes_(s + 1, all_coarse_nodes);
  }

  tensors_initialized_ = true;
  
  // Return result
  return ComputeFinalAmplitude_();
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
  trial.layer_splits.resize(scales_.size());

  // Scale-0 updates.
  for (const auto& [site, tensor] : replacements) {
    trial.layer_updates[0][NodeId_(site.row(), site.col())] = tensor;
  }

  if (trial.layer_updates[0].empty()) {
     trial.amplitude = ComputeFinalAmplitude_();
     return trial;
  }

  PropagateReplacements_(trial.layer_updates, &trial.layer_splits);

  const size_t last = trial.layer_updates.size() - 1;
  trial.amplitude = ComputeFinalAmplitude_(
      trial.layer_updates[last].empty() ? nullptr : &trial.layer_updates[last]);
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
    for (auto& [id, tensor] : trial.layer_updates[s]) {
      scales_[s].tens[id] = std::move(tensor);
    }

    // Transplant pre-computed split data from trial into the scale cache.
    // Falls back to EnsureSplitCacheForNodes_ only for nodes without stored splits.
    if (s < trial.layer_splits.size() && !trial.layer_splits[s].empty()) {
      std::vector<uint32_t> need_recompute;
      for (const auto& [id, _] : trial.layer_updates[s]) {
        auto split_it = trial.layer_splits[s].find(id);
        if (split_it != trial.layer_splits[s].end()) {
          auto& sd = split_it->second;
          auto& layer = scales_[s];
          layer.split_type[id] = static_cast<typename ScaleCache::SplitType>(sd.type);
          layer.split_U[id] = std::move(sd.U);
          layer.split_Vt[id] = std::move(sd.Vt);
          layer.split_S_inv_sqrt[id] = std::move(sd.S_inv_sqrt);
          layer.split_P[id] = std::move(sd.P);
          layer.split_Q[id] = std::move(sd.Q);
        } else {
          need_recompute.push_back(id);
        }
      }
      if (!need_recompute.empty()) {
        EnsureSplitCacheForNodes_(s, need_recompute);
      }
    } else {
      // No stored splits at this scale; fall back to full recompute.
      std::vector<uint32_t> modified_nodes;
      for (const auto& [id, _] : trial.layer_updates[s]) {
        modified_nodes.push_back(id);
      }
      EnsureSplitCacheForNodes_(s, modified_nodes);
    }
  }
}

template <typename TenElemT, typename QNT>
TenElemT TRGContractor<TenElemT, QNT>::EvaluateReplacement(
    const std::vector<std::pair<SiteIdx, Tensor>>& replacements) const {
  if (bc_ != BoundaryCondition::Periodic) throw std::logic_error("TRGContractor::EvaluateReplacement: call Init(tn) first.");
  if (!trunc_params_.has_value()) throw std::logic_error("TRGContractor::EvaluateReplacement: truncation params are not set.");
  if (!tensors_initialized_) throw std::logic_error("TRGContractor::EvaluateReplacement: Trace(tn) must be called at least once to initialize cache.");

  if (replacements.empty()) {
    return ComputeFinalAmplitude_();
  }

  std::vector<std::map<uint32_t, Tensor>> layer_updates(scales_.size());
  for (const auto& [site, tensor] : replacements) {
    layer_updates[0][NodeId_(site.row(), site.col())] = tensor;
  }

  PropagateReplacements_(layer_updates, nullptr);

  const size_t last = layer_updates.size() - 1;
  return ComputeFinalAmplitude_(
      layer_updates[last].empty() ? nullptr : &layer_updates[last]);
}

template <typename TenElemT, typename QNT>
void TRGContractor<TenElemT, QNT>::PropagateReplacements_(
    std::vector<std::map<uint32_t, Tensor>>& layer_updates,
    std::vector<std::map<uint32_t, TrialSplitData>>* trial_splits) const {
  const bool compute_isometry = (trial_splits != nullptr);

  for (size_t s = 0; s < scales_.size() - 1; ++s) {
    if (layer_updates[s].empty()) break;

    const auto& fine_layer = scales_[s];
    const auto& coarse_layer = scales_[s + 1];
    const bool even_to_odd = (s % 2 == 0);

    // Identify affected coarse nodes.
    std::set<uint32_t> dirty_coarse_nodes;
    for (const auto& [f_id, _] : layer_updates[s]) {
      const auto& parents = fine_layer.fine_to_coarse[f_id];
      if (parents[0] != 0xFFFFFFFF) dirty_coarse_nodes.insert(parents[0]);
      if (parents[1] != 0xFFFFFFFF) dirty_coarse_nodes.insert(parents[1]);
    }

    // Local storage for lightweight splits (evaluate path only).
    std::map<uint32_t, SplitResult> local_splits;

    for (uint32_t c_id : dirty_coarse_nodes) {
      const auto& children = coarse_layer.coarse_to_fine[c_id];

      auto is_dirty = [&](uint32_t id) -> bool {
        return layer_updates[s].count(id) > 0;
      };
      auto get_fine_tensor = [&](uint32_t id) -> const Tensor& {
        auto it = layer_updates[s].find(id);
        if (it != layer_updates[s].end()) return it->second;
        return fine_layer.tens[id];
      };

      // Get P/Q for a dirty child: compute via SplitNode_ and cache in the appropriate storage.
      // Returns references to P and Q that remain valid until the end of this coarse-node iteration.
      auto get_split = [&](uint32_t child_id, int role) -> std::pair<const Tensor&, const Tensor&> {
        if (!is_dirty(child_id)) {
          return {fine_layer.split_P.at(child_id), fine_layer.split_Q.at(child_id)};
        }
        auto st = ChildSplitType_(s, child_id, role);
        if (compute_isometry) {
          // Trial path: store full SVD in trial_splits.
          auto it = (*trial_splits)[s].find(child_id);
          if (it != (*trial_splits)[s].end())
            return {it->second.P, it->second.Q};
          auto result = SplitNode_(get_fine_tensor(child_id), st, true);
          TrialSplitData sd;
          sd.type = static_cast<uint8_t>(result.split_type);
          sd.U = std::move(result.U);
          sd.Vt = std::move(result.Vt);
          sd.S_inv_sqrt = std::move(result.S_inv_sqrt);
          sd.P = std::move(result.P);
          sd.Q = std::move(result.Q);
          auto [pos, _] = (*trial_splits)[s].emplace(child_id, std::move(sd));
          return {pos->second.P, pos->second.Q};
        } else {
          // Evaluate path: store lightweight split locally.
          auto it = local_splits.find(child_id);
          if (it != local_splits.end())
            return {it->second.P, it->second.Q};
          auto result = SplitNode_(get_fine_tensor(child_id), st, false);
          auto [pos, _] = local_splits.emplace(child_id, std::move(result));
          return {pos->second.P, pos->second.Q};
        }
      };

      if (even_to_odd) {
        // Plaquette (even -> odd), children = {TL, TR, BL, BR}.
        auto [P_TL, Q_TL] = get_split(children[0], 0);
        auto [P_TR, Q_TR] = get_split(children[1], 1);
        auto [P_BL, Q_BL] = get_split(children[2], 2);
        auto [P_BR, Q_BR] = get_split(children[3], 3);
        layer_updates[s + 1][c_id] = ContractPlaquetteCore_(Q_TL, Q_TR, P_BL, P_BR);
      } else {
        // Diamond (odd -> even), children = {N, E, S, W}.
        auto [Np, Nq] = get_split(children[0], 0);
        auto [Ep, Eq] = get_split(children[1], 1);
        auto [Sp, Sq] = get_split(children[2], 2);
        auto [Wp, Wq] = get_split(children[3], 3);
        layer_updates[s + 1][c_id] = ContractDiamondCore_(Np, Ep, Sq, Wq);
      }
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
TenElemT TRGContractor<TenElemT, QNT>::ContractFinal3x3_(const std::array<const Tensor*, 9>& T3x3) const {
  using qlten::Contract;

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

    // Close the row in one contraction:
    // - A1.R <-> A2.L
    // - A0.L <-> A2.R
    // t01 axes: [A0.L,A0.D,A0.U, A1.D,A1.R,A1.U]
    Tensor closed;
    Contract(&t01, {0, 4}, &A2, {2, 0}, &closed);
    // closed axes: [A0.D,A0.U, A1.D,A1.U, A2.D,A2.U]
    return closed;
  };

  for (const Tensor* t : T3x3) {
    if (t == nullptr || t->IsDefault()) {
      throw std::logic_error("TRGContractor::ContractFinal3x3_: default/null terminal tensor.");
    }
  }

  Tensor R0 = row_ring(*T3x3[0], *T3x3[1], *T3x3[2]);
  Tensor R1 = row_ring(*T3x3[3], *T3x3[4], *T3x3[5]);
  Tensor R2 = row_ring(*T3x3[6], *T3x3[7], *T3x3[8]);

  // Contract vertically between rows: connect D (OUT) with U (IN).
  Tensor M01;
  Contract(&R0, {0, 2, 4}, &R1, {1, 3, 5}, &M01);

  // Close vertical PBC in one contraction:
  // - M01[0,1,2] (R0.U0,U1,U2) <-> R2[0,2,4] (R2.D0,D1,D2)
  // - M01[3,4,5] (R1.D0,D1,D2) <-> R2[1,3,5] (R2.U0,U1,U2)
  Tensor out;
  Contract(&M01, {0, 1, 2, 3, 4, 5}, &R2, {0, 2, 4, 1, 3, 5}, &out);
  return out();
}

template <typename TenElemT, typename QNT>
std::array<typename TRGContractor<TenElemT, QNT>::Tensor, 4>
TRGContractor<TenElemT, QNT>::PunchAllHoleFinal2x2_(
    const std::array<Tensor, 4>& T2x2) const {
  using qlten::Contract;
  // Exact 2x2 PBC holes by contracting the other 3 tensors for each site.
  //
  // Id layout (row-major):
  //   0 1
  //   2 3
  //
  // For a 2x2 torus, each nearest-neighbor pair is connected by TWO bonds.
  std::array<Tensor, 4> holes;

  // Shared intermediate: ab contracts horizontal bonds of row 0.
  Tensor ab;
  Contract(&T2x2[0], {0, 2}, &T2x2[1], {2, 0}, &ab);

  // Hole 0: remove T[0], contract T[1]-T[3] vertically then with T[2].
  {
    Tensor bd;
    Contract(&T2x2[1], {1, 3}, &T2x2[3], {3, 1}, &bd);
    Contract(&bd, {2, 3}, &T2x2[2], {2, 0}, &holes[0]);
    holes[0].Transpose({1, 3, 0, 2});
  }
  // Hole 1: remove T[1], contract T[0]-T[2] vertically then with T[3].
  {
    Tensor ac;
    Contract(&T2x2[0], {1, 3}, &T2x2[2], {3, 1}, &ac);
    Contract(&ac, {2, 3}, &T2x2[3], {2, 0}, &holes[1]);
    holes[1].Transpose({1, 3, 0, 2});
  }
  // Hole 2: remove T[2], use shared ab then contract with T[3].
  {
    Contract(&ab, {2, 3}, &T2x2[3], {3, 1}, &holes[2]);
    holes[2].Transpose({3, 1, 2, 0});
  }
  // Hole 3: remove T[3], use shared ab then contract with T[2].
  {
    Contract(&ab, {0, 1}, &T2x2[2], {3, 1}, &holes[3]);
    holes[3].Transpose({3, 1, 2, 0});
  }
  return holes;
}

template <typename TenElemT, typename QNT>
std::array<typename TRGContractor<TenElemT, QNT>::Tensor, 9>
TRGContractor<TenElemT, QNT>::PunchAllHoleFinal3x3_(
    const std::array<const Tensor*, 9>& T3x3) const {
  using qlten::Contract;

  for (const Tensor* t : T3x3) {
    if (t == nullptr || t->IsDefault()) {
      throw std::logic_error("TRGContractor::PunchAllHoleFinal3x3_: site tensor is default/null.");
    }
  }

  // Contract one periodic row of 3 tensors into rank-6: [D0,U0,D1,U1,D2,U2].
  auto row_ring = [](const Tensor& A0, const Tensor& A1, const Tensor& A2) -> Tensor {
    Tensor t01;
    Contract(&A0, {2}, &A1, {0}, &t01);
    // t01 axes: [A0.L, A0.D, A0.U, A1.D, A1.R, A1.U]
    Tensor row;
    Contract(&t01, {0, 4}, &A2, {2, 0}, &row);
    // row axes: [A0.D, A0.U, A1.D, A1.U, A2.D, A2.U]
    return row;
  };

  auto mod3 = [](int x) -> uint8_t { return static_cast<uint8_t>((x % 3 + 3) % 3); };
  auto site_id = [](uint8_t r, uint8_t c) -> uint32_t { return uint32_t(r) * 3u + uint32_t(c); };

  std::array<Tensor, 9> holes;

  for (uint8_t rr = 0; rr < 3; ++rr) {
    for (uint8_t cc = 0; cc < 3; ++cc) {
      const uint32_t rid = site_id(rr, cc);

      // Relative indexing from the removed site.
      auto rel_id = [&](int dr, int dc) -> uint32_t {
        return site_id(mod3(int(rr) + dr), mod3(int(cc) + dc));
      };

      const Tensor& T01 = *T3x3[rel_id(0, 1)];
      const Tensor& T02 = *T3x3[rel_id(0, 2)];
      const Tensor& T10 = *T3x3[rel_id(1, 0)];
      const Tensor& T11 = *T3x3[rel_id(1, 1)];
      const Tensor& T12 = *T3x3[rel_id(1, 2)];
      const Tensor& T20 = *T3x3[rel_id(2, 0)];
      const Tensor& T21 = *T3x3[rel_id(2, 1)];
      const Tensor& T22 = *T3x3[rel_id(2, 2)];

      // Row around the hole (open chain of the 2 remaining tensors in the same row).
      Tensor R0;
      Contract(&T01, {2}, &T02, {0}, &R0);
      // R0 axes: [T01.L, T01.D, T01.U, T02.D, T02.R, T02.U]

      // Middle and bottom rows (each closed horizontally).
      Tensor R1 = row_ring(T10, T11, T12);
      Tensor R2 = row_ring(T20, T21, T22);

      // Vertically merge middle and bottom rows: R1.D <-> R2.U
      Tensor R12;
      Contract(&R1, {0, 2, 4}, &R2, {1, 3, 5}, &R12);
      // R12 axes: [U1_c0, U1_c1, U1_c2, D2_c0, D2_c1, D2_c2]

      // Final merge: connect R0 vertical legs to R12
      // R0.T01.D(1) <-> R12.U1_c1(1)
      // R0.T02.D(3) <-> R12.U1_c2(2)
      // R0.T01.U(2) <-> R12.D2_c1(4)
      // R0.T02.U(5) <-> R12.D2_c2(5)
      Tensor hole;
      Contract(&R0, {1, 3, 2, 5}, &R12, {1, 2, 4, 5}, &hole);

      if (hole.Rank() != 4) {
        throw std::logic_error("TRGContractor::PunchAllHoleFinal3x3_: result is not rank-4.");
      }
      hole.Transpose({1, 2, 0, 3});  // [R,L,D,U] -> [L,D,R,U]
      holes[rid] = std::move(hole);
    }
  }
  return holes;
}

// ---- Batch PunchAllHoles implementation ----
template <typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>
TRGContractor<TenElemT, QNT>::PunchAllHolesImpl_() const {
  const size_t last = scales_.size() - 1;
  const size_t top_size = scales_.at(last).tens.size();

  if (top_size != 9 && top_size != 4)
    throw std::logic_error("TRGContractor::PunchAllHolesImpl_: last scale must be 3x3 (size 9) or 2x2 (size 4).");

  // Step 1: Compute ALL top-level holes (not just ancestors of one site)
  std::map<uint32_t, Tensor> holes_cur;
  {
    const auto& top = scales_.at(last).tens;
    if (top_size == 9) {
      const std::array<const Tensor*, 9> t3x3 = {&top[0], &top[1], &top[2],
                                                  &top[3], &top[4], &top[5],
                                                  &top[6], &top[7], &top[8]};
      auto all = PunchAllHoleFinal3x3_(t3x3);
      for (uint32_t id = 0; id < 9; ++id) {
        holes_cur.emplace(id, std::move(all[id]));
      }
    } else {
      const std::array<Tensor, 4> t2x2 = {top[0], top[1], top[2], top[3]};
      auto all = PunchAllHoleFinal2x2_(t2x2);
      for (uint32_t id = 0; id < 4; ++id) {
        holes_cur.emplace(id, std::move(all[id]));
      }
    }
  }

  // Step 2: Backprop layer by layer, iterating per-coarse-parent to memoize intermediates.
  // For each coarse parent, we build intermediates (tmp0/tmp1 or SW/NE) once, then extract
  // per-child holes for all 4 children cheaply. This avoids 4x redundant contractions.
  for (size_t s = last; s-- > 0;) {
    using qlten::Contract;
    const size_t num_nodes = scales_.at(s).tens.size();
    const size_t num_coarse = scales_.at(s + 1).tens.size();
    const bool even_to_odd = (s % 2 == 0);

    // Accumulator: child_id -> {sum_of_contributions, count}
    std::vector<Tensor> hole_accum(num_nodes);
    std::vector<int> hole_cnt(num_nodes, 0);

    for (uint32_t pid = 0; pid < static_cast<uint32_t>(num_coarse); ++pid) {
      auto it = holes_cur.find(pid);
      if (it == holes_cur.end()) continue;
      const Tensor& H_parent = it->second;
      const auto& children = scales_.at(s + 1).coarse_to_fine.at(pid);

      if (even_to_odd) {
        // Plaquette backprop: build tmp0, tmp1 once, then extract holes for TL, TR, BL, BR.
        const Tensor& Q_TL = scales_.at(s).split_Q.at(children[0]);
        const Tensor& Q_TR = scales_.at(s).split_Q.at(children[1]);
        const Tensor& P_BL = scales_.at(s).split_P.at(children[2]);
        const Tensor& P_BR = scales_.at(s).split_P.at(children[3]);

        Tensor tmp0, tmp1;
        Contract(&P_BL, {1}, &P_BR, {0}, &tmp0);
        Contract(&Q_TR, {0}, &Q_TL, {2}, &tmp1);

        // H_tmp2 = H_parent with the plaquette transpose
        Tensor H_tmp2 = H_parent;
        H_tmp2.Transpose({3, 2, 1, 0});

        // Compute H_tmp0 and H_tmp1 (shared by all children)
        Tensor H_tmp0, H_tmp1;
        {
          Tensor t;
          Contract(&H_tmp2, {2, 3}, &tmp1, {1, 2}, &t);
          t.Transpose({0, 3, 2, 1});
          H_tmp0 = std::move(t);

          Tensor t2;
          Contract(&H_tmp2, {0, 1}, &tmp0, {0, 3}, &t2);
          t2.Transpose({3, 0, 1, 2});
          H_tmp1 = std::move(t2);
        }

        // Child 0 (TL): Q piece -> H_QTL from H_tmp1
        {
          Tensor H_QTL;
          Contract(&H_tmp1, {0, 1}, &Q_TR, {1, 2}, &H_QTL);
          Tensor contrib = LinearSplitAdjointToHole_(s, children[0], nullptr, &H_QTL);
          if (hole_accum[children[0]].IsDefault()) hole_accum[children[0]] = std::move(contrib);
          else hole_accum[children[0]] = hole_accum[children[0]] + contrib;
          ++hole_cnt[children[0]];
        }
        // Child 1 (TR): Q piece -> H_QTR from H_tmp1
        {
          Tensor t;
          Contract(&H_tmp1, {2, 3}, &Q_TL, {0, 1}, &t);
          t.Transpose({2, 0, 1});
          Tensor contrib = LinearSplitAdjointToHole_(s, children[1], nullptr, &t);
          if (hole_accum[children[1]].IsDefault()) hole_accum[children[1]] = std::move(contrib);
          else hole_accum[children[1]] = hole_accum[children[1]] + contrib;
          ++hole_cnt[children[1]];
        }
        // Child 2 (BL): P piece -> H_PBL from H_tmp0
        {
          Tensor t;
          Contract(&H_tmp0, {2, 3}, &P_BR, {1, 2}, &t);
          t.Transpose({0, 2, 1});
          Tensor contrib = LinearSplitAdjointToHole_(s, children[2], &t, nullptr);
          if (hole_accum[children[2]].IsDefault()) hole_accum[children[2]] = std::move(contrib);
          else hole_accum[children[2]] = hole_accum[children[2]] + contrib;
          ++hole_cnt[children[2]];
        }
        // Child 3 (BR): P piece -> H_PBR from H_tmp0
        {
          Tensor t;
          Contract(&H_tmp0, {0, 1}, &P_BL, {0, 2}, &t);
          t.Transpose({2, 0, 1});
          Tensor contrib = LinearSplitAdjointToHole_(s, children[3], &t, nullptr);
          if (hole_accum[children[3]].IsDefault()) hole_accum[children[3]] = std::move(contrib);
          else hole_accum[children[3]] = hole_accum[children[3]] + contrib;
          ++hole_cnt[children[3]];
        }
      } else {
        // Diamond backprop: build SW, NE once, then extract holes for N, E, S, W.
        const Tensor& Np = scales_.at(s).split_P.at(children[0]);
        const Tensor& Ep = scales_.at(s).split_P.at(children[1]);
        const Tensor& Sq = scales_.at(s).split_Q.at(children[2]);
        const Tensor& Wq = scales_.at(s).split_Q.at(children[3]);

        Tensor SW, NE;
        Contract(&Sq, {0}, &Wq, {2}, &SW);
        Contract(&Np, {1}, &Ep, {0}, &NE);

        Tensor H_out_pre = H_parent;
        H_out_pre.Transpose({1, 0, 3, 2});

        // Compute H_SW and H_NE (shared by children)
        Tensor H_SW, H_NE;
        {
          Tensor tmp;
          Contract(&H_out_pre, {2, 3}, &NE, {0, 3}, &tmp);
          tmp.Transpose({3, 0, 1, 2});
          H_SW = std::move(tmp);

          Tensor tmp2;
          Contract(&H_out_pre, {0, 1}, &SW, {1, 2}, &tmp2);
          tmp2.Transpose({0, 3, 2, 1});
          H_NE = std::move(tmp2);
        }

        // Child 0 (N): P piece -> H_Np from H_NE
        {
          Tensor t;
          Contract(&H_NE, {2, 3}, &Ep, {1, 2}, &t);
          t.Transpose({0, 2, 1});
          Tensor contrib = LinearSplitAdjointToHole_(s, children[0], &t, nullptr);
          if (hole_accum[children[0]].IsDefault()) hole_accum[children[0]] = std::move(contrib);
          else hole_accum[children[0]] = hole_accum[children[0]] + contrib;
          ++hole_cnt[children[0]];
        }
        // Child 1 (E): P piece -> H_Ep from H_NE
        {
          Tensor t;
          Contract(&H_NE, {0, 1}, &Np, {0, 2}, &t);
          t.Transpose({2, 0, 1});
          Tensor contrib = LinearSplitAdjointToHole_(s, children[1], &t, nullptr);
          if (hole_accum[children[1]].IsDefault()) hole_accum[children[1]] = std::move(contrib);
          else hole_accum[children[1]] = hole_accum[children[1]] + contrib;
          ++hole_cnt[children[1]];
        }
        // Child 2 (S): Q piece -> H_Sq from H_SW
        {
          Tensor t;
          Contract(&H_SW, {2, 3}, &Wq, {0, 1}, &t);
          t.Transpose({2, 0, 1});
          Tensor contrib = LinearSplitAdjointToHole_(s, children[2], nullptr, &t);
          if (hole_accum[children[2]].IsDefault()) hole_accum[children[2]] = std::move(contrib);
          else hole_accum[children[2]] = hole_accum[children[2]] + contrib;
          ++hole_cnt[children[2]];
        }
        // Child 3 (W): Q piece -> H_Wq from H_SW
        {
          Tensor H_Wq;
          Contract(&H_SW, {0, 1}, &Sq, {1, 2}, &H_Wq);
          Tensor contrib = LinearSplitAdjointToHole_(s, children[3], nullptr, &H_Wq);
          if (hole_accum[children[3]].IsDefault()) hole_accum[children[3]] = std::move(contrib);
          else hole_accum[children[3]] = hole_accum[children[3]] + contrib;
          ++hole_cnt[children[3]];
        }
      }
    }

    // Average contributions and build holes_prev
    std::map<uint32_t, Tensor> holes_prev;
    for (uint32_t id = 0; id < static_cast<uint32_t>(num_nodes); ++id) {
      if (hole_cnt[id] == 0)
        throw std::logic_error("TRGContractor::PunchAllHolesImpl_: failed to compute hole (no valid parent contributions).");
      if (hole_cnt[id] > 1) hole_accum[id] = hole_accum[id] * TenElemT(RealT(1.0 / double(hole_cnt[id])));
      holes_prev.emplace(id, std::move(hole_accum[id]));
    }
    holes_cur = std::move(holes_prev);
  }

  // Step 3: Assemble result into TensorNetwork2D
  TensorNetwork2D<TenElemT, QNT> result(rows_, cols_, BoundaryCondition::Periodic);
  for (size_t r = 0; r < rows_; ++r) {
    for (size_t c = 0; c < cols_; ++c) {
      const uint32_t node_id = NodeId_(r, c);
      auto it = holes_cur.find(node_id);
      if (it == holes_cur.end())
        throw std::logic_error("TRGContractor::PunchAllHolesImpl_: missing hole for site.");
      result({r, c}) = std::move(it->second);
    }
  }
  return result;
}

template <typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>
TRGContractor<TenElemT, QNT>::PunchAllHoles(const TensorNetwork2D<TenElemT, QNT>& tn) const {
  if (bc_ != BoundaryCondition::Periodic)
    throw std::logic_error("TRGContractor::PunchAllHoles: call Init(tn) first.");
  if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic)
    throw std::invalid_argument("TRGContractor::PunchAllHoles: tn must be periodic.");

  // Handle small cases directly (no RG needed)
  if (rows_ == 2 && cols_ == 2) {
    std::array<Tensor, 4> t2x2 = {tn({0, 0}), tn({0, 1}), tn({1, 0}), tn({1, 1})};
    auto all = PunchAllHoleFinal2x2_(t2x2);
    TensorNetwork2D<TenElemT, QNT> result(2, 2, BoundaryCondition::Periodic);
    for (size_t r = 0; r < 2; ++r)
      for (size_t c = 0; c < 2; ++c)
        result({r, c}) = std::move(all[NodeId_(r, c)]);
    return result;
  }

  if (rows_ == 3 && cols_ == 3) {
    const std::array<const Tensor*, 9> t3x3 = {&tn({0, 0}), &tn({0, 1}), &tn({0, 2}),
                                                &tn({1, 0}), &tn({1, 1}), &tn({1, 2}),
                                                &tn({2, 0}), &tn({2, 1}), &tn({2, 2})};
    auto all = PunchAllHoleFinal3x3_(t3x3);
    TensorNetwork2D<TenElemT, QNT> result(3, 3, BoundaryCondition::Periodic);
    for (size_t r = 0; r < 3; ++r)
      for (size_t c = 0; c < 3; ++c)
        result({r, c}) = std::move(all[NodeId_(r, c)]);
    return result;
  }

  if (!tensors_initialized_)
    throw std::logic_error(
        "TRGContractor::PunchAllHoles: Trace(tn) must be called at least once to initialize cache.");

  const bool is_pow2 = IsPowerOfTwo_(rows_);
  const bool is_3pow2 = (rows_ % 3 == 0) && IsPowerOfTwo_(rows_ / 3);

  if ((rows_ == cols_) && (is_pow2 || is_3pow2)) {
    return PunchAllHolesImpl_();
  }

  throw std::logic_error(
      "TRGContractor::PunchAllHoles: only 2x2, 3x3 and N=2^k or 3*2^k periodic torus is supported currently.");
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
