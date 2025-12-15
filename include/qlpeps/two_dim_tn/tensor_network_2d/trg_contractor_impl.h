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
  if (!IsPowerOfTwo_(rows_)) {
    throw std::invalid_argument("TRGContractor::Init: requires n = 2^m (power-of-two linear size).");
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
  
  size_t n = rows_;
  size_t scale_idx = 0;
  
  while (n > 1) {
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
      
      n = n / 2; // Reduce n for next iteration
    }
    
    scale_idx++;
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
      return ContractFinal1x1_(scales_.back().tens[0]);
  }

  // Reload dirty tensors from TN
  for (uint32_t id : dirty_scale0_) {
     auto coords = Coord_(id);
     scales_[0].tens[id] = tn({coords.first, coords.second});
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
              auto tl = SplitType0_(T0); 
              auto tr = SplitType1_(T1);
              auto bl = SplitType1_(T2);
              auto br = SplitType0_(T3);
              
              const Tensor& Q0 = tl.Q; // TL.Q
              const Tensor& Q1 = tr.Q; // TR.Q
              const Tensor& P1 = bl.P; // BL.P
              const Tensor& P0 = br.P; // BR.P

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
              
              coarse_layer.tens[c_id] = out;
          }
      }
      
      dirty_current = std::move(dirty_next);
  }

  // Final 1x1 Trace
  return ContractFinal1x1_(scales_.back().tens[0]);
}

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
    trial.amplitude = ContractFinal1x1_(scales_.back().tens[0]);
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
    auto it = trial.layer_updates[last].find(0);
    if (it != trial.layer_updates[last].end()) {
      trial.amplitude = ContractFinal1x1_(it->second);
      return trial;
    }
  }
  trial.amplitude = ContractFinal1x1_(scales_.back().tens[0]);
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

}  // namespace qlpeps

#endif  // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_IMPL_H
