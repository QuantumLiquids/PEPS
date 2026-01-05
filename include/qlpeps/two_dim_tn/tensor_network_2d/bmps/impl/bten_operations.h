// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2025-01
 *
 * Description: Unified BTen (Boundary Tensor) operations for BMPS contraction.
 * 
 * This file provides standalone static functions for BTen operations that can be
 * shared between BMPSContractor and BMPSWalker, following the DRY principle.
 *
 * BTen Index Conventions:
 * =======================
 * 
 * For LEFT BTen (3-rank bosonic, 4-rank fermionic):
 *   - index 0: connects to UP BMPS tensor's RIGHT index
 *   - index 1: connects to site tensor's LEFT index  
 *   - index 2: connects to DOWN BMPS tensor's LEFT index
 *   - index 3 (fermionic only): trivial parity index
 *
 * For RIGHT BTen (3-rank bosonic, 4-rank fermionic):
 *   - index 0: connects to DOWN BMPS tensor's RIGHT index
 *   - index 1: connects to site tensor's RIGHT index
 *   - index 2: connects to UP BMPS tensor's LEFT index
 *   - index 3 (fermionic only): trivial parity index
 *
 * MPS Storage Order (Critical!):
 * ==============================
 *   - UP BMPS: Reversed storage. bmps[0] = rightmost col, bmps[N-1] = leftmost col
 *   - DOWN BMPS: Normal storage. bmps[0] = leftmost col, bmps[N-1] = rightmost col
 *   - LEFT BMPS: Normal storage.
 *   - RIGHT BMPS: Reversed storage.
 */

#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BTEN_OPERATIONS_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BTEN_OPERATIONS_H

#include "qlten/qlten.h"
#include "qlpeps/basic.h"

namespace qlpeps {
namespace bten_ops {

using qlten::QLTensor;
using qlten::Index;
using qlten::QNSector;
using qlten::InverseIndex;
using qlten::Contract;

/**
 * @brief Create a vacuum BTen for LEFT position.
 *
 * The vacuum BTen represents an "empty" boundary at col=-1, ready to absorb col=0.
 * Its indices are constructed to be compatible with the first GrowBTenLeftStep.
 *
 * @param up_mps_col0 UP BMPS tensor at col=0 (from reversed storage: bmps[N-1])
 * @param site_col0 Site tensor at col=0 (mpo[0])
 * @param down_mps_col0 DOWN BMPS tensor at col=0 (opposite[0])
 * @return Vacuum BTen tensor
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> CreateVacuumBTenLeft(
    const QLTensor<TenElemT, QNT>& up_mps_col0,
    const QLTensor<TenElemT, QNT>& site_col0,
    const QLTensor<TenElemT, QNT>& down_mps_col0) {
  
  using Tensor = QLTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  
  // Vacuum BTen indices:
  // - index0: must match up_mps[2] for Contract(up_mps, bten, 2, 0, ...)
  // - index1: derived from site's LEFT index
  // - index2: derived from down_mps's LEFT index
  IndexT index0 = InverseIndex(up_mps_col0.GetIndex(2));   // UP col0 RIGHT
  IndexT index1 = InverseIndex(site_col0.GetIndex(0));     // site col0 LEFT  
  IndexT index2 = InverseIndex(down_mps_col0.GetIndex(0)); // DOWN col0 LEFT
  
  Tensor vacuum_bten;
  if constexpr (Tensor::IsFermionic()) {
    // Fermionic vacuum BTen needs extra trivial parity index
    auto qn = index0.GetQNSct(0).GetQn();
    auto qn0 = qn + (-qn);  // Zero QN
    auto trivial_index = IndexT({QNSector(qn0, 1)}, qlten::IN);
    vacuum_bten = Tensor({index0, index1, index2, trivial_index});
    vacuum_bten({0, 0, 0, 0}) = TenElemT(1.0);
    assert(vacuum_bten.Div().IsFermionParityEven() && 
           "Fermionic vacuum BTen must have even parity");
  } else {
    vacuum_bten = Tensor({index0, index1, index2});
    vacuum_bten({0, 0, 0}) = TenElemT(1.0);
  }
  
  return vacuum_bten;
}

/**
 * @brief Create a vacuum BTen for RIGHT position.
 *
 * The vacuum BTen represents an "empty" boundary at col=N, ready to absorb col=N-1.
 *
 * @param down_mps_colN1 DOWN BMPS tensor at col=N-1 (opposite[N-1])
 * @param site_colN1 Site tensor at col=N-1 (mpo[N-1])
 * @param up_mps_colN1 UP BMPS tensor at col=N-1 (from reversed storage: bmps[0])
 * @return Vacuum BTen tensor
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> CreateVacuumBTenRight(
    const QLTensor<TenElemT, QNT>& down_mps_colN1,
    const QLTensor<TenElemT, QNT>& site_colN1,
    const QLTensor<TenElemT, QNT>& up_mps_colN1) {
  
  using Tensor = QLTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  
  // RIGHT BTen indices for absorbing from right to left
  IndexT index0 = InverseIndex(down_mps_colN1.GetIndex(2)); // DOWN colN-1 RIGHT
  IndexT index1 = InverseIndex(site_colN1.GetIndex(2));     // site colN-1 RIGHT
  IndexT index2 = InverseIndex(up_mps_colN1.GetIndex(0));   // UP colN-1 LEFT
  
  Tensor vacuum_bten;
  if constexpr (Tensor::IsFermionic()) {
    auto qn = index0.GetQNSct(0).GetQn();
    auto qn0 = qn + (-qn);
    auto trivial_index = IndexT({QNSector(qn0, 1)}, qlten::IN);
    vacuum_bten = Tensor({index0, index1, index2, trivial_index});
    vacuum_bten({0, 0, 0, 0}) = TenElemT(1.0);
    assert(vacuum_bten.Div().IsFermionParityEven() &&
           "Fermionic vacuum BTen must have even parity");
  } else {
    vacuum_bten = Tensor({index0, index1, index2});
    vacuum_bten({0, 0, 0}) = TenElemT(1.0);
  }
  
  return vacuum_bten;
}

/**
 * @brief Grow LEFT BTen by absorbing one column.
 *
 * Contraction pattern (LEFT case):
 *   1. Contract up_mps[R=2] with bten[0]
 *   2. Contract result with site[L=0, R=2] (after transpose for fermionic)
 *   3. Contract result with down_mps[L=0, P=1]
 *
 * @param bten Current BTen (will absorb next column)
 * @param up_mps UP BMPS tensor for this column
 * @param site Site tensor for this column
 * @param down_mps DOWN BMPS tensor for this column
 * @return New BTen after absorption
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> GrowBTenLeftStep(
    const QLTensor<TenElemT, QNT>& bten,
    const QLTensor<TenElemT, QNT>& up_mps,
    QLTensor<TenElemT, QNT> site,  // non-const: may be transposed
    const QLTensor<TenElemT, QNT>& down_mps) {
  
  using Tensor = QLTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  
  Tensor next_bten;
  
  if constexpr (Tensor::IsFermionic()) {
    // Fermionic contraction with FuseIndex for correct fermion signs
    Tensor tmp1, tmp2, tmp3;
    Contract<TenElemT, QNT, true, true>(up_mps, bten, 2, 0, 1, tmp1);
    tmp1.FuseIndex(0, 5);
    site.Transpose({3, 0, 1, 2, 4});  // LEFT case transpose
    Contract<TenElemT, QNT, false, false>(tmp1, site, 2, 0, 2, tmp2);
    tmp2.FuseIndex(1, 5);
    Contract(&tmp2, {1, 3}, &down_mps, {0, 1}, &tmp3);
    tmp3.FuseIndex(0, 4);
    tmp3.Transpose({1, 2, 3, 0});
    next_bten = std::move(tmp3);
  } else {
    // Bosonic contraction
    Tensor tmp1, tmp2;
    Contract<TenElemT, QNT, true, true>(up_mps, bten, 2, 0, 1, tmp1);
    Contract<TenElemT, QNT, false, false>(tmp1, site, 1, 3, 2, tmp2);
    Contract(&tmp2, {0, 2}, &down_mps, {0, 1}, &next_bten);
  }
  
  return next_bten;
}

/**
 * @brief Grow RIGHT BTen by absorbing one column (from right to left).
 *
 * Contraction pattern (RIGHT case):
 *   1. Contract down_mps[R=2] with bten[0]
 *   2. Contract result with site[D=1] (no transpose needed)
 *   3. Contract result with up_mps[L=0, P=1]
 *
 * Note: Unlike LEFT case, RIGHT case does NOT transpose the site tensor.
 * The contraction indices are different: site[D=1] instead of site[L=0].
 *
 * @param bten Current BTen
 * @param down_mps DOWN BMPS tensor for this column
 * @param site Site tensor for this column (const - no transpose needed)
 * @param up_mps UP BMPS tensor for this column
 * @return New BTen after absorption
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> GrowBTenRightStep(
    const QLTensor<TenElemT, QNT>& bten,
    const QLTensor<TenElemT, QNT>& down_mps,
    const QLTensor<TenElemT, QNT>& site,  // const: no transpose for RIGHT
    const QLTensor<TenElemT, QNT>& up_mps) {
  
  using Tensor = QLTensor<TenElemT, QNT>;
  Tensor next_bten;
  
  if constexpr (Tensor::IsFermionic()) {
    // Matches GrowFullBTen RIGHT case exactly
    Tensor tmp1, tmp2, tmp3;
    Contract<TenElemT, QNT, true, true>(down_mps, bten, 2, 0, 1, tmp1);
    tmp1.FuseIndex(0, 5);
    Contract<TenElemT, QNT, false, false>(tmp1, site, 2, 1, 2, tmp2);  // site[D=1]
    tmp2.FuseIndex(1, 4);  // Note: 4, not 5
    Contract(&tmp2, {1, 3}, &up_mps, {0, 1}, &tmp3);
    tmp3.FuseIndex(0, 4);
    tmp3.Transpose({1, 2, 3, 0});
    next_bten = std::move(tmp3);
  } else {
    Tensor tmp1, tmp2;
    Contract<TenElemT, QNT, true, true>(down_mps, bten, 2, 0, 1, tmp1);
    Contract<TenElemT, QNT, false, false>(tmp1, site, 1, 1, 2, tmp2);
    Contract(&tmp2, {0, 2}, &up_mps, {0, 1}, &next_bten);
  }
  
  return next_bten;
}

/**
 * @brief Compute trace using LEFT BTen, MPS tensors, site, and RIGHT BTen.
 *
 * This matches BMPSContractor::ReplaceOneSiteTrace contraction pattern.
 * Contracts: up_mps -- left_bten -- site -- down_mps -- right_bten
 *
 * For HORIZONTAL orientation (row contraction):
 *   - up_mps: UP BMPS tensor at this column
 *   - down_mps: DOWN BMPS tensor at this column
 *   - left_bten: LEFT BTen covering [0, col)
 *   - right_bten: RIGHT BTen covering (col, N-1]
 *   - site: Site tensor at col
 *
 * @return Scalar trace value
 */
template<typename TenElemT, typename QNT>
TenElemT TraceBTen(
    const QLTensor<TenElemT, QNT>& up_mps,
    const QLTensor<TenElemT, QNT>& left_bten,
    const QLTensor<TenElemT, QNT>& site,
    const QLTensor<TenElemT, QNT>& down_mps,
    const QLTensor<TenElemT, QNT>& right_bten) {
  
  using Tensor = QLTensor<TenElemT, QNT>;
  Tensor tmp[4];
  
  if constexpr (Tensor::IsFermionic()) {
    // Fermionic contraction with FuseIndex
    Contract<TenElemT, QNT, true, true>(up_mps, left_bten, 2, 0, 1, tmp[0]);
    tmp[0].FuseIndex(0, 5);
    Contract(tmp, {2, 3}, &site, {3, 0}, tmp + 1);
    tmp[1].FuseIndex(0, 5);
    Contract<TenElemT, QNT, false, true>(tmp[1], down_mps, 2, 0, 2, tmp[2]);
    tmp[2].FuseIndex(1, 4);
    Contract(tmp + 2, {3, 1, 2}, &right_bten, {0, 1, 2}, tmp + 3);
    return tmp[3].GetElem({0, 0});
  } else {
    // Bosonic contraction
    Contract<TenElemT, QNT, true, true>(up_mps, left_bten, 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, false>(tmp[0], site, 1, 3, 2, tmp[1]);
    Contract(&tmp[1], {0, 2}, &down_mps, {0, 1}, &tmp[2]);
    Contract(&tmp[2], {0, 1, 2}, &right_bten, {2, 1, 0}, &tmp[3]);
    return tmp[3]();
  }
}

} // namespace bten_ops
} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BTEN_OPERATIONS_H

