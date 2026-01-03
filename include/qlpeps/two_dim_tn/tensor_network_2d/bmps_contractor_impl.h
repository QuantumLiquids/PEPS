// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-12-11
*
* Description: QuantumLiquids/PEPS project. The BMPS Contractor implementation.
*/

#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_BMPS_CONTRACTOR_IMPL_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_BMPS_CONTRACTOR_IMPL_H

#include "bmps_contractor.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h" // For TensorNetwork2D definition

namespace qlpeps {

using qlmps::IN;
using qlmps::OUT;

// ============================================================================
//                    BMPS CONTRACTOR CONVENTIONS & REFERENCE
// ============================================================================
//
// This document explains the critical index conventions, storage orders, and
// contraction patterns used throughout BMPSContractor. Understanding these is
// essential for correctly implementing new features.
//
// ============================================================================
// 1. TENSOR INDEX CONVENTIONS
// ============================================================================
//
// ## 1.1 Site Tensor (TN node, 4-leg tensor)
//
//            U (index 3)
//            |
//     L -----+------ R
//   (idx 0)  |    (idx 2)
//            D (index 1)
//
// Indices: (L=0, D=1, R=2, U=3)
// Physical meaning: virtual bonds connecting to neighboring sites in 2D lattice.
//
// ## 1.2 BMPS Tensor (Boundary MPS tensor, 3-leg tensor)
//
//       P (index 1)
//       |
//  L ---+--- R
// (0)       (2)
//
// Indices: (L=0, P=1, R=2)
//
// NAMING CLARIFICATION:
// - From MPS perspective: index 1 is the "physical" index of the MPS.
// - From PEPS/TN2D perspective: index 1 is still a VIRTUAL bond that connects
//   to a site tensor's virtual index (e.g., U or D).
// - We use "P" for brevity, but remember it's NOT a true physical index
//   of the underlying quantum system.
//
// Connection example (DOWN BMPS at row r):
//   BMPS_tensor[P=1] connects to site_tensor[D=1] at row r.
//
// ============================================================================
// 2. BMPS STORAGE ORDER (CRITICAL!)
// ============================================================================
//
// Different BMPS positions have different storage conventions:
//
// ## 2.1 DOWN BMPS (Normal Order)
//
//   Storage:  bmps[0]  bmps[1]  bmps[2]  ...  bmps[N-1]
//   Columns:    0        1        2             N-1
//
//   bmps[i] corresponds to column i. Left-to-right storage.
//
// ## 2.2 UP BMPS (Reversed Order!) *** IMPORTANT ***
//
//   Storage:  bmps[0]  bmps[1]  bmps[2]  ...  bmps[N-1]
//   Columns:   N-1      N-2      N-3            0
//
//   bmps[i] corresponds to column (N-1-i). Right-to-left storage!
//
// Why reversed? The UP BMPS is constructed by absorbing rows from top to bottom,
// and the internal MPS bond direction naturally reverses compared to DOWN.
//
// ## 2.3 Mapping between storage index and column index
//
// For N columns (0 to N-1):
//   - DOWN BMPS: storage_idx = col
//   - UP BMPS:   storage_idx = N - 1 - col
//
// ============================================================================
// 3. BTen (BOUNDARY TENSOR) STRUCTURE
// ============================================================================
//
// BTen represents the partially contracted environment along a row or column.
// Used for efficient trace calculations when scanning across multiple sites.
//
// IMPORTANT: Opposite BTens have REVERSED index ordering by design!
// See bmps_contractor.h for complete documentation.
//
// Summary for horizontal (LEFT/RIGHT) pair:
//   LEFT BTen:  [0]=UP_R,   [1]=site_L, [2]=DOWN_L  (order: TOP→BOTTOM)
//   RIGHT BTen: [0]=DOWN_R, [1]=site_R, [2]=UP_L    (order: BOTTOM→TOP, reversed!)
//
// When contracting opposite BTens, use reversed index matching:
//   Contract(&left_result, {0, 1, 2}, &right_result, {2, 1, 0}, &scalar);
//
// ## 3.1 LEFT BTen (grows from left to right)
//
//       +---------+
//       | LEFT    |====> index 0: UP[R]   (top boundary)
//       | BTen    |====> index 1: site[L] (middle, site tensor)
//       |         |====> index 2: DOWN[L] (bottom boundary)
//       +---------+
//
// ## 3.2 RIGHT BTen (grows from right to left)
//
//                       +-------+
//   UP[L=0]   <---[2]---|       |   (connects to UP boundary)
//   site[R=2] <---[1]---| RIGHT |   (connects to site tensor)
//   DOWN[R=2] <---[0]---| BTen  |   (connects to DOWN boundary)
//                       +-------+
//
// ============================================================================
// 4. CONTRACTION PATTERNS (for GrowFullBTen)
// ============================================================================
//
// ## 4.1 LEFT case (bosonic):
//
//   Contraction Order:
//   1. UP_BMPS[R]  connects with  BTen[0]      -> forms Tmp1
//   2. Tmp1        connects with  Site[U, L]   -> forms Tmp2
//   3. Tmp2        connects with  DOWN_BMPS[L] -> New BTen
//
// ## 4.2 RIGHT case (bosonic):
//
//   Contraction Order:
//   1. DOWN_BMPS[R] connects with  BTen[0]      -> forms Tmp1
//   2. Tmp1         connects with  Site[D, R]   -> forms Tmp2
//   3. Tmp2         connects with  UP_BMPS[L]   -> New BTen
//
// ============================================================================
// 5. ASCII DIAGRAM: Row Contraction Alignment
// ============================================================================
//
// Alignment of vector indices vs physical columns (for N=4):
//
//   Physical Col:     0            1            2            3
//   (Left -> Right)
//
//   UP BMPS vec:    [3]          [2]          [1]          [0]    (Reversed Storage!)
//                    |            |            |            |
//   Site Tensor:    [0]          [1]          [2]          [3]    (Normal Storage)
//                    |            |            |            |
//   DOWN BMPS vec:  [0]          [1]          [2]          [3]    (Normal Storage)
//
//
// LEFT BTen grows from left (col 0 -> 3):
//
//   +------+ 
//   |      |--- [UP vec 3]
//   | LEFT |--- [Site 0  ] ...
//   | BTen |--- [DOWN 0  ]
//   |  0   |
//   |      |
//   +------+
//
// RIGHT BTen grows from right (col 3 -> 0):
//
//                                            +-------+   
//                               [UP vec 0]---|       |
//                          ...  [Site 3  ]---| RIGHT |
//                          ...  [DOWN 3  ]---| BTen  |
//                                            |  0    |
//                                            +-------+
//
// ============================================================================

namespace {
template<bool is_fermion>
inline std::vector<size_t> GenMpoTen1TransposeAxesForBrowBTen2(const size_t post) {
  if constexpr (is_fermion) {
    return {(post + 3) % 4,
            (post) % 4,
            (post + 2) % 4,
            (post + 1) % 4,
            4};
  } else {
    return {(post + 3) % 4,
            (post) % 4,
            (post + 2) % 4,
            (post + 1) % 4};
  }
}

inline std::vector<size_t> GenMpoTen2TransposeAxesForFermionGrowBTen2(const size_t post) {
  return {(post + 3) % 4,
          (post) % 4,
          (post + 1) % 4,
          (post + 2) % 4,
          4};
}

/**
 * Helper for GrowBTen2Step.
 *
 * @tparam QNT : use for distinguish fermion or boson
 */
template<typename QNT>
void SetUpCoordInfoForGrowBTen2(const BTenPOSITION post,
                                const size_t slice_num1,
                                const size_t bten_size,
                                const size_t rows,
                                const size_t cols,
                                size_t &N, //outputs start
                                SiteIdx &grown_site1,
                                SiteIdx &grown_site2,
                                size_t &mps1_idx,
                                size_t &mps2_idx,
                                std::vector<size_t> &mpo1_transpose_axes) {
  switch (post) {
    case DOWN: {
      const size_t col = slice_num1;
      N = rows;
      grown_site1 = {N - bten_size, col};
      grown_site2 = {N - bten_size, col + 1};
      mps1_idx = col;
      mps2_idx = cols - 1 - (col + 1);
      break;
    }
    case UP: {
      const size_t col = slice_num1;
      N = rows;
      grown_site1 = {bten_size - 1, col + 1};
      grown_site2 = {bten_size - 1, col};
      mps1_idx = cols - 1 - (col + 1);
      mps2_idx = col;
      break;
    }
    case LEFT: {
      const size_t row = slice_num1;
      N = cols;
      grown_site1 = {row, bten_size - 1};
      grown_site2 = {row + 1, bten_size - 1};
      mps1_idx = row;
      mps2_idx = rows - 1 - (row + 1);
      break;
    }
    case RIGHT: {
      const size_t row = slice_num1;
      N = cols;
      grown_site1 = {row + 1, N - bten_size};
      grown_site2 = {row, N - bten_size};
      mps1_idx = rows - 1 - (row + 1);
      mps2_idx = row;
      break;
    }
  }
  mpo1_transpose_axes = GenMpoTen1TransposeAxesForBrowBTen2<Index<QNT>::IsFermionic()>(post);
}

/*
 * e.g. bten_position = LEFT
 *
 *           ++-------mps_ten1---
 *           ||          |
 *           ||          |
 *  BTEN-LEFT||-------mpo_ten----
 *           ||          |
 *           ||          |
 *           ++-------mps_ten2---
 *  mpo_ten is the original mpo which haven't been transposed
*/
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> FermionGrowBTenStep(
    const BTenPOSITION bten_position,
    const QLTensor<TenElemT, QNT> &bten,
    const QLTensor<TenElemT, QNT> &mps_ten1,
    QLTensor<TenElemT, QNT> mpo_ten,
    const QLTensor<TenElemT, QNT> &mps_ten2
) {
  switch (bten_position) {
    case LEFT : {
      mpo_ten.Transpose({3, 0, 1, 2, 4});
      break;
    }
    case DOWN : {
      break;
    }
    case RIGHT : {
      mpo_ten.Transpose({1, 2, 3, 0, 4});
      break;
    }
    case UP : {
      mpo_ten.Transpose({2, 3, 0, 1, 4});
    }
  }
  QLTensor<TenElemT, QNT> tmp1, tmp2, next_bten;
  Contract<TenElemT, QNT, true, true>(mps_ten1, bten, 2, 0, 1, tmp1);
  tmp1.FuseIndex(0, 5);
  Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 2, 0, 2, tmp2);
  tmp2.FuseIndex(1, 5);
  Contract(&tmp2, {1, 3}, &mps_ten2, {0, 1}, &next_bten);
  next_bten.FuseIndex(0, 4);
  next_bten.Transpose({1, 2, 3, 0});
  return next_bten;
}

/**
 * Helper for GrowBTen2Step.
 * It defines the final contraction for the GrowBTen2Step.
 * The indices of bten2, mps_ten1 and mps_ten2 follow the original tensors indices order.
 * The mpo_ten1 needs to be transposed as defined in GrowBTen2Step.
 * For bosonic tensor, mpo_ten2 is original mpo tensor;
 * while for fermionic tensor, mpo_ten2 should be tranposed as defined in GrowBTen2Step.
 * Parameter ctrct_mpo_start_idx only useful for bosonic case.
 * @return
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> GrowBTen2StepAfterTransposedMPOTens(
    const QLTensor<TenElemT, QNT> &bten2,  //input boundary tensor 2
    const QLTensor<TenElemT, QNT> &mps_ten1,
    const QLTensor<TenElemT, QNT> &mps_ten2,
    const QLTensor<TenElemT, QNT> &mpo_ten1,
    const QLTensor<TenElemT, QNT> &mpo_ten2,
    const size_t ctrct_mpo_start_idx) {
  using Tensor = QLTensor<TenElemT, QNT>;
  Tensor next_bten2; //return
  Tensor tmp1, tmp2, tmp3, tmp4;
  if constexpr (Tensor::IsFermionic()) {
    Contract<TenElemT, QNT, true, true>(mps_ten1, bten2, 2, 0, 1, tmp1);
    tmp1.FuseIndex(0, 6);
    Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 2, 0, 2, tmp2);
    tmp2.FuseIndex(2, 6);
    tmp2.Transpose({1, 0, 2, 3, 4, 5});
    Contract<TenElemT, QNT, true, true>(tmp2, mpo_ten2, 5, 0, 2, tmp3);
    tmp3.FuseIndex(0, 6);
    Contract(&tmp3, {1, 4}, &mps_ten2, {0, 1}, &next_bten2);
    next_bten2.FuseIndex(0, 5);
    next_bten2.Transpose({1, 2, 3, 4, 0});
  } else {
    Contract<TenElemT, QNT, true, true>(mps_ten1, bten2, 2, 0, 1, tmp1);
    Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
    Contract<TenElemT, QNT, false, false>(tmp2, mpo_ten2, 4, ctrct_mpo_start_idx, 2, tmp3);
    Contract(&tmp3, {0, 3}, &mps_ten2, {0, 1}, &next_bten2);
  }
  return next_bten2;
}
}

template<typename TenElemT, typename QNT>
BMPSContractor<TenElemT, QNT>::BMPSContractor(size_t rows, size_t cols)
    : rows_(rows), cols_(cols) {
  for (size_t post_int = 0; post_int < 4; post_int++) {
    const BMPSPOSITION post = static_cast<BMPSPOSITION>(post_int);
    bmps_set_.insert(std::make_pair(post, std::vector<BMPS<TenElemT, QNT>>()));
    // We can reserve based on rows/cols, but exact max depends on direction
    size_t max_len = (post == UP || post == DOWN) ? rows : cols;
    bmps_set_[post].reserve(max_len);

    bten_set_.insert(std::make_pair(static_cast<BTenPOSITION>(post_int), std::vector<Tensor>()));
    bten_set2_.insert(std::make_pair(static_cast<BTenPOSITION>(post_int), std::vector<Tensor>()));
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::Init(const TensorNetwork2D<TenElemT, QNT>& tn) {
  assert(tn.rows() == rows_ && tn.cols() == cols_);
  for (size_t post_int = 0; post_int < 4; post_int++) {
    bmps_set_[static_cast<BMPSPOSITION>(post_int)].clear();
    InitBMPS(tn, static_cast<BMPSPOSITION>(post_int));
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::InitBMPS(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION post) {
  assert(tn.GetBoundaryCondition() == BoundaryCondition::Open && "BMPS initialization is only valid for Open Boundary Condition");
  assert(bmps_set_.at(post).empty());
  const size_t mps_size = tn.length(Rotate(Orientation(post)));
  std::vector<Index<QNT>> boundary_indices;
  boundary_indices.reserve(mps_size);

  switch (post) {
    case LEFT : {
      for (size_t i = 0; i < mps_size; i++) {
        boundary_indices.push_back(InverseIndex(tn({i, 0}).GetIndex(post)));
      }
      break;
    }
    case DOWN: {
      for (size_t i = 0; i < mps_size; i++) {
        boundary_indices.push_back(InverseIndex(tn({tn.rows() - 1, i}).GetIndex(post)));
      }
      break;
    }
    case RIGHT: {
      for (size_t i = 0; i < mps_size; i++) {
        boundary_indices.push_back(InverseIndex(tn({tn.rows() - i - 1, tn.cols() - 1}).GetIndex(post)));
      }
      break;
    }
    case UP: {
      for (size_t i = 0; i < mps_size; i++) {
        boundary_indices.push_back(InverseIndex(tn({0, tn.cols() - i - 1}).GetIndex(post)));
      }
      break;
    }
  }
  BMPS<TenElemT, QNT> boundary_bmps(post, boundary_indices);
  bmps_set_[post].push_back(boundary_bmps);
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
BMPSContractor<TenElemT, QNT>::GenerateBMPSApproach(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION post, const BMPSTruncateParams<RealT> &trunc_para) {
  DeleteInnerBMPS(post);
  GrowFullBMPS(tn, Opposite(post), trunc_para);
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
size_t BMPSContractor<TenElemT, QNT>::GrowBMPSStep(const BMPSPOSITION position,
                                                    TransferMPO mpo,
                                                    const BMPSTruncateParams<RealT> &trunc_para) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  bmps_set.push_back(
      bmps_set.back().MultipleMPO(mpo, trunc_para.compress_scheme,
                                  trunc_para.D_min, trunc_para.D_max, trunc_para.trunc_err,
                                  trunc_para.convergence_tol,
                                  trunc_para.iter_max));
  return bmps_set.size();
}

template<typename TenElemT, typename QNT>
size_t BMPSContractor<TenElemT, QNT>::GrowBMPSStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  size_t existed_bmps_num = bmps_set.size();
  assert(existed_bmps_num > 0);
  size_t mpo_num;
  if (position == UP || position == LEFT) {
    mpo_num = existed_bmps_num - 1;
  } else if (position == DOWN) {
    mpo_num = tn.rows() - existed_bmps_num;
  } else { //RIGHT
    mpo_num = tn.cols() - existed_bmps_num;
  }
  const TransferMPO &mpo = tn.get_slice(mpo_num, Rotate(Orientation(position)));
  return GrowBMPSStep(position, mpo, trunc_para);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::GrowFullBMPS(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  size_t existed_bmps_size = bmps_set.size();
  assert(existed_bmps_size > 0);
  size_t rows = tn.rows();
  size_t cols = tn.cols();
  switch (position) {
    case DOWN: {
      for (size_t row = rows - existed_bmps_size; row > 0; row--) {
        const TransferMPO &mpo = tn.get_row(row);
        GrowBMPSStep(position, mpo, trunc_para);
      }
      break;
    }
    case UP: {
      for (size_t row = existed_bmps_size - 1; row < rows - 1; row++) {
        const TransferMPO &mpo = tn.get_row(row);
        GrowBMPSStep(position, mpo, trunc_para);
      }
      break;
    }
    case LEFT: {
      for (size_t col = existed_bmps_size - 1; col < cols - 1; col++) {
        const TransferMPO &mpo = tn.get_col(col);
        GrowBMPSStep(position, mpo, trunc_para);
      }
      break;
    }
    case RIGHT: {
      for (size_t col = cols - existed_bmps_size; col > 0; col--) {
        const TransferMPO &mpo = tn.get_col(col);
        GrowBMPSStep(position, mpo, trunc_para);
      }
      break;
    }
  }
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
BMPSContractor<TenElemT, QNT>::GrowBMPSForRow(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t row, const BMPSTruncateParams<RealT> &trunc_para) {
  const size_t rows = tn.rows();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set_down = bmps_set_[DOWN];
  for (size_t row_bmps = rows - bmps_set_down.size(); row_bmps > row; row_bmps--) {
    const TransferMPO &mpo = tn.get_row(row_bmps);
    GrowBMPSStep(DOWN, mpo, trunc_para);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_up = bmps_set_[UP];
  for (size_t row_bmps = bmps_set_up.size() - 1; row_bmps < row; row_bmps++) {
    const TransferMPO &mpo = tn.get_row(row_bmps);
    GrowBMPSStep(UP, mpo, trunc_para);
  }
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
BMPSContractor<TenElemT, QNT>::GrowBMPSForCol(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t col, const BMPSTruncateParams<RealT> &trunc_para) {
  const size_t cols = tn.cols();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set_right = bmps_set_[RIGHT];
  for (size_t col_bmps = cols - bmps_set_right.size(); col_bmps > col; col_bmps--) {
    const TransferMPO &mpo = tn.get_col(col_bmps);
    GrowBMPSStep(RIGHT, mpo, trunc_para);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_left = bmps_set_[LEFT];
  for (size_t col_bmps = bmps_set_left.size() - 1; col_bmps < col; col_bmps++) {
    const TransferMPO &mpo = tn.get_col(col_bmps);
    GrowBMPSStep(LEFT, mpo, trunc_para);
  }
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
const std::pair<BMPS<TenElemT, QNT>, BMPS<TenElemT, QNT> >
BMPSContractor<TenElemT, QNT>::GetBMPSForRow(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t row, const BMPSTruncateParams<RealT> &trunc_para) {
  const size_t rows = tn.rows();
  GrowBMPSForRow(tn, row, trunc_para);
  BMPST &up_bmps = bmps_set_[UP][row];
  BMPST &down_bmps = bmps_set_[DOWN][rows - 1 - row];

  return std::pair(up_bmps, down_bmps);
}

template<typename TenElemT, typename QNT>
const std::pair<BMPS<TenElemT, QNT>, BMPS<TenElemT, QNT> >
BMPSContractor<TenElemT, QNT>::GetBMPSForCol(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t col, const BMPSTruncateParams<RealT> &trunc_para) {
  const size_t cols = tn.cols();
  GrowBMPSForCol(tn, col, trunc_para);
  BMPST &left_bmps = bmps_set_[LEFT][col];
  BMPST &right_bmps = bmps_set_[RIGHT][cols - 1 - col];
  return std::pair(left_bmps, right_bmps);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::ShiftBMPSWindow(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para) {
  bmps_set_[position].pop_back();
  auto oppo_post = Opposite(position);
  GrowBMPSStep(tn, oppo_post, trunc_para);
}

template<typename TenElemT, typename QNT>
typename BMPSContractor<TenElemT, QNT>::Tensor 
BMPSContractor<TenElemT, QNT>::PunchHole(const TensorNetwork2D<TenElemT, QNT>& tn, const qlpeps::SiteIdx &site,
                                         const qlpeps::BondOrientation mps_orient) const {
  const Tensor *left_ten, *down_ten, *right_ten, *up_ten;
  const size_t row = site[0];
  const size_t col = site[1];
  
#ifndef NDEBUG
  if (mps_orient == HORIZONTAL) {
    up_ten = &(bmps_set_.at(UP).at(row)[tn.cols() - col - 1]);
    down_ten = &(bmps_set_.at(DOWN).at(tn.rows() - row - 1)[col]);
    left_ten = &(bten_set_.at(LEFT).at(col));
    right_ten = &(bten_set_.at(RIGHT).at(tn.cols() - col - 1));
  } else {
    up_ten = &(bten_set_.at(UP).at(row));
    down_ten = &(bten_set_.at(DOWN).at(tn.rows() - row - 1));
    left_ten = &(bmps_set_.at(LEFT).at(col)[row]);
    right_ten = &(bmps_set_.at(RIGHT).at(tn.cols() - col - 1)[tn.rows() - row - 1]);
  }
#else
  if (mps_orient == HORIZONTAL) {
    up_ten = &(bmps_set_.at(UP)[row][tn.cols() - col - 1]);
    down_ten = &(bmps_set_.at(DOWN)[tn.rows() - row - 1][col]);
    left_ten = &(bten_set_.at(LEFT)[col]);
    right_ten = &(bten_set_.at(RIGHT)[tn.cols() - col - 1]);
  } else {
    up_ten = &(bten_set_.at(UP)[row]);
    down_ten = &(bten_set_.at(DOWN)[tn.rows() - row - 1]);
    left_ten = &(bmps_set_.at(LEFT)[col][row]);
    right_ten = &(bmps_set_.at(RIGHT)[tn.cols() - col - 1][tn.rows() - row - 1]);
  }
#endif
  Tensor tmp1, tmp2, res_ten;
  if constexpr (Tensor::IsFermionic()) {
    Contract<TenElemT, QNT, false, true>(*left_ten, *down_ten, 2, 0, 1, tmp1);
    tmp1.FuseIndex(0, 5);
    Contract<TenElemT, QNT, true, true>(tmp1, *right_ten, 4, 0, 1, tmp2);
    tmp2.FuseIndex(0, 6);
    Contract(&tmp2, {5, 1}, up_ten, {0, 2}, &res_ten);
    res_ten.FuseIndex(0, 5);// the first index is the trivial index
  } else {
    Contract(left_ten, {2}, down_ten, {0}, &tmp1);
    Contract(right_ten, {2}, up_ten, {0}, &tmp2);
    Contract(&tmp1, {0, 3}, &tmp2, {3, 0}, &res_ten);
  }
  return res_ten;
}

template<typename TenElemT, typename QNT>
bool BMPSContractor<TenElemT, QNT>::DirectionCheck() const {
  for (const auto &[direction, bmps_vec] : bmps_set_) {
    for (const BMPS<TenElemT, QNT> &bmps : bmps_vec) {
      if (bmps.Direction() != direction) {
        std::cout << "direction : " << direction << std::endl;
        std::cout << "bmps.Direction() : " << bmps.Direction() << std::endl;
        return false;
      }
      assert(bmps.Direction() == direction);
    }
  }
  return true;
}

// --------------------------------------------------------
// BTen operations
// --------------------------------------------------------

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::InitBTen(const TensorNetwork2D<TenElemT, QNT>& tn, const qlpeps::BTenPOSITION position, const size_t slice_num) {
  bten_set_[position].clear();
  using IndexT = Index<QNT>;
  IndexT index0, index1, index2;
  switch (position) {
    case DOWN: {
      const size_t col = slice_num;
      index0 = InverseIndex(bmps_set_[LEFT][col](tn.rows() - 1)->GetIndex(2));
      index1 = InverseIndex(tn({tn.rows() - 1, col}).GetIndex(position));
      index2 = InverseIndex(bmps_set_[RIGHT][tn.cols() - col - 1](0)->GetIndex(0));
      break;
    }
    case UP: {
      const size_t col = slice_num;
      index0 = InverseIndex(bmps_set_[RIGHT][tn.cols() - col - 1](tn.rows() - 1)->GetIndex(2));
      index1 = InverseIndex(tn({0, col}).GetIndex(position));
      index2 = InverseIndex(bmps_set_[LEFT][col](0)->GetIndex(0));
      break;
    }
    case LEFT: {
      const size_t row = slice_num;
      index0 = InverseIndex(bmps_set_[UP][row](tn.cols() - 1)->GetIndex(2));
      index1 = InverseIndex(tn({row, 0}).GetIndex(position));
      index2 = InverseIndex(bmps_set_[DOWN][tn.rows() - row - 1](0)->GetIndex(0));
      break;
    }
    case RIGHT: {
      const size_t row = slice_num;
      index0 = InverseIndex(bmps_set_[DOWN][tn.rows() - row - 1](tn.cols() - 1)->GetIndex(2));
      index1 = InverseIndex(tn({row, tn.cols() - 1}).GetIndex(position));
      index2 = InverseIndex(bmps_set_[UP][row](0)->GetIndex(0));
      break;
    }
  }
  Tensor ten;
  if constexpr (Tensor::IsFermionic()) {
    auto qn = index0.GetQNSct(0).GetQn();
    auto qn0 = qn + (-qn);
    auto trivial_index_out = Index<QNT>({QNSector(qn0, 1)}, IN);
    ten = Tensor({index0, index1, index2, trivial_index_out});
    ten({0, 0, 0, 0}) = TenElemT(1.0);
    assert(ten.Div().IsFermionParityEven());
  } else {
    ten = Tensor({index0, index1, index2});
    ten({0, 0, 0}) = TenElemT(1.0);
  }
  bten_set_[position].emplace_back(ten);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::TruncateBTen(const qlpeps::BTenPOSITION position, const size_t length) {
  auto &btens = bten_set_.at(position);
  if (btens.size() > length) {
    btens.resize(length);
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::InitBTen2(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1) {
  bten_set2_[position].clear();
  using IndexT = Index<QNT>;
  IndexT index0, index1, index2, index3;

  switch (position) {
    case DOWN: {
      const size_t col1 = slice_num1;
      const size_t col2 = col1 + 1;
      index0 = InverseIndex(bmps_set_[LEFT][col1](tn.rows() - 1)->GetIndex(2));
      index1 = InverseIndex(tn({tn.rows() - 1, col1}).GetIndex(position));
      index2 = InverseIndex(tn({tn.rows() - 1, col2}).GetIndex(position));
      index3 = InverseIndex(bmps_set_[RIGHT][tn.cols() - col2 - 1](0)->GetIndex(0));
      break;
    }
    case UP: {
      const size_t col1 = slice_num1;
      const size_t col2 = col1 + 1;
      index0 = InverseIndex(bmps_set_[RIGHT][tn.cols() - col2 - 1](tn.rows() - 1)->GetIndex(2));
      index1 = InverseIndex(tn({0, col2}).GetIndex(position));
      index2 = InverseIndex(tn({0, col1}).GetIndex(position));
      index3 = InverseIndex(bmps_set_[LEFT][col1](0)->GetIndex(0));
      break;
    }
    case LEFT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      index0 = InverseIndex(bmps_set_[UP][row1](tn.cols() - 1)->GetIndex(2));
      index1 = InverseIndex(tn({row1, 0}).GetIndex(position));
      index2 = InverseIndex(tn({row2, 0}).GetIndex(position));
      index3 = InverseIndex(bmps_set_[DOWN][tn.rows() - row2 - 1](0)->GetIndex(0));
      break;
    }
    case RIGHT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      index0 = InverseIndex(bmps_set_[DOWN][tn.rows() - row2 - 1](tn.cols() - 1)->GetIndex(2));
      index1 = InverseIndex(tn({row2, tn.cols() - 1}).GetIndex(position));
      index2 = InverseIndex(tn({row1, tn.cols() - 1}).GetIndex(position));
      index3 = InverseIndex(bmps_set_[UP][row1](0)->GetIndex(0));
      break;
    }
  }
  Tensor ten;
  if constexpr (Tensor::IsFermionic()) {
    auto qn = index0.GetQNSct(0).GetQn();
    auto qn0 = qn + (-qn);
    auto trivial_index_out = Index<QNT>({QNSector(qn0, 1)}, IN);
    ten = Tensor({index0, index1, index2, index3, trivial_index_out});
    ten({0, 0, 0, 0, 0}) = TenElemT(1.0);
  } else {
    ten = Tensor({index0, index1, index2, index3});
    ten({0, 0, 0, 0}) = TenElemT(1.0);
  }
  bten_set2_[position].emplace_back(ten);
}

/**
 * @brief Grows the boundary tensor (BTen) by absorbing sites along a slice.
 *
 * ## Contraction Pattern for LEFT position (bosonic):
 *
 * @verbatim
 *   +-------+      up_mps[R,P]      +-------+
 *   | BTen  |=========*=============| BTen' |
 *   | (k)   |      site[U,L,D]      | (k+1) |
 *   |       |=========*=============|       |
 *   |       |     down_mps[L,P]     |       |
 *   |       |=========*=============|       |
 *   +-------+                       +-------+
 *
 *   Contract order:
 *   1. up_mps[R=2] with BTen[0]
 *   2. Result with site[U=3,L=0]  
 *   3. Result with down_mps[L=0,P=1]
 *   
 *   Key: UP BMPS is reversed, so up_mps[R] connects to BTen[0] (pointing left).
 * @endverbatim
 *
 * ## Contraction Pattern for RIGHT position (bosonic):
 *
 * RIGHT BTen grows from right to left (absorbing columns N-1, N-2, ...).
 *
 * @verbatim
 *   +-------+    up_mps[L,P]     +-------+
 *   | BTen' |=========*==========| BTen  |
 *   | (k+1) |     site[D,R,U]    | (k)   |   <-- BTen(k) is on the RIGHT
 *   |       |=========*==========|       |
 *   |       |   down_mps[P,R]    |       |
 *   |       |=========*==========|       |
 *   +-------+                    +-------+
 *
 *   Contract order:
 *   1. down_mps[R=2] with BTen[0]
 *   2. Result with site[D=1,R=2]
 *   3. Result with up_mps[L=0,P=1]
 * @endverbatim
 *
 * @note The UP BMPS uses reversed storage (see file header for details).
 */
template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::GrowFullBTen(const TensorNetwork2D<TenElemT, QNT>& tn,
                                                  const qlpeps::BTenPOSITION position,
                                                  const size_t slice_num,
                                                  const size_t remain_sites,
                                                  bool init) {
  if (init) {
    InitBTen(tn, position, slice_num);
  }
  size_t start_idx = bten_set_[position].size() - 1;
  std::vector<Tensor> &btens = bten_set_[position];
  switch (position) {
    case DOWN: {
      const size_t col = slice_num;
      const TransferMPO &mpo = tn.get_col(col);
      const size_t N = mpo.size(); // tn.cols()
      const size_t end_idx = N - remain_sites;
      auto &left_bmps = bmps_set_[LEFT][col];
      auto &right_bmps = bmps_set_[RIGHT][tn.cols() - col - 1];
      for (size_t i = start_idx; i < end_idx; i++) {
        auto &left_mps_ten = left_bmps[N - i - 1];
        auto &right_mps_ten = right_bmps[i];
        auto &mpo_ten = *mpo[N - i - 1];
        Tensor tmp1, tmp2, tmp3;
        if constexpr (Tensor::IsFermionic()) {
          Contract<TenElemT, QNT, true, true>(left_mps_ten, btens.back(), 2, 0, 1, tmp1);
          tmp1.FuseIndex(0, 5);
          Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 2, 0, 2, tmp2);
          tmp2.FuseIndex(1, 5);
          Contract(&tmp2, {1, 3}, &right_mps_ten, {0, 1}, &tmp3);
          tmp3.FuseIndex(0, 4);
          tmp3.Transpose({1, 2, 3, 0});
        } else {
          Contract<TenElemT, QNT, true, true>(left_mps_ten, btens.back(), 2, 0, 1, tmp1);
          Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 0, 2, tmp2);
          Contract(&tmp2, {0, 2}, &right_mps_ten, {0, 1}, &tmp3);
        }
        btens.emplace_back(tmp3);
      }
      break;
    }
    case UP: {
      const size_t col = slice_num;
      const TransferMPO &mpo = tn.get_col(col);
      const size_t N = mpo.size(); // tn.cols()
      const size_t end_idx = N - remain_sites;
      auto &left_bmps = bmps_set_[LEFT][col];
      auto &right_bmps = bmps_set_[RIGHT][tn.cols() - col - 1];
      for (size_t i = start_idx; i < end_idx; i++) {
        auto &left_mps_ten = left_bmps[i];
        auto &right_mps_ten = right_bmps[N - i - 1];
        auto &mpo_ten = *mpo[i];
        Tensor tmp1, tmp2, tmp3;
        if constexpr (Tensor::IsFermionic()) {
          Contract<TenElemT, QNT, true, true>(right_mps_ten, btens.back(), 2, 0, 1, tmp1);
          tmp1.FuseIndex(0, 5);
          Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 2, 2, 2, tmp2);
          tmp2.FuseIndex(1, 3);
          Contract(&tmp2, {1, 3}, &left_mps_ten, {0, 1}, &tmp3);
          tmp3.FuseIndex(0, 4);
          tmp3.Transpose({1, 2, 3, 0});
        } else {
          Contract<TenElemT, QNT, true, true>(right_mps_ten, btens.back(), 2, 0, 1, tmp1);
          Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 2, 2, tmp2);
          Contract(&tmp2, {0, 2}, &left_mps_ten, {0, 1}, &tmp3);
        }
        btens.emplace_back(tmp3);
      }
      break;
    }
    case LEFT: {
      const size_t row = slice_num;
      const TransferMPO &mpo = tn.get_row(row);
      const size_t N = mpo.size(); // tn.cols()
      const size_t end_idx = N - remain_sites;
      auto &up_bmps = bmps_set_[UP][row];
      auto &down_bmps = bmps_set_[DOWN][tn.rows() - row - 1];
      for (size_t i = start_idx; i < end_idx; i++) {
        auto &up_mps_ten = up_bmps[N - i - 1];
        auto &down_mps_ten = down_bmps[i];
        Tensor tmp1, tmp2, tmp3;
        if constexpr (Tensor::IsFermionic()) {
          Contract<TenElemT, QNT, true, true>(up_mps_ten, btens.back(), 2, 0, 1, tmp1);
          tmp1.FuseIndex(0, 5);
          auto mpo_ten = *mpo[i];
          mpo_ten.Transpose({3, 0, 1, 2, 4});
          Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 2, 0, 2, tmp2);
          tmp2.FuseIndex(1, 5);
          Contract(&tmp2, {1, 3}, &down_mps_ten, {0, 1}, &tmp3);
          tmp3.FuseIndex(0, 4);
          tmp3.Transpose({1, 2, 3, 0});
        } else {
          Contract<TenElemT, QNT, true, true>(up_mps_ten, btens.back(), 2, 0, 1, tmp1);
          Contract<TenElemT, QNT, false, false>(tmp1, *mpo[i], 1, 3, 2, tmp2);
          Contract(&tmp2, {0, 2}, &down_mps_ten, {0, 1}, &tmp3);
        }
        btens.emplace_back(tmp3);
      }
      break;
    }
    case RIGHT: {
      const size_t row = slice_num;
      const TransferMPO &mpo = tn.get_row(row);
      const size_t N = mpo.size(); // tn.cols()
      const size_t end_idx = N - remain_sites;
      auto &up_bmps = bmps_set_[UP][row];
      auto &down_bmps = bmps_set_[DOWN][tn.rows() - row - 1];
      for (size_t i = start_idx; i < end_idx; i++) {
        auto &up_mps_ten = up_bmps[i];
        auto &down_mps_ten = down_bmps[N - i - 1];
        auto &mpo_ten = *mpo[N - i - 1];
        Tensor tmp1, tmp2, tmp3;
        if constexpr (Tensor::IsFermionic()) {
          Contract<TenElemT, QNT, true, true>(down_mps_ten, btens.back(), 2, 0, 1, tmp1);
          tmp1.FuseIndex(0, 5);
          Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 2, 1, 2, tmp2);
          tmp2.FuseIndex(1, 4);
          Contract(&tmp2, {1, 3}, &up_mps_ten, {0, 1}, &tmp3);
          tmp3.FuseIndex(0, 4);
          tmp3.Transpose({1, 2, 3, 0});
        } else {
          Contract<TenElemT, QNT, true, true>(down_mps_ten, btens.back(), 2, 0, 1, tmp1);
          Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 1, 2, tmp2);
          Contract(&tmp2, {0, 2}, &up_mps_ten, {0, 1}, &tmp3);
        }
        btens.emplace_back(tmp3);
      }
      break;
    }
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::GrowFullBTen2(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post, const size_t slice_num1,
                                                   const size_t remain_sites, bool init) {
  if (init) {
    InitBTen2(tn, post, slice_num1);
  }
  size_t ctrct_mpo_start_idx = (size_t(post) + 3) % 4;
  BMPSPOSITION pre_post = BMPSPOSITION(ctrct_mpo_start_idx);
  BMPSPOSITION next_post = BMPSPOSITION((size_t(post) + 1) % 4);
  size_t start_idx = bten_set2_[post].size() - 1;
  std::vector<Tensor> &btens = bten_set2_[post];
  TransferMPO mpo1, mpo2;
  size_t N, end_idx;
  BMPS<TenElemT, QNT> *bmps_pre, *bmps_post;
  std::vector<size_t> mpo_ten_transpose_axes;
  switch (post) {
    case DOWN: {
      const size_t col1 = slice_num1;
      const size_t col2 = slice_num1 + 1;
      mpo1 = tn.get_col(col1);
      mpo2 = tn.get_col(col2);
      std::reverse(mpo1.begin(), mpo1.end());
      std::reverse(mpo2.begin(), mpo2.end());
      N = mpo1.size(); // tn.rows();
      bmps_pre = &bmps_set_[pre_post][col1];
      bmps_post = &bmps_set_[next_post][tn.cols() - 1 - col2];
      break;
    }
    case RIGHT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      mpo1 = tn.get_row(row2);
      mpo2 = tn.get_row(row1);
      std::reverse(mpo1.begin(), mpo1.end());
      std::reverse(mpo2.begin(), mpo2.end());
      N = mpo1.size(); // tn.cols()
      const size_t mps1_num = tn.rows() - 1 - row2;
      const size_t mps2_num = row1;
      bmps_pre = &bmps_set_[pre_post][mps1_num];
      bmps_post = &bmps_set_[next_post][mps2_num];
      break;
    }
    case UP: {
      const size_t col1 = slice_num1;
      const size_t col2 = slice_num1 + 1;
      mpo1 = tn.get_col(col2);
      mpo2 = tn.get_col(col1);
      N = mpo1.size(); // tn.rows();
      bmps_pre = &bmps_set_[RIGHT][tn.cols() - 1 - col2];
      bmps_post = &bmps_set_[LEFT][col1];
      break;
    }
    case LEFT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      mpo1 = tn.get_row(row1);
      mpo2 = tn.get_row(row2);
      N = mpo1.size(); // tn.cols()
      const size_t mps1_num = row1;
      const size_t mps2_num = tn.rows() - 1 - row2;
      bmps_pre = &bmps_set_[pre_post][mps1_num];
      bmps_post = &bmps_set_[next_post][mps2_num];
      break;
    }
  }
  mpo_ten_transpose_axes = GenMpoTen1TransposeAxesForBrowBTen2<Tensor::IsFermionic()>(post);
  end_idx = N - remain_sites;
  for (size_t i = start_idx; i < end_idx; i++) {
    auto &mps_ten1 = (*bmps_pre)[N - i - 1];
    auto &mps_ten2 = (*bmps_post)[i];
    Tensor mpo_ten1 = *mpo1[i];
    mpo_ten1.Transpose(mpo_ten_transpose_axes);
    Tensor tmp1, tmp2, tmp3, next_bten;

    if constexpr (Tensor::IsFermionic()) {
      auto mpo_ten2 = *mpo2[i];
      auto mpo2_trans_axes = GenMpoTen2TransposeAxesForFermionGrowBTen2(post);
      if (post != DOWN)
        mpo_ten2.Transpose({mpo2_trans_axes});

      Contract<TenElemT, QNT, true, true>(mps_ten1, bten_set2_.at(post).back(), 2, 0, 1, tmp1);
      tmp1.FuseIndex(0, 6);
      Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 2, 0, 2, tmp2);
      tmp2.FuseIndex(2, 6);
      tmp2.Transpose({1, 0, 2, 3, 4, 5});
      Contract<TenElemT, QNT, true, true>(tmp2, mpo_ten2, 5, 0, 2, tmp3);
      tmp3.FuseIndex(0, 6);
      Contract(&tmp3, {1, 4}, &mps_ten2, {0, 1}, &next_bten);
      next_bten.FuseIndex(0, 5);
      next_bten.Transpose({1, 2, 3, 4, 0});
    } else {
      auto &mpo_ten2 = *mpo2[i];
      Contract<TenElemT, QNT, true, true>(mps_ten1, btens.back(), 2, 0, 1, tmp1);
      Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
      Contract<TenElemT, QNT, false, false>(tmp2, mpo_ten2, 4, ctrct_mpo_start_idx, 2, tmp3);
      Contract(&tmp3, {0, 3}, &mps_ten2, {0, 1}, &next_bten);
    }
    btens.emplace_back(next_bten);
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::GrowBTen2Step(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post, const size_t slice_num1) {
  size_t ctrct_mpo_start_idx = (size_t(post) + 3) % 4;
  BMPSPOSITION pre_post = BMPSPOSITION(ctrct_mpo_start_idx);
  BMPSPOSITION next_post = BMPSPOSITION((size_t(post) + 1) % 4);
  const size_t bten_size = bten_set2_.at(post).size();
#ifndef NDEBUG
  assert(bmps_set_.at(pre_post).size() + bmps_set_.at(next_post).size() >= tn.length(Orientation(pre_post)));
  assert(bten_set2_.at(post).size() > 0 &&
      bten_set2_.at(post).size() <= bmps_set_.at(pre_post).back().size()); // has been initialled
#endif
  SiteIdx grown_site1, grown_site2;
  size_t N; //mps length
  size_t mps1_idx, mps2_idx;
  std::vector<size_t> mpo1_transpose_axes;

  SetUpCoordInfoForGrowBTen2<QNT>(post, slice_num1, bten_size, tn.rows(), tn.cols(),
                                  N, grown_site1, grown_site2, mps1_idx, mps2_idx, mpo1_transpose_axes);

  auto mpo_ten1 = tn({grown_site1});
  mpo_ten1.Transpose(mpo1_transpose_axes);
  const auto &mps_ten1 = bmps_set_.at(pre_post)[mps1_idx][N - bten_size];
  const auto &mps_ten2 = bmps_set_.at(next_post)[mps2_idx][bten_size - 1];
  Tensor *mpo_ten2;
  if constexpr (Tensor::IsFermionic()) {
    mpo_ten2 = new Tensor(tn({grown_site2}));
    if (post != DOWN) {
      auto mpo2_trans_axes = GenMpoTen2TransposeAxesForFermionGrowBTen2(post);
      mpo_ten2->Transpose(mpo2_trans_axes);
    }
  } else {
    mpo_ten2 = const_cast<Tensor*>(&tn({grown_site2}));
  }
  auto next_bten = GrowBTen2StepAfterTransposedMPOTens(bten_set2_.at(post).back(),
                                                       mps_ten1,
                                                       mps_ten2,
                                                       mpo_ten1,
                                                       *mpo_ten2,
                                                       ctrct_mpo_start_idx);
  if constexpr (Tensor::IsFermionic()) {
    delete mpo_ten2;
  }
  bten_set2_[post].emplace_back(next_bten);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::ShiftBTenWindow(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position) {
  bten_set_[position].pop_back();
  GrowBTenStep(tn, Opposite(position));
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::ShiftBTen2Window(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1) {
  bten_set2_[position].pop_back();
  GrowBTen2Step(tn, Opposite(position), slice_num1);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::GrowBTenStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post) {
  size_t ctrct_mpo_start_idx = (size_t(post) + 3) % 4;
  BMPSPOSITION pre_post = BMPSPOSITION(ctrct_mpo_start_idx);
  BMPSPOSITION next_post = BMPSPOSITION((size_t(post) + 1) % 4);
#ifndef NDEBUG
  assert(bmps_set_.at(pre_post).size() + bmps_set_.at(next_post).size() == tn.length(Orientation(pre_post)) + 1);
  assert(bten_set_.at(post).size() > 0 &&
      bten_set_.at(post).size() <= bmps_set_.at(pre_post).back().size()); // has been initialled
#endif
  Tensor tmp1, tmp2, next_bten;
  Tensor *mps_ten1, *mps_ten2;
  SiteIdx grown_site;
  size_t N; //mps length
  const size_t bten_size = bten_set_.at(post).size();
  switch (post) {
    case DOWN: {
      const size_t col = bmps_set_[LEFT].size() - 1;
      N = tn.rows();
      grown_site = {N - bten_size, col};
      break;
    }
    case UP: {
      const size_t col = bmps_set_[LEFT].size() - 1;
      N = tn.rows();
      grown_site = {bten_size - 1, col};
      break;
    }
    case LEFT: {
      const size_t row = bmps_set_[UP].size() - 1;
      N = tn.cols();
      grown_site = {row, bten_size - 1};
      break;
    }
    case RIGHT: {
      const size_t row = bmps_set_[UP].size() - 1;
      N = tn.cols();
      grown_site = {row, N - bten_size};
      break;
    }
  }
  mps_ten1 = &bmps_set_[pre_post].back()[N - bten_size];
  mps_ten2 = &bmps_set_[next_post].back()[bten_size - 1];
  if constexpr (Tensor::IsFermionic()) {
    next_bten =
        FermionGrowBTenStep(post, bten_set_.at(post).back(),
                            *mps_ten1, tn({grown_site}), *mps_ten2);
  } else {
    Contract<TenElemT, QNT, true, true>(*mps_ten1, bten_set_.at(post).back(), 2, 0, 1, tmp1);
    Contract<TenElemT, QNT, false, false>(tmp1, tn({grown_site}), 1, ctrct_mpo_start_idx, 2, tmp2);
    Contract(&tmp2, {0, 2}, mps_ten2, {0, 1}, &next_bten);
  }
  bten_set_[post].emplace_back(next_bten);
}

// --------------------------------------------------------
// Trace Methods
// --------------------------------------------------------

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::Trace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const BondOrientation bond_dir) const {
  SiteIdx site_b(site_a);
  if (bond_dir == HORIZONTAL) {
    site_b.col() += 1;
  } else {
    site_b.row() += 1;
  }
  return Trace(tn, site_a, site_b, bond_dir);
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::Trace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const SiteIdx &site_b,
                                               const BondOrientation bond_dir) const {
  const Tensor &ten_a = tn({site_a});
  const Tensor &ten_b = tn({site_b});
  return ReplaceNNSiteTrace(tn, site_a, site_b, bond_dir, ten_a, ten_b);
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::ReplaceOneSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site,
                                                             const Tensor &replace_ten,
                                                             const BondOrientation mps_orient) const {
  Tensor tmp[4];
  const size_t row = site.row();
  const size_t col = site.col();
#ifndef NDEBUG
  // check the environment tensors are sufficient
  if (mps_orient == HORIZONTAL) {
    assert(bmps_set_.at(UP).size() > row);
    assert(bmps_set_.at(DOWN).size() + 1 > tn.rows() - row);
    assert(bten_set_.at(LEFT).size() > col);
    assert(bten_set_.at(RIGHT).size() + 1 > tn.cols() - col);
  } else {
    assert(bmps_set_.at(LEFT).size() > col);
    assert(bmps_set_.at(RIGHT).size() + 1 > tn.cols() - col);
    assert(bten_set_.at(UP).size() > row);
    assert(bten_set_.at(DOWN).size() + 1 > tn.rows() - row);
  }
#endif
  const Tensor *up_ten, *down_ten, *left_ten, *right_ten;
  if (mps_orient == HORIZONTAL) {
    const Tensor &up_mps_ten = bmps_set_.at(UP)[row][tn.cols() - col - 1];
    const Tensor &down_mps_ten = bmps_set_.at(DOWN)[tn.rows() - row - 1][col];
    const Tensor &left_bten = bten_set_.at(LEFT)[col];
    const Tensor &right_bten = bten_set_.at(RIGHT)[tn.cols() - col - 1];
    up_ten = &up_mps_ten;
    down_ten = &down_mps_ten;
    left_ten = &left_bten;
    right_ten = &right_bten;
  } else {
    const Tensor &left_mps_ten = bmps_set_.at(LEFT)[col][row];
    const Tensor &right_mps_ten = bmps_set_.at(RIGHT)[tn.cols() - col - 1][tn.rows() - row - 1];
    const Tensor &up_bten = bten_set_.at(UP)[row];
    const Tensor &down_bten = bten_set_.at(DOWN)[tn.rows() - row - 1];
    up_ten = &up_bten;
    down_ten = &down_bten;
    left_ten = &left_mps_ten;
    right_ten = &right_mps_ten;
  }
  if constexpr (Tensor::IsFermionic()) {
    // assume the particle numbers in the configuration of the single-layer tensor network, are always even.
    Contract<TenElemT, QNT, true, true>(*up_ten, *left_ten, 2, 0, 1, tmp[0]);
    tmp[0].FuseIndex(0, 5);
    Contract(tmp, {2, 3}, &replace_ten, {3, 0}, tmp + 1);
    tmp[1].FuseIndex(0, 5);
    Contract<TenElemT, QNT, false, true>(tmp[1], *down_ten, 2, 0, 2, tmp[2]);
    tmp[2].FuseIndex(1, 4);
    Contract(tmp + 2, {3, 1, 2}, right_ten, {0, 1, 2}, tmp + 3);
    return tmp[3].GetElem({0, 0});
  } else {
    Contract<TenElemT, QNT, true, true>(*up_ten, *left_ten, 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, false>(tmp[0], replace_ten, 1, 3, 2, tmp[1]);
    Contract(&tmp[1], {0, 2}, down_ten, {0, 1}, &tmp[2]);
    Contract(&tmp[2], {0, 1, 2}, right_ten, {2, 1, 0}, &tmp[3]);
    return tmp[3]();
  }
}

template<typename TenElemT, typename QNT>
TenElemT
BMPSContractor<TenElemT, QNT>::ReplaceNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const SiteIdx &site_b,
                                                   const BondOrientation bond_dir,
                                                   const Tensor &ten_a,
                                                   const Tensor &ten_b) const {
#ifndef NDEBUG
  if (bond_dir == HORIZONTAL) {
    assert(site_a.row() == site_b.row());
    assert(site_a.col() + 1 == site_b.col());
    size_t bond_row = site_a.row();
    assert(bmps_set_.at(UP).size() > bond_row);
    assert(bmps_set_.at(DOWN).size() + 1 > tn.rows() - bond_row);
    assert(bten_set_.at(LEFT).size() > site_a[1]);
    assert(bten_set_.at(RIGHT).size() + 1 > tn.cols() - site_b[1]);
  } else {
    assert(site_a.row() + 1 == site_b.row());
    assert(site_a.col() == site_b.col());
    size_t bond_col = site_a.col();
    assert(bmps_set_.at(LEFT).size() > bond_col);
    assert(bmps_set_.at(RIGHT).size() + 1 > tn.cols() - bond_col);
    assert(bten_set_.at(UP).size() > site_a.row());
    assert(bten_set_.at(DOWN).size() + 1 > tn.rows() - site_b[0]);
  }
#endif
  Tensor tmp[7];
  if (bond_dir == HORIZONTAL) {
    const size_t row = site_a[0];
    const size_t col_a = site_a[1];

    const Tensor &up_mps_ten_a = bmps_set_.at(UP)[row][tn.cols() - col_a - 1];
    const Tensor &down_mps_ten_a = bmps_set_.at(DOWN)[tn.rows() - row - 1][col_a];
    if constexpr (Tensor::IsFermionic()) {
      Contract<TenElemT, QNT, true, true>(up_mps_ten_a, bten_set_.at(LEFT)[col_a], 2, 0, 1, tmp[0]);
      tmp[0].FuseIndex(0, 5);
      Contract(tmp, {2, 3}, &ten_a, {3, 0}, tmp + 1);
      Contract(tmp + 1, {2, 3}, &down_mps_ten_a, {0, 1}, &tmp[2]);
      tmp[2].FuseIndex(0, 5); // the first index of tmp[2] is the trivial index
    } else {
      Contract<TenElemT, QNT, true, true>(up_mps_ten_a, bten_set_.at(LEFT)[col_a], 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, false>(tmp[0], ten_a, 1, 3, 2, tmp[1]);
      Contract(&tmp[1], {0, 2}, &down_mps_ten_a, {0, 1}, &tmp[2]);
    }

    size_t col_b = site_b[1];
    const Tensor &up_mps_ten_b = bmps_set_.at(UP)[row][tn.cols() - col_b - 1];
    const Tensor &down_mps_ten_b = bmps_set_.at(DOWN)[tn.rows() - row - 1][col_b];
    if constexpr (Tensor::IsFermionic()) {
      Contract<TenElemT, QNT, true, true>(down_mps_ten_b,
                                          bten_set_.at(RIGHT)[tn.cols() - col_b - 1],
                                          2,
                                          0,
                                          1,
                                          tmp[3]);
      tmp[3].FuseIndex(0, 5);
      Contract(tmp + 3, {2, 3}, &ten_b, {1, 2}, tmp + 4);
      Contract(&tmp[4], {2, 4}, &up_mps_ten_b, {0, 1}, &tmp[5]);
      tmp[5].FuseIndex(0, 5);
    } else {
      Contract<TenElemT, QNT, true, true>(down_mps_ten_b,
                                          bten_set_.at(RIGHT)[tn.cols() - col_b - 1],
                                          2,
                                          0,
                                          1,
                                          tmp[3]);
      Contract<TenElemT, QNT, false, false>(tmp[3], ten_b, 1, 1, 2, tmp[4]);
      Contract(&tmp[4], {0, 2}, &up_mps_ten_b, {0, 1}, &tmp[5]);
    }
  } else {
    const size_t col = site_a[1];
    const size_t row_a = site_a[0];
    const Tensor &left_mps_ten_a = bmps_set_.at(LEFT)[col][row_a];
    const Tensor &right_mps_ten_a = bmps_set_.at(RIGHT)[tn.cols() - col - 1][tn.rows() - row_a - 1];
    if constexpr (Tensor::IsFermionic()) {
      Contract<TenElemT, QNT, true, true>(right_mps_ten_a, bten_set_.at(UP)[row_a], 2, 0, 1, tmp[0]);
      tmp[0].FuseIndex(0, 5);
      Contract(tmp, {2, 3}, &ten_a, {2, 3}, &tmp[1]);
      Contract(&tmp[1], {2, 3}, &left_mps_ten_a, {0, 1}, &tmp[2]);
      tmp[2].FuseIndex(0, 5);
    } else {
      Contract<TenElemT, QNT, true, true>(right_mps_ten_a, bten_set_.at(UP)[row_a], 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, false>(tmp[0], ten_a, 1, 2, 2, tmp[1]);
      Contract(&tmp[1], {0, 2}, &left_mps_ten_a, {0, 1}, &tmp[2]);
    }

    const size_t row_b = site_b[0];
    const Tensor &left_mps_ten_b = bmps_set_.at(LEFT)[col][row_b];
    const Tensor &right_mps_ten_b = bmps_set_.at(RIGHT)[tn.cols() - col - 1][tn.rows() - row_b - 1];
    if constexpr (Tensor::IsFermionic()) {
      Contract<TenElemT, QNT, true, true>(left_mps_ten_b,
                                          bten_set_.at(DOWN)[tn.rows() - row_b - 1],
                                          2, 0, 1, tmp[3]);
      tmp[3].FuseIndex(0, 5);
      Contract(&tmp[3], {2, 3}, &ten_b, {0, 1}, &tmp[4]);
      Contract(&tmp[4], {2, 3}, &right_mps_ten_b, {0, 1}, &tmp[5]);
      tmp[5].FuseIndex(0, 5);
    } else {
      Contract<TenElemT, QNT, true, true>(left_mps_ten_b,
                                          bten_set_.at(DOWN)[tn.rows() - row_b - 1],
                                          2,
                                          0,
                                          1,
                                          tmp[3]);
      Contract<TenElemT, QNT, false, false>(tmp[3], ten_b, 1, 0, 2, tmp[4]);
      Contract(&tmp[4], {0, 2}, &right_mps_ten_b, {0, 1}, &tmp[5]);
    }
  }
  if constexpr (Tensor::IsFermionic()) {
    Contract(&tmp[2], {1, 2, 4}, &tmp[5], {4, 2, 1}, &tmp[6]);
    tmp[6].Transpose({0, 1, 3, 2});
    return tmp[6].GetElem({0, 0, 0, 0});
  } else {
    Contract(&tmp[2], {0, 1, 2}, &tmp[5], {2, 1, 0}, &tmp[6]);
    return tmp[6]();
  }
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::ReplaceNNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &left_up_site,
                                                             const DIAGONAL_DIR nnn_dir,
                                                             const BondOrientation mps_orient,
                                                             const Tensor &ten_left,
                                                             const Tensor &ten_right) const {
  const size_t row1 = left_up_site[0];
  const size_t row2 = row1 + 1;
  const size_t col1 = left_up_site[1];
  const size_t col2 = col1 + 1;
  Tensor tmp[9];
  if (mps_orient == HORIZONTAL) {
#ifndef NDEBUG
    const Tensor &mps_ten1 = bmps_set_.at(UP).at(row1)[tn.cols() - col1 - 1];
    const Tensor &mps_ten2 = bmps_set_.at(DOWN).at(tn.rows() - 1 - row2)[col1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN).at(tn.rows() - 1 - row2)[col2];
    const Tensor &mps_ten4 = bmps_set_.at(UP).at(row1)[tn.cols() - col2 - 1];
#else
    const Tensor &mps_ten1 = bmps_set_.at(UP)[row1][tn.cols() - col1 - 1];
    const Tensor &mps_ten2 = bmps_set_.at(DOWN)[tn.rows() - 1 - row2][col1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN)[tn.rows() - 1 - row2][col2];
    const Tensor &mps_ten4 = bmps_set_.at(UP)[row1][tn.cols() - col2 - 1];
#endif

    Tensor *mpo_ten0, *mpo_ten2;
    const Tensor *mpo_ten1, *mpo_ten3;
    const Tensor &left_bten = bten_set2_.at(LEFT)[col1];
    const Tensor &right_bten = bten_set2_.at(RIGHT)[tn.cols() - col2 - 1];
    if (nnn_dir == LEFTUP_TO_RIGHTDOWN) {
      if (Tensor::IsFermionic()) {
        mpo_ten0 = const_cast<Tensor *>(&ten_left);
        mpo_ten2 = const_cast<Tensor *>(&ten_right);
      } else {
        mpo_ten0 = new Tensor(ten_left);
        mpo_ten2 = new Tensor(ten_right);
      }
      mpo_ten1 = &tn({row2, col1});
      mpo_ten3 = &tn({row1, col2});
    } else { //LEFTDOWN_TO_RIGHTUP
      if (Tensor::IsFermionic()) {
        mpo_ten0 = const_cast<Tensor *>(&tn({row1, col1}));
        mpo_ten2 = const_cast<Tensor *>(&tn({row2, col2}));
      } else {
        mpo_ten0 = new Tensor(tn({row1, col1}));
        mpo_ten2 = new Tensor(tn({row2, col2}));
      }
      mpo_ten1 = (&ten_left);
      mpo_ten3 = (&ten_right);
    }
    if (Tensor::IsFermionic()) {
      Contract<TenElemT, QNT, true, true>(mps_ten1, left_bten, 2, 0, 1, tmp[0]);
      tmp[0].FuseIndex(0, 6);
      Contract(tmp, {2, 3}, mpo_ten0, {3, 0}, tmp + 1);
      Contract(tmp + 1, {2, 4}, mpo_ten1, {0, 3}, tmp + 2);
      Contract(tmp + 2, {2, 5}, &mps_ten2, {0, 1}, tmp + 3);
      tmp[3].FuseIndex(0, 7);

      Contract<TenElemT, QNT, true, true>(mps_ten3, right_bten, 2, 0, 1, tmp[4]);
      tmp[4].FuseIndex(0, 6);
      Contract(tmp + 4, {2, 3}, mpo_ten2, {1, 2}, tmp + 5);
      Contract(tmp + 5, {5, 2}, mpo_ten3, {1, 2}, tmp + 6);
      Contract(tmp + 6, {2, 6}, &mps_ten4, {0, 1}, tmp + 7);
      tmp[7].FuseIndex(0, 7);

      Contract(tmp + 3, {1, 2, 4, 6}, tmp + 7, {6, 4, 2, 1}, &tmp[8]);
      tmp[8].Transpose({0, 3, 2, 5, 1, 4});
      return tmp[8]({0, 0, 0, 0, 0, 0});
    } else {
      mpo_ten0->Transpose({3, 0, 2, 1});
      mpo_ten2->Transpose({1, 2, 0, 3});

      Contract<TenElemT, QNT, true, true>(mps_ten1, left_bten, 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, false>(tmp[0], *mpo_ten0, 1, 0, 2, tmp[1]);
      Contract<TenElemT, QNT, false, false>(tmp[1], *mpo_ten1, 4, 3, 2, tmp[2]);
      Contract(&tmp[2], {0, 3}, &mps_ten2, {0, 1}, &tmp[3]);

      Contract<TenElemT, QNT, true, true>(mps_ten3, right_bten, 2, 0, 1, tmp[4]);
      Contract<TenElemT, QNT, false, false>(tmp[4], *mpo_ten2, 1, 0, 2, tmp[5]);
      Contract<TenElemT, QNT, false, false>(tmp[5], *mpo_ten3, 4, 1, 2, tmp[6]);
      Contract(&tmp[6], {0, 3}, &mps_ten4, {0, 1}, &tmp[7]);
      Contract(&tmp[3], {0, 1, 2, 3}, &tmp[7], {3, 2, 1, 0}, &tmp[8]);

      delete mpo_ten0;
      delete mpo_ten2;
      return tmp[8]();
    }
  } else { //mps_orient == VERTICAL
    assert(!Tensor::IsFermionic());
    Tensor mpo_ten[4];
    Tensor tmp[9];
#ifndef NDEBUG
    const Tensor &mps_ten1 = bmps_set_.at(LEFT).at(col1)[row2];
    const Tensor &mps_ten2 = bmps_set_.at(RIGHT).at(tn.cols() - 1 - col2)[tn.rows() - 1 - row2];
    const Tensor &mps_ten3 = bmps_set_.at(LEFT).at(col1)[row1];
    const Tensor &mps_ten4 = bmps_set_.at(RIGHT).at(tn.cols() - 1 - col2)[tn.rows() - 1 - row1];
#else
    const Tensor &mps_ten1 = bmps_set_.at(LEFT)[col1][row2];
    const Tensor &mps_ten2 = bmps_set_.at(RIGHT)[tn.cols() - 1 - col2][tn.rows() - 1 - row2];
    const Tensor &mps_ten3 = bmps_set_.at(LEFT)[col1][row1];
    const Tensor &mps_ten4 = bmps_set_.at(RIGHT)[tn.cols() - 1 - col2][tn.rows() - 1 - row1];
#endif
    const Tensor &top_bten = bten_set2_.at(UP)[row1];
    const Tensor &bottom_bten = bten_set2_.at(DOWN)[tn.rows() - row2 - 1];

    if (nnn_dir == LEFTUP_TO_RIGHTDOWN) {
      mpo_ten[0] = tn({row2, col1});
      mpo_ten[1] = ten_right;
      mpo_ten[2] = ten_left;
      mpo_ten[3] = tn({row1, col2});
    } else { //LEFTDOWN_TO_RIGHTUP
      mpo_ten[0] = ten_left;
      mpo_ten[1] = tn({row2, col2});
      mpo_ten[2] = tn({row1, col1});
      mpo_ten[3] = ten_right;
    }
    mpo_ten[0].Transpose({0, 1, 3, 2});
    mpo_ten[3].Transpose({2, 3, 1, 0});

    Contract<TenElemT, QNT, true, true>(mps_ten1, bottom_bten, 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, true>(tmp[0], mpo_ten[0], 1, 0, 2, tmp[1]);
    Contract<TenElemT, QNT, false, true>(tmp[1], mpo_ten[1], 4, 0, 2, tmp[2]);
    Contract(&tmp[2], {0, 3}, &mps_ten2, {0, 1}, &tmp[3]);

    Contract<TenElemT, QNT, true, true>(mps_ten4, top_bten, 2, 0, 1, tmp[4]);
    Contract<TenElemT, QNT, false, false>(tmp[4], mpo_ten[3], 1, 0, 2, tmp[5]);
    Contract<TenElemT, QNT, false, false>(tmp[5], mpo_ten[2], 4, 2, 2, tmp[6]);
    Contract(&tmp[6], {0, 3}, &mps_ten3, {0, 1}, &tmp[7]);

    Contract(&tmp[3], {0, 1, 2, 3}, &tmp[7], {3, 2, 1, 0}, &tmp[8]);
    return tmp[8]();
  }
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::ReplaceTNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site0,
                                                             const BondOrientation mps_orient,
                                                             const Tensor &replaced_ten0,
                                                             const Tensor &replaced_ten1,
                                                             const Tensor &replaced_ten2) const {
  Tensor tmp[10];
  if (mps_orient == HORIZONTAL) {
    const size_t row = site0.row();
    const size_t col0 = site0.col();
    const size_t col1 = col0 + 1;
    const size_t col2 = col0 + 2;
#ifndef NDEBUG
    const Tensor &mps_ten0 = bmps_set_.at(UP).at(row)[tn.cols() - col0 - 1];
    const Tensor &mps_ten1 = bmps_set_.at(DOWN).at(tn.rows() - 1 - row)[col0];
    const Tensor &mps_ten2 = bmps_set_.at(UP).at(row)[tn.cols() - col1 - 1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN).at(tn.rows() - 1 - row)[col1];
    const Tensor &mps_ten4 = bmps_set_.at(UP).at(row)[tn.cols() - col2 - 1];
    const Tensor &mps_ten5 = bmps_set_.at(DOWN).at(tn.rows() - 1 - row)[col2];
    const Tensor &left_bten = bten_set_.at(LEFT).at(col0);
    const Tensor &right_bten = bten_set_.at(RIGHT).at(tn.cols() - col2 - 1);
#else
    const Tensor &mps_ten0 = bmps_set_.at(UP)[row][tn.cols() - col0 - 1];
    const Tensor &mps_ten1 = bmps_set_.at(DOWN)[tn.rows() - 1 - row][col0];
    const Tensor &mps_ten2 = bmps_set_.at(UP)[row][tn.cols() - col1 - 1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN)[tn.rows() - 1 - row][col1];
    const Tensor &mps_ten4 = bmps_set_.at(UP)[row][tn.cols() - col2 - 1];
    const Tensor &mps_ten5 = bmps_set_.at(DOWN)[tn.rows() - 1 - row][col2];
    const Tensor &left_bten = bten_set_.at(LEFT)[col0];
    const Tensor &right_bten = bten_set_.at(RIGHT)[tn.cols() - col2 - 1];
#endif
    if constexpr (Tensor::IsFermionic()) {
      Tensor next_left_bten1 =
          FermionGrowBTenStep(LEFT, left_bten,
                              mps_ten0, replaced_ten0, mps_ten1);
      Tensor next_left_bten2 =
          FermionGrowBTenStep(LEFT, next_left_bten1,
                              mps_ten2, replaced_ten1, mps_ten3);
      Tensor next_left_bten3 =
          FermionGrowBTenStep(LEFT, next_left_bten2,
                              mps_ten4, replaced_ten2, mps_ten5);
      Contract(&next_left_bten3, {0, 1, 2}, &right_bten, {2, 1, 0}, tmp);
      return tmp[0]({0, 0});
    } else { // Boson
      const Tensor &mpo_ten0 = replaced_ten0,
          mpo_ten1 = replaced_ten1,
          mpo_ten2 = replaced_ten2;

      Contract<TenElemT, QNT, true, true>(mps_ten0, left_bten, 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, false>(tmp[0], mpo_ten0, 1, 3, 2, tmp[1]);
      Contract(&tmp[1], {0, 2}, &mps_ten1, {0, 1}, &tmp[2]);

      Contract<TenElemT, QNT, true, true>(mps_ten2, tmp[2], 2, 0, 1, tmp[3]);
      Contract<TenElemT, QNT, false, false>(tmp[3], mpo_ten1, 1, 3, 2, tmp[4]);
      Contract(&tmp[4], {0, 2}, &mps_ten3, {0, 1}, &tmp[5]);

      Contract<TenElemT, QNT, true, true>(mps_ten4, tmp[5], 2, 0, 1, tmp[6]);
      Contract<TenElemT, QNT, false, false>(tmp[6], mpo_ten2, 1, 3, 2, tmp[7]);
      Contract(&tmp[7], {0, 2}, &mps_ten5, {0, 1}, &tmp[8]);

      Contract(&tmp[8], {0, 1, 2}, &right_bten, {2, 1, 0}, &tmp[9]);
      return tmp[9]();
    }
  } else { // mps_orient == VERTICAL
    const size_t col = site0[1];
    const size_t row0 = site0[0];
    const size_t row1 = row0 + 1;
    const size_t row2 = row1 + 1;

    const Tensor &mps_ten0 = bmps_set_.at(RIGHT)[tn.cols() - 1 - col][tn.rows() - 1 - row0];
    const Tensor &mps_ten1 = bmps_set_.at(LEFT)[col][row0];
    const Tensor &mps_ten2 = bmps_set_.at(RIGHT)[tn.cols() - 1 - col][tn.rows() - 1 - row1];
    const Tensor &mps_ten3 = bmps_set_.at(LEFT)[col][row1];
    const Tensor &mps_ten4 = bmps_set_.at(RIGHT)[tn.cols() - 1 - col][tn.rows() - 1 - row2];
    const Tensor &mps_ten5 = bmps_set_.at(LEFT)[col][row2];

    const Tensor &top_bten = bten_set_.at(UP)[row0];
    const Tensor &bottom_bten = bten_set_.at(DOWN)[tn.rows() - row2 - 1];

    if constexpr (Tensor::IsFermionic()) {
      Tensor next_up_bten1 =
          FermionGrowBTenStep(UP, top_bten,
                              mps_ten0, replaced_ten0, mps_ten1);
      Tensor next_up_bten2 =
          FermionGrowBTenStep(UP, next_up_bten1,
                              mps_ten2, replaced_ten1, mps_ten3);
      Tensor next_up_bten3 =
          FermionGrowBTenStep(UP, next_up_bten2,
                              mps_ten4, replaced_ten2, mps_ten5);
      Contract(&next_up_bten3, {0, 1, 2}, &bottom_bten, {2, 1, 0}, tmp);
      return tmp[0]({0, 0});
    } else { // Boson
      Contract<TenElemT, QNT, true, true>(mps_ten0, top_bten, 2, 0, 1, tmp[0]);
      Contract<TenElemT, QNT, false, false>(tmp[0], replaced_ten0, 1, 2, 2, tmp[1]);
      Contract(&tmp[1], {0, 2}, &mps_ten1, {0, 1}, &tmp[2]);

      Contract<TenElemT, QNT, true, true>(mps_ten2, tmp[2], 2, 0, 1, tmp[3]);
      Contract<TenElemT, QNT, false, false>(tmp[3], replaced_ten1, 1, 2, 2, tmp[4]);
      Contract(&tmp[4], {0, 2}, &mps_ten3, {0, 1}, &tmp[5]);

      Contract<TenElemT, QNT, true, true>(mps_ten4, tmp[5], 2, 0, 1, tmp[6]);
      Contract<TenElemT, QNT, false, false>(tmp[6], replaced_ten2, 1, 2, 2, tmp[7]);
      Contract(&tmp[7], {0, 2}, &mps_ten5, {0, 1}, &tmp[8]);

      Contract(&tmp[8], {0, 1, 2}, &bottom_bten, {2, 1, 0}, &tmp[9]);
      return tmp[9]();
    }
  }
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::ReplaceSqrt5DistTwoSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &left_up_site,
                                                                      const DIAGONAL_DIR sqrt5link_dir,
                                                                      const BondOrientation mps_orient, //mps orientation is the same with longer side orientation
                                                                      const Tensor &ten_left,
                                                                      const Tensor &ten_right) const {
  assert(!Tensor::IsFermionic());
  Tensor mpo_ten[6];
  Tensor tmp[13];
  if (mps_orient == HORIZONTAL) {
    const size_t row1 = left_up_site.row();
    const size_t row2 = row1 + 1;
    const size_t col1 = left_up_site.col();
    const size_t col2 = col1 + 1;
    const size_t col3 = col2 + 1;

    const Tensor &mps_ten1 = bmps_set_.at(UP)[row1][tn.cols() - col1 - 1];
    const Tensor &mps_ten2 = bmps_set_.at(DOWN)[tn.rows() - 1 - row2][col1];
    const Tensor &mps_ten3 = bmps_set_.at(UP)[row1][tn.cols() - col2 - 1];
    const Tensor &mps_ten4 = bmps_set_.at(DOWN)[tn.rows() - 1 - row2][col2];
    const Tensor &mps_ten5 = bmps_set_.at(UP)[row1][tn.cols() - col3 - 1];
    const Tensor &mps_ten6 = bmps_set_.at(DOWN)[tn.rows() - 1 - row2][col3];

    const Tensor &left_bten = bten_set2_.at(LEFT)[col1];
    const Tensor &right_bten = bten_set2_.at(RIGHT)[tn.cols() - col3 - 1];

    if (sqrt5link_dir == LEFTUP_TO_RIGHTDOWN) {
      mpo_ten[0] = ten_left;
      mpo_ten[1] = tn({row2, col1});
      mpo_ten[4] = tn({row1, col3});
      mpo_ten[5] = ten_right;
    } else { //LEFTDOWN_TO_RIGHTUP
      mpo_ten[0] = tn({row1, col1});
      mpo_ten[1] = ten_left;
      mpo_ten[4] = ten_right;
      mpo_ten[5] = tn({row2, col3});
    }
    mpo_ten[2] = tn({row1, col2});
    mpo_ten[3] = tn({row2, col2});

    mpo_ten[0].Transpose({3, 0, 2, 1});
    mpo_ten[2].Transpose({3, 0, 2, 1});
    mpo_ten[5].Transpose({1, 2, 0, 3});

    Contract<TenElemT, QNT, true, true>(mps_ten1, left_bten, 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, false>(tmp[0], mpo_ten[0], 1, 0, 2, tmp[1]);
    Contract<TenElemT, QNT, false, false>(tmp[1], mpo_ten[1], 4, 3, 2, tmp[2]);
    Contract(&tmp[2], {0, 3}, &mps_ten2, {0, 1}, &tmp[3]);

    Contract<TenElemT, QNT, true, true>(mps_ten6, right_bten, 2, 0, 1, tmp[4]);
    Contract<TenElemT, QNT, false, false>(tmp[4], mpo_ten[5], 1, 0, 2, tmp[5]);
    Contract<TenElemT, QNT, false, false>(tmp[5], mpo_ten[4], 4, 1, 2, tmp[6]);
    Contract(&tmp[6], {0, 3}, &mps_ten5, {0, 1}, &tmp[7]);

    Contract<TenElemT, QNT, true, true>(mps_ten3, tmp[3], 2, 0, 1, tmp[8]);
    Contract<TenElemT, QNT, false, false>(tmp[8], mpo_ten[2], 1, 0, 2, tmp[9]);
    Contract<TenElemT, QNT, false, false>(tmp[9], mpo_ten[3], 4, 3, 2, tmp[10]);
    Contract(&tmp[10], {0, 3}, &mps_ten4, {0, 1}, &tmp[11]);

  } else { //mps_orient = VERTICAL
    const size_t row1 = left_up_site.row();
    const size_t row2 = row1 + 1;
    const size_t row3 = row2 + 1;
    const size_t col1 = left_up_site.col();
    const size_t col2 = col1 + 1;

    const Tensor &mps_ten1 = bmps_set_.at(LEFT)[col1][row3];
    const Tensor &mps_ten2 = bmps_set_.at(RIGHT)[tn.cols() - 1 - col2][tn.rows() - 1 - row3];
    const Tensor &mps_ten3 = bmps_set_.at(LEFT)[col1][row2];
    const Tensor &mps_ten4 = bmps_set_.at(RIGHT)[tn.cols() - 1 - col2][tn.rows() - 1 - row2];
    const Tensor &mps_ten5 = bmps_set_.at(LEFT)[col1][row1];
    const Tensor &mps_ten6 = bmps_set_.at(RIGHT)[tn.cols() - 1 - col2][tn.rows() - 1 - row1];

    const Tensor &top_bten = bten_set2_.at(UP)[row1];
    const Tensor &bottom_bten = bten_set2_.at(DOWN)[tn.rows() - row3 - 1];

    mpo_ten[2] = tn({row2, col1});
    mpo_ten[3] = tn({row2, col2});
    if (sqrt5link_dir == LEFTUP_TO_RIGHTDOWN) {
      mpo_ten[0] = tn({row3, col1});
      mpo_ten[1] = ten_right;
      mpo_ten[4] = ten_left;
      mpo_ten[5] = tn({row1, col2});
    } else { //LEFTDOWN_TO_RIGHTUP
      mpo_ten[0] = ten_left;
      mpo_ten[1] = tn({row3, col2});
      mpo_ten[4] = tn({row1, col1});
      mpo_ten[5] = ten_right;
    }
    mpo_ten[0].Transpose({0, 1, 3, 2});
    mpo_ten[2].Transpose({0, 1, 3, 2});
    mpo_ten[5].Transpose({2, 3, 1, 0});

    Contract<TenElemT, QNT, true, true>(mps_ten1, bottom_bten, 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, true>(tmp[0], mpo_ten[0], 1, 0, 2, tmp[1]);
    Contract<TenElemT, QNT, false, true>(tmp[1], mpo_ten[1], 4, 0, 2, tmp[2]);
    Contract(&tmp[2], {0, 3}, &mps_ten2, {0, 1}, &tmp[3]);

    Contract<TenElemT, QNT, true, true>(mps_ten6, top_bten, 2, 0, 1, tmp[4]);
    Contract<TenElemT, QNT, false, true>(tmp[4], mpo_ten[5], 1, 0, 2, tmp[5]);
    Contract<TenElemT, QNT, false, false>(tmp[5], mpo_ten[4], 4, 2, 2, tmp[6]);
    Contract(&tmp[6], {0, 3}, &mps_ten5, {0, 1}, &tmp[7]);

    Contract<TenElemT, QNT, true, true>(mps_ten3, tmp[3], 2, 0, 1, tmp[8]);
    Contract<TenElemT, QNT, false, true>(tmp[8], mpo_ten[2], 1, 0, 2, tmp[9]);
    Contract<TenElemT, QNT, false, true>(tmp[9], mpo_ten[3], 4, 0, 2, tmp[10]);
    Contract(&tmp[10], {0, 3}, &mps_ten4, {0, 1}, &tmp[11]);

  }
  Contract(&tmp[11], {0, 1, 2, 3}, &tmp[7], {3, 2, 1, 0}, &tmp[12]);
  return tmp[12]();
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::EraseEnvsAfterUpdate(const SiteIdx &site) {
  const size_t row = site[0];
  const size_t col = site[1];
  if (bmps_set_.at(LEFT).size() > col + 1) {
    bmps_set_[LEFT].erase(bmps_set_[LEFT].cbegin() + col + 1, bmps_set_[LEFT].end());
  }

  if (bmps_set_.at(UP).size() > row + 1) {
    bmps_set_[UP].erase(bmps_set_[UP].cbegin() + row + 1, bmps_set_[UP].end());
  }

  size_t down_allow_mps_num = rows_ - row;
  if (bmps_set_.at(DOWN).size() > down_allow_mps_num) {
    bmps_set_[DOWN].erase(bmps_set_[DOWN].cbegin() + down_allow_mps_num, bmps_set_[DOWN].end());
  }

  size_t right_allow_mps_num = cols_ - col;
  if (bmps_set_.at(RIGHT).size() > right_allow_mps_num) {
    bmps_set_[RIGHT].erase(bmps_set_[RIGHT].cbegin() + right_allow_mps_num, bmps_set_[RIGHT].end());
  }

  // Targeted invalidation for cached boundary tensors (BTen/BTen2).
  //
  // These caches are anchored at boundaries and indexed by the starting site coordinate
  // (e.g. LEFT bten at index 'col' corresponds to the environment up to that column).
  // After a local update at (row,col), any cached entries that cross this site become stale.
  // Truncate them in-place so caller-side local measurement routines can keep using the contractor
  // without rebuilding everything from scratch.
  auto truncate_bten = [](auto &bten_map, BTenPOSITION pos, size_t keep) {
    auto it = bten_map.find(pos);
    if (it == bten_map.end()) {
      return;
    }
    auto &vec = it->second;
    if (keep == 0) {
      vec.clear();
      return;
    }
    if (vec.size() > keep) {
      vec.resize(keep);
    }
  };
  truncate_bten(bten_set_, LEFT, col + 1);
  truncate_bten(bten_set_, UP, row + 1);
  truncate_bten(bten_set_, RIGHT, cols_ - col);
  truncate_bten(bten_set_, DOWN, rows_ - row);
  truncate_bten(bten_set2_, LEFT, col + 1);
  truncate_bten(bten_set2_, UP, row + 1);
  truncate_bten(bten_set2_, RIGHT, cols_ - col);
  truncate_bten(bten_set2_, DOWN, rows_ - row);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::CheckInvalidateEnvs(const SiteIdx &site) const {
#ifndef NDEBUG
  const size_t row = site[0];
  const size_t col = site[1];

  // After invalidation for a local update at (row,col), we must not keep BMPS caches
  // beyond that site in any direction, otherwise stale environments may be reused.
  assert(bmps_set_.at(LEFT).size() <= col + 1);
  assert(bmps_set_.at(UP).size() <= row + 1);

  const size_t down_allow_mps_num = rows_ - row;
  const size_t right_allow_mps_num = cols_ - col;
  assert(bmps_set_.at(DOWN).size() <= down_allow_mps_num);
  assert(bmps_set_.at(RIGHT).size() <= right_allow_mps_num);

  // BTens should not cross (row,col) after invalidation.
  auto check_bten = [](const auto &bten_map, BTenPOSITION pos, size_t max_len) {
    auto it = bten_map.find(pos);
    if (it == bten_map.end()) {
      return;
    }
    assert(it->second.size() <= max_len);
  };
  check_bten(bten_set_, LEFT, col + 1);
  check_bten(bten_set_, UP, row + 1);
  check_bten(bten_set_, RIGHT, cols_ - col);
  check_bten(bten_set_, DOWN, rows_ - row);
  check_bten(bten_set2_, LEFT, col + 1);
  check_bten(bten_set2_, UP, row + 1);
  check_bten(bten_set2_, RIGHT, cols_ - col);
  check_bten(bten_set2_, DOWN, rows_ - row);
#else
  (void)site;
#endif
}

// --------------------------------------------------------
// BMPSWalker Methods
// --------------------------------------------------------

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::Evolve(const TransferMPO& mpo, const BMPSTruncateParams<RealT> &trunc_para) {
  // MultipleMPO requires non-const ref due to internal alignment logic; make a local copy
  TransferMPO mpo_copy = mpo;
  bmps_ = bmps_.MultipleMPO(mpo_copy, trunc_para.compress_scheme,
                            trunc_para.D_min, trunc_para.D_max, trunc_para.trunc_err,
                            trunc_para.convergence_tol,
                            trunc_para.iter_max);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::EvolveStep(const BMPSTruncateParams<RealT> &trunc_para) {
  assert(stack_size_ > 0);
  size_t mpo_num;
  if (pos_ == UP || pos_ == LEFT) {
    // If stack_size is 1 (vacuum), next is row 0. mpo_num = 0.
    mpo_num = stack_size_ - 1;
  } else if (pos_ == DOWN) {
    // If stack_size is 1 (vacuum), next is row rows-1.
    // mpo_num = rows - 1.
    mpo_num = tn_.rows() - stack_size_;
  } else { // RIGHT
    mpo_num = tn_.cols() - stack_size_;
  }

  // Safety check for boundary over-evolution
  if (pos_ == UP && mpo_num >= tn_.rows() - 1) return; 
  if (pos_ == LEFT && mpo_num >= tn_.cols() - 1) return;
  // For DOWN and RIGHT, index goes 0..N-1, so check underflow/bounds if size_t wasn't unsigned
  
  const TransferMPO &mpo = tn_.get_slice(mpo_num, Rotate(Orientation(pos_)));
  Evolve(mpo, trunc_para);
  stack_size_++;
}

template<typename TenElemT, typename QNT>
typename BMPSContractor<TenElemT, QNT>::BMPSWalker 
BMPSContractor<TenElemT, QNT>::GetWalker(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION position) const {
  const auto& stack = bmps_set_.at(position);
  assert(!stack.empty() && "Cannot create Walker from empty BMPS stack");
  // Copy the top BMPS
  return BMPSWalker(tn, stack.back(), position, stack.size());
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::BMPSWalker::ContractRow(const TransferMPO& mpo, const BMPS<TenElemT, QNT>& opposite_boundary) const {
  // Contract <bmps_ | mpo | opposite_boundary> to get a scalar overlap.
  //
  // Tensor network structure (for UP walker, DOWN opposite):
  //   top[col] (UP BMPS tensor)
  //      |
  //   mpo[col] (TN site tensor)  
  //      |
  //   bot[col] (DOWN BMPS tensor)
  //
  // Index conventions:
  //   BMPS tensor (boson): 0:left, 1:physical, 2:right
  //   TN site tensor: 0:left, 1:down, 2:right, 3:up
  //
  // CRITICAL: Storage order differs by direction!
  //   - UP BMPS (bmps_): Reversed. bmps_[0] = rightmost column (col=N-1).
  //   - DOWN BMPS (opposite_boundary): Normal. opposite[0] = leftmost column (col=0).
  //   - site MPO: Normal. mpo[0] = leftmost column (col=0).
  //
  // Due to reversed UP storage, we contract from col=N-1 to col=0 (right-to-left)
  // to properly match virtual bonds between adjacent BMPS tensors.
  //
  // Connection pattern when contracting column (left) with accumulator (right):
  //   - acc.top_R connects column.top_L (UP BMPS internal bond)
  //   - column.site_R connects acc.site_L (site tensor horizontal bond)
  //   - column.bot_R connects acc.bot_L (DOWN BMPS internal bond)
  //
  // For OBC:
  //   - col=0: left bonds (top_L, site_L, bot_L) are trivial (dim=1)
  //   - col=N-1: right bonds (top_R, site_R, bot_R) are trivial (dim=1)
  
  using Tensor = qlten::QLTensor<TenElemT, QNT>;
  
  const size_t N = bmps_.size();
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
#ifndef NDEBUG
    std::cerr << "[ContractRow] Size mismatch: N=" << N 
              << ", mpo.size()=" << mpo.size()
              << ", opposite.size()=" << opposite_boundary.size() << std::endl;
#endif
    return TenElemT(0);
  }
  
  // Only handle UP-DOWN case for horizontal row contraction
  if (pos_ != UP || opposite_boundary.Direction() != DOWN) {
#ifndef NDEBUG
    std::cerr << "[ContractRow] Direction mismatch: pos_=" << static_cast<int>(pos_) 
              << " (expected UP=" << static_cast<int>(UP) << ")"
              << ", opposite.Direction()=" << static_cast<int>(opposite_boundary.Direction())
              << " (expected DOWN=" << static_cast<int>(DOWN) << ")" << std::endl;
#endif
    return TenElemT(0);
  }
  
  // Build accumulator by contracting RIGHT to LEFT (following UP BMPS storage order)
  // UP BMPS is stored reversed: bmps_[0] = rightmost column (N-1), bmps_[N-1] = leftmost column (0)
  // For UP BMPS: bmps_[i].right (idx2) connects bmps_[i+1].left (idx0)
  // For DOWN BMPS (normal order): opposite[i].right (idx2) connects opposite[i+1].left (idx0)
  //
  // To match both, we contract from col=N-1 to col=0 (right to left)
  // - UP BMPS: bmps_[0] to bmps_[N-1], i.e., storage order
  // - DOWN BMPS: opposite[N-1] to opposite[0], i.e., reverse storage order
  //
  // Each column tensor after contraction has indices:
  //   (top_L, top_R, site_L, site_R, bot_L, bot_R) = (0, 1, 2, 3, 4, 5)
  // Left bonds: (0, 2, 4), Right bonds: (1, 3, 5)

  Tensor accumulator;
  
  // Contract from col=N-1 (rightmost) to col=0 (leftmost)
  for (size_t i = 0; i < N; ++i) {
    size_t col = N - 1 - i;  // col goes from N-1 down to 0
    
    // UP BMPS: bmps_[i] corresponds to col = N-1-i
    // So for col, we use bmps_[N-1-col] = bmps_[N-1-(N-1-i)] = bmps_[i]
    const Tensor& top = bmps_[i];
    const Tensor& site = *mpo[col];
    const Tensor& bot = opposite_boundary[col];
    
    // Contract top[1] with site[3] (physical bond)
    Tensor top_site;
    qlten::Contract(&top, {1}, &site, {3}, &top_site);
    // top_site indices: (top_L, top_R, site_L, site_D, site_R) = (0, 1, 2, 3, 4)
    
    // Contract top_site[3] with bot[1] (physical bond)
    Tensor column;
    qlten::Contract(&top_site, {3}, &bot, {1}, &column);
    // column indices: (top_L, top_R, site_L, site_R, bot_L, bot_R) = (0, 1, 2, 3, 4, 5)
    
    if (i == 0) {
      // First iteration (rightmost column): just store the column tensor
      // At OBC right boundary: top_L (idx0), site_R (idx3), bot_R (idx5) are trivial
      accumulator = std::move(column);
    } else if (i == 1) {
      // Second iteration: contract to connect the two columns
      // Connection pattern (col is to the LEFT of acc in space):
      //   - UP BMPS: acc.top_R connects col.top_L (bmps_[i-1].idx2 connects bmps_[i].idx0)
      //   - site: col.site_R connects acc.site_L (site[col].idx2 connects site[col+1].idx0)
      //   - DOWN BMPS: col.bot_R connects acc.bot_L (opposite[col].idx2 connects opposite[col+1].idx0)
      // So: acc{top_R, site_L, bot_L} = acc{1, 2, 4} connects col{top_L, site_R, bot_R} = col{0, 3, 5}
      Tensor new_acc;
      qlten::Contract(&accumulator, {1, 2, 4}, &column, {0, 3, 5}, &new_acc);
      // Result indices: (acc:0, acc:3, acc:5, col:1, col:2, col:4)
      //               = (top_L_first, site_R_first, bot_R_first, top_R_col, site_L_col, bot_L_col)
      //               = (0, 1, 2, 3, 4, 5)
      // For next iteration: left bonds are (3, 4, 5), right bonds are (0, 1, 2)
      accumulator = std::move(new_acc);
    } else {
      // Subsequent iterations: 
      // acc indices: (top_L_first, site_R_first, bot_R_first, top_R_prev, site_L_prev, bot_L_prev)
      //            = (0, 1, 2, 3, 4, 5)
      // Connect: acc{top_R, site_L, bot_L} = acc{3, 4, 5} with col{top_L, site_R, bot_R} = col{0, 3, 5}
      Tensor new_acc;
      qlten::Contract(&accumulator, {3, 4, 5}, &column, {0, 3, 5}, &new_acc);
      // Result indices: (acc:0, acc:1, acc:2, col:1, col:2, col:4)
      //               = (top_L_first, site_R_first, bot_R_first, top_R_col, site_L_col, bot_L_col)
      accumulator = std::move(new_acc);
    }
  }
  
  // After all columns, accumulator has 6 indices:
  // (top_L_0, site_L_0, bot_L_0, top_R_{N-1}, site_R_{N-1}, bot_R_{N-1})
  // For OBC, all 6 indices are trivial (dim=1)
  
  size_t rank = accumulator.Rank();
  if (rank == 0) {
    return accumulator();
  }
  
  // Verify all indices are trivial and extract the scalar
  std::vector<size_t> coords(rank, 0);
  bool all_trivial = true;
  for (size_t i = 0; i < rank; ++i) {
    if (accumulator.GetIndex(i).dim() != 1) {
      all_trivial = false;
#ifndef NDEBUG
      std::cerr << "[ContractRow] Non-trivial index at position " << i 
                << ", dim=" << accumulator.GetIndex(i).dim() 
                << ", total rank=" << rank << std::endl;
#endif
    }
  }
  
  if (!all_trivial) {
#ifndef NDEBUG
    std::cerr << "[ContractRow] N=" << N 
              << ", bmps_.size()=" << bmps_.size()
              << ", opposite.size()=" << opposite_boundary.size()
              << ", opposite.Direction()=" << static_cast<int>(opposite_boundary.Direction())
              << ", pos_=" << static_cast<int>(pos_) << std::endl;
    // Print first and last BMPS tensor shapes
    if (N > 0) {
      std::cerr << "[ContractRow] bmps_[0] rank=" << bmps_[0].Rank();
      for (size_t j = 0; j < bmps_[0].Rank(); ++j) {
        std::cerr << " idx" << j << ".dim=" << bmps_[0].GetIndex(j).dim();
      }
      std::cerr << std::endl;
      std::cerr << "[ContractRow] bmps_[N-1] rank=" << bmps_[N-1].Rank();
      for (size_t j = 0; j < bmps_[N-1].Rank(); ++j) {
        std::cerr << " idx" << j << ".dim=" << bmps_[N-1].GetIndex(j).dim();
      }
      std::cerr << std::endl;
      std::cerr << "[ContractRow] opposite[0] rank=" << opposite_boundary[0].Rank();
      for (size_t j = 0; j < opposite_boundary[0].Rank(); ++j) {
        std::cerr << " idx" << j << ".dim=" << opposite_boundary[0].GetIndex(j).dim();
      }
      std::cerr << std::endl;
    }
#endif
    return TenElemT(0);
  }
  
  return accumulator.GetElem(coords);
}

// ============================================================================
//                    BTen Cache Implementation (BMPSWalker)
// ============================================================================
//
// This section implements BTen caching for BMPSWalker, enabling O(Lx) row
// contraction instead of O(Lx²) when computing multiple traces on the same row.
//
// ## Index Conventions
// BMPSWalker BTen uses the SAME index conventions as BMPSContractor::InitBTen().
// See bmps_contractor.h for complete documentation of all 4-direction BTen indices.
//
// For UP walker with DOWN opposite (horizontal row contraction):
//   - LEFT BTen:  (UP_R, site_L, DOWN_L) - matches InitBTen(LEFT, row)
//   - RIGHT BTen: (DOWN_R, site_R, UP_L) - matches InitBTen(RIGHT, row)
//
// ## Storage Order Reminder
//   - UP BMPS (walker.bmps_): REVERSED. bmps_[i] → col = N-1-i
//   - DOWN BMPS (opposite):   Normal.   opposite[i] → col = i
//   - Site MPO (mpo):         Normal.   mpo[i] → col = i
//
// ## Contraction Patterns (bosonic case)
// Uses the same patterns as GrowFullBTen:
//
// ### GrowBTenLeftStep (LEFT pattern):
//   Contract<true,true>(up_mps_ten, btens.back(), 2, 0, 1, tmp1);
//   Contract<false,false>(tmp1, mpo_ten, 1, 3, 2, tmp2);
//   Contract(&tmp2, {0, 2}, &down_mps_ten, {0, 1}, &tmp3);
//
// ### GrowBTenRightStep (RIGHT pattern):
//   Contract<true,true>(down_mps_ten, btens.back(), 2, 0, 1, tmp1);
//   Contract<false,false>(tmp1, mpo_ten, 1, 1, 2, tmp2);
//   Contract(&tmp2, {0, 2}, &up_mps_ten, {0, 1}, &tmp3);
//
// ### TraceWithBTen: Contract at column k
//
//   +---------+     up[k]      +---------+
//   | LEFT[k] |=====[L,R]=====| RIGHT   |
//   |         |    site[k]    | [N-1-k] |
//   |         |====[L,R,U,D]==|         |
//   |         |    down[k]    |         |
//   |         |=====[L,R]=====|         |
//   +---------+               +---------+
//         ↓
//      Scalar result

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::InitBTenLeft(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary, 
    size_t target_col) {
  using IndexT = qlten::Index<QNT>;
  
  const size_t N = bmps_.size();
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
    return;
  }
  
  bten_left_.clear();
  bten_left_col_ = 0;
  
  // Following the same pattern as BMPSContractor::InitBTen(LEFT, row):
  // vacuum BTen connects via:
  //   index0: UP BMPS last tensor (col=0 in reversed) RIGHT index
  //   index1: site tensor at col=0 LEFT index
  //   index2: DOWN BMPS first tensor (col=0) LEFT index
  
  const Tensor& up_ten_col0 = bmps_[N - 1];  // UP reversed: bmps_[N-1] = col 0
  const Tensor& site_col0 = *mpo[0];
  const Tensor& down_ten_col0 = opposite_boundary[0];
  
  // Match InitBTen LEFT pattern: index0 from UP[R], index1 from site[L], index2 from DOWN[L]
  IndexT index0 = qlten::InverseIndex(up_ten_col0.GetIndex(2));    // UP col0 RIGHT
  IndexT index1 = qlten::InverseIndex(site_col0.GetIndex(0));      // site col0 LEFT
  IndexT index2 = qlten::InverseIndex(down_ten_col0.GetIndex(0));  // DOWN col0 LEFT
  
#ifndef NDEBUG
  std::cerr << "[InitBTenLeft] N=" << N << ", target_col=" << target_col << std::endl;
  std::cerr << "[InitBTenLeft] up_ten_col0: "; up_ten_col0.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[InitBTenLeft] site_col0: "; site_col0.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[InitBTenLeft] down_ten_col0: "; down_ten_col0.ConciseShow(0); std::cerr << std::endl;
#endif
  
  Tensor vacuum_bten({index0, index1, index2});
  vacuum_bten({0, 0, 0}) = TenElemT(1.0);
  
#ifndef NDEBUG
  std::cerr << "[InitBTenLeft] vacuum_bten: "; vacuum_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  bten_left_.push_back(std::move(vacuum_bten));
  
  // Grow to target_col
  while (bten_left_col_ < target_col && bten_left_col_ < N) {
    GrowBTenLeftStep(mpo, opposite_boundary);
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::InitBTenRight(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary, 
    size_t target_col) {
  using IndexT = qlten::Index<QNT>;
  
  const size_t N = bmps_.size();
  if (N == 0 || mpo.size() != N || opposite_boundary.size() != N) {
    return;
  }
  
  bten_right_.clear();
  bten_right_col_ = N;  // Right edge starts at N (nothing absorbed yet)
  
  // Following the same pattern as BMPSContractor::InitBTen(RIGHT, row):
  // vacuum BTen connects via:
  //   index0: DOWN BMPS last tensor (col=N-1) RIGHT index
  //   index1: site tensor at col=N-1 RIGHT index
  //   index2: UP BMPS first tensor (col=N-1 in reversed) LEFT index
  
  const Tensor& down_ten_colN1 = opposite_boundary[N - 1];
  const Tensor& site_colN1 = *mpo[N - 1];
  const Tensor& up_ten_colN1 = bmps_[0];  // UP reversed: bmps_[0] = col N-1
  
  // Match InitBTen RIGHT pattern
  IndexT index0 = qlten::InverseIndex(down_ten_colN1.GetIndex(2));  // DOWN col=N-1 RIGHT
  IndexT index1 = qlten::InverseIndex(site_colN1.GetIndex(2));      // site col=N-1 RIGHT
  IndexT index2 = qlten::InverseIndex(up_ten_colN1.GetIndex(0));    // UP col=N-1 LEFT
  
#ifndef NDEBUG
  std::cerr << "[InitBTenRight] N=" << N << ", target_col=" << target_col << std::endl;
  std::cerr << "[InitBTenRight] down_ten_colN1: "; down_ten_colN1.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[InitBTenRight] site_colN1: "; site_colN1.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[InitBTenRight] up_ten_colN1: "; up_ten_colN1.ConciseShow(0); std::cerr << std::endl;
#endif
  
  Tensor vacuum_bten({index0, index1, index2});
  vacuum_bten({0, 0, 0}) = TenElemT(1.0);
  
#ifndef NDEBUG
  std::cerr << "[InitBTenRight] vacuum_bten: "; vacuum_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  bten_right_.push_back(std::move(vacuum_bten));
  
  // Grow to target_col (towards left)
  // target_col means we want RIGHT BTen to cover (target_col, N-1]
  // So we need bten_right_col_ == target_col + 1
  while (bten_right_col_ > target_col + 1 && bten_right_col_ > 0) {
    GrowBTenRightStep(mpo, opposite_boundary);
  }
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::GrowBTenLeftStep(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary) {
  const size_t N = bmps_.size();
  if (bten_left_.empty() || bten_left_col_ >= N) {
    return;
  }
  
  // Absorb column at bten_left_col_
  const size_t col = bten_left_col_;
  
  // Get tensors for this column (same as GrowFullBTen LEFT case)
  const Tensor& up_mps_ten = bmps_[N - 1 - col];  // UP reversed
  const Tensor& mpo_ten = *mpo[col];
  const Tensor& down_mps_ten = opposite_boundary[col];
  const Tensor& left_bten = bten_left_.back();

#ifndef NDEBUG
  std::cerr << "[GrowBTenLeftStep] col=" << col << std::endl;
  std::cerr << "[GrowBTenLeftStep] up_mps_ten: "; up_mps_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenLeftStep] mpo_ten: "; mpo_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenLeftStep] down_mps_ten: "; down_mps_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenLeftStep] left_bten: "; left_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  // Use EXACT same contraction pattern as GrowFullBTen LEFT case (bosonic):
  // Contract<TenElemT, QNT, true, true>(up_mps_ten, btens.back(), 2, 0, 1, tmp1);
  // Contract<TenElemT, QNT, false, false>(tmp1, *mpo[i], 1, 3, 2, tmp2);
  // Contract(&tmp2, {0, 2}, &down_mps_ten, {0, 1}, &tmp3);
  
  Tensor tmp1, tmp2, next_bten;
  qlten::Contract<TenElemT, QNT, true, true>(up_mps_ten, left_bten, 2, 0, 1, tmp1);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenLeftStep] tmp1 after up*bten: "; tmp1.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 3, 2, tmp2);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenLeftStep] tmp2 after tmp1*mpo: "; tmp2.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract(&tmp2, {0, 2}, &down_mps_ten, {0, 1}, &next_bten);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenLeftStep] next_bten: "; next_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  bten_left_.push_back(std::move(next_bten));
  bten_left_col_++;
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BMPSWalker::GrowBTenRightStep(
    const TransferMPO& mpo, 
    const BMPS<TenElemT, QNT>& opposite_boundary) {
  const size_t N = bmps_.size();
  if (bten_right_.empty() || bten_right_col_ == 0) {
    return;
  }
  
  // Absorb column at bten_right_col_ - 1 (moving left)
  const size_t col = bten_right_col_ - 1;
  
  // Get tensors for this column (same indexing as GrowFullBTen RIGHT case)
  // In GrowFullBTen RIGHT: i-th iteration processes col = N-1-i
  // Here we use col directly, so we need:
  // up_mps_ten = up_bmps[N-1-col] = bmps_[N-1-col] but wait...
  // Actually in GrowFullBTen RIGHT: up_mps_ten = up_bmps[i] where col=N-1-i
  // So when col=N-1, i=0, up_mps_ten = up_bmps[0]
  // Since UP BMPS is reversed: up_bmps[0] corresponds to col=N-1. Correct!
  // When col=N-2, i=1, up_mps_ten = up_bmps[1] corresponds to col=N-2. Correct!
  
  const Tensor& up_mps_ten = bmps_[N - 1 - col];  // UP reversed
  const Tensor& down_mps_ten = opposite_boundary[col];
  const Tensor& mpo_ten = *mpo[col];
  const Tensor& right_bten = bten_right_.back();
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenRightStep] col=" << col << std::endl;
  std::cerr << "[GrowBTenRightStep] down_mps_ten: "; down_mps_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenRightStep] mpo_ten: "; mpo_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenRightStep] up_mps_ten: "; up_mps_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[GrowBTenRightStep] right_bten: "; right_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  // Use EXACT same contraction pattern as GrowFullBTen RIGHT case (bosonic):
  // Contract<TenElemT, QNT, true, true>(down_mps_ten, btens.back(), 2, 0, 1, tmp1);
  // Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 1, 2, tmp2);
  // Contract(&tmp2, {0, 2}, &up_mps_ten, {0, 1}, &tmp3);
  
  Tensor tmp1, tmp2, next_bten;
  qlten::Contract<TenElemT, QNT, true, true>(down_mps_ten, right_bten, 2, 0, 1, tmp1);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenRightStep] tmp1 after down*bten: "; tmp1.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 1, 2, tmp2);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenRightStep] tmp2 after tmp1*mpo: "; tmp2.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract(&tmp2, {0, 2}, &up_mps_ten, {0, 1}, &next_bten);
  
#ifndef NDEBUG
  std::cerr << "[GrowBTenRightStep] next_bten: "; next_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  bten_right_.push_back(std::move(next_bten));
  bten_right_col_--;
}

template<typename TenElemT, typename QNT>
TenElemT BMPSContractor<TenElemT, QNT>::BMPSWalker::TraceWithBTen(
    const Tensor& site, 
    size_t site_col, 
    const BMPS<TenElemT, QNT>& opposite_boundary) const {
  const size_t N = bmps_.size();
  
  // Check if BTen caches are available and cover the required columns
  // LEFT BTen should cover [0, site_col), so bten_left_col_ >= site_col
  // RIGHT BTen should cover (site_col, N-1], so bten_right_col_ <= site_col + 1
  if (bten_left_.empty() || bten_right_.empty()) {
    return TenElemT(0);  // BTen not initialized
  }
  
  if (bten_left_col_ < site_col || bten_right_col_ > site_col + 1) {
    return TenElemT(0);  // BTen doesn't cover the required range
  }
  
  // Get the appropriate BTen
  // bten_left_[k] covers [0, k-1] after k grow steps, so for [0, site_col) we need bten_left_[site_col]
  // But bten_left_[0] is vacuum, bten_left_[k] covers [0, k-1]
  // Actually: bten_left_[0] is vacuum (before col 0), bten_left_[k] has absorbed cols [0, k-1]
  // So to cover [0, site_col), we need bten_left_[site_col]
  const Tensor& left_bten = bten_left_[site_col];
  
  // bten_right_[0] is vacuum (after col N-1), bten_right_[k] has absorbed cols [N-k, N-1]
  // For (site_col, N-1], we need to have absorbed cols [site_col+1, N-1]
  // Number of cols = N-1 - site_col = N - 1 - site_col
  // So we need bten_right_[N - 1 - site_col]
  const size_t right_idx = N - 1 - site_col;
  if (right_idx >= bten_right_.size()) {
    return TenElemT(0);
  }
  const Tensor& right_bten = bten_right_[right_idx];
  
  // Get tensors at site_col
  const Tensor& up_ten = bmps_[N - 1 - site_col];  // UP reversed
  const Tensor& down_ten = opposite_boundary[site_col];
  
#ifndef NDEBUG
  std::cerr << "[TraceWithBTen] site_col=" << site_col << ", right_idx=" << right_idx << std::endl;
  std::cerr << "[TraceWithBTen] left_bten: "; left_bten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[TraceWithBTen] up_ten: "; up_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[TraceWithBTen] site: "; site.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[TraceWithBTen] down_ten: "; down_ten.ConciseShow(0); std::cerr << std::endl;
  std::cerr << "[TraceWithBTen] right_bten: "; right_bten.ConciseShow(0); std::cerr << std::endl;
#endif
  
  // The BTen structure from GrowFullBTen (LEFT case):
  // After contracting up_mps_ten[2] with bten[0], then mpo, then down_mps_ten,
  // the result has indices connecting to the RIGHT side of the current column.
  // So left_bten's indices should connect to column's LEFT sides, not the MPS tensors' L indices.
  //
  // Looking at the contraction in GrowFullBTen more carefully:
  // Contract<true,true>(up_mps_ten, btens.back(), 2, 0, 1, tmp1)
  // This contracts up_mps_ten[R=2] with bten[0].
  // So bten[0] should be compatible with up_mps_ten[R], meaning bten[0] connects from the RIGHT.
  //
  // This is confusing. Let me just follow the same contraction pattern as GrowFullBTen
  // but for calculating the full trace.
  //
  // For a single-column trace, we need to contract:
  // left_bten -- up_mps_ten -- right_bten
  //            |-- mpo_ten --|
  //            -- down_mps_ten --
  //
  // Using GrowFullBTen pattern:
  // Step 1: Contract up_mps_ten with left_bten (same as GrowBTenLeftStep)
  // Step 2: Contract result with mpo_ten
  // Step 3: Contract result with down_mps_ten
  // Step 4: Contract result with right_bten
  
  Tensor tmp1, tmp2, tmp3, result;
  
  // Follow GrowBTenLeftStep pattern exactly for first 3 contractions
  qlten::Contract<TenElemT, QNT, true, true>(up_ten, left_bten, 2, 0, 1, tmp1);
  
#ifndef NDEBUG
  std::cerr << "[TraceWithBTen] tmp1: "; tmp1.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract<TenElemT, QNT, false, false>(tmp1, site, 1, 3, 2, tmp2);
  
#ifndef NDEBUG
  std::cerr << "[TraceWithBTen] tmp2: "; tmp2.ConciseShow(0); std::cerr << std::endl;
#endif
  
  qlten::Contract(&tmp2, {0, 2}, &down_ten, {0, 1}, &tmp3);
  
#ifndef NDEBUG
  std::cerr << "[TraceWithBTen] tmp3: "; tmp3.ConciseShow(0); std::cerr << std::endl;
#endif
  
  // Now tmp3 has the same structure as a LEFT BTen after absorbing one column
  // tmp3 indices should match what right_bten expects to contract with
  // right_bten from GrowFullBTen RIGHT case has indices after Contract pattern:
  // Contract<true,true>(down_mps_ten, btens.back(), 2, 0, 1, tmp1)
  // So right_bten[0] connects to down_mps_ten[R]
  // This means tmp3 needs to be contracted with right_bten appropriately
  
  // Contract all remaining indices
  qlten::Contract(&tmp3, {0, 1, 2}, &right_bten, {2, 1, 0}, &result);
  
  // Result should be a scalar
  if (result.Rank() != 0) {
    return TenElemT(0);
  }
  
  return result();
}

} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_BMPS_CONTRACTOR_IMPL_H
