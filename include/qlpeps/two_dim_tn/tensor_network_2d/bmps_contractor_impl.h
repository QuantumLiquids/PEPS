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
 * Helper for GrowBTen2Step_.
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
 * Helper for GrowBTen2Step_.
 * It defines the final contraction for the GrowBTen2Step_.
 * The indices of bten2, mps_ten1 and mps_ten2 follow the original tensors indices order.
 * The mpo_ten1 needs to be transposed as defined in GrowBTen2Step_.
 * For bosonic tensor, mpo_ten2 is original mpo tensor;
 * while for fermionic tensor, mpo_ten2 should be tranposed as defined in GrowBTen2Step_.
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
size_t BMPSContractor<TenElemT, QNT>::GrowBMPSStep_(const BMPSPOSITION position,
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
size_t BMPSContractor<TenElemT, QNT>::GrowBMPSStep_(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para) {
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
  return GrowBMPSStep_(position, mpo, trunc_para);
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
        GrowBMPSStep_(position, mpo, trunc_para);
      }
      break;
    }
    case UP: {
      for (size_t row = existed_bmps_size - 1; row < rows - 1; row++) {
        const TransferMPO &mpo = tn.get_row(row);
        GrowBMPSStep_(position, mpo, trunc_para);
      }
      break;
    }
    case LEFT: {
      for (size_t col = existed_bmps_size - 1; col < cols - 1; col++) {
        const TransferMPO &mpo = tn.get_col(col);
        GrowBMPSStep_(position, mpo, trunc_para);
      }
    }
    case RIGHT: {
      for (size_t col = cols - existed_bmps_size; col > 0; col--) {
        const TransferMPO &mpo = tn.get_col(col);
        GrowBMPSStep_(position, mpo, trunc_para);
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
    GrowBMPSStep_(DOWN, mpo, trunc_para);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_up = bmps_set_[UP];
  for (size_t row_bmps = bmps_set_up.size() - 1; row_bmps < row; row_bmps++) {
    const TransferMPO &mpo = tn.get_row(row_bmps);
    GrowBMPSStep_(UP, mpo, trunc_para);
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
    GrowBMPSStep_(RIGHT, mpo, trunc_para);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_left = bmps_set_[LEFT];
  for (size_t col_bmps = bmps_set_left.size() - 1; col_bmps < col; col_bmps++) {
    const TransferMPO &mpo = tn.get_col(col_bmps);
    GrowBMPSStep_(LEFT, mpo, trunc_para);
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
void BMPSContractor<TenElemT, QNT>::BMPSMoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para) {
  bmps_set_[position].pop_back();
  auto oppo_post = Opposite(position);
  GrowBMPSStep_(tn, oppo_post, trunc_para);
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
void BMPSContractor<TenElemT, QNT>::GrowBTen2Step_(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post, const size_t slice_num1) {
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
void BMPSContractor<TenElemT, QNT>::BTenMoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position) {
  bten_set_[position].pop_back();
  GrowBTenStep(tn, Opposite(position));
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::BTen2MoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1) {
  bten_set2_[position].pop_back();
  GrowBTen2Step_(tn, Opposite(position), slice_num1);
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
void BMPSContractor<TenElemT, QNT>::InvalidateEnvs(const SiteIdx &site) {
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
}

} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_BMPS_CONTRACTOR_IMPL_H
