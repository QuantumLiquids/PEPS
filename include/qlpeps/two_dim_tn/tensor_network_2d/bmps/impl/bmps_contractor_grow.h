// SPDX-License-Identifier: LGPL-3.0-only
#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_GROW_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_GROW_H

#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bmps_contractor_helpers.h"

namespace qlpeps {

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
BMPSContractor<TenElemT, QNT>::GenerateBMPSApproach(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION post) {
  DeleteInnerBMPS(post);
  GrowFullBMPS(tn, Opposite(post));
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
size_t BMPSContractor<TenElemT, QNT>::GrowBMPSStep(const BMPSPOSITION position,
                                                   TransferMPO mpo) {
  const auto &trunc_para = GetTruncateParams();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  bmps_set.push_back(
      bmps_set.back().MultiplyMPO(mpo, trunc_para.compress_scheme,
                                  trunc_para.D_min, trunc_para.D_max, trunc_para.trunc_err,
                                  trunc_para.convergence_tol,
                                  trunc_para.iter_max));
  return bmps_set.size();
}

template<typename TenElemT, typename QNT>
size_t BMPSContractor<TenElemT, QNT>::GrowBMPSStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position) {
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
  return GrowBMPSStep(position, mpo);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::GrowFullBMPS(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  size_t existed_bmps_size = bmps_set.size();
  assert(existed_bmps_size > 0);
  size_t rows = tn.rows();
  size_t cols = tn.cols();
  switch (position) {
    case DOWN: {
      for (size_t row = rows - existed_bmps_size; row > 0; row--) {
        const TransferMPO &mpo = tn.get_row(row);
        GrowBMPSStep(position, mpo);
      }
      break;
    }
    case UP: {
      for (size_t row = existed_bmps_size - 1; row < rows - 1; row++) {
        const TransferMPO &mpo = tn.get_row(row);
        GrowBMPSStep(position, mpo);
      }
      break;
    }
    case LEFT: {
      for (size_t col = existed_bmps_size - 1; col < cols - 1; col++) {
        const TransferMPO &mpo = tn.get_col(col);
        GrowBMPSStep(position, mpo);
      }
      break;
    }
    case RIGHT: {
      for (size_t col = cols - existed_bmps_size; col > 0; col--) {
        const TransferMPO &mpo = tn.get_col(col);
        GrowBMPSStep(position, mpo);
      }
      break;
    }
  }
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
BMPSContractor<TenElemT, QNT>::GrowBMPSForRow(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t row) {
  const size_t rows = tn.rows();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set_down = bmps_set_[DOWN];
  for (size_t row_bmps = rows - bmps_set_down.size(); row_bmps > row; row_bmps--) {
    const TransferMPO &mpo = tn.get_row(row_bmps);
    GrowBMPSStep(DOWN, mpo);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_up = bmps_set_[UP];
  for (size_t row_bmps = bmps_set_up.size() - 1; row_bmps < row; row_bmps++) {
    const TransferMPO &mpo = tn.get_row(row_bmps);
    GrowBMPSStep(UP, mpo);
  }
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
BMPSContractor<TenElemT, QNT>::GrowBMPSForCol(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t col) {
  const size_t cols = tn.cols();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set_right = bmps_set_[RIGHT];
  for (size_t col_bmps = cols - bmps_set_right.size(); col_bmps > col; col_bmps--) {
    const TransferMPO &mpo = tn.get_col(col_bmps);
    GrowBMPSStep(RIGHT, mpo);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_left = bmps_set_[LEFT];
  for (size_t col_bmps = bmps_set_left.size() - 1; col_bmps < col; col_bmps++) {
    const TransferMPO &mpo = tn.get_col(col_bmps);
    GrowBMPSStep(LEFT, mpo);
  }
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
const std::pair<BMPS<TenElemT, QNT>, BMPS<TenElemT, QNT> >
BMPSContractor<TenElemT, QNT>::GetBMPSForRow(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t row) {
  GrowBMPSForRow(tn, row);
  BMPST &up_bmps = bmps_set_[UP][row];
  BMPST &down_bmps = BMPSAtSlice_(DOWN, row);

  return std::pair(up_bmps, down_bmps);
}

template<typename TenElemT, typename QNT>
const std::pair<BMPS<TenElemT, QNT>, BMPS<TenElemT, QNT> >
BMPSContractor<TenElemT, QNT>::GetBMPSForCol(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t col) {
  GrowBMPSForCol(tn, col);
  BMPST &left_bmps = bmps_set_[LEFT][col];
  BMPST &right_bmps = BMPSAtSlice_(RIGHT, col);
  return std::pair(left_bmps, right_bmps);
}

template<typename TenElemT, typename QNT>
void BMPSContractor<TenElemT, QNT>::ShiftBMPSWindow(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position) {
  bmps_set_[position].pop_back();
  auto oppo_post = Opposite(position);
  GrowBMPSStep(tn, oppo_post);
}

template<typename TenElemT, typename QNT>
typename BMPSContractor<TenElemT, QNT>::Tensor 
BMPSContractor<TenElemT, QNT>::PunchHole(const TensorNetwork2D<TenElemT, QNT>& tn, const qlpeps::SiteIdx &site,
                                         const qlpeps::BondOrientation mps_orient) const {
  const Tensor *left_ten, *down_ten, *right_ten, *up_ten;
  const size_t row = site[0];
  const size_t col = site[1];
  
  if (mps_orient == HORIZONTAL) {
    up_ten = &(BMPSAtSlice_(UP, row).AtLogicalCol(col));
    down_ten = &(BMPSAtSlice_(DOWN, row).AtLogicalCol(col));
    left_ten = &(bten_set_.at(LEFT)[col]);
    right_ten = &(BTenAtSlice_(RIGHT, col));
  } else {
    up_ten = &(bten_set_.at(UP)[row]);
    down_ten = &(BTenAtSlice_(DOWN, row));
    left_ten = &(BMPSAtSlice_(LEFT, col).AtLogicalCol(row));
    right_ten = &(BMPSAtSlice_(RIGHT, col).AtLogicalCol(row));
  }
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
      auto &left_bmps = BMPSAtSlice_(LEFT, col);
      auto &right_bmps = BMPSAtSlice_(RIGHT, col);
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
      auto &left_bmps = BMPSAtSlice_(LEFT, col);
      auto &right_bmps = BMPSAtSlice_(RIGHT, col);
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
      auto &up_bmps = BMPSAtSlice_(UP, row);
      auto &down_bmps = BMPSAtSlice_(DOWN, row);
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
      auto &up_bmps = BMPSAtSlice_(UP, row);
      auto &down_bmps = BMPSAtSlice_(DOWN, row);
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
      bmps_pre = &BMPSAtSlice_(LEFT, col1);
      bmps_post = &BMPSAtSlice_(RIGHT, col2);
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
      bmps_pre = &BMPSAtSlice_(DOWN, row2);
      bmps_post = &BMPSAtSlice_(UP, row1);
      break;
    }
    case UP: {
      const size_t col1 = slice_num1;
      const size_t col2 = slice_num1 + 1;
      mpo1 = tn.get_col(col2);
      mpo2 = tn.get_col(col1);
      N = mpo1.size(); // tn.rows();
      bmps_pre = &BMPSAtSlice_(RIGHT, col2);
      bmps_post = &BMPSAtSlice_(LEFT, col1);
      break;
    }
    case LEFT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      mpo1 = tn.get_row(row1);
      mpo2 = tn.get_row(row2);
      N = mpo1.size(); // tn.cols()
      bmps_pre = &BMPSAtSlice_(UP, row1);
      bmps_post = &BMPSAtSlice_(DOWN, row2);
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

} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_GROW_H
