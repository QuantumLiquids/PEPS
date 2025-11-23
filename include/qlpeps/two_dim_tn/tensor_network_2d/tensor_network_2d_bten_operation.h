// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-07
*
* Description: QuantumLiquids/PEPS project. The 2-dimensional tensor network class, implementation for operations on boundary tensors.
*/

#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BTEN_OPERATION_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BTEN_OPERATION_H

namespace qlpeps {
using qlmps::IN;
using qlmps::OUT;

/**
 * @brief Initialize the boundary tensor for a given position and slice number.
 * 
 * @param position The position of the boundary tensor (DOWN, UP, LEFT, RIGHT).
 * @param slice_num row or column number for the boundary tensor.
 */
template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::InitBTen(const qlpeps::BTenPOSITION position, const size_t slice_num) {
  bten_set_[position].clear();
  IndexT index0, index1, index2;
  switch (position) {
    case DOWN: {
      const size_t col = slice_num;
      index0 = InverseIndex(bmps_set_[LEFT][col](this->rows() - 1)->GetIndex(2));
      index1 = InverseIndex((*this)(this->rows() - 1, col)->GetIndex(position));
      index2 = InverseIndex(bmps_set_[RIGHT][this->cols() - col - 1](0)->GetIndex(0));
      break;
    }
    case UP: {
      const size_t col = slice_num;
      index0 = InverseIndex(bmps_set_[RIGHT][this->cols() - col - 1](this->rows() - 1)->GetIndex(2));
      index1 = InverseIndex((*this)(0, col)->GetIndex(position));
      index2 = InverseIndex(bmps_set_[LEFT][col](0)->GetIndex(0));
      break;
    }
    case LEFT: {
      const size_t row = slice_num;
      index0 = InverseIndex(bmps_set_[UP][row](this->cols() - 1)->GetIndex(2));
      index1 = InverseIndex((*this)(row, 0)->GetIndex(position));
      index2 = InverseIndex(bmps_set_[DOWN][this->rows() - row - 1](0)->GetIndex(0));
      break;
    }
    case RIGHT: {
      const size_t row = slice_num;
      index0 = InverseIndex(bmps_set_[DOWN][this->rows() - row - 1](this->cols() - 1)->GetIndex(2));
      index1 = InverseIndex((*this)(row, this->cols() - 1)->GetIndex(position));
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
void TensorNetwork2D<TenElemT, QNT>::TruncateBTen(const qlpeps::BTenPOSITION position, const size_t length) {
  auto &btens = bten_set_.at(position);
  if (btens.size() > length) {
    btens.resize(length);
  }
}
///< slice_num1 is the smaller row/col value.
template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::InitBTen2(const BTenPOSITION position, const size_t slice_num1) {
  bten_set2_[position].clear();
  IndexT index0, index1, index2, index3;

  switch (position) {
    case DOWN: {
      const size_t col1 = slice_num1;
      const size_t col2 = col1 + 1;
      index0 = InverseIndex(bmps_set_[LEFT][col1](this->rows() - 1)->GetIndex(2));
      index1 = InverseIndex((*this)(this->rows() - 1, col1)->GetIndex(position));
      index2 = InverseIndex((*this)(this->rows() - 1, col2)->GetIndex(position));
      index3 = InverseIndex(bmps_set_[RIGHT][this->cols() - col2 - 1](0)->GetIndex(0));
      break;
    }
    case UP: {
      const size_t col1 = slice_num1;
      const size_t col2 = col1 + 1;
      index0 = InverseIndex(bmps_set_[RIGHT][this->cols() - col2 - 1](this->rows() - 1)->GetIndex(2));
      index1 = InverseIndex((*this)(0, col2)->GetIndex(position));
      index2 = InverseIndex((*this)(0, col1)->GetIndex(position));
      index3 = InverseIndex(bmps_set_[LEFT][col1](0)->GetIndex(0));
      break;
    }
    case LEFT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      index0 = InverseIndex(bmps_set_[UP][row1](this->cols() - 1)->GetIndex(2));
      index1 = InverseIndex((*this)(row1, 0)->GetIndex(position));
      index2 = InverseIndex((*this)(row2, 0)->GetIndex(position));
      index3 = InverseIndex(bmps_set_[DOWN][this->rows() - row2 - 1](0)->GetIndex(0));
      break;
    }
    case RIGHT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      index0 = InverseIndex(bmps_set_[DOWN][this->rows() - row2 - 1](this->cols() - 1)->GetIndex(2));
      index1 = InverseIndex((*this)(row2, this->cols() - 1)->GetIndex(position));
      index2 = InverseIndex((*this)(row1, this->cols() - 1)->GetIndex(position));
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
void TensorNetwork2D<TenElemT, QNT>::GrowFullBTen(const qlpeps::BTenPOSITION position,
                                                  const size_t slice_num,
                                                  const size_t remain_sites,
                                                  bool init) {
  if (init) {
    InitBTen(position, slice_num);
  }
  size_t start_idx = bten_set_[position].size() - 1;
  std::vector<Tensor> &btens = bten_set_[position];
  switch (position) {
    case DOWN: {
      const size_t col = slice_num;
      const TransferMPO &mpo = this->get_col(col);
      const size_t N = mpo.size(); // this->cols()
      const size_t end_idx = N - remain_sites;
//      GrowBMPSForCol(col);
      auto &left_bmps = bmps_set_[LEFT][col];
      auto &right_bmps = bmps_set_[RIGHT][this->cols() - col - 1];
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
      const TransferMPO &mpo = this->get_col(col);
      const size_t N = mpo.size(); // this->cols()
      const size_t end_idx = N - remain_sites;
//      GrowBMPSForCol(col);
      auto &left_bmps = bmps_set_[LEFT][col];
      auto &right_bmps = bmps_set_[RIGHT][this->cols() - col - 1];
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
      const TransferMPO &mpo = this->get_row(row);
      const size_t N = mpo.size(); // this->cols()
      const size_t end_idx = N - remain_sites;
//      GrowBMPSForRow(row);
      auto &up_bmps = bmps_set_[UP][row];
      auto &down_bmps = bmps_set_[DOWN][this->rows() - row - 1];
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
      const TransferMPO &mpo = this->get_row(row);
      const size_t N = mpo.size(); // this->cols()
      const size_t end_idx = N - remain_sites;
//      GrowBMPSForRow(row);
      auto &up_bmps = bmps_set_[UP][row];
      auto &down_bmps = bmps_set_[DOWN][this->rows() - row - 1];
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

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::GrowFullBTen2(const BTenPOSITION post, const size_t slice_num1,
                                                   const size_t remain_sites, bool init) {
  if (init) {
    InitBTen2(post, slice_num1);
  }
  size_t ctrct_mpo_start_idx = (size_t(post) + 3) % 4;
  BMPSPOSITION pre_post = BMPSPOSITION(ctrct_mpo_start_idx);
  BMPSPOSITION next_post = BMPSPOSITION((size_t(post) + 1) % 4);
  size_t start_idx = bten_set2_[post].size() - 1;
  std::vector<Tensor> &btens = bten_set2_[post];
  TransferMPO mpo1, mpo2;
  size_t N, end_idx;
  BMPST *bmps_pre, *bmps_post;
  std::vector<size_t> mpo_ten_transpose_axes;
  switch (post) {
    case DOWN: {
      /*
       *            0         1       2         3
       *            |         |       |         |
       *            |         |       |         |
       * BTEN2-DOWN ++=========================++
       */
      const size_t col1 = slice_num1;
      const size_t col2 = slice_num1 + 1;
      mpo1 = this->get_col(col1);
      mpo2 = this->get_col(col2);
      std::reverse(mpo1.begin(), mpo1.end());
      std::reverse(mpo2.begin(), mpo2.end());
      N = mpo1.size(); // this->rows();
      bmps_pre = &bmps_set_[pre_post][col1];
      bmps_post = &bmps_set_[next_post][this->cols() - 1 - col2];
      break;
    }
    case RIGHT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      mpo1 = this->get_row(row2);
      mpo2 = this->get_row(row1);
      std::reverse(mpo1.begin(), mpo1.end());
      std::reverse(mpo2.begin(), mpo2.end());
      N = mpo1.size(); // this->cols()
      const size_t mps1_num = this->rows() - 1 - row2;
      const size_t mps2_num = row1;
      bmps_pre = &bmps_set_[pre_post][mps1_num];
      bmps_post = &bmps_set_[next_post][mps2_num];
      break;
    }
    case UP: {
      /*
       *        MPS-LEFT                  MPS-RIGHT
       * BTEN-UP   ++=========================++
       *           |         |       |         |
       *           |         |       |         |
       */
      const size_t col1 = slice_num1;
      const size_t col2 = slice_num1 + 1;
      mpo1 = this->get_col(col2);
      mpo2 = this->get_col(col1);
      N = mpo1.size(); // this->rows();
      bmps_pre = &bmps_set_[RIGHT][this->cols() - 1 - col2];
      bmps_post = &bmps_set_[LEFT][col1];
      break;
    }
    case LEFT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      mpo1 = this->get_row(row1);
      mpo2 = this->get_row(row2);
      N = mpo1.size(); // this->cols()
      const size_t mps1_num = row1;
      const size_t mps2_num = this->rows() - 1 - row2;
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
void TensorNetwork2D<TenElemT, QNT>::BTenMoveStep(const BTenPOSITION position) {
  bten_set_[position].pop_back();
  GrowBTenStep(Opposite(position));
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::BTen2MoveStep(const BTenPOSITION position, const size_t slice_num1) {
  bten_set2_[position].pop_back();
  GrowBTen2Step_(Opposite(position), slice_num1);
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

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::GrowBTenStep(const BTenPOSITION post) {
  size_t ctrct_mpo_start_idx = (size_t(post) + 3) % 4;
  BMPSPOSITION pre_post = BMPSPOSITION(ctrct_mpo_start_idx);
  BMPSPOSITION next_post = BMPSPOSITION((size_t(post) + 1) % 4);
#ifndef NDEBUG
  assert(bmps_set_.at(pre_post).size() + bmps_set_.at(next_post).size() == this->length(Orientation(pre_post)) + 1);
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
      N = this->rows();
      grown_site = {N - bten_size, col};
      break;
    }
    case UP: {
      const size_t col = bmps_set_[LEFT].size() - 1;
      N = this->rows();
      grown_site = {bten_size - 1, col};
      break;
    }
    case LEFT: {
      const size_t row = bmps_set_[UP].size() - 1;
      N = this->cols();
      grown_site = {row, bten_size - 1};
      break;
    }
    case RIGHT: {
      const size_t row = bmps_set_[UP].size() - 1;
      N = this->cols();
      grown_site = {row, N - bten_size};
      break;
    }
  }
  mps_ten1 = &bmps_set_[pre_post].back()[N - bten_size];
  mps_ten2 = &bmps_set_[next_post].back()[bten_size - 1];
  if constexpr (Tensor::IsFermionic()) {
    next_bten =
        FermionGrowBTenStep(post, bten_set_.at(post).back(),
                            *mps_ten1, (*this)(grown_site), *mps_ten2);
  } else {
    Contract<TenElemT, QNT, true, true>(*mps_ten1, bten_set_.at(post).back(), 2, 0, 1, tmp1);
    Contract<TenElemT, QNT, false, false>(tmp1, (*this)(grown_site), 1, ctrct_mpo_start_idx, 2, tmp2);
    Contract(&tmp2, {0, 2}, mps_ten2, {0, 1}, &next_bten);
  }
  bten_set_[post].emplace_back(next_bten);
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
      /*
       *            |         |          |          |
       *            |      (gr_site1) (gr_site2)    |
       * TN ROW  mps_ten1---mpo_ten1---mpo_ten2---mps_ten2
       *            |         |          |          |
       *            0         1          2          3
       *            |         |          |          |
       * BTEN2-DOWN ++=============================++
      */
      const size_t col = slice_num1;
      N = rows;
      grown_site1 = {N - bten_size, col};
      grown_site2 = {N - bten_size, col + 1};
      mps1_idx = col;
      mps2_idx = cols - 1 - (col + 1);
      break;
    }
    case UP: {
      /*
       *        MPS-LEFT                      MPS-RIGHT
       * BTEN2-UP    ++=============================++
       *             |         |          |          |
       *             3         2          1          0 (indices order of bten2-up)
       *             |         |          |          |
       * TN ROW   mps_ten2---mpo_ten2----mpo_ten1---mps_ten1
       *             |      (gr_site2) (gr_site1)    |
       *             |         |          |          |
      */
      const size_t col = slice_num1;
      N = rows;
      grown_site1 = {bten_size - 1, col + 1};
      grown_site2 = {bten_size - 1, col};
      mps1_idx = cols - 1 - (col + 1);
      mps2_idx = col;
      break;
    }
    case LEFT: {
      /*
        *       BTEN2-LEFT
        * MPS UP    ++-----mps_ten1--
        *           ||        |
        *           ||        |
        * TN ROW1   ||------mpo_ten1(grown_site1)---
        *           ||        |
        *           ||        |
        * TN ROW2   ||------mpo_ten2(grown_site2)---
        *           ||        |
        *           ||        |
        * MPS DOWN  ++-----mps_ten2--
        *
       */
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

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::GrowBTen2Step_(const BTenPOSITION post, const size_t slice_num1) {
  size_t ctrct_mpo_start_idx = (size_t(post) + 3) % 4;
  BMPSPOSITION pre_post = BMPSPOSITION(ctrct_mpo_start_idx);
  BMPSPOSITION next_post = BMPSPOSITION((size_t(post) + 1) % 4);
  const size_t bten_size = bten_set2_.at(post).size();
#ifndef NDEBUG
  assert(bmps_set_.at(pre_post).size() + bmps_set_.at(next_post).size() >= this->length(Orientation(pre_post)));
  assert(bten_set2_.at(post).size() > 0 &&
      bten_set2_.at(post).size() <= bmps_set_.at(pre_post).back().size()); // has been initialled
#endif
  SiteIdx grown_site1, grown_site2;
  size_t N; //mps length
  size_t mps1_idx, mps2_idx;
  std::vector<size_t> mpo1_transpose_axes;

  SetUpCoordInfoForGrowBTen2<QNT>(post, slice_num1, bten_size, this->rows(), this->cols(),
                                  N, grown_site1, grown_site2, mps1_idx, mps2_idx, mpo1_transpose_axes);

  auto mpo_ten1 = (*this)(grown_site1);
  mpo_ten1.Transpose(mpo1_transpose_axes);
  const auto &mps_ten1 = bmps_set_.at(pre_post)[mps1_idx][N - bten_size];
  const auto &mps_ten2 = bmps_set_.at(next_post)[mps2_idx][bten_size - 1];
  Tensor *mpo_ten2;
  if constexpr (Tensor::IsFermionic()) {
    mpo_ten2 = new Tensor((*this)(grown_site2));
    if (post != DOWN) {
      auto mpo2_trans_axes = GenMpoTen2TransposeAxesForFermionGrowBTen2(post);
      mpo_ten2->Transpose(mpo2_trans_axes);
    }
  } else {
    mpo_ten2 = &(*this)(grown_site2);
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
}//qlpeps
#endif //QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BTEN_OPERATION_H
