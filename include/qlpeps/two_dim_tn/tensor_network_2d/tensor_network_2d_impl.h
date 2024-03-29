// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-07
*
* Description: QuantumLiquids/PEPS project. The 2-dimensional tensor network class, implementation.
*/


#ifndef QLPEPS_VMC_PEPS_TENSOR_NETWORK_2D_IMPL_H
#define QLPEPS_VMC_PEPS_TENSOR_NETWORK_2D_IMPL_H

namespace qlpeps {
using namespace qlten;

template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>::TensorNetwork2D(const size_t rows, const size_t cols)
    : TenMatrix<QLTensor<TenElemT, QNT>>(rows, cols) {
  for (size_t post_int = 0; post_int < 4; post_int++) {
    const BMPSPOSITION post = static_cast<BMPSPOSITION>(post_int);
    bmps_set_.insert(std::make_pair(post, std::vector<BMPS<TenElemT, QNT>>()));
    const size_t mps_max_num = this->length(Orientation(post));
    bmps_set_[post].reserve(mps_max_num);

    bten_set_.insert(std::make_pair(static_cast<BTenPOSITION>(post_int), std::vector<Tensor>()));
  }
}

template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>::TensorNetwork2D(const SplitIndexTPS<TenElemT, QNT> &tps, const Configuration &config)
    :TensorNetwork2D(tps.rows(), tps.cols()) {
  for (size_t row = 0; row < tps.rows(); row++) {
    for (size_t col = 0; col < tps.cols(); col++) {
      (*this)({row, col}) = tps({row, col})[config({row, col})];
    }
  }

  //generate the boundary bmps_set_[position][0]
  InitBMPS();
}

template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT> &TensorNetwork2D<TenElemT, QNT>::operator=(const TensorNetwork2D<TenElemT, QNT> &tn) {
  TenMatrix<Tensor>::operator=(tn);
  //Question : directly set bmps_set_ = tn.bmps_set_ induce bug, the position in map is inconsistent with the position
  // inside the bmps. Why?
  for (BMPSPOSITION post : {LEFT, DOWN, RIGHT, UP}) {
    bmps_set_[post] = tn.bmps_set_.at(post);
  }
  bten_set_ = tn.bten_set_;
  return *this;
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::InitBMPS(void) {
  for (size_t post_int = 0; post_int < 4; post_int++) {
    InitBMPS((BMPSPOSITION) post_int);
  }
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::InitBMPS(const qlpeps::BMPSPOSITION post) {
  SiteIdx refer_site;
  if (post < 2) {
    refer_site = {this->rows() - 1, 0}; // left,down
  } else {
    refer_site = {0, this->cols() - 1}; //right, up
  }
  IndexT idx = InverseIndex((*this)(refer_site).GetIndex(post));
  const size_t mps_size = this->length(Rotate(Orientation(post)));
  BMPS<TenElemT, QNT> boundary_bmps(post, mps_size, idx);
  bmps_set_[post].push_back(boundary_bmps);
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
TensorNetwork2D<TenElemT, QNT>::GenerateBMPSApproach(BMPSPOSITION post, const BMPSTruncatePara &trunc_para) {
  DeleteInnerBMPS(post);
  GrowFullBMPS(Opposite(post), trunc_para);
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
size_t TensorNetwork2D<TenElemT, QNT>::GrowBMPSStep_(const BMPSPOSITION position,
                                                     TransferMPO mpo,
                                                     const BMPSTruncatePara &trunc_para) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  bmps_set.push_back(
      bmps_set.back().MultipleMPO(mpo, trunc_para.compress_scheme,
                                  trunc_para.D_min, trunc_para.D_max, trunc_para.trunc_err,
                                  trunc_para.convergence_tol,
                                  trunc_para.iter_max));
  return bmps_set.size();
}

template<typename TenElemT, typename QNT>
size_t TensorNetwork2D<TenElemT, QNT>::GrowBMPSStep_(const BMPSPOSITION position, const BMPSTruncatePara &trunc_para) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  size_t existed_bmps_num = bmps_set.size();
  assert(existed_bmps_num > 0);
  size_t mpo_num;
  if (position == UP || position == LEFT) {
    mpo_num = existed_bmps_num - 1;
  } else if (position == DOWN) {
    mpo_num = this->rows() - existed_bmps_num;
  } else { //RIGHT
    mpo_num = this->cols() - existed_bmps_num;
  }
  const TransferMPO &mpo = this->get_slice(mpo_num, Rotate(Orientation(position)));
  return GrowBMPSStep_(position, mpo, trunc_para);
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::GrowFullBMPS(const BMPSPOSITION position, const BMPSTruncatePara &trunc_para) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  size_t existed_bmps_size = bmps_set.size();
  assert(existed_bmps_size > 0);
  size_t rows = this->rows();
  size_t cols = this->cols();
  switch (position) {
    case DOWN: {
      for (size_t row = rows - existed_bmps_size; row > 0; row--) {
        const TransferMPO &mpo = this->get_row(row);
        GrowBMPSStep_(position, mpo, trunc_para);
      }
      break;
    }
    case UP: {
      for (size_t row = existed_bmps_size - 1; row < rows - 1; row++) {
        const TransferMPO &mpo = this->get_row(row);
        GrowBMPSStep_(position, mpo, trunc_para);
      }
      break;
    }
    case LEFT: {
      for (size_t col = existed_bmps_size - 1; col < cols - 1; col++) {
        const TransferMPO &mpo = this->get_col(col);
        GrowBMPSStep_(position, mpo, trunc_para);
      }
    }
    case RIGHT: {
      for (size_t col = cols - existed_bmps_size; col > 0; col--) {
        const TransferMPO &mpo = this->get_col(col);
        GrowBMPSStep_(position, mpo, trunc_para);
      }
      break;
    }
  }
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
TensorNetwork2D<TenElemT, QNT>::GrowBMPSForRow(const size_t row, const BMPSTruncatePara &trunc_para) {
  const size_t rows = this->rows();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set_down = bmps_set_[DOWN];
  for (size_t row_bmps = rows - bmps_set_down.size(); row_bmps > row; row_bmps--) {
    const TransferMPO &mpo = this->get_row(row_bmps);
    GrowBMPSStep_(DOWN, mpo, trunc_para);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_up = bmps_set_[UP];
  for (size_t row_bmps = bmps_set_up.size() - 1; row_bmps < row; row_bmps++) {
    const TransferMPO &mpo = this->get_row(row_bmps);
    GrowBMPSStep_(UP, mpo, trunc_para);
  }
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
TensorNetwork2D<TenElemT, QNT>::GrowBMPSForCol(const size_t col, const BMPSTruncatePara &trunc_para) {
  const size_t cols = this->cols();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set_right = bmps_set_[RIGHT];
  for (size_t col_bmps = cols - bmps_set_right.size(); col_bmps > col; col_bmps--) {
    const TransferMPO &mpo = this->get_col(col_bmps);
    GrowBMPSStep_(RIGHT, mpo, trunc_para);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_left = bmps_set_[LEFT];
  for (size_t col_bmps = bmps_set_left.size() - 1; col_bmps < col; col_bmps++) {
    const TransferMPO &mpo = this->get_col(col_bmps);
    GrowBMPSStep_(LEFT, mpo, trunc_para);
  }
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
const std::pair<BMPS<TenElemT, QNT>, BMPS<TenElemT, QNT> >
TensorNetwork2D<TenElemT, QNT>::GetBMPSForRow(const size_t row, const BMPSTruncatePara &trunc_para) {
  const size_t rows = this->rows();
  GrowBMPSForRow(row);
  BMPST &up_bmps = bmps_set_[UP][row];
  BMPST &down_bmps = bmps_set_[DOWN][rows - 1 - row];

  return std::pair(up_bmps, down_bmps);
}

template<typename TenElemT, typename QNT>
const std::pair<BMPS<TenElemT, QNT>, BMPS<TenElemT, QNT> >
TensorNetwork2D<TenElemT, QNT>::GetBMPSForCol(const size_t col, const BMPSTruncatePara &trunc_para) {
  const size_t cols = this->cols();
  GrowBMPSForCol(col);
  BMPST &left_bmps = bmps_set_[LEFT][col];
  BMPST &right_bmps = bmps_set_[RIGHT][cols - 1 - col];
  return std::pair(left_bmps, right_bmps);
}

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
  Tensor ten({index0, index1, index2});
  ten({0, 0, 0}) = TenElemT(1.0);
  bten_set_[position].emplace_back(ten);
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::TruncateBTen(const qlpeps::BTenPOSITION position, const size_t length) {
  auto &btens = bten_set_.at(position);
  if (btens.size() > length) {
    btens.resize(length);
  }
}
///< slice_num1 is the small row/col value.
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
  Tensor ten({index0, index1, index2, index3});
  ten({0, 0, 0, 0}) = TenElemT(1.0);
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
        Contract<TenElemT, QNT, true, true>(left_mps_ten, btens.back(), 2, 0, 1, tmp1);
        Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 0, 2, tmp2);
        Contract(&tmp2, {0, 2}, &right_mps_ten, {0, 1}, &tmp3);
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
        Contract<TenElemT, QNT, true, true>(right_mps_ten, btens.back(), 2, 0, 1, tmp1);
        Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 2, 2, tmp2);
        Contract(&tmp2, {0, 2}, &left_mps_ten, {0, 1}, &tmp3);
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
        auto &mpo_ten = *mpo[i];
        Tensor tmp1, tmp2, tmp3;
        Contract<TenElemT, QNT, true, true>(up_mps_ten, btens.back(), 2, 0, 1, tmp1);
        Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 3, 2, tmp2);
        Contract(&tmp2, {0, 2}, &down_mps_ten, {0, 1}, &tmp3);
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
        Contract<TenElemT, QNT, true, true>(down_mps_ten, btens.back(), 2, 0, 1, tmp1);
        Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, 1, 2, tmp2);
        Contract(&tmp2, {0, 2}, &up_mps_ten, {0, 1}, &tmp3);
        btens.emplace_back(tmp3);
      }
      break;
    }
  }
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
      const TransferMPO &mpo1 = this->get_col(col1);
      const TransferMPO &mpo2 = this->get_col(col2);
      const size_t N = mpo1.size(); // this->rows();
      const size_t end_idx = N - remain_sites;
      auto &left_bmps = bmps_set_[pre_post][col1];
      auto &right_bmps = bmps_set_[next_post][this->cols() - 1 - col2];

      for (size_t i = start_idx; i < end_idx; i++) {
        auto &mps_ten1 = left_bmps[N - i - 1];
        auto &mps_ten2 = right_bmps[i];
        Tensor mpo_ten1 = *mpo1[N - i - 1];
        mpo_ten1.Transpose({0, 1, 3, 2});
        auto &mpo_ten2 = *mpo2[N - i - 1];

        Tensor tmp1, tmp2, tmp3, next_bten;
        Contract<TenElemT, QNT, true, true>(mps_ten1, btens.back(), 2, 0, 1, tmp1);
        Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
        Contract<TenElemT, QNT, false, false>(tmp2, mpo_ten2, 4, ctrct_mpo_start_idx, 2, tmp3);
        Contract(&tmp3, {0, 3}, &mps_ten2, {0, 1}, &next_bten);
        btens.emplace_back(next_bten);
      }
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
      const TransferMPO &mpo1 = this->get_col(col2);
      const TransferMPO &mpo2 = this->get_col(col1);
      const size_t N = mpo1.size(); // this->rows();
      const size_t end_idx = N - remain_sites;
      auto &right_bmps = bmps_set_[RIGHT][this->cols() - 1 - col2];
      auto &left_bmps = bmps_set_[LEFT][col1];
      for (size_t i = start_idx; i < end_idx; i++) {
        auto &mps_ten1 = right_bmps[N - i - 1];
        auto &mps_ten2 = left_bmps[i];

        Tensor mpo_ten1 = *mpo1[i];
        mpo_ten1.Transpose({2, 3, 1, 0});
        auto &mpo_ten2 = *mpo2[i];
        Tensor tmp1, tmp2, tmp3, next_bten;
        Contract<TenElemT, QNT, true, true>(mps_ten1, btens.back(), 2, 0, 1, tmp1);
        Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
        Contract<TenElemT, QNT, false, false>(tmp2, mpo_ten2, 4, ctrct_mpo_start_idx, 2, tmp3);
        Contract(&tmp3, {0, 3}, &mps_ten2, {0, 1}, &next_bten);
        btens.emplace_back(next_bten);
      }
      break;
    }
    case LEFT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      const TransferMPO &mpo1 = this->get_row(row1);
      const TransferMPO &mpo2 = this->get_row(row2);
      const size_t N = mpo1.size(); // this->cols()
      const size_t end_idx = N - remain_sites;
      const size_t mps1_num = row1;
      const size_t mps2_num = this->rows() - 1 - row2;
      auto &up_bmps = bmps_set_[pre_post][mps1_num];
      auto &down_bmps = bmps_set_[next_post][mps2_num];
      for (size_t i = start_idx; i < end_idx; i++) {
        auto &mps_ten1 = up_bmps[N - i - 1];
        auto &mps_ten2 = down_bmps[i];
        Tensor mpo_ten1 = *mpo1[i];
        mpo_ten1.Transpose({3, 0, 2, 1});
        auto &mpo_ten2 = *mpo2[i];
        Tensor tmp1, tmp2, tmp3, next_bten;
        Contract<TenElemT, QNT, true, true>(mps_ten1, btens.back(), 2, 0, 1, tmp1);
        Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 1, 0, 2, tmp2); // O(D^7) complexity
        Contract<TenElemT, QNT, false, false>(tmp2, mpo_ten2, 4, ctrct_mpo_start_idx, 2, tmp3);
        Contract(&tmp3, {0, 3}, &mps_ten2, {0, 1}, &next_bten);
        btens.emplace_back(next_bten);
      }
      break;
    }
    case RIGHT: {
      const size_t row1 = slice_num1;
      const size_t row2 = row1 + 1;
      const TransferMPO &mpo1 = this->get_row(row2);
      const TransferMPO &mpo2 = this->get_row(row1);
      const size_t N = mpo1.size(); // this->cols()
      const size_t end_idx = N - remain_sites;
      const size_t mps1_num = this->rows() - 1 - row2;
      const size_t mps2_num = row1;
      auto &down_bmps = bmps_set_[pre_post][mps1_num];
      auto &up_bmps = bmps_set_[next_post][mps2_num];
      for (size_t i = start_idx; i < end_idx; i++) {
        auto &mps_ten1 = down_bmps[N - i - 1];
        auto &mps_ten2 = up_bmps[i];
        Tensor mpo_ten1 = *mpo1[N - i - 1];
        mpo_ten1.Transpose({1, 2, 0, 3});
        auto &mpo_ten2 = *mpo2[N - i - 1];
        Tensor tmp1, tmp2, tmp3, next_bten;
        Contract<TenElemT, QNT, true, true>(mps_ten1, btens.back(), 2, 0, 1, tmp1);
        Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
        Contract<TenElemT, QNT, false, false>(tmp2, mpo_ten2, 4, ctrct_mpo_start_idx, 2, tmp3);
        Contract(&tmp3, {0, 3}, &mps_ten2, {0, 1}, &next_bten);
        btens.emplace_back(next_bten);
      }
      break;
    }
  }
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::BMPSMoveStep(const BMPSPOSITION position, const BMPSTruncatePara &trunc_para) {
  bmps_set_[position].pop_back();
  auto oppo_post = Opposite(position);
  GrowBMPSStep_(oppo_post, trunc_para);
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
      grown_site[0] = N - bten_size;
      grown_site[1] = col;
      break;
    }
    case UP: {
      const size_t col = bmps_set_[LEFT].size() - 1;
      N = this->rows();
      grown_site[0] = bten_size - 1;
      grown_site[1] = col;
      break;
    }
    case LEFT: {
      const size_t row = bmps_set_[UP].size() - 1;
      N = this->cols();
      grown_site[0] = row;
      grown_site[1] = bten_size - 1;
      break;
    }
    case RIGHT: {
      const size_t row = bmps_set_[UP].size() - 1;
      N = this->cols();
      grown_site[0] = row;
      grown_site[1] = N - bten_size;
      break;
    }
  }
  mps_ten1 = &bmps_set_[pre_post].back()[N - bten_size];
  mps_ten2 = &bmps_set_[next_post].back()[bten_size - 1];
  Contract<TenElemT, QNT, true, true>(*mps_ten1, bten_set_.at(post).back(), 2, 0, 1, tmp1);
  Contract<TenElemT, QNT, false, false>(tmp1, (*this)(grown_site), 1, ctrct_mpo_start_idx, 2, tmp2);
  Contract(&tmp2, {0, 2}, mps_ten2, {0, 1}, &next_bten);
  bten_set_[post].emplace_back(next_bten);
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::GrowBTen2Step_(const BTenPOSITION post, const size_t slice_num1) {
  size_t ctrct_mpo_start_idx = (size_t(post) + 3) % 4;
  BMPSPOSITION pre_post = BMPSPOSITION(ctrct_mpo_start_idx);
  BMPSPOSITION next_post = BMPSPOSITION((size_t(post) + 1) % 4);
#ifndef NDEBUG
  assert(bmps_set_.at(pre_post).size() + bmps_set_.at(next_post).size() >= this->length(Orientation(pre_post)));
  assert(bten_set2_.at(post).size() > 0 &&
      bten_set2_.at(post).size() <= bmps_set_.at(pre_post).back().size()); // has been initialled
#endif
  Tensor tmp1, tmp2, tmp3, next_bten;
  Tensor *mps_ten1, *mpo_ten2, *mps_ten2;
  Tensor mpo_ten1;
  SiteIdx grown_site1, grown_site2;
  size_t N; //mps length
  const size_t bten_size = bten_set2_.at(post).size();
  size_t mps1_num, mps2_num;
  switch (post) {
    case DOWN: {
      /*
       *            0         1       2         3
       *            |         |       |         |
       *            |         |       |         |
       * BTEN2-DOWN ++=========================++
      */
      const size_t col = slice_num1;
      N = this->rows();
      grown_site1[0] = N - bten_size;
      grown_site1[1] = col;
      grown_site2[0] = N - bten_size;
      grown_site2[1] = col + 1;
      mps1_num = col;
      mps2_num = this->cols() - 1 - (col + 1);
      mpo_ten1 = (*this)(grown_site1);
      mpo_ten1.Transpose({0, 1, 3, 2});
      break;
    }
    case UP: {
      /*
       *        MPS-LEFT                  MPS-RIGHT
       * BTEN-UP     ++=========================++
       *             |         |       |         |
       *             |         |       |         |
       * TN ROW1  mps_ten2--gr_site2--mpo_ten1---mps_ten1
       *             |         |       |         |
       *             |         |       |         |
      */
      const size_t col = slice_num1;
      N = this->rows();
      grown_site1 = {bten_size - 1, col + 1};
      grown_site2 = {bten_size - 1, col};
      mps1_num = this->cols() - 1 - (col + 1);
      mps2_num = col;
      mpo_ten1 = (*this)({bten_size - 1, col + 1});
      mpo_ten1.Transpose({2, 3, 1, 0});
      break;
    }
    case LEFT: {
      const size_t row = slice_num1;
      N = this->cols();
      grown_site1[0] = row;
      grown_site1[1] = bten_size - 1;
      grown_site2[0] = row + 1;
      grown_site2[1] = bten_size - 1;
      mps1_num = row;
      mps2_num = this->rows() - 1 - (row + 1);
      mpo_ten1 = (*this)(grown_site1);
      mpo_ten1.Transpose({3, 0, 2, 1});
      break;
    }
    case RIGHT: {
      const size_t row = slice_num1;
      N = this->cols();
      grown_site1[0] = row;
      grown_site1[1] = N - bten_size;
      grown_site2[0] = row + 1;
      grown_site2[1] = N - bten_size;
      mps1_num = this->rows() - 1 - (row + 1);
      mps2_num = row;
      mpo_ten1 = (*this)(grown_site1);
      mpo_ten1.Transpose({1, 2, 0, 3});
      break;
    }
  }
  mps_ten1 = &bmps_set_.at(pre_post)[mps1_num][N - bten_size];
  mps_ten2 = &bmps_set_.at(next_post)[mps2_num][bten_size - 1];
  Contract<TenElemT, QNT, true, true>(*mps_ten1, bten_set2_.at(post).back(), 2, 0, 1, tmp1);
  Contract<TenElemT, QNT, false, true>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
  Contract<TenElemT, QNT, false, false>(tmp2, (*this)(grown_site2), 4, ctrct_mpo_start_idx, 2, tmp3);
  Contract(&tmp3, {0, 3}, mps_ten2, {0, 1}, &next_bten);
  bten_set2_[post].emplace_back(next_bten);
}

template<typename TenElemT, typename QNT>
TenElemT TensorNetwork2D<TenElemT, QNT>::Trace(const SiteIdx &site_a, const BondOrientation bond_dir) const {
  SiteIdx site_b(site_a);
  if (bond_dir == HORIZONTAL) {
    site_b[1] += 1;
  } else {
    site_b[0] += 1;
  }
  return Trace(site_a, site_b, bond_dir);
}

template<typename TenElemT, typename QNT>
TenElemT TensorNetwork2D<TenElemT, QNT>::Trace(const SiteIdx &site_a, const SiteIdx &site_b,
                                               const BondOrientation bond_dir) const {
  const Tensor &ten_a = (*this)(site_a);
  const Tensor &ten_b = (*this)(site_b);
  return ReplaceNNSiteTrace(site_a, site_b, bond_dir, ten_a, ten_b);
}

template<typename TenElemT, typename QNT>
TenElemT TensorNetwork2D<TenElemT, QNT>::ReplaceOneSiteTrace(const SiteIdx &site,
                                                             const TensorNetwork2D::Tensor &replace_ten,
                                                             const BondOrientation mps_orient) const {
  //here suppose mps along the horizontal direction.
  Tensor tmp[4];
  const size_t row = site[0];
  const size_t col = site[1];
  if (mps_orient == HORIZONTAL) {
#ifndef NDEBUG
    assert(bmps_set_.at(UP).size() > row);
    assert(bmps_set_.at(DOWN).size() + 1 > this->rows() - row);
    assert(bten_set_.at(LEFT).size() > col);
    assert(bten_set_.at(RIGHT).size() + 1 > this->cols() - col);
#endif
    const Tensor &up_mps_ten = bmps_set_.at(UP)[row][this->cols() - col - 1];
    const Tensor &down_mps_ten = bmps_set_.at(DOWN)[this->rows() - row - 1][col];
    Contract<TenElemT, QNT, true, true>(up_mps_ten, bten_set_.at(LEFT)[col], 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, false>(tmp[0], replace_ten, 1, 3, 2, tmp[1]);
    Contract(&tmp[1], {0, 2}, &down_mps_ten, {0, 1}, &tmp[2]);
    Contract(&tmp[2], {0, 1, 2}, &bten_set_.at(RIGHT)[this->cols() - col - 1], {2, 1, 0}, &tmp[3]);
  } else {
    const Tensor &left_mps_ten_a = bmps_set_.at(LEFT)[col][row];
    const Tensor &right_mps_ten_a = bmps_set_.at(RIGHT)[this->cols() - col - 1][this->rows() - row - 1];
    Contract<TenElemT, QNT, true, true>(right_mps_ten_a, bten_set_.at(UP)[row], 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, false>(tmp[0], replace_ten, 1, 2, 2, tmp[1]);
    Contract(&tmp[1], {0, 2}, &left_mps_ten_a, {0, 1}, &tmp[2]);
    Contract(&tmp[2], {0, 1, 2}, &bten_set_.at(DOWN)[this->rows() - row - 1], {2, 1, 0}, &tmp[3]);
  }
  return tmp[3]();
}

template<typename TenElemT, typename QNT>
TenElemT
TensorNetwork2D<TenElemT, QNT>::ReplaceNNSiteTrace(const SiteIdx &site_a, const SiteIdx &site_b,
                                                   const BondOrientation bond_dir,
                                                   const Tensor &ten_a,
                                                   const Tensor &ten_b) const {
#ifndef NDEBUG
  if (bond_dir == HORIZONTAL) {
    assert(site_a[0] == site_b[0]);
    assert(site_a[1] + 1 == site_b[1]);
    size_t bond_row = site_a[0];
    assert(bmps_set_.at(UP).size() > bond_row);
    assert(bmps_set_.at(DOWN).size() + 1 > this->rows() - bond_row);
    assert(bten_set_.at(LEFT).size() > site_a[1]);
    assert(bten_set_.at(RIGHT).size() + 1 > this->cols() - site_b[1]);
  } else {
    assert(site_a[0] + 1 == site_b[0]);
    assert(site_a[1] == site_b[1]);
    size_t bond_col = site_a[1];
    assert(bmps_set_.at(LEFT).size() > bond_col);
    assert(bmps_set_.at(RIGHT).size() + 1 > this->cols() - bond_col);
    assert(bten_set_.at(UP).size() > site_a[0]);
    assert(bten_set_.at(DOWN).size() + 1 > this->rows() - site_b[0]);
  }
#endif
  Tensor tmp[7];
  if (bond_dir == HORIZONTAL) {
    /*
     *        BTEN-LEFT             BTEN-RIGHT
     * MPS UP    ++-------+------+-------++
     *           ||       |      |       ||
     *           ||       |      |       ||
     * TN ROW    ||----site_a--site_b----||
     *           ||       |      |       ||
     *           ||       |      |       ||
     * MPS DOWN  ++-------+------+-------++
     *
    */
    const size_t row = site_a[0];
    const size_t col_a = site_a[1];

    const Tensor &up_mps_ten_a = bmps_set_.at(UP)[row][this->cols() - col_a - 1];
    const Tensor &down_mps_ten_a = bmps_set_.at(DOWN)[this->rows() - row - 1][col_a];

    Contract<TenElemT, QNT, true, true>(up_mps_ten_a, bten_set_.at(LEFT)[col_a], 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, false>(tmp[0], ten_a, 1, 3, 2, tmp[1]);
    Contract(&tmp[1], {0, 2}, &down_mps_ten_a, {0, 1}, &tmp[2]);

    size_t col_b = site_b[1];
    const Tensor up_mps_ten_b = bmps_set_.at(UP)[row][this->cols() - col_b - 1];
    const Tensor down_mps_ten_b = bmps_set_.at(DOWN)[this->rows() - row - 1][col_b];

    Contract<TenElemT, QNT, true, true>(down_mps_ten_b, bten_set_.at(RIGHT)[this->cols() - col_b - 1], 2, 0, 1, tmp[3]);
    Contract<TenElemT, QNT, false, false>(tmp[3], ten_b, 1, 1, 2, tmp[4]);
    Contract(&tmp[4], {0, 2}, &up_mps_ten_b, {0, 1}, &tmp[5]);
  } else {
    /*
     *        BMPS-LEFT         BMPS-RIGHT
     * BTEN-UP   +-------+------+
     *           |       |      |
     *           |       |      |
     * TN ROW1   +----site_a----+
     *           |       |      |
     *           |       |      |
     * TN ROW2   +----site_b----+
     *           |       |      |
     *           |       |      |
     * BTEN-DOWN +-------+------+
     *
    */
    const size_t col = site_a[1];
    const size_t row_a = site_a[0];

    const Tensor &left_mps_ten_a = bmps_set_.at(LEFT)[col][row_a];
    const Tensor &right_mps_ten_a = bmps_set_.at(RIGHT)[this->cols() - col - 1][this->rows() - row_a - 1];
    Contract<TenElemT, QNT, true, true>(right_mps_ten_a, bten_set_.at(UP)[row_a], 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, false>(tmp[0], ten_a, 1, 2, 2, tmp[1]);
    Contract(&tmp[1], {0, 2}, &left_mps_ten_a, {0, 1}, &tmp[2]);

    const size_t row_b = site_b[0];
    const Tensor &left_mps_ten_b = bmps_set_.at(LEFT)[col][row_b];
    const Tensor &right_mps_ten_b = bmps_set_.at(RIGHT)[this->cols() - col - 1][this->rows() - row_b - 1];
    Contract<TenElemT, QNT, true, true>(left_mps_ten_b, bten_set_.at(DOWN)[this->rows() - row_b - 1], 2, 0, 1, tmp[3]);
    Contract<TenElemT, QNT, false, false>(tmp[3], ten_b, 1, 0, 2, tmp[4]);
    Contract(&tmp[4], {0, 2}, &right_mps_ten_b, {0, 1}, &tmp[5]);
  }
  Contract(&tmp[2], {0, 1, 2}, &tmp[5], {2, 1, 0}, &tmp[6]);
  return tmp[6]();
}

template<typename TenElemT, typename QNT>
TenElemT TensorNetwork2D<TenElemT, QNT>::ReplaceNNNSiteTrace(const SiteIdx &left_up_site,
                                                             const DIAGONAL_DIR nnn_dir,
                                                             const BondOrientation mps_orient,
                                                             const TensorNetwork2D::Tensor &ten_left,
                                                             const TensorNetwork2D::Tensor &ten_right) const {
  const size_t row1 = left_up_site[0];
  const size_t row2 = row1 + 1;
  const size_t col1 = left_up_site[1];
  const size_t col2 = col1 + 1;
  Tensor tmp[9];
  if (mps_orient == HORIZONTAL) {
    /*
     *       BTEN-LEFT                       BTEN-RIGHT
     * MPS UP    ++-----mps_ten1--mps_ten4------++
     *           ||        |         |          ||
     *           ||        |         |          ||
     * TN ROW1   ||------mpo_t0----mpo_t3-------||
     *           ||        |         |          ||
     *           ||        |         |          ||
     * TN ROW2   ||------mpo_t1----mpo_t2-------||
     *           ||        |         |          ||
     *           ||        |         |          ||
     * MPS DOWN  ++-----mps_ten2--mps_ten3------++
     *
    */
#ifndef NDEBUG
    const Tensor &mps_ten1 = bmps_set_.at(UP).at(row1)[this->cols() - col1 - 1];
    const Tensor &mps_ten2 = bmps_set_.at(DOWN).at(this->rows() - 1 - row2)[col1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN).at(this->rows() - 1 - row2)[col2];
    const Tensor &mps_ten4 = bmps_set_.at(UP).at(row1)[this->cols() - col2 - 1];
#else
    const Tensor &mps_ten1 = bmps_set_.at(UP)[row1][this->cols() - col1 - 1];
    const Tensor &mps_ten2 = bmps_set_.at(DOWN)[this->rows() - 1 - row2][col1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN)[this->rows() - 1 - row2][col2];
    const Tensor &mps_ten4 = bmps_set_.at(UP)[row1][this->cols() - col2 - 1];
#endif

    Tensor mpo_ten1, mpo_ten3;
    const Tensor *mpo_ten2, *mpo_ten4;
    const Tensor &left_bten = bten_set2_.at(LEFT)[col1];
    const Tensor &right_bten = bten_set2_.at(RIGHT)[this->cols() - col2 - 1];
    if (nnn_dir == LEFTUP_TO_RIGHTDOWN) {
      mpo_ten1 = ten_left;
      mpo_ten2 = (*this)(row2, col1);
      mpo_ten3 = ten_right;
      mpo_ten4 = (*this)(row1, col2);
    } else { //LEFTDOWN_TO_RIGHTUP
      mpo_ten1 = (*this)({row1, col1});
      mpo_ten2 = &ten_left;
      mpo_ten3 = (*this)({row2, col2});
      mpo_ten4 = &ten_right;
    }
    mpo_ten1.Transpose({3, 0, 2, 1});
    mpo_ten3.Transpose({1, 2, 0, 3});

    Contract<TenElemT, QNT, true, true>(mps_ten1, left_bten, 2, 0, 1, tmp[0]);
    Contract<TenElemT, QNT, false, false>(tmp[0], mpo_ten1, 1, 0, 2, tmp[1]);
    Contract<TenElemT, QNT, false, false>(tmp[1], *mpo_ten2, 4, 3, 2, tmp[2]);
    Contract(&tmp[2], {0, 3}, &mps_ten2, {0, 1}, &tmp[3]);

    Contract<TenElemT, QNT, true, true>(mps_ten3, right_bten, 2, 0, 1, tmp[4]);
    Contract<TenElemT, QNT, false, false>(tmp[4], mpo_ten3, 1, 0, 2, tmp[5]);
    Contract<TenElemT, QNT, false, false>(tmp[5], *mpo_ten4, 4, 1, 2, tmp[6]);
    Contract(&tmp[6], {0, 3}, &mps_ten4, {0, 1}, &tmp[7]);
    Contract(&tmp[3], {0, 1, 2, 3}, &tmp[7], {3, 2, 1, 0}, &tmp[8]);
    return tmp[8]();
  } else { //mps_orient == VERTICAL
    /*
     *        MPS-LEFT                  MPS-RIGHT
     * BTEN-UP   ++=========================++
     *           |         |       |         |
     *           |         |       |         |
     * TN ROW1   mps3----mpo2----mpo3------mps4
     *           |         |       |         |
     *           |         |       |         |
     * TN ROW2   mps1----mpo0----mpo1------mps2
     *           |         |       |         |
     *           |         |       |         |
     * BTEN-DOWN ++=========================++
     *
    */
    Tensor mpo_ten[4];
    Tensor tmp[9];
    const size_t row1 = left_up_site[0];
    const size_t row2 = row1 + 1;
    const size_t col1 = left_up_site[1];
    const size_t col2 = col1 + 1;
#ifndef NDEBUG
    const Tensor &mps_ten1 = bmps_set_.at(LEFT).at(col1)[row2];
    const Tensor &mps_ten2 = bmps_set_.at(RIGHT).at(this->cols() - 1 - col2)[this->rows() - 1 - row2];
    const Tensor &mps_ten3 = bmps_set_.at(LEFT).at(col1)[row1];
    const Tensor &mps_ten4 = bmps_set_.at(RIGHT).at(this->cols() - 1 - col2)[this->rows() - 1 - row1];
#else
    const Tensor &mps_ten1 = bmps_set_.at(LEFT)[col1][row2];
    const Tensor &mps_ten2 = bmps_set_.at(RIGHT)[this->cols() - 1 - col2][this->rows() - 1 - row2];
    const Tensor &mps_ten3 = bmps_set_.at(LEFT)[col1][row1];
    const Tensor &mps_ten4 = bmps_set_.at(RIGHT)[this->cols() - 1 - col2][this->rows() - 1 - row1];
#endif
    const Tensor &top_bten = bten_set2_.at(UP)[row1];
    const Tensor &bottom_bten = bten_set2_.at(DOWN)[this->rows() - row2 - 1];

    if (nnn_dir == LEFTUP_TO_RIGHTDOWN) {
      mpo_ten[0] = (*this)({row2, col1});
      mpo_ten[1] = ten_right;
      mpo_ten[2] = ten_left;
      mpo_ten[3] = (*this)({row1, col2});
    } else { //LEFTDOWN_TO_RIGHTUP
      mpo_ten[0] = ten_left;
      mpo_ten[1] = (*this)({row2, col2});
      mpo_ten[2] = (*this)({row1, col1});
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
TenElemT TensorNetwork2D<TenElemT, QNT>::ReplaceTNNSiteTrace(const SiteIdx &site0,
                                                             const BondOrientation mps_orient,
                                                             const TensorNetwork2D::Tensor &replaced_ten0,
                                                             const TensorNetwork2D::Tensor &replaced_ten1,
                                                             const TensorNetwork2D::Tensor &replaced_ten2) const {
  Tensor tmp[10];
  if (mps_orient == HORIZONTAL) {
    /*
     *       BTEN-LEFT                               BTEN-RIGHT
     * MPS UP    ++-----mps_ten0--mps_ten2--mps_ten4------++
     *           ||        |         |         |          ||
     *           ||        |         |         |          ||
     * TN ROW    ||------mpo_t0----mpo_t1----mpo_t2-------||
     *           ||        |         |         |          ||
     *           ||        |         |         |          ||
     * MPS DOWN  ++-----mps_ten1--mps_ten3--mps_ten5------++
     *
    */
    const size_t row = site0.row();
    const size_t col0 = site0.col();
    const size_t col1 = col0 + 1;
    const size_t col2 = col0 + 2;
    const SiteIdx site1 = {row, col1};
    const SiteIdx site2 = {row, col2};
#ifndef NDEBUG
    const Tensor &mps_ten0 = bmps_set_.at(UP).at(row)[this->cols() - col0 - 1];
    const Tensor &mps_ten1 = bmps_set_.at(DOWN).at(this->rows() - 1 - row)[col0];
    const Tensor &mps_ten2 = bmps_set_.at(UP).at(row)[this->cols() - col1 - 1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN).at(this->rows() - 1 - row)[col1];
    const Tensor &mps_ten4 = bmps_set_.at(UP).at(row)[this->cols() - col2 - 1];
    const Tensor &mps_ten5 = bmps_set_.at(DOWN).at(this->rows() - 1 - row)[col2];
    const Tensor &left_bten = bten_set_.at(LEFT).at(col0);
    const Tensor &right_bten = bten_set_.at(RIGHT).at(this->cols() - col2 - 1);
#else
    const Tensor &mps_ten0 = bmps_set_.at(UP)[row][this->cols() - col0 - 1];
    const Tensor &mps_ten1 = bmps_set_.at(DOWN)[this->rows() - 1 - row][col0];
    const Tensor &mps_ten2 = bmps_set_.at(UP)[row][this->cols() - col1 - 1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN)[this->rows() - 1 - row][col1];
    const Tensor &mps_ten4 = bmps_set_.at(UP)[row][this->cols() - col2 - 1];
    const Tensor &mps_ten5 = bmps_set_.at(DOWN)[this->rows() - 1 - row][col2];
    const Tensor &left_bten = bten_set_.at(LEFT)[col0];
    const Tensor &right_bten = bten_set_.at(RIGHT)[this->cols() - col2 - 1];
#endif

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
  } else {
    /*
     *        MPS-LEFT            MPS-RIGHT
     * BTEN-UP   ++=================++
     *           |         |         |
     *           |         |         |
     * TN ROW1   mps1----mpo0-------mps0
     *           |         |         |
     *           |         |         |
     * TN ROW2   mps3----mpo1-------mps2
     *           |         |         |
     *           |         |         |
     * TN ROW3   mps5----mpo2-------mps4
     *           |         |         |
     *           |         |         |
     * BTEN-DOWN ++=================++
     *
     */
    const size_t col = site0[1];
    const size_t row0 = site0[0];
    const size_t row1 = row0 + 1;
    const size_t row2 = row1 + 1;
    const SiteIdx site1 = {row1, col};
    const SiteIdx site2 = {row2, col};

    const Tensor &mps_ten0 = bmps_set_.at(RIGHT)[this->cols() - 1 - col][this->rows() - 1 - row0];
    const Tensor &mps_ten1 = bmps_set_.at(LEFT)[col][row0];
    const Tensor &mps_ten2 = bmps_set_.at(RIGHT)[this->cols() - 1 - col][this->rows() - 1 - row1];
    const Tensor &mps_ten3 = bmps_set_.at(LEFT)[col][row1];
    const Tensor &mps_ten4 = bmps_set_.at(RIGHT)[this->cols() - 1 - col][this->rows() - 1 - row2];
    const Tensor &mps_ten5 = bmps_set_.at(LEFT)[col][row2];

    const Tensor &top_bten = bten_set_.at(UP)[row0];
    const Tensor &bottom_bten = bten_set_.at(DOWN)[this->rows() - row2 - 1];

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

template<typename TenElemT, typename QNT>
TenElemT TensorNetwork2D<TenElemT, QNT>::ReplaceSqrt5DistTwoSiteTrace(const SiteIdx &left_up_site,
                                                                      const DIAGONAL_DIR sqrt5link_dir,
                                                                      const BondOrientation mps_orient, //mps orientation is the same with longer side orientation
                                                                      const Tensor &ten_left,
                                                                      const Tensor &ten_right) const {
  Tensor mpo_ten[6];
  Tensor tmp[13];
  if (mps_orient == HORIZONTAL) {
    /*
     *       BTEN-LEFT                             BTEN-RIGHT
     * MPS UP    ++---mps_ten1--mps_ten3--mps_ten5----++
     *           ||      |         |         |        ||
     *           ||      |         |         |        ||
     * TN ROW1   ||----mpo_t0----mpo_t2----mpo_t4-----||
     *           ||      |         |         |        ||
     *           ||      |         |         |        ||
     * TN ROW2   ||----mpo_t1----mpo_t3----mpo_t5-----||
     *           ||      |         |         |        ||
     *           ||      |         |         |        ||
     * MPS DOWN  ++---mps_ten2--mps_ten4--mps_ten6----++
     *
     */
    const size_t row1 = left_up_site.row();
    const size_t row2 = row1 + 1;
    const size_t col1 = left_up_site.col();
    const size_t col2 = col1 + 1;
    const size_t col3 = col2 + 1;

    const Tensor &mps_ten1 = bmps_set_.at(UP)[row1][this->cols() - col1 - 1];
    const Tensor &mps_ten2 = bmps_set_.at(DOWN)[this->rows() - 1 - row2][col1];
    const Tensor &mps_ten3 = bmps_set_.at(UP)[row1][this->cols() - col2 - 1];
    const Tensor &mps_ten4 = bmps_set_.at(DOWN)[this->rows() - 1 - row2][col2];
    const Tensor &mps_ten5 = bmps_set_.at(UP)[row1][this->cols() - col3 - 1];
    const Tensor &mps_ten6 = bmps_set_.at(DOWN)[this->rows() - 1 - row2][col3];

    const Tensor &left_bten = bten_set2_.at(LEFT)[col1];
    const Tensor &right_bten = bten_set2_.at(RIGHT)[this->cols() - col3 - 1];

    if (sqrt5link_dir == LEFTUP_TO_RIGHTDOWN) {
      mpo_ten[0] = ten_left;
      mpo_ten[1] = (*this)({row2, col1});
      mpo_ten[4] = (*this)({row1, col3});
      mpo_ten[5] = ten_right;
    } else { //LEFTDOWN_TO_RIGHTUP
      mpo_ten[0] = (*this)({row1, col1});
      mpo_ten[1] = ten_left;
      mpo_ten[4] = ten_right;
      mpo_ten[5] = (*this)({row2, col3});
    }
    mpo_ten[2] = (*this)({row1, col2});
    mpo_ten[3] = (*this)({row2, col2});

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
    /*
     *        MPS-LEFT                  MPS-RIGHT
     * BTEN-UP   ++=========================++
     *           |         |       |         |
     *           |         |       |         |
     * TN ROW1   mps5----mpo4----mpo5------mps6
     *           |         |       |         |
     *           |         |       |         |
     * TN ROW2   mps3----mpo2----mpo3------mps4
     *           |         |       |         |
     *           |         |       |         |
     * TN ROW3   mps1----mpo0----mpo1------mps2
     *           |         |       |         |
     *           |         |       |         |
     * BTEN-DOWN ++=========================++
     *
     */
    const size_t row1 = left_up_site.row();
    const size_t row2 = row1 + 1;
    const size_t row3 = row2 + 1;
    const size_t col1 = left_up_site.col();
    const size_t col2 = col1 + 1;

    const Tensor &mps_ten1 = bmps_set_.at(LEFT)[col1][row3];
    const Tensor &mps_ten2 = bmps_set_.at(RIGHT)[this->cols() - 1 - col2][this->rows() - 1 - row3];
    const Tensor &mps_ten3 = bmps_set_.at(LEFT)[col1][row2];
    const Tensor &mps_ten4 = bmps_set_.at(RIGHT)[this->cols() - 1 - col2][this->rows() - 1 - row2];
    const Tensor &mps_ten5 = bmps_set_.at(LEFT)[col1][row1];
    const Tensor &mps_ten6 = bmps_set_.at(RIGHT)[this->cols() - 1 - col2][this->rows() - 1 - row1];

    const Tensor &top_bten = bten_set2_.at(UP)[row1];
    const Tensor &bottom_bten = bten_set2_.at(DOWN)[this->rows() - row3 - 1];

    mpo_ten[2] = (*this)({row2, col1});
    mpo_ten[3] = (*this)({row2, col2});
    if (sqrt5link_dir == LEFTUP_TO_RIGHTDOWN) {
      mpo_ten[0] = (*this)({row3, col1});
      mpo_ten[1] = ten_right;
      mpo_ten[4] = ten_left;
      mpo_ten[5] = (*this)({row1, col2});
    } else { //LEFTDOWN_TO_RIGHTUP
      mpo_ten[0] = ten_left;
      mpo_ten[1] = (*this)({row3, col2});
      mpo_ten[4] = (*this)({row1, col1});
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
QLTensor<TenElemT, QNT> TensorNetwork2D<TenElemT, QNT>::PunchHole(const qlpeps::SiteIdx &site,
                                                                  const qlpeps::BondOrientation mps_orient) const {
  const Tensor *left_ten, *down_ten, *right_ten, *up_ten;
  const size_t row = site[0];
  const size_t col = site[1];
  if (mps_orient == HORIZONTAL) {
    up_ten = &(bmps_set_.at(UP)[row][this->cols() - col - 1]);
    down_ten = &(bmps_set_.at(DOWN)[this->rows() - row - 1][col]);
    left_ten = &(bten_set_.at(LEFT)[col]);
    right_ten = &(bten_set_.at(RIGHT)[this->cols() - col - 1]);
  } else {
    up_ten = &(bten_set_.at(UP)[row]);
    down_ten = &(bten_set_.at(DOWN)[this->rows() - row - 1]);
    left_ten = &(bmps_set_.at(LEFT)[col][row]);
    right_ten = &(bmps_set_.at(RIGHT)[this->cols() - col - 1][this->rows() - row - 1]);
  }
  Tensor tmp1, tmp2, res_ten;
  Contract(left_ten, {2}, down_ten, {0}, &tmp1);
  Contract(right_ten, {2}, up_ten, {0}, &tmp2);
  Contract(&tmp1, {0, 3}, &tmp2, {3, 0}, &res_ten);
  return res_ten;
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::UpdateSiteConfig(const qlpeps::SiteIdx &site, const size_t update_config,
                                                      const SplitIndexTPS<TenElemT, QNT> &istps, bool check_envs) {
  (*this)(site) = istps(site)[update_config];
  if (check_envs) {
    const size_t row = site[0];
    const size_t col = site[1];
    if (bmps_set_.at(LEFT).size() > col + 1) {
      bmps_set_[LEFT].erase(bmps_set_[LEFT].cbegin() + col + 1, bmps_set_[LEFT].end());
    }

    if (bmps_set_.at(UP).size() > row + 1) {
      bmps_set_[UP].erase(bmps_set_[UP].cbegin() + row + 1, bmps_set_[UP].end());
    }

    size_t down_allow_mps_num = this->rows() - row;
    if (bmps_set_.at(DOWN).size() > down_allow_mps_num) {
      bmps_set_[DOWN].erase(bmps_set_[DOWN].cbegin() + down_allow_mps_num, bmps_set_[DOWN].end());
    }

    size_t right_allow_mps_num = this->cols() - col;
    if (bmps_set_.at(RIGHT).size() > right_allow_mps_num) {
      bmps_set_[RIGHT].erase(bmps_set_[RIGHT].cbegin() + right_allow_mps_num, bmps_set_[RIGHT].end());
    }
  }
}

template<typename TenElemT, typename QNT>
bool TensorNetwork2D<TenElemT, QNT>::DirectionCheck() const {
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
}///qlpeps

#endif //QLPEPS_VMC_PEPS_TENSOR_NETWORK_2D_IMPL_H
