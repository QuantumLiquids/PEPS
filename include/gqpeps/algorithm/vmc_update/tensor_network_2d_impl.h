// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-07
*
* Description: GraceQ/VMC-PEPS project. The 2-dimensional tensor network class, implementation.
*/


#ifndef GRACEQ_VMC_PEPS_TENSOR_NETWORK_2D_IMPL_H
#define GRACEQ_VMC_PEPS_TENSOR_NETWORK_2D_IMPL_H

namespace gqpeps {
using namespace gqten;


template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>::TensorNetwork2D(const size_t rows, const size_t cols)
    :TensorNetwork2D(rows, cols, TruncatePara(1, 1, 0)) {}


template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>::TensorNetwork2D(const size_t rows, const size_t cols, const TruncatePara &trunc_para)
    : TenMatrix<GQTensor<TenElemT, QNT>>(rows, cols), trunc_para_(trunc_para) {
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
    : TensorNetwork2D(tps, config, TruncatePara(1, 1, 0.0)) {}

template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT>::TensorNetwork2D(const SplitIndexTPS<TenElemT, QNT> &tps, const Configuration &config,
                                                const TruncatePara &trunc_para)
    :TensorNetwork2D(tps.rows(), tps.cols(), trunc_para) {
  for (size_t row = 0; row < tps.rows(); row++) {
    for (size_t col = 0; col < tps.cols(); col++) {
      (*this)({row, col}) = tps({row, col})[config({row, col})];
    }
  }

  //generate the boundary bmps_set_[position][0]
  for (size_t post_int = 0; post_int < 4; post_int++) {
    const BMPSPOSITION post = BMPSPOSITION(post_int);
    SiteIdx refer_site;
    if (post_int < 2) {
      refer_site = {this->rows() - 1, 0}; // left,down
    } else {
      refer_site = {0, this->cols() - 1}; //right, up
    }
    IndexT idx = InverseIndex((*this)(refer_site).GetIndex(post));
    const size_t mps_size = this->length(Rotate(Orientation(post)));
    BMPS<TenElemT, QNT> boundary_bmps(post, mps_size, idx);
    bmps_set_[post].push_back(boundary_bmps);
  }
}


template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT> &TensorNetwork2D<TenElemT, QNT>::operator=(const TensorNetwork2D<TenElemT, QNT> &tn) {
  TenMatrix<Tensor>::operator=(tn);
  bmps_set_ = tn.bmps_set_;
  bten_set_ = tn.bten_set_;
  trunc_para_ = tn.trunc_para_;
  return *this;
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
TensorNetwork2D<TenElemT, QNT>::GenerateBMPSApproach(BMPSPOSITION post) {
  DeleteInnerBMPS(post);
  GrowFullBMPS(Opposite(post));
  return bmps_set_;
}


template<typename TenElemT, typename QNT>
size_t TensorNetwork2D<TenElemT, QNT>::GrowBMPSStep_(const BMPSPOSITION position,
                                                     const TransferMPO &mpo) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  bmps_set.push_back(bmps_set.back().MultipleMPO(mpo, trunc_para_.D_min, trunc_para_.D_max, trunc_para_.trunc_err));
  return bmps_set.size();
}


template<typename TenElemT, typename QNT>
size_t TensorNetwork2D<TenElemT, QNT>::GrowBMPSStep_(const BMPSPOSITION position) {
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
  return GrowBMPSStep_(position, mpo);
}


template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::GrowFullBMPS(const BMPSPOSITION position) {
  std::vector<BMPS<TenElemT, QNT>> &bmps_set = bmps_set_[position];
  size_t existed_bmps_size = bmps_set.size();
  assert(existed_bmps_size > 0);
  size_t rows = this->rows();
  size_t cols = this->cols();
  switch (position) {
    case DOWN: {
      for (size_t row = rows - existed_bmps_size; row > 0; row--) {
        const TransferMPO &mpo = this->get_row(row);
        GrowBMPSStep_(position, mpo);
      }
      break;
    }
    case UP: {
      for (size_t row = existed_bmps_size - 1; row < rows - 1; row++) {
        const TransferMPO &mpo = this->get_row(row);
        GrowBMPSStep_(position, mpo);
      }
      break;
    }
    case LEFT: {
      for (size_t col = existed_bmps_size - 1; col < cols - 1; col++) {
        const TransferMPO &mpo = this->get_col(col);
        GrowBMPSStep_(position, mpo);
      }
    }
    case RIGHT: {
      for (size_t col = cols - existed_bmps_size; col > 0; col--) {
        const TransferMPO &mpo = this->get_col(col);
        GrowBMPSStep_(position, mpo);
      }
      break;
    }
  }
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
TensorNetwork2D<TenElemT, QNT>::GrowBMPSForRow(const size_t row) {
  const size_t rows = this->rows();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set_down = bmps_set_[DOWN];
  for (size_t row_bmps = rows - bmps_set_down.size(); row_bmps > row; row_bmps--) {
    const TransferMPO &mpo = this->get_row(row_bmps);
    GrowBMPSStep_(DOWN, mpo);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_up = bmps_set_[UP];
  for (size_t row_bmps = bmps_set_up.size() - 1; row_bmps < row; row_bmps++) {
    const TransferMPO &mpo = this->get_row(row_bmps);
    GrowBMPSStep_(UP, mpo);
  }
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
TensorNetwork2D<TenElemT, QNT>::GrowBMPSForCol(const size_t col) {
  const size_t cols = this->cols();
  std::vector<BMPS<TenElemT, QNT>> &bmps_set_right = bmps_set_[RIGHT];
  for (size_t col_bmps = cols - bmps_set_right.size(); col_bmps > col; col_bmps--) {
    const TransferMPO &mpo = this->get_col(col_bmps);
    GrowBMPSStep_(RIGHT, mpo);
  }

  std::vector<BMPS<TenElemT, QNT>> &bmps_set_left = bmps_set_[LEFT];
  for (size_t col_bmps = bmps_set_left.size() - 1; col_bmps < col; col_bmps++) {
    const TransferMPO &mpo = this->get_col(col_bmps);
    GrowBMPSStep_(LEFT, mpo);
  }
  return bmps_set_;
}

template<typename TenElemT, typename QNT>
const std::pair<BMPS<TenElemT, QNT>, BMPS<TenElemT, QNT> >
TensorNetwork2D<TenElemT, QNT>::GetBMPSForRow(const size_t row) {
  const size_t rows = this->rows();
  GrowBMPSForRow(row);
  BMPST &up_bmps = bmps_set_[UP][row];
  BMPST &down_bmps = bmps_set_[DOWN][rows - 1 - row];

  return std::pair(up_bmps, down_bmps);
}

template<typename TenElemT, typename QNT>
const std::pair<BMPS<TenElemT, QNT>, BMPS<TenElemT, QNT> >
TensorNetwork2D<TenElemT, QNT>::GetBMPSForCol(const size_t col) {
  const size_t cols = this->cols();
  GrowBMPSForCol(col);
  BMPST &left_bmps = bmps_set_[LEFT][col];
  BMPST &right_bmps = bmps_set_[RIGHT][cols - 1 - col];
  return std::pair(left_bmps, right_bmps);
}


template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::InitBTen(const gqpeps::BTenPOSITION position, const size_t slice_num) {
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
void TensorNetwork2D<TenElemT, QNT>::GrowFullBTen(const gqpeps::BTenPOSITION position,
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
      GrowBMPSForCol(col);
      auto &left_bmps = bmps_set_[LEFT].back();
      auto &right_bmps = bmps_set_[RIGHT].back();
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
      GrowBMPSForCol(col);
      auto &left_bmps = bmps_set_[LEFT].back();
      auto &right_bmps = bmps_set_[RIGHT].back();
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
      GrowBMPSForRow(row);
      auto &up_bmps = bmps_set_[UP].back();
      auto &down_bmps = bmps_set_[DOWN].back();
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
      GrowBMPSForRow(row);
      auto &up_bmps = bmps_set_[UP].back();
      auto &down_bmps = bmps_set_[DOWN].back();
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
      std::cout << "TODO code." << std::endl;
      exit(1);
      break;
    }
    case UP: {
      std::cout << "TODO code." << std::endl;
      exit(1);
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
        Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
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
        Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
        Contract<TenElemT, QNT, false, false>(tmp2, mpo_ten2, 4, ctrct_mpo_start_idx, 2, tmp3);
        Contract(&tmp3, {0, 3}, &mps_ten2, {0, 1}, &next_bten);
        btens.emplace_back(next_bten);
      }
      break;
    }
  }
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::BMPSMoveStep(const BMPSPOSITION position) {
  bmps_set_[position].pop_back();
  auto oppo_post = Opposite(position);
  GrowBMPSStep_(oppo_post);
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::BTenMoveStep(const BTenPOSITION position) {
  bten_set_[position].pop_back();
  GrowBTenStep_(Opposite(position));
}


template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::BTen2MoveStep(const BTenPOSITION position, const size_t slice_num1) {
  bten_set2_[position].pop_back();
  GrowBTen2Step_(Opposite(position), slice_num1);
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::GrowBTenStep_(const BTenPOSITION post) {
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
      const size_t col = slice_num1;
      N = this->rows();
      grown_site1[0] = bten_size - 1;
      grown_site1[1] = col;
      grown_site2[0] = bten_size - 1;
      grown_site2[1] = col + 1;
      mps1_num = this->cols() - 1 - (col + 1);
      mps2_num = col;
      mpo_ten1 = (*this)(grown_site1);
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
  mps_ten1 = &bmps_set_[pre_post][mps1_num][N - bten_size];
  mps_ten2 = &bmps_set_[next_post][mps2_num][bten_size - 1];
  Contract<TenElemT, QNT, true, true>(*mps_ten1, bten_set2_.at(post).back(), 2, 0, 1, tmp1);
  Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten1, 1, 0, 2, tmp2);
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
                                                             const TensorNetwork2D::Tensor &replace_ten) const {
  //here suppose mps along the horizontal direction.
  Tensor tmp[4];
  const size_t row = site[0];
  const size_t col = site[1];
  const Tensor &up_mps_ten = bmps_set_.at(UP)[row][this->cols() - col - 1];
  const Tensor &down_mps_ten = bmps_set_.at(DOWN)[this->rows() - row - 1][col];
  Contract<TenElemT, QNT, true, true>(up_mps_ten, bten_set_.at(LEFT)[col], 2, 0, 1, tmp[0]);
  Contract<TenElemT, QNT, false, false>(tmp[0], replace_ten, 1, 3, 2, tmp[1]);
  Contract(&tmp[1], {0, 2}, &down_mps_ten, {0, 1}, &tmp[2]);
  Contract(&tmp[2], {0, 1, 2}, &bten_set_.at(RIGHT)[this->cols() - col - 1], {2, 1, 0}, &tmp[3]);
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
  if (mps_orient == HORIZONTAL) {
    const size_t row1 = left_up_site[0];
    const size_t row2 = row1 + 1;
    const size_t col1 = left_up_site[1];
    const size_t col2 = col1 + 1;
    const Tensor &mps_ten1 = bmps_set_.at(UP)[row1][this->cols() - col1 - 1];
    const Tensor &mps_ten2 = bmps_set_.at(DOWN)[this->rows() - 1 - row2][col1];
    const Tensor &mps_ten3 = bmps_set_.at(DOWN)[this->rows() - 1 - row2][col2];
    const Tensor &mps_ten4 = bmps_set_.at(UP)[row1][this->cols() - col2 - 1];
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
    Tensor tmp[9];

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
  } else {
    std::cout << "TODO code" << std::endl;
    exit(1);
    return 0;
  }
}


template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> TensorNetwork2D<TenElemT, QNT>::PunchHole(const gqpeps::SiteIdx &site,
                                                                  const gqpeps::BondOrientation mps_orient) const {
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
void TensorNetwork2D<TenElemT, QNT>::UpdateSiteConfig(const gqpeps::SiteIdx &site, const size_t update_config,
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


}///gqpeps

#endif //GRACEQ_VMC_PEPS_TENSOR_NETWORK_2D_IMPL_H
