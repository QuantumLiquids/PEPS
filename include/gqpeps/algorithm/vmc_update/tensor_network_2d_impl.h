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
void TensorNetwork2D<TenElemT, QNT>::BMPSMoveStep(const BMPSPOSITION position) {
  bmps_set_[position].pop_back();
  auto oppo_post = Opposite(position);
  GrowBMPSStep_(oppo_post);
}

template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::BTenMoveStep(const BTenPOSITION position) {
  Tensor tmp1, tmp2, tmp3;
  bten_set_[position].pop_back();
  GrowBTenStep_(Opposite(position));
}


template<typename TenElemT, typename QNT>
void TensorNetwork2D<TenElemT, QNT>::GrowBTenStep_(const BTenPOSITION post) {
  Tensor tmp1, tmp2, next_bten;
  size_t ctrct_mpo_start_idx = (size_t(post) + 3) % 4;
#ifndef NDEBUG
  BMPSPOSITION pre_post = BMPSPOSITION(ctrct_mpo_start_idx);
  BMPSPOSITION next_post = BMPSPOSITION((size_t(post) + 1) % 4);
  assert(bmps_set_.at(pre_post).size() + bmps_set_.at(next_post).size() == this->length(Orientation(pre_post)) + 1);
  assert(bten_set_.at(post).size() > 0 &&
         bten_set_.at(post).size() <= bmps_set_.at(pre_post).back().size()); // has been initialled
#endif
  switch (post) {
    case DOWN: {// up_ten +1, down_ten -1;
      const size_t col = bmps_set_[LEFT].size() - 1;
      const TransferMPO &mpo = this->get_col(col);
      const size_t N = mpo.size();
      const size_t mpo_idx = N - bten_set_.at(post).size();
      auto &left_mps_ten = bmps_set_[LEFT].back()[mpo_idx];
      auto &right_mps_ten = bmps_set_[RIGHT].back()[N - mpo_idx - 1];
      auto &mpo_ten = *mpo[mpo_idx];
      Contract<TenElemT, QNT, true, true>(left_mps_ten, bten_set_.at(post).back(), 2, 0, 1, tmp1);
      Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, ctrct_mpo_start_idx, 2, tmp2);
      Contract(&tmp2, {0, 2}, &right_mps_ten, {0, 1}, &next_bten);
      break;
    }
    case UP: {
      const size_t col = bmps_set_[LEFT].size() - 1;
      const TransferMPO &mpo = this->get_col(col);
      const size_t N = mpo.size();
      const size_t mpo_idx = bten_set_.at(post).size() - 1;

      auto &left_mps_ten = bmps_set_[LEFT].back()[mpo_idx];
      auto &right_mps_ten = bmps_set_[RIGHT].back()[N - mpo_idx - 1];
      auto &mpo_ten = *mpo[mpo_idx];
      Contract<TenElemT, QNT, true, true>(right_mps_ten, bten_set_.at(post).back(), 2, 0, 1, tmp1);
      Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, ctrct_mpo_start_idx, 2, tmp2);
      Contract(&tmp2, {0, 2}, &left_mps_ten, {0, 1}, &next_bten);
      break;
    }
    case LEFT: {
      const size_t row = bmps_set_[UP].size() - 1;
      const TransferMPO &mpo = this->get_row(row);
      const size_t N = mpo.size();
      const size_t mpo_idx = bten_set_.at(post).size() - 1;

      auto &up_mps_ten = bmps_set_[UP].back()[N - mpo_idx - 1];
      auto &down_mps_ten = bmps_set_[DOWN].back()[mpo_idx];
      auto &mpo_ten = *mpo[mpo_idx];

      Contract<TenElemT, QNT, true, true>(up_mps_ten, bten_set_.at(post).back(), 2, 0, 1, tmp1);
      Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, ctrct_mpo_start_idx, 2, tmp2);
      Contract(&tmp2, {0, 2}, &down_mps_ten, {0, 1}, &next_bten);
      break;
    }
    case RIGHT: {
      const size_t row = bmps_set_[UP].size() - 1;
      const TransferMPO &mpo = this->get_row(row);
      const size_t N = mpo.size();
      const size_t mpo_idx = N - bten_set_.at(post).size();

      auto &up_mps_ten = bmps_set_[UP].back()[N - mpo_idx - 1];
      auto &down_mps_ten = bmps_set_[DOWN].back()[mpo_idx];
      auto &mpo_ten = *mpo[mpo_idx];

      Contract<TenElemT, QNT, true, true>(down_mps_ten, bten_set_.at(post).back(), 2, 0, 1, tmp1);
      Contract<TenElemT, QNT, false, false>(tmp1, mpo_ten, 1, ctrct_mpo_start_idx, 2, tmp2);
      Contract(&tmp2, {0, 2}, &up_mps_ten, {0, 1}, &next_bten);
      break;
    }
  }
  bten_set_[post].emplace_back(next_bten);
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
