// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-07
*
* Description: QuantumLiquids/PEPS project. The 2-dimensional tensor network class, implementation.
*/


#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BASIC_IMPL_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BASIC_IMPL_H

#include <stdexcept>
#include <string>

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
  assert(config.rows() == tps.rows());
  assert(config.cols() == tps.cols());
  
  // Input validation - detect uninitialized TPS early
  for (size_t row = 0; row < tps.rows(); row++) {
    for (size_t col = 0; col < tps.cols(); col++) {
      const auto& site_tensors = tps({row, col});
      
      size_t config_val = config({row, col});
      
      // Check for empty tensor vector first
      if (site_tensors.empty()) {
        throw std::invalid_argument("TensorNetwork2D: TPS at site (" + std::to_string(row) + 
                                   ", " + std::to_string(col) + ") has empty tensor vector. " +
                                   "TPS must be properly initialized before creating TensorNetwork2D.");
      }
      
      // Check configuration bounds
      if (config_val >= site_tensors.size()) {
        throw std::out_of_range("TensorNetwork2D: Configuration value " + std::to_string(config_val) + 
                               " at site (" + std::to_string(row) + ", " + std::to_string(col) + 
                               ") exceeds TPS physical dimension " + std::to_string(site_tensors.size()));
      }
      
      // Check if the specific tensor component needed is default
      if (site_tensors[config_val].IsDefault()) {
        throw std::invalid_argument("TensorNetwork2D: TPS tensor at site (" + std::to_string(row) + 
                                   ", " + std::to_string(col) + ") component " + std::to_string(config_val) + 
                                   " is default (uninitialized). TPS must be properly initialized before creating TensorNetwork2D.");
      }
      
#ifndef NDEBUG
      (*this)({row, col}) = tps({row, col}).at(config({row, col}));
#else
      (*this)({row, col}) = tps({row, col})[config({row, col})];
#endif
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
  assert(bmps_set_.at(post).empty());
  const size_t mps_size = this->length(Rotate(Orientation(post)));
  std::vector<IndexT> boundary_indices;
  boundary_indices.reserve(mps_size);

  switch (post) {
    case LEFT : {
      for (size_t i = 0; i < mps_size; i++) {
        boundary_indices.push_back(InverseIndex((*this)({i, 0}).GetIndex(post)));
      }
      break;
    }
    case DOWN: {
      for (size_t i = 0; i < mps_size; i++) {
        boundary_indices.push_back(InverseIndex((*this)({this->rows() - 1, i}).GetIndex(post)));
      }
      break;
    }
    case RIGHT: {
      for (size_t i = 0; i < mps_size; i++) {
        boundary_indices.push_back(InverseIndex((*this)({this->rows() - i - 1, this->cols() - 1}).GetIndex(post)));
      }
      break;
    }
    case UP: {
      for (size_t i = 0; i < mps_size; i++) {
        boundary_indices.push_back(InverseIndex((*this)({0, this->cols() - i - 1}).GetIndex(post)));
      }
      break;
    }
  }
  BMPS<TenElemT, QNT> boundary_bmps(post, boundary_indices);
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
void TensorNetwork2D<TenElemT, QNT>::BMPSMoveStep(const BMPSPOSITION position, const BMPSTruncatePara &trunc_para) {
  bmps_set_[position].pop_back();
  auto oppo_post = Opposite(position);
  GrowBMPSStep_(oppo_post, trunc_para);
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> TensorNetwork2D<TenElemT, QNT>::PunchHole(const qlpeps::SiteIdx &site,
                                                                  const qlpeps::BondOrientation mps_orient) const {
  const Tensor *left_ten, *down_ten, *right_ten, *up_ten;
  const size_t row = site[0];
  const size_t col = site[1];
#ifndef NDEBUG
  if (mps_orient == HORIZONTAL) {
    up_ten = &(bmps_set_.at(UP).at(row)[this->cols() - col - 1]);
    down_ten = &(bmps_set_.at(DOWN).at(this->rows() - row - 1)[col]);
    left_ten = &(bten_set_.at(LEFT).at(col));
    right_ten = &(bten_set_.at(RIGHT).at(this->cols() - col - 1));
  } else {
    up_ten = &(bten_set_.at(UP).at(row));
    down_ten = &(bten_set_.at(DOWN).at(this->rows() - row - 1));
    left_ten = &(bmps_set_.at(LEFT).at(col)[row]);
    right_ten = &(bmps_set_.at(RIGHT).at(this->cols() - col - 1)[this->rows() - row - 1]);
  }
#else
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
void TensorNetwork2D<TenElemT, QNT>::UpdateSiteTensor(const qlpeps::SiteIdx &site, const size_t update_config,
                                                      const SplitIndexTPS<TenElemT, QNT> &sitps, bool check_envs) {
  (*this)(site) = sitps(site)[update_config];
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

#endif //QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_BASIC_IMPL_H
