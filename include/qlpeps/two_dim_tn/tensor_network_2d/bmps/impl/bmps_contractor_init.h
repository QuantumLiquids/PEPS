// SPDX-License-Identifier: LGPL-3.0-only
#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_INIT_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_INIT_H

#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"

namespace qlpeps {

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

} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_INIT_H

