// SPDX-License-Identifier: LGPL-3.0-only
#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_HELPERS_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_HELPERS_H

#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"

namespace qlpeps {

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
} // anonymous namespace

} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_HELPERS_H

