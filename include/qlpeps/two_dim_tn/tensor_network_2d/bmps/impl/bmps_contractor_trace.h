// SPDX-License-Identifier: LGPL-3.0-only
#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_TRACE_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_TRACE_H

#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/impl/bmps_contractor_helpers.h"

namespace qlpeps {

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

} // namespace qlpeps

#endif // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_IMPL_BMPS_CONTRACTOR_TRACE_H

