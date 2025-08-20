// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-07
*
* Description: QuantumLiquids/PEPS project. The 2-dimensional tensor network class, implementation.
*/
#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_TRACE_IMPL_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_TRACE_IMPL_H

namespace qlpeps {

template<typename TenElemT, typename QNT>
TenElemT TensorNetwork2D<TenElemT, QNT>::Trace(const SiteIdx &site_a, const BondOrientation bond_dir) const {
  SiteIdx site_b(site_a);
  if (bond_dir == HORIZONTAL) {
    site_b.col() += 1;
  } else {
    site_b.row() += 1;
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
  Tensor tmp[4];
  const size_t row = site.row();
  const size_t col = site.col();
#ifndef NDEBUG
  // check the environment tensors are sufficient
  if (mps_orient == HORIZONTAL) {
    assert(bmps_set_.at(UP).size() > row);
    assert(bmps_set_.at(DOWN).size() + 1 > this->rows() - row);
    assert(bten_set_.at(LEFT).size() > col);
    assert(bten_set_.at(RIGHT).size() + 1 > this->cols() - col);
  } else {
    assert(bmps_set_.at(LEFT).size() > col);
    assert(bmps_set_.at(RIGHT).size() + 1 > this->cols() - col);
    assert(bten_set_.at(UP).size() > row);
    assert(bten_set_.at(DOWN).size() + 1 > this->rows() - row);
  }
#endif
  const Tensor *up_ten, *down_ten, *left_ten, *right_ten;
  if (mps_orient == HORIZONTAL) {
    const Tensor &up_mps_ten = bmps_set_.at(UP)[row][this->cols() - col - 1];
    const Tensor &down_mps_ten = bmps_set_.at(DOWN)[this->rows() - row - 1][col];
    const Tensor &left_bten = bten_set_.at(LEFT)[col];
    const Tensor &right_bten = bten_set_.at(RIGHT)[this->cols() - col - 1];
    up_ten = &up_mps_ten;
    down_ten = &down_mps_ten;
    left_ten = &left_bten;
    right_ten = &right_bten;
  } else {
    const Tensor &left_mps_ten = bmps_set_.at(LEFT)[col][row];
    const Tensor &right_mps_ten = bmps_set_.at(RIGHT)[this->cols() - col - 1][this->rows() - row - 1];
    const Tensor &up_bten = bten_set_.at(UP)[row];
    const Tensor &down_bten = bten_set_.at(DOWN)[this->rows() - row - 1];
    up_ten = &up_bten;
    down_ten = &down_bten;
    left_ten = &left_mps_ten;
    right_ten = &right_mps_ten;
//    Contract<TenElemT, QNT, true, true>(right_mps_ten, up_bten, 2, 0, 1, tmp[0]);
//    Contract<TenElemT, QNT, false, false>(tmp[0], replace_ten, 1, 2, 2, tmp[1]);
//    Contract(&tmp[1], {0, 2}, &left_mps_ten, {0, 1}, &tmp[2]);
//    Contract(&tmp[2], {0, 1, 2}, &down_ten, {2, 1, 0}, &tmp[3]);
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
TensorNetwork2D<TenElemT, QNT>::ReplaceNNSiteTrace(const SiteIdx &site_a, const SiteIdx &site_b,
                                                   const BondOrientation bond_dir,
                                                   const Tensor &ten_a,
                                                   const Tensor &ten_b) const {
#ifndef NDEBUG
  if (bond_dir == HORIZONTAL) {
    assert(site_a.row() == site_b.row());
    assert(site_a.col() + 1 == site_b.col());
    size_t bond_row = site_a.row();
    assert(bmps_set_.at(UP).size() > bond_row);
    assert(bmps_set_.at(DOWN).size() + 1 > this->rows() - bond_row);
    assert(bten_set_.at(LEFT).size() > site_a[1]);
    assert(bten_set_.at(RIGHT).size() + 1 > this->cols() - site_b[1]);
  } else {
    assert(site_a.row() + 1 == site_b.row());
    assert(site_a.col() == site_b.col());
    size_t bond_col = site_a.col();
    assert(bmps_set_.at(LEFT).size() > bond_col);
    assert(bmps_set_.at(RIGHT).size() + 1 > this->cols() - bond_col);
    assert(bten_set_.at(UP).size() > site_a.row());
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
    const Tensor &up_mps_ten_b = bmps_set_.at(UP)[row][this->cols() - col_b - 1];
    const Tensor &down_mps_ten_b = bmps_set_.at(DOWN)[this->rows() - row - 1][col_b];
    if constexpr (Tensor::IsFermionic()) {
      Contract<TenElemT, QNT, true, true>(down_mps_ten_b,
                                          bten_set_.at(RIGHT)[this->cols() - col_b - 1],
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
                                          bten_set_.at(RIGHT)[this->cols() - col_b - 1],
                                          2,
                                          0,
                                          1,
                                          tmp[3]);
      Contract<TenElemT, QNT, false, false>(tmp[3], ten_b, 1, 1, 2, tmp[4]);
      Contract(&tmp[4], {0, 2}, &up_mps_ten_b, {0, 1}, &tmp[5]);
    }
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
    const Tensor &right_mps_ten_b = bmps_set_.at(RIGHT)[this->cols() - col - 1][this->rows() - row_b - 1];
    if constexpr (Tensor::IsFermionic()) {
      Contract<TenElemT, QNT, true, true>(left_mps_ten_b,
                                          bten_set_.at(DOWN)[this->rows() - row_b - 1],
                                          2, 0, 1, tmp[3]);
      tmp[3].FuseIndex(0, 5);
      Contract(&tmp[3], {2, 3}, &ten_b, {0, 1}, &tmp[4]);
      Contract(&tmp[4], {2, 3}, &right_mps_ten_b, {0, 1}, &tmp[5]);
      tmp[5].FuseIndex(0, 5);
    } else {
      Contract<TenElemT, QNT, true, true>(left_mps_ten_b,
                                          bten_set_.at(DOWN)[this->rows() - row_b - 1],
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
    // make sure the trivial indices of the sites a and b are connected
    return tmp[6].GetElem({0, 0, 0, 0});
  } else {
    Contract(&tmp[2], {0, 1, 2}, &tmp[5], {2, 1, 0}, &tmp[6]);
    return tmp[6]();
  }
} //ReplaceNNSiteTrace

/**
 * Trace the 2 by 2 cluster of tensors by replacing two next-nearest neighbor tensors with another two given tensors.
 *
 * For fermion, the trivial indices of the two diagonal tensors are moved to continuous.
 * More explicitly, the order of the trivial indices of the traced tensor is:
 * (environment, site 1, site 3, site 0, site 2)
 * (As for the definition of the site position of site 0,1,2,3, please refer to the mpo tensor figures inside the function).
 * This make sure the operators like NNN hopping acts on continuous fermion operators.
 *
 *
 * @param left_up_site  defines the left-up site of the 2 by 2 cluster.
 * @param nnn_dir       defines the direction of the next-nearest neighbor tensors (LEFTUP_TO_RIGHTDOWN or LEFTDOWN_TO_RIGHTUP).
 * @param mps_orient    defines the orientation of the MPS (HORIZONTAL or VERTICAL).
 * @param ten_left      the tensor to replace the left-up or left-down tensor in the cluster.
 * @param ten_right     the tensor to replace the right-up or right-down tensor in the cluster.
 * @return  trace
 */
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
     *      BTEN2-LEFT                      BTEN2-RIGHT
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

    Tensor *mpo_ten0, *mpo_ten2;
    const Tensor *mpo_ten1, *mpo_ten3;
    const Tensor &left_bten = bten_set2_.at(LEFT)[col1];
    const Tensor &right_bten = bten_set2_.at(RIGHT)[this->cols() - col2 - 1];
    if (nnn_dir == LEFTUP_TO_RIGHTDOWN) {
      if (Tensor::IsFermionic()) {
        mpo_ten0 = const_cast<Tensor *>(&ten_left);
        mpo_ten2 = const_cast<Tensor *>(&ten_right);
      } else {
        mpo_ten0 = new Tensor(ten_left);
        mpo_ten2 = new Tensor(ten_right);
      }
      mpo_ten1 = ((*this)(row2, col1));
      mpo_ten3 = ((*this)(row1, col2));
    } else { //LEFTDOWN_TO_RIGHTUP
      if (Tensor::IsFermionic()) {
        mpo_ten0 = const_cast<Tensor *>((*this)(row1, col1));
        mpo_ten2 = const_cast<Tensor *>((*this)(row2, col2));
      } else {
        mpo_ten0 = new Tensor((*this)({row1, col1}));
        mpo_ten2 = new Tensor((*this)({row2, col2}));
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
    assert(!Tensor::IsFermionic());
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

///< For fermion case, the sign of the output is meaningless.
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
TenElemT TensorNetwork2D<TenElemT, QNT>::ReplaceSqrt5DistTwoSiteTrace(const SiteIdx &left_up_site,
                                                                      const DIAGONAL_DIR sqrt5link_dir,
                                                                      const BondOrientation mps_orient, //mps orientation is the same with longer side orientation
                                                                      const Tensor &ten_left,
                                                                      const Tensor &ten_right) const {
  assert(!Tensor::IsFermionic());
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
}
#endif //QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TENSOR_NETWORK_2D_TRACE_IMPL_H
