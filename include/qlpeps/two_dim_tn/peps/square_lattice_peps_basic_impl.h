// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-21
*
* Description: QuantumLiquids/PEPS project. The generic PEPS class, implementation.
*/


#ifndef VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_BASIC_IMPL_H
#define VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_BASIC_IMPL_H

#include "qlmps/utilities.h"       //mock_qlten::SVD

namespace qlpeps {
using namespace qlten;
using qlmps::mock_qlten::SVD;

template<typename TenElemT, typename QNT>
SquareLatticePEPS<TenElemT, QNT>::
SquareLatticePEPS(const HilbertSpaces<QNT> &hilbert_spaces):
    rows_(hilbert_spaces.size()),
    cols_(hilbert_spaces[0].size()),
    Gamma(hilbert_spaces.size(), hilbert_spaces[0].size()),
    lambda_vert(hilbert_spaces.size() + 1, hilbert_spaces[0].size()),
    lambda_horiz(hilbert_spaces.size(), hilbert_spaces[0].size() + 1) {
#ifndef NDEBUG
  for (size_t i = 0; i < hilbert_spaces.size(); i++) {
    for (size_t j = 0; j < hilbert_spaces[0].size(); j++) {
      assert(hilbert_spaces[i][j].GetDir() == qlten::OUT);
    }
  }
  assert(rows_ > 0 && cols_ > 0);
#endif
//  QNT qn_site000 = hilbert_spaces[0][0].GetQNSct(0).GetQn();

  Index<QNT> index0_in({QNSector(qn0_, 1)}, IN), index0_out({QNSector(qn0_, 1)}, OUT);

  for (size_t row = 0; row < lambda_vert.rows(); row++) {
    for (size_t col = 0; col < lambda_vert.cols(); col++) {
      DTensor &the_lambda = lambda_vert({row, col});
      the_lambda = DTensor({index0_in, index0_out});
      the_lambda({0, 0}) = (1.0);
    }
  }

  for (size_t row = 0; row < lambda_horiz.rows(); row++) {
    for (size_t col = 0; col < lambda_horiz.cols(); col++) {
      DTensor &the_lambda = lambda_horiz({row, col});
      the_lambda = DTensor({index0_in, index0_out});
      the_lambda({0, 0}) = (1.0);
    }
  }

  for (size_t row = 0; row < Gamma.rows(); row++) {
    for (size_t col = 0; col < Gamma.cols(); col++) {
      Gamma({row, col}) = TenT({index0_in, index0_out, index0_out, index0_in, hilbert_spaces[row][col]});
    }
  }
}

// Helper function to generate a Hilbert space with the same local hilbert space repeated
template<typename QNT>
HilbertSpaces<QNT> GenerateHilbertSpace(size_t rows, size_t cols, const Index<QNT> &local_hilbert_space) {
  HilbertSpaces<QNT> hilbert_spaces(rows, std::vector<Index<QNT>>(cols, local_hilbert_space));
  return hilbert_spaces;
}

template<typename TenElemT, typename QNT>
SquareLatticePEPS<TenElemT, QNT>::SquareLatticePEPS(const Index<QNT> &local_hilbert_space, size_t rows, size_t cols)
    : SquareLatticePEPS(GenerateHilbertSpace(rows, cols, local_hilbert_space)) {}

/**
 * Initial PEPS as a direct product state, according to activates configuration
 *
 * @param activates represent the direct product state; should has the same size with the PEPS
 *
 * @return
 * The gamma tensors are all assumed have the following index directions:
 *
 *          3
 *          |
 *          ↑
 *          |
 *  0-->--Gamma-->--2
 *          |
 *          ↓
 *          |
 *          1
 *
 *  and physical index 4 points out.
 *
 *  Every gamma tensor has Div==0, except the most right-lower one.
 *  All the external virtual indices of the PEPS, has dimension 1, and QN=0 (qn0)
 *
 */
template<typename TenElemT, typename QNT>
void SquareLatticePEPS<TenElemT, QNT>::Initial(std::vector<std::vector<size_t>> &activates) {
  Index<QNT> index0_in({QNSector(qn0_, 1)}, IN), index0_out({QNSector(qn0_, 1)}, OUT);
  // The lambda tensors surrounding the PEPS
  DTensor surrounding_lam = DTensor({index0_in, index0_out});
  surrounding_lam({0, 0}) = 1.0;
  // upper vertical lambda
  for (size_t col = 0; col < cols_; col++) {
    const size_t row = 0;
    lambda_vert({row, col}) = surrounding_lam;
  }

  //lower vertical lambda
  for (size_t col = 0; col < cols_; col++) {
    const size_t row = this->rows_;
    lambda_vert({row, col}) = surrounding_lam;
  }

  //left horizontal lambda
  for (size_t row = 0; row < rows_; row++) {
    const size_t col = 0;
    lambda_horiz({row, col}) = surrounding_lam;
  }
  //right horizontal lambda
  for (size_t row = 0; row < rows_ - 1; row++) {
    const size_t col = this->cols_;
    lambda_horiz({row, col}) = surrounding_lam;
  }

  //horizontal lambda except the last row are also set as trivial
  for (size_t row = 0; row < rows_ - 1; row++) {
    for (size_t col = 0; col < cols_; col++) {
      lambda_horiz({row, col}) = surrounding_lam;
    }
  }

  // row = 0
  for (size_t col = 0; col < cols_; col++) {
    const size_t row = 0;
    Index<QNT> index0 = lambda_horiz({row, col}).GetIndex(0);
    Index<QNT> index2 = index0_out;
    Index<QNT> index3 = lambda_vert({row, col}).GetIndex(0);
    Index<QNT> phy_idx = Gamma({row, col}).GetIndex(4);
    QNT index1_qn = index0.GetQNSct(0).GetQn()  // in
        + index3.GetQNSct(0).GetQn()        //in
        - index2.GetQNSct(0).GetQn()      //out
        - phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
    Index<QNT> index1({QNSector(index1_qn, 1)}, OUT);
    Gamma({row, col}) = TenT({index0, index1, index2, index3, phy_idx});
    Gamma({row, col})({0, 0, 0, 0, activates[row][col]}) = 1.0;
    //lambda_horiz tensors in first row have been set.
  }

  for (size_t row = 1; row < rows_ - 1; row++) {
    // set a layer of lambda_vert
    for (size_t col = 0; col < cols_; col++) {
      lambda_vert({row, col}) = DTensor({InverseIndex(Gamma({row - 1, col}).GetIndex(1)),
                                         Gamma({row - 1, col}).GetIndex(1)});
      lambda_vert({row, col})({0, 0}) = 1.0;
    }
    //set a layer of gamma tensors
    for (size_t col = 0; col < cols_; col++) {
      Index<QNT> index0 = lambda_horiz({row, col}).GetIndex(0);
      Index<QNT> index2 = index0_out;
      Index<QNT> index3 = lambda_vert({row, col}).GetIndex(0);
      Index<QNT> phy_idx = Gamma({row, col}).GetIndex(4);
      QNT index1_qn = index0.GetQNSct(0).GetQn()  // in
          + index3.GetQNSct(0).GetQn()        //in
          - index2.GetQNSct(0).GetQn()      //out
          - phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
      Index<QNT> index1({QNSector(index1_qn, 1)}, OUT);
      Gamma({row, col}) = TenT({index0, index1, index2, index3, phy_idx});
      Gamma({row, col})({0, 0, 0, 0, activates[row][col]}) = 1.0;
    }
    //use the default lambda_horiz tensors in these layers
  }

  for (size_t col = 0; col < cols_; col++) {
    const size_t row = rows_ - 1;
    lambda_vert({row, col}) = DTensor({InverseIndex(Gamma({row - 1, col}).GetIndex(1)),
                                       Gamma({row - 1, col}).GetIndex(1)});
    lambda_vert({row, col})({0, 0}) = 1.0;
  }

  //last layer of gamma and horizontal lambda
  for (size_t col = 0; col < cols_; col++) {
    const size_t row = rows_ - 1;
    Index<QNT> index0 = lambda_horiz({row, col}).GetIndex(0);
    Index<QNT> index1 = lambda_vert({row + 1, col}).GetIndex(1);
    Index<QNT> index3 = lambda_vert({row, col}).GetIndex(0);
    Index<QNT> phy_idx = Gamma({row, col}).GetIndex(4);
    QNT index2_qn = index0.GetQNSct(0).GetQn()  // in
        + index3.GetQNSct(0).GetQn()        //in
        - index1.GetQNSct(0).GetQn()      //out
        - phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
    Index<QNT> index2({QNSector(index2_qn, 1)}, OUT);
    Gamma({row, col}) = TenT({index0, index1, index2, index3, phy_idx});
    Gamma({row, col})({0, 0, 0, 0, activates[row][col]}) = 1.0;

    lambda_horiz({row, col + 1}) = DTensor({InverseIndex(index2), index2});
    lambda_horiz({row, col + 1})({0, 0}) = 1.0;
  }
  if constexpr (Index<QNT>::IsFermionic()) {
    if (lambda_horiz({rows_ - 1, cols_}).GetIndex(0).GetQNSct(0).GetQn().IsFermionParityOdd()) {
      std::cout << "warning : the direct product PEPS as odd fermion parity!" << std::endl;
    }
  }
}

template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> SquareLatticePEPS<TenElemT, QNT>::GetMinMaxBondDim(void) const {
  size_t dmax(0), dmin(lambda_vert({1, 1}).GetShape()[0]);
  for (size_t row = 0; row < rows_; row++) {
    for (size_t col = 0; col < cols_; col++) {
      size_t d = lambda_vert({row, col}).GetShape()[0];
      dmax = std::max(dmax, d);
      if (row > 0)
        dmin = std::min(dmin, d);
      d = lambda_horiz({row, col}).GetShape()[0];
      dmax = std::max(dmax, d);
      if (col > 0)
        dmin = std::min(dmin, d);
    }
  }
  return std::pair<size_t, size_t>(dmin, dmax);
}

template<typename TenElemT, typename QNT>
size_t SquareLatticePEPS<TenElemT, QNT>::GetMaxBondDim(void) const {
  size_t dmax = 0;
  for (size_t row = 0; row < rows_; row++) {
    for (size_t col = 0; col < cols_; col++) {
      size_t d = lambda_vert({row, col}).GetShape()[0];
      dmax = std::max(dmax, d);
      d = lambda_horiz({row, col}).GetShape()[0];
      dmax = std::max(dmax, d);
    }
  }
  return dmax;
}

template<typename TenElemT, typename QNT>
bool SquareLatticePEPS<TenElemT, QNT>::IsBondDimensionEven(void) const {
  size_t d = lambda_vert({1, 0}).GetShape()[0];

  for (size_t row = 1; row < rows_; row++) {
    for (size_t col = 0; col < cols_; col++) {
      if (d != lambda_vert({row, col}).GetShape()[0]) {
        return false;
      }
    }
  }

  for (size_t row = 0; row < rows_; row++) {
    for (size_t col = 1; col < cols_; col++) {
      if (d != lambda_horiz({row, col}).GetShape()[0]) {
        return false;
      }
    }
  }

  return true;
}

template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::NormalizeAllTensor() {
  double norm(1.0);
  for (auto &gamma : Gamma) {
    norm *= gamma.QuasiNormalize();
  }

  for (auto &lambda : lambda_vert) {
    norm *= lambda.QuasiNormalize();
  }
  for (auto &lambda : lambda_vert) {
    norm *= lambda.QuasiNormalize();
  }
  return norm;
}

template<typename TenElemT, typename QNT>
bool SquareLatticePEPS<TenElemT, QNT>::operator==(const SquareLatticePEPS<TenElemT, QNT> &rhs) const {
  // Check if the number of rows and columns are the same
  if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
    return false;
  }

  // Check if Gamma elements are equal
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      if (Gamma({row, col}) != rhs.Gamma({row, col})) {
        return false;
      }
    }
  }

  // Check if lambda_vert elements are equal
  for (size_t row = 0; row <= rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      if (lambda_vert({row, col}) != rhs.lambda_vert({row, col})) {
        return false;
      }
    }
  }

  // Check if lambda_horiz elements are equal
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col <= cols_; ++col) {
      if (lambda_horiz({row, col}) != rhs.lambda_horiz({row, col})) {
        return false;
      }
    }
  }

  // If all elements are equal, return true
  return true;
}

template<typename TenElemT, typename QNT>
bool SquareLatticePEPS<TenElemT, QNT>::Dump(const std::string path) const {
  // Dump Gamma, lambda_vert, and lambda_horiz tensors one by one
  if (!qlmps::IsPathExist(path)) { qlmps::CreatPath(path); }
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "gamma_ten_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!Gamma.DumpTen(row, col, filename)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump Gamma tensor
      }
    }
  }

  for (size_t row = 0; row <= rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_vert.DumpTen(row, col, filename)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col <= cols_; ++col) {
      std::string filename = path + "lam_h_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_horiz.DumpTen(row, col, filename)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_horiz tensor
      }
    }
  }

  return true; // Successfully dumped all tensors
}

template<typename TenElemT, typename QNT>
bool SquareLatticePEPS<TenElemT, QNT>::Dump(const std::string path, bool release_mem) {
  // Dump Gamma, lambda_vert, and lambda_horiz tensors one by one
  if (!qlmps::IsPathExist(path)) { qlmps::CreatPath(path); }
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "/gamma_ten_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!Gamma.DumpTen(row, col, filename, release_mem)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump Gamma tensor
      }
    }
  }

  for (size_t row = 0; row <= rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "/lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_vert.DumpTen(row, col, filename, release_mem)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col <= cols_; ++col) {
      std::string filename = path + "/lam_h_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_horiz.DumpTen(row, col, filename, release_mem)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_horiz tensor
      }
    }
  }

  return true; // Successfully dumped all tensors
}

template<typename TenElemT, typename QNT>
bool SquareLatticePEPS<TenElemT, QNT>::Load(const std::string path) {
  // Load Gamma, lambda_vert, and lambda_horiz tensors one by one
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "/gamma_ten_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!Gamma.LoadTen(row, col, filename)) {
        std::cout << "Failed to load tensor from file: " << filename << std::endl;
        return false; // Failed to dump Gamma tensor
      }
    }
  }

  for (size_t row = 0; row <= rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "/lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_vert.LoadTen(row, col, filename)) {
        std::cout << "Failed to load tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col <= cols_; ++col) {
      std::string filename = path + "/lam_h_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_horiz.LoadTen(row, col, filename)) {
        std::cout << "Failed to load tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_horiz tensor
      }
    }
  }

  return true; // Successfully loaded all tensors
}

template<typename TenElemT, typename QNT>
SquareLatticePEPS<TenElemT, QNT>::operator TPS<TenElemT, QNT>() const {
  auto tps = TPS<TenElemT, QNT>(rows_, cols_);
  for (size_t row = 0; row < rows_; row++) {
    for (size_t col = 0; col < cols_; col++) {
      tps.alloc(row, col);
      const DTensor lam_left_sqrt = ElementWiseSqrt(lambda_horiz({row, col}));
      const DTensor lam_right_sqrt = ElementWiseSqrt(lambda_horiz({row, col + 1}));
      const DTensor lam_up_sqrt = ElementWiseSqrt(lambda_vert({row, col}));
      const DTensor lam_down_sqrt = ElementWiseSqrt(lambda_vert({row + 1, col}));

      TenT tmp[3];
      Contract<TenElemT, QNT, true, true>(lam_up_sqrt, Gamma({row, col}), 1, 3, 1, tmp[0]);
      Contract<TenElemT, QNT, false, true>(lam_right_sqrt, tmp[0], 0, 4, 1, tmp[1]);
      Contract<TenElemT, QNT, false, true>(lam_down_sqrt, tmp[1], 0, 4, 1, tmp[2]);
      Contract<TenElemT, QNT, true, true>(lam_left_sqrt, tmp[2], 1, 4, 1, tps({row, col}));
#ifndef NDEBUG
      auto physical_index = Gamma(row, col)->GetIndex(4);
      assert(physical_index == tps(row, col)->GetIndex(4));
#endif
    }
  }

  return tps;
}

/**
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param site
 * @return physical index 0, auxiliary indexes follow original order
 *
 * res:
 *          3
 *          |
 *          |
 *    0-----T------2   and physical idx = 4
 *          |
 *          |
 *          1
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> SquareLatticePEPS<TenElemT, QNT>::EatSurroundLambdas_(const SiteIdx &site) const {
  TenT tmp_ten[3], res;
  const size_t row = site[0], col = site[1];
  Contract<TenElemT, QNT, false, false>(Gamma(site), lambda_horiz(site), 0, 1, 1, tmp_ten[0]);
  Contract<TenElemT, QNT, false, true>(tmp_ten[0], lambda_vert({row + 1, col}), 0, 0, 1, tmp_ten[1]);
  Contract<TenElemT, QNT, false, true>(tmp_ten[1], lambda_horiz({row, col + 1}), 0, 0, 1, tmp_ten[2]);
  Contract<TenElemT, QNT, false, false>(tmp_ten[2], lambda_vert(site), 0, 1, 1, res);
  return res;
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT>
SquareLatticePEPS<TenElemT, QNT>::Eat3SurroundLambdas_(const SiteIdx &site,
                                                       const BTenPOSITION remaining_idx) const {
  size_t row = site[0], col = site[1];
  TenT tmp_ten[2], res;
  switch (remaining_idx) {
    case RIGHT: {
      Contract<TenElemT, QNT, true, false>(Gamma({row, col}), lambda_vert({row, col}), 3, 1, 1, tmp_ten[0]);
      Contract<TenElemT, QNT, false, false>(tmp_ten[0], lambda_horiz({row, col}), 1, 1, 1, tmp_ten[1]);
      Contract<TenElemT, QNT, false, true>(tmp_ten[1], lambda_vert({row + 1, col}), 0, 0, 1, res);
      res.Transpose({1, 3, 4, 2, 0});
      return res;
      /*      0
       *      |
       *  1--res--4, physical index = 3
       *      |
       *      2
       */
    }
    case LEFT: {
      Contract<TenElemT, QNT, false, false>(Gamma({row, col}), lambda_vert({row, col}), 3, 1, 1, tmp_ten[0]);
      Contract<TenElemT, QNT, false, true>(tmp_ten[0], lambda_horiz({row, col + 1}), 3, 0, 1, tmp_ten[1]);
      Contract<TenElemT, QNT, false, false>(lambda_vert({row + 1, col}), tmp_ten[1], 0, 3, 1, res);
      return res;
      /*      2
       *      |
       *  4--res--1, physical index = 3
       *      |
       *      0
       */
    }
    case DOWN: {
      Contract<TenElemT, QNT, true, true>(lambda_horiz({row, col}), Gamma({row, col}), 1, 0, 1, tmp_ten[0]);
      Contract<TenElemT, QNT, true, true>(lambda_vert({row, col}), tmp_ten[0], 1, 3, 1, tmp_ten[1]);
      Contract<TenElemT, QNT, false, false>(lambda_horiz({row, col + 1}), tmp_ten[1], 0, 4, 1, res);
      res.Transpose({0, 1, 3, 2, 4});
      return res;
      /*      1
       *      |
       *  2--res--0, physical index = 3
       *      |
       *      4
       */
    }
    case UP: {
      Contract<TenElemT, QNT, true, true>(lambda_horiz({row, col}), Gamma({row, col}), 1, 0, 1, tmp_ten[0]);
      Contract<TenElemT, QNT, false, true>(lambda_vert({row + 1, col}), tmp_ten[0], 0, 1, 1, tmp_ten[1]);
      Contract<TenElemT, QNT, false, true>(lambda_horiz({row, col + 1}), tmp_ten[1], 0, 1, 1, res);
      res.Transpose({0, 4, 3, 2, 1});
      return res;
      /*      4
       *      |
       *  2--res--0, physical index = 3
       *      |
       *      1
       */
    }
    default: {
      return TenT();
    }
  }
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT>
SquareLatticePEPS<TenElemT, QNT>::QTenSplitOutLambdas_(const QLTensor<TenElemT, QNT> &q, const SiteIdx &site,
                                                       const BTenPOSITION remaining_idx,
                                                       double inv_tolerance) const {
  TenT tmp_ten[2], res;
  DTensor inv_lambda;
  const size_t row = site[0], col = site[1];
  switch (remaining_idx) {
    case RIGHT: {
      /* input:
       *      0
       *      |
       *  1---q---3
       *      |
       *      2
       */
      inv_lambda = ElementWiseInv(lambda_vert({row, col}), inv_tolerance);
      Contract<TenElemT, QNT, false, false>(q, inv_lambda, 0, 1, 1, tmp_ten[0]);
      inv_lambda = ElementWiseInv(lambda_horiz({row, col}), inv_tolerance);
      Contract<TenElemT, QNT, false, false>(tmp_ten[0], inv_lambda, 0, 1, 1, tmp_ten[1]);
      inv_lambda = ElementWiseInv(lambda_vert({row + 1, col}), inv_tolerance);
      Contract<TenElemT, QNT, false, true>(tmp_ten[1], inv_lambda, 0, 0, 1, res);
      /*  output:
       *      1
       *      |
       *  2--res--0
       *      |
       *      3
       */
      return res;
    }
    case LEFT: {
      /* input:
       *      2
       *      |
       *  3---q---1
       *      |
       *      0
       */
      inv_lambda = ElementWiseInv(lambda_vert({row + 1, col}), inv_tolerance);
      Contract<TenElemT, QNT, false, false>(q, inv_lambda, 0, 0, 1, tmp_ten[0]);
      inv_lambda = ElementWiseInv(lambda_horiz({row, col + 1}), inv_tolerance);
      Contract<TenElemT, QNT, false, false>(tmp_ten[0], inv_lambda, 0, 0, 1, tmp_ten[1]);
      inv_lambda = ElementWiseInv(lambda_vert({row, col}), inv_tolerance);
      Contract<TenElemT, QNT, false, true>(tmp_ten[1], inv_lambda, 0, 1, 1, res);
      /*  output:
       *      3
       *      |
       *  0--res--2
       *      |
       *      1
       */
      return res;
    }
    case DOWN: {
      /*      1
       *      |
       *  2---q---0
       *      |
       *      3
       */
      inv_lambda = ElementWiseInv(lambda_vert(site), inv_tolerance);
      Contract<TenElemT, QNT, true, true>(inv_lambda, q, 1, 1, 1, tmp_ten[0]);
      inv_lambda = ElementWiseInv(lambda_horiz(site), inv_tolerance);
      Contract<TenElemT, QNT, true, true>(inv_lambda, tmp_ten[0], 1, 1, 1, tmp_ten[1]);
      inv_lambda = ElementWiseInv(lambda_horiz({row, col + 1}), inv_tolerance);
      Contract<TenElemT, QNT, false, true>(inv_lambda, tmp_ten[1], 0, 2, 1, res);
      /*  output:
       *      1
       *      |
       *  2--res--0
       *      |
       *      3
       */
      return res;
    }
    case UP: {
      /*      3
       *      |
       *  2---q---0
       *      |
       *      1
       */
      inv_lambda = ElementWiseInv(lambda_horiz({row, col + 1}), inv_tolerance);
      Contract<TenElemT, QNT, false, true>(inv_lambda, q, 0, 0, 1, tmp_ten[0]);
      inv_lambda = ElementWiseInv(lambda_vert({row + 1, col}), inv_tolerance);
      Contract<TenElemT, QNT, false, true>(inv_lambda, tmp_ten[0], 0, 1, 1, tmp_ten[1]);
      inv_lambda = ElementWiseInv(lambda_horiz({row, col}), inv_tolerance);
      Contract<TenElemT, QNT, true, true>(inv_lambda, tmp_ten[1], 1, 1, 1, res);
      /*  output:
       *      1
       *      |
       *  0--res--2
       *      |
       *      3
       */
      return res;
    }
  }
  return res;
}
}//qlpeps

#endif //VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_BASIC_IMPL_H
