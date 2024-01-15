// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-21
*
* Description: GraceQ/VMC-PEPS project. The generic PEPS class, implementation.
*/


#ifndef VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_IMPL_H
#define VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_IMPL_H

#include "gqmps2/utilities.h"       //mock_gqten::SVD

namespace gqpeps {
using namespace gqten;
using gqmps2::mock_gqten::SVD;

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
      assert(hilbert_spaces[i][j].GetDir() == gqten::OUT);
    }
  }
  assert(rows_ > 0 && cols_ > 0);
#endif
//  QNT qn_site000 = hilbert_spaces[0][0].GetQNSct(0).GetQn();

  Index<QNT> index0_in({QNSector(qn0_, 1)}, IN), index0_out({QNSector(qn0_, 1)}, OUT);

  for (size_t row = 0; row < lambda_vert.rows(); row++) {
    for (size_t col = 0; col < lambda_vert.cols(); col++) {
      DTensor &the_lambda = lambda_vert({row, col});
      the_lambda = TenT({index0_in, index0_out});
      the_lambda({0, 0}) = TenElemT(1.0);
    }
  }

  for (size_t row = 0; row < lambda_horiz.rows(); row++) {
    for (size_t col = 0; col < lambda_horiz.cols(); col++) {
      DTensor &the_lambda = lambda_horiz({row, col});
      the_lambda = TenT({index0_in, index0_out});
      the_lambda({0, 0}) = TenElemT(1.0);
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

///< activates has the same size with the SquareLatticePEPS
template<typename TenElemT, typename QNT>
void SquareLatticePEPS<TenElemT, QNT>::Initial(std::vector<std::vector<size_t>> &activates) {
  for (size_t row = 0; row < rows_; row++) {
    for (size_t col = 0; col < cols_; col++) {
      TenT &the_gamma = Gamma({row, col});
      the_gamma({0, 0, 0, 0, activates[row][col]}) = 1.0;
    }
  }

  for (size_t row = 0; row < lambda_vert.rows(); row++) {
    for (size_t col = 0; col < lambda_vert.cols(); col++) {
      auto &the_lambda = lambda_vert({row, col});
      if (the_lambda.GetBlkSparDataTen().GetActualRawDataSize() == 0) {
        the_lambda({0, 0}) = TenElemT(1.0);
      }
    }
  }

  for (size_t row = 0; row < lambda_horiz.rows(); row++) {
    for (size_t col = 0; col < lambda_horiz.cols(); col++) {
      auto &the_lambda = lambda_horiz({row, col});
      if (the_lambda.GetBlkSparDataTen().GetActualRawDataSize() == 0) {
        the_lambda({0, 0}) = TenElemT(1.0);
      }
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
double SquareLatticePEPS<TenElemT, QNT>::NearestNeighborSiteProject(const TenT &gate_ten, const SiteIdx &site,
                                                                    const BondOrientation &orientation,
                                                                    const SimpleUpdateTruncatePara &trunc_para) {
  double norm;
  const size_t row = site[0], col = site[1];
  TenT tmp_ten[7];
  TenT q0, r0, q1, r1;
  TenT u, vt;
  double actual_trunc_err;
  size_t actual_D;
  TenT inv_lambda;
#ifndef NDEBUG
  auto physical_index = Gamma(row, col)->GetIndex(4);
#endif
  switch (orientation) {
    case HORIZONTAL: {
      /*                          0                                         0
       *                          |                                         |
       *                    Lam_v[rows_][cols_]                          Lam_v[rows_][cols_+1]
       *                          |                                         |
       *                          1                                         1
       *                          3                                         3
       *                          |                                         |
       *0-Lam_h[rows_][cols_]-1 0-Gamma[rows_][cols_]-2 0-Lam_h[rows_][cols_+1]-1 0-Gamma[rows_][cols_+1]-2 0-Lam_h[rows_][cols_+2]-1
       *                          |                                         |
       *                          1                                         1
       *                          0                                         0
       *                          |                                         |
       *                  Lam_v[rows_+1][cols_]                          Lam_v[rows_+1][cols_+1]
       *                          |                                         |
       *                          1                                         1
       */
      const size_t lcol = col;
      const size_t rcol = lcol + 1;
      assert(rcol < Cols());
      const SiteIdx r_site = {row, rcol};

      //Contract left site 3 lambdas
      tmp_ten[0] = Eat3SurroundLambdas_(site, RIGHT);
      QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);

      //Contract right site 3 lambdas
      tmp_ten[1] = Eat3SurroundLambdas_(r_site, LEFT);
      QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);

      Contract(&r0, {2}, lambda_horiz(row, rcol), {0}, tmp_ten + 2);
      Contract<TenElemT, QNT, true, false>(tmp_ten[2], r1, 2, 2, 1, tmp_ten[3]);
      Contract(tmp_ten + 3, {1, 3}, &gate_ten, {0, 1}, tmp_ten + 4);

      tmp_ten[4].Transpose({0, 2, 1, 3});
      lambda_horiz({row, rcol}) = TenT();
      SVD(tmp_ten + 4, 2, qn0_,
          trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
          &u, lambda_horiz(row, rcol), &vt,
          &actual_trunc_err, &actual_D);

      // hand over lambdas from q0, q1, contract u or vt, setting Gammas
      //left site
      tmp_ten[5] = QTenSplitOutLambdas_(q0, site, RIGHT, trunc_para.trunc_err);
      Gamma(site) = TenT();
      Contract<TenElemT, QNT, false, true>(tmp_ten[5], u, 0, 0, 1, Gamma(site));
      Gamma(site).Transpose({1, 2, 4, 0, 3});

      tmp_ten[6] = QTenSplitOutLambdas_(q1, r_site, LEFT, trunc_para.trunc_err);
      Gamma(r_site) = TenT();
      Contract<TenElemT, QNT, false, false>(tmp_ten[6], vt, 0, 1, 1, Gamma(r_site));
      Gamma(r_site).Transpose({4, 0, 1, 2, 3});
      norm = lambda_horiz(row, col + 1)->Normalize();
      break;
    }
    case VERTICAL: {
      /*                           0
      *                            |
      *                     Lam_v[rows_][cols_]
      *                            |
      *                            1
      *                            3
      *                            |
      *  0-Lam_h[rows_][cols_]-1 0-Gamma[rows_][cols_]-2 0-Lam_h[rows_][cols_+1]-1
      *                            |
      *                            1
      *                            0
      *                            |
      *                    Lam_v[rows_+1][cols_]
      *                            |
      *                            1
      *                            3
      *                            |
      *0-Lam_h[rows_+1][cols_]-1 0-Gamma[rows_+1][cols_]-2 0-Lam_h[rows_+1][cols_+1]-1
      *                            |
      *                            1
      *                            0
      *                            |
      *                    Lam_v[rows_+2][cols_]
      *                            |
      *                            1
      */
      assert(row + 1 < this->Rows());
      tmp_ten[0] = Eat3SurroundLambdas_(site, DOWN);
      QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);

      tmp_ten[1] = Eat3SurroundLambdas_({row + 1, col}, UP);
      QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);

      Contract<TenElemT, QNT, true, true>(r0, lambda_vert({row + 1, col}), 2, 0, 1, tmp_ten[2]);
      Contract<TenElemT, QNT, true, false>(tmp_ten[2], r1, 2, 2, 1, tmp_ten[3]);
      Contract(tmp_ten + 3, {1, 3}, &gate_ten, {0, 1}, tmp_ten + 4);

      tmp_ten[4].Transpose({0, 2, 1, 3});
      lambda_vert({row + 1, col}) = TenT();
      SVD(tmp_ten + 4, 2, qn0_,
          trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
          &u, lambda_vert(row + 1, col), &vt,
          &actual_trunc_err, &actual_D);\

      // hand over lambdas from q0, q1, contract u or vt, setting Gammas
      tmp_ten[5] = QTenSplitOutLambdas_(q0, site, DOWN, trunc_para.trunc_err);
      Gamma(site) = TenT();
      Contract<TenElemT, QNT, true, true>(tmp_ten[5], u, 3, 0, 1, Gamma(site));
      Gamma(site).Transpose({2, 4, 0, 1, 3});

      tmp_ten[6] = QTenSplitOutLambdas_(q1, {row + 1, col}, UP, trunc_para.trunc_err);
      Gamma({row + 1, col}) = TenT();
      Contract<TenElemT, QNT, true, true>(tmp_ten[6], vt, 1, 1, 1, Gamma({row + 1, col}));
      Gamma({row + 1, col}).Transpose({2, 1, 0, 4, 3});

      norm = lambda_vert(row + 1, col)->Normalize();
      break;
    }
    default: {
      std::cout << "We suppose square lattice now." << std::endl;
    }
  }
#ifndef NDEBUG
  assert(physical_index == Gamma(row, col)->GetIndex(4));
  assert(Gamma(row, col)->GetIndex(1) == lambda_vert(row + 1, col)->GetIndex(1));
  for (size_t i = 0; i < 7; i++) {
    assert(!std::isnan(*tmp_ten[i].GetBlkSparDataTen().GetActualRawDataPtr()));
  }
#endif
  return norm;
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT>
SquareLatticePEPS<TenElemT, QNT>::QTenSplitOutLambdas_(const GQTensor<TenElemT, QNT> &q, const SiteIdx &site,
                                                       const BTenPOSITION remaining_idx,
                                                       double inv_tolerance) const {
  TenT tmp_ten[2], res, inv_lambda;
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

template<typename TenElemT, typename QNT>
bool SquareLatticePEPS<TenElemT, QNT>::Dump(const std::string path) const {
  // Dump Gamma, lambda_vert, and lambda_horiz tensors one by one
  if (!gqmps2::IsPathExist(path)) { gqmps2::CreatPath(path); }
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "gamma_ten_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
      if (!Gamma.DumpTen(row, col, filename)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump Gamma tensor
      }
    }
  }

  for (size_t row = 0; row <= rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
      if (!lambda_vert.DumpTen(row, col, filename)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col <= cols_; ++col) {
      std::string filename = path + "lam_h_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
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
  if (!gqmps2::IsPathExist(path)) { gqmps2::CreatPath(path); }
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "/gamma_ten_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
      if (!Gamma.DumpTen(row, col, filename, release_mem)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump Gamma tensor
      }
    }
  }

  for (size_t row = 0; row <= rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "/lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
      if (!lambda_vert.DumpTen(row, col, filename, release_mem)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col <= cols_; ++col) {
      std::string filename = path + "/lam_h_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
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
      std::string filename = path + "/gamma_ten_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
      if (!Gamma.LoadTen(row, col, filename)) {
        std::cout << "Failed to load tensor from file: " << filename << std::endl;
        return false; // Failed to dump Gamma tensor
      }
    }
  }

  for (size_t row = 0; row <= rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "/lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
      if (!lambda_vert.LoadTen(row, col, filename)) {
        std::cout << "Failed to load tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col <= cols_; ++col) {
      std::string filename = path + "/lam_h_" + std::to_string(row) + "-" + std::to_string(col) + ".gqten";
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
      const TenT lam_left_sqrt = ElementWiseSqrt(lambda_horiz({row, col}));
      const TenT lam_right_sqrt = ElementWiseSqrt(lambda_horiz({row, col + 1}));
      const TenT lam_up_sqrt = ElementWiseSqrt(lambda_vert({row, col}));
      const TenT lam_down_sqrt = ElementWiseSqrt(lambda_vert({row + 1, col}));

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

template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::SingleSiteProject(const SquareLatticePEPS::TenT &gate_ten,
                                                           const SiteIdx &site) {
//  size_t row = site[0], col = site[1];
//  TenT eaten_lam_ten = EatSurroundLambdas_(site);
//  TenT tmp_ten[5];
  TenT tmp_ten;
  Contract<TenElemT, QNT, true, true>(Gamma(site), gate_ten, 4, 0, 1, tmp_ten);
  Gamma(site) = std::move(tmp_ten);
//  TenT u, vt;
//  DTensor *s = new DTensor();
//  SVD(tmp_ten, 1, qn0_, &u, s, &vt);
//  if (col != 0) {
//    Contract(&Gamma({row, col - 1}), 2, &u, 0, &tmp_ten[1]);
//    tmp_ten[1].Transpose({0, 1, 4, 2, 3});
//    Gamma({row, col - 1}) = std::move(tmp_ten[1]);
//  }
//  delete lambda_horiz(row, col);
//  lambda_horiz(row, col) = s;
//  Contract<TenElemT, QNT, false, false>(vt, *s, 0, 1, 1, tmp_ten[2]);
//
//  s = new DTensor();
//  u = TenT();
//  vt = TenT();
//  SVD(tmp_ten[2], 1, qn0_, &u, s, &vt);
//  if (row != this->Rows() - 1) {
//    Contract(&Gamma({row + 1, col}), 3, &u, 0, &tmp_ten[3]);
//    tmp_ten[3].Transpose({0, 1, 2, 4, 3});
//    Gamma({row + 1, col}) = std::move(tmp_ten[1]);
//  }

  return 1;
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
GQTensor<TenElemT, QNT> SquareLatticePEPS<TenElemT, QNT>::EatSurroundLambdas_(const SiteIdx &site) const {
  TenT tmp_ten[3], res;
  const size_t row = site[0], col = site[1];
  Contract<TenElemT, QNT, false, false>(Gamma(site), lambda_horiz(site), 0, 1, 1, tmp_ten[0]);
  Contract<TenElemT, QNT, false, true>(tmp_ten[0], lambda_vert({row + 1, col}), 0, 0, 1, tmp_ten[1]);
  Contract<TenElemT, QNT, false, true>(tmp_ten[1], lambda_horiz({row, col + 1}), 0, 0, 1, tmp_ten[2]);
  Contract<TenElemT, QNT, false, false>(tmp_ten[2], lambda_vert(site), 0, 1, 1, res);
  return res;
}

/**
 *
 *      |             |
 *      |             |
 * -----B-------------A--------
 *      |             |
 *      |             |
 *      |             |
 * -----C----------------------
 *      |             |
 *      |             |
 *
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param gate_ten  order of the indexes: upper-right site; upper-left site; lower-left site.
 * @param upper_site
 * @param trunc_para
 * @return
 */
template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::UpperLeftTriangleProject(const SquareLatticePEPS::TenT &gate_ten,
                                                                  const SiteIdx &left_upper_site,
                                                                  const SimpleUpdateTruncatePara &trunc_para) {
#ifndef NDEBUG
  auto physical_index = Gamma(left_upper_site).GetIndex(4);
#endif
  double norm = 1;
  size_t row = left_upper_site[0], col = left_upper_site[1];
  SiteIdx right_site = {row, col + 1};
  SiteIdx lower_site = {row + 1, col};
  TenT tmp_ten[11], q0, r0, q1, r1;
  tmp_ten[0] = Eat3SurroundLambdas_(right_site, LEFT);
  QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);
  tmp_ten[1] = Eat3SurroundLambdas_(lower_site, UP);
  QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);
  tmp_ten[2] = EatSurroundLambdas_(left_upper_site);
  Contract<TenElemT, QNT, false, false>(tmp_ten[2], r1, 2, 2, 1, tmp_ten[3]);
  Contract<TenElemT, QNT, true, true>(r0, tmp_ten[3], 2, 0, 1, tmp_ten[4]);
  Contract(tmp_ten + 4, {1, 3, 6}, &gate_ten, {0, 1, 2}, tmp_ten + 5);
  /**
   *  tmp_ten[5] is a rank-7 tensor
   *         1
   *         |
   *  2--tmp_ten[5]--0, physical index = 4,5,6, with order upper-right site->upper-left site->lower-left site.
   *        |
   *        3
   */

  tmp_ten[5].Transpose({0, 4, 5, 1, 2, 6, 3});
  TenT u1, vt1, u2, vt2;
  DTensor s1, s2;
  double trunc_err;
  size_t D;
  gqten::SVD(tmp_ten + 5, 5, tmp_ten[5].Div(),
             trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u1, &s1, &vt1, &trunc_err, &D);
  norm *= s1.Normalize();
  lambda_vert({lower_site}) = s1;
  tmp_ten[6] = QTenSplitOutLambdas_(q1, lower_site, UP, trunc_para.trunc_err);

  Gamma(lower_site) = TenT();
  Contract<TenElemT, QNT, false, false>(vt1, tmp_ten[6], 2, 1, 1, Gamma(lower_site));
  Gamma(lower_site).Transpose({4, 3, 2, 0, 1});
  Contract(&u1, {5}, &s1, {0}, &tmp_ten[7]);
  gqten::SVD(tmp_ten + 7, 2, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u2, &s2, &vt2, &trunc_err, &D);

  /**
 *       2
 *       |
 *  3---vt2--0, physical index = 1
 *      |
 *      4
 */
  norm *= s2.Normalize();
  lambda_horiz({right_site}) = s2;
  lambda_horiz({right_site}).Transpose({1, 0});
  tmp_ten[8] = QTenSplitOutLambdas_(q0, right_site, LEFT, trunc_para.trunc_err);
  Gamma(right_site) = TenT();
  Contract<TenElemT, QNT, false, true>(tmp_ten[8], u2, 0, 0, 1, Gamma(right_site));
  Gamma(right_site).Transpose({4, 0, 1, 2, 3});

  auto inv_lam = ElementWiseInv(s1, trunc_para.trunc_err);
  Contract(&vt2, {4}, &inv_lam, {0}, &tmp_ten[9]);
  inv_lam = ElementWiseInv(lambda_vert({row, col}), trunc_para.trunc_err);
  Contract<TenElemT, QNT, false, true>(tmp_ten[9], inv_lam, 2, 1, 1, tmp_ten[10]);
  inv_lam = ElementWiseInv(lambda_horiz({row, col}), trunc_para.trunc_err);
  Gamma({left_upper_site}) = TenT();
  Contract<TenElemT, QNT, false, true>(tmp_ten[10], inv_lam, 0, 1, 1, Gamma({left_upper_site}));
  Gamma({left_upper_site}).Transpose({4, 0, 1, 3, 2});
#ifndef NDEBUG
  assert(physical_index == Gamma(left_upper_site).GetIndex(4));
  assert(physical_index == Gamma(right_site).GetIndex(4));
  assert(physical_index == Gamma(lower_site).GetIndex(4));
#endif
  return norm;
}

/**
 *
 *      |             |
 *      |             |
 * -------------------A--------
 *      |             |
 *      |             |
 *      |             |
 * -----C-------------B--------
 *      |             |
 *      |             |
 * @tparam TenElemT
 * @tparam QNT
 * @param gate_ten  order of the indexes: upper-right site; lower-right site; lower-left site.
 * @param upper_site
 * @param trunc_para
 * @return
 */
template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::LowerRightTriangleProject(const SquareLatticePEPS::TenT &gate_ten,
                                                                   const SiteIdx &upper_site,
                                                                   const SimpleUpdateTruncatePara &trunc_para) {
#ifndef NDEBUG
  auto physical_index = Gamma(upper_site).GetIndex(4);
#endif
  double norm = 1;
  size_t row = upper_site[0], col = upper_site[1];
  SiteIdx left_site = {row + 1, col - 1};
  SiteIdx right_down_site = {row + 1, col};
  TenT tmp_ten[11], q0, r0, q1, r1;
  tmp_ten[0] = Eat3SurroundLambdas_(left_site, RIGHT);
  QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);
  tmp_ten[1] = Eat3SurroundLambdas_(upper_site, DOWN);
  QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);
  tmp_ten[2] = EatSurroundLambdas_(right_down_site);
  Contract<TenElemT, QNT, true, false>(r1, tmp_ten[2], 2, 4, 1, tmp_ten[3]);
  Contract<TenElemT, QNT, false, false>(tmp_ten[3], r0, 3, 2, 1, tmp_ten[4]);
  Contract(tmp_ten + 4, {3, 4, 6}, &gate_ten, {0, 1, 2}, tmp_ten + 5);
  /**
   *  tmp_ten[5] is a rank-7 tensor
   *         2
   *         |
   *  3--tmp_ten[5]--1, physical index = 4,5,6
   *        |
   *        0
   */
  tmp_ten[5].Transpose({6, 3, 0, 1, 5, 4, 2});
  TenT u1, vt1, u2, vt2;
  DTensor s1, s2;
  double trunc_err;
  size_t D;
  gqten::SVD(tmp_ten + 5, 5, tmp_ten[5].Div(),
             trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u1, &s1, &vt1, &trunc_err, &D);
  norm *= s1.Normalize();
  lambda_vert({right_down_site}) = s1;
  lambda_vert({right_down_site}).Transpose({1, 0});
  tmp_ten[6] = QTenSplitOutLambdas_(q1, upper_site, DOWN, trunc_para.trunc_err);

  Gamma(upper_site) = TenT();
  Contract<TenElemT, QNT, true, true>(tmp_ten[6], vt1, 3, 2, 1, Gamma(upper_site));
  Gamma(upper_site).Transpose({2, 3, 0, 1, 4});
  Contract(&u1, {5}, &s1, {0}, &tmp_ten[7]);
  gqten::SVD(tmp_ten + 7, 2, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u2, &s2, &vt2, &trunc_err, &D);
  /**
   *       4
   *       |
   *  0---vt2--2, physical index = 3
   *       |
   *       1
   */
  norm *= s2.Normalize();
  lambda_horiz({right_down_site}) = s2;
  tmp_ten[8] = QTenSplitOutLambdas_(q0, left_site, RIGHT, trunc_para.trunc_err);
  Gamma(left_site) = TenT();
  Contract<TenElemT, QNT, false, false>(tmp_ten[8], u2, 0, 1, 1, Gamma(left_site));
  Gamma(left_site).Transpose({1, 2, 3, 0, 4});
  auto inv_lam = ElementWiseInv(s1, trunc_para.trunc_err);
  Contract(&vt2, {4}, &inv_lam, {0}, &tmp_ten[9]);
  inv_lam = ElementWiseInv(lambda_vert({row + 2, col}), trunc_para.trunc_err);
  Contract<TenElemT, QNT, false, true>(tmp_ten[9], inv_lam, 1, 0, 1, tmp_ten[10]);
  inv_lam = ElementWiseInv(lambda_horiz({row + 1, col + 1}), trunc_para.trunc_err);
  Gamma({right_down_site}) = TenT();
  Contract<TenElemT, QNT, false, true>(tmp_ten[10], inv_lam, 0, 0, 1, Gamma({right_down_site}));
  Gamma({right_down_site}).Transpose({2, 3, 4, 1, 0});
#ifndef NDEBUG
  assert(physical_index == Gamma(upper_site).GetIndex(4));
  assert(physical_index == Gamma(left_site).GetIndex(4));
  assert(physical_index == Gamma(right_down_site).GetIndex(4));
#endif
  return norm;
}

/**
 *
 *      |             |
 *      |             |
 * -----A----------------------
 *      |             |
 *      |             |
 *      |             |
 * -----B-------------C--------
 *      |             |
 *      |             |
 * @tparam TenElemT
 * @tparam QNT
 * @param gate_ten  order of the indexes: upper-left site; lower-left site; lower-right site.
 * @param upper_site
 * @param trunc_para
 * @return
 */
template<typename TenElemT, typename QNT>
double SquareLatticePEPS<TenElemT, QNT>::LowerLeftTriangleProject(const GQTensor<TenElemT, QNT> &gate_ten,
                                                                  const gqpeps::SiteIdx &upper_left_site,
                                                                  const gqpeps::SimpleUpdateTruncatePara &trunc_para) {
  double norm = 1;
  size_t row = upper_left_site[0], col = upper_left_site[1];
  SiteIdx lower_left_site = {row + 1, col};
  SiteIdx lower_right_site = {row + 1, col + 1};
#ifndef NDEBUG
  auto index_1 = Gamma(upper_left_site).GetIndexes();
  auto index_2 = Gamma(lower_left_site).GetIndexes();
  auto index_3 = Gamma(lower_right_site).GetIndexes();
#endif
  TenT tmp_ten[11], q0, r0, q1, r1;
  tmp_ten[0] = Eat3SurroundLambdas_(upper_left_site, DOWN);
  QR(tmp_ten, 3, tmp_ten[0].Div(), &q0, &r0);
  tmp_ten[1] = Eat3SurroundLambdas_(lower_right_site, LEFT);
  QR(tmp_ten + 1, 3, tmp_ten[1].Div(), &q1, &r1);
  /**
   *       2
   *       |
   *  3---q1---1
   *      |
   *      0
   *
   */
  tmp_ten[2] = EatSurroundLambdas_(lower_left_site);
  Contract<TenElemT, QNT, true, false>(r0, tmp_ten[2], 2, 3, 1, tmp_ten[3]);
  Contract<TenElemT, QNT, true, false>(tmp_ten[3], r1, 5, 2, 1, tmp_ten[4]);
  Contract(tmp_ten + 4, {1, 2, 6}, &gate_ten, {0, 1, 2}, tmp_ten + 5);

  /**
   *  tmp_ten[5] is a rank-7 tensor
   *         0
   *         |
   *  1--tmp_ten[5]--3, physical index = 4,5,6, with order upper-left site->lower-left site->lower-right site.
   *        |
   *        2
   */
  tmp_ten[5].Transpose({0, 4, 5, 1, 2, 6, 3}); //(0,4) for upper site, (5,1,2) for left-lower site, (6,3) for right site
  TenT u1, vt1, u2, vt2;
  DTensor s1, s2;
  double trunc_err;
  size_t D;
  gqten::SVD(tmp_ten + 5, 5, tmp_ten[5].Div(),
             trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u1, &s1, &vt1, &trunc_err, &D);
  lambda_horiz({lower_right_site}) = s1;
  tmp_ten[6] = QTenSplitOutLambdas_(q1, lower_right_site, LEFT, trunc_para.trunc_err);
  /**
   *       3
   *       |
   *  0--tmp6--2    no phy idx
   *      |
   *      1
   */
  Gamma(lower_right_site) = TenT();
  Contract<TenElemT, QNT, true, true>(vt1, tmp_ten[6], 2, 0, 1, Gamma(lower_right_site));
  Gamma(lower_right_site).Transpose({0, 2, 3, 4, 1});

  Contract(&u1, {5}, &s1, {0}, &tmp_ten[7]);
  gqten::SVD(tmp_ten + 7, 2, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max,
             &u2, &s2, &vt2, &trunc_err, &D);
  /**
   *       0
   *       |
   *  2---vt2--4, physical index = 1
   *       |
   *       3
   */
#ifndef NDEBUG
  assert(vt2.GetIndex(1) == index_2.back());
#endif
  norm *= s2.Normalize();
  lambda_vert({lower_left_site}) = s2;
  tmp_ten[8] = QTenSplitOutLambdas_(q0, upper_left_site, DOWN, trunc_para.trunc_err);
/*
 *        1
 *        |
 *  2--tmp_ten[8]--0
 *        |
 *        3
 */
  Gamma(upper_left_site) = TenT();
  Contract<TenElemT, QNT, true, true>(tmp_ten[8], u2, 3, 0, 1, Gamma(upper_left_site));
  Gamma(upper_left_site).Transpose({2, 4, 0, 1, 3});

  auto inv_lam = ElementWiseInv(s1, trunc_para.trunc_err);
  Contract(&vt2, {4}, &inv_lam, {0}, &tmp_ten[9]);
  inv_lam = ElementWiseInv(lambda_vert({row + 2, col}), trunc_para.trunc_err);
  Contract<TenElemT, QNT, false, true>(tmp_ten[9], inv_lam, 3, 0, 1, tmp_ten[10]);
  inv_lam = ElementWiseInv(lambda_horiz({row + 1, col}), trunc_para.trunc_err);
  Gamma({lower_left_site}) = TenT();
  Contract<TenElemT, QNT, false, false>(tmp_ten[10], inv_lam, 3, 1, 1, Gamma({lower_left_site}));
  Gamma({lower_left_site}).Transpose({4, 0, 1, 2, 3});
#ifndef NDEBUG
  auto index_1p = Gamma(upper_left_site).GetIndexes();
  auto index_2p = Gamma(lower_left_site).GetIndexes();
  auto index_3p = Gamma(lower_right_site).GetIndexes();
  assert(index_1.back() == index_1p.back());
  assert(index_2.back() == index_2p.back());
  assert(index_3.back() == index_3p.back());
#endif
  return norm;
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT>
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
}//gqpeps

#endif //VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_IMPL_H
