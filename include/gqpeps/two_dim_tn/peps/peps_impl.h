// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-21
*
* Description: GraceQ/VMC-PEPS project. The generic PEPS class, implementation.
*/


#ifndef VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_IMPL_H
#define VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_IMPL_H

#include "gqpeps/two_dim_tn/peps/peps.h"

namespace gqpeps {
using namespace gqten;

template<typename TenElemT, typename QNT>
PEPS<TenElemT, QNT>::PEPS(const HilbertSpaces<QNT> &hilbert_spaces): rows_(hilbert_spaces.size()),
                                                                     cols_(hilbert_spaces[0].size()),
                                                                     Gamma(hilbert_spaces.size(),
                                                                           hilbert_spaces[0].size()),
                                                                     lambda_vert(hilbert_spaces.size() + 1,
                                                                                 hilbert_spaces[0].size()),
                                                                     lambda_horiz(hilbert_spaces.size(),
                                                                                  hilbert_spaces[0].size() + 1) {
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

// Helper function to generate a Hilbert space with the same local hilber space repeated
template<typename QNT>
HilbertSpaces<QNT> GenerateHilbertSpace(size_t rows, size_t cols, const Index<QNT> &local_hilbert_space) {
  HilbertSpaces<QNT> hilbert_spaces(rows, std::vector<Index<QNT>>(cols, local_hilbert_space));
  return hilbert_spaces;
}

template<typename TenElemT, typename QNT>
PEPS<TenElemT, QNT>::PEPS(const Index<QNT> &local_hilbert_space, size_t rows, size_t cols): PEPS(
    GenerateHilbertSpace(rows, cols, local_hilbert_space)) {}

template<typename TenElemT, typename QNT>
size_t PEPS<TenElemT, QNT>::GetMaxBondDimension(void) const {
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
bool PEPS<TenElemT, QNT>::IsBondDimensionEven(void) const {
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
bool PEPS<TenElemT, QNT>::operator==(const PEPS<TenElemT, QNT> &rhs) const {
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

///< activates has the same size with the PEPS
template<typename TenElemT, typename QNT>
void PEPS<TenElemT, QNT>::Initial(std::vector<std::vector<size_t>> &activates) {
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

//TODO: check
template<typename TenElemT, typename QNT>
double PEPS<TenElemT, QNT>::NearestNeighborSiteProject(const PEPS::GateT &gate, const SiteIdx &site,
                                                       const BondOrientation &direction,
                                                       const TruncatePara &trunc_para) {
  double norm;
  const size_t row = site[0], col = site[1];
  switch (direction) {
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
      const SiteIdx &l_site = site;
      const SiteIdx r_site = {row, rcol};
      TenT tmp_ten[15];

      //Contract left site 3 lambdas
      Contract(Gamma(row, lcol), {3}, lambda_vert(row, lcol), {1}, &tmp_ten[0]);
      Contract(&tmp_ten[0], {0}, lambda_horiz(row, lcol), {1}, &tmp_ten[1]);
      Contract(&tmp_ten[1], {0}, lambda_vert(row + 1, lcol), {0}, &tmp_ten[2]);

      //Contract gate left tensor
      Contract(&tmp_ten[2], {1}, &gate.first, {0}, &tmp_ten[3]);
      tmp_ten[3].Transpose({1, 2, 3, 4, 5, 0});

      TenT q, r;
      QR(&tmp_ten[3], 4, tmp_ten[3].Div(), &q, &r);
      Contract(&r, lambda_horiz(row, lcol + 1), {{2},
                                                 {0}}, &tmp_ten[4]);
      Contract(&tmp_ten[4], &gate.second, {{1},
                                           {0}}, &tmp_ten[5]);

      //Contract right site 3 lambda
      Contract(Gamma(row, rcol), lambda_vert(row, rcol), {{3},
                                                          {1}}, &tmp_ten[6]);
      Contract(&tmp_ten[6], lambda_vert(row + 1, rcol), {{1},
                                                         {0}}, &tmp_ten[7]);
      Contract(&tmp_ten[7], lambda_horiz(row, rcol + 1), {{1},
                                                          {0}}, &tmp_ten[8]);
      Contract(tmp_ten + 5, {1, 2}, tmp_ten + 8, {0,
                                                  1}, &tmp_ten[9]);

      TenT u, vt;
      DTensor s;

      double actual_trunc_err;
      size_t actual_D;
      SVD(&tmp_ten[9], 1, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max, &u, &s, &vt,
          &actual_trunc_err,
          &actual_D);
      Contract(&q, &u, {{4},
                        {0}}, &tmp_ten[10]);

      lambda_horiz({row, rcol}) = std::move(s);

      // hand over lambdas from tmp_ten[10] and vt, setting Gammas
      // left:
      TenT inv_lambda = lambda_vert({row, lcol});
      inv_lambda.ElementWiseInv(trunc_para.trunc_err);
      Contract(&inv_lambda, &tmp_ten[10], {{1},
                                           {0}}, &tmp_ten[11]);
      inv_lambda = lambda_horiz({row, lcol});
      inv_lambda.ElementWiseInv(trunc_para.trunc_err);
      Contract(&tmp_ten[11], &inv_lambda, {{1},
                                           {1}}, &tmp_ten[12]);
      inv_lambda = lambda_vert({row + 1, lcol});
      inv_lambda.ElementWiseInv(trunc_para.trunc_err);
      Gamma(l_site) = TenT();
      Contract(&tmp_ten[12], &inv_lambda, {{1},
                                           {0}}, Gamma(row, lcol));
      Gamma(row, lcol)->Transpose({3, 4, 2, 0, 1});
      //right
      inv_lambda = lambda_vert({row, rcol});
      inv_lambda.ElementWiseInv(trunc_para.trunc_err);
      Contract(&vt, &inv_lambda, {{2},
                                  {1}}, &tmp_ten[13]);
      inv_lambda = lambda_horiz({row, rcol + 1});
      inv_lambda.ElementWiseInv(trunc_para.trunc_err);
      Contract(&tmp_ten[13], &inv_lambda, {{3},
                                           {0}}, &tmp_ten[14]);
      inv_lambda = lambda_vert({row + 1, rcol});
      inv_lambda.ElementWiseInv(trunc_para.trunc_err);
      Gamma(r_site) = TenT();
      Contract(&tmp_ten[14], &inv_lambda, {{2},
                                           {0}}, &Gamma(r_site));
      Gamma(row, rcol)->Transpose({0, 4, 3, 2, 1});
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
      TenT gate_ten;
      auto physical_index = Gamma(row, col)->GetIndex(4);
      Contract(&gate.first, &gate.second, {{2},
                                           {0}}, &gate_ten);
      TenT tmp_ten[15];

      /* Plain contraction
      Contract(lambda_horiz(row, col), Gamma(row, col), {{1},
                                                         {0}}, tmp_ten);
      Contract(lambda_vert(row, col), tmp_ten, {{1},
                                                {3}}, tmp_ten + 1);
      Contract(lambda_horiz(row, col + 1), tmp_ten + 1, {{0},
                                                         {3}}, tmp_ten + 2);

      Tensor q0, r0, q1, r1;
      QR(tmp_ten + 2, 3, tmp_ten[2].Div(), &q0, &r0);

      Contract(lambda_horiz(row + 1, col), {1}, Gamma(row + 1, col), {0}, tmp_ten + 3);
      Contract(lambda_vert(row + 2, col), {0}, tmp_ten + 3, {1}, tmp_ten + 4);
      Contract(lambda_horiz(row + 1, col + 1), {0}, tmp_ten + 4, {2}, tmp_ten + 5);

      QR(tmp_ten + 5, 3, tmp_ten[5].Div(), &q1, &r1);

      Contract(&r0, {1}, lambda_vert(row + 1, col), {0}, tmp_ten + 6);
      Contract(tmp_ten + 6, {2}, &r1, {1}, tmp_ten + 7);
      Contract(tmp_ten + 7, {1, 3}, &gate_ten, {0, 2}, tmp_ten + 8);
      tmp_ten[8].Transpose({0, 2, 1, 3});

      Tensor u, vt;
      DTensor s;
      double actual_trunc_err;
      size_t actual_D;
      SVD(tmp_ten + 8, 2, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max, &u, &s, &vt, &actual_trunc_err,
          &actual_D);
      lambda_vert({row + 1, col}) = std::move(s);
      // hand over lambdas from q0, q1, contract u or vt, setting Gammas
      Tensor inv_lambda;
      inv_lambda = ElementWiseInv(lambda_vert(site), trunc_para.trunc_err);
      Contract(&inv_lambda, {1}, &q0, {1}, tmp_ten + 9);
      inv_lambda = ElementWiseInv(lambda_horiz(site), trunc_para.trunc_err);
      Contract(&inv_lambda, {1}, tmp_ten + 9, {2}, tmp_ten + 10);
      inv_lambda = ElementWiseInv(lambda_horiz({row, col + 1}), trunc_para.trunc_err);
      Contract(&inv_lambda, {0}, tmp_ten + 10, {2}, tmp_ten + 11);
      Gamma({row, col}) = Tensor();
      Contract(tmp_ten + 11, {3}, &u, {0}, Gamma(row, col));
      Gamma(row, col)->Transpose({1, 4, 0, 2, 3});

      inv_lambda = ElementWiseInv(lambda_horiz({row + 1, col + 1}), trunc_para.trunc_err);
      Contract(&inv_lambda, {0}, &q1, {0}, tmp_ten + 12);
      inv_lambda = ElementWiseInv(lambda_vert({row + 2, col}), trunc_para.trunc_err);
      Contract(&inv_lambda, {0}, tmp_ten + 12, {1}, tmp_ten + 13);
      inv_lambda = ElementWiseInv(lambda_horiz({row + 1, col}), trunc_para.trunc_err);
      Contract(&inv_lambda, {1}, tmp_ten + 13, {2}, tmp_ten + 14);
      Gamma({row + 1, col}) = Tensor();
      Contract(tmp_ten + 14, {3}, &vt, {1}, Gamma(row + 1, col));

       */
      // fast contraction

      Contract<TenElemT, QNT, true, true>(lambda_horiz({row, col}), Gamma({row, col}), 1, 0, 1, tmp_ten[0]);

      Contract<TenElemT, QNT, true, true>(lambda_vert({row, col}), tmp_ten[0], 1, 3, 1, tmp_ten[1]);

      Contract<TenElemT, QNT, false, false>(lambda_horiz({row, col + 1}), tmp_ten[1], 0, 4, 1, tmp_ten[2]);

      tmp_ten[2].Transpose({0, 1, 3, 2, 4});

      TenT q0, r0, q1, r1;
      QR(tmp_ten + 2, 3, tmp_ten[2].Div(), &q0, &r0);

      Contract<TenElemT, QNT, true, true>(lambda_horiz({row + 1, col}), Gamma({row + 1, col}), 1, 0, 1, tmp_ten[3]);
      Contract<TenElemT, QNT, false, true>(lambda_vert({row + 2, col}), tmp_ten[3], 0, 1, 1, tmp_ten[4]);
      Contract<TenElemT, QNT, false, true>(lambda_horiz({row + 1, col + 1}), tmp_ten[4], 0, 1, 1, tmp_ten[5]);
      tmp_ten[5].Transpose({0, 4, 3, 2, 1});

      QR(tmp_ten + 5, 3, tmp_ten[5].Div(), &q1, &r1);

      Contract<TenElemT, QNT, true, true>(r0, lambda_vert({row + 1, col}), 2, 0, 1, tmp_ten[6]);
      Contract<TenElemT, QNT, true, false>(tmp_ten[6], r1, 2, 2, 1, tmp_ten[7]);
      Contract(tmp_ten + 7, {1, 3}, &gate_ten, {0, 2}, tmp_ten + 8);

      tmp_ten[8].Transpose({0, 2, 1, 3});
//      tmp_ten[8].Show(2);
      TenT u, vt;
      DTensor s;
      double actual_trunc_err;
      size_t actual_D;
      SVD(tmp_ten + 8, 2, qn0_, trunc_para.trunc_err, trunc_para.D_min, trunc_para.D_max, &u, &s, &vt,
          &actual_trunc_err,
          &actual_D);
      lambda_vert({row + 1, col}) = std::move(s);
      // hand over lambdas from q0, q1, contract u or vt, setting Gammas

      TenT inv_lambda;
      inv_lambda = ElementWiseInv(lambda_vert(site), trunc_para.trunc_err);
      Contract<TenElemT, QNT, true, true>(inv_lambda, q0, 1, 1, 1, tmp_ten[9]);
      inv_lambda = ElementWiseInv(lambda_horiz(site), trunc_para.trunc_err);
      Contract<TenElemT, QNT, true, true>(inv_lambda, tmp_ten[9], 1, 1, 1, tmp_ten[10]);
      inv_lambda = ElementWiseInv(lambda_horiz({row, col + 1}), trunc_para.trunc_err);
      Contract<TenElemT, QNT, false, true>(inv_lambda, tmp_ten[10], 0, 2, 1, tmp_ten[11]);
      Gamma({row, col}) = TenT();
      Contract<TenElemT, QNT, true, true>(tmp_ten[11], u, 3, 0, 1, Gamma({row, col}));
      Gamma(row, col)->Transpose({2, 4, 0, 1, 3});

      inv_lambda = ElementWiseInv(lambda_horiz({row + 1, col + 1}), trunc_para.trunc_err);
      Contract<TenElemT, QNT, false, true>(inv_lambda, q1, 0, 0, 1, tmp_ten[12]);
      inv_lambda = ElementWiseInv(lambda_vert({row + 2, col}), trunc_para.trunc_err);
      Contract<TenElemT, QNT, false, true>(inv_lambda, tmp_ten[12], 0, 1, 1, tmp_ten[13]);
      inv_lambda = ElementWiseInv(lambda_horiz({row + 1, col}), trunc_para.trunc_err);
      Contract<TenElemT, QNT, true, true>(inv_lambda, tmp_ten[13], 1, 1, 1, tmp_ten[14]);
      Gamma({row + 1, col}) = TenT();
      Contract<TenElemT, QNT, true, true>(tmp_ten[14], vt, 1, 1, 1, Gamma({row + 1, col}));
      Gamma({row + 1, col}).Transpose({2, 1, 0, 4, 3});
#ifndef NDEBUG
      assert(physical_index == Gamma(row, col)->GetIndex(4));
      assert(physical_index == Gamma(row + 1, col)->GetIndex(4));
      assert(Gamma(row, col)->GetIndex(1) == lambda_vert(row + 1, col)->GetIndex(1));
      assert(Gamma(row + 1, col)->GetIndex(3) == lambda_vert(row + 1, col)->GetIndex(0));

      for (size_t i = 0; i < 15; i++) {
        assert(!std::isnan(*tmp_ten[i].GetBlkSparDataTen().GetActualRawDataPtr()));
      }

#endif
      norm = lambda_vert(row + 1, col)->Normalize();
      break;
    }
    default: {

      std::cout << "We suppose square lattice now." << std::endl;
    }
  }

  return norm;
}

template<typename TenElemT, typename QNT>
bool PEPS<TenElemT, QNT>::Dump(const std::string path) const {
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
bool PEPS<TenElemT, QNT>::Dump(const std::string path, bool release_mem) {
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
bool PEPS<TenElemT, QNT>::Load(const std::string path) {
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
TPS<TenElemT, QNT> PEPS<TenElemT, QNT>::ToTPS(void) const {
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

}//gqpeps

#endif //VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_IMPL_H
