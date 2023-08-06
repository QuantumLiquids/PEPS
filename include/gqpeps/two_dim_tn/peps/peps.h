// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: GraceQ/VMC-PEPS project. The generic PEPS class.
*/


#ifndef VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_H
#define VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/framework/ten_matrix.h"
#include "gqmps2/utilities.h"             //CreatPath
#include "gqpeps/consts.h"              //kPepsPath
#include "gqpeps/two_dim_tn/tps/tps.h"  //ToTPS()
#include "gqpeps/basic.h"               //BondDirection, TruncatePara

namespace gqpeps {
using namespace gqten;

template<typename TenElemT, typename QNT>
using Gate = std::pair<GQTensor<TenElemT, QNT>, GQTensor<TenElemT, QNT>>;

template<typename QNT>
using HilbertSpaces = std::vector<std::vector<Index<QNT>>>;
//Inner vector indices correspond to column indices
//Direction out


/**
 *           3
 *           |
 *   0--Gamma[rows_][cols_]--2,   also contain physical index 4
 *           |
 *           1
 *
 *
 *   0--Lambda[rows_][cols_]--1
 *
 *
 *       0
 *       |
 *   Lambda[rows_][cols_]
 *       |
 *       1
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class PEPS {
 public:
  using TenT = GQTensor<TenElemT, QNT>;
  using DTensor = GQTensor<GQTEN_Double, QNT>;
  using GateT = Gate<TenElemT, QNT>;

//  // Constructor with size
//  PEPS(size_t rows, size_t cols) : rows_(rows), cols_(cols), Gamma(rows, cols), lambda_vert(rows + 1, cols),
//                                   lambda_horiz(rows, cols + 1) {}

  PEPS(const HilbertSpaces<QNT> &hilbert_spaces);
  PEPS(const Index<QNT> &local_hilbert_space, size_t rows, size_t cols);

  // Copy constructor
  PEPS(const PEPS<TenElemT, QNT> &rhs)
      : rows_(rhs.rows_), cols_(rhs.cols_), Gamma(rhs.Gamma), lambda_vert(rhs.lambda_vert),
        lambda_horiz(rhs.lambda_horiz) {}

  // Move constructor
  PEPS(PEPS<TenElemT, QNT> &&rhs) noexcept
      : rows_(rhs.rows_), cols_(rhs.cols_), Gamma(std::move(rhs.Gamma)), lambda_vert(std::move(rhs.lambda_vert)),
        lambda_horiz(std::move(rhs.lambda_horiz)) {}

  // Assignment operator
  PEPS<TenElemT, QNT> &operator=(const PEPS<TenElemT, QNT> &rhs) {
    if (this != &rhs) {
      rows_ = rhs.rows_;
      cols_ = rhs.cols_;
      Gamma = rhs.Gamma;
      lambda_vert = rhs.lambda_vert;
      lambda_horiz = rhs.lambda_horiz;
    }
    return *this;
  }

  // Move assignment operator
  PEPS<TenElemT, QNT> &operator=(PEPS<TenElemT, QNT> &&rhs) noexcept {
    if (this != &rhs) {
      rows_ = rhs.rows_;
      cols_ = rhs.cols_;
      Gamma = std::move(rhs.Gamma);
      lambda_vert = std::move(rhs.lambda_vert);
      lambda_horiz = std::move(rhs.lambda_horiz);
    }
    return *this;
  }

  bool operator==(const PEPS<TenElemT, QNT> &rhs) const;

  bool operator!=(const PEPS<TenElemT, QNT> &rhs) const {
    return !(*this == rhs);
  }

  void Initial(std::vector<std::vector<size_t>> &activates);

  // Function to get the number of rows in the PEPS
  size_t Rows(void) const {
    return rows_;
  }

  // Function to get the number of columns in the PEPS
  size_t Cols(void) const {
    return cols_;
  }

  size_t GetMaxBondDimension(void) const;
  // if the bond dimension of each lambda is the same, except boundary gamma
  bool IsBondDimensionEven(void) const;

  double NearestNeighborSiteProject(
      const GateT &gate,
      const SiteIdx &site,
      const BondDirection &direction,
      const TruncatePara &trunc_para
  );

  bool Dump(const std::string path = kPepsPath) const;
  bool Dump(const std::string path = kPepsPath, bool release_mem = false);
  bool Load(const std::string path = kPepsPath);

  TPS<TenElemT, QNT> ToTPS(void) const;

  TenMatrix<TenT> Gamma; // The rank-5 projection tensors;
  TenMatrix<DTensor> lambda_vert; // vertical singular value tensors;
  TenMatrix<DTensor> lambda_horiz; // horizontal singular value tensors;

 private:
  static const QNT qn0_;

  size_t rows_; // Number of rows in the PEPS
  size_t cols_; // Number of columns in the PEPS
};

template<typename TenElemT, typename QNT>
const QNT PEPS<TenElemT, QNT>::qn0_ = QNT::Zero();

}

#include "gqpeps/two_dim_tn/peps/peps_impl.h"

#endif //VMC_PEPS_TWO_DIM_TN_PEPS_PEPS_H
