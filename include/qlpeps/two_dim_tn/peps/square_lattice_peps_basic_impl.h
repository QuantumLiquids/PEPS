// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-21
*
* Description: QuantumLiquids/PEPS project. Square Lattice PEPS implementation (Basic).
*/


#ifndef QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_BASIC_IMPL_H
#define QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_BASIC_IMPL_H

#include "qlmps/utilities.h"       //mock_qlten::SVD
#include "qlpeps/utility/filesystem_utils.h"

namespace qlpeps {
using qlmps::mock_qlten::SVD;

template<typename TenElemT, typename QNT>
SquareLatticePEPS<TenElemT, QNT>::
SquareLatticePEPS(const HilbertSpaces<QNT> &hilbert_spaces, BoundaryCondition bc):
    rows_(hilbert_spaces.size()),
    cols_(hilbert_spaces[0].size()),
    boundary_condition_(bc),
    Gamma(hilbert_spaces.size(), hilbert_spaces[0].size()),
    lambda_vert(bc == BoundaryCondition::Open ? hilbert_spaces.size() + 1 : hilbert_spaces.size(), hilbert_spaces[0].size()),
    lambda_horiz(hilbert_spaces.size(), bc == BoundaryCondition::Open ? hilbert_spaces[0].size() + 1 : hilbert_spaces[0].size()) {
#ifndef NDEBUG
//  for (size_t i = 0; i < hilbert_spaces.size(); i++) {
//    for (size_t j = 0; j < hilbert_spaces[0].size(); j++) {
//      assert(hilbert_spaces[i][j].GetDir() == qlten::OUT);
//    }
//  }
  assert(rows_ > 0 && cols_ > 0);
#endif
//  QNT qn_site000 = hilbert_spaces[0][0].GetQNSct(0).GetQn();

  Index<QNT> index0_in({QNSector(qn0_, 1)}, IN), index0_out({QNSector(qn0_, 1)}, OUT);

  for (size_t row = 0; row < lambda_vert.rows(); row++) {
    for (size_t col = 0; col < lambda_vert.cols(); col++) {
      DTenT &the_lambda = lambda_vert({row, col});
      the_lambda = DTenT({index0_in, index0_out});
      the_lambda({0, 0}) = (1.0);
    }
  }

  for (size_t row = 0; row < lambda_horiz.rows(); row++) {
    for (size_t col = 0; col < lambda_horiz.cols(); col++) {
      DTenT &the_lambda = lambda_horiz({row, col});
      the_lambda = DTenT({index0_in, index0_out});
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
SquareLatticePEPS<TenElemT, QNT>::SquareLatticePEPS(const Index<QNT> &local_hilbert_space, size_t rows, size_t cols, BoundaryCondition bc)
    : SquareLatticePEPS(GenerateHilbertSpace(rows, cols, local_hilbert_space), bc) {}

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
  Index<QNT> virtual_index0_in({QNSector(qn0_, 1)}, IN),
      virtual_index0_out({QNSector(qn0_, 1)}, OUT);
  // The lambda tensors surrounding the PEPS
  DTenT surrounding_lam = DTenT({virtual_index0_in, virtual_index0_out});
  surrounding_lam({0, 0}) = 1.0;

  // Initialize Lambdas
  // 1. Upper vertical (Row 0 inputs) - Always needed
  for (size_t col = 0; col < cols_; col++) {
    lambda_vert({0, col}) = surrounding_lam;
  }

  // 2. Lower vertical - Only for OBC
  if (boundary_condition_ == BoundaryCondition::Open) {
    for (size_t col = 0; col < cols_; col++) {
      lambda_vert({rows_, col}) = surrounding_lam;
    }
  }

  // 3. Left horizontal (Col 0 inputs) - Always needed
  for (size_t row = 0; row < rows_; row++) {
    lambda_horiz({row, 0}) = surrounding_lam;
  }

  // 4. Right horizontal - Only for OBC
  if (boundary_condition_ == BoundaryCondition::Open) {
    for (size_t row = 0; row < rows_; row++) {
      lambda_horiz({row, cols_}) = surrounding_lam;
    }
  }

  // 5. Initialize internal horizontal lambdas to trivial (needed for logic below)
  // For Comb strategy, all rows except the last one have trivial horizontal bonds.
  // The last row's horizontal lambdas are derived (except the first one), so we don't pre-fill them
  // to ensure we don't mask logic errors with trivial tensors.
  for (size_t row = 0; row < rows_ - 1; row++) {
    for (size_t col = 0; col < cols_; col++) {
      lambda_horiz({row, col}) = surrounding_lam;
    }
  }

  // --- Row 0 Construction ---
  for (size_t col = 0; col < cols_; col++) {
    const size_t row = 0;
    Index<QNT> index0 = lambda_horiz({row, col}).GetIndex(0);
    Index<QNT> index2 = virtual_index0_out; // Default East to 0 (Comb strategy)
    Index<QNT> index3 = lambda_vert({row, col}).GetIndex(0);
    Index<QNT> phy_idx = Gamma({row, col}).GetIndex(4);

    QNT index1_qn = index0.GetQNSct(0).GetQn()  // in
        + index3.GetQNSct(0).GetQn()            // in
        - index2.GetQNSct(0).GetQn();           // out

    if (phy_idx.GetDir() == OUT) {
      index1_qn += -phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
    } else {
      index1_qn += phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
    }

    Index<QNT> index1({QNSector(index1_qn, 1)}, OUT);
    Gamma({row, col}) = TenT({index0, index1, index2, index3, phy_idx});
    Gamma({row, col})({0, 0, 0, 0, activates[row][col]}) = 1.0;
  }

  // --- Middle Rows Construction ---
  for (size_t row = 1; row < rows_ - 1; row++) {
    // set a layer of lambda_vert
    for (size_t col = 0; col < cols_; col++) {
      lambda_vert({row, col}) = DTenT({InverseIndex(Gamma({row - 1, col}).GetIndex(1)),
                                       Gamma({row - 1, col}).GetIndex(1)});
      lambda_vert({row, col})({0, 0}) = 1.0;
    }
    // Set a layer of gamma tensors
    for (size_t col = 0; col < cols_; col++) {
      Index<QNT> index0 = lambda_horiz({row, col}).GetIndex(0);
      Index<QNT> index2 = virtual_index0_out; // Default East to 0
      Index<QNT> index3 = lambda_vert({row, col}).GetIndex(0);
      Index<QNT> phy_idx = Gamma({row, col}).GetIndex(4);

      QNT index1_qn = index0.GetQNSct(0).GetQn()   // in
          + index3.GetQNSct(0).GetQn()             // in
          - index2.GetQNSct(0).GetQn();            // out

      if (phy_idx.GetDir() == OUT) {
        index1_qn += -phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
      } else {
        index1_qn += phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
      }

      Index<QNT> index1({QNSector(index1_qn, 1)}, OUT);
      Gamma({row, col}) = TenT({index0, index1, index2, index3, phy_idx});
      Gamma({row, col})({0, 0, 0, 0, activates[row][col]}) = 1.0;
    }
  }

  // --- Lambda Vert between (Rows-2) and (Rows-1) ---
  // This is the input for the last row
  if (rows_ > 1) { // Guard for 1-row case
    size_t row = rows_ - 1;
    for (size_t col = 0; col < cols_; col++) {
      lambda_vert({row, col}) = DTenT({InverseIndex(Gamma({row - 1, col}).GetIndex(1)),
                                       Gamma({row - 1, col}).GetIndex(1)});
      lambda_vert({row, col})({0, 0}) = 1.0;
    }
  }

  // --- Last Layer (Bottom Row) Construction ---
  for (size_t col = 0; col < cols_; col++) {
    const size_t row = rows_ - 1;
    Index<QNT> index0 = lambda_horiz({row, col}).GetIndex(0);
    Index<QNT> index3 = lambda_vert({row, col}).GetIndex(0);
    Index<QNT> phy_idx = Gamma({row, col}).GetIndex(4);

    // Determine South Index (index1)
    Index<QNT> index1 = GetLambdaVertSouth(row, col).GetIndex(1);

    // Determine East Index (index2)
    QNT index2_qn = index0.GetQNSct(0).GetQn()  // in
        + index3.GetQNSct(0).GetQn()            // in
        - index1.GetQNSct(0).GetQn();           // out

    if (phy_idx.GetDir() == OUT) {
      index2_qn += -phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
    } else {
      index2_qn += phy_idx.GetQNSctFromActualCoor(activates[row][col]).GetQn();
    }

    // PBC Closure for Last Column
    if (boundary_condition_ == BoundaryCondition::Periodic && col == cols_ - 1) {
      index2_qn = qn0_;
    }

    Index<QNT> index2({QNSector(index2_qn, 1)}, OUT);
    Gamma({row, col}) = TenT({index0, index1, index2, index3, phy_idx});
    Gamma({row, col})({0, 0, 0, 0, activates[row][col]}) = 1.0;

    // Set next horizontal lambda
    if (col < cols_ - 1) {
      lambda_horiz({row, col + 1}) = DTenT({InverseIndex(index2), index2});
      lambda_horiz({row, col + 1})({0, 0}) = 1.0;
    } else if (boundary_condition_ == BoundaryCondition::Open) {
      // Only set the boundary lambda in OBC
      lambda_horiz({row, col + 1}) = DTenT({InverseIndex(index2), index2});
      lambda_horiz({row, col + 1})({0, 0}) = 1.0;
    }
  }

  // General warning for the whole system (legacy check)
  if constexpr (Index<QNT>::IsFermionic()) {
    if (boundary_condition_ == BoundaryCondition::Open) {
      if (lambda_horiz({rows_ - 1, cols_}).GetIndex(0).GetQNSct(0).GetQn().IsFermionParityOdd()) {
        std::cout << "warning : the direct product PEPS has odd fermion parity!" << std::endl;
      }
    } else if (boundary_condition_ == BoundaryCondition::Periodic) {
      if (Gamma({rows_ - 1, cols_-1}).Div().IsFermionParityOdd()) {
        std::cout << "warning : the direct product PEPS has odd fermion parity!" << std::endl;
      }
    }
  }
}

template<typename TenElemT, typename QNT>
std::pair<size_t, size_t> SquareLatticePEPS<TenElemT, QNT>::GetMinMaxBondDim(void) const {
  if (lambda_vert.size() == 0 && lambda_horiz.size() == 0) {
    return {0, 0};
  }
  
  size_t dmax(0);
  // Initialize dmin with the first available element if possible, otherwise max value?
  // Safety: lambda_vert usually has elements.
  size_t dmin = lambda_vert.size() > 0 ? lambda_vert({0, 0}).GetShape()[0] : 
                (lambda_horiz.size() > 0 ? lambda_horiz({0, 0}).GetShape()[0] : 0);

  for (const auto &lambda : lambda_vert) {
    size_t d = lambda.GetShape()[0];
    dmax = std::max(dmax, d);
    dmin = std::min(dmin, d);
  }

  for (const auto &lambda : lambda_horiz) {
    size_t d = lambda.GetShape()[0];
    dmax = std::max(dmax, d);
    dmin = std::min(dmin, d);
  }
  return std::pair<size_t, size_t>(dmin, dmax);
}

template<typename TenElemT, typename QNT>
size_t SquareLatticePEPS<TenElemT, QNT>::GetMaxBondDim(void) const {
  size_t dmax = 0;
  for (const auto &lambda : lambda_vert) {
    size_t d = lambda.GetShape()[0];
    dmax = std::max(dmax, d);
  }
  for (const auto &lambda : lambda_horiz) {
    size_t d = lambda.GetShape()[0];
    dmax = std::max(dmax, d);
  }
  return dmax;
}

template<typename TenElemT, typename QNT>
bool SquareLatticePEPS<TenElemT, QNT>::IsBondDimensionUniform(void) const {
  if (rows_ <= 1 || cols_ <= 1) {
    // Too small to distinguish bulk/boundary meaningfully for this check, or just return true.
    // If 1x1, true.
    return true;
  }
  size_t d = lambda_vert({1, 0}).GetShape()[0]; // work for both PBC and OBC.

  if (boundary_condition_ == BoundaryCondition::Open) {
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
  } else {
    for (auto& lambda : lambda_vert) {
      if ( d != lambda.GetShape()[0] ) {
        return false;
      }
    }
    for (auto& lambda : lambda_horiz) {
      if ( d != lambda.GetShape()[0] ) {
        return false;
      }
    }
  }

  return true;
}

template<typename TenElemT, typename QNT>
typename SquareLatticePEPS<TenElemT, QNT>::RealT SquareLatticePEPS<TenElemT, QNT>::NormalizeAllTensor() {
  RealT norm(1.0);
  for (auto &gamma : Gamma) {
    norm *= gamma.QuasiNormalize();
  }

  for (auto &lambda : lambda_vert) {
    norm *= lambda.QuasiNormalize();
  }
  for (auto &lambda : lambda_horiz) { 
    norm *= lambda.QuasiNormalize();
  }
  return norm;
}

template<typename TenElemT, typename QNT>
bool SquareLatticePEPS<TenElemT, QNT>::operator==(const SquareLatticePEPS<TenElemT, QNT> &rhs) const {
  // Check if the number of rows and columns are the same
  if (rows_ != rhs.rows_ || cols_ != rhs.cols_ || boundary_condition_ != rhs.boundary_condition_) {
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
  for (size_t row = 0; row < lambda_vert.rows(); ++row) {
    for (size_t col = 0; col < lambda_vert.cols(); ++col) {
      if (lambda_vert({row, col}) != rhs.lambda_vert({row, col})) {
        return false;
      }
    }
  }

  // Check if lambda_horiz elements are equal
  for (size_t row = 0; row < lambda_horiz.rows(); ++row) {
    for (size_t col = 0; col < lambda_horiz.cols(); ++col) {
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
  EnsureDirectoryExists(path);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "gamma_ten_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!Gamma.DumpTen(row, col, filename)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump Gamma tensor
      }
    }
  }

  for (size_t row = 0; row < lambda_vert.rows(); ++row) {
    for (size_t col = 0; col < lambda_vert.cols(); ++col) {
      std::string filename = path + "lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_vert.DumpTen(row, col, filename)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < lambda_horiz.rows(); ++row) {
    for (size_t col = 0; col < lambda_horiz.cols(); ++col) {
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
  EnsureDirectoryExists(path);
  for (size_t row = 0; row < rows_; ++row) {
    for (size_t col = 0; col < cols_; ++col) {
      std::string filename = path + "/gamma_ten_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!Gamma.DumpTen(row, col, filename, release_mem)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump Gamma tensor
      }
    }
  }

  for (size_t row = 0; row < lambda_vert.rows(); ++row) {
    for (size_t col = 0; col < lambda_vert.cols(); ++col) {
      std::string filename = path + "/lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_vert.DumpTen(row, col, filename, release_mem)) {
        std::cout << "Failed to dump tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < lambda_horiz.rows(); ++row) {
    for (size_t col = 0; col < lambda_horiz.cols(); ++col) {
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

  for (size_t row = 0; row < lambda_vert.rows(); ++row) {
    for (size_t col = 0; col < lambda_vert.cols(); ++col) {
      std::string filename = path + "/lam_v_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_vert.LoadTen(row, col, filename)) {
        std::cout << "Failed to load tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_vert tensor
      }
    }
  }

  for (size_t row = 0; row < lambda_horiz.rows(); ++row) {
    for (size_t col = 0; col < lambda_horiz.cols(); ++col) {
      std::string filename = path + "/lam_h_" + std::to_string(row) + "-" + std::to_string(col) + ".qlten";
      if (!lambda_horiz.LoadTen(row, col, filename)) {
        std::cout << "Failed to load tensor from file: " << filename << std::endl;
        return false; // Failed to dump lambda_horiz tensor
      }
    }
  }

  return true; // Successfully loaded all tensors
}

template<typename ElemT, typename QNT>
QLTensor<ElemT, QNT> SquareRootDiagMat(
    const QLTensor<ElemT, QNT> &positive_diag_mat
) {
  if (!QLTensor<ElemT, QNT>::IsFermionic() || positive_diag_mat.GetIndex(0).GetDir() == IN) {
    QLTensor<ElemT, QNT> sqrt = positive_diag_mat;
    for (size_t i = 0; i < sqrt.GetShape()[0]; i++) {
      ElemT elem = sqrt({i, i});
      if (elem > 0) {
        sqrt({i, i}) = std::sqrt(elem);
      } else if (elem < 0) {
        std::cout << "error: trying to find square root of " << std::scientific << elem << std::endl;
        exit(1);
      }
    }
    return sqrt;
  } else {
    std::cout << "Square Root for fermion tensor with first index OUT is not well defined." << std::endl;
    exit(1);
  }
}

template<typename TenElemT, typename QNT>
IndexVec<QNT> SquareLatticePEPS<TenElemT, QNT>::GatherAllIndices() const {
  IndexVec<QNT> index_vec;
  index_vec.reserve(rows_ * cols_ * 9); // Approximation
  for (auto &gamma : Gamma) {
    for (size_t i = 0; i < gamma.Rank(); i++) {
      index_vec.push_back(gamma.GetIndex(i));
    }
  }
  for (auto &lambda : lambda_vert) {
    for (size_t i = 0; i < lambda.Rank(); i++) {
      index_vec.push_back(lambda.GetIndex(i));
    }
  }
  for (auto &lambda : lambda_horiz) {
    for (size_t i = 0; i < lambda.Rank(); i++) {
      index_vec.push_back(lambda.GetIndex(i));
    }
  }
  return index_vec;
}

/**
 * Assumption here: two ends of lambda have different directions.
 */
template<typename TenElemT, typename QNT>
void SquareLatticePEPS<TenElemT, QNT>::RegularizeIndexDir() {
  if (boundary_condition_ == BoundaryCondition::Open) { // surrounding indices should have expected directions.
    for (size_t col = 0; col < this->cols_; col++) {
      assert(lambda_vert({0, col}).GetIndex(0).GetDir() == IN);
      assert(lambda_vert({rows_, col}).GetIndex(0).GetDir() == IN);
    }
    for (size_t row = 0; row < this->rows_; row++) {
      assert(lambda_horiz({row, 0}).GetIndex(0).GetDir() == IN);
      assert(lambda_horiz({row, cols_}).GetIndex(0).GetDir() == IN);
    }
  }

  //fix for internal lambdas
  // range:
  // OBC: 1 to rows-1 (internal only)
  // PBC: 0 to rows-1 (all are internal)
  size_t start_row = (boundary_condition_ == BoundaryCondition::Open) ? 1 : 0;

  for (size_t row = start_row; row < this->rows_; row++) {
    for (size_t col = 0; col < this->cols_; col++) {
      DTenT& lam = lambda_vert({row, col});
      if (lam.GetIndex(0).GetDir() != IN) {
        DTenT u, vt;
        DTenT s;
        qlmps::mock_qlten::SVD<RealT, QNT>(&lam, 1, qn0_, &u, &s, &vt);
        lam = std::move(s);
        TenT tmp0, tmp1;

        // Update Gamma Above (North side of lambda)
        // OBC: row-1. PBC: (row-1 + rows) % rows
        size_t row_above = (row + rows_ - 1) % rows_;

        Contract(&Gamma({row_above, col}), {1}, &u, {0}, &tmp0);
        Gamma({row_above, col}) = tmp0;
        Gamma({row_above, col}).Transpose({0, 4, 1, 2, 3});

        Contract(&vt, {1}, &Gamma({row, col}), {3}, &tmp1);
        Gamma({row, col}) = tmp1;
        Gamma({row, col}).Transpose({1, 2, 3, 0, 4});
      }
    }
  }

  size_t start_col = (boundary_condition_ == BoundaryCondition::Open) ? 1 : 0;
  for (size_t row = 0; row < this->rows_; row++) {
    for (size_t col = start_col; col < this->cols_; col++) {
      DTenT& lam = lambda_horiz({row, col});
      if (lam.GetIndex(0).GetDir() != IN) {
        DTenT u, vt;
        DTenT s;
        qlmps::mock_qlten::SVD(&lam, 1, qn0_, &u, &s, &vt);
        lam = std::move(s);
        TenT tmp0, tmp1;

        // Update Gamma Left (West side of lambda)
        // OBC: col-1. PBC: (col-1 + cols) % cols
        size_t col_left = (col + cols_ - 1) % cols_;

        Contract(&Gamma({row, col_left}), {2}, &u, {0}, &tmp0);
        Gamma({row, col_left}) = tmp0;
        Gamma({row, col_left}).Transpose({0, 1, 4, 2, 3});

        Contract(&vt, {1}, &Gamma({row, col}), {0}, &tmp1);
        Gamma({row, col}) = tmp1;
      }
    }
  }

}

template<typename TenElemT, typename QNT>
TPS<TenElemT, QNT>
SquareLatticePEPS<TenElemT, QNT>::ToTPS() const {
  auto tps = TPS<TenElemT, QNT>(rows_, cols_);
  SquareLatticePEPS peps_copy = (*this);
  if constexpr (TenT::IsFermionic()) {
    peps_copy.RegularizeIndexDir();
  }
  for (size_t row = 0; row < rows_; row++) {
    for (size_t col = 0; col < cols_; col++) {
      tps.alloc(row, col);
      // PBC Support: Use Accessors
      const DTenT lam_left_sqrt = SquareRootDiagMat(peps_copy.GetLambdaHorizWest(row, col));
      const DTenT lam_right_sqrt = SquareRootDiagMat(peps_copy.GetLambdaHorizEast(row, col));
      const DTenT lam_up_sqrt = SquareRootDiagMat(peps_copy.GetLambdaVertNorth(row, col));
      const DTenT lam_down_sqrt = SquareRootDiagMat(peps_copy.GetLambdaVertSouth(row, col));

      TenT tmp[3];
      Contract<TenElemT, QNT, true, true>(lam_up_sqrt, peps_copy.Gamma({row, col}), 1, 3, 1, tmp[0]);
      Contract<TenElemT, QNT, false, true>(lam_right_sqrt, tmp[0], 0, 4, 1, tmp[1]);
      Contract<TenElemT, QNT, false, true>(lam_down_sqrt, tmp[1], 0, 4, 1, tmp[2]);
      Contract<TenElemT, QNT, true, true>(lam_left_sqrt, tmp[2], 1, 4, 1, tps({row, col}));
#ifndef NDEBUG
      auto physical_index = peps_copy.Gamma(row, col)->GetIndex(4);
      assert(physical_index == tps(row, col)->GetIndex(4));
#endif
    }
  }
  return tps;
}

template<typename TenElemT, typename QNT>
SquareLatticePEPS<TenElemT, QNT>::operator TPS<TenElemT, QNT>() const {
  return ToTPS();
}

/**
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param site
 * @return physical index 0, auxiliary indexes follow original order
 *
 * res:
 *          4
 *          |
 *          |
 *    1-----T------3   and physical idx = 0
 *          |
 *          |
 *          2
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> SquareLatticePEPS<TenElemT, QNT>::EatSurroundLambdas_(const SiteIdx &site) const {
  TenT tmp_ten[3], res;
  const size_t row = site[0], col = site[1];
  Contract<TenElemT, QNT, false, false>(Gamma(site), GetLambdaHorizWest(row, col), 0, 1, 1, tmp_ten[0]);
  Contract<TenElemT, QNT, false, true>(tmp_ten[0], GetLambdaVertSouth(row, col), 0, 0, 1, tmp_ten[1]);
  Contract<TenElemT, QNT, false, true>(tmp_ten[1], GetLambdaHorizEast(row, col), 0, 0, 1, tmp_ten[2]);
  Contract<TenElemT, QNT, false, false>(tmp_ten[2], GetLambdaVertNorth(row, col), 0, 1, 1, res);
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
      Contract<TenElemT, QNT, true, false>(Gamma({row, col}), GetLambdaVertNorth(row, col), 3, 1, 1, tmp_ten[0]);
      Contract<TenElemT, QNT, false, false>(tmp_ten[0], GetLambdaHorizWest(row, col), 1, 1, 1, tmp_ten[1]);
      Contract<TenElemT, QNT, false, true>(tmp_ten[1], GetLambdaVertSouth(row, col), 0, 0, 1, res);
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
      Contract<TenElemT, QNT, false, false>(Gamma({row, col}), GetLambdaVertNorth(row, col), 3, 1, 1, tmp_ten[0]);
      Contract<TenElemT, QNT, false, true>(tmp_ten[0], GetLambdaHorizEast(row, col), 3, 0, 1, tmp_ten[1]);
      Contract<TenElemT, QNT, false, false>(GetLambdaVertSouth(row, col), tmp_ten[1], 0, 3, 1, res);
      return res;
      /*      2
       *      |
       *  4--res--1, physical index = 3
       *      |
       *      0
       */
    }
    case DOWN: {
      Contract<TenElemT, QNT, true, true>(GetLambdaHorizWest(row, col), Gamma({row, col}), 1, 0, 1, tmp_ten[0]);
      Contract<TenElemT, QNT, true, true>(GetLambdaVertNorth(row, col), tmp_ten[0], 1, 3, 1, tmp_ten[1]);
      Contract<TenElemT, QNT, false, false>(GetLambdaHorizEast(row, col), tmp_ten[1], 0, 4, 1, res);
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
      Contract<TenElemT, QNT, true, true>(GetLambdaHorizWest(row, col), Gamma({row, col}), 1, 0, 1, tmp_ten[0]);
      Contract<TenElemT, QNT, false, true>(GetLambdaVertSouth(row, col), tmp_ten[0], 0, 1, 1, tmp_ten[1]);
      Contract<TenElemT, QNT, false, true>(GetLambdaHorizEast(row, col), tmp_ten[1], 0, 1, 1, res);
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
                                                       typename SquareLatticePEPS<TenElemT, QNT>::RealT inv_tolerance) const {
  TenT tmp_ten[2], res;
  DTenT inv_lambda;
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
      inv_lambda = DiagMatInv(GetLambdaVertNorth(row, col), inv_tolerance);
      Contract<TenElemT, QNT, false, false>(q, inv_lambda, 0, 1, 1, tmp_ten[0]);
      inv_lambda = DiagMatInv(GetLambdaHorizWest(row, col), inv_tolerance);
      Contract<TenElemT, QNT, false, false>(tmp_ten[0], inv_lambda, 0, 1, 1, tmp_ten[1]);
      inv_lambda = DiagMatInv(GetLambdaVertSouth(row, col), inv_tolerance);
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
      inv_lambda = DiagMatInv(GetLambdaVertSouth(row, col), inv_tolerance);
      Contract<TenElemT, QNT, false, false>(q, inv_lambda, 0, 0, 1, tmp_ten[0]);
      inv_lambda = DiagMatInv(GetLambdaHorizEast(row, col), inv_tolerance);
      Contract<TenElemT, QNT, false, false>(tmp_ten[0], inv_lambda, 0, 0, 1, tmp_ten[1]);
      inv_lambda = DiagMatInv(GetLambdaVertNorth(row, col), inv_tolerance);
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
      inv_lambda = DiagMatInv(GetLambdaVertNorth(row, col), inv_tolerance);
      Contract<TenElemT, QNT, true, true>(inv_lambda, q, 1, 1, 1, tmp_ten[0]);
      inv_lambda = DiagMatInv(GetLambdaHorizWest(row, col), inv_tolerance);
      Contract<TenElemT, QNT, true, true>(inv_lambda, tmp_ten[0], 1, 1, 1, tmp_ten[1]);
      inv_lambda = DiagMatInv(GetLambdaHorizEast(row, col), inv_tolerance);
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
      inv_lambda = DiagMatInv(GetLambdaHorizEast(row, col), inv_tolerance);
      Contract<TenElemT, QNT, false, true>(inv_lambda, q, 0, 0, 1, tmp_ten[0]);
      inv_lambda = DiagMatInv(GetLambdaVertSouth(row, col), inv_tolerance);
      Contract<TenElemT, QNT, false, true>(inv_lambda, tmp_ten[0], 0, 1, 1, tmp_ten[1]);
      inv_lambda = DiagMatInv(GetLambdaHorizWest(row, col), inv_tolerance);
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

#endif //QLPEPS_TWO_DIM_TN_PEPS_SQUARE_LATTICE_PEPS_BASIC_IMPL_H
