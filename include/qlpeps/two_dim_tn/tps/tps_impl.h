// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: QuantumLiquids/PEPS project. The generic tensor product state (TPS) class, implementation.
*/

#ifndef QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_IMPL_H
#define QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_IMPL_H

#include "qlpeps/two_dim_tn/tps/tps.h"
#include "qlpeps/utility/filesystem_utils.h"

namespace qlpeps {

template<typename TenElemT, typename QNT>
TensorNetwork2D<TenElemT, QNT> TPS<TenElemT, QNT>::Project(const Configuration &config) const {
  const size_t rows = this->rows();
  const size_t cols = this->cols();
  TensorNetwork2D<TenElemT, QNT> tn(rows, cols, boundary_condition_);

  //TODO: optimize
  Index<QNT> physical_index; // We suppose each site has the same hilbert space
  physical_index = (*this)(0, 0)->GetIndex(4);
  auto physical_index_inv = InverseIndex(physical_index);
  const size_t type_of_config = physical_index.dim();
  TenT project_tens[type_of_config];
  for (size_t i = 0; i < type_of_config; i++) {
    project_tens[i] = TenT({physical_index_inv});
    project_tens[i]({i}) = TenElemT(1.0);
  }

  for (size_t row = 0; row < this->rows(); row++) {
    for (size_t col = 0; col < this->cols(); col++) {
      tn(row, col)->alloc();
      size_t local_config = config({row, col});
      Contract((*this)(row, col), 4, project_tens[local_config], 0, tn(row, col));
    }
  }
  return tn;
}

template<typename TenElemT, typename QNT>
void TPS<TenElemT, QNT>::UpdateConfigurationTN(const std::vector<SiteIdx> &site_set, const std::vector<size_t> &config,
                                               TensorNetwork2D<TenElemT, QNT> &tn2d) const {
  assert(boundary_condition_ == tn2d.GetBoundaryCondition() && "Boundary condition mismatch between TPS and TensorNetwork2D");

  Index<QNT> physical_index; // We suppose each site has the same hilbert space
  physical_index = (*this)(0, 0)->GetIndex(4);
  auto physical_index_inv = InverseIndex(physical_index);
  const size_t type_of_config = physical_index.dim();
  TenT project_tens[type_of_config];
  for (size_t i = 0; i < type_of_config; i++) {
    project_tens[i] = TenT({physical_index_inv});
    project_tens[i]({i}) = TenElemT(1.0);
  }


  for (size_t i = 0; i < site_set.size(); i++) {
    const SiteIdx &site = site_set[i];
    size_t local_config = config[i];
    tn2d(site).alloc();
    Contract(&(*this)(site), 4, project_tens[local_config], 0, &tn2d(site));
  }
}

template<typename TenElemT, typename QNT>
size_t TPS<TenElemT, QNT>::GetMaxBondDimension(void) const {
  size_t dmax = 0;
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      const TenT *tensor = (*this)(row, col);
      dmax = std::max(dmax, tensor->GetShape()[0]);
      dmax = std::max(dmax, tensor->GetShape()[1]);
    }
  }
  return dmax;
}

///< OBC
template<typename TenElemT, typename QNT>
bool TPS<TenElemT, QNT>::IsBondDimensionUniform(void) const {
  size_t d = (*this)(0, 0)->GetShape()[1];
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      const TenT *tensor = (*this)(row, col);
      bool check_vertical = (row != 0) || (boundary_condition_ == BoundaryCondition::Periodic);
      if (check_vertical && d != tensor->GetShape()[3]) {
        return false;
      }
      bool check_horizontal = (col != 0) || (boundary_condition_ == BoundaryCondition::Periodic);
      if (check_horizontal && d != tensor->GetShape()[0]) {
        return false;
      }
    }
  }
  return true;
}

template<typename TenElemT, typename QNT>
void TPS<TenElemT, QNT>::Dump(const std::string &tps_path, const bool release_mem) {
  EnsureDirectoryExists(tps_path);
  // Write Metadata
  {
    std::string meta_file = tps_path + "/tps_meta.txt";
    std::ofstream ofs(meta_file);
    if (ofs) {
      ofs << this->rows() << " " << this->cols() << " " << static_cast<int>(boundary_condition_) << std::endl;
    }
  }
  std::string file;
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      file = GenTPSTenName(tps_path, row, col);
      this->DumpTen(row, col, file, release_mem);
    }
  }
}

template<typename TenElemT, typename QNT>
bool TPS<TenElemT, QNT>::Load(const std::string &tps_path) {
  std::string meta_file = tps_path + "/tps_meta.txt";
  std::ifstream ifs(meta_file);
  if (ifs) {
    size_t r, c;
    int bc_int;
    ifs >> r >> c >> bc_int;
    if (ifs) {
      if (r != this->rows() || c != this->cols()) {
         *this = TPS<TenElemT, QNT>(r, c, static_cast<BoundaryCondition>(bc_int));
      } else {
         boundary_condition_ = static_cast<BoundaryCondition>(bc_int);
      }
    }
  }

  std::string file;
  for (size_t row = 0; row < this->rows(); ++row) {
    for (size_t col = 0; col < this->cols(); ++col) {
      file = GenTPSTenName(tps_path, row, col);
      if (!(this->LoadTen(row, col, file))) {
        return false;
      };
    }
  }
  return true;
}

// =====================================================================
// WaveFunctionSum Implementation
// =====================================================================

namespace detail {
/**
 * @brief Determine which virtual indices need to be expanded at a given site.
 * 
 * For OBC, boundary indices (dim=1) are NOT expanded.
 * For PBC, all 4 virtual indices are expanded.
 * 
 * @param row Current row index
 * @param col Current column index
 * @param rows Total number of rows
 * @param cols Total number of columns
 * @param bc Boundary condition
 * @return Vector of index numbers to expand (subset of {0, 1, 2, 3})
 */
inline std::vector<size_t> GetExpandIndices_(
    size_t row, size_t col,
    size_t rows, size_t cols,
    BoundaryCondition bc
) {
  std::vector<size_t> expand_idx_nums;
  
  if (bc == BoundaryCondition::Periodic) {
    // PBC: expand all 4 virtual indices
    expand_idx_nums = {0, 1, 2, 3};
  } else {
    // OBC: only expand non-boundary indices
    // Index 0 (West): expand if col > 0
    if (col > 0) {
      expand_idx_nums.push_back(0);
    }
    // Index 1 (South): expand if row < rows-1
    if (row < rows - 1) {
      expand_idx_nums.push_back(1);
    }
    // Index 2 (East): expand if col < cols-1
    if (col < cols - 1) {
      expand_idx_nums.push_back(2);
    }
    // Index 3 (North): expand if row > 0
    if (row > 0) {
      expand_idx_nums.push_back(3);
    }
  }
  return expand_idx_nums;
}
} // namespace detail

template<typename TenElemT, typename QNT>
TPS<TenElemT, QNT> WaveFunctionSum(
    const TPS<TenElemT, QNT> &tps1,
    const TPS<TenElemT, QNT> &tps2
) {
  using TenT = QLTensor<TenElemT, QNT>;
  
  // Precondition checks
  assert(tps1.rows() == tps2.rows() && "TPS dimensions must match");
  assert(tps1.cols() == tps2.cols() && "TPS dimensions must match");
  assert(tps1.GetBoundaryCondition() == tps2.GetBoundaryCondition() && 
         "Boundary conditions must match");
  
  const size_t rows = tps1.rows();
  const size_t cols = tps1.cols();
  const BoundaryCondition bc = tps1.GetBoundaryCondition();
  
  TPS<TenElemT, QNT> result(rows, cols, bc);
  
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      const TenT &ten1 = *tps1(row, col);
      const TenT &ten2 = *tps2(row, col);
      
      // Physical indices must match
      assert(ten1.GetIndex(4) == ten2.GetIndex(4) && 
             "Physical indices must match at each site");
      
      std::vector<size_t> expand_idx_nums = 
          detail::GetExpandIndices_(row, col, rows, cols, bc);
      
      if (expand_idx_nums.empty()) {
        // No expansion needed (corner case for 1x1 OBC lattice)
        result({row, col}) = ten1 + ten2;
      } else {
        TenT expanded_ten;
        Expand(&ten1, &ten2, expand_idx_nums, &expanded_ten);
        result({row, col}) = std::move(expanded_ten);
      }
    }
  }
  
  return result;
}

template<typename TenElemT, typename QNT>
TPS<TenElemT, QNT> WaveFunctionSum(
    TenElemT alpha, const TPS<TenElemT, QNT> &tps1,
    TenElemT beta, const TPS<TenElemT, QNT> &tps2
) {
  using TenT = QLTensor<TenElemT, QNT>;
  
  // Precondition checks
  assert(tps1.rows() == tps2.rows() && "TPS dimensions must match");
  assert(tps1.cols() == tps2.cols() && "TPS dimensions must match");
  assert(tps1.GetBoundaryCondition() == tps2.GetBoundaryCondition() && 
         "Boundary conditions must match");
  
  const size_t rows = tps1.rows();
  const size_t cols = tps1.cols();
  const size_t N = rows * cols;
  const BoundaryCondition bc = tps1.GetBoundaryCondition();
  
  // Compute per-site scaling factors: alpha^(1/N), beta^(1/N)
  // For complex numbers, use principal root
  double alpha_scale, beta_scale;
  if constexpr (std::is_same_v<TenElemT, double>) {
    alpha_scale = std::pow(std::abs(alpha), 1.0 / N);
    beta_scale = std::pow(std::abs(beta), 1.0 / N);
    if (alpha < 0 && N % 2 == 1) alpha_scale = -alpha_scale;
    if (beta < 0 && N % 2 == 1) beta_scale = -beta_scale;
  } else {
    // Complex case: use magnitude
    alpha_scale = std::pow(std::abs(alpha), 1.0 / N);
    beta_scale = std::pow(std::abs(beta), 1.0 / N);
  }
  
  TPS<TenElemT, QNT> result(rows, cols, bc);
  
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      TenT ten1_scaled = *tps1(row, col) * TenElemT(alpha_scale);
      TenT ten2_scaled = *tps2(row, col) * TenElemT(beta_scale);
      
      std::vector<size_t> expand_idx_nums = 
          detail::GetExpandIndices_(row, col, rows, cols, bc);
      
      if (expand_idx_nums.empty()) {
        result({row, col}) = ten1_scaled + ten2_scaled;
      } else {
        TenT expanded_ten;
        Expand(&ten1_scaled, &ten2_scaled, expand_idx_nums, &expanded_ten);
        result({row, col}) = std::move(expanded_ten);
      }
    }
  }
  
  return result;
}

template<typename TenElemT, typename QNT>
TPS<TenElemT, QNT> WaveFunctionSum(
    const std::vector<TPS<TenElemT, QNT>> &tps_list,
    const std::vector<TenElemT> &coefficients
) {
  assert(!tps_list.empty() && "tps_list must be non-empty");
  assert(coefficients.empty() || coefficients.size() == tps_list.size() && 
         "coefficients must be empty or match tps_list size");
  
  if (tps_list.size() == 1) {
    if (coefficients.empty()) {
      return tps_list[0];
    } else {
      // Scale by coefficient (distributed across sites)
      TPS<TenElemT, QNT> result = tps_list[0];
      const size_t N = result.rows() * result.cols();
      double scale = std::pow(std::abs(coefficients[0]), 1.0 / N);
      for (size_t row = 0; row < result.rows(); ++row) {
        for (size_t col = 0; col < result.cols(); ++col) {
          result({row, col}) *= TenElemT(scale);
        }
      }
      return result;
    }
  }
  
  // Successive pairwise direct sums
  TPS<TenElemT, QNT> result = coefficients.empty() 
      ? WaveFunctionSum(tps_list[0], tps_list[1])
      : WaveFunctionSum(coefficients[0], tps_list[0], coefficients[1], tps_list[1]);
  
  for (size_t i = 2; i < tps_list.size(); ++i) {
    if (coefficients.empty()) {
      result = WaveFunctionSum(result, tps_list[i]);
    } else {
      // For successive sums with coefficients, we need to handle this carefully.
      // The accumulated result should be treated as having coefficient 1.
      result = WaveFunctionSum(TenElemT(1.0), result, coefficients[i], tps_list[i]);
    }
  }
  
  return result;
}

} // qlpeps

#endif //QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_IMPL_H
