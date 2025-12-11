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
  TensorNetwork2D<TenElemT, QNT> tn(rows, cols);

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

} // qlpeps

#endif //QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_IMPL_H
