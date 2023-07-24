// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-19
*
* Description: GraceQ/VMC-PEPS project. The generic tensor product state (TPS) class.
*/

/**
@file tps.h
@brief The generic tensor product state (TPS) class.
*/

#ifndef GQPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H
#define GQPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/framework/ten_matrix.h"
#include "gqmps2/utilities.h"             //CreatPath
#include "gqpeps/consts.h"              //kTpsPath
#include "gqpeps/two_dim_tn/tps/tensor_network_2d.h"

namespace gqpeps {
using namespace gqten;

using Configuration = DuoMatrix<size_t>;

// Helpers
inline std::string GenTPSTenName(const std::string &tps_path, const size_t row, const size_t col) {
  return tps_path + "/" +
         kTpsTenBaseName + std::to_string(row) + "_" + std::to_string(col) + "." + kGQTenFileSuffix;
}

template<typename TenElemT, typename QNT>
class TPS : public TenMatrix<GQTensor<TenElemT, QNT>> {
 public:
  using TenT = GQTensor<TenElemT, QNT>;

  TensorNetwork2D<TenElemT, QNT> ProjectToConfiguration(
      const Configuration &config
  ) const {
    const size_t rows = this->rows();
    const size_t cols = this->cols();
    TensorNetwork2D<TenElemT, QNT> tn(rows, cols);
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

  /**
  Dump TPS to HDD.

  @param tps_path Path to the TPS directory.
  @param release_mem Whether release memory after dump.
  */
  void Dump(
      const std::string &tps_path = kTpsPath,
      const bool release_mem = false
  ) {
    if (!gqmps2::IsPathExist(tps_path)) { gqmps2::CreatPath(tps_path); }
    std::string file;
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        file = GenTPSTenName(tps_path, row, col);
        this->DumpTen(row, col, file, release_mem);
      }
    }
  }

  /**
  Load TPS from HDD.

  @param tps_path Path to the TPS directory.
  */
  void Load(const std::string &tps_path = kTpsPath) {
    std::string file;
    for (size_t row = 0; row < this->rows(); ++row) {
      for (size_t col = 0; col < this->cols(); ++col) {
        file = GenTPSTenName(tps_path, row, col);
        this->LoadTen(row, col, file);
      }
    }
  }
};

} // gqpeps

#endif //GQPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H
