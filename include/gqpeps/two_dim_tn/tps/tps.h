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
#include "gqpeps/two_dim_tn/framework/ten_matrix.h"     // TenMatrix
#include "gqmps2/utilities.h"                           // CreatPath
#include "gqpeps/consts.h"                              // kTpsPath
#include "gqpeps/algorithm/vmc_update/tensor_network_2d.h"
#include "gqpeps/two_dim_tn/tps/configuration.h"

namespace gqpeps {
using namespace gqten;

// Helpers
inline std::string GenTPSTenName(const std::string &tps_path, const size_t row, const size_t col) {
  return tps_path + "/" +
         kTpsTenBaseName + std::to_string(row) + "_" + std::to_string(col) + "." + kGQTenFileSuffix;
}

template<typename TenElemT, typename QNT>
class TPS : public TenMatrix<GQTensor<TenElemT, QNT>> {
 public:
  using TenT = GQTensor<TenElemT, QNT>;

  TPS(const size_t rows, const size_t cols) : TenMatrix<GQTensor<TenElemT, QNT>>(rows, cols) {}

  TensorNetwork2D<TenElemT, QNT> Project(const Configuration &config) const;

  void UpdateConfigurationTN(const std::vector<SiteIdx> &site_set,
                             const std::vector<size_t> &config,
                             TensorNetwork2D<TenElemT, QNT> &tn2d) const;

  size_t GetMaxBondDimension(void) const;

  // if the bond dimension of each lambda is the same, except boundary gamma
  bool IsBondDimensionEven(void) const;

  /**
  Dump TPS to HDD.

  @param tps_path Path to the TPS directory.
  @param release_mem Whether release memory after dump.
  */
  void Dump(const std::string &tps_path = kTpsPath, const bool release_mem = false);

  /**
  Load TPS from HDD.

  @param tps_path Path to the TPS directory.
  */
  bool Load(const std::string &tps_path = kTpsPath);
};

} // gqpeps

#include "gqpeps/two_dim_tn/tps/tps_impl.h"

#endif //GQPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H
