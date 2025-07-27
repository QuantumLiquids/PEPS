// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-19
*
* Description: QuantumLiquids/PEPS project. The generic tensor product state (TPS) class.
*/

/**
@file tps.h
@brief The generic tensor product state (TPS) class.
*/

#ifndef QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H
#define QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/framework/ten_matrix.h"     // TenMatrix
#include "qlmps/utilities.h"                           // CreatPath
#include "qlpeps/consts.h"                              // kTpsPath
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/vmc_basic/configuration.h"

namespace qlpeps {
using namespace qlten;

// Helpers
inline std::string GenTPSTenName(const std::string &tps_path, const size_t row, const size_t col) {
  return tps_path + "/" +
         kTpsTenBaseName + std::to_string(row) + "_" + std::to_string(col) + "." + kQLTenFileSuffix;
}

template<typename TenElemT, typename QNT>
class TPS : public TenMatrix<QLTensor<TenElemT, QNT>> {
 public:
  using TenT = QLTensor<TenElemT, QNT>;

  TPS(const size_t rows, const size_t cols) : TenMatrix<QLTensor<TenElemT, QNT>>(rows, cols) {}

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

} // qlpeps

#include "qlpeps/two_dim_tn/tps/tps_impl.h"

#endif //QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H
