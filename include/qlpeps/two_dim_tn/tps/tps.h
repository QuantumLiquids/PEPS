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
#include "qlpeps/two_dim_tn/common/boundary_condition.h"

namespace qlpeps {
using namespace qlten;

// Helpers
inline std::string GenTPSTenName(const std::string &tps_path, const size_t row, const size_t col) {
  return tps_path + "/" +
         kTpsTenBaseName + std::to_string(row) + "_" + std::to_string(col) + "." + kQLTenFileSuffix;
}

/**
 * @brief Generic Tensor Product State (TPS) class for 2D quantum lattice systems.
 * 
 * This class represents a state on a 2D square lattice where each site contains a rank-5 tensor.
 * It serves as a base representation for PEPS (Projected Entangled Pair States).
 *
 * Tensor Index Convention:
 * ------------------------
 * Each tensor A[row][col] has 5 indices ordered as:
 * (West, South, East, North, Physical) = (0, 1, 2, 3, 4)
 * 
 *          3 (North)
 *          |
 *          v
 *          |
 *  0 -->-- A -->-- 2 (East)
 * (West)   |
 *          v
 *          |
 *          1 (South)
 *
 * Directions follow the standard screen coordinates (row increases downwards, col increases rightwards).
 * - Index 0 (West) connects to (row, col-1) [Index 2 East]
 * - Index 1 (South) connects to (row+1, col) [Index 3 North]
 * - Index 2 (East) connects to (row, col+1) [Index 0 West]
 * - Index 3 (North) connects to (row-1, col) [Index 1 South]
 * - Index 4 is the Physical index (local Hilbert space)
 *
 *
 * Boundary Conditions:
 * --------------------
 * Supports both Open (OBC) and Periodic (PBC) boundary conditions.
 * - OBC: Boundary indices are dummy indices of dimension 1.
 * - PBC: 
 *   - (row, 0) West connects to (row, Cols-1) East
 *   - (0, col) North connects to (Rows-1, col) South
 *
 * @tparam TenElemT Tensor element type (e.g., double, complex<double>)
 * @tparam QNT Quantum Number type (e.g., U1QN)
 */
template<typename TenElemT, typename QNT>
class TPS : public TenMatrix<QLTensor<TenElemT, QNT>> {
 public:
  using TenT = QLTensor<TenElemT, QNT>;

  TPS(const size_t rows, const size_t cols, const BoundaryCondition bc = BoundaryCondition::Open) 
      : TenMatrix<QLTensor<TenElemT, QNT>>(rows, cols), boundary_condition_(bc) {}

  TensorNetwork2D<TenElemT, QNT> Project(const Configuration &config) const;

  BoundaryCondition GetBoundaryCondition() const { return boundary_condition_; }

  void UpdateConfigurationTN(const std::vector<SiteIdx> &site_set,
                             const std::vector<size_t> &config,
                             TensorNetwork2D<TenElemT, QNT> &tn2d) const;

  size_t GetMaxBondDimension(void) const;

  // if the bond dimension of each lambda is the same, except boundary gamma
  bool IsBondDimensionUniform(void) const;

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

 private:
  BoundaryCondition boundary_condition_;
};

} // qlpeps

#include "qlpeps/two_dim_tn/tps/tps_impl.h"

#endif //QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H
