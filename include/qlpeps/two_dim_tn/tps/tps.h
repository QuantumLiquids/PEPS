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

// =====================================================================
// Wave Function Superposition for TPS
// =====================================================================

/**
 * @brief Wave function sum: \f$|\psi\rangle = |\psi_1\rangle + |\psi_2\rangle\f$
 *
 * Computes the quantum superposition of two tensor product states by expanding 
 * (direct sum) all virtual bond indices.
 *
 * Implementation:
 * ---------------
 * For each site tensor \f$A\f$ and \f$B\f$, the result tensor \f$C\f$ is computed via:
 *
 * \f[
 *   C^{s}_{(l_1 l_2)(d_1 d_2)(r_1 r_2)(u_1 u_2)} = 
 *   \begin{pmatrix} A^{s}_{l_1 d_1 r_1 u_1} & 0 \\ 0 & B^{s}_{l_2 d_2 r_2 u_2} \end{pmatrix}
 * \f]
 *
 * where \f$s\f$ is the physical index, and \f$(l, d, r, u)\f$ are the virtual indices
 * (West, South, East, North). The bond dimension of \f$C\f$ is \f$D_1 + D_2\f$.
 *
 * Boundary Conditions:
 * --------------------
 * - OBC: Boundary indices (dim=1) are NOT expanded. Only bulk virtual indices are expanded.
 * - PBC: All four virtual indices are expanded at every site.
 *
 * @tparam TenElemT Tensor element type (e.g., double, std::complex<double>)
 * @tparam QNT Quantum number type (e.g., U1QN)
 *
 * @param tps1 First TPS wave function \f$|\psi_1\rangle\f$
 * @param tps2 Second TPS wave function \f$|\psi_2\rangle\f$
 *
 * @return TPS representing \f$|\psi_1\rangle + |\psi_2\rangle\f$
 *
 * @pre tps1 and tps2 must have the same lattice dimensions
 * @pre tps1 and tps2 must have the same boundary condition
 * @pre Physical indices at corresponding sites must match exactly
 *
 * @note This is NOT element-wise tensor addition. Use operator+ on tensors for that.
 * @note The returned TPS has bond dimension \f$D_1 + D_2\f$ (or \f$D\f$ at boundaries for OBC)
 *
 * @see Expand() in TensorToolkit for the underlying tensor direct sum operation
 */
template<typename TenElemT, typename QNT>
TPS<TenElemT, QNT> WaveFunctionSum(
    const TPS<TenElemT, QNT> &tps1,
    const TPS<TenElemT, QNT> &tps2
);

/**
 * @brief Wave function sum with coefficients: \f$|\psi\rangle = \alpha|\psi_1\rangle + \beta|\psi_2\rangle\f$
 *
 * Computes a weighted superposition of two tensor product states. Each tensor in
 * tps1 is scaled by \f$\alpha^{1/N}\f$ and each tensor in tps2 is scaled by 
 * \f$\beta^{1/N}\f$, where \f$N\f$ is the total number of sites.
 *
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 *
 * @param alpha Coefficient for \f$|\psi_1\rangle\f$
 * @param tps1 First TPS wave function \f$|\psi_1\rangle\f$
 * @param beta Coefficient for \f$|\psi_2\rangle\f$
 * @param tps2 Second TPS wave function \f$|\psi_2\rangle\f$
 *
 * @return TPS representing \f$\alpha|\psi_1\rangle + \beta|\psi_2\rangle\f$
 *
 * @note The scaling is distributed across all sites to maintain numerical stability.
 *       Each site tensor is scaled by \f$\alpha^{1/N}\f$ or \f$\beta^{1/N}\f$ respectively.
 */
template<typename TenElemT, typename QNT>
TPS<TenElemT, QNT> WaveFunctionSum(
    TenElemT alpha, const TPS<TenElemT, QNT> &tps1,
    TenElemT beta, const TPS<TenElemT, QNT> &tps2
);

/**
 * @brief Wave function sum of N states: \f$|\psi\rangle = \sum_i c_i |\psi_i\rangle\f$
 *
 * Computes the superposition of multiple tensor product states with optional coefficients.
 * This is implemented as successive pairwise wave function sums.
 *
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 *
 * @param tps_list Vector of TPS wave functions \f$\{|\psi_i\rangle\}\f$
 * @param coefficients Optional vector of coefficients \f$\{c_i\}\f$. 
 *                     If empty, all coefficients default to 1.
 *
 * @return TPS representing \f$\sum_i c_i |\psi_i\rangle\f$
 *
 * @pre tps_list must be non-empty
 * @pre If coefficients is non-empty, it must have the same size as tps_list
 * @pre All TPS in tps_list must have the same dimensions and boundary conditions
 *
 * @note The resulting bond dimension is \f$\sum_i D_i\f$
 */
template<typename TenElemT, typename QNT>
TPS<TenElemT, QNT> WaveFunctionSum(
    const std::vector<TPS<TenElemT, QNT>> &tps_list,
    const std::vector<TenElemT> &coefficients = {}
);

} // qlpeps

#include "qlpeps/two_dim_tn/tps/tps_impl.h"

#endif //QLPEPS_VMC_PEPS_TWO_DIM_TN_TPS_TPS_H
