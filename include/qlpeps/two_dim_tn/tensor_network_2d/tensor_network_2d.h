// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-24
*
* Description: QuantumLiquids/PEPS project. The 2-dimensional tensor network class.
*/

#ifndef QLPEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H
#define QLPEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/framework/ten_matrix.h"
#include "qlpeps/two_dim_tn/framework/site_idx.h"
#include "qlpeps/basic.h"                           //BMPSTruncatePara
#include "qlpeps/vmc_basic/configuration.h"    //Configure
#include "qlpeps/two_dim_tn/common/boundary_condition.h"

namespace qlpeps {

//forward declaration
template<typename TenElemT, typename QNT>
class SplitIndexTPS;

using BTenPOSITION = BMPSPOSITION;

/**
 * @brief 2-dimensional finite-size tensor network data container
 * 
 * This class implements a 2D tensor network with the following features:
 * - Open Boundary Condition (OBC)
 * - Support for both bosonic and fermionic systems
 * 
 * It purely holds the tensor data and grid structure.
 * Algorithm logic and workspace (BMPS environments) have been moved to BMPSContractor.
 * 
 * For bosonic systems, each tensor has 4 physical indices with ordering:
 *         3
 *         |
 *      0--t--2
 *         |
 *         1
 *
 *  For fermion tensor network, there is additional 1-dim index which are used to
 *  match the even parities of the tensors, which is the last index.
 */
template<typename TenElemT, typename QNT>
class TensorNetwork2D : public TenMatrix<qlten::QLTensor<TenElemT, QNT>> {
  using IndexT = qlten::Index<QNT>;
  using Tensor = qlten::QLTensor<TenElemT, QNT>;
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
 public:
  /**
   * @brief Constructs empty tensor network with specified dimensions
   * 
   * @param rows Number of rows in the network
   * @param cols Number of columns in the network
   */
  TensorNetwork2D(const size_t rows, const size_t cols, const BoundaryCondition bc = BoundaryCondition::Open);

  /**
   * @brief Constructs tensor network by projecting the state from split-index TPS with specific configuration
   * 
   * @param tps Source split-index TPS
   * @param config Network configuration parameters
   */
  TensorNetwork2D(const SplitIndexTPS<TenElemT, QNT> &tps, const Configuration &config);

  TensorNetwork2D<TenElemT, QNT> &operator=(const TensorNetwork2D<TenElemT, QNT> &tn);

  BoundaryCondition GetBoundaryCondition() const { return boundary_condition_; }

  /**
   * @brief Updates one site tensor by projecting from wavefunction (in form of split-index TPS) and new local configuration
   * 
   * @param site Target site index
   * @param new_config New configuration
   * @param tps The global wave-function (in form of split-index TPS)
   */
  void UpdateSiteTensor(const SiteIdx &site, const size_t new_config, const SITPS &tps);

 private:
  BoundaryCondition boundary_condition_;
};

}//qlpeps

#include "tensor_network_2d_basic_impl.h"

#endif //QLPEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H

