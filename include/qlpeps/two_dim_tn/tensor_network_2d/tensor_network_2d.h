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
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"
#include "qlpeps/basic.h"                           //BMPSTruncatePara
#include "qlpeps/vmc_basic/configuration.h"    //Configure

namespace qlpeps {
using namespace qlten;

//forward declaration
template<typename TenElemT, typename QNT>
class SplitIndexTPS;

using BTenPOSITION = BMPSPOSITION;

/**
 * @brief 2-dimensional finite-size tensor network with environment tensors
 * 
 * This class implements a 2D tensor network with the following features:
 * - Open Boundary Condition (OBC)
 * - Network contraction using Boundary MPS (BMPS) method
 * - Support for both bosonic and fermionic systems
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
 *
 *  @note Trace functions for fermion tensor network also return a c-number. But
 *  by the definition, the wave-function components of the fermion wave function
 *  are also associated with the order of the single-particle fermion quantum number (means, the site indices).
 *  When calling these trace functions, one should carefully investigate the default
 *  fermion orders, by, carefully reading the code (so sad), so that to make sure
 *  they give what you want.
 */
template<typename TenElemT, typename QNT>
class TensorNetwork2D : public TenMatrix<QLTensor<TenElemT, QNT>> {
  using IndexT = Index<QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;
  using TransferMPO = std::vector<Tensor *>;
  using BMPST = BMPS<TenElemT, QNT>;
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  /**
   * @brief Constructs empty tensor network with specified dimensions
   * 
   * @param rows Number of rows in the network
   * @param cols Number of columns in the network
   * @note Boundary MPS data is not initialized
   */
  TensorNetwork2D(const size_t rows, const size_t cols);

  /**
   * @brief Constructs tensor network by projecting the state from split-index TPS with specific configuration
   * 
   * @param tps Source split-index TPS
   * @param config Network configuration parameters
   * @note Initializes both network tensors and boundary MPS
   */
  TensorNetwork2D(const SplitIndexTPS<TenElemT, QNT> &tps, const Configuration &config);

  TensorNetwork2D<TenElemT, QNT> &operator=(const TensorNetwork2D<TenElemT, QNT> &tn);

  /**
   * @brief Updates one site tensor by projecting from wavefunction (in form of split-index TPS) and new local configuration
   * 
   * @param site Target site index
   * @param new_config New configuration
   * @param tps The global wave-function (in form of split-index TPS)
   * @param check_envs Whether to check and delete incompatibility environment tensors
   */
  void UpdateSiteTensor(const SiteIdx &site, const size_t new_config, const SITPS &tps,
                        bool check_envs = true);

  const std::vector<BMPS<TenElemT, QNT>> &GetBMPS(const BMPSPOSITION position) const {
    return bmps_set_[position];
  }

  void InitBMPS();

  void InitBMPS(const BMPSPOSITION post);

  /**
   * @brief Generates BMPS approach to specified position
   * 
   * Grows boundary MPS towards the given boundary of the bulk:
   * - Iteratively absorbs MPO layers
   * - Applies truncation according to truncation parameters
   * 
   * @param post Starting position for BMPS growth
   * @param trunc_para Truncation parameters controlling accuracy
   * @return Map of position to vector of boundary MPS
   */
  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GenerateBMPSApproach(BMPSPOSITION post, const BMPSTruncatePara &trunc_para);

  /**
   * Generate the boundary MPS for the row-th MPO.
   * During the calculation, we assume the existed mpo data is suitable for current 2d tensor network.
   * @param row
   * @return
   */
  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GrowBMPSForRow(const size_t row, const BMPSTruncatePara &trunc_para);

  /**
   * Same functionality with GrowBMPSForRow but only return the corresponding boundary MPS
   * @param row
   * @return
   */
  const std::pair<BMPST, BMPST> GetBMPSForRow(const size_t row, const BMPSTruncatePara &trunc_para);

  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GrowBMPSForCol(const size_t col, const BMPSTruncatePara &trunc_para);

  const std::pair<BMPST, BMPST> GetBMPSForCol(const size_t col, const BMPSTruncatePara &trunc_para);

  void BMPSMoveStep(const BMPSPOSITION position, const BMPSTruncatePara &trunc_para);

  void GrowFullBMPS(const BMPSPOSITION position, const BMPSTruncatePara &trunc_para);

  void DeleteInnerBMPS(const BMPSPOSITION position) {
    if (!bmps_set_[position].empty()) {
      bmps_set_[position].erase(bmps_set_[position].begin() + 1, bmps_set_[position].end());
    }
  }

  /**
   * We assume the corresponding boundary MPS has constructed
   *
   * bmps don't change. only change boundary tensors
   * @param position
   * @param mpo_num row or col number
   */
  void InitBTen(const BTenPOSITION position, const size_t slice_num);

  void InitBTen2(const BTenPOSITION position, const size_t slice_num1);

  /**
 *
 * @param post the postion of the boundary tensor which will be grown.
 * @note the data of bmps should just clip one layer of mpo.
 */
  void GrowBTenStep(const BTenPOSITION post);

  //< if init = false, the existed environment tensor data is correct.
  void
  GrowFullBTen(const BTenPOSITION position, const size_t slice_num, const size_t remain_sites = 2, bool init = true);

  void
  GrowFullBTen2(const BTenPOSITION post, const size_t slice_num1, const size_t remain_sites = 2, bool init = true);

  /**
   * we assume the boundary mps and boundary tensors has form a square
   * @param position the direction they move
   */
  void BTenMoveStep(const BTenPOSITION position);

  void TruncateBTen(const BTenPOSITION position, const size_t length);
  void BTen2MoveStep(const BTenPOSITION position, const size_t slice_num1);

  /**
   * @brief Calculates trace by contracting environment tensors around NN bond
   * 
   * Assumes all required environment tensors are available.
   * For fermionic systems, users should carefully consider the implicit ordering.
   * 
   * @param site_a First site index
   * @param bond_dir Bond direction
   * @return Trace value
   */
  TenElemT Trace(const SiteIdx &site_a, const BondOrientation bond_dir) const;

  TenElemT Trace(const SiteIdx &site_a, const SiteIdx &site_b, const BondOrientation bond_dir) const;

  TenElemT ReplaceOneSiteTrace(const SiteIdx &site, const Tensor &replace_ten, const BondOrientation mps_orient) const;

  // There are some redundancy information but it will help users to check the calling.
  TenElemT ReplaceNNSiteTrace(const SiteIdx &site_a, const SiteIdx &site_b,
                              const BondOrientation bond_dir,
                              const Tensor &ten_a, const Tensor &ten_b) const;

  TenElemT ReplaceNNNSiteTrace(const SiteIdx &left_up_site,
                               const DIAGONAL_DIR nnn_dir,
                               const BondOrientation mps_orient,
                               const Tensor &ten_left, const Tensor &ten_right) const;
  //Third Nearest-Neighbor
  TenElemT ReplaceTNNSiteTrace(const SiteIdx &site0,
                               const BondOrientation mps_orient,
                               const Tensor &replaced_ten0,
                               const Tensor &replaced_ten1,
                               const Tensor &replaced_ten2) const;

  TenElemT ReplaceSqrt5DistTwoSiteTrace(const SiteIdx &left_up_site,
                                        const DIAGONAL_DIR sqrt5link_dir,
                                        const BondOrientation mps_orient, //mps orientation is the same with longer side orientation
                                        const Tensor &ten_left, const Tensor &ten_right) const;

  Tensor PunchHole(const SiteIdx &site, const BondOrientation mps_orient) const;

  ///< Debug function
  bool DirectionCheck() const;
 private:
  /**
 * grow one step for the boundary MPS
 *
 * @param position
 * @return
 */
  size_t GrowBMPSStep_(const BMPSPOSITION position, TransferMPO, const BMPSTruncatePara &);

  size_t GrowBMPSStep_(const BMPSPOSITION position, const BMPSTruncatePara &);

  /**
   *
   * @param post
   * @param slice_num1 for post = left/right, slice_num is the row number.
   *                   for post = up/down, slice_num is the col number.
   *                   slice_num1 is the smaller one.
   */
  void GrowBTen2Step_(const BTenPOSITION post, const size_t slice_num1);

  /** bmps_set_
   * left bmps: mps are numbered from left to right, mps tensors are numbered from top to bottom
   * down bmps: mps are numbered from bottom to top, mps tensors are numbered from left to right
   * right bmps: mps are numbered from right to left, mps tensors are numbered from bottom to top
   * up bmps: mps are numbered from top to bottom mps tensors are numbered from right to left
   */
  std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> bmps_set_;
  std::map<BTenPOSITION, std::vector<Tensor>> bten_set_;  // for 1 layer between two bmps
  std::map<BTenPOSITION, std::vector<Tensor>> bten_set2_; // for 2 layers between two bmps
};

}//qlpeps

#include "tensor_network_2d_basic_impl.h"
#include "tensor_network_2d_bten_operation.h"
#include "tensor_network_2d_trace_impl.h"

#endif //QLPEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H
