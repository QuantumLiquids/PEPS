// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-24
*
* Description: GraceQ/VMC-PEPS project. The 2-dimensional tensor network class.
*/

#ifndef VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H
#define VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/framework/ten_matrix.h"
#include "gqpeps/ond_dim_tn/boundary_mps/bmps.h"
#include "gqpeps/basic.h"                           //BMPSTruncatePara
#include "gqpeps/two_dim_tn/tps/configuration.h"    //Configure

namespace gqpeps {
using namespace gqten;

//forward declaration
template<typename TenElemT, typename QNT>
class SplitIndexTPS;

using BTenPOSITION = BMPSPOSITION;

/**  2-dimensional finite-size tensor network and its environments (boundary MPS and so on)
 *         3
 *         |
 *      0--t--2
 *         |
 *         1
 * @tparam TenElemT
 * @tparam QNT
 */
template<typename TenElemT, typename QNT>
class TensorNetwork2D : public TenMatrix<GQTensor<TenElemT, QNT>> {
  using IndexT = Index<QNT>;
  using Tensor = GQTensor<TenElemT, QNT>;
  using TransferMPO = std::vector<Tensor *>;
  using BMPST = BMPS<TenElemT, QNT>;
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  //constructor
  //without initialization of the data of boundary mps
  TensorNetwork2D(const size_t rows, const size_t cols);

  //with initialization of the data of boundary mps
  TensorNetwork2D(const SplitIndexTPS<TenElemT, QNT> &tps, const Configuration &config);

  TensorNetwork2D<TenElemT, QNT> &operator=(const TensorNetwork2D<TenElemT, QNT> &tn);

  const std::vector<BMPS<TenElemT, QNT>> &GetBMPS(const BMPSPOSITION position) const {
    return bmps_set_[position];
  }

  void InitBMPS();

  void InitBMPS(const BMPSPOSITION post);

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
    bmps_set_[position].erase(bmps_set_[position].begin() + 1, bmps_set_[position].end());
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

  void UpdateSiteConfig(const SiteIdx &site, const size_t update_config, const SITPS &tps,
                        bool check_envs = false);

  /**
   * Calculate the trace by contracting the environment tensors around a NN bond
   * We assume we have gotten all of the environment tensors
   * @param site_a
   * @param bond_dir
   * @return
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

  TenElemT ReplaceSqrt5DistTwoSiteTrace(const SiteIdx &left_up_site,
                                        const DIAGONAL_DIR sqrt5link_dir,
                                        const BondOrientation mps_orient, //mps orientation is the same with longer side orientation
                                        const Tensor &ten_left, const Tensor &ten_right) const;

  Tensor PunchHole(const SiteIdx &site, const BondOrientation mps_orient) const;

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

}//gqpeps

#include "tensor_network_2d_impl.h"

#endif //VMC_PEPS_TWO_DIM_TN_TPS_TENSOR_NETWORK_2D_H
