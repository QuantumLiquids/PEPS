// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-12-11
*
* Description: QuantumLiquids/PEPS project. The BMPS Contractor for 2D Tensor Networks.
*/

#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_BMPS_CONTRACTOR_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_BMPS_CONTRACTOR_H

#include <map>
#include <vector>
#include "qlten/qlten.h"
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"
#include "qlpeps/two_dim_tn/framework/site_idx.h"
#include "qlpeps/basic.h"
#include "qlpeps/two_dim_tn/common/boundary_condition.h"

namespace qlpeps {

// Forward declaration
template<typename TenElemT, typename QNT>
class TensorNetwork2D;

using BTenPOSITION = BMPSPOSITION;

/**
 * @brief BMPS Contractor for 2D Tensor Networks.
 * 
 * This class encapsulates the Boundary MPS (BMPS) contraction algorithm state and logic.
 * It separates the "solver" (algorithm/workspace) from the "data" (TensorNetwork2D).
 */
template<typename TenElemT, typename QNT>
class BMPSContractor {
 public:
  using Tensor = qlten::QLTensor<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using BMPST = BMPS<TenElemT, QNT>;
  using TransferMPO = std::vector<Tensor *>;

  /**
   * @brief Constructor
   * @param rows Number of rows in the network
   * @param cols Number of columns in the network
   */
  BMPSContractor(size_t rows, size_t cols);

  /**
   * @brief Initialize boundary MPS sets based on the TensorNetwork2D
   * @param tn The tensor network data
   */
  void Init(const TensorNetwork2D<TenElemT, QNT>& tn);

  // Accessors
  const std::vector<BMPS<TenElemT, QNT>>& GetBMPS(BMPSPOSITION position) const {
    return bmps_set_.at(position);
  }

  // --- BMPS Growth Methods ---

  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GenerateBMPSApproach(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION post, const BMPSTruncateParams<RealT> &trunc_para);

  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GrowBMPSForRow(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t row, const BMPSTruncateParams<RealT> &trunc_para);

  const std::pair<BMPST, BMPST> GetBMPSForRow(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t row, const BMPSTruncateParams<RealT> &trunc_para);

  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GrowBMPSForCol(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t col, const BMPSTruncateParams<RealT> &trunc_para);

  const std::pair<BMPST, BMPST> GetBMPSForCol(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t col, const BMPSTruncateParams<RealT> &trunc_para);

  void BMPSMoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para);

  void GrowFullBMPS(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para);

  void DeleteInnerBMPS(const BMPSPOSITION position) {
    if (!bmps_set_[position].empty()) {
      bmps_set_[position].erase(bmps_set_[position].begin() + 1, bmps_set_[position].end());
    }
  }

  // --- Boundary Tensor (BTen) Methods ---

  void InitBTen(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num);

  void InitBTen2(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1);

  void GrowBTenStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post);

  void GrowFullBTen(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num, const size_t remain_sites = 2, bool init = true);

  void GrowFullBTen2(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post, const size_t slice_num1, const size_t remain_sites = 2, bool init = true);

  void BTenMoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position);

  void TruncateBTen(const BTenPOSITION position, const size_t length);
  
  void BTen2MoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1);

  // --- Trace Methods ---

  /**
   * @note Trace functions for fermion tensor network also return a c-number. But
   * by the definition, the wave-function components of the fermion wave function
   * are also associated with the order of the single-particle fermion quantum number (means, the site indices).
   * When calling these trace functions, one should carefully investigate the default
   * fermion orders, by, carefully reading the code (so sad), so that to make sure
   * they give what you want.
   */
  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const BondOrientation bond_dir) const;

  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const SiteIdx &site_b, const BondOrientation bond_dir) const;

  TenElemT ReplaceOneSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site, const Tensor &replace_ten, const BondOrientation mps_orient) const;

  TenElemT ReplaceNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const SiteIdx &site_b,
                              const BondOrientation bond_dir,
                              const Tensor &ten_a, const Tensor &ten_b) const;

  TenElemT ReplaceNNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &left_up_site,
                               const DIAGONAL_DIR nnn_dir,
                               const BondOrientation mps_orient,
                               const Tensor &ten_left, const Tensor &ten_right) const;

  TenElemT ReplaceTNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site0,
                               const BondOrientation mps_orient,
                               const Tensor &replaced_ten0,
                               const Tensor &replaced_ten1,
                               const Tensor &replaced_ten2) const;

  TenElemT ReplaceSqrt5DistTwoSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &left_up_site,
                                        const DIAGONAL_DIR sqrt5link_dir,
                                        const BondOrientation mps_orient, 
                                        const Tensor &ten_left, const Tensor &ten_right) const;

  Tensor PunchHole(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site, const BondOrientation mps_orient) const;

  bool DirectionCheck() const;

  void InvalidateEnvs(const SiteIdx &site);

 private:
  void InitBMPS(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION post);

  size_t GrowBMPSStep_(const BMPSPOSITION position, TransferMPO, const BMPSTruncateParams<RealT> &);

  size_t GrowBMPSStep_(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &);

  void GrowBTen2Step_(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post, const size_t slice_num1);

  std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> bmps_set_;
  std::map<BTenPOSITION, std::vector<Tensor>> bten_set_;  // for 1 layer between two bmps
  std::map<BTenPOSITION, std::vector<Tensor>> bten_set2_; // for 2 layers between two bmps
  
  size_t rows_;
  size_t cols_;
};

} // namespace qlpeps

#include "bmps_contractor_impl.h"

#endif //QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_BMPS_CONTRACTOR_H
