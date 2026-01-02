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
  /// @brief Truncation parameters type for this contractor (enables generic code in TPSWaveFunctionComponent)
  using TruncateParams = BMPSTruncateParams<RealT>;

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

  /**
   * @brief Generates BMPS steps towards a specific position.
   * 
   * This method advances the boundary MPS contraction from the boundaries towards the specified `post` position.
   * It is typically used to prepare the environment for calculating observables at a specific location or row/column.
   *
   * @param tn The tensor network.
   * @param post The target position/direction where the BMPS should converge or be available.
   * @param trunc_para Parameters controlling the truncation of the boundary MPS bond dimension.
   * @return A const reference to the map of all BMPS vectors maintained by the contractor.
   */
  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GenerateBMPSApproach(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION post, const BMPSTruncateParams<RealT> &trunc_para);

  /**
   * @brief Grows the boundary MPS from top/bottom to reach the specified row.
   * 
   * This ensures that the UP and DOWN boundary MPS are grown until they reach the vicinity of `row`.
   * Specifically, after this call, the environments needed to contract the row `row` should be ready.
   * 
   * @param tn The tensor network.
   * @param row The target row index (0-indexed).
   * @param trunc_para Truncation parameters.
   * @return A const reference to the internal map of BMPS vectors.
   */
  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GrowBMPSForRow(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t row, const BMPSTruncateParams<RealT> &trunc_para);

  /**
   * @brief Retrieves the pair of BMPS (Top and Bottom environments) surrounding a specific row.
   * 
   * This function first ensures the BMPS are grown to the required depth and then returns
   * the specific UP and DOWN BMPS that "sandwich" the given `row`.
   * 
   * @param tn The tensor network.
   * @param row The target row index.
   * @param trunc_para Truncation parameters.
   * @return A pair of BMPS: {UP_BMPS, DOWN_BMPS} effectively enclosing the row.
   */
  const std::pair<BMPST, BMPST> GetBMPSForRow(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t row, const BMPSTruncateParams<RealT> &trunc_para);

  /**
   * @brief Grows the boundary MPS from left/right to reach the specified column.
   * 
   * Similar to GrowBMPSForRow, but for LEFT and RIGHT boundary MPS converging on `col`.
   * 
   * @param tn The tensor network.
   * @param col The target column index.
   * @param trunc_para Truncation parameters.
   * @return A const reference to the internal map of BMPS vectors.
   */
  const std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> &
  GrowBMPSForCol(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t col, const BMPSTruncateParams<RealT> &trunc_para);

  /**
   * @brief Retrieves the pair of BMPS (Left and Right environments) surrounding a specific column.
   * 
   * @param tn The tensor network.
   * @param col The target column index.
   * @param trunc_para Truncation parameters.
   * @return A pair of BMPS: {LEFT_BMPS, RIGHT_BMPS} effectively enclosing the column.
   */
  const std::pair<BMPST, BMPST> GetBMPSForCol(const TensorNetwork2D<TenElemT, QNT>& tn, const size_t col, const BMPSTruncateParams<RealT> &trunc_para);

  /**
   * @brief Advances the BMPS at the specified position by one step (one row or column).
   * 
   * This absorbs one layer of the bulk tensor network into the boundary MPS at `position`.
   * 
   * @param tn The tensor network.
   * @param position The boundary position (UP, DOWN, LEFT, RIGHT) to advance.
   * @param trunc_para Truncation parameters for the bond compression after absorption.
   */
  void BMPSMoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para);

  /**
   * @brief Fully grows the BMPS from the boundary at `position` across the entire network.
   * 
   * This effectively sweeps the boundary MPS through the whole system.
   * 
   * @param tn The tensor network.
   * @param position The starting boundary position.
   * @param trunc_para Truncation parameters.
   */
  void GrowFullBMPS(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para);

  /**
   * @brief Removes inner BMPS layers to save memory, keeping only the boundary.
   * 
   * @param position The boundary position stack to clean up.
   */
  void DeleteInnerBMPS(const BMPSPOSITION position) {
    if (!bmps_set_[position].empty()) {
      bmps_set_[position].erase(bmps_set_[position].begin() + 1, bmps_set_[position].end());
    }
  }

  // --- Boundary Tensor (BTen) Methods ---

  /**
   * @brief Initializes the boundary tensor (environment) set for a specific position and slice.
   * 
   * "BTen" (Boundary Tensor) usually refers to the environment tensors accumulated during the contraction
   * of a strip/slice, orthogonal to the BMPS direction.
   * 
   * @param tn The tensor network.
   * @param position The position/direction of the boundary.
   * @param slice_num The index of the slice (row or column) being processed.
   */
  void InitBTen(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num);

  /**
   * @brief Initializes a double-slice boundary tensor set (spanning two adjacent rows/columns).
   * 
   * Used when contracting a width-2 strip simultaneously (e.g., for NN interactions or 2-site updates).
   * @note This refers to two geometric lattice slices, distinct from the "double-layer" (ket+bra) concept in PEPS.
   * 
   * @param tn The tensor network.
   * @param position The position/direction.
   * @param slice_num1 The index of the first slice in the pair.
   */
  void InitBTen2(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1);

  /**
   * @brief Advances the single-layer boundary tensor by one site.
   * 
   * @param tn The tensor network.
   * @param post The position/direction of the boundary tensor.
   */
  void GrowBTenStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post);

  /**
   * @brief Fully grows the single-layer boundary tensor for a given slice.
   * 
   * @param tn The tensor network.
   * @param position The position/direction.
   * @param slice_num The slice index.
   * @param remain_sites Number of sites to leave uncontracted (e.g., for local operations). Default is 2.
   * @param init Whether to initialize the BTen set before growing. Default is true.
   */
  void GrowFullBTen(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num, const size_t remain_sites = 2, bool init = true);

  /**
   * @brief Fully grows the double-slice boundary tensor for a pair of slices.
   * 
   * @param tn The tensor network.
   * @param post The position/direction.
   * @param slice_num1 The first slice index.
   * @param remain_sites Number of sites to leave uncontracted. Default is 2.
   * @param init Whether to initialize the BTen set before growing. Default is true.
   */
  void GrowFullBTen2(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post, const size_t slice_num1, const size_t remain_sites = 2, bool init = true);

  /**
   * @brief Moves the single-layer boundary tensor one step.
   * 
   * @param tn The tensor network.
   * @param position The position/direction.
   */
  void BTenMoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position);

  /**
   * @brief Truncates the boundary tensor set to a specific length.
   * 
   * Useful for managing memory or resetting state.
   * 
   * @param position The position/direction.
   * @param length The new size of the BTen vector.
   */
  void TruncateBTen(const BTenPOSITION position, const size_t length);
  
  /**
   * @brief Moves the double-slice boundary tensor one step.
   * 
   * @param tn The tensor network.
   * @param position The position/direction.
   * @param slice_num1 The first slice index (used for context).
   */
  void BTen2MoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1);

  // --- Trace Methods ---

  /**
   * @brief Trace/contraction methods for computing wavefunction amplitudes.
   *
   * @warning **Fermion Sign Consistency**
   *
   * For fermionic tensor networks, the wavefunction amplitude \f$\Psi(S)\f$ depends
   * not only on the configuration \f$S\f$, but also on the **contraction path**
   * (i.e., the order in which parity indices are fused via `FuseIndex`).
   *
   * **Key rule**: When computing amplitude ratios \f$\Psi(S')/\Psi(S)\f$, both
   * numerator and denominator **must** be evaluated using the same contraction path
   * (same `Trace` or `ReplaceXXXTrace` call with identical environment setup).
   * Otherwise, the relative sign may be inconsistent.
   *
   * **Practical implication**: In `EvaluateBondEnergy` for fermion models, always
   * recalculate `psi = Trace(...)` locally rather than reusing a cached value
   * computed elsewhere. See `docs/dev/design/math/fermion-sign-in-bmps-contraction.md`
   * for the mathematical derivation.
   *
   * @see ReplaceNNSiteTrace, ReplaceNNNSiteTrace for amplitude ratio computations
   */
  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const BondOrientation bond_dir) const;

  /**
   * @brief Calculates the trace of the network with two sites considered.
   * 
   * @param tn The tensor network.
   * @param site_a The first site.
   * @param site_b The second site.
   * @param bond_dir The bond orientation.
   * @return The scalar result.
   */
  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const SiteIdx &site_b, const BondOrientation bond_dir) const;

  /**
   * @brief Replaces one site's tensor and calculates the trace.
   * 
   * Useful for calculating single-site observables or wave function amplitudes with local modifications.
   * 
   * @param tn The tensor network.
   * @param site The site to replace.
   * @param replace_ten The new tensor to place at `site`.
   * @param mps_orient The orientation of the boundary MPS used for contraction.
   * @return The scalar result.
   */
  TenElemT ReplaceOneSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site, const Tensor &replace_ten, const BondOrientation mps_orient) const;

  /**
   * @brief Replaces two nearest-neighbor sites' tensors and calculates the trace.
   * 
   * Useful for calculating two-site observables or wave function amplitudes with local modifications.
   * 
   * @param tn The tensor network.
   * @param site_a The first site.
   * @param site_b The second site.
   * @param bond_dir The direction of the bond connecting the two sites.
   * @param ten_a The replacement tensor for site A.
   * @param ten_b The replacement tensor for site B.
   * @return The scalar result.
   */
  TenElemT ReplaceNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site_a, const SiteIdx &site_b,
                              const BondOrientation bond_dir,
                              const Tensor &ten_a, const Tensor &ten_b) const;

  /**
   * @brief Replaces two next-nearest-neighbor (diagonal) sites and calculates the trace.
   * 
   * Useful for calculating next-nearest-neighbor (diagonal) two-site observables or wave function amplitudes with local modifications.
   * 
   * @param tn The tensor network.
   * @param left_up_site The site at the top-left of the diagonal pair.
   * @param nnn_dir The diagonal direction (e.g., UP_RIGHT, UP_LEFT).
   * @param mps_orient The MPS orientation.
   * @param ten_left The replacement tensor for the left site.
   * @param ten_right The replacement tensor for the right site.
   * @return The scalar result.
   */
  TenElemT ReplaceNNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &left_up_site,
                               const DIAGONAL_DIR nnn_dir,
                               const BondOrientation mps_orient,
                               const Tensor &ten_left, const Tensor &ten_right) const;

  /**
   * @brief Replaces third-nearest-neighbor sites and calculates the trace.
   * 
   * Useful for calculating three-site observables or wave function amplitudes with local modifications.
   * 
   * @param tn The tensor network.
   * @param site0 The first site (reference).
   * @param mps_orient The MPS orientation.
   * @param replaced_ten0 Replacement for site 0.
   * @param replaced_ten1 Replacement for site 1.
   * @param replaced_ten2 Replacement for site 2.
   * @return The scalar result.
   */
  TenElemT ReplaceTNNSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site0,
                               const BondOrientation mps_orient,
                               const Tensor &replaced_ten0,
                               const Tensor &replaced_ten1,
                               const Tensor &replaced_ten2) const;

  /**
   * @brief Replaces two sites separated by a "knight's move" (sqrt(5) distance) and calculates trace.
   * 
   * Useful for calculating long-range sqrt(5) distance two-site observables or wave function amplitudes with local modifications.
   * 
   * @param tn The tensor network.
   * @param left_up_site The top-left site reference.
   * @param sqrt5link_dir The direction of the long-range link.
   * @param mps_orient The MPS orientation.
   * @param ten_left Replacement for the left site.
   * @param ten_right Replacement for the right site.
   * @return The scalar result.
   */
  TenElemT ReplaceSqrt5DistTwoSiteTrace(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &left_up_site,
                                        const DIAGONAL_DIR sqrt5link_dir,
                                        const BondOrientation mps_orient, 
                                        const Tensor &ten_left, const Tensor &ten_right) const;

  /**
   * @brief Contracts the entire network except for one site, returning the environment tensor for that site.
   * 
   * This effectively "punches a hole" in the network at `site`, returning the contraction of everything else.
   * The result is the environment tensor \f$E_i\f$ such that the total contraction is \f$C = T_i \cdot E_i\f$,
   * where \f$T_i\f$ is the tensor at site `i` in the input network `tn`.
   * 
   * @param tn The tensor network.
   * @param site The site to leave open.
   * @param mps_orient The MPS orientation used for contraction.
   * @return The environment tensor for the specified site.
   */
  Tensor PunchHole(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx &site, const BondOrientation mps_orient) const;

  /**
   * @brief Checks if the internal directions/orientations are consistent.
   * 
   * @note Design Critique:
   * This function exists because the relationship between `BMPS` objects and the `BMPSContractor`
   * is coupled in a way that allows state drift. Ideally, `BMPS` should be immutable or strictly
   * managed such that invalid states are unrepresentable. 
   * Currently, it serves as a runtime sanitizer for the complex stateful logic of BMPS growth. 
   * It should always be true in correct implementation.
   * 
   * @return True if consistent, false otherwise.
   */
  bool DirectionCheck() const;

  /**
   * @brief Erase cached environments affected by a local tensor update.
   *
   * This is a state-mutating operation: it truncates cached BMPS layers so that
   * subsequent environment growth recomputes the affected region instead of using stale data.
   */
  void EraseEnvsAfterUpdate(const SiteIdx &site);

  /**
   * @brief Debug-only check that cached environments have been invalidated for the given site.
   *
   * This function MUST NOT mutate state. It exists to catch missing invalidation in debug builds.
   * In release builds it is a no-op.
   */
  void CheckInvalidateEnvs(const SiteIdx &site) const;

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
