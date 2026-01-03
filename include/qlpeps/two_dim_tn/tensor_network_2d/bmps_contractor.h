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
   * @brief Shifts the BMPS environment window by one slice.
   * 
   * This pops the outermost BMPS at `position` and grows a new one at `Opposite(position)`.
   * The net effect is that the "sandwich" environment moves one row/column toward `position`.
   * 
   * @param tn The tensor network.
   * @param position The boundary position (UP, DOWN, LEFT, RIGHT) to shift toward.
   * @param trunc_para Truncation parameters for the bond compression after absorption.
   */
  void ShiftBMPSWindow(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para);

  /// @deprecated Use ShiftBMPSWindow instead
  [[deprecated("Use ShiftBMPSWindow instead")]]
  void BMPSMoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &trunc_para) {
    ShiftBMPSWindow(tn, position, trunc_para);
  }

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

 
  /**
   * @brief A lightweight, independent walker for Boundary MPS evolution.
   *
   * The BMPSWalker holds a detached copy of a BMPS state and can evolve independently
   * from the main BMPSContractor's state stack.
   *
   * ## Usage Scenario
   * - "Forking" the evolution logic to measure non-local observables (e.g., structure factors).
   * - Implementing "Excited State Propagation" where an operator is injected and propagated.
   *
   * The Walker holds a reference to the TensorNetwork2D to automatically resolve
   * MPOs for evolution steps (via EvolveStep), but it does not modify the TN or the Contractor.
   *
   * ## BMPS Storage Order (Critical for ContractRow)
   * 
   * The storage order differs by direction (see also BMPS class documentation):
   * 
   * - **UP BMPS**: Reversed storage. `bmps_[0]` corresponds to the rightmost column (col = N-1),
   *   `bmps_[N-1]` corresponds to the leftmost column (col = 0).
   *   Virtual bond connectivity: `bmps_[i].right (idx2)` connects `bmps_[i+1].left (idx0)`.
   * 
   * - **DOWN BMPS**: Normal storage. `bmps_[0]` corresponds to the leftmost column (col = 0),
   *   `bmps_[N-1]` corresponds to the rightmost column (col = N-1).
   *   Virtual bond connectivity: `bmps_[i].right (idx2)` connects `bmps_[i+1].left (idx0)`.
   * 
   * - **LEFT BMPS**: Normal storage (top to bottom).
   * - **RIGHT BMPS**: Reversed storage (bottom to top).
   * 
   * This asymmetry must be accounted for when performing sandwich contractions in ContractRow().
   */
  class BMPSWalker {
   public:
    /**
     * @brief Construct a new BMPSWalker.
     * @param tn Reference to the TensorNetwork (must outlive the Walker).
     * @param bmps The initial boundary MPS state (will be copied/moved in).
     * @param pos The direction of evolution (UP, DOWN, LEFT, RIGHT).
     * @param current_stack_size The number of layers already absorbed in the initial 'bmps'.
     *                           (Equivalent to bmps_set_[pos].size() at creation).
     */
    BMPSWalker(const TensorNetwork2D<TenElemT, QNT>& tn,
               BMPS<TenElemT, QNT> bmps,
               BMPSPOSITION pos,
               size_t current_stack_size)
        : tn_(tn), bmps_(std::move(bmps)), pos_(pos), stack_size_(current_stack_size) {}

    /**
     * @brief Evolve the walker by one step according to the lattice structure.
     *
     * Automatically determines the next row/column MPO to absorb based on current position
     * and stack size. Increments the internal stack size counter.
     *
     * @param trunc_para Truncation parameters.
     */
    void EvolveStep(const BMPSTruncateParams<RealT> &trunc_para);

    /**
     * @brief Evolve using a manually provided MPO.
     *
     * Useful for applying operators or custom evolution steps.
     * Updates the internal BMPS state.
     *
     * @note This does NOT increment the internal 'stack_size' counter. If this MPO corresponds
     * to a physical lattice layer, you might want to manually track that or use EvolveStep instead.
     *
     * @param mpo The MPO to absorb.
     * @param trunc_para Truncation parameters.
     */
    void Evolve(const TransferMPO& mpo, const BMPSTruncateParams<RealT> &trunc_para);

    /**
     * @brief Access the current Boundary MPS state.
     */
    const BMPS<TenElemT, QNT>& GetBMPS() const { return bmps_; }

    /**
     * @brief Access mutable Boundary MPS state (e.g., to apply local operators directly).
     */
    BMPS<TenElemT, QNT>& GetBMPSRef() { return bmps_; }

    /**
     * @brief Contracts the walker's boundary with a single row/column MPO and an opposing boundary.
     *
     * This calculates the scalar overlap: `<Walker_Boundary | MPO | Opposite_Boundary>`.
     * Useful for measuring observables or overlaps in the middle of evolution without modifying the Walker.
     *
     * ## Tensor Network Structure (for UP walker with DOWN opposite)
     * @verbatim
     *   top[col] (UP BMPS tensor)
     *      |
     *   mpo[col] (TN site tensor)  
     *      |
     *   bot[col] (DOWN BMPS tensor)
     * @endverbatim
     *
     * ## Index Conventions
     * - BMPS tensor (boson): `0: left, 1: physical, 2: right`
     * - TN site tensor: `0: left, 1: down, 2: right, 3: up`
     *
     * ## Storage Order Considerations
     * - **UP BMPS** (walker): Reversed storage. `bmps_[0]` = rightmost column (col=N-1).
     * - **DOWN BMPS** (opposite): Normal storage. `opposite[0]` = leftmost column (col=0).
     * - **Site tensors** (mpo): Normal order. `mpo[0]` = leftmost column (col=0).
     *
     * Due to the reversed UP storage, the contraction proceeds from col=N-1 to col=0 (right-to-left)
     * to properly match virtual bond indices between adjacent BMPS tensors.
     *
     * ## Connection Pattern
     * When contracting column tensors from right to left:
     * - `accumulator.top_R` connects `column.top_L` (UP BMPS internal bond)
     * - `column.site_R` connects `accumulator.site_L` (site tensor horizontal bond)
     * - `column.bot_R` connects `accumulator.bot_L` (DOWN BMPS internal bond)
     *
     * @param mpo The MPO row/column to sandwich (normal order: mpo[0] = leftmost).
     * @param opposite_boundary The boundary MPS from the opposite direction.
     * @return The scalar contraction result (overlap). Returns 0 if dimensions mismatch or direction is unsupported.
     *
     * @note Currently only supports UP walker with DOWN opposite_boundary (horizontal row contraction).
     */
    TenElemT ContractRow(const TransferMPO& mpo, const BMPS<TenElemT, QNT>& opposite_boundary) const;

    // ==================== BTen Cache Methods ====================
    // These methods provide efficient caching for multiple trace calculations
    // on the same row with different site replacements.

    /**
     * @brief Initialize the BTen cache for efficient multi-site trace calculations.
     * 
     * This pre-computes the LEFT boundary tensors from col=0 up to (but not including) `target_col`.
     * After calling this, you can use TraceWithBTen() to efficiently compute traces
     * with different site replacements at `target_col` or beyond.
     *
     * @param mpo The MPO row (normal order: mpo[0] = leftmost).
     * @param opposite_boundary The opposing BMPS (e.g., DOWN boundary for UP walker).
     * @param target_col The column index up to which to pre-compute the LEFT BTen.
     */
    void InitBTenLeft(const TransferMPO& mpo, const BMPS<TenElemT, QNT>& opposite_boundary, size_t target_col);

    /**
     * @brief Initialize the RIGHT BTen cache.
     * 
     * Pre-computes the RIGHT boundary tensors from col=N-1 down to (but not including) `target_col`.
     *
     * @param mpo The MPO row (normal order: mpo[0] = leftmost).
     * @param opposite_boundary The opposing BMPS.
     * @param target_col The column index down to which to pre-compute the RIGHT BTen.
     */
    void InitBTenRight(const TransferMPO& mpo, const BMPS<TenElemT, QNT>& opposite_boundary, size_t target_col);

    /**
     * @brief Grow the LEFT BTen by one column.
     * 
     * Absorbs the column at the current LEFT BTen edge and advances the edge by one.
     *
     * @param mpo The MPO row.
     * @param opposite_boundary The opposing BMPS.
     */
    void GrowBTenLeftStep(const TransferMPO& mpo, const BMPS<TenElemT, QNT>& opposite_boundary);

    /**
     * @brief Grow the RIGHT BTen by one column (towards left).
     *
     * @param mpo The MPO row.
     * @param opposite_boundary The opposing BMPS.
     */
    void GrowBTenRightStep(const TransferMPO& mpo, const BMPS<TenElemT, QNT>& opposite_boundary);

    /**
     * @brief Compute trace at a specific site using cached BTen.
     * 
     * This is much more efficient than ContractRow when you need to compute
     * multiple traces on the same row with different site replacements.
     *
     * ## Prerequisite
     * Either:
     * - Call InitBTenLeft() to grow LEFT BTen up to `site_col`, then use this with RIGHT BTen grown from right, or
     * - Have both LEFT and RIGHT BTen meeting at `site_col`.
     *
     * @param site The replacement site tensor.
     * @param site_col The column index of the site.
     * @param opposite_boundary The opposing BMPS.
     * @return The scalar trace result.
     */
    TenElemT TraceWithBTen(const Tensor& site, size_t site_col, const BMPS<TenElemT, QNT>& opposite_boundary) const;

    /**
     * @brief Clear all BTen caches.
     */
    void ClearBTen() { bten_left_.clear(); bten_right_.clear(); bten_left_col_ = 0; bten_right_col_ = 0; }

    /**
     * @brief Get the current left BTen edge column (exclusive upper bound).
     */
    size_t GetBTenLeftCol() const { return bten_left_col_; }

    /**
     * @brief Get the current right BTen edge column (exclusive lower bound).
     */
    size_t GetBTenRightCol() const { return bten_right_col_; }

    /**
     * @brief Get the current underlying stack size.
     * Indicates how many lattice layers have been effectively absorbed into the current boundary.
     */
    size_t GetStackSize() const { return stack_size_; }

    /**
     * @brief Get the evolution direction.
     */
    BMPSPOSITION GetPosition() const { return pos_; }

   private:
    const TensorNetwork2D<TenElemT, QNT>& tn_;
    BMPS<TenElemT, QNT> bmps_;
    BMPSPOSITION pos_;
    size_t stack_size_;

    // BTen cache for efficient multi-site trace calculations
    // bten_left_[i] represents the environment from col=0 to col=i (inclusive)
    // bten_right_[i] represents the environment from col=N-1 to col=N-1-i (inclusive)
    mutable std::vector<Tensor> bten_left_;
    mutable std::vector<Tensor> bten_right_;
    mutable size_t bten_left_col_ = 0;   // Current left edge (exclusive): BTen covers [0, bten_left_col_)
    mutable size_t bten_right_col_ = 0;  // Current right edge (exclusive): BTen covers (bten_right_col_, N-1]
  };

  /**
   * @brief Creates a BMPSWalker initialized with the current boundary state at the given position.
   *
   * @param tn The tensor network (must be the one being contracted).
   * @param position The boundary position to fork from.
   * @return BMPSWalker A detached walker ready for independent evolution.
   */
  BMPSWalker GetWalker(const TensorNetwork2D<TenElemT, QNT>& tn, BMPSPOSITION position) const;

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
   * @brief Shifts the single-slice boundary tensor window by one site.
   * 
   * This pops the outermost BTen at `position` and grows a new one at `Opposite(position)`.
   * 
   * @param tn The tensor network.
   * @param position The position/direction to shift toward.
   */
  void ShiftBTenWindow(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position);

  /// @deprecated Use ShiftBTenWindow instead
  [[deprecated("Use ShiftBTenWindow instead")]]
  void BTenMoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position) {
    ShiftBTenWindow(tn, position);
  }

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
   * @brief Shifts the double-slice boundary tensor window by one site.
   * 
   * This pops the outermost BTen2 at `position` and grows a new one at `Opposite(position)`.
   * 
   * @param tn The tensor network.
   * @param position The position/direction to shift toward.
   * @param slice_num1 The first slice index (used for context).
   */
  void ShiftBTen2Window(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1);

  /// @deprecated Use ShiftBTen2Window instead
  [[deprecated("Use ShiftBTen2Window instead")]]
  void BTen2MoveStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION position, const size_t slice_num1) {
    ShiftBTen2Window(tn, position, slice_num1);
  }

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

  void InitBMPS(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION post);

  size_t GrowBMPSStep(const BMPSPOSITION position, TransferMPO, const BMPSTruncateParams<RealT> &);

  size_t GrowBMPSStep(const TensorNetwork2D<TenElemT, QNT>& tn, const BMPSPOSITION position, const BMPSTruncateParams<RealT> &);

  void GrowBTen2Step(const TensorNetwork2D<TenElemT, QNT>& tn, const BTenPOSITION post, const size_t slice_num1);

 private:
  std::map<BMPSPOSITION, std::vector<BMPS<TenElemT, QNT>>> bmps_set_;
  std::map<BTenPOSITION, std::vector<Tensor>> bten_set_;  // for 1 layer between two bmps
  std::map<BTenPOSITION, std::vector<Tensor>> bten_set2_; // for 2 layers between two bmps
  
  size_t rows_;
  size_t cols_;
};

} // namespace qlpeps

#include "bmps_contractor_impl.h"

#endif //QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_BMPS_CONTRACTOR_H
