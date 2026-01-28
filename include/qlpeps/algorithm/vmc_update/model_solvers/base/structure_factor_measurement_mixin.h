// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-01-02
*
* Description: QuantumLiquids/PEPS project. Structure Factor Measurement Mixin.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BASE_STRUCTURE_FACTOR_MEASUREMENT_MIXIN_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BASE_STRUCTURE_FACTOR_MEASUREMENT_MIXIN_H

#include "qlpeps/basic.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h"

namespace qlpeps {

/**
 * @brief Mixin for measuring Structure Factor (S+S- correlations) using BMPSWalker.
 *
 * This mixin provides the `MeasureStructureFactor` method, which utilizes the
 * `BMPSWalker` to efficiently calculate non-local correlations by "forking"
 * the boundary evolution.
 *
 * ## Algorithm Overview
 * 
 * For spin structure factor \f$ S(\mathbf{q}) = \sum_{i,j} e^{i\mathbf{q}\cdot(\mathbf{r}_i - \mathbf{r}_j)} 
 * \langle S_i^+ S_j^- \rangle \f$, we need to compute all pair correlations \f$ \langle S_i^+ S_j^- \rangle \f$.
 *
 * The "Excited State Propagation" algorithm works as follows:
 * 1. Grow UP BMPS from top, creating main_walker at some reference row.
 * 2. For each source site (y1, x1) with spin-down:
 *    - Fork the walker: `excited_walker = main_walker`
 *    - Inject S+ operator (replace site tensor with flipped spin) and absorb row y1
 * 3. For each target row y2 > y1:
 *    - For each target site (y2, x2) with spin-up:
 *      - Build target MPO with S- at (y2, x2)
 *      - ContractRow with the appropriate DOWN boundary to get the overlap
 *    - Absorb row y2 with standard MPO (for next y2 iteration)
 *
 * ## BMPS Storage Order
 * 
 * - **UP BMPS** (walker): Reversed storage. `bmps_[0]` = rightmost column.
 * - **DOWN BMPS** (from contractor): Normal storage. `down_stack[k]` has absorbed k layers from bottom.
 *   - `down_stack[0]`: vacuum (no rows absorbed)
 *   - `down_stack[Ly-1-y]`: absorbed rows [y+1, Ly-1], ready for row y contraction
 *
 * ## Requirements
 * - ModelType must provide `GetSiteTensor(split_index_tps, row, col, spin_val)`.
 * - Contractor must have DOWN BMPS grown to sufficient depth before calling this method.
 * - This implementation assumes spin-1/2 S+ / S- operators.
 *
 * @tparam ModelType The derived model class (CRTP pattern).
 */
template<typename ModelType>
class StructureFactorMeasurementMixin {
 public:
  void SetEnableStructureFactor(bool enable) { enable_structure_factor_measurement_ = enable; }

  bool IsStructureFactorEnabled() const { return enable_structure_factor_measurement_; }

  /**
   * @brief Measure S+S- correlations across the lattice using Excited State Propagation.
   *
   * @param tn The tensor network representing the physical state.
   * @param split_index_tps The split-index TPS for obtaining site tensors.
   * @param contractor The BMPS contractor with pre-grown boundaries.
   * @param config The spin configuration (0 = down, 1 = up).
   * @param[out] out Observable map to store results under key "SpSm_cross".
   * @param trunc_para Truncation parameters for BMPS evolution.
   *
   * Results are stored as flat tuples: {y1, x1, y2, x2, val, ...} where:
   * - (y1, x1): source site where S+ is applied
   * - (y2, x2): target site where S- is applied
   * - val: the correlation value (ratio of overlaps, needs normalization by caller)
   *
   * ## Implementation with BTen Optimization
   * 
   * Uses BTen (Boundary Tensor) caching for efficient multi-site trace calculations.
   * For each target row y2, the algorithm:
   * 1. Builds complete LEFT BTen from col=0 to col=Lx-1 using standard_mpo
   * 2. Scans x2 from right to left, using TraceWithBTen for O(1) per-site calculation
   * 3. Incrementally grows RIGHT BTen as x2 moves left
   *
   * This reduces complexity from O(Lx^2) to O(Lx) per target row.
   */
  template<typename TenElemT, typename QNT>
  void MeasureStructureFactor(
      const TensorNetwork2D<TenElemT, QNT>& tn,
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      const BMPSContractor<TenElemT, QNT>& contractor,
      const Configuration& config,
      ObservableMap<TenElemT>& out) {
    
    if (!enable_structure_factor_measurement_) return;

    using Tensor = qlten::QLTensor<TenElemT, QNT>;
    using TransferMPO = std::vector<Tensor *>;
    const auto& trunc_params = contractor.GetTruncateParams();

    const size_t Ly = tn.rows();
    const size_t Lx = tn.cols();

    std::vector<TenElemT> spsm_cross;
    
    // Get the DOWN BMPS stack for bottom environments
    const auto& down_stack = contractor.GetBMPS(DOWN);
    
    // Get the UP BMPS stack for vacuum state
    const auto& up_stack = contractor.GetBMPS(UP);
    if (up_stack.empty()) {
      std::cerr << "[MeasureStructureFactor] ERROR: UP BMPS stack is empty!" << std::endl;
      return;
    }
    
    // Create main_walker from the VACUUM state (up_stack[0]), not from stack.back()!
    // This way we start from the top of the lattice and can evolve row by row.
    auto main_walker = typename BMPSContractor<TenElemT, QNT>::BMPSWalker(
        tn, up_stack[0], UP, 1, trunc_params);

    // Iterate over all source sites (y1, x1) and target sites (y2, x2) with y2 > y1
    // Record ALL pairs including zeros for consistent vector length across samples
    for (size_t y1 = 0; y1 < Ly - 1; ++y1) {  // y1 < Ly-1 because we need at least one row below
      for (size_t x1 = 0; x1 < Lx; ++x1) {
        // Check if S+ can act: S+|↓⟩ = |↑⟩, but S+|↑⟩ = 0
        const bool source_is_spin_down = (config({y1, x1}) == 0);
        
        // Fork the walker at row y1 (only if source can contribute)
        auto excited_walker = main_walker;
        
        // Build excited MPO: replace tensor at (y1, x1) with flipped spin (S+ applied)
        TransferMPO excited_mpo_ptrs = tn.get_row(y1);
        Tensor excited_tensor = static_cast<const ModelType*>(this)->GetSiteTensor(split_index_tps, y1, x1, 1);
        if (source_is_spin_down) {
          excited_mpo_ptrs[x1] = &excited_tensor;
        }
        // If source is spin-up, S+|↑⟩ = 0, all correlations from this source are zero
        // We still evolve to maintain consistent loop structure
        
        // Absorb the excited row y1
        excited_walker.Evolve(excited_mpo_ptrs);
        
        // Propagate to target rows y2 > y1
        for (size_t y2 = y1 + 1; y2 < Ly; ++y2) {
          // DOWN stack index for row y2: need boundary that has absorbed rows [y2+1, Ly-1]
          size_t down_stack_idx = Ly - 1 - y2;
          
          if (down_stack_idx >= down_stack.size()) {
            // Still record zeros for all x2 in this row
            for (size_t x2 = 0; x2 < Lx; ++x2) {
              spsm_cross.push_back(TenElemT(y1));
              spsm_cross.push_back(TenElemT(x1));
              spsm_cross.push_back(TenElemT(y2));
              spsm_cross.push_back(TenElemT(x2));
              spsm_cross.push_back(TenElemT(0));
            }
            continue;
          }
          const auto& bottom_env = down_stack[down_stack_idx];
          
          // Build standard MPO for this row
          TransferMPO standard_mpo = tn.get_row(y2);
          
          // BTen optimization: Build LEFT BTen fully, then scan right-to-left
          // Initialize LEFT BTen to cover all columns (for any x2 we might need)
          excited_walker.InitBTenLeft(standard_mpo, bottom_env, Lx);
          
          // Initialize RIGHT BTen at the right boundary (nothing absorbed yet)
          excited_walker.InitBTenRight(standard_mpo, bottom_env, Lx - 1);
          
          // Temporary storage for results (to maintain left-to-right order in output)
          std::vector<TenElemT> row_results(Lx, TenElemT(0));
          
          // Scan from right to left for efficient BTen-based trace
          for (size_t x2_rev = 0; x2_rev < Lx; ++x2_rev) {
            size_t x2 = Lx - 1 - x2_rev;  // x2 goes from Lx-1 down to 0
            
            // Only compute non-zero if S+ and S- both contribute:
            // S+|↓⟩ ≠ 0 (source spin-down) AND S-|↑⟩ ≠ 0 (target spin-up)
            const bool target_is_spin_up = (config({y2, x2}) == 1);
            
            if (source_is_spin_down && target_is_spin_up) {
              // Get the S- tensor (spin flipped from 1 to 0)
              Tensor target_tensor = static_cast<const ModelType*>(this)->GetSiteTensor(split_index_tps, y2, x2, 0);
              
              // Use TraceWithBTen for O(1) single-site trace
              // BTen is guaranteed to be initialized by InitBTenLeft/Right above
              row_results[x2] = excited_walker.TraceWithBTen(target_tensor, x2, bottom_env);
            }
            
            // Grow RIGHT BTen by one step (absorb column x2 into RIGHT BTen)
            if (x2 > 0) {
              excited_walker.GrowBTenRightStep(standard_mpo, bottom_env);
            }
          }
          
          // Output results in left-to-right order
          for (size_t x2 = 0; x2 < Lx; ++x2) {
            spsm_cross.push_back(TenElemT(y1));
            spsm_cross.push_back(TenElemT(x1));
            spsm_cross.push_back(TenElemT(y2));
            spsm_cross.push_back(TenElemT(x2));
            spsm_cross.push_back(row_results[x2]);
          }
          
          // Clear BTen cache before evolving
          excited_walker.ClearBTen();
          
          // Absorb row y2 with standard MPO for next y2 iteration
          if (y2 < Ly - 1) {
            excited_walker.Evolve(standard_mpo);
          }
        }
      }
      
      // Advance main_walker by absorbing row y1
      TransferMPO standard_mpo = tn.get_row(y1);
      main_walker.Evolve(standard_mpo);
    }
    
    if (!spsm_cross.empty()) {
      out["SpSm_cross"] = std::move(spsm_cross);
    }
  }

 protected:
  bool enable_structure_factor_measurement_ = false;
};

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BASE_STRUCTURE_FACTOR_MEASUREMENT_MIXIN_H
