// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-01-04
*
* Description: QuantumLiquids/PEPS project. Singlet Pair Correlation Measurement Mixin for t-J Model.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BASE_SINGLET_PAIR_CORRELATION_MEASUREMENT_MIXIN_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BASE_SINGLET_PAIR_CORRELATION_MEASUREMENT_MIXIN_H

#include <array>
#include <iostream>
#include <sstream>
#include "qlpeps/basic.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h"
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h"
#include "qlpeps/vmc_basic/tj_single_site_state.h"

namespace qlpeps {

/**
 * @brief Mixin for measuring singlet pair correlation functions in t-J model.
 *
 * Measures the superconducting pairing correlation:
 * \f[
 *   \langle \Delta^\dagger(\text{ref\_bond}) \Delta(\text{target\_bond}) \rangle
 * \f]
 * where the singlet pair operator is:
 * \f[
 *   \Delta^\dagger_{ij} = \frac{1}{\sqrt{2}} (c_{i\uparrow}^\dagger c_{j\downarrow}^\dagger 
 *                         - c_{i\downarrow}^\dagger c_{j\uparrow}^\dagger)
 * \f]
 *
 * ## Algorithm Overview (Excited State Propagation)
 *
 * For bond pairs where reference bond is at row y1 and target bond is at row y2 > y1:
 * 1. Create main_walker from UP BMPS vacuum, evolve row by row.
 * 2. For each reference bond (y1, x1) with both sites empty:
 *    - Fork walker
 *    - Inject Δ† operator: replace site tensors with (up,down) - (down,up) combination
 *    - Evolve through row y1
 * 3. For each target bond (y2, x2) with (up,down) or (down,up):
 *    - Contract with Δ operator: replace site tensors with (empty,empty)
 *    - Compute overlap ratio
 *
 * ## Physical Interpretation
 *
 * In t-J model:
 * - Δ† creates a Cooper pair from two empty sites
 * - Δ annihilates a Cooper pair to create two empty sites
 * - ⟨Δ†Δ⟩ measures d-wave superconducting pairing correlation
 *
 * ## Current Implementation
 *
 * Supports:
 * - Horizontal reference bonds only (ref along x-direction)
 * - Horizontal target bonds only (target along x-direction)
 * - Only bonds in different rows (y2 > y1, no site overlap)
 * - Skip same-row correlations to avoid complex 4-site contractions
 *
 * TODO: Vertical target bonds require BTen2 in BMPSWalker, not yet implemented.
 *
 * ## Selection Rules
 *
 * For non-zero contribution:
 * - Reference bond: both sites must be empty (state = 2)
 * - Target bond: must be (up,down) or (down,up) pair
 *
 * ## Output Format
 *
 * MeasureSingletPairCorrelation() stores only correlation values (one per bond pair).
 * Use GenerateSCPairCorrIndexMapping() to get coordinate mapping:
 *
 * @code
 * // Get values from MC measurement
 * auto values = registry["SC_singlet_pair_corr"];  // size = num_pairs
 *
 * // Generate coordinate mapping (once, for post-processing)
 * auto mapping = SingletPairCorrelationMixin<M>::GenerateSCPairCorrIndexMapping(Ly, Lx);
 * for (size_t i = 0; i < mapping.size(); ++i) {
 *     auto [ref_y, ref_x, ref_orient, tgt_y, tgt_x, tgt_orient] = mapping[i];
 *     std::cout << "Pair " << i << ": bond(" << ref_y << "," << ref_x << ")@" << ref_orient
 *               << " -> bond(" << tgt_y << "," << tgt_x << ")@" << tgt_orient
 *               << " = " << values[i] << std::endl;
 * }
 * @endcode
 *
 * @tparam ModelType The derived model class (CRTP pattern).
 */
template<typename ModelType>
class SingletPairCorrelationMixin {
 public:
  void SetEnableSingletPairCorrelation(bool enable) { enable_singlet_pair_correlation_ = enable; }

  bool IsSingletPairCorrelationEnabled() const { return enable_singlet_pair_correlation_; }

  /**
   * @brief Compute the number of singlet pair correlation bond pairs.
   *
   * For horizontal reference bonds at row y1, targets are horizontal bonds at rows y2 > y1.
   * Reference bonds: (Ly-1) rows × (Lx-1) cols.
   * For ref at y1: (Ly-1-y1) target rows × (Lx-1) target bonds per row.
   *
   * @param Ly Number of rows.
   * @param Lx Number of columns.
   * @return Total number of bond pairs.
   */
  static size_t ComputeNumSCPairs(size_t Ly, size_t Lx) {
    if (Ly < 2 || Lx < 2) return 0;
    size_t count = 0;
    for (size_t y1 = 0; y1 < Ly - 1; ++y1) {
      for (size_t x1 = 0; x1 < Lx - 1; ++x1) {
        count += (Ly - 1 - y1) * (Lx - 1);
      }
    }
    return count;
  }

  /**
   * @brief Generate coordinate mapping for singlet pair correlation indices.
   *
   * Each entry: {ref_y, ref_x, ref_orient, tgt_y, tgt_x, tgt_orient}
   * where orient = 0 for horizontal bonds.
   *
   * This should be called once when dumping results, not per MC sample.
   * The ordering matches MeasureSingletPairCorrelation() output exactly.
   *
   * @param Ly Number of rows.
   * @param Lx Number of columns.
   * @return Vector of coordinate tuples, one per bond pair.
   */
  static std::vector<std::array<size_t, 6>> GenerateSCPairCorrIndexMapping(size_t Ly, size_t Lx) {
    std::vector<std::array<size_t, 6>> mapping;
    if (Ly < 2 || Lx < 2) return mapping;

    mapping.reserve(ComputeNumSCPairs(Ly, Lx));

    // Enumerate all horizontal reference bonds (y1, x1)-(y1, x1+1)
    for (size_t y1 = 0; y1 < Ly - 1; ++y1) {
      for (size_t x1 = 0; x1 < Lx - 1; ++x1) {
        // Enumerate all horizontal target bonds in rows y2 > y1
        for (size_t y2 = y1 + 1; y2 < Ly; ++y2) {
          for (size_t x2 = 0; x2 < Lx - 1; ++x2) {
            mapping.push_back({y1, x1, 0, y2, x2, 0});  // 0 = horizontal
          }
        }
      }
    }

    return mapping;
  }

  /**
   * @brief Generate coordinate mapping as a string for file output.
   *
   * Format:
   * # idx ref_y ref_x ref_orient tgt_y tgt_x tgt_orient
   * 0 0 0 0 1 0 0
   * 1 0 0 0 1 1 0
   * ...
   *
   * This is suitable for use with ObservableMeta::coord_generator.
   *
   * @param Ly Number of rows.
   * @param Lx Number of columns.
   * @return Coordinate mapping as string content.
   */
  static std::string GenerateSCPairCorrCoordString(size_t Ly, size_t Lx) {
    auto mapping = GenerateSCPairCorrIndexMapping(Ly, Lx);
    std::ostringstream oss;
    oss << "# idx ref_y ref_x ref_orient tgt_y tgt_x tgt_orient\n";
    for (size_t i = 0; i < mapping.size(); ++i) {
      const auto& m = mapping[i];
      oss << i << " " << m[0] << " " << m[1] << " " << m[2] << " "
          << m[3] << " " << m[4] << " " << m[5] << "\n";
    }
    return oss.str();
  }

  /**
   * @brief Measure singlet pair correlations between horizontal bonds.
   *
   * Uses the "Excited State Propagation" algorithm with BTen optimization:
   * 1. Fork walker at reference bond row, inject Δ† operator
   * 2. For each target row, use BTen-based trace with 2-site replacement
   *
   * @param tn The tensor network representing the physical state.
   * @param split_index_tps The split-index TPS for obtaining site tensors.
   * @param contractor The BMPS contractor with pre-grown boundaries.
   * @param config The t-J configuration (0=up, 1=down, 2=empty).
   * @param[out] out Observable map to store results under key "SC_singlet_pair_corr".
   * @param trunc_para Truncation parameters for BMPS evolution.
   *
   * Results stored as correlation values only (one per bond pair).
   * Use GenerateSCPairCorrIndexMapping() to obtain coordinate mapping.
   */
  template<typename TenElemT, typename QNT>
  void MeasureSingletPairCorrelation(
      const TensorNetwork2D<TenElemT, QNT>& tn,
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      const BMPSContractor<TenElemT, QNT>& contractor,
      const Configuration& config,
      ObservableMap<TenElemT>& out,
      const BMPSTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type>& trunc_para) {
    
    if (!enable_singlet_pair_correlation_) return;

    using Tensor = qlten::QLTensor<TenElemT, QNT>;
    using TransferMPO = std::vector<Tensor *>;

    const size_t Ly = tn.rows();
    const size_t Lx = tn.cols();

    if (Ly < 2 || Lx < 2) return;  // Need at least 2x2 lattice

    std::vector<TenElemT> sc_corr;
    
    // Get the DOWN BMPS stack for bottom environments
    const auto& down_stack = contractor.GetBMPS(DOWN);
    
    // Get the UP BMPS stack for vacuum state
    const auto& up_stack = contractor.GetBMPS(UP);
    if (up_stack.empty()) {
      std::cerr << "[MeasureSingletPairCorrelation] ERROR: UP BMPS stack is empty!" << std::endl;
      return;
    }
    
    // Create main_walker from the VACUUM state (up_stack[0])
    auto main_walker = typename BMPSContractor<TenElemT, QNT>::BMPSWalker(
        tn, up_stack[0], UP, 1);

    // Enumerate all horizontal reference bonds
    for (size_t y1 = 0; y1 < Ly - 1; ++y1) {  // y1 < Ly-1 because we need at least one row below
      for (size_t x1 = 0; x1 < Lx - 1; ++x1) {  // x1 < Lx-1 for horizontal bond
        
        const SiteIdx site1_ref{y1, x1};
        const SiteIdx site2_ref{y1, x1 + 1};
        
        // Check if Δ† can act: both sites must be empty
        const bool ref_valid = (config(site1_ref) == static_cast<size_t>(tJSingleSiteState::Empty) &&
                                config(site2_ref) == static_cast<size_t>(tJSingleSiteState::Empty));
        
        // Fork the walker at row y1
        auto excited_walker = main_walker;
        
        // Build excited MPOs for Δ† action
        // Δ†|empty,empty⟩ = (|↑,↓⟩ - |↓,↑⟩)/√2
        TransferMPO excited_mpo_up_down = tn.get_row(y1);
        TransferMPO excited_mpo_down_up = tn.get_row(y1);
        
        Tensor tensor_up = GetSiteTensorImpl<TenElemT, QNT>(split_index_tps, y1, x1, 
                                                            static_cast<size_t>(tJSingleSiteState::SpinUp));
        Tensor tensor_down_at_x1 = GetSiteTensorImpl<TenElemT, QNT>(split_index_tps, y1, x1, 
                                                                    static_cast<size_t>(tJSingleSiteState::SpinDown));
        Tensor tensor_down = GetSiteTensorImpl<TenElemT, QNT>(split_index_tps, y1, x1 + 1, 
                                                              static_cast<size_t>(tJSingleSiteState::SpinDown));
        Tensor tensor_up_at_x1p1 = GetSiteTensorImpl<TenElemT, QNT>(split_index_tps, y1, x1 + 1, 
                                                                    static_cast<size_t>(tJSingleSiteState::SpinUp));
        
        if (ref_valid) {
          excited_mpo_up_down[x1] = &tensor_up;
          excited_mpo_up_down[x1 + 1] = &tensor_down;
          
          excited_mpo_down_up[x1] = &tensor_down_at_x1;
          excited_mpo_down_up[x1 + 1] = &tensor_up_at_x1p1;
        }
        
        // Evolve two excited walkers (for +/- combination)
        auto excited_walker_up_down = excited_walker;
        auto excited_walker_down_up = excited_walker;
        
        excited_walker_up_down.Evolve(excited_mpo_up_down, trunc_para);
        excited_walker_down_up.Evolve(excited_mpo_down_up, trunc_para);
        
        // Propagate to target rows y2 > y1
        for (size_t y2 = y1 + 1; y2 < Ly; ++y2) {
          // DOWN stack index for row y2
          size_t down_stack_idx = Ly - 1 - y2;
          
          if (down_stack_idx >= down_stack.size()) {
            // Record zeros for all horizontal bonds in this row
            for (size_t x2 = 0; x2 < Lx - 1; ++x2) {
              sc_corr.push_back(TenElemT(0));  // value only
            }
            continue;
          }
          const auto& bottom_env = down_stack[down_stack_idx];
          
          // Build standard and target MPOs for this row
          TransferMPO standard_mpo = tn.get_row(y2);
          
          // Initialize BTen for both walkers
          excited_walker_up_down.InitBTenLeft(standard_mpo, bottom_env, Lx);
          excited_walker_up_down.InitBTenRight(standard_mpo, bottom_env, Lx - 1);
          excited_walker_down_up.InitBTenLeft(standard_mpo, bottom_env, Lx);
          excited_walker_down_up.InitBTenRight(standard_mpo, bottom_env, Lx - 1);
          
          // Results for this target row
          std::vector<TenElemT> row_results(Lx - 1, TenElemT(0));
          
          // Enumerate target horizontal bonds
          for (size_t x2 = 0; x2 < Lx - 1; ++x2) {
            const SiteIdx site1_tgt{y2, x2};
            const SiteIdx site2_tgt{y2, x2 + 1};
            
            // Check if Δ can act: need (up,down) or (down,up) pair
            const bool tgt_is_up_down = (config(site1_tgt) == static_cast<size_t>(tJSingleSiteState::SpinUp) &&
                                          config(site2_tgt) == static_cast<size_t>(tJSingleSiteState::SpinDown));
            const bool tgt_is_down_up = (config(site1_tgt) == static_cast<size_t>(tJSingleSiteState::SpinDown) &&
                                          config(site2_tgt) == static_cast<size_t>(tJSingleSiteState::SpinUp));
            
            if (ref_valid && (tgt_is_up_down || tgt_is_down_up)) {
              // For two-site replacement, we need to use TraceWithBTen twice or build a combined trace.
              // Simpler approach: use the full row contraction with modified MPO.
              // Build target MPO with Δ operator: replace with (empty, empty)
              TransferMPO target_mpo = tn.get_row(y2);
              Tensor empty_tensor1 = GetSiteTensorImpl<TenElemT, QNT>(split_index_tps, y2, x2,
                                                                      static_cast<size_t>(tJSingleSiteState::Empty));
              Tensor empty_tensor2 = GetSiteTensorImpl<TenElemT, QNT>(split_index_tps, y2, x2 + 1,
                                                                      static_cast<size_t>(tJSingleSiteState::Empty));
              target_mpo[x2] = &empty_tensor1;
              target_mpo[x2 + 1] = &empty_tensor2;
              
              // Use TraceWithTwoSiteReplacement: compute the trace with two adjacent sites replaced.
              // This requires summing over all BTen configurations between the two sites.
              TenElemT overlap_up_down = ComputeTwoSiteTrace_(
                  excited_walker_up_down, target_mpo, bottom_env, x2);
              TenElemT overlap_down_up = ComputeTwoSiteTrace_(
                  excited_walker_down_up, target_mpo, bottom_env, x2);
              
              if (tgt_is_up_down) {
                // ⟨Δ†Δ⟩ = (1/2) [⟨↑↓|00⟩ - ⟨↓↑|00⟩]
                row_results[x2] = TenElemT(0.5) * ComplexConjugate(overlap_up_down - overlap_down_up);
              } else {  // tgt_is_down_up
                // ⟨Δ†Δ⟩ = (1/2) [-⟨↑↓|00⟩ + ⟨↓↑|00⟩]
                row_results[x2] = TenElemT(0.5) * ComplexConjugate(-overlap_up_down + overlap_down_up);
              }
            }
          }
          
          // Output horizontal results (values only)
          for (size_t x2 = 0; x2 < Lx - 1; ++x2) {
            sc_corr.push_back(row_results[x2]);
          }
          
          // TODO: Vertical target bonds (y2, x2) -> (y2+1, x2)
          // Requires BTen2 implementation in BMPSWalker for cross-row contraction.
          
          // Clear BTen cache before evolving
          excited_walker_up_down.ClearBTen();
          excited_walker_down_up.ClearBTen();
          
          // Evolve excited walkers to next row (with standard MPO)
          if (y2 < Ly - 1) {
            excited_walker_up_down.Evolve(standard_mpo, trunc_para);
            excited_walker_down_up.Evolve(standard_mpo, trunc_para);
          }
        }
      }
      
      // Advance main_walker by absorbing row y1
      TransferMPO standard_mpo = tn.get_row(y1);
      main_walker.Evolve(standard_mpo, trunc_para);
    }
    
    if (!sc_corr.empty()) {
      out["SC_singlet_pair_corr"] = std::move(sc_corr);
    }
  }

 private:
  /**
   * @brief Compute trace with two adjacent sites replaced at columns x and x+1.
   *
   * Performs a full row contraction with the walker's BMPS (excited state),
   * the target MPO (with replacement tensors), and the bottom environment.
   *
   * @param walker The excited walker.
   * @param mpo The modified MPO with replacement tensors at x and x+1.
   * @param bottom_env The bottom BMPS boundary.
   * @param x The left column of the two-site replacement (unused, for future optimization).
   * @return The trace value.
   */
  template<typename TenElemT, typename QNT>
  TenElemT ComputeTwoSiteTrace_(
      typename BMPSContractor<TenElemT, QNT>::BMPSWalker& walker,
      const std::vector<qlten::QLTensor<TenElemT, QNT>*>& mpo,
      const BMPS<TenElemT, QNT>& bottom_env,
      [[maybe_unused]] size_t x) const {
    
    using Tensor = qlten::QLTensor<TenElemT, QNT>;
    const size_t N = mpo.size();
    const auto& bmps = walker.GetBMPS();
    
    if (N == 0 || bmps.size() != N || bottom_env.size() != N) {
      return TenElemT(0);
    }
    
    // Full row contraction: contract all columns from left to right
    // This is simpler and more robust than trying to use BTen for two-site replacement.
    //
    // Tensor network structure (for UP walker with DOWN bottom_env):
    //   UP BMPS (reversed storage): bmps[N-1-col] = column col
    //   MPO: mpo[col] = site tensor at column col
    //   DOWN BMPS (normal storage): bottom_env[col] = column col
    //
    // For each column, contract:
    //   top[physical=1] with site[up=3] -> (top_L, top_R, site_L, site_D, site_R)
    //   then [site_D=3] with bot[physical=1] -> (top_L, top_R, site_L, site_R, bot_L, bot_R)
    //
    // Connect columns:
    //   acc[top_R, site_R, bot_R] with col[top_L, site_L, bot_L]
    
    Tensor acc;
    for (size_t col = 0; col < N; ++col) {
      const Tensor& top = bmps[N - 1 - col];  // UP reversed
      const Tensor& site = *mpo[col];
      const Tensor& bot = bottom_env[col];
      
      // top: (left, physical, right) = (0, 1, 2)
      // site: (left, down, right, up) = (0, 1, 2, 3)
      // bot: (left, physical, right) = (0, 1, 2)
      
      // Contract top[1] with site[3]
      Tensor top_site;
      qlten::Contract(&top, {1}, &site, {3}, &top_site);
      // top_site: (top_L, top_R, site_L, site_D, site_R) = (0, 1, 2, 3, 4)
      
      // Contract top_site[3] with bot[1]
      Tensor col_tensor;
      qlten::Contract(&top_site, {3}, &bot, {1}, &col_tensor);
      // col_tensor: (top_L, top_R, site_L, site_R, bot_L, bot_R) = (0, 1, 2, 3, 4, 5)
      
      if (col == 0) {
        acc = std::move(col_tensor);
      } else {
        // Connect acc's right bonds to col_tensor's left bonds
        // acc: (..., top_R, site_R, bot_R) at (r-3, r-2, r-1)
        // col_tensor: (top_L, top_R, site_L, site_R, bot_L, bot_R) = (0, 1, 2, 3, 4, 5)
        // Connect: acc{r-3, r-2, r-1} with col{0, 2, 4}
        size_t r = acc.Rank();
        Tensor new_acc;
        qlten::Contract(&acc, {r-3, r-2, r-1}, &col_tensor, {0, 2, 4}, &new_acc);
        acc = std::move(new_acc);
      }
    }
    
    // acc should now have 6 trivial indices for OBC
    if (acc.Rank() == 0) {
      return acc();
    }
    
    // Extract scalar from trivial indices
    std::vector<size_t> coords(acc.Rank(), 0);
    return acc.GetElem(coords);
  }

 protected:
  bool enable_singlet_pair_correlation_ = false;

 private:
  /**
   * @brief Get site tensor for a given configuration.
   * 
   * Uses CRTP to delegate to the derived model class.
   */
  template<typename TenElemT, typename QNT>
  qlten::QLTensor<TenElemT, QNT> GetSiteTensorImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      size_t row, size_t col, size_t state_val) const {
    return (*split_index_tps)({row, col})[state_val];
  }
};

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BASE_SINGLET_PAIR_CORRELATION_MEASUREMENT_MIXIN_H

