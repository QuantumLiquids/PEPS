// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-01-04
*
* Description: QuantumLiquids/PEPS project. Singlet Pair Correlation Measurement Mixin for t-J Model.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BASE_SINGLET_PAIR_CORRELATION_MEASUREMENT_MIXIN_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BASE_SINGLET_PAIR_CORRELATION_MEASUREMENT_MIXIN_H

#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <sstream>
#include <utility>
#include <vector>
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
   * @brief Set selected reference bonds for singlet pair correlation measurement.
   *
   * Only the specified reference bonds will be measured, reducing computational cost.
   * Each bond is specified by (y, x) coordinates of the left site.
   * Empty list means all horizontal reference bonds will be measured.
   *
   * @param bonds Vector of (y, x) coordinates for reference bonds.
   */
  void SetSelectedRefBonds(std::vector<std::pair<size_t, size_t>> bonds) {
    selected_ref_bonds_ = std::move(bonds);
  }

  /**
   * @brief Get the currently selected reference bonds.
   * @return Const reference to the selected bonds vector.
   */
  const std::vector<std::pair<size_t, size_t>>& GetSelectedRefBonds() const {
    return selected_ref_bonds_;
  }

  /**
   * @brief Compute the number of singlet pair correlation bond pairs.
   *
   * For horizontal reference bonds at row y1, targets are horizontal bonds at rows y2 > y1.
   * Reference bonds: (Ly-1) rows × (Lx-1) cols.
   * For ref at y1: (Ly-1-y1) target rows × (Lx-1) target bonds per row.
   *
   * @param Ly Number of rows.
   * @param Lx Number of columns.
   * @param ref_bonds Optional list of selected reference bonds (y, x). Empty means all.
   * @return Total number of bond pairs.
   */
  static size_t ComputeNumSCPairs(
      size_t Ly, size_t Lx,
      const std::vector<std::pair<size_t, size_t>>& ref_bonds = {}) {
    if (Ly < 2 || Lx < 2) return 0;
    size_t count = 0;
    
    if (ref_bonds.empty()) {
      // All reference bonds
      for (size_t y1 = 0; y1 < Ly - 1; ++y1) {
        for (size_t x1 = 0; x1 < Lx - 1; ++x1) {
          count += (Ly - 1 - y1) * (Lx - 1);
        }
      }
    } else {
      // Only selected reference bonds
      for (const auto& bond : ref_bonds) {
        size_t y1 = bond.first;
        size_t x1 = bond.second;
        if (y1 < Ly - 1 && x1 < Lx - 1) {
          count += (Ly - 1 - y1) * (Lx - 1);
        }
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
   * @param ref_bonds Optional list of selected reference bonds (y, x). Empty means all.
   * @return Vector of coordinate tuples, one per bond pair.
   */
  static std::vector<std::array<size_t, 6>> GenerateSCPairCorrIndexMapping(
      size_t Ly, size_t Lx,
      const std::vector<std::pair<size_t, size_t>>& ref_bonds = {}) {
    std::vector<std::array<size_t, 6>> mapping;
    if (Ly < 2 || Lx < 2) return mapping;

    mapping.reserve(ComputeNumSCPairs(Ly, Lx, ref_bonds));

    // Helper lambda to add target bonds for a given reference bond
    auto add_targets = [&](size_t y1, size_t x1) {
      for (size_t y2 = y1 + 1; y2 < Ly; ++y2) {
        for (size_t x2 = 0; x2 < Lx - 1; ++x2) {
          mapping.push_back({y1, x1, 0, y2, x2, 0});  // 0 = horizontal
        }
      }
    };

    if (ref_bonds.empty()) {
      // Enumerate all horizontal reference bonds (y1, x1)-(y1, x1+1)
      for (size_t y1 = 0; y1 < Ly - 1; ++y1) {
        for (size_t x1 = 0; x1 < Lx - 1; ++x1) {
          add_targets(y1, x1);
        }
      }
    } else {
      // Only selected reference bonds (sorted by y then x for consistent ordering)
      std::vector<std::pair<size_t, size_t>> sorted_bonds;
      for (const auto& bond : ref_bonds) {
        if (bond.first < Ly - 1 && bond.second < Lx - 1) {
          sorted_bonds.push_back(bond);
        }
      }
      std::sort(sorted_bonds.begin(), sorted_bonds.end());
      for (const auto& bond : sorted_bonds) {
        add_targets(bond.first, bond.second);
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
   * @param ref_bonds Optional list of selected reference bonds (y, x). Empty means all.
   * @return Coordinate mapping as string content.
   */
  static std::string GenerateSCPairCorrCoordString(
      size_t Ly, size_t Lx,
      const std::vector<std::pair<size_t, size_t>>& ref_bonds = {}) {
    auto mapping = GenerateSCPairCorrIndexMapping(Ly, Lx, ref_bonds);
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
   * 2. For each target row, use BTen-based two-site trace (O(Lx) per row vs O(Lx²) naive)
   *
   * ## BTen Optimization for Two-Site Replacement
   *
   * For target bond (x2, x2+1):
   * - LEFT BTen covers [0, x2)
   * - Two-site contraction at x2 and x2+1
   * - RIGHT BTen covers (x2+1, Lx-1]
   *
   * Scan from right to left:
   * 1. Initialize LEFT BTen to cover all columns
   * 2. Initialize RIGHT BTen at rightmost position (vacuum)
   * 3. For x2 from Lx-2 down to 0:
   *    - Compute trace using left_bten[x2] + two-site + right_bten[N-2-x2]
   *    - Grow RIGHT BTen by one step (absorb column x2+1 for next iteration)
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
    
    // FIXME: Fermion sign is NOT correctly handled.
    // The current implementation traces out parity indices during contraction,
    // losing fermion sign information. Need to preserve target bond's physical
    // indices through BMPS contraction and extract sign at the end.
    // The results should not only miss some overall sign for each correlation,
    // So we temporarily keep the code since the framework to calculate the environment window may be useful
    // in the future.
    // The possible ways to fix it:
    // 1. Include the fusion tree information in the BMPS Contractor
    // 2. Keep the parity indices of the tensors in the reference bond
    // 3. Keep the whole parity indices of 2D tensor network in contracting the Amplitude.
    throw std::runtime_error("MeasureSingletPairCorrelation: Fermion sign is NOT correctly handled. ");
    if (!enable_singlet_pair_correlation_) return;

    using Tensor = qlten::QLTensor<TenElemT, QNT>;
    using TransferMPO = std::vector<Tensor *>;

    const size_t Ly = tn.rows();
    const size_t Lx = tn.cols();

    if (Ly < 2 || Lx < 2) return;  // Need at least 2x2 lattice

    std::vector<TenElemT> sc_corr;
    
    // Get the DOWN BMPS stack for bottom environments
    const auto& down_stack = contractor.GetBMPS(DOWN);
    
    // Get the UP BMPS stack - use contractor's cached vacuum BMPS
    const auto& up_stack = contractor.GetBMPS(UP);
    if (up_stack.empty()) {
      throw std::runtime_error(
          "MeasureSingletPairCorrelation: UP BMPS stack is empty. "
          "Contractor not properly initialized.");
    }

    // Build reference bonds set for fast lookup
    // If selected_ref_bonds_ is empty, process all bonds
    std::set<std::pair<size_t, size_t>> ref_bonds_set;
    size_t max_ref_y = Ly - 2;  // Default: process all rows
    if (!selected_ref_bonds_.empty()) {
      max_ref_y = 0;  // Reset when filtering to avoid unnecessary iterations
      for (const auto& bond : selected_ref_bonds_) {
        // Validate and add to set
        if (bond.first < Ly - 1 && bond.second < Lx - 1) {
          ref_bonds_set.insert(bond);
          max_ref_y = std::max(max_ref_y, bond.first);
        }
      }
      if (ref_bonds_set.empty()) return;  // No valid bonds selected
    }
    const bool filter_enabled = !ref_bonds_set.empty();

    // Verify UP stack depth - must cover all reference rows
    // up_stack[y1] represents boundary state after absorbing rows [0, y1-1]
    if (up_stack.size() <= max_ref_y) {
      throw std::runtime_error(
          "MeasureSingletPairCorrelation: UP BMPS stack insufficient. "
          "Need depth " + std::to_string(max_ref_y + 1) + 
          ", got " + std::to_string(up_stack.size()) + 
          ". Call contractor.GrowFullBMPS(tn, UP, trunc_para) before measurement.");
    }

    // Enumerate horizontal reference bonds
    for (size_t y1 = 0; y1 <= max_ref_y; ++y1) {
      for (size_t x1 = 0; x1 < Lx - 1; ++x1) {  // x1 < Lx-1 for horizontal bond
        // Skip if filtering is enabled and this bond is not selected
        if (filter_enabled && ref_bonds_set.find({y1, x1}) == ref_bonds_set.end()) {
          continue;
        }
        
        const SiteIdx site1_ref{y1, x1};
        const SiteIdx site2_ref{y1, x1 + 1};
        
        // Check if Δ† can act: both sites must be empty
        const bool ref_valid = (config(site1_ref) == static_cast<size_t>(tJSingleSiteState::Empty) &&
                                config(site2_ref) == static_cast<size_t>(tJSingleSiteState::Empty));
        
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
        
        // Fork walkers directly from contractor's pre-computed BMPS at row y1
        // up_stack[y1] = boundary state after absorbing rows [0, y1-1]
        auto excited_walker_up_down = typename BMPSContractor<TenElemT, QNT>::BMPSWalker(
            tn, up_stack[y1], UP, y1);
        auto excited_walker_down_up = typename BMPSContractor<TenElemT, QNT>::BMPSWalker(
            tn, up_stack[y1], UP, y1);
        auto original_walker = typename BMPSContractor<TenElemT, QNT>::BMPSWalker(
            tn, up_stack[y1], UP, y1);
        
        // Evolve walkers with their respective MPOs at ref row
        TransferMPO original_mpo_y1 = tn.get_row(y1);  // Original config (ref=empty)
        excited_walker_up_down.Evolve(excited_mpo_up_down, trunc_para);
        excited_walker_down_up.Evolve(excited_mpo_down_up, trunc_para);
        original_walker.Evolve(original_mpo_y1, trunc_para);
        
        // Track how many rows walkers have absorbed
        // After Evolve(mpo_y1), they have absorbed rows [0, y1]
        size_t excited_absorbed_rows = y1 + 1;
        size_t original_absorbed_rows = y1 + 1;
        
        // Propagate to target rows y2 > y1
        for (size_t y2 = y1 + 1; y2 < Ly; ++y2) {
          // DOWN stack index for row y2
          size_t down_stack_idx = Ly - 1 - y2;
          
          if (down_stack_idx >= down_stack.size()) {
            throw std::runtime_error(
                "MeasureSingletPairCorrelation: DOWN BMPS stack insufficient. "
                "Need depth " + std::to_string(down_stack_idx + 1) + 
                ", got " + std::to_string(down_stack.size()) + 
                ". Call contractor.GrowFullBMPS(tn, DOWN, trunc_para) before measurement.");
          }
          const auto& bottom_env = down_stack[down_stack_idx];
          
          // Evolve all walkers to have absorbed rows [0, y2-1]
          // so they can use row y2's MPO for BTen
          // InitBTenLeft requires: walker absorbed [0, y2-1], MPO is row y2, bottom_env is [y2+1, Ly-1]
          while (excited_absorbed_rows < y2) {
            TransferMPO evolve_mpo = tn.get_row(excited_absorbed_rows);
            excited_walker_up_down.Evolve(evolve_mpo, trunc_para);
            excited_walker_down_up.Evolve(evolve_mpo, trunc_para);
            excited_absorbed_rows++;
          }
          while (original_absorbed_rows < y2) {
            TransferMPO evolve_mpo = tn.get_row(original_absorbed_rows);
            original_walker.Evolve(evolve_mpo, trunc_para);
            original_absorbed_rows++;
          }
          
          // Build standard MPO for this row (y2)
          TransferMPO standard_mpo = tn.get_row(y2);
          
          // BTen optimization: scan left to right with ShiftBTenWindow.
          //
          // For two-site replacement at (x2, x2+1):
          // - LEFT BTen covers [0, x2)
          // - RIGHT BTen covers (x2+1, Lx-1]
          // Total: O(Lx) per row instead of O(Lx²)
          //
          // Initialize: LEFT = vacuum, RIGHT covers [2, Lx-1]
          excited_walker_up_down.InitBTenLeft(standard_mpo, bottom_env, 0);
          excited_walker_down_up.InitBTenLeft(standard_mpo, bottom_env, 0);
          excited_walker_up_down.InitBTenRight(standard_mpo, bottom_env, 1);
          excited_walker_down_up.InitBTenRight(standard_mpo, bottom_env, 1);
          
          original_walker.InitBTenLeft(standard_mpo, bottom_env, 0);
          original_walker.InitBTenRight(standard_mpo, bottom_env, 1);
          
          // Prepare empty tensors for target bond (reused across x2 in this row).
          // Note: Cannot hoist outside y2 loop because TPS tensors at different rows
          // may have different auxiliary space dimensions.
          std::vector<Tensor> empty_tensors(Lx);
          for (size_t x = 0; x < Lx; ++x) {
            empty_tensors[x] = GetSiteTensorImpl<TenElemT, QNT>(split_index_tps, y2, x,
                                                                 static_cast<size_t>(tJSingleSiteState::Empty));
          }
          
          // Scan from left to right
          for (size_t x2 = 0; x2 < Lx - 1; ++x2) {
            const SiteIdx site1_tgt{y2, x2};
            const SiteIdx site2_tgt{y2, x2 + 1};
            
            // Check if Δ can act: need (up,down) or (down,up) pair
            const bool tgt_is_up_down = (config(site1_tgt) == static_cast<size_t>(tJSingleSiteState::SpinUp) &&
                                          config(site2_tgt) == static_cast<size_t>(tJSingleSiteState::SpinDown));
            const bool tgt_is_down_up = (config(site1_tgt) == static_cast<size_t>(tJSingleSiteState::SpinDown) &&
                                          config(site2_tgt) == static_cast<size_t>(tJSingleSiteState::SpinUp));
            
            TenElemT result = TenElemT(0);
            if (ref_valid && (tgt_is_up_down || tgt_is_down_up)) {
              // Get original tensors for target bond (spin_pair in current config)
              Tensor original_tensor_x2 = GetSiteTensorImpl<TenElemT, QNT>(
                  split_index_tps, y2, x2, config(site1_tgt));
              Tensor original_tensor_x2_plus_1 = GetSiteTensorImpl<TenElemT, QNT>(
                  split_index_tps, y2, x2 + 1, config(site2_tgt));
              
              // Compute psi_original using BMPSWalker::TraceWithTwoSiteBTen
              TenElemT psi_original = original_walker.TraceWithTwoSiteBTen(
                  original_tensor_x2, original_tensor_x2_plus_1, x2, standard_mpo, bottom_env);
              
              // Compute excited overlap: target bond replaced with empty (Δ action)
              TenElemT overlap_up_down = excited_walker_up_down.TraceWithTwoSiteBTen(
                  empty_tensors[x2], empty_tensors[x2 + 1], x2, standard_mpo, bottom_env);
              TenElemT overlap_down_up = excited_walker_down_up.TraceWithTwoSiteBTen(
                  empty_tensors[x2], empty_tensors[x2 + 1], x2, standard_mpo, bottom_env);
              
              // In VMC, configurations are sampled by |ψ(σ)|². 
              // psi_original = 0 should never happen for a sampled configuration.
              if (psi_original == TenElemT(0)) {
                throw std::runtime_error(
                    "MeasureSingletPairCorrelation: psi_original = 0 for sampled configuration. "
                    "This indicates a bug in sampling or tensor network contraction.");
              }
              
              // Compute normalized local estimator
              TenElemT ratio_up_down = overlap_up_down / psi_original;
              TenElemT ratio_down_up = overlap_down_up / psi_original;
              
              if (tgt_is_up_down) {
                result = TenElemT(0.5) * ComplexConjugate(ratio_up_down - ratio_down_up);
              } else {  // tgt_is_down_up
                result = TenElemT(0.5) * ComplexConjugate(-ratio_up_down + ratio_down_up);
              }
            }
            
            sc_corr.push_back(result);
            
            // Shift BTen window to the right for next iteration
            if (x2 < Lx - 2) {
              excited_walker_up_down.ShiftBTenWindow(standard_mpo, bottom_env, RIGHT);
              excited_walker_down_up.ShiftBTenWindow(standard_mpo, bottom_env, RIGHT);
              original_walker.ShiftBTenWindow(standard_mpo, bottom_env, RIGHT);
            }
          }
          
          // TODO: Vertical target bonds (y2, x2) -> (y2+1, x2)
          // Requires BTen2 implementation in BMPSWalker for cross-row contraction.
          
          // Clear BTen cache before evolving
          excited_walker_up_down.ClearBTen();
          excited_walker_down_up.ClearBTen();
          original_walker.ClearBTen();
          
          // Note: We DON'T evolve walkers here anymore.
          // The while loops at the beginning of each y2 iteration handle evolution.
          // This ensures absorbed_rows counters stay consistent.
        }
      }
    }
    
    if (!sc_corr.empty()) {
      out["SC_singlet_pair_corr"] = std::move(sc_corr);
    }
  }

 protected:
  bool enable_singlet_pair_correlation_ = false;
  std::vector<std::pair<size_t, size_t>> selected_ref_bonds_;  ///< Selected reference bonds (y, x)

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

