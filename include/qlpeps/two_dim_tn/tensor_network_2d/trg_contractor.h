// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2025-12-12
 *
 * Description: QuantumLiquids/PEPS project.
 * Finite-size TRG (Navy–Levin) contractor for PBC TensorNetwork2D.
 */

#ifndef QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_H
#define QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "qlten/qlten.h"
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"  // BMPSTruncateParams (reused for TRG SVD truncation)
#include "qlpeps/basic.h"
#include "qlpeps/two_dim_tn/common/boundary_condition.h"
#include "qlpeps/two_dim_tn/framework/site_idx.h"

namespace qlpeps {

// Forward declaration
template <typename TenElemT, typename QNT>
class TensorNetwork2D;

/**
 * @brief Finite-size TRG contractor (Navy–Levin, simplest checkerboard plaquette coarse-graining).
 *
 * This class is designed for **Periodic Boundary Condition (PBC)** contraction of a 2D tensor
 * network stored in `TensorNetwork2D`.
 *
 * @section trg_rg_convention RG convention (must be consistent everywhere)
 *
 * We use the **checkerboard plaquette** coarse-graining that maps:
 * \f[
 * N_{s+1} = \frac{1}{2} N_s, \quad L_{s+1} = \frac{1}{\sqrt{2}} L_s
 * \f]
 * and causes a **45° rotation** of the effective embedding on odd steps. This is the convention
 * matching the user's requirement "64 -> 32 -> 16" for an 8x8 lattice.
 *
 * @subsection trg_legs Leg order
 * The leg order follows `TensorNetwork2D`'s bosonic convention:
 *
 * \code
 *        3 (up)
 *        |
 * 0(left)-t-2(right)
 *        |
 *        1 (down)
 * \endcode
 *
 * For PBC scale-0 graph:
 * - leg 0 connects to left neighbor's leg 2
 * - leg 2 connects to right neighbor's leg 0
 * - leg 1 connects to down neighbor's leg 3
 * - leg 3 connects to up neighbor's leg 1
 *
 * @subsection trg_ab AB sublattice and SVD split direction
 * We assign sublattice labels on scale 0 by parity:
 * \f[
 * A: (x+y)\bmod 2 = 0,\quad B: (x+y)\bmod 2 = 1
 * \f]
 *
 * The Navy–Levin TRG step requires **different SVD split directions** on A/B to keep the
 * coarse-graining pattern consistent. This contractor stores that per-node convention and
 * uses it during each coarse-graining step.
 *
 * @note Implementation status (as of 2025-12):
 * - **Finite-size bosonic `Trace()` is implemented** using the checkerboard plaquette TRG pipeline
 *   (even->odd->even until the final 2x2 even lattice), with SVD truncation.
 * - Final contraction uses an exact 2x2 PBC contraction (no additional truncation step).
 * - Truncation parameters must be explicitly provided via `SetTruncateParams()` (no hidden defaults).
 * - Fermionic TRG is not implemented.
 * - Incremental updates (`InvalidateEnvs` influence-cone propagation), `Replace*Trace`, and `PunchHole`
 *   are not implemented yet (currently only full `Trace()` is supported).
 *
 * @tparam TenElemT Tensor element type (e.g. double, complex<double>)
 * @tparam QNT Quantum number type
 */
template <typename TenElemT, typename QNT>
class TRGContractor {
 public:
  using Tensor = qlten::QLTensor<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  /**
   * @brief Truncation control for the SVD splits inside TRG.
   *
   * This reuses `BMPSTruncateParams<RealT>` as a plain parameter bundle.
   *
   * Only the following fields are used by TRG:
   * - `trunc_err`: target truncation error passed to `qlten::SVD`
   * - `D_min`: minimum bond dimension kept after truncation
   * - `D_max`: maximum bond dimension kept after truncation
   *
   * @note Any other fields in `BMPSTruncateParams` are ignored by TRG.
   */
  using TruncateParams = qlpeps::BMPSTruncateParams<RealT>;

  /**
   * @brief Construct an empty contractor with no geometry.
   *
   * Call `Init(tn)` before `Trace(tn)` / trial APIs.
   */
  TRGContractor() = default;

  /**
   * @brief Construct with geometry only.
   *
   * @param rows Number of rows of the scale-0 network.
   * @param cols Number of columns of the scale-0 network.
   *
   * @note This only stores the geometry; you still need to call `Init(tn)`.
   */
  TRGContractor(size_t rows, size_t cols) { ResetGeometry_(rows, cols); }

  /**
   * @brief Construct with geometry and truncation parameters.
   *
   * @param rows Number of rows of the scale-0 network.
   * @param cols Number of columns of the scale-0 network.
   * @param trunc_params Truncation parameters used in all SVD splits.
   *
   * @note This only stores the geometry/params; you still need to call `Init(tn)`.
   */
  TRGContractor(size_t rows, size_t cols, const TruncateParams& trunc_params)
      : trunc_params_(trunc_params) {
    ResetGeometry_(rows, cols);
  }

  /**
   * @brief Set the SVD truncation parameters used by TRG.
   *
   * This does not rebuild topology nor clear caches.
   */
  void SetTruncateParams(const TruncateParams& trunc_params) { trunc_params_ = trunc_params; }

  /**
   * @brief Get the current truncation parameters.
   *
   * @throws std::logic_error if truncation params were never set.
   */
  const TruncateParams& GetTruncateParams() const {
    if (!trunc_params_.has_value()) {
      throw std::logic_error("TRGContractor::GetTruncateParams: truncation params are not set.");
    }
    return *trunc_params_;
  }

  /**
   * @brief Initialize internal scale-0 graph and clear caches.
   *
   * This builds the multi-scale topology (connectivity + fine/coarse mapping) and clears all
   * cached tensors.
   *
   * @param tn Tensor network container. Requirements:
   * - `tn` must have `BoundaryCondition::Periodic`
   * - the lattice must be square (`rows == cols`)
   * - the linear size must be \(2^m\) (power-of-two), so the RG flow reaches 1x1 exactly
   *
   * @note `Init()` does **not** copy any tensors from `tn`. The actual tensors are loaded lazily
   * on the first `Trace(tn)` call.
   *
   * @throws std::invalid_argument if geometry/boundary condition requirements are violated.
   */
  void Init(const TensorNetwork2D<TenElemT, QNT>& tn);

  /**
   * @brief Contract the whole 2D tensor network and return the amplitude Z.
   *
   * For TRG this is a fixed pipeline: coarse-grain repeatedly (even->odd->even->...) until the
   * final 2x2 even lattice, then contract the 2x2 PBC torus exactly (no further truncation).
   *
   * Caching behavior:
   * - First call after `Init(tn)` treats all scale-0 tensors as dirty and caches all scales.
   * - After calling `InvalidateEnvs(site)` (possibly multiple times), the next `Trace(tn)` will
   *   reload those scale-0 tensors from `tn` and recompute only the affected coarse tensors.
   * - If nothing is dirty, this returns the cached final 1x1 contraction.
   *
   * @param tn Tensor network to contract. Must match the geometry passed to `Init(tn)`.
   * @return The scalar contraction result (partition function / amplitude).
   *
   * @warning Not `const`: TRG stores and updates multi-scale caches.
   *
   * @throws std::logic_error if `Init(tn)` has not been called or truncation params are not set.
   * @throws std::invalid_argument if `tn` is not periodic.
   */
  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn);

  /**
   * @brief Compatibility adapter for existing call sites that pass a bond location/orientation.
   *
   * TRG's contraction result does not depend on these parameters; they are ignored.
   */
  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn,
                 const SiteIdx& /*site_a*/,
                 const BondOrientation /*bond_dir*/) {
    return Trace(tn);
  }

  /**
   * @brief Trial object produced by a "shadow" update.
   *
   * This captures the *minimal* set of coarse tensors that would change under the replacements
   * along the TRG flow. The trial can later be committed (swap-in) or discarded.
   *
   * Design note:
   * - This is intentionally a value type so VMC code can keep it short-term between
   *   TrialAmplitude and Accept/Reject.
   * - Only affected nodes are stored for each scale, so this is much smaller than cloning
   *   the whole multi-scale cache.
   */
  struct Trial {
    TenElemT amplitude{};
    // layer_updates[s] stores the tensors that would change at scale s (node_id -> tensor).
    std::vector<std::map<uint32_t, Tensor>> layer_updates;
  };

  /**
   * @brief Create a trial contraction result under @p replacements without modifying caches.
   *
   * This performs a "shadow" RG propagation starting from scale-0 `replacements`, and stores
   * only the affected tensors at each scale into the returned `Trial`.
   *
   * @param replacements Scale-0 site tensor replacements, identified by `SiteIdx`.
   * @return A `Trial` containing the would-be updated coarse tensors and the corresponding
   * scalar amplitude in `Trial::amplitude`.
   *
   * @note This method requires an already initialized and clean cache. In other words, call
   * `Trace(tn)` at least once and make sure there are no pending dirtiness seeds.
   *
   * @warning This does not mutate the `TensorNetwork2D`. It is the caller's responsibility to
   * keep the external `tn` consistent with `CommitTrial()` if they later commit.
   *
   * @throws std::logic_error if `Init(tn)` has not been called, truncation params are not set,
   * cache has not been initialized by `Trace(tn)`, or the cache is currently dirty.
   */
  Trial BeginTrialWithReplacement(const std::vector<std::pair<SiteIdx, Tensor>>& replacements) const;

  /**
   * @brief Commit a previously created trial into the internal cache.
   *
   * Preconditions:
   * - Cache must be clean (no pending dirtiness from `InvalidateEnvs()`).
   * - `trial` must have been created from the current clean cache (`Trial` topology matches).
   *
   * @param trial Trial object returned by `BeginTrialWithReplacement()`.
   *
   * @warning This only swaps tensors into this contractor's internal cache. It does not update
   * the external `TensorNetwork2D`. Passing an inconsistent `tn` to subsequent `Trace(tn)` calls
   * is a user error (it will typically show up once you invalidate/reload scale-0 tensors).
   *
   * @throws std::logic_error if `Init(tn)` has not been called, truncation params are not set,
   * cache has not been initialized by `Trace(tn)`, or the cache is currently dirty.
   * @throws std::invalid_argument if `trial` was created under a different topology.
   */
  void CommitTrial(Trial&& trial);

  /**
   * @brief Compute the "hole" environment tensor at @p site.
   *
   * The returned tensor has rank 4 and represents the contraction of the whole network
   * with the site tensor at @p site removed, leaving the four bond legs open.
   *
   * Current implementation status:
   * - Only supports the 2x2 periodic torus by exact contraction (a terminator for future
   *   recursive/iterative PunchHole on larger systems).
   * - For larger sizes this is not implemented yet and will throw.
   *
   * Leg convention of the returned hole tensor matches the removed site tensor:
   * \code
   *        3 (up)
   *        |
   * 0(left)-H-2(right)
   *        |
   *        1 (down)
   * \endcode
   *
   * @param tn Tensor network container (must be periodic).
   * @param site Scale-0 site to remove.
   * @return Rank-4 hole environment tensor.
   *
   * @throws std::logic_error if `Init(tn)` has not been called.
   * @throws std::invalid_argument if `tn` is not periodic.
   * @throws std::logic_error if called for sizes other than 2x2 (for now).
   */
  Tensor PunchHole(const TensorNetwork2D<TenElemT, QNT>& tn, const SiteIdx& site) const;

  /**
   * @brief Mark caches affected by a local tensor update at @p site.
   *
   * This only records a "dirty seed" on scale 0. The influence propagation across scales is
   * handled lazily on the next `Trace(tn)` call.
   *
   * @param site Scale-0 lattice site whose tensor has changed in the external `tn`.
   *
   * @throws std::logic_error if `Init(tn)` has not been called.
   */
  void InvalidateEnvs(const SiteIdx& site);

  /**
   * @brief Clear all cached scales and dirty state.
   *
   * This drops all cached tensors and dirtiness seeds, and resets the "cache initialized" flag.
   * It does not change geometry/boundary-condition/truncation parameters.
   */
  void ClearCache();

 private:
  enum class SubLattice : uint8_t { A = 0, B = 1 };
  enum class SplitDir : uint8_t { Horizontal = 0, Vertical = 1 };

  struct Neighbor {
    uint32_t node = 0;
    uint8_t leg = 0;  // 0..3
  };

  struct Graph {
    // For each node: 4 neighbors, each identified by (node, leg).
    std::vector<std::array<Neighbor, 4>> nbr;
    std::vector<SubLattice> sublattice;
    std::vector<SplitDir> split_dir;

    size_t Size() const { return nbr.size(); }
  };

  struct ScaleCache {
    Graph graph;
    std::vector<Tensor> tens;

    // Mapping between scales (filled when scale s -> s+1 is built).
    // fine_to_coarse[fine] has up to 2 coarse parents in checkerboard TRG.
    std::vector<std::array<uint32_t, 2>> fine_to_coarse;
    std::vector<std::array<uint32_t, 4>> coarse_to_fine;
  };

  struct SplitARes {
    Tensor P;  // "NW" piece: (leg0, leg3, alpha)
    Tensor Q;  // "SE" piece: (alpha, leg1, leg2)
  };
  struct SplitBRes {
    Tensor Q;  // "SW/N" piece: (leg0, leg1, alpha)
    Tensor P;  // "NE/S" piece: (alpha, leg2, leg3)
  };

  static bool IsPowerOfTwo_(size_t n) { return n != 0 && ((n & (n - 1)) == 0); }

  void ResetGeometry_(size_t rows, size_t cols);
  uint32_t NodeId_(size_t row, size_t col) const {
    return static_cast<uint32_t>(row * cols_ + col);
  }
  std::pair<size_t, size_t> Coord_(uint32_t node) const {
    return {static_cast<size_t>(node) / cols_, static_cast<size_t>(node) % cols_};
  }

  void BuildScale0GraphPBCSquare_();
  void BuildTopology_();
  
  // Helpers for incremental updates
  void MarkDirtySeed_(uint32_t node);
  
  // SVD Splitters
  SplitARes SplitType0_(const Tensor& T_in) const;
  SplitBRes SplitType1_(const Tensor& T_in) const;

  // Contractions
  Tensor ContractPlaquette_(const std::vector<Tensor>& fine_tens, uint32_t coarse_idx, size_t n_fine);
  Tensor ContractDiamond_(const std::vector<Tensor>& fine_tens, uint32_t coarse_idx, size_t n_fine_embed);
  TenElemT ContractFinal1x1_(const Tensor& T) const;
  TenElemT ContractFinal2x2_(const std::array<Tensor, 4>& T2x2) const;
  Tensor PunchHoleFinal2x2_(const std::array<Tensor, 4>& T2x2, uint32_t removed_id) const;

  size_t rows_ = 0;
  size_t cols_ = 0;
  BoundaryCondition bc_ = BoundaryCondition::Open;
  bool tensors_initialized_ = false;

  // Multi-scale cache: scales_[0] corresponds to the original network (scale 0).
  std::vector<ScaleCache> scales_;

  // Dirty seeds on scale 0; influence propagation across scales is handled lazily.
  std::unordered_set<uint32_t> dirty_scale0_;

  // Must be set by caller (via SetTruncateParams or the ctor taking trunc_params).
  std::optional<TruncateParams> trunc_params_;
};

}  // namespace qlpeps

#include "trg_contractor_impl.h"

#endif  // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_H


