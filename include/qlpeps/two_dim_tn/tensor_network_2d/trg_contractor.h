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
   * - If the cache is uninitialized, this loads all tensors from `tn` and performs a full TRG contraction.
   * - If the cache is initialized, this returns the **cached** result. It ignores `tn` unless
   *   `ClearCache()` was called.
   * - To update the state after modifying tensors, use `CommitTrial()` or `ClearCache()`.
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

#if defined(QLPEPS_UNITTEST)
  /**
   * @brief Expose split pieces for module-level unit tests only.
   *
   * @note This is intentionally guarded by QLPEPS_UNITTEST so it is not part of the
   * production API surface.
   */
  std::pair<Tensor, Tensor> SplitType0PiecesForTest(const Tensor& T) const {
    auto r = SplitType0_(T);
    return {std::move(r.P), std::move(r.Q)};
  }
  std::pair<Tensor, Tensor> SplitType1PiecesForTest(const Tensor& T) const {
    auto r = SplitType1_(T);
    return {std::move(r.P), std::move(r.Q)};
  }
#endif

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
   * @note Requires initialized cache.
   *
   * @warning This does not mutate the `TensorNetwork2D`. It is the caller's responsibility to
   * keep the external `tn` consistent with `CommitTrial()` if they later commit.
   *
   * @throws std::logic_error if `Init(tn)` has not been called, truncation params are not set,
   * or cache has not been initialized by `Trace(tn)`.
   */
  Trial BeginTrialWithReplacement(const std::vector<std::pair<SiteIdx, Tensor>>& replacements) const;

  /**
   * @brief Commit a previously created trial into the internal cache.
   *
   * Preconditions:
   * - `trial` must have been created from the current clean cache (`Trial` topology matches).
   *
   * @param trial Trial object returned by `BeginTrialWithReplacement()`.
   *
   * @warning This only swaps tensors into this contractor's internal cache. It does not update
   * the external `TensorNetwork2D`. Passing an inconsistent `tn` to subsequent `Trace(tn)` calls
   * is a user error (it will typically show up once you invalidate/reload scale-0 tensors).
   *
   * @throws std::logic_error if `Init(tn)` has not been called, truncation params are not set,
   * or cache has not been initialized by `Trace(tn)`.
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
   * @brief Clear all cached scales and dirty state.
   *
   * This drops all cached tensors, and resets the "cache initialized" flag.
   * It does not change geometry/boundary-condition/truncation parameters.
   */
  void ClearCache();

#if defined(QLPEPS_UNITTEST)
  /**
   * @brief Test-only baseline: build a 4x4 hole by brute probing (very expensive).
   *
   * This method is meant for **unit tests only** as a correctness reference for future
   * optimized impurity-TRG implementations. It is not intended for production use.
   *
   * Behavior:
   * - Supports 4x4 PBC only.
   * - Requires cache to be initialized and clean (call `Trace(tn)` once, and no pending invalidations).
   * - Returns the hole tensor in the **original leg space** of the removed site:
   *   legs are ordered `[L,D,R,U]` and directions are inverted (dual space) so that
   *   `Contract(hole, tn(site))` yields a scalar.
   *
   * @warning Complexity is O(product of bond dims at the site) trial contractions.
   */
  Tensor PunchHoleBaselineByProbingForTest(const TensorNetwork2D<TenElemT, QNT>& tn,
                                          const SiteIdx& site) const;
#endif

 private:
  /**
   * @brief Internal implementation for 4x4 "punch-hole" using cached finite-size TRG graph.
   *
   * @details
   * This routine mirrors the fixed contraction graph used by @ref Trace by reverse-mode contraction
   * (adjoint pullback) through the cached split pieces and local plaquette/diamond contractions.
   *
   * @par A note on the historical "0.25" normalization issue
   * Earlier revisions contained a hard-coded normalization:
   * \f[
   *   \mathrm{hole} \leftarrow 0.25 \times \mathrm{hole},
   * \f]
   * to satisfy unit tests requiring \f$\langle \mathrm{hole}_i, T_i\rangle \approx Z\f$.
   *
   * The *possible cause* is **systematic over-counting induced by the 4x4 PBC topology encoding**:
   *
   * - In @ref BuildTopology_ we build `fine_to_coarse` such that a fine node may have **two coarse
   *   parents** on each RG step (checkerboard plaquette step and the embedded diamond step).
   * - In @ref PunchHole4x4_ we then accumulate pullbacks across parent contexts.
   *
   * On a 4x4 torus there are exactly two RG steps (scale0->scale1 and scale1->scale2), hence a naive
   * "sum over both parents" leads to an overall multiplicity of \f$2\times2 = 4\f$ in the composed
   * pullback, which shows up as \f$\langle \mathrm{hole}, T_{\text{site}}\rangle \approx 4 Z\f$.
   *
   * @par Truncation and "frozen" SVD pieces
   * This code uses a *linearized split* for backpropagation: it treats the SVD factors
   * \f$(U,S,V)\f$ from the forward pass as **frozen** when mapping a site tensor perturbation
   * \f$\delta T\f$ to split-piece perturbations \f$(\delta P,\delta Q)\f$.
   *
   * - With nonzero truncation (finite `D_max`), the true TRG contraction is not strictly linear in a
   *   single site tensor, so \f$\langle \mathrm{hole},T\rangle\f$ is only required to match \f$Z\f$
   *   approximately.
   * - The observed factor-4 mismatch, however, is a **topology/multiplicity** effect and can persist
   *   even when truncation is disabled (it is not "caused by truncation").
   *
   * @par What to do during future refactors
   * Prefer eliminating the double-cover at the computation-graph level (single-cover RG mapping, or
   * an impurity-list style algorithm as in thesis Fig. 3.29 / Fortran reference) rather than relying
   * on an opaque global rescale.
   *
   * Suggested regression checks when refactoring:
   * - Use a 3x3 terminator (or other terminator variants) to ensure the hole propagation remains
   *   graph-consistent.
   * - Test larger sizes (e.g. 8x8) to ensure no size-dependent "magic constants" remain.
   * - A degenerate diagnostic case: only one row has nontrivial bond dimension (others are trivial),
   *   so the network reduces to a matrix trace; this helps isolate multiplicity/double-cover issues.
   */
  // Generic backprop hole from a small terminator (2x2 or 3x3) down to scale-0.
  // Handles N=2^k (terminating at 2x2) and N=3*2^k (terminating at 3x3).
  // Includes automatic averaging of parent contributions to correct for checkerboard multiplicity.
  Tensor PunchHoleBackpropGeneric_(const SiteIdx& site) const;

  // ---- PunchHole helpers (shared by future terminators / larger sizes) ----
  int ParentSlot_(size_t scale, uint32_t fine_id, uint32_t parent_id) const;

  // Adjoint pullback from split-piece holes back to the original rank-4 tensor hole,
  // using the *linearized* (frozen) forward SVD factors cached per (node, parent-slot).
  Tensor LinearSplitAdjointToHole_(size_t scale,
                                  uint32_t node,
                                  uint32_t parent,
                                  const Tensor* H_P,
                                  const Tensor* H_Q) const;

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

    // --- Split cache (P/Q pieces) ---
    //
    // For impurity TRG / PunchHole we must reuse the same split pieces as used in the forward TRG
    // contraction. Storing them explicitly keeps the later hole code simple and avoids repeated SVD.
    //
    // Each *edge context* (fine node -> one of its coarse parents) is split by exactly one of
    // {SplitType0_, SplitType1_} during the forward TRG. This matters because for checkerboard PBC
    // a fine node typically participates in TWO parents, and (critically) it may play different
    // roles in each parent (e.g. odd-layer node can be E in one diamond but W in another).
    //
    // Therefore, for impurity TRG / PunchHole we must cache split data per (node, parent-slot),
    // not per node.
    //
    // Split types:
    // - type0: P(l, u, a_out), Q(a_in, d, r)
    // - type1: Q(l, d, a_out), P(a_in, r, u)
    enum class SplitType : uint8_t { Type0 = 0, Type1 = 1 };
    std::vector<std::array<SplitType, 2>> split_type;   // per parent-slot (0/1)
    std::vector<std::array<Tensor, 2>> split_P;         // per parent-slot
    std::vector<std::array<Tensor, 2>> split_Q;         // per parent-slot

    // --- Split isometries (for impurity TRG / PunchHole linearization) ---
    //
    // Store the *fixed* SVD isometries and diag reweighting from the forward pass, so that
    // for any updated tensor X at this node we can compute its split pieces (P_X, Q_X) in the
    // same truncated alpha basis *linearly*:
    //
    // Type0 (after perm {0,3,1,2}): A = U S V^T
    //   P_X = (A_X * V) * S^{-1/2}
    //   Q_X = S^{-1/2} * (U^T * A_X)
    //
    // Type1 (no perm): A = U S V^T
    //   Q_X = (A_X * V) * S^{-1/2}
    //   P_X = S^{-1/2} * (U^T * A_X)
    //
    // Here S^{-1/2} is taken elementwise on the retained singular values (diag tensor).
    std::vector<std::array<Tensor, 2>> split_U;           // per parent-slot
    std::vector<std::array<Tensor, 2>> split_Vt;          // per parent-slot
    std::vector<std::array<qlten::QLTensor<RealT, QNT>, 2>> split_S_inv_sqrt; // per parent-slot
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
  // void MarkDirtySeed_(uint32_t node);

  // Split cache helpers (for Trace/PunchHole)
  void EnsureSplitCacheForNodes_(size_t scale, const std::set<uint32_t>& nodes);
  
  // SVD Splitters
  SplitARes SplitType0_(const Tensor& T_in) const;
  SplitBRes SplitType1_(const Tensor& T_in) const;

  // Core contraction logic
  // Returns: tmp2.Transpose({3, 2, 1, 0})
  static Tensor ContractPlaquetteCore_(const Tensor& Q0, const Tensor& Q1, const Tensor& P1, const Tensor& P0);
  
  // Returns: out.Transpose({1, 0, 3, 2})
  static Tensor ContractDiamondCore_(const Tensor& Np, const Tensor& Ep, const Tensor& Sq, const Tensor& Wq);

  TenElemT ContractFinal1x1_(const Tensor& T) const;
  TenElemT ContractFinal2x2_(const std::array<Tensor, 4>& T2x2) const;
  TenElemT ContractFinal3x3_(const std::array<Tensor, 9>& T3x3) const;
  Tensor PunchHoleFinal2x2_(const std::array<Tensor, 4>& T2x2, uint32_t removed_id) const;
  Tensor PunchHoleFinal3x3_(const std::array<Tensor, 9>& T3x3, uint32_t removed_id) const;

  TenElemT GetCachedAmplitude_() const;

  size_t rows_ = 0;
  size_t cols_ = 0;
  BoundaryCondition bc_ = BoundaryCondition::Open;
  bool tensors_initialized_ = false;

  // Multi-scale cache: scales_[0] corresponds to the original network (scale 0).
  std::vector<ScaleCache> scales_;

  // Dirty seeds on scale 0; influence propagation across scales is handled lazily.
  // std::unordered_set<uint32_t> dirty_scale0_;

  // Must be set by caller (via SetTruncateParams or the ctor taking trunc_params).
  std::optional<TruncateParams> trunc_params_;
};

}  // namespace qlpeps

#include "trg_contractor_impl.h"

#endif  // QLPEPS_TWO_DIM_TN_TENSOR_NETWORK_2D_TRG_CONTRACTOR_H
