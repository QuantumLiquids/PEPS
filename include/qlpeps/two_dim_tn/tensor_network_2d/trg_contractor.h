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
 *   (even->odd->even until 1x1), with SVD truncation.
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
  using TruncateParams = qlpeps::BMPSTruncateParams<RealT>;

  TRGContractor() = default;
  TRGContractor(size_t rows, size_t cols) { ResetGeometry_(rows, cols); }
  TRGContractor(size_t rows, size_t cols, const TruncateParams& trunc_params)
      : trunc_params_(trunc_params) {
    ResetGeometry_(rows, cols);
  }

  void SetTruncateParams(const TruncateParams& trunc_params) { trunc_params_ = trunc_params; }
  const TruncateParams& GetTruncateParams() const {
    if (!trunc_params_.has_value()) {
      throw std::logic_error("TRGContractor::GetTruncateParams: truncation params are not set.");
    }
    return *trunc_params_;
  }

  /**
   * @brief Initialize internal scale-0 graph and clear caches.
   *
   * @param tn Tensor network data container (must be PBC square with size 2^m)
   */
  void Init(const TensorNetwork2D<TenElemT, QNT>& tn);

  /**
   * @brief Contract the whole 2D tensor network and return the amplitude Z.
   *
   * For TRG this is a "fixed pipeline" operation: it depends only on `tn`.
   */
  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn) const;

  /**
   * @brief Compatibility adapter for existing call sites that pass a bond location/orientation.
   *
   * TRG's contraction result does not depend on these parameters; they are ignored.
   */
  TenElemT Trace(const TensorNetwork2D<TenElemT, QNT>& tn,
                 const SiteIdx& /*site_a*/,
                 const BondOrientation /*bond_dir*/) const {
    return Trace(tn);
  }

  /**
   * @brief Mark caches affected by a local tensor update at @p site.
   *
   * For TRG, the "influence cone" expands across scales. This method only records the dirty
   * seed(s); the propagation is performed lazily on the next Trace/Replace call.
   */
  void InvalidateEnvs(const SiteIdx& site);

  /**
   * @brief Clear all cached scales and dirty state.
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

  static bool IsPowerOfTwo_(size_t n) { return n != 0 && ((n & (n - 1)) == 0); }

  void ResetGeometry_(size_t rows, size_t cols);
  uint32_t NodeId_(size_t row, size_t col) const {
    return static_cast<uint32_t>(row * cols_ + col);
  }
  std::pair<size_t, size_t> Coord_(uint32_t node) const {
    return {static_cast<size_t>(node) / cols_, static_cast<size_t>(node) % cols_};
  }

  void BuildScale0GraphPBCSquare_();
  void MarkDirtySeed_(uint32_t node);

  size_t rows_ = 0;
  size_t cols_ = 0;
  BoundaryCondition bc_ = BoundaryCondition::Open;

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


