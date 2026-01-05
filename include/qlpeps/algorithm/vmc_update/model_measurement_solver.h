/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: QuantumLiquids/PEPS project. Model Measurement Solver base class. Also an example on how to write a ModelEnergySolver.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_MEASUREMENT_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_MEASUREMENT_SOLVER_H

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"                // SplitIndexTPS
#include "qlpeps/vmc_basic/wave_function_component.h"             // TPSWaveFunctionComponent
#include "qlpeps/algorithm/vmc_update/psi_consistency.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <complex>
#include <functional>
#include <limits>
#include <type_traits>

namespace qlpeps {

// Legacy ObservablesLocal has been migrated to ObservableMap

/** Minimal observable registry return type for new API. */
template<typename ElemT>
using ObservableMap = std::unordered_map<std::string, std::vector<ElemT>>; // key -> flat values

/**
 * @brief Observable metadata description used by registry-based API.
 */
struct ObservableMeta {
  std::string key;                    ///< Unique key, e.g., "energy", "spin_z"
  std::string description;            ///< Concise physical meaning (English)
  std::vector<size_t> shape;          ///< Data shape. Empty for scalar; lattice-aware entries should use runtime sizes
  std::vector<std::string> index_labels; ///< Optional axis labels, e.g., {"y","x"} or {"bond_y","bond_x"}

  /**
   * @brief Optional coordinate mapping generator.
   *
   * If set, DumpData() will call this function to generate coordinate mapping file
   * (stats/<key>_coords.txt). Signature: (ly, lx) -> file content string.
   *
   * Use this for observables with non-trivial index semantics (e.g., bond pair correlations,
   * structure factor). The generator should produce human-readable text describing the mapping
   * from linear index to physical coordinates.
   */
  std::function<std::string(size_t, size_t)> coord_generator;
};

/**
 * @brief Base class for registry-based Monte Carlo measurements on TPS (CRTP).
 *
 * This class defines the minimal "observable registry" API required by the new
 * measurement pipeline described in the RFC “Observable Registry and Results Organization”.
 * Concrete model solvers should override:
 *   - DescribeObservables(size_t ly, size_t lx): return a list of ObservableMeta, each carrying a stable key
 *     (e.g., "energy", "spin_z", "bond_energy_h"), human-readable description, and
 *     optional shape/index labels.
 *   - EvaluateObservables(sitps, tps_sample): return key -> flat values for a single MC sample.
 *
 * Notes:
 *   - Do not expose large intermediate lists in outputs. If a solver needs to provide raw
 *     wavefunction amplitudes for consistency analysis, place them under key "psi_list".
 *     The executor will convert them into "psi_mean" (complex scalar) and "psi_rel_err"
 *     (real scalar) and will not persist "psi_list" to disk.
 *   - Legacy one-/two-point categories are no longer part of the API surface. Use stable
 *     keys and metadata to describe shapes and index semantics instead.
 *
 * @tparam ConcreteModelSolver Derived model solver type (CRTP)
 */
template<typename ConcreteModelSolver>
class ModelMeasurementSolver {
 public:
  ModelMeasurementSolver(void) = default;

  // New API: describe and evaluate observables via registry.
  // Default returns empty; concrete solvers should override without relying on legacy paths.
  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> * /*sitps*/,
      TPSWaveFunctionComponent<TenElemT, QNT> * /*tps_sample*/
  ) { return {}; }

  std::vector<ObservableMeta> DescribeObservables(size_t /*ly*/, size_t /*lx*/) const { return {}; }
  const double wave_function_component_measure_accuracy = 1E-3;

  /**
   * @brief Sample-level psi summary API (non-registered, no statistics).
   *
   * Derived classes should provide BuildPsiList(...) if they can construct
   * a meaningful psi list for the current model/lattice. This default
   * implementation returns zeros when no psi list is available.
   */
  template<typename TenElemT>
  struct PsiSummary {
    TenElemT psi_mean;   ///< mean amplitude
    double psi_rel_err;  ///< radius_rel = max_i |psi_i - mean| / |mean|
  };

  template<typename TenElemT, typename QNT>
  PsiSummary<TenElemT> EvaluatePsiSummary(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample
  ) const {
    // Fast path: use cached summary from the last EvaluateObservables call
    if (last_psi_valid_) {
      last_psi_valid_ = false; // consume once per sample
      TenElemT mean = static_cast<TenElemT>(last_psi_mean_re_);
      if constexpr (!std::is_same_v<TenElemT, double>) {
        mean = TenElemT(last_psi_mean_re_, last_psi_mean_im_);
      }
      return PsiSummary<TenElemT>{mean, last_psi_rel_err_};
    }
    (void)sitps; (void)tps_sample;
    auto *self = static_cast<const ConcreteModelSolver *>(this);
    std::vector<TenElemT> psi_list = self->template BuildPsiList<TenElemT, QNT>(sitps, tps_sample);
    return ComputePsiSummary(psi_list);
  }

 protected:
  template<typename TenElemT, typename QNT>
  std::vector<TenElemT> BuildPsiList(
      const SplitIndexTPS<TenElemT, QNT> * /*sitps*/,
      TPSWaveFunctionComponent<TenElemT, QNT> * /*tps_sample*/
  ) const { return {}; }

  /**
   * @brief Estimate the mean amplitude and a conservative relative error.
   *
   * Tensor-network contractions yield multiple amplitudes for the same configuration by
   * finalizing the network at different positions; the spread reflects truncation noise.
   * For bosonic models these samples cluster tightly, but fermionic simulations often show
   * two clusters separated by an almost-\f$\pi\f$ phase due to sign conventions.
   *
   * This helper first selects the largest-magnitude amplitude as a reference and flips any
   * sample whose overlap \f$\Re[\psi_i \psi_\mathrm{ref}^*]\f$ is negative, effectively
   * aligning the two fermionic branches. Afterwards it reports the arithmetic mean and the
   * maximum deviation normalized by \f$|\langle\psi\rangle|\f$.
   *
   * The contraction pipeline guarantees that all amplitudes are well separated from the
   * numeric limit, hence no extra normalization is required when forming the overlap.
   *
   * @tparam TenElemT Scalar type of the amplitudes (real or complex).
   * @param psi_list All contraction results obtained for a single configuration.
   * @return Mean amplitude and relative radius of the aligned cluster.
   */
  template<typename TenElemT>
  PsiSummary<TenElemT> ComputePsiSummary(const std::vector<TenElemT> &psi_list) const {
    const auto s = ComputePsiConsistencySummaryAligned(psi_list);
    return PsiSummary<TenElemT>{s.psi_mean, s.psi_rel_err};
  }

  // Cache setter for derived classes to avoid double traversal per sample
  template<typename TenElemT>
  void SetLastPsiSummary(const TenElemT &psi_mean, double psi_rel_err) const {
    if constexpr (std::is_same_v<TenElemT, double>) {
      last_psi_mean_re_ = psi_mean;
      last_psi_mean_im_ = 0.0;
    } else {
      last_psi_mean_re_ = static_cast<double>(std::real(psi_mean));
      last_psi_mean_im_ = static_cast<double>(std::imag(psi_mean));
    }
    last_psi_rel_err_ = psi_rel_err;
    last_psi_valid_ = true;
  }

 protected:
  // Lightweight, type-agnostic cache of last psi summary; consumed by EvaluatePsiSummary()
  mutable bool last_psi_valid_ = false;
  mutable double last_psi_mean_re_ = 0.0;
  mutable double last_psi_mean_im_ = 0.0;
  mutable double last_psi_rel_err_ = 0.0;
};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_MEASUREMENT_SOLVER_H
