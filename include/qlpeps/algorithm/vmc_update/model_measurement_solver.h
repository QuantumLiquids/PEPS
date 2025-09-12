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
#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // WaveFunctionAmplitudeConsistencyCheck
#include <unordered_map>
#include <string>
#include <vector>

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
  std::vector<size_t> shape;          ///< Data shape. Empty for scalar; {Ly,Lx} for per-site fields
  std::vector<std::string> index_labels; ///< Optional labels, e.g., {"y","x"}
};

/**
 * @brief Base class for registry-based Monte Carlo measurements on TPS (CRTP).
 *
 * This class defines the minimal "observable registry" API required by the new
 * measurement pipeline described in the RFC “Observable Registry and Results Organization”.
 * Concrete model solvers should override:
 *   - DescribeObservables(): return a list of ObservableMeta, each carrying a stable key
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

  std::vector<ObservableMeta> DescribeObservables() const { return {}; }
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

  template<typename TenElemT>
  PsiSummary<TenElemT> ComputePsiSummary(const std::vector<TenElemT> &psi_list) const {
    PsiSummary<TenElemT> out{};
    if (psi_list.empty()) {
      out.psi_mean = TenElemT(0);
      out.psi_rel_err = 0.0;
      return out;
    }
    TenElemT mean(0);
    for (const auto &v : psi_list) mean += v;
    mean = mean / static_cast<double>(psi_list.size());
    auto abs_val = [](const TenElemT &v) -> double { return static_cast<double>(std::abs(v)); };
    const double denom = std::max(abs_val(mean), 1e-300);
    double max_dev = 0.0;
    for (const auto &v : psi_list) {
      max_dev = std::max(max_dev, abs_val(v - mean));
    }
    out.psi_mean = mean;
    out.psi_rel_err = max_dev / denom;
    return out;
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
