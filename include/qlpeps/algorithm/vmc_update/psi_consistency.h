// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
*
* Description: Shared utilities for wavefunction amplitude (psi) consistency checks.
*
 * Background (BMPS contractor truncation inconsistency):
 * - In BMPS-contractor based contractions we often compute the same configuration amplitude
 *   \f$\Psi(S)\f$ multiple times by "finalizing" the partially contracted 2D network at
 *   different cursor positions (different trace/closure locations).
 * - With exact contractions these values should be identical. In practice, BMPS truncation
 *   (finite bond dimension, SVD cut, etc.) makes each closure path introduce slightly
 *   different approximation errors, so the resulting \f$\Psi(S)\f$ values can differ.
 * - This header provides a stable, complex-aware metric `psi_rel_err` to quantify the spread
 *   caused by truncation. Large `psi_rel_err` typically indicates truncation is too aggressive.
 *
* Notes:
* - This header centralizes the "complex-aware alignment + relative radius" logic used by
*   both ModelEnergySolver and ModelMeasurementSolver.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_PSI_CONSISTENCY_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_PSI_CONSISTENCY_H

#include <algorithm>
#include <complex>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace qlpeps {

/**
 * @brief Psi(S) consistency warning controls.
 *
 * Used as a sub-field of RuntimeParams and applied by executors/evaluators.
 *
 * Notes:
 * - `master_only=true` makes the warning budget "global" (printed by master rank only).
 * - `max_warnings` limits the number of printed messages (per executor instance).
 */
struct PsiConsistencyWarningParams {
  bool enabled = true;
  bool master_only = false;
  double threshold = 1e-3;
  size_t max_warnings = 50;
  size_t max_print_elems = 8;
};

/**
 * @brief Configuration rescue controls.
 *
 * When enabled, MonteCarloEngine attempts to rescue invalid configurations (e.g., due to
 * near-zero amplitude or Empty tensor exceptions from BMPS contraction) by broadcasting
 * a valid configuration from another MPI rank.
 */
struct ConfigurationRescueParams {
  bool enabled = true;  ///< Whether to attempt configuration rescue across MPI ranks

  /**
   * @brief Minimum amplitude threshold for valid configurations.
   *
   * Configurations with |amplitude| <= this threshold are considered invalid and will
   * trigger rescue if enabled. The default value is std::numeric_limits<double>::min(),
   * which is the most permissive setting (only truly zero amplitudes are invalid).
   *
   * For stricter control, consider using larger values such as:
   * - 1e-100: reasonable for most physical applications
   * - sqrt(DBL_MIN) ≈ 1.5e-154: ensures |amplitude|^2 is representable
   *
   * Note: Monte Carlo updates use amplitude ratios, so small amplitudes don't directly
   * cause numerical issues there. The main concern is warmup efficiency and edge cases.
   */
  double amplitude_min_threshold = std::numeric_limits<double>::min();

  /**
   * @brief Maximum amplitude threshold for valid configurations.
   *
   * Configurations with |amplitude| >= this threshold are considered invalid and will
   * trigger rescue if enabled. The default value is std::numeric_limits<double>::max(),
   * which is the most permissive setting.
   */
  double amplitude_max_threshold = std::numeric_limits<double>::max();
};

/**
 * @brief Executor-level runtime parameter pack.
 *
 * This is intended to live alongside other top-level parameter packs (optimizer/mc/peps).
 * Extend this struct when new runtime behavior categories are introduced.
 */
struct RuntimeParams {
  PsiConsistencyWarningParams psi_consistency;
  ConfigurationRescueParams config_rescue;
};

/**
 * @brief Psi consistency summary for one Monte-Carlo configuration.
 *
 * - psi_mean: aligned arithmetic mean of all contraction results in psi_list
 * - psi_rel_err: radius_rel = max_i |psi_i - psi_mean| / |psi_mean|
 */
template<typename TenElemT>
struct PsiConsistencySummary {
  TenElemT psi_mean{};
  double psi_rel_err = 0.0;
};

/**
 * @brief Compute psi_mean and psi_rel_err after aligning fermionic sign/phase branches.
 *
 * Alignment:
 * - pick the element with largest magnitude as reference psi_ref
 * - flip sign for samples with negative overlap: Re[psi_i * conj(psi_ref)] < 0
 *
 * This is robust for complex amplitudes and matches the measurement pipeline behavior.
 */
template<typename TenElemT>
inline PsiConsistencySummary<TenElemT> ComputePsiConsistencySummaryAligned(
    const std::vector<TenElemT> &psi_list) {
  PsiConsistencySummary<TenElemT> out{};
  if (psi_list.empty()) {
    out.psi_mean = TenElemT(0);
    out.psi_rel_err = 0.0;
    return out;
  }

  // Choose the largest magnitude as reference to stabilize phase/sign alignment.
  constexpr double kReferenceTol = 1e-14;
  size_t ref_index = 0;
  double ref_abs = 0.0;
  for (size_t i = 0; i < psi_list.size(); ++i) {
    const double mag = static_cast<double>(std::abs(psi_list[i]));
    if (mag > ref_abs) {
      ref_abs = mag;
      ref_index = i;
    }
  }
  const bool ref_valid = ref_abs > kReferenceTol;
  const TenElemT psi_ref = psi_list[ref_index];

  TenElemT mean(0);
  std::vector<TenElemT> aligned;
  aligned.reserve(psi_list.size());
  using std::conj;
  using std::real;
  for (const auto &psi_val : psi_list) {
    TenElemT aligned_val = psi_val;
    if (ref_valid) {
      const auto overlap = psi_val * conj(psi_ref);
      if (static_cast<double>(real(overlap)) < 0.0) {
        aligned_val = -psi_val;
      }
    }
    aligned.push_back(aligned_val);
    mean += aligned_val;
  }
  mean = mean / static_cast<double>(psi_list.size());

  auto abs_val = [](const TenElemT &v) -> double { return static_cast<double>(std::abs(v)); };
  const double denom = std::max(abs_val(mean), std::numeric_limits<double>::epsilon());
  double max_dev = 0.0;
  for (const auto &v : aligned) {
    max_dev = std::max(max_dev, abs_val(v - mean));
  }

  out.psi_mean = mean;
  out.psi_rel_err = max_dev / denom;
  return out;
}

template<typename TenElemT>
inline std::string FormatPsiListTruncated(const std::vector<TenElemT> &psi_list,
                                          size_t max_elems) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  oss << "[";
  const size_t n = psi_list.size();
  const size_t m = std::min(n, max_elems);
  for (size_t i = 0; i < m; ++i) {
    if (i) { oss << ", "; }
    oss << psi_list[i];
  }
  if (n > m) {
    oss << ", ... (n=" << n << ")";
  }
  oss << "]";
  return oss.str();
}

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_PSI_CONSISTENCY_H


