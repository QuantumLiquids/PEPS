/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-04
*
* Description: QuantumLiquids/PEPS project. Abstract class of wavefunction component.
*/

#ifndef QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
#define QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "qlpeps/vmc_basic/configuration.h"         // Configuration
#include "qlpeps/one_dim_tn/boundary_mps/bmps.h"    // BMPSTruncateParams
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"  // SplitIndexTPS
#include "qlpeps/vmc_basic/jastrow_factor.h"        // JastrowFactor
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h" // BMPSContractor
namespace qlpeps {

namespace detail {
template <class Contractor, class TenElemT, class QNT>
concept HasBMPSWorkflow = requires(Contractor c, TensorNetwork2D<TenElemT, QNT>& tn) {
  c.GrowBMPSForRow(tn, 0);
  c.GrowFullBTen(tn, RIGHT, 0, 2, true);
  c.InitBTen(tn, LEFT, 0);
  c.Trace(tn, SiteIdx{0, 0}, HORIZONTAL);
};

template <class Contractor>
concept HasSetTruncateParams = requires(Contractor c, const typename Contractor::TruncateParams& tp) {
  c.SetTruncateParams(tp);
};

template <class Contractor>
concept HasTrialType = requires { typename Contractor::Trial; };

template <class Contractor>
concept HasClearCache = requires(Contractor c) {
  c.ClearCache();
};

template <class Contractor, bool kHasTrial = HasTrialType<Contractor>>
struct TrialTokenHelper_ {
  using type = std::monostate;
};

template <class Contractor>
struct TrialTokenHelper_<Contractor, true> {
  using type = typename Contractor::Trial;
};

template <class Contractor>
using TrialTokenT = typename TrialTokenHelper_<Contractor>::type;

template <class Contractor, class SiteIdxT, class TensorT>
concept HasBeginTrialWithReplacement =
    HasTrialType<Contractor> &&
    requires(const Contractor c, const std::vector<std::pair<SiteIdxT, TensorT>>& repl) {
      { c.BeginTrialWithReplacement(repl) } -> std::same_as<typename Contractor::Trial>;
    };

template <class Contractor, class SiteIdxT, class TensorT>
concept HasTraceWithReplacement = requires(const Contractor c, const std::vector<std::pair<SiteIdxT, TensorT>>& repl) {
  c.TraceWithReplacement(repl);
};

template <class Contractor>
concept HasCommitTrial =
    HasTrialType<Contractor> &&
    requires(Contractor c, typename Contractor::Trial t) {
      c.CommitTrial(std::move(t));
    };

template <class Contractor, class SiteIdxT>
concept HasCheckInvalidateEnvs = requires(const Contractor c, const SiteIdxT &site) {
  c.CheckInvalidateEnvs(site);
};

template <class Contractor, class SiteIdxT>
concept HasEraseEnvsAfterUpdate = requires(Contractor c, const SiteIdxT &site) {
  c.EraseEnvsAfterUpdate(site);
};
}  // namespace detail

///< abstract wave function component, useless up to now.
template<typename TenElemT>
struct WaveFunctionComponent {
  Configuration config;
  TenElemT amplitude;

  WaveFunctionComponent(const size_t rows, const size_t cols) :
      config(rows, cols), amplitude(0) {}
  WaveFunctionComponent(const Configuration &config) : config(config), amplitude(0) {}
};

struct NoDress {
  NoDress() = default;
};

/**
 * The absolute value of Jastrow factor is no sense. Only calculate the ratio of Jastrow factor.
 */
struct JastrowDress {
 public:
  JastrowFactor jastrow;
  DensityConfig density_config; // number of particle in each site

  JastrowDress(const JastrowFactor &jastrow) : jastrow(jastrow) {}

  template<typename... Args>
  void UpdateLocalDensity(const Args... site_density_pairs) {
    (UpdateSingleSiteDensity_(site_density_pairs.first, site_density_pairs.second), ...);
  }
 private:
  void UpdateSingleSiteDensity_(const SiteIdx &site, size_t new_density) {
    density_config(site) = new_density;
  }
};

/**
 * Wave function component based on tensor product state (TPS)
 *
 * The default wave function type is PEPS
 *
 * For with Dress = JastrowDress, Jastrow-factor dressed PEPS wave function.
 * For Jastrow-factor dressed PEPS wave function, note in the implementation,
 * The member `amplitude` naturally encode the amplitude of PEPS part, rather the overall amplitude.
 * So here may need careful.
 *
 * We hope it can support the spin inversion symmetry in the future.
 */
template<typename TenElemT, typename QNT, typename Dress = qlpeps::NoDress, template<typename, typename> class ContractorT = BMPSContractor>
struct TPSWaveFunctionComponent {
 public:
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using Contractor = ContractorT<TenElemT, QNT>;
  using Tensor = typename Contractor::Tensor;
  /// @brief Use the contractor's own TruncateParams type for type-safe parameter passing
  using TruncateParams = typename Contractor::TruncateParams;
  using TrialToken = detail::TrialTokenT<Contractor>;
  ///< No initialized construct. considering to be removed in future.
  TPSWaveFunctionComponent(const size_t rows, const size_t cols, const TruncateParams &truncate_para) :
      config(rows, cols), amplitude(0), tn(rows, cols), trun_para(truncate_para), contractor(rows, cols) {
    // IMPORTANT:
    // This constructor does NOT initialize `tn` with projected site tensors.
    // Calling contractor.Init(tn) here would dereference default tensors and crash.
    // Proper initialization must be done by
    // using the constructor taking (sitps, config, truncate_para).
  }

  TPSWaveFunctionComponent(const SplitIndexTPS<TenElemT, QNT> &sitps,
                           const Configuration &config,
                           const TruncateParams &truncate_para)
      : config(config), tn(config.rows(), config.cols()), trun_para(truncate_para), contractor(config.rows(), config.cols()) {
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config); // projection
    contractor.Init(tn);
    EvaluateAmplitude();
  }

  const TenElemT &GetAmplitude(void) const { return amplitude; }
    Configuration &GetConfig(void) { return config; }
  size_t GetConfiguration(const SiteIdx &site) const { return config(site); }

  /**
   * @param sitps
   * @param occupancy_num
   */
  void RandomInit(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  const std::vector<size_t> &occupancy_num) {
    this->config.Random(occupancy_num);
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, this->config);
    contractor.Init(tn);
    EvaluateAmplitude();
  }

  void ReplaceGlobalConfig(const SplitIndexTPS<TenElemT, QNT> &sitps, const Configuration &config_new) {
    this->config = config_new;
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config);
    contractor.Init(tn);
    EvaluateAmplitude();
  }

  TenElemT EvaluateAmplitude() {
    if (tn.rows() == 0 || tn.cols() == 0 || tn({0, 0}).IsDefault()) {
      throw std::logic_error("TPSWaveFunctionComponent::EvaluateAmplitude: TensorNetwork2D is not initialized.");
    }

    if constexpr (detail::HasSetTruncateParams<Contractor>) {
      contractor.SetTruncateParams(this->trun_para);
    }

    if constexpr (detail::HasBMPSWorkflow<Contractor, TenElemT, QNT>) {
      contractor.GrowBMPSForRow(tn, 0);
      contractor.GrowFullBTen(tn, RIGHT, 0, 2, true);
      contractor.InitBTen(tn, LEFT, 0);
      this->amplitude = contractor.Trace(tn, {0, 0}, HORIZONTAL);
    } else {
      this->amplitude = contractor.Trace(tn);
    }

    if (!IsAmplitudeSquareLegal()) {
      std::cout << "warning : wavefunction amplitude = "
                << this->amplitude
                << ", square of amplitude will be illegal! "
                << std::endl;
    }
    return this->amplitude;
  }

  /**
   * @brief Start a trial move by temporarily replacing local tensors, without touching tn/config.
   *
   * This is the only safe place to create "shadow caches" for a trial move. The resulting trial
   * token is kept until AcceptTrial/RejectTrial.
   *
   * @param replacements (site, new_tensor) list.
   * @param new_configs (site, new_config) list (must correspond to replacements).
   */
  struct Trial {
    TenElemT amplitude{};
    std::vector<std::pair<SiteIdx, Tensor>> replacements;
    std::vector<std::pair<SiteIdx, size_t>> new_configs;
    TrialToken token{};
  };

  Trial BeginTrial(const std::vector<std::pair<SiteIdx, Tensor>>& replacements,
                   const std::vector<std::pair<SiteIdx, size_t>>& new_configs) {
    if (detail::HasSetTruncateParams<Contractor>) {
      contractor.SetTruncateParams(this->trun_para);
    }

    Trial trial;
    trial.replacements = replacements;
    trial.new_configs = new_configs;

    if constexpr (detail::HasBeginTrialWithReplacement<Contractor, SiteIdx, Tensor> && detail::HasCommitTrial<Contractor>) {
      trial.token = contractor.BeginTrialWithReplacement(trial.replacements);
      trial.amplitude = trial.token.amplitude;
    } else if constexpr (detail::HasTraceWithReplacement<Contractor, SiteIdx, Tensor>) {
      // Fallback: compute amplitude only; AcceptTrial will force a full re-trace if needed.
      trial.amplitude = contractor.TraceWithReplacement(trial.replacements);
    } else {
      throw std::logic_error("BeginTrial: contractor does not support trial evaluation.");
    }

    return trial;
  }

  /**
   * @brief Commit the pending trial: update config/tn, and swap-in contractor caches if supported.
   */
  void AcceptTrial(Trial&& trial, const SplitIndexTPS<TenElemT, QNT>& sitps) {

    // 1) Commit contractor caches first (so it doesn't see a dirty flag from UpdateSingleSite_()).
    if constexpr (detail::HasCommitTrial<Contractor>) {
      // Only meaningful if token holds a real trial.
      if constexpr (!std::is_same_v<TrialToken, std::monostate>) {
        contractor.CommitTrial(std::move(trial.token));
      }
    }

    // 2) Apply to config + TN (source of truth for tensors).
    for (const auto& sc : trial.new_configs) {
      const SiteIdx& site = sc.first;
      const size_t cfg = sc.second;
      config(site) = cfg;
      tn.UpdateSiteTensor(site, cfg, sitps);
#ifndef NDEBUG
      // If we updated the TN directly, we must ensure contractor caches cannot go stale.
      // - Preferred: CommitTrial(trial_token) for contractors that support trials.
      // - Fallback: EraseEnvsAfterUpdate(site) for BMPS.
      // - Fallback: ClearCache() for TRG (or other contractors with global caches).
      static_assert(detail::HasCommitTrial<Contractor> ||
                        detail::HasEraseEnvsAfterUpdate<Contractor, SiteIdx> ||
                        detail::HasClearCache<Contractor>,
                    "TPSWaveFunctionComponent::AcceptTrial requires the contractor to support either "
                    "CommitTrial(trial_token), EraseEnvsAfterUpdate(site), or ClearCache() to avoid stale cached environments.");
#endif

      // If CommitTrial is unavailable, we *must* invalidate caches after direct TN updates.
      if constexpr (!detail::HasCommitTrial<Contractor>) {
        if constexpr (detail::HasEraseEnvsAfterUpdate<Contractor, SiteIdx>) {
          contractor.EraseEnvsAfterUpdate(site);
        } else if constexpr (detail::HasClearCache<Contractor>) {
          contractor.ClearCache();
        } 
#ifndef NDEBUG
        // Debug-only: verify that invalidation actually removed stale environments (BMPS only).
        if constexpr (detail::HasCheckInvalidateEnvs<Contractor, SiteIdx>) {
          contractor.CheckInvalidateEnvs(site);
        }
#endif
      }
    }

    // 3) Update amplitude and clear trial.
    amplitude = trial.amplitude;
  }

  /**
   * @brief Discard the pending trial (no state changes).
   */
  void RejectTrial(Trial&& /*trial*/) {}

  bool IsAmplitudeSquareLegal() const {
    const double min_positive = std::numeric_limits<double>::min();
    const double max_positive = std::numeric_limits<double>::max();
    return !std::isnan(std::abs(amplitude))
        && std::abs(amplitude) > std::sqrt(min_positive)
        && std::abs(amplitude) < std::sqrt(max_positive);
  }

  [[nodiscard]] bool IsZero() const { return (amplitude == TenElemT(0)); }

  /**
   * @brief Updates local configuration and amplitude for arbitrary number of sites
   * 
   * @param sitps Split-index TPS containing the wavefunction
   * @param new_amplitude New wavefunction amplitude after update
   * @param site_configs Pairs of (site, new_config)
   * 
   * Usage examples:
   * @code
   *   // Single site update
   *   component.UpdateLocal(sitps, new_amplitude, {site1, config1});
   *   
   *   // Two sites update
   *   component.UpdateLocal(sitps, new_amplitude, 
   *                        {site1, config1}, 
   *                        {site2, config2});
   *   
   *   // Multiple sites update
   *   component.UpdateLocal(sitps, new_amplitude,
   *                        {site1, config1},
   *                        {site2, config2},
   *                        {site3, config3});
   * @endcode
   */
  // Direct-update API: only supported for BMPS-style contractors that can erase
  // cached environments incrementally after a local TN update.
  template<typename... Args>
  void UpdateLocal(const SplitIndexTPS<TenElemT, QNT> &sitps,
                   const TenElemT new_amplitude,
                   const Args... site_configs)
    requires(detail::HasEraseEnvsAfterUpdate<Contractor, SiteIdx>) {
    (UpdateSingleSite_(site_configs.first, site_configs.second, sitps), ...);
    amplitude = new_amplitude;
  }

  Configuration config;
  TenElemT amplitude;
  TensorNetwork2D<TenElemT, QNT> tn;
  Contractor contractor;
  TruncateParams trun_para;  ///< Uses contractor-specific truncation params type
  Dress dress;
 private:
  void UpdateSingleSite_(const SiteIdx &site,
                         const size_t new_config,
                         const SplitIndexTPS<TenElemT, QNT> &sitps) {
    config(site) = new_config;
    tn.UpdateSiteTensor(site, new_config, sitps); 
#ifndef NDEBUG
    static_assert(detail::HasEraseEnvsAfterUpdate<Contractor, SiteIdx>,
                  "UpdateSingleSite_ is a direct TensorNetwork2D update API and is only supported "
                  "for contractors providing EraseEnvsAfterUpdate(site) (e.g. BMPS).");
#endif
    // Direct TN update path: BMPS must erase cached environments to avoid stale reuse.
    contractor.EraseEnvsAfterUpdate(site);
#ifndef NDEBUG
    if constexpr (detail::HasCheckInvalidateEnvs<Contractor, SiteIdx>) {
      contractor.CheckInvalidateEnvs(site);
    }
#endif
  }
};

/**
 * @brief Check if wavefunction amplitude is valid for Monte Carlo sampling.
 *
 * Checks that the amplitude magnitude is within the given thresholds and is not NaN or Inf.
 *
 * @tparam WFC Wavefunction component type with amplitude member
 * @param wfc The wavefunction component to check
 * @param min_threshold Minimum amplitude threshold (must be explicitly specified)
 * @param max_threshold Maximum amplitude threshold (must be explicitly specified)
 * @return true if amplitude is valid for MC sampling
 */
template<typename WFC>
bool CheckWaveFunctionAmplitudeValidity(
    const WFC &wfc,
    double min_threshold,
    double max_threshold) {
  const double amp_mag = std::abs(wfc.amplitude);
  return !std::isnan(amp_mag) && !std::isinf(amp_mag) &&
         amp_mag > min_threshold &&
         amp_mag < max_threshold;
}

}//qlpeps


#endif //QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
