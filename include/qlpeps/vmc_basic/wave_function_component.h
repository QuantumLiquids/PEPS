/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-04
*
* Description: QuantumLiquids/PEPS project. Abstract class of wavefunction component.
*/

#ifndef QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
#define QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H

#include "qlpeps/vmc_basic/configuration.h"         // Configuration
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"    // BMPSTruncatePara
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"  // SplitIndexTPS
#include "qlpeps/vmc_basic/jastrow_factor.h"        // JastrowFactor
namespace qlpeps {

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
template<typename TenElemT, typename QNT, typename Dress = qlpeps::NoDress>
struct TPSWaveFunctionComponent {
 public:

  ///< No initialized construct. considering to be removed in future.
  TPSWaveFunctionComponent(const size_t rows, const size_t cols, const BMPSTruncatePara &truncate_para) :
      config(rows, cols), amplitude(0), tn(rows, cols), trun_para(truncate_para) {}

  TPSWaveFunctionComponent(const SplitIndexTPS<TenElemT, QNT> &sitps,
                           const Configuration &config,
                           const BMPSTruncatePara &truncate_para)
      : config(config), tn(config.rows(), config.cols()), trun_para(truncate_para) {
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config); // projection
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
    EvaluateAmplitude();
  }

  void ReplaceGlobalConfig(const SplitIndexTPS<TenElemT, QNT> &sitps, const Configuration &config_new) {
    this->config = config_new;
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config);
    EvaluateAmplitude();
  }

  TenElemT EvaluateAmplitude() {
    tn.GrowBMPSForRow(0, this->trun_para);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    this->amplitude = tn.Trace({0, 0}, HORIZONTAL);
    if (!IsAmplitudeSquareLegal()) {
      std::cout << "warning : wavefunction amplitude = "
                << this->amplitude
                << ", square of amplitude will be illegal! "
                << std::endl;
    }
    return this->amplitude;
  }

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
  template<typename... Args>
  void UpdateLocal(const SplitIndexTPS<TenElemT, QNT> &sitps,
                   const TenElemT new_amplitude,
                   const Args... site_configs) {
    (UpdateSingleSite_(site_configs.first, site_configs.second, sitps), ...);
    amplitude = new_amplitude;
  }

  Configuration config;
  TenElemT amplitude;
  TensorNetwork2D<TenElemT, QNT> tn;
  BMPSTruncatePara trun_para;
  Dress dress;
 private:
  void UpdateSingleSite_(const SiteIdx &site,
                         const size_t new_config,
                         const SplitIndexTPS<TenElemT, QNT> &sitps) {
    config(site) = new_config;
    tn.UpdateSiteTensor(site, new_config, sitps);
  }
};

template<typename MonteCarloSweepUpdater>
bool CheckWaveFunctionAmplitudeValidity(const MonteCarloSweepUpdater &tps_sample) {
  return (std::abs(tps_sample.amplitude) > std::numeric_limits<double>::min()) &&
      (std::abs(tps_sample.amplitude) < std::numeric_limits<double>::max()) &&
      !std::isnan(std::abs(tps_sample.amplitude)) && !std::isinf(std::abs(tps_sample.amplitude));
}

}//qlpeps


#endif //QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
