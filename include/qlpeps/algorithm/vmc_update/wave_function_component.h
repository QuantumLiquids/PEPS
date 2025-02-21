/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-04
*
* Description: QuantumLiquids/PEPS project. Abstract class of wavefunction component.
*/

#ifndef QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
#define QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H

#include "qlpeps/two_dim_tn/tps/configuration.h"    //Configuration
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"    //BMPSTruncatePara
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"  //SplitIndexTPS

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

/**
 * Wave function component based on tensor product state (TPS)
 *
 *
 *
 */
template<typename TenElemT, typename QNT>
struct TPSWaveFunctionComponent {
 public:
  Configuration config;
  TenElemT amplitude;
  TensorNetwork2D<TenElemT, QNT> tn;
  static BMPSTruncatePara trun_para;

  ///< No initialized construct. considering to be removed in future.
  TPSWaveFunctionComponent(const size_t rows, const size_t cols, const BMPSTruncatePara &truncate_para) :
      config(rows, cols), amplitude(0), tn(rows, cols) { trun_para = truncate_para; }

  TPSWaveFunctionComponent(const SplitIndexTPS<TenElemT, QNT> &sitps,
                           const Configuration &config,
                           const BMPSTruncatePara &truncate_para)
      : config(config), tn(config.rows(), config.cols()) {
    trun_para = truncate_para;
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, config);
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
  }

  /**
   * @param sitps
   * @param occupancy_num
   */
  void RandomInit(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  const std::vector<size_t> &occupancy_num) {
    this->config.Random(occupancy_num);
    tn = TensorNetwork2D<TenElemT, QNT>(sitps, this->config);
    tn.GrowBMPSForRow(0, this->trun_para);
    tn.GrowFullBTen(RIGHT, 0, 2, true);
    tn.InitBTen(LEFT, 0);
    this->amplitude = tn.Trace({0, 0}, HORIZONTAL);
  }

  bool IsAmplitudeSquareLegal() const {
    const double min_positive = std::numeric_limits<double>::min();
    const double max_positive = std::numeric_limits<double>::max();
    return !std::isnan(std::abs(amplitude))
        && std::abs(amplitude) > std::sqrt(min_positive)
        && std::abs(amplitude) < std::sqrt(max_positive);
  }

  [[nodiscard]] bool IsZero() const { return (amplitude == TenElemT(0)); }
};

template<typename MonteCarloSweepUpdater>
bool CheckWaveFunctionAmplitudeValidity(const MonteCarloSweepUpdater &tps_sample) {
  return (std::abs(tps_sample.amplitude) > std::numeric_limits<double>::min()) &&
      (std::abs(tps_sample.amplitude) < std::numeric_limits<double>::max()) &&
      !std::isnan(std::abs(tps_sample.amplitude)) && !std::isinf(std::abs(tps_sample.amplitude));
}

template<typename TenElemT, typename QNT>
BMPSTruncatePara TPSWaveFunctionComponent<TenElemT, QNT>::trun_para;
}//qlpeps


#endif //QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
