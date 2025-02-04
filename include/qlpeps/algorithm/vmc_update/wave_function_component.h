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

template<typename TenElemT, typename QNT>
class WaveFunctionComponent {
 public:
  Configuration config;
  TenElemT amplitude;

  static std::optional<BMPSTruncatePara> trun_para;

  WaveFunctionComponent(const size_t rows, const size_t cols) :
      config(rows, cols), amplitude(0) {}
  WaveFunctionComponent(const Configuration &config) : config(config), amplitude(0) {}

  bool IsZero() const { return (amplitude == TenElemT(0)); }

  virtual void MonteCarloSweepUpdate(const SplitIndexTPS<TenElemT, QNT> &sitps,
                                     std::uniform_real_distribution<double> &u_double,
                                     std::vector<double> &accept_rates) = 0;
};

template<typename ElemT>
bool IsAmplitudeSquareLegal(const ElemT &amplitude) {
  const double min_positive = std::numeric_limits<double>::min();
  const double max_positive = std::numeric_limits<double>::max();
  return std::abs(amplitude) > std::sqrt(min_positive)
      && std::abs(amplitude) < std::sqrt(max_positive);
}

template<typename WaveFunctionComponentType>
bool CheckWaveFunctionAmplitudeValidity(const WaveFunctionComponentType &tps_sample) {
  return (std::abs(tps_sample.amplitude) > std::numeric_limits<double>::epsilon()) &&
      !std::isnan(std::abs(tps_sample.amplitude)) && !std::isinf(std::abs(tps_sample.amplitude));
}

template<typename TenElemT, typename QNT>
std::optional<BMPSTruncatePara> WaveFunctionComponent<TenElemT, QNT>::trun_para;
}//qlpeps


#endif //QLPEPS_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
