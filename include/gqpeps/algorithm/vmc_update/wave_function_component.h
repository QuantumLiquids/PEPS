/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-04
*
* Description: GraceQ/VMC-PEPS project. Abstract class of wavefunction component.
*/

#ifndef GRACEQ_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
#define GRACEQ_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H

#include "gqpeps/two_dim_tn/tps/configuration.h"    //Configuration
#include "gqpeps/ond_dim_tn/boundary_mps/bmps.h"    //BMPSTruncatePara
#include "gqpeps/two_dim_tn/tps/split_index_tps.h"  //SplitIndexTPS

namespace gqpeps {

template<typename TenElemT, typename QNT>
class WaveFunctionComponent {
 public:
  Configuration config;
  TenElemT amplitude;

  //try to think a better design
  static BMPSTruncatePara trun_para;

  WaveFunctionComponent(const size_t rows, const size_t cols) :
      config(rows, cols), amplitude(0) {}
  WaveFunctionComponent(const Configuration &config) : config(config), amplitude(0) {}

  virtual void MonteCarloSweepUpdate(const SplitIndexTPS<TenElemT, QNT> &sitps,
                                     std::uniform_real_distribution<double> &u_double,
                                     std::vector<double> &accept_rates) = 0;
};

template<typename TenElemT, typename QNT>
BMPSTruncatePara WaveFunctionComponent<TenElemT, QNT>::trun_para = BMPSTruncatePara(0, 0, 0.0);

}//gqpeps


#endif //GRACEQ_VMC_PEPS_ALGORITHM_VMC_UPDATE_WAVE_FUNCTION_COMPONENT_H
