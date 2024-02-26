/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver base class. Also an example on how to write a ModelEnergySolver.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"      //SplitIndexTPS
#include "qlpeps/algorithm/vmc_update/wave_function_component_classes/square_tps_sample_nn_exchange.h"     //SquareTPSSampleNNExchange

namespace qlpeps {

template<typename TenElemT, typename QNT>
class ModelEnergySolver {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  ModelEnergySolver(void) = default;

  template<typename WaveFunctionComponentType, bool calchols = true>
  TenElemT CalEnergyAndHoles(
      const SITPS *sitps,
      WaveFunctionComponentType *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res  // the return value
  ) {
    TenElemT energy(0);
    return energy;
  }
 protected:
};

//helper
template<typename ElemT>
bool WaveFunctionAmplitudeConsistencyCheck(
    const std::vector<ElemT> &psi_list,
    const double critical_bias
) {
  std::vector<double> abs_psi(psi_list.size());
  std::transform(psi_list.begin(), psi_list.end(), abs_psi.begin(), [](const ElemT &value) {
    return std::abs(value);
  });
  double max_abs = *std::max_element(abs_psi.begin(), abs_psi.end());
  double min_abs = *std::min_element(abs_psi.begin(), abs_psi.end());

  double estimate_wavefunction_bias = (max_abs - min_abs) / max_abs;

  if (estimate_wavefunction_bias > critical_bias) {
    std::cout << "inconsistent wave function amplitudes :" << std::endl;
    for (const auto &element : psi_list) {
      std::cout << element << " ";
    }
    std::cout << std::endl;
    return false;
  }
  return true;
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
