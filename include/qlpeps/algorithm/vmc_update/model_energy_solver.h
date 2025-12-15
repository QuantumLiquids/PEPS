/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver base class.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H

#include "qlten/qlten.h"
#include "qlpeps/vmc_basic/wave_function_component.h"  //TPSWaveFunctionComponent

namespace qlpeps {

//helper
template<typename ElemT>
bool WaveFunctionAmplitudeConsistencyCheck(
    const std::vector<ElemT> &psi_list,
    const double critical_bias
) {
  if (psi_list.empty()) {
    return true;
  }
  std::vector<double> abs_psi(psi_list.size());
  std::transform(psi_list.begin(), psi_list.end(), abs_psi.begin(), [](const ElemT &value) {
    return std::abs(value);
  });
  double max_abs = *std::max_element(abs_psi.begin(), abs_psi.end());
  double min_abs = *std::min_element(abs_psi.begin(), abs_psi.end());

  double estimate_wavefunction_bias = (max_abs - min_abs) / max_abs;

  if (estimate_wavefunction_bias > critical_bias) {
    std::cout << "inconsistent wave function amplitudes : "
              << "(" << min_abs << ", " << max_abs << ")"
              << std::endl;
    return false;
  }
  return true;
}

/**
 * @brief ModelEnergySolver is a base class used for calculating the energy and gradient hole
 *        during the optimization of the Tensor Product State (TPS).
 *        The class rewrite the operator() so that it works as a functor which calculate
 *        the energy and gradient info upon specific Monte-Carlo samples
 *        It uses CRTP (Curiously Recurring Template Pattern).
 *
 * In the inherited class, the function CalEnergyAndHolesImpl should be defined to evaluate
 * the energy and holes of 2D tensor-network (without additional divide on psi).
 * @tparam ConcreteModelSolver the derived class
 */
template<typename ConcreteModelSolver>
class ModelEnergySolver {
 public:
  ModelEnergySolver(void) = default;
  ModelEnergySolver(const double wave_function_component_accuracy) : wave_function_component_accuracy(
      wave_function_component_accuracy) {}
  /**
   *
   * @tparam calchols   whether calculate the gradient hole sample data and return in hole_res
   * @param sitps       the TPS wave function
   * @param tps_sample  the wave function component
   * @param hole_res    the gradient hole sample data, valid only when calchols==true
   * @return  evaluated total energy in current Monte Carlo samples
   */
  template<typename TenElemT, typename QNT, bool calchols, typename WaveFunctionComponentT>
  TenElemT CalEnergyAndHoles(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      WaveFunctionComponentT *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res  // the return value
  ) {
    std::vector<TenElemT> psi_list;

//    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
//    const Configuration &sample_config = tps_sample->config;
//    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
//    TenElemT energy = static_cast<ConcreteModelSolver *>(this)->template CalEnergyAndHolesImpl<calchols>(sitps,
//                                                                                                         sample_config,
//                                                                                                         sample_tn,
//                                                                                                         trunc_para,
//                                                                                                         hole_res,
//                                                                                                         psi_list);
    TenElemT energy = static_cast<ConcreteModelSolver *>(this)
                          ->template CalEnergyAndHolesImpl<TenElemT, QNT, calchols>(sitps, tps_sample, hole_res, psi_list);

    WaveFunctionAmplitudeConsistencyCheck(psi_list, wave_function_component_accuracy);
    return energy;
  }

  template<typename TenElemT, typename QNT, typename WaveFunctionComponentT>
  TenElemT CalEnergy(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      WaveFunctionComponentT *tps_sample
  ) {
    TensorNetwork2D<TenElemT, QNT> hole_res(1, 1);
    return CalEnergyAndHoles<TenElemT, QNT, false>(sitps, tps_sample, hole_res);
  }

  const double wave_function_component_accuracy = 1E-3;
};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
