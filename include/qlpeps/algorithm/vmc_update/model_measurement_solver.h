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

namespace qlpeps {

template<typename ElemT>
struct ObservablesLocal {
  ElemT energy_loc;
  std::vector<ElemT> bond_energys_loc;
  std::vector<ElemT> one_point_functions_loc;
  std::vector<ElemT> two_point_functions_loc;

  ObservablesLocal(void) = default;
};

/**
 * @brief ModelMeasurementSolver is a base class used for Monte-Carlo based measurements
 *        on TPS. It rewrite the operator() so that it works as a functor which calculate
 *        the local observations upon specific Monte-Carlo samples
 *        It uses CRTP (Curiously Recurring Template Pattern).
 *
 * In the inherited class, the function SampleMeasureImpl will define the model-dependent
 * measurement actions.
 * @tparam ConcreteModelSolver the derived class
 */
template<typename ConcreteModelSolver>
class ModelMeasurementSolver {
 public:
  ModelMeasurementSolver(void) = default;

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> operator()(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample
  ) {
    std::vector<TenElemT> psi_list;
//    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
//    const Configuration &sample_config = tps_sample->config;
//    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
//    /**  Evaluate total energy, bond energy, and correlation functions from current MC sample  **/
//    ObservablesLocal<TenElemT> res_loc = static_cast<ConcreteModelSolver *>(this)->SampleMeasureImpl(sitps,
//                                                                                                     sample_config,
//                                                                                                     sample_tn,
//                                                                                                     trunc_para,
//                                                                                                     psi_list);
    /**  Evaluate total energy, bond energy, and correlation functions from current MC sample  **/
    ObservablesLocal<TenElemT> res_loc = static_cast<ConcreteModelSolver *>(this)->SampleMeasureImpl(sitps,
                                                                                                     tps_sample,
                                                                                                     psi_list);
    WaveFunctionAmplitudeConsistencyCheck(psi_list, wave_function_component_measure_accuracy);
    return res_loc;
  }
  const double wave_function_component_measure_accuracy = 1E-3;
};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_MEASUREMENT_SOLVER_H
