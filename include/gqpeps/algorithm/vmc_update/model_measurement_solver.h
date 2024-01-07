/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: GraceQ/VMC-PEPS project. Model Measurement Solver base class. Also an example on how to write a ModelEnergySolver.
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_MEASUREMENT_SOLVER_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_MEASUREMENT_SOLVER_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/tps/split_index_tps.h"      //SplitIndexTPS

namespace gqpeps {

template<typename ElemT>
struct ObservablesLocal {
  ElemT energy_loc;
  std::vector<ElemT> bond_energys_loc;
  std::vector<ElemT> one_point_functions_loc;
  std::vector<ElemT> two_point_functions_loc;

  ObservablesLocal(void) = default;
};

template<typename TenElemT, typename QNT>
class ModelMeasurementSolver {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  ModelMeasurementSolver(void) = default;

  template<typename WaveFunctionComponentType>
  ObservablesLocal<TenElemT> SampleMeasure(
      const SITPS *sitps,
      WaveFunctionComponentType *tps_sample
  ) {
    return ObservablesLocal<TenElemT>();
  }
 protected:
};

}//gqpeps

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_MEASUREMENT_SOLVER_H
