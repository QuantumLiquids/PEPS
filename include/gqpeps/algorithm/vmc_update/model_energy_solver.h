/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: GraceQ/VMC-PEPS project. Model Energy Solver base class. Also an example on how to write a ModelEnergySolver.
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/tps/split_index_tps.h"      //SplitIndexTPS
#include "gqpeps/algorithm/vmc_update/wave_function_component_classes/square_tps_sample_nn_flip.h"     //SquareTPSSampleNNFlip

namespace gqpeps {

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

}//gqpeps

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
