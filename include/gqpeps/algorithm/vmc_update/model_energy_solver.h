/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: GraceQ/VMC-PEPS project. Model Energy Solver base class
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/tps/split_index_tps.h"      //SplitIndexTPS
#include "gqpeps/algorithm/vmc_update/tps_sample.h"     //TPSSample

namespace gqpeps {

template<typename TenElemT, typename QNT>
class ModelEnergySolver {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  ModelEnergySolver(void) = default;

  virtual TenElemT CalEnergyAndHoles(
      const SITPS *sitps,
      TPSSample<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res  // the return value
  ) {
    TenElemT energy(0);
    return energy;
  }

  virtual TenElemT CalEnergy(
      const SITPS *sitps,
      TPSSample<TenElemT, QNT> *tps_sample
  ) {
    TenElemT energy(0);
    return energy;
  }
 protected:
};

}//gqpeps

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
