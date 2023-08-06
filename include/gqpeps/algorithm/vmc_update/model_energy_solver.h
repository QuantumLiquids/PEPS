/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: GraceQ/VMC-PEPS project. Model Energy Solver base class
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H

#include "gqten/gqten.h"
#include "gqpeps/two_dim_tn/tps/tps.h"
#include "gqpeps/two_dim_tn/tps/tensor_network_2d.h"

namespace gqpeps {
using namespace gqten;

template<typename TenElemT, typename QNT>
class ModelEnergySolver {
  using TenT = GQTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
 public:
  ModelEnergySolver(const TPST *tps,
                    const TensorNetwork2D<TenElemT, QNT> *proj_tn,
                    const Configuration *config,
                    std::vector<BMPS<TenElemT, QNT>> *bmps_down_set,
                    std::vector<BMPS<TenElemT, QNT>> *bmps_up_set,
                    std::vector<BMPS<TenElemT, QNT>> *bmps_left_set,
                    std::vector<BMPS<TenElemT, QNT>> *bmps_right_set)
      : tps_(tps),
        proj_tn_(proj_tn),
        config_(config),
        bmps_down_set_(bmps_down_set),
        bmps_up_set_(bmps_up_set),
        bmps_left_set_(bmps_left_set),
        bmps_right_set_(bmps_right_set) {
  }

  virtual TenElemT GetEnergy(void);

 private:
  const TPST *tps_;
  const TensorNetwork2D<TenElemT, QNT> *proj_tn_;
  const Configuration *config_;

  std::vector<BMPS<TenElemT, QNT>> *bmps_down_set_;
  std::vector<BMPS<TenElemT, QNT>> *bmps_up_set_;
  std::vector<BMPS<TenElemT, QNT>> *bmps_left_set_;
  std::vector<BMPS<TenElemT, QNT>> *bmps_right_set_;
};

}

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
