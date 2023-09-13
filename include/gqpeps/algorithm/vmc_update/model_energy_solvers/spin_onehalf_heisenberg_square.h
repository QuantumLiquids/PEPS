/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-12
*
* Description: GraceQ/VMC-PEPS project. Model Energy Solver for spin-1/2 AFM Heisenberg model in square lattice
*/

#ifndef GRACEQ_VMC_PEPS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
#define GRACEQ_VMC_PEPS_SPIN_ONEHALF_HEISENBERG_SQUARE_H

#include "gqpeps/algorithm/vmc_update/model_energy_solver.h"    //ModelEnergySolver


namespace gqpeps {
using namespace gqten;


template<typename TenElemT, typename QNT>
class SpinOneHalfHeisenbergSquare : public ModelEnergySolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  using ModelEnergySolver<TenElemT, QNT>::ModelEnergySolver;

  TenElemT CalEnergyAndHoles(
      TensorNetwork2D<TenElemT, QNT> &hole_res
  ) override;
};

template<typename TenElemT, typename QNT>
TenElemT SpinOneHalfHeisenbergSquare<TenElemT, QNT>::CalEnergyAndHoles(TensorNetwork2D<TenElemT, QNT> &hole_res) {
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = this->tps_sample_->tn;
  const Configuration &config = this->tps_sample_->config;
  TenElemT inv_psi = 1.0 / (this->tps_sample_->amplitude);
  tn.GenerateBMPSApproach(UP);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL));
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        if (config(site1) == config(site2)) {
          energy += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, HORIZONTAL,
                                                  (*this->split_index_tps_)(site1)[config(site2)],
                                                  (*this->split_index_tps_)(site2)[config(site1)]);
          energy += (-0.25 + psi_ex * inv_psi * 0.5);
        }
        tn.BTenMoveStep(RIGHT);
      }
    }
    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN);
    }
  }

  //Calculate vertical bond energy contribution
  tn.GenerateBMPSApproach(LEFT);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        energy += 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*this->split_index_tps_)(site1)[config(site2)],
                                                (*this->split_index_tps_)(site2)[config(site1)]);
        energy += (-0.25 + psi_ex * inv_psi * 0.5);
      }
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT);
    }
  }
  return energy;
}

}//gqpeps




#endif //GRACEQ_VMC_PEPS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
