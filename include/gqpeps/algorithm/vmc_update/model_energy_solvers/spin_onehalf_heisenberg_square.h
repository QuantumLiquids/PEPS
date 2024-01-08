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

  template<typename WaveFunctionComponentType, bool calchols = true>
  TenElemT CalEnergyAndHoles(
      const SITPS *sitps,
      WaveFunctionComponentType *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res
  );

};

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType, bool calchols>
TenElemT SpinOneHalfHeisenbergSquare<TenElemT, QNT>::CalEnergyAndHoles(const SITPS *split_index_tps,
                                                                       WaveFunctionComponentType *tps_sample,
                                                                       TensorNetwork2D<TenElemT, QNT> &hole_res) {
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = SquareTPSSampleNNFlip<TenElemT, QNT>::trun_para;
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    // update the amplitude so that the error of ratio of amplitude can reduce by cancellation.
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL));
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        if (config(site1) == config(site2)) {
          energy += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, HORIZONTAL,
                                                  (*split_index_tps)(site1)[config(site2)],
                                                  (*split_index_tps)(site2)[config(site1)]);
          if (std::abs(psi_ex * inv_psi) > 1.0e8) {
            std::cerr << "psi_exchange : " << std::scientific << psi_ex
                      << ", psi_0 : " << std::scientific << tps_sample->amplitude
                      << std::endl;
          }
          energy += (-0.25 + psi_ex * inv_psi * 0.5);
        }
        tn.BTenMoveStep(RIGHT);
      }
    }
    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  //Calculate vertical bond energy contribution
  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    tps_sample->amplitude = tn.Trace({0, col}, VERTICAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        energy += 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        if (std::abs(psi_ex * inv_psi) > 1.0e8) {
          std::cerr << "psi_exchange : " << std::scientific << psi_ex
                    << ", psi_0 : " << std::scientific << tps_sample->amplitude
                    << std::endl;
        }
        energy += (-0.25 + psi_ex * inv_psi * 0.5);
      }
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  return energy;
}
}//gqpeps




#endif //GRACEQ_VMC_PEPS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
