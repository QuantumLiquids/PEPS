/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-20
*
* Description: GraceQ/VMC-PEPS project. Model Energy Solver for spin-1/2 J1-J2 Heisenberg model in square lattice
*/

#ifndef GRACEQ_VMC_PEPS_SPIN_ONEHALF_SQUAREJ1J2_H
#define GRACEQ_VMC_PEPS_SPIN_ONEHALF_SQUAREJ1J2_H

#include "gqpeps/algorithm/vmc_update/model_energy_solver.h"    //ModelEnergySolver

namespace gqpeps {
using namespace gqten;

template<typename TenElemT, typename QNT>
class SpinOneHalfJ1J2HeisenbergSquare : public ModelEnergySolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SpinOneHalfJ1J2HeisenbergSquare(void) = delete;

  SpinOneHalfJ1J2HeisenbergSquare(double j2) : j2_(j2) {}

  TenElemT CalEnergyAndHoles(
      const SITPS *sitps,
      TPSSample<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res
  ) override;

  TenElemT CalEnergy(
      const SITPS *sitps,
      TPSSample<TenElemT, QNT> *tps_sample
  ) override;

 private:
  double j2_;
};

template<typename TenElemT, typename QNT>
TenElemT SpinOneHalfJ1J2HeisenbergSquare<TenElemT, QNT>::CalEnergyAndHoles(const SITPS *split_index_tps,
                                                                           TPSSample<TenElemT, QNT> *tps_sample,
                                                                           TensorNetwork2D<TenElemT, QNT> &hole_res) {
  TenElemT e1(0), e2(0); // energy in J1 and J2 bond respectively
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = TPSSample<TenElemT, QNT>::trun_para;
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
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
          e1 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, HORIZONTAL,
                                                  (*split_index_tps)(site1)[config(site2)],
                                                  (*split_index_tps)(site2)[config(site1)]);
          e1 += (-0.25 + psi_ex * inv_psi * 0.5);
        }
        tn.BTenMoveStep(RIGHT);
      }
    }
    if (row < tn.rows() - 1) {
      //calculate J2 energy
      tn.InitBTen2(LEFT, row);
      tn.GrowFullBTen2(RIGHT, row, 2, true);

      for (size_t col = 0; col < tn.cols() - 1; col++) {
        //Calculate diagonal energy contribution
        SiteIdx site1 = {row, col};
        SiteIdx site2 = {row + 1, col + 1};
        if (config(site1) == config(site2)) {
          e2 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace(site1,
                                                   LEFTUP_TO_RIGHTDOWN,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],
                                                   (*split_index_tps)(site2)[config(site1)]);
          e2 += (-0.25 + psi_ex * inv_psi * 0.5);
        }

        site1 = {row + 1, col}; //left-down
        site2 = {row, col + 1}; //right-up
        if (config(site1) == config(site2)) {
          e2 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTDOWN_TO_RIGHTUP,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          e2 += (-0.25 + psi_ex * inv_psi * 0.5);
        }
        tn.BTen2MoveStep(RIGHT, row);
      }
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  //Calculate vertical bond energy contribution
  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        e1 += 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        e1 += (-0.25 + psi_ex * inv_psi * 0.5);
      }
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  return e1 + j2_ * e2;
}

template<typename TenElemT, typename QNT>
TenElemT SpinOneHalfJ1J2HeisenbergSquare<TenElemT, QNT>::CalEnergy(const SITPS *split_index_tps,
                                                                   TPSSample<TenElemT, QNT> *tps_sample) {
  TenElemT e1(0), e2(0); // energy in J1 and J2 bond respectively
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = TPSSample<TenElemT, QNT>::trun_para;
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    for (size_t col = 0; col < tn.cols() - 1; col++) {
      const SiteIdx site1 = {row, col};
      //Calculate horizontal bond energy contribution
      const SiteIdx site2 = {row, col + 1};
      if (config(site1) == config(site2)) {
        e1 += 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, HORIZONTAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        e1 += (-0.25 + psi_ex * inv_psi * 0.5);
      }
      tn.BTenMoveStep(RIGHT);
    }
    if (row < tn.rows() - 1) {
      //calculate J2 energy
      tn.InitBTen2(LEFT, row);
      tn.GrowFullBTen2(RIGHT, row, 2, true);

      for (size_t col = 0; col < tn.cols() - 1; col++) {
        //Calculate diagonal energy contribution
        SiteIdx site1 = {row, col};
        SiteIdx site2 = {row + 1, col + 1};
        if (config(site1) == config(site2)) {
          e2 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace(site1,
                                                   LEFTUP_TO_RIGHTDOWN,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],
                                                   (*split_index_tps)(site2)[config(site1)]);
          e2 += (-0.25 + psi_ex * inv_psi * 0.5);
        }

        site1 = {row + 1, col}; //left-down
        site2 = {row, col + 1}; //right-up
        if (config(site1) == config(site2)) {
          e2 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTDOWN_TO_RIGHTUP,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          e2 += (-0.25 + psi_ex * inv_psi * 0.5);
        }
        tn.BTen2MoveStep(RIGHT, row);
      }
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  //Calculate vertical bond energy contribution
  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        e1 += 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        e1 += (-0.25 + psi_ex * inv_psi * 0.5);
      }
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  return e1 + j2_ * e2;
}

}//gqpeps


#endif //GRACEQ_VMC_PEPS_SPIN_ONEHALF_SQUAREJ1J2_H
