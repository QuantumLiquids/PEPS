/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-12-30
*
* Description: GraceQ/VMC-PEPS project. Model Energy Solver for spin-1/2 Triangle Heisenberg J1-J2 model on square PEPS
*/

#ifndef GRACEQ_GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H
#define GRACEQ_GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H

#include "gqpeps/algorithm/vmc_update/model_energy_solver.h"    //ModelEnergySolver

namespace gqpeps {
using namespace gqten;

template<typename TenElemT, typename QNT>
class SpinOneHalfTriJ1J2HeisenbergSqrPEPS : public ModelEnergySolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SpinOneHalfTriJ1J2HeisenbergSqrPEPS(void) = delete;

  SpinOneHalfTriJ1J2HeisenbergSqrPEPS(double j2) : j2_(j2) {}

  template<typename WaveFunctionComponentType, bool calchols = true>
  TenElemT CalEnergyAndHoles(
      const SITPS *sitps,
      WaveFunctionComponentType *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res
  );

  template<typename WaveFunctionComponentType>
  ObservablesLocal<TenElemT> SampleMeasure(
      const SITPS *sitps,
      WaveFunctionComponentType *tps_sample
  );
 private:
  double j2_;
};

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType, bool calchols>
TenElemT SpinOneHalfTriJ1J2HeisenbergSqrPEPS<TenElemT, QNT>::
CalEnergyAndHoles(const SITPS *split_index_tps,
                  WaveFunctionComponentType *tps_sample,
                  TensorNetwork2D<TenElemT, QNT> &hole_res) {
  TenElemT e1(0), e2(0); // energy in J1 and J2 bond respectively
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = SquareTPSSampleNNFlip<TenElemT, QNT>::trun_para;
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
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
      tn.InitBTen2(LEFT, row);
      tn.GrowFullBTen2(RIGHT, row, 2, true);

      for (size_t col = 0; col < tn.cols() - 1; col++) {
        //Calculate diagonal J1 energy contribution
        SiteIdx site1 = {row + 1, col}; //left-down
        SiteIdx site2 = {row, col + 1}; //right-up
        if (config(site1) == config(site2)) {
          e1 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTDOWN_TO_RIGHTUP,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          e1 += (-0.25 + psi_ex * inv_psi * 0.5);
        }
        //Calculate J2 contribution
        site1 = {row, col}; //left-top
        site2 = {row + 1, col + 1}; //right-bottom
        if (config(site1) == config(site2)) {
          e2 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTUP_TO_RIGHTDOWN,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          e2 += (-0.25 + psi_ex * inv_psi * 0.5);
        }

        if (col < tn.cols() - 2) {
          SiteIdx site1 = {row + 1, col}; //left-bottom
          SiteIdx site2 = {row, col + 2}; //right-top
          if (config(site1) == config(site2)) {
            e2 += 0.25;
          } else {
            TenElemT psi_ex = tn.ReplaceSqrt5DistTwoSiteTrace({row, col},
                                                              LEFTDOWN_TO_RIGHTUP,
                                                              HORIZONTAL,
                                                              (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                              (*split_index_tps)(site2)[config(site1)]);
            e2 += (-0.25 + psi_ex * inv_psi * 0.5);
          }
        }
        tn.BTen2MoveStep(RIGHT, row);
      }
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    tps_sample->amplitude = tn.Trace({0, col}, VERTICAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    //Calculate vertical bond energy contribution
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
      tn.InitBTen2(UP, col);
      tn.GrowFullBTen2(DOWN, col, 3, true);
      //Calculate J2 energy contribution
      for (size_t row = 0; row < tn.rows() - 2; row++) {
        const SiteIdx site1 = {row + 2, col};
        const SiteIdx site2 = {row, col + 1};
        if (config(site1) == config(site2)) {
          e2 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceSqrt5DistTwoSiteTrace({row, col},
                                                            LEFTDOWN_TO_RIGHTUP,
                                                            VERTICAL,
                                                            (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                            (*split_index_tps)(site2)[config(site1)]);
          e2 += (-0.25 + psi_ex * inv_psi * 0.5);
        }
        if ((int) row < (int) tn.rows() - 3) {
          tn.BTen2MoveStep(DOWN, col);
        }
      }
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  return e1 + j2_ * e2;
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType>
ObservablesLocal<TenElemT> SpinOneHalfTriJ1J2HeisenbergSqrPEPS<TenElemT, QNT>::SampleMeasure(
    const SITPS *split_index_tps,
    WaveFunctionComponentType *tps_sample
) {





}

}//gqpeps
#endif //GRACEQ_GQPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H
