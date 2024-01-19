/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-28
*
* Description: GraceQ/VMC-PEPS project. Model Energy Solver for spin-1/2 Triangle Heisenberg model on square PEPS
*/

#ifndef GRACEQ_VMC_PEPS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H
#define GRACEQ_VMC_PEPS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H

#include "gqpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "gqpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver

namespace gqpeps {
using namespace gqten;

template<typename TenElemT, typename QNT>
class SpinOneHalfTriHeisenbergSqrPEPS : public ModelEnergySolver<TenElemT, QNT>, ModelMeasurementSolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SpinOneHalfTriHeisenbergSqrPEPS(void) = default;

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
};

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType, bool calchols>
TenElemT SpinOneHalfTriHeisenbergSqrPEPS<TenElemT, QNT>::CalEnergyAndHoles(const SITPS *split_index_tps,
                                                                           WaveFunctionComponentType *tps_sample,
                                                                           TensorNetwork2D<TenElemT, QNT> &hole_res) {
  TenElemT e(0);
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
          e += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, HORIZONTAL,
                                                  (*split_index_tps)(site1)[config(site2)],
                                                  (*split_index_tps)(site2)[config(site1)]);
          e += (-0.25 + psi_ex * inv_psi * 0.5);
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
        SiteIdx site1 = {row + 1, col}; //left-down
        SiteIdx site2 = {row, col + 1}; //right-up
        if (config(site1) == config(site2)) {
          e += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTDOWN_TO_RIGHTUP,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          e += (-0.25 + psi_ex * inv_psi * 0.5);
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
    tps_sample->amplitude = tn.Trace({0, col}, VERTICAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        e += 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        e += (-0.25 + psi_ex * inv_psi * 0.5);
      }
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  return e;
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType>
ObservablesLocal<TenElemT> SpinOneHalfTriHeisenbergSqrPEPS<TenElemT, QNT>::SampleMeasure(
    const SITPS *split_index_tps,
    WaveFunctionComponentType *tps_sample
) {
  ObservablesLocal<TenElemT> res;

  TenElemT energy_loc(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const size_t lx = tn.cols();
  res.bond_energys_loc.reserve(tn.rows() * tn.cols() * 3);
  res.two_point_functions_loc.reserve(tn.cols() / 2 * 3);
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
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        if (config(site1) == config(site2)) {
          energy_loc += 0.25;
          res.bond_energys_loc.push_back(0.25);
        } else {
          TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, HORIZONTAL,
                                                  (*split_index_tps)(site1)[config(site2)],
                                                  (*split_index_tps)(site2)[config(site1)]);
          auto bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
          energy_loc += bond_energy;
          res.bond_energys_loc.push_back(bond_energy);
        }
        tn.BTenMoveStep(RIGHT);
      }
    }

    if (row == tn.rows() / 2) { //measure correlation in the middle bonds
      SiteIdx site1 = {row, lx / 4};

      // sz(i) * sz(j)
      double sz1 = config(site1) - 0.5;
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        double sz2 = config(site2) - 0.5;
        res.two_point_functions_loc.push_back(sz1 * sz2);
      }


      std::vector<TenElemT> diag_corr(lx / 2);// sp(i) * sm(j) or sm(i) * sp(j), the valid channel
      tn(site1) = (*split_index_tps)(site1)[1 - config(site1)]; //temporally change
      tn.TruncateBTen(LEFT, lx / 4 + 1); // may be above two lines should be summarized as an API
      tn.GrowBTenStep(LEFT);//left boundary tensor just across Lx/4
      tn.GrowFullBTen(RIGHT, row, lx / 4 + 2, false); //environment for Lx/4 + 1 site
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        //sm(i) * sp(j)
        if (config(site2) == config(site1)) {
          diag_corr[i-1] = 0.0;
        } else {
          TenElemT psi_ex = tn.ReplaceOneSiteTrace(site2, (*split_index_tps)(site2)[1 - config(site2)], HORIZONTAL);
          diag_corr[i-1] = (psi_ex * inv_psi);
        }
        tn.BTenMoveStep(RIGHT);
      }
      tn(site1) = (*split_index_tps)(site1)[config(site1)]; // change back

      if (config(site1) == 1) {
        for (size_t i = 1; i <= lx / 2; i++) {  //sp(i) * sm(j) = 0
          res.two_point_functions_loc.push_back(0.0);
        }
        res.two_point_functions_loc.insert(res.two_point_functions_loc.end(), diag_corr.begin(), diag_corr.end());
      } else {
        res.two_point_functions_loc.insert(res.two_point_functions_loc.end(), diag_corr.begin(), diag_corr.end());
        for (size_t i = 1; i <= lx / 2; i++) {  //sm(i) * sp(j) = 0
          res.two_point_functions_loc.push_back(0.0);
        }
      }
    }

    if (row < tn.rows() - 1) {
      //calculate J2 energy
      tn.InitBTen2(LEFT, row);
      tn.GrowFullBTen2(RIGHT, row, 2, true);

      for (size_t col = 0; col < tn.cols() - 1; col++) {
        //Calculate diagonal energy contribution
        SiteIdx site1 = {row + 1, col}; //left-down
        SiteIdx site2 = {row, col + 1}; //right-up
        if (config(site1) == config(site2)) {
          energy_loc += 0.25;
          res.bond_energys_loc.push_back(0.25);
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTDOWN_TO_RIGHTUP,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          auto bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
          energy_loc += bond_energy;
          res.bond_energys_loc.push_back(bond_energy);
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
    tps_sample->amplitude = tn.Trace({0, col}, VERTICAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        energy_loc += 0.25;
        res.bond_energys_loc.push_back(0.25);
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        auto bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
        energy_loc += bond_energy;
        res.bond_energys_loc.push_back(bond_energy);
      }
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  res.energy_loc = energy_loc;
  res.one_point_functions_loc.reserve(tn.rows() * tn.cols());
  for (auto &spin_config : config) {
    res.one_point_functions_loc.push_back((double) spin_config - 0.5);
  }
  return res;
}

}//gqpeps



#endif //GRACEQ_VMC_PEPS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H
