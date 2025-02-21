/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-20
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 J1-J2 Heisenberg model in square lattice
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "spin_onehalf_heisenberg_square.h"                       // EvaluateNNBondEnergyForAFMHeisenbergModel
namespace qlpeps {
using namespace qlten;

class SpinOneHalfJ1J2HeisenbergSquare : public ModelEnergySolver<SpinOneHalfJ1J2HeisenbergSquare>,
                                        public ModelMeasurementSolver<SpinOneHalfJ1J2HeisenbergSquare> {
 public:
  using ModelEnergySolver<SpinOneHalfJ1J2HeisenbergSquare>::CalEnergyAndHoles;
  using ModelMeasurementSolver<SpinOneHalfJ1J2HeisenbergSquare>::operator();

  SpinOneHalfJ1J2HeisenbergSquare(void) = delete;

  SpinOneHalfJ1J2HeisenbergSquare(double j2) : j2_(j2) {}

  template<typename TenElemT, typename QNT, bool calchols = true>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  );

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> SampleMeasureImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      std::vector<TenElemT> &psi_list
  );

 private:
  double j2_;
};

template<typename TenElemT, typename QNT>
TenElemT EvaluateNNNSSForAFMHeisenbergModel(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    const TenElemT inv_psi
) {
  if (config1 == config2) {
    return 0.25;
  } else {
    SiteIdx left_up_site;
    if (diagonal_dir == LEFTUP_TO_RIGHTDOWN) {
      left_up_site = site1;
    } else {
      left_up_site = {site2.row(), site1.col()};
    }
    TenElemT psi_ex = tn.ReplaceNNNSiteTrace(left_up_site,
                                             diagonal_dir,
                                             HORIZONTAL,
                                             split_index_tps_on_site1[config2],
                                             split_index_tps_on_site2[config1]);
    TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
    return (-0.25 + ratio * 0.5);
  }
}

template<typename TenElemT, typename QNT>
ObservablesLocal<TenElemT> SpinOneHalfJ1J2HeisenbergSquare::SampleMeasureImpl(
    const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
    TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
    std::vector<TenElemT> &psi_list
) {
  ObservablesLocal<TenElemT> res;
  TenElemT e1(0), e2(0); // energy in J1 and J2 bond respectively
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const size_t lx = tn.cols();
  res.bond_energys_loc.reserve(tn.rows() * tn.cols() * 4);
  res.two_point_functions_loc.reserve(tn.cols() / 2 * 3);
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = TPSWaveFunctionComponent<TenElemT, QNT>::trun_para;
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(lx + tn.rows());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_list.push_back(tps_sample->amplitude);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        TenElemT bond_energy = EvaluateNNBondEnergyForAFMHeisenbergModel(site1, site2,
                                                                         config(site1),
                                                                         config(site2),
                                                                         HORIZONTAL,
                                                                         tn,
                                                                         (*split_index_tps)(site1),
                                                                         (*split_index_tps)(site2),
                                                                         inv_psi);
        res.bond_energys_loc.push_back(bond_energy);
        e1 += bond_energy;
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
      tn.GrowBTenStep(LEFT);
      tn.GrowFullBTen(RIGHT, row, lx / 4 + 2, false);
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        //sm(i) * sp(j)
        if (config(site2) == config(site1)) {
          diag_corr[i - 1] = 0.0;
        } else {
          TenElemT psi_ex = tn.ReplaceOneSiteTrace(site2, (*split_index_tps)(site2)[1 - config(site2)], HORIZONTAL);
          diag_corr[i - 1] = ComplexConjugate(psi_ex * inv_psi);
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
        SiteIdx site1 = {row, col};
        SiteIdx site2 = {row + 1, col + 1};
        TenElemT diag_ss = EvaluateNNNSSForAFMHeisenbergModel(site1, site2,
                                                              config(site1),
                                                              config(site2),
                                                              LEFTUP_TO_RIGHTDOWN,
                                                              tn,
                                                              (*split_index_tps)(site1),
                                                              (*split_index_tps)(site2),
                                                              inv_psi);
        res.bond_energys_loc.push_back(diag_ss * j2_);
        e2 += diag_ss;
        site1 = {row + 1, col}; //left-down
        site2 = {row, col + 1}; //right-up
        diag_ss = EvaluateNNNSSForAFMHeisenbergModel(site1, site2,
                                                     config(site1),
                                                     config(site2),
                                                     LEFTDOWN_TO_RIGHTUP,
                                                     tn,
                                                     (*split_index_tps)(site1),
                                                     (*split_index_tps)(site2),
                                                     inv_psi);
        res.bond_energys_loc.push_back(diag_ss * j2_);
        e2 += diag_ss;
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
    psi_list.push_back(tps_sample->amplitude);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      TenElemT bond_energy = EvaluateNNBondEnergyForAFMHeisenbergModel(site1,
                                                                       site2,
                                                                       config(site1),
                                                                       config(site2),
                                                                       VERTICAL,
                                                                       tn,
                                                                       (*split_index_tps)(site1),
                                                                       (*split_index_tps)(site2),
                                                                       inv_psi);
      res.bond_energys_loc.push_back(bond_energy);
      e1 += bond_energy;
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  res.energy_loc = e1 + j2_ * e2;
  res.one_point_functions_loc.reserve(tn.rows() * tn.cols());
  for (auto &spin_config : config) {
    res.one_point_functions_loc.push_back((double) spin_config - 0.5);
  }
  return res;
}

template<typename TenElemT, typename QNT, bool calchols>
TenElemT SpinOneHalfJ1J2HeisenbergSquare::
CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                      TensorNetwork2D<TenElemT, QNT> &hole_res,
                      std::vector<TenElemT> &psi_list) {
  TenElemT e1(0), e2(0); // energy in J1 and J2 bond respectively
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = TPSWaveFunctionComponent<TenElemT, QNT>::trun_para;
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_list.push_back(tps_sample->amplitude);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL));
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        e1 += EvaluateNNBondEnergyForAFMHeisenbergModel(site1,
                                                        site2,
                                                        config(site1),
                                                        config(site2),
                                                        HORIZONTAL,
                                                        tn,
                                                        (*split_index_tps)(site1),
                                                        (*split_index_tps)(site2),
                                                        inv_psi);
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
        e2 += EvaluateNNNSSForAFMHeisenbergModel(site1, site2,
                                                 config(site1),
                                                 config(site2),
                                                 LEFTUP_TO_RIGHTDOWN,
                                                 tn,
                                                 (*split_index_tps)(site1),
                                                 (*split_index_tps)(site2),
                                                 inv_psi);

        site1 = {row + 1, col}; //left-down
        site2 = {row, col + 1}; //right-up
        e2 += EvaluateNNNSSForAFMHeisenbergModel(site1, site2,
                                                 config(site1),
                                                 config(site2),
                                                 LEFTDOWN_TO_RIGHTUP,
                                                 tn,
                                                 (*split_index_tps)(site1),
                                                 (*split_index_tps)(site2),
                                                 inv_psi);
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
    psi_list.push_back(tps_sample->amplitude);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      e1 += EvaluateNNBondEnergyForAFMHeisenbergModel(site1,
                                                      site2,
                                                      config(site1),
                                                      config(site2),
                                                      VERTICAL,
                                                      tn,
                                                      (*split_index_tps)(site1),
                                                      (*split_index_tps)(site2),
                                                      inv_psi);
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
}//qlpeps


#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_H
