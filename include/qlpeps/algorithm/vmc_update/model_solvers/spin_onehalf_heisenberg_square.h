/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-12
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 AFM Heisenberg model in square lattice
*/

#ifndef QLPEPS_VMC_PEPS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
#define QLPEPS_VMC_PEPS_SPIN_ONEHALF_HEISENBERG_SQUARE_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      //ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               //ComplexConjugate

namespace qlpeps {
using namespace qlten;

template<typename TenElemT, typename QNT>
class SpinOneHalfHeisenbergSquare : public ModelEnergySolver<TenElemT, QNT>, ModelMeasurementSolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  using ModelEnergySolver<TenElemT, QNT>::ModelEnergySolver;

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
};

template<typename TenElemT, typename QNT>
TenElemT EvaluateNNBondEnergyForAFMHeisenbergModel(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    const TenElemT inv_psi
) {
  if (config1 == config2) {
    return 0.25;
  } else {
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[config2],
                                            split_index_tps_on_site2[config1]);
    TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
    return (-0.25 + ratio * 0.5);
  }
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType, bool calchols>
TenElemT SpinOneHalfHeisenbergSquare<TenElemT, QNT>::CalEnergyAndHoles(const SITPS *split_index_tps,
                                                                       WaveFunctionComponentType *tps_sample,
                                                                       TensorNetwork2D<TenElemT, QNT> &hole_res) {
  const double bond_energy_extremly_large = 1.0e5;
  TenElemT energy(0);
  bool has_unreasonable_bond_energy(false);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = WaveFunctionComponentType::trun_para.value();
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  std::vector<TenElemT> psi_gather;
  psi_gather.reserve(tn.rows() + tn.cols());
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    // update the amplitude so that the error of ratio of amplitude can reduce by cancellation.
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_gather.push_back(tps_sample->amplitude);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL)); // natural match to complex number wave-function case.
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        TenElemT bond_energy = EvaluateNNBondEnergyForAFMHeisenbergModel(site1,
                                                                         site2,
                                                                         config(site1),
                                                                         config(site2),
                                                                         HORIZONTAL,
                                                                         tn,
                                                                         (*split_index_tps)(site1),
                                                                         (*split_index_tps)(site2),
                                                                         inv_psi);
        if (std::abs(bond_energy) > bond_energy_extremly_large) {
          std::cout << "Warning [Unreasonable bond energy]: "
                    << "Site : (" << row << ", " << col << ") "
                    << "Bond Orient :" << " H, "
                    << ", psi_0 : " << std::scientific << tps_sample->amplitude
                    << ", bond energy : " << std::scientific << bond_energy
                    << std::endl;
          // set the bond energy = 0.0
          has_unreasonable_bond_energy = true;
        } else {
          energy += bond_energy;
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
    psi_gather.push_back(tps_sample->amplitude);
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
      energy += bond_energy;
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  WaveFunctionAmplitudeConsistencyCheck(psi_gather, 0.01);
  return energy;
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType>
ObservablesLocal<TenElemT> SpinOneHalfHeisenbergSquare<TenElemT, QNT>::SampleMeasure(
    const SITPS *split_index_tps,
    WaveFunctionComponentType *tps_sample
) {
  ObservablesLocal<TenElemT> res;
  TenElemT energy(0);
  const double bond_energy_extremly_large = 1.0e5;
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const size_t lx = tn.cols(), ly = tn.rows();
  res.bond_energys_loc.reserve(lx * ly * 2);
  res.two_point_functions_loc.reserve(lx / 2 * 3);
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = WaveFunctionComponentType::trun_para.value();
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  std::vector<TenElemT> psi_gather;
  psi_gather.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < ly; row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    // update the amplitude so that the error of ratio of amplitude can reduce by cancellation.
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_gather.push_back(tps_sample->amplitude);
    for (size_t col = 0; col < lx - 1; col++) {
      //Calculate horizontal bond energy contribution
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row, col + 1};
      TenElemT horizontal_bond_energy = EvaluateNNBondEnergyForAFMHeisenbergModel(site1,
                                                                                  site2,
                                                                                  config(site1),
                                                                                  config(site2),
                                                                                  HORIZONTAL,
                                                                                  tn,
                                                                                  (*split_index_tps)(site1),
                                                                                  (*split_index_tps)(site2),
                                                                                  inv_psi);
      energy += horizontal_bond_energy;
      res.bond_energys_loc.push_back(horizontal_bond_energy);
      tn.BTenMoveStep(RIGHT);
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

      std::vector<TenElemT> off_diag_corr(lx / 2);// sp(i) * sm(j) or sm(i) * sp(j), the valid channel
      tn(site1) = (*split_index_tps)(site1)[1 - config(site1)]; //temporally change
      tn.TruncateBTen(LEFT, lx / 4 + 1); // may be above two lines should be summarized as an API
      tn.GrowBTenStep(LEFT);//left boundary tensor just across Lx/4
      tn.GrowFullBTen(RIGHT, row, lx / 4 + 2, false); //environment for Lx/4 + 1 site
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        //sm(i) * sp(j)
        if (config(site2) == config(site1)) {
          off_diag_corr[i - 1] = 0.0;
        } else {
          TenElemT psi_ex = tn.ReplaceOneSiteTrace(site2, (*split_index_tps)(site2)[1 - config(site2)], HORIZONTAL);
          off_diag_corr[i - 1] = (ComplexConjugate(psi_ex * inv_psi));
        }
        tn.BTenMoveStep(RIGHT);
      }
      tn(site1) = (*split_index_tps)(site1)[config(site1)]; // change back

      if (config(site1) == 1) {
        for (size_t i = 1; i <= lx / 2; i++) {  //sp(i) * sm(j) = 0
          res.two_point_functions_loc.push_back(0.0);
        }
        res.two_point_functions_loc.insert(res.two_point_functions_loc.end(), off_diag_corr.begin(), off_diag_corr.end());
      } else {
        res.two_point_functions_loc.insert(res.two_point_functions_loc.end(), off_diag_corr.begin(), off_diag_corr.end());
        for (size_t i = 1; i <= lx / 2; i++) {  //sm(i) * sp(j) = 0
          res.two_point_functions_loc.push_back(0.0);
        }
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
    psi_gather.push_back(tps_sample->amplitude);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      TenElemT vertical_bond_energy = EvaluateNNBondEnergyForAFMHeisenbergModel(site1,
                                                                                site2,
                                                                                config(site1),
                                                                                config(site2),
                                                                                VERTICAL,
                                                                                tn,
                                                                                (*split_index_tps)(site1),
                                                                                (*split_index_tps)(site2),
                                                                                inv_psi);
      energy += vertical_bond_energy;
      res.bond_energys_loc.push_back(vertical_bond_energy);
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  res.energy_loc = energy;
  res.one_point_functions_loc.reserve(tn.rows() * tn.cols());
  for (auto &spin_config : config) {
    res.one_point_functions_loc.push_back((double) spin_config - 0.5);
  }
  WaveFunctionAmplitudeConsistencyCheck(psi_gather, 0.01);
  return res;
}

}//qlpeps




#endif //QLPEPS_VMC_PEPS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
