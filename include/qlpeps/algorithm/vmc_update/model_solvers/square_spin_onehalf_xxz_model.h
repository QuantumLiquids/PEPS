/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-12
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 AFM Heisenberg model in square lattice
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_HEISENBERG_SQUARE_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/algorithm/vmc_update/model_solvers/square_nn_energy_solver.h"

namespace qlpeps {
using namespace qlten;

/**
 *  $$H = sum_<i,j> (J_z * S^z_i \cdot S^z_j + J_{xy} * ( S^x_i \cdot S^x_j  +  S^y_i \cdot S^y_j ))- h_{00} * S^z_{00}$$
 * $S^{\alpha}_i$ are spin-1/2 operator, h_{00} is the pinning field in corner.
 */
class SquareSpinOneHalfXXZModel : public SquareNNModelEnergySolver<SquareSpinOneHalfXXZModel>,
                                  public ModelMeasurementSolver<SquareSpinOneHalfXXZModel> {
 public:
  SquareSpinOneHalfXXZModel(void) : jz_(1), jxy_(1), pinning00_(0) {};
  SquareSpinOneHalfXXZModel(double jz, double jxy, double pinning_field_00)
      : jz_(jz), jxy_(jxy), pinning00_(pinning_field_00) {};
  using SquareNNModelEnergySolver::CalEnergyAndHoles;
  using SquareNNModelEnergySolver::CalEnergyAndHolesImpl;
  using ModelMeasurementSolver<SquareSpinOneHalfXXZModel>::operator();

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      const TenElemT inv_psi
  ) {
    if (config1 == config2) {
      return 0.25 * jz_;
    } else {
      TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                              split_index_tps_on_site1[config2],
                                              split_index_tps_on_site2[config1]);
      TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
      return (-0.25 * jz_ + ratio * 0.5 * jxy_);
    }
  }

  double Config2Sz(const size_t config) { return double(config) - 0.5; }

  double EvaluateTotalOnsiteEnergy(const Configuration &config) {
    return -pinning00_ * Config2Sz(config({0, 0}));
  }

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> SampleMeasureImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      std::vector<TenElemT> &psi_list
  ) {
    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
    const Configuration &sample_config = tps_sample->config;
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
    return this->SampleMeasureImpl(split_index_tps, sample_config, sample_tn, trunc_para, psi_list);
  }

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> SampleMeasureImpl(
      const SplitIndexTPS<TenElemT, QNT> *,
      const Configuration &,
      TensorNetwork2D<TenElemT, QNT> &,
      const BMPSTruncatePara &,
      std::vector<TenElemT> &
  );
 private:
  const double jz_;
  const double jxy_;
  const double pinning00_;
};

template<typename TenElemT, typename QNT>
ObservablesLocal<TenElemT> SquareSpinOneHalfXXZModel::SampleMeasureImpl(
    const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
    const Configuration &config,
    TensorNetwork2D<TenElemT, QNT> &tn,
    const BMPSTruncatePara &trunc_para,
    std::vector<TenElemT> &psi_list
) {
  ObservablesLocal<TenElemT> res;
  TenElemT energy(0);
  const double bond_energy_extremly_large = 1.0e5;
  const size_t lx = tn.cols(), ly = tn.rows();
  res.bond_energys_loc.reserve(lx * ly * 2);
  res.two_point_functions_loc.reserve(lx / 2 * 3);
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < ly; row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    // update the amplitude so that the error of ratio of amplitude can reduce by cancellation.
    auto psi = tn.Trace({row, 0}, HORIZONTAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
    for (size_t col = 0; col < lx - 1; col++) {
      //Calculate horizontal bond energy contribution
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row, col + 1};
      TenElemT horizontal_bond_energy = EvaluateBondEnergy(site1,
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
        double sz2 = double(config(site2)) - 0.5;
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
        res.two_point_functions_loc.insert(res.two_point_functions_loc.end(),
                                           off_diag_corr.begin(),
                                           off_diag_corr.end());
      } else {
        res.two_point_functions_loc.insert(res.two_point_functions_loc.end(),
                                           off_diag_corr.begin(),
                                           off_diag_corr.end());
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
    auto psi = tn.Trace({0, col}, VERTICAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      TenElemT vertical_bond_energy = EvaluateBondEnergy(site1,
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
    res.one_point_functions_loc.push_back(Config2Sz(spin_config));
  }
  return res;
}

}//qlpeps




#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
