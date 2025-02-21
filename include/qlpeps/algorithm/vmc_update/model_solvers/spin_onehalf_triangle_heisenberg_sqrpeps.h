/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-28
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 Triangle Heisenberg model on square PEPS
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate

namespace qlpeps {
using namespace qlten;

class SpinOneHalfTriHeisenbergSqrPEPS : public ModelEnergySolver<SpinOneHalfTriHeisenbergSqrPEPS>,
                                        public ModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS> {
 public:
  SpinOneHalfTriHeisenbergSqrPEPS(void) = default;

  using ModelEnergySolver<SpinOneHalfTriHeisenbergSqrPEPS>::CalEnergyAndHoles;
  using ModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS>::operator();

  template<typename TenElemT, typename QNT, bool calchols>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  ) {
    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
    const Configuration &sample_config = tps_sample->config;
    const BMPSTruncatePara &trunc_para = TPSWaveFunctionComponent<TenElemT, QNT>::trun_para;
    return this->template CalEnergyAndHolesImpl<TenElemT, QNT, calchols>(split_index_tps,
                                                                         sample_config,
                                                                         sample_tn,
                                                                         trunc_para,
                                                                         hole_res,
                                                                         psi_list);
  }

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> SampleMeasureImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      std::vector<TenElemT> &psi_list
  ) {
    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
    const Configuration &sample_config = tps_sample->config;
    const BMPSTruncatePara &trunc_para = TPSWaveFunctionComponent<TenElemT, QNT>::trun_para;
    return this->SampleMeasureImpl(split_index_tps, sample_config, sample_tn, trunc_para, psi_list);
  }

  template<typename TenElemT, typename QNT, bool calchols>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      const Configuration &config,
      TensorNetwork2D<TenElemT, QNT> &tn,
      const BMPSTruncatePara &trunc_para,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  );

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> SampleMeasureImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      const Configuration &config,
      TensorNetwork2D<TenElemT, QNT> &tn,
      const BMPSTruncatePara &trunc_para,
      std::vector<TenElemT> &psi_list
  );

 private:
  template<typename TenElemT>
  std::vector<TenElemT> CalculateSzAll2AllCorrelation_(
      const Configuration &config
  ) const;
};

template<typename TenElemT, typename QNT, bool calchols>
TenElemT SpinOneHalfTriHeisenbergSqrPEPS::
CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                      const qlpeps::Configuration &config,
                      TensorNetwork2D<TenElemT, QNT> &tn,
                      const qlpeps::BMPSTruncatePara &trunc_para,
                      TensorNetwork2D<TenElemT, QNT> &hole_res,
                      std::vector<TenElemT> &psi_list) {
  TenElemT e(0);
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    auto psi = tn.Trace({row, 0}, HORIZONTAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
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
          e += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
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
          e += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
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
    auto psi = tn.Trace({0, col}, VERTICAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        e += 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        e += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
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
ObservablesLocal<TenElemT> SpinOneHalfTriHeisenbergSqrPEPS::
SampleMeasureImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                  const qlpeps::Configuration &config,
                  TensorNetwork2D<TenElemT, QNT> &tn,
                  const qlpeps::BMPSTruncatePara &trunc_para,
                  std::vector<TenElemT> &psi_list) {
  ObservablesLocal<TenElemT> res;

  TenElemT energy_loc(0);
  const size_t lx = tn.cols(), ly = tn.rows(), N = tn.size();
  res.bond_energys_loc.reserve(N * 3);
  res.two_point_functions_loc.reserve(lx / 2 * 3 + N * N);
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < ly; row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    auto psi = tn.Trace({row, 0}, HORIZONTAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
    for (size_t col = 0; col < lx; col++) {
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
          auto bond_energy = (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
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
          diag_corr[i - 1] = 0.0;
        } else {
          TenElemT psi_ex = tn.ReplaceOneSiteTrace(site2, (*split_index_tps)(site2)[1 - config(site2)], HORIZONTAL);
          diag_corr[i - 1] = (ComplexConjugate(psi_ex * inv_psi));
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
          auto bond_energy = (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
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
    auto psi = tn.Trace({0, col}, VERTICAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
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
        auto bond_energy = (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
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
  std::vector<TenElemT> all2all_sz_corr = CalculateSzAll2AllCorrelation_<TenElemT>(config);
  res.two_point_functions_loc.insert(res.two_point_functions_loc.end(), all2all_sz_corr.begin(), all2all_sz_corr.end());

  return res;
}

template<typename TenElemT>
std::vector<TenElemT> SpinOneHalfTriHeisenbergSqrPEPS::CalculateSzAll2AllCorrelation_(const qlpeps::Configuration &config) const {
  std::vector<TenElemT> res;
  size_t N = config.size();
  res.reserve(N * N);
  for (auto &c1 : config) {
    for (auto &c2 : config) {
      if (c1 == c2) {
        res.push_back(0.25);
      } else {
        res.push_back(-0.25);
      }
    }
  }
  return res;
}
}//qlpeps



#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H
