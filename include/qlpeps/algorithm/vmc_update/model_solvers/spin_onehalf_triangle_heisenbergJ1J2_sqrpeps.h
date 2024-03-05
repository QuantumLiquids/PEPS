/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-12-30
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 Triangle Heisenberg J1-J2 model on square PEPS
*/

#ifndef QLPEPS_QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H
#define QLPEPS_QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"    //ModelEnergySolver

namespace qlpeps {
using namespace qlten;

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
  std::vector<TenElemT> CalculateSzAll2AllCorrelation_(const qlpeps::Configuration &config) const;
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
  const BMPSTruncatePara &trunc_para = SquareTPSSampleNNExchange<TenElemT, QNT>::trun_para;
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  std::vector<TenElemT> psi_gather;
  psi_gather.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_gather.push_back(tps_sample->amplitude);
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
    psi_gather.push_back(tps_sample->amplitude);
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
  WaveFunctionAmplitudeConsistencyCheck(psi_gather, 0.03);
  return e1 + j2_ * e2;
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType>
ObservablesLocal<TenElemT> SpinOneHalfTriJ1J2HeisenbergSqrPEPS<TenElemT, QNT>::SampleMeasure(
    const SITPS *split_index_tps,
    WaveFunctionComponentType *tps_sample
) {
  ObservablesLocal<TenElemT> res;

  TenElemT e1(0), e2(0); // energy in J1 and J2 bond respectively
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const size_t lx = tn.cols();
  res.bond_energys_loc.reserve(tn.rows() * lx * 6);
  res.two_point_functions_loc.reserve(tn.cols() / 2 * 3);
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = SquareTPSSampleNNExchange<TenElemT, QNT>::trun_para;
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
        TenElemT horizontal_bond_energy;
        if (config(site1) == config(site2)) {
          horizontal_bond_energy = 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, HORIZONTAL,
                                                  (*split_index_tps)(site1)[config(site2)],
                                                  (*split_index_tps)(site2)[config(site1)]);
          horizontal_bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
        }
        e1 += horizontal_bond_energy;
        res.bond_energys_loc.push_back(horizontal_bond_energy);
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
          diag_corr[i - 1] = (psi_ex * inv_psi);
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
      tn.InitBTen2(LEFT, row);
      tn.GrowFullBTen2(RIGHT, row, 2, true);

      for (size_t col = 0; col < tn.cols() - 1; col++) {
        //Calculate diagonal J1 energy contribution
        SiteIdx site1 = {row + 1, col}; //left-down
        SiteIdx site2 = {row, col + 1}; //right-up
        TenElemT diagonal_nn_bond_energy;
        if (config(site1) == config(site2)) {
          diagonal_nn_bond_energy = 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTDOWN_TO_RIGHTUP,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          diagonal_nn_bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
        }
        e1 += diagonal_nn_bond_energy;
        res.bond_energys_loc.push_back(diagonal_nn_bond_energy);
        //Calculate J2 contribution
        site1 = {row, col}; //left-top
        site2 = {row + 1, col + 1}; //right-bottom
        TenElemT nnn_bond_energy;
        if (config(site1) == config(site2)) {
          nnn_bond_energy = 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTUP_TO_RIGHTDOWN,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          nnn_bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
        }
        e2 += nnn_bond_energy;
        res.bond_energys_loc.push_back(nnn_bond_energy * j2_);

        if (col < tn.cols() - 2) {
          SiteIdx site1 = {row + 1, col}; //left-bottom
          SiteIdx site2 = {row, col + 2}; //right-top
          TenElemT nnn_bond_energy;
          if (config(site1) == config(site2)) {
            nnn_bond_energy = 0.25;
          } else {
            TenElemT psi_ex = tn.ReplaceSqrt5DistTwoSiteTrace({row, col},
                                                              LEFTDOWN_TO_RIGHTUP,
                                                              HORIZONTAL,
                                                              (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                              (*split_index_tps)(site2)[config(site1)]);
            nnn_bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
          }
          e2 += nnn_bond_energy;
          res.bond_energys_loc.push_back(nnn_bond_energy * j2_);
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
      TenElemT vertical_nn_bond_energy;
      if (config(site1) == config(site2)) {
        vertical_nn_bond_energy = 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        vertical_nn_bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
      }
      e1 += vertical_nn_bond_energy;
      res.bond_energys_loc.push_back(vertical_nn_bond_energy);
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
        TenElemT nnn_bond_energy;
        if (config(site1) == config(site2)) {
          nnn_bond_energy = 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceSqrt5DistTwoSiteTrace({row, col},
                                                            LEFTDOWN_TO_RIGHTUP,
                                                            VERTICAL,
                                                            (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                            (*split_index_tps)(site2)[config(site1)]);
          nnn_bond_energy = (-0.25 + psi_ex * inv_psi * 0.5);
        }
        e2 += nnn_bond_energy;
        res.bond_energys_loc.push_back(nnn_bond_energy * j2_);
        if ((int) row < (int) tn.rows() - 3) {
          tn.BTen2MoveStep(DOWN, col);
        }
      }
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  res.energy_loc = e1 + e2 * j2_;
  res.one_point_functions_loc.reserve(tn.rows() * lx);
  for (auto &spin_config : config) {
    res.one_point_functions_loc.push_back((double) spin_config - 0.5);
  }

  std::vector<TenElemT> all2all_sz_corr = CalculateSzAll2AllCorrelation_(config);
  res.two_point_functions_loc.insert(res.two_point_functions_loc.end(), all2all_sz_corr.begin(), all2all_sz_corr.end());

  return res;
}

template<typename TenElemT, typename QNT>
std::vector<TenElemT> SpinOneHalfTriJ1J2HeisenbergSqrPEPS<TenElemT,
                                                          QNT>::CalculateSzAll2AllCorrelation_(const qlpeps::Configuration &config) const {
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
#endif //QLPEPS_QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H
