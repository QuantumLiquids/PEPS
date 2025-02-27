/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2025-02-20
 *
 * Description: QuantumLiquids/PEPS project.
 * Model Energy Solver for in square lattice models, with only NN bond energy contributions.
 * Like the XXZ model, the spinless free fermion, the t-J and the Hubbard models.
 * The on-site energy only allow the diagonal terms like the Hubbard repulsion or chemical potential.
 *
 * To use the class, one should inherit the class and define the member function EvaluateBondEnergy
 * and EvaluateTotalOnsiteEnergy.
*/


#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate

namespace qlpeps {

///< CRTP Base class
template<class ExplicitlyModel>
class SquareNNModelEnergySolver : public ModelEnergySolver<SquareNNModelEnergySolver<ExplicitlyModel>> {
 public:
  using ModelEnergySolver<SquareNNModelEnergySolver<ExplicitlyModel>>::CalEnergyAndHoles;
  template<typename TenElemT, typename QNT, bool calchols = true>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  );
};

template<class ExplicitlyModel>
template<typename TenElemT, typename QNT, bool calchols>
TenElemT SquareNNModelEnergySolver<ExplicitlyModel>::
CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                      TensorNetwork2D<TenElemT, QNT> &hole_res,
                      std::vector<TenElemT> &psi_list) {
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    bool psi_added = false; // only valid for Ferimonic case
    TenElemT inv_psi; // only useful for Bosonic case
    if constexpr (!Index<QNT>::IsFermionic()) {
      auto psi = tn.Trace({row, 0}, HORIZONTAL);
      inv_psi = 1.0 / psi;
      psi_list.push_back(psi);
    }
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL)); // natural match to complex number wave-function case.
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        // Declaration on EvaluateBondEnergy is different for Fermion and Boson
        // Fermion should return the psi value, while boson should input inv_psi
        if constexpr (Index<QNT>::IsFermionic()) {
          std::optional<TenElemT> psi;
          energy += static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                             site2,
                                                                             (config(site1)),
                                                                             (config(site2)),
                                                                             HORIZONTAL,
                                                                             tn,
                                                                             (*split_index_tps)(site1),
                                                                             (*split_index_tps)(site2),
                                                                             psi);
          if (!psi_added && psi.has_value()) {
            psi_list.push_back(psi.value());
            psi_added = true;
          }
        } else {
          energy += static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                             site2,
                                                                             (config(site1)),
                                                                             (config(site2)),
                                                                             HORIZONTAL,
                                                                             tn,
                                                                             (*split_index_tps)(site1),
                                                                             (*split_index_tps)(site2),
                                                                             inv_psi);
        }
        if (col < tn.cols() - 1) { // need to calculate the hole
          tn.BTenMoveStep(RIGHT);
        }
      }
    }
    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    bool psi_added = false; // valid for fermion
    TenElemT inv_psi; //valid for boson
    if constexpr (!Index<QNT>::IsFermionic()) {
      auto psi = tn.Trace({0, col}, VERTICAL);
      inv_psi = 1.0 / psi;
      psi_list.push_back(psi);
    }
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if constexpr (Index<QNT>::IsFermionic()) {
        std::optional<TenElemT> psi;
        energy += static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                           site2,
                                                                           (config(site1)),
                                                                           (config(site2)),
                                                                           VERTICAL,
                                                                           tn,
                                                                           (*split_index_tps)(site1),
                                                                           (*split_index_tps)(site2),
                                                                           psi);
        if (!psi_added && psi.has_value()) {
          psi_list.push_back(psi.value());
          psi_added = true;
        }
      } else {
        energy += static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                           site2,
                                                                           (config(site1)),
                                                                           (config(site2)),
                                                                           VERTICAL,
                                                                           tn,
                                                                           (*split_index_tps)(site1),
                                                                           (*split_index_tps)(site2),
                                                                           inv_psi);
      }
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }

  energy += static_cast<ExplicitlyModel *>(this)->EvaluateTotalOnsiteEnergy(config);
  // Can be extended to adding the general diagonal energy contribution.
  return energy;
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H
