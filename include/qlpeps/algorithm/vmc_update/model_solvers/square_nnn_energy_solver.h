/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2025-02-27
 *
 * Description: QuantumLiquids/PEPS project.
 * Model Energy Solver for in square lattice models, with NN & NNN bond energy contributions.
 * Like the J1-J2 XXZ model. (Only boson model support upto now)
 * The on-site energy only allow the diagonal terms like magnetic pinning field.
 *
 * To use the class, one should inherit the class and define the member function
 * EvaluateBondEnergy
 * and EvaluateTotalOnsiteEnergy.
*/


#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NNN_FERMION_ENERGY_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NNN_FERMION_ENERGY_SOLVER_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/algorithm/vmc_update/model_solvers/bond_traversal_mixin.h"

namespace qlpeps {

/**
 * SquareNNNModelEnergySolver is the base class to define generic
 * nearest-neighbor model energy solver on the square lattices,
 * work for both energy & gradient evaluation.
 * The CRTP technique is used to realize the Polymorphism.
 *
 * To be implemented by derived classes:
 * - EvaluateBondEnergy: Computes energy contribution for a bond between site1 and site2.
 *   Fermionic models should also optionally return the wavefunction amplitude (psi).
 * - EvaluateTotalOnsiteEnergy: Sums all on-site energy terms (e.g., chemical potential, Hubbard U).
 */
template<class ExplicitlyModel, bool has_nnn_interaction = true>
class SquareNNNModelEnergySolver : public ModelEnergySolver<SquareNNNModelEnergySolver<ExplicitlyModel,
                                                                                       has_nnn_interaction>> {
 public:
  using ModelEnergySolver<SquareNNNModelEnergySolver<ExplicitlyModel, has_nnn_interaction>>::CalEnergyAndHoles;
  template<typename TenElemT, typename QNT, bool calchols = true>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *,
      TPSWaveFunctionComponent<TenElemT, QNT> *,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  );

  template<typename TenElemT, typename QNT, bool calchols = true>
  void CalHorizontalBondEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *,
      TPSWaveFunctionComponent<TenElemT, QNT> *,
      TensorNetwork2D<TenElemT, QNT> &,
      std::vector<TenElemT> &, // gather the bond energy
      std::vector<TenElemT> &  // gather wave function amplitude
  );

  ///< assume the boundary MPS has obtained
  template<typename TenElemT, typename QNT, bool calchols = true>
  void CalHorizontalBondEnergyAndHolesSweepRowImpl(
      const size_t row,
      const SplitIndexTPS<TenElemT, QNT> *,
      TPSWaveFunctionComponent<TenElemT, QNT> *,
      TensorNetwork2D<TenElemT, QNT> &,
      std::vector<TenElemT> &,
      std::vector<TenElemT> &
  );

  template<typename TenElemT, typename QNT>
  void CalVerticalBondEnergyImpl(
      const SplitIndexTPS<TenElemT, QNT> *,
      TPSWaveFunctionComponent<TenElemT, QNT> *,
      std::vector<TenElemT> &, // gather the bond energy
      std::vector<TenElemT> &
  );
};

template<class ExplicitlyModel, bool has_nnn_interaction>
template<typename TenElemT, typename QNT, bool calchols>
TenElemT SquareNNNModelEnergySolver<ExplicitlyModel, has_nnn_interaction>::
CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                      TensorNetwork2D<TenElemT, QNT> &hole_res,
                      std::vector<TenElemT> &psi_list) {
  static_assert(
      std::is_base_of_v<SquareNNNModelEnergySolver<ExplicitlyModel, has_nnn_interaction>, ExplicitlyModel>,
      "ExplicitlyModel must inherit from SquareNNNModelEnergySolver<ExplicitlyModel, has_nnn_interaction>"
  );

  std::vector<TenElemT> bond_energy_set;
  bond_energy_set.reserve(2 * split_index_tps->size());
  this->template CalHorizontalBondEnergyAndHolesImpl<TenElemT, QNT, calchols>(
      split_index_tps, tps_sample, hole_res, bond_energy_set, psi_list
  );
  this->CalVerticalBondEnergyImpl(split_index_tps, tps_sample, bond_energy_set, psi_list);
  TenElemT bond_energy_total = std::reduce(bond_energy_set.begin(), bond_energy_set.end());

  auto energy_onsite = static_cast<ExplicitlyModel *>(this)->EvaluateTotalOnsiteEnergy(tps_sample->config);
  // Can be extended to adding the general diagonal energy contribution.
  return bond_energy_total + energy_onsite;
}

template<class ExplicitlyModel, bool has_nnn_interaction>
template<typename TenElemT, typename QNT, bool calchols>
void SquareNNNModelEnergySolver<ExplicitlyModel, has_nnn_interaction>::
CalHorizontalBondEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                                    TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                                    TensorNetwork2D<TenElemT, QNT> &hole_res,
                                    std::vector<TenElemT> &bond_energy_set,
                                    std::vector<TenElemT> &psi_list) {
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    this->template CalHorizontalBondEnergyAndHolesSweepRowImpl<TenElemT, QNT, calchols>(row,
                                                                                        split_index_tps,
                                                                                        tps_sample,
                                                                                        hole_res,
                                                                                        bond_energy_set,
                                                                                        psi_list);
    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }
}

template<class ExplicitlyModel, bool has_nnn_interaction>
template<typename TenElemT, typename QNT, bool calchols>
void SquareNNNModelEnergySolver<ExplicitlyModel, has_nnn_interaction>::
CalHorizontalBondEnergyAndHolesSweepRowImpl(const size_t row,
                                            const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                                            TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                                            TensorNetwork2D<TenElemT, QNT> &hole_res,
                                            std::vector<TenElemT> &bond_energy_set,
                                            std::vector<TenElemT> &psi_list) {
  auto &tn = tps_sample->tn;
  tn.InitBTen(LEFT, row);
  tn.GrowFullBTen(RIGHT, row, 1, true);
  bool psi_added = false; // only valid for Ferimonic case
  TenElemT inv_psi; // only useful for Bosonic case
  if constexpr (!Index<QNT>::IsFermionic()) {
    auto psi = tn.Trace({row, 0}, HORIZONTAL);
    if (psi == TenElemT(0)) [[unlikely]] {
      throw std::runtime_error("Wavefunction amplitude is near zero, causing division by zero.");
    }
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
      TenElemT bond_energy;
      if constexpr (Index<QNT>::IsFermionic()) {
        std::optional<TenElemT> psi;
        bond_energy = static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                               site2,
                                                                               (tps_sample->config(site1)),
                                                                               (tps_sample->config(site2)),
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
        bond_energy = static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                               site2,
                                                                               (tps_sample->config(site1)),
                                                                               (tps_sample->config(site2)),
                                                                               HORIZONTAL,
                                                                               tn,
                                                                               (*split_index_tps)(site1),
                                                                               (*split_index_tps)(site2),
                                                                               inv_psi);
      }
      bond_energy_set.push_back(bond_energy);
      tn.BTenMoveStep(RIGHT);
    }
  }
  if constexpr (has_nnn_interaction) {
    if (row < tn.rows() - 1) {
      // NNN energy contribution
      tn.InitBTen2(LEFT, row);
      tn.GrowFullBTen2(RIGHT, row, 2, true);
      for (size_t col = 0; col < tn.cols() - 1; col++) {
        TenElemT nnn_energy(0);
        SiteIdx site1 = {row, col};
        SiteIdx site2 = {row + 1, col + 1};
        // inv_psi may be updated accordingly to improve accuracy
        std::optional<TenElemT> psi; // only used for fermion model
        if constexpr (!Index<QNT>::IsFermionic()) { //boson code
          nnn_energy = static_cast<ExplicitlyModel *>(this)->EvaluateNNNEnergy(site1, site2,
                                                                               tps_sample->config(site1),
                                                                               tps_sample->config(site2),
                                                                               LEFTUP_TO_RIGHTDOWN,
                                                                               tn,
                                                                               (*split_index_tps)(site1),
                                                                               (*split_index_tps)(site2),
                                                                               inv_psi);
        } else {
          nnn_energy = static_cast<ExplicitlyModel *>(this)->EvaluateNNNEnergy(site1, site2,
                                                                               (tps_sample->config(site1)),
                                                                               (tps_sample->config(site2)),
                                                                               LEFTUP_TO_RIGHTDOWN,
                                                                               tn,
                                                                               (*split_index_tps)(site1),
                                                                               (*split_index_tps)(site2),
                                                                               psi);
        }

        site1 = {row + 1, col}; //left-down
        site2 = {row, col + 1}; //right-up
        if constexpr (!Index<QNT>::IsFermionic()) {
          nnn_energy += static_cast<ExplicitlyModel *>(this)->EvaluateNNNEnergy(site1, site2,
                                                                                tps_sample->config(site1),
                                                                                tps_sample->config(site2),
                                                                                LEFTDOWN_TO_RIGHTUP,
                                                                                tn,
                                                                                (*split_index_tps)(site1),
                                                                                (*split_index_tps)(site2),
                                                                                inv_psi);
        } else {
          nnn_energy += static_cast<ExplicitlyModel *>(this)->EvaluateNNNEnergy(site1, site2,
                                                                                (tps_sample->config(site1)),
                                                                                (tps_sample->config(site2)),
                                                                                LEFTDOWN_TO_RIGHTUP,
                                                                                tn,
                                                                                (*split_index_tps)(site1),
                                                                                (*split_index_tps)(site2),
                                                                                psi);
        }
        tn.BTen2MoveStep(RIGHT, row);
        bond_energy_set.push_back(nnn_energy);
      }
    }
  } // evaluate NNN energy.
}

template<class ExplicitlyModel, bool has_nnn_interaction>
template<typename TenElemT, typename QNT>
void SquareNNNModelEnergySolver<ExplicitlyModel, has_nnn_interaction>::
CalVerticalBondEnergyImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                          TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                          std::vector<TenElemT> &bond_energy_set,
                          std::vector<TenElemT> &psi_list) {
  const Configuration &config = tps_sample->config;
  BondTraversalMixin::TraverseVerticalBonds(
      tps_sample->tn,
      tps_sample->trun_para,
      [&, split_index_tps](const SiteIdx &site1,
                           const SiteIdx &site2,
                           const BondOrientation bond_orient,
                           const TenElemT &inv_psi) {
        TenElemT bond_energy;
        std::optional<TenElemT> fermion_psi;
        if constexpr (Index<QNT>::IsFermionic()) {
          bond_energy =
              static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                       site2,
                                                                       (config(site1)),
                                                                       (config(site2)),
                                                                       bond_orient,
                                                                       tps_sample->tn,
                                                                       (*split_index_tps)(site1),
                                                                       (*split_index_tps)(site2),
                                                                       fermion_psi);
        } else {
          bond_energy =
              static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                       site2,
                                                                       (config(site1)),
                                                                       (config(site2)),
                                                                       bond_orient,
                                                                       tps_sample->tn,
                                                                       (*split_index_tps)(site1),
                                                                       (*split_index_tps)(site2),
                                                                       inv_psi);
        }
        bond_energy_set.push_back(bond_energy);
      },
      psi_list
  );
}
}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NNN_FERMION_ENERGY_SOLVER_H
