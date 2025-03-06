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

template<class ExplicitlyModel>
template<typename TenElemT, typename QNT, bool calchols>
TenElemT SquareNNModelEnergySolver<ExplicitlyModel>::
CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                      TensorNetwork2D<TenElemT, QNT> &hole_res,
                      std::vector<TenElemT> &psi_list) {
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

template<class ExplicitlyModel>
template<typename TenElemT, typename QNT, bool calchols>
void SquareNNModelEnergySolver<ExplicitlyModel>::
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

template<class ExplicitlyModel>
template<typename TenElemT, typename QNT, bool calchols>
void SquareNNModelEnergySolver<ExplicitlyModel>::
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
}

template<class ExplicitlyModel>
template<typename TenElemT, typename QNT>
void SquareNNModelEnergySolver<ExplicitlyModel>::
CalVerticalBondEnergyImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                          TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                          std::vector<TenElemT> &bond_energy_set,
                          std::vector<TenElemT> &psi_list) {
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
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
      TenElemT bond_energy;
      if constexpr (Index<QNT>::IsFermionic()) {
        std::optional<TenElemT> psi;
        bond_energy = static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
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
        bond_energy = static_cast<ExplicitlyModel *>(this)->EvaluateBondEnergy(site1,
                                                                               site2,
                                                                               (config(site1)),
                                                                               (config(site2)),
                                                                               VERTICAL,
                                                                               tn,
                                                                               (*split_index_tps)(site1),
                                                                               (*split_index_tps)(site2),
                                                                               inv_psi);
      }
      bond_energy_set.push_back(bond_energy);
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H
