/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2025-02-20
 *
 * Description: QuantumLiquids/PEPS project.
 * Model Energy Solver for the fermion models in square lattice, with only NN bond energy contributions.
 * Like the spinless free fermion, the t-J and the Hubbard models.
 * The on-site energy only allow the diagonal terms like the Hubbard repulsion or chemical potential.
 *
 * To use the class, one should inherit the class and define the member function EvaluateBondEnergy
 * and
*/


#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_ENERGY_SOLVER_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate

namespace qlpeps {

///< CRTP
template<class ExplicitlyModel>
class SquareNNFermionModelEnergySolver : public ModelEnergySolver<SquareNNFermionModelEnergySolver<ExplicitlyModel>> {
 public:
  using ModelEnergySolver<SquareNNFermionModelEnergySolver<ExplicitlyModel>>::CalEnergyAndHoles;
  template<typename TenElemT, typename QNT, bool calchols = true>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<double> &psi_list
  );
};

template<class ExplicitlyModel>
template<typename TenElemT, typename QNT, bool calchols>
TenElemT SquareNNFermionModelEnergySolver<ExplicitlyModel>::
CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                      TensorNetwork2D<TenElemT, QNT> &hole_res,
                      std::vector<double> &psi_list) {
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = TPSWaveFunctionComponent<TenElemT, QNT>::trun_para;
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    bool psi_added = false;
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL)); // natural match to complex number wave-function case.
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
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
        if (col < tn.cols() - 2) {
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
    bool psi_added = false;
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
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
