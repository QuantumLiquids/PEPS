/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-20
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 J1-J2 Heisenberg model in square lattice
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_OBC_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_OBC_H

#include "square_spin_onehalf_xxz_obc.h"          // SquareSpinOneHalfXXZModelMixIn
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_model_measurement_solver.h"

namespace qlpeps {
using namespace qlten;

/**
 * J_1-J_2 XXZ Model on square lattice
 * 
 * Hamiltonian:
 * $$H = \sum_{\langle i,j \rangle} (J_{z1} S^z_i S^z_j + J_{xy1} (S^x_i S^x_j + S^y_i S^y_j))$$
 * $$   + \sum_{\langle\langle i,j \rangle\rangle} (J_{z2} S^z_i S^z_j + J_{xy2} (S^x_i S^x_j + S^y_i S^y_j)) - h_{00} S^z_{00}$$
 * 
 * where:
 * - First sum over nearest-neighbor (NN) bonds <i,j>
 * - Second sum over next-nearest-neighbor (NNN) bonds <<i,j>>
 * - J_{z1}, J_{xy1}: NN coupling constants for Ising and XY interactions
 * - J_{z2}, J_{xy2}: NNN coupling constants for Ising and XY interactions  
 * - h_{00}: pinning field at corner site (0,0)
 * - For J_z = 0: reduces to planar XY limit
 * - Supports competing interactions and magnetic frustration effects
 */
class SquareSpinOneHalfJ1J2XXZModelOBC : public SquareNNNModelEnergySolver<SquareSpinOneHalfJ1J2XXZModelOBC>,
                                      public SquareNNNModelMeasurementSolver<SquareSpinOneHalfJ1J2XXZModelOBC>,
                                      public SquareSpinOneHalfXXZModelMixIn {
 public:
  using SquareNNNModelMeasurementSolver<SquareSpinOneHalfJ1J2XXZModelOBC>::EvaluateObservables;
  using SquareNNNModelMeasurementSolver<SquareSpinOneHalfJ1J2XXZModelOBC>::DescribeObservables;

  SquareSpinOneHalfJ1J2XXZModelOBC(void) = delete;

  ///< J1-J2 Heisenberg model
  SquareSpinOneHalfJ1J2XXZModelOBC(double j2) :
      SquareSpinOneHalfXXZModelMixIn(1, 1, j2, j2, 0) {}
  ///< Generic construction
  SquareSpinOneHalfJ1J2XXZModelOBC(double jz, double jxy, double jz2, double jxy2, double pinning_field00) :
      SquareSpinOneHalfXXZModelMixIn(jz, jxy, jz2, jxy2, pinning_field00) {}

  // Unified row-hook (registry-based) for off-diagonal observables along a row.
  // This is invoked by SquareNNNModelMeasurementSolver via BondTraversalMixin.
  template<typename TenElemT, typename QNT>
  void EvaluateOffDiagOrderInRow(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                                 TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                                 const size_t row,
                                 const TenElemT inv_psi,
                                 ObservableMap<TenElemT> &out) const {
    auto &tn = tps_sample->tn;
    const auto &config = tps_sample->config;
    const size_t ly = tn.rows();
    const size_t lx = tn.cols();
    if (ly == 0 || lx == 0) {
      return;
    }
    if (row != ly / 2) {
      return;
    }

    auto &contractor = tps_sample->contractor;
    std::vector<TenElemT> diag_corr;
    diag_corr.reserve(lx / 2);
    MeasureSpinOneHalfOffDiagOrderInRow(split_index_tps, tn, contractor, diag_corr, inv_psi, config, row);

    std::vector<TenElemT> SmSp_row = diag_corr;
    std::vector<TenElemT> SpSm_row(diag_corr.size(), TenElemT(0));
    const SiteIdx site1{row, lx / 4};
    if (config(site1) == 0) {
      SpSm_row = diag_corr;
      std::fill(SmSp_row.begin(), SmSp_row.end(), TenElemT(0));
    }
    if (!SmSp_row.empty()) out["SmSp_row"] = std::move(SmSp_row);
    if (!SpSm_row.empty()) out["SpSm_row"] = std::move(SpSm_row);
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    auto base = this->SquareNNNModelMeasurementSolver<SquareSpinOneHalfJ1J2XXZModelOBC>::DescribeObservables(ly, lx);
    for (auto &meta : base) {
      if (meta.key == "spin_z" || meta.key == "charge") {
        meta.shape = {ly, lx};
        meta.index_labels = {"y", "x"};
      }
      if (meta.key == "bond_energy_h") {
        meta.shape = {ly, (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_v") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), lx};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_dr") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_ur") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
    }
    base.push_back({"SmSp_row", "Row Sm(i)Sp(j) along middle row (flat)", {lx / 2}, {"segment"}});
    base.push_back({"SpSm_row", "Row Sp(i)Sm(j) along middle row (flat)", {lx / 2}, {"segment"}});
    return base;
  }
};//SquareSpinOneHalfJ1J2XXZModelOBC

using SquareSpinOneHalfJ1J2XXZModel [[deprecated("Use SquareSpinOneHalfJ1J2XXZModelOBC instead.")]] =
    SquareSpinOneHalfJ1J2XXZModelOBC;

}//qlpeps


#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_OBC_H
