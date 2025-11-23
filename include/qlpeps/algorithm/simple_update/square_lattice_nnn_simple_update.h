// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-09
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project.
*              Simple Update for nearest-neighbor and next-nearest-neighbor interaction models in square lattice
*/

#ifndef QLPEPS_VMC_PEPS_SQUARE_LATTICE_NNN_SIMPLE_UPDATE_H
#define QLPEPS_VMC_PEPS_SQUARE_LATTICE_NNN_SIMPLE_UPDATE_H

#include "qlpeps/algorithm/simple_update/simple_update.h"

namespace qlpeps {

using namespace qlten;

/**
 * Simple update for square lattice model with uniform NN and NNN interaction.
 * TODO: introduce on-site terms
 */
template<typename TenElemT, typename QNT>
class SquareLatticeNNNSimpleUpdateExecutor : public SimpleUpdateExecutor<TenElemT, QNT> {
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
  using RealT = typename SimpleUpdateExecutor<TenElemT, QNT>::RealT;

 public:
  /**
   *
   * @param ham_nn Hamiltonian term on NN bond, rank-4 tensor
   * @param ham_nnn Hamiltonian term on NNN link, rank-4 tensor
   */
  SquareLatticeNNNSimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                       const PEPST &peps_initial,
                                       const Tensor &ham_nn,
                                       const Tensor &ham_nnn) :
      SimpleUpdateExecutor<TenElemT, QNT>(update_para, peps_initial), \
                                           ham_nn_(ham_nn) {
    /*      1             3
            |             |
            v             v
            |---ham_nnn---|
            v             v
            |             |
            0             2
     */
    Tensor id = Eye<TenElemT>(ham_nn.GetIndex(0));
    if (Tensor::IsFermionic() && id.GetIndex(0).GetDir() == OUT) { //should be
      id.ActFermionPOps();
    }
    Tensor extend_ham_nnn, extend_ham_nn_left, extend_ham_nn_right;
    Contract(&ham_nnn, {}, &id, {}, &extend_ham_nnn);
    extend_ham_nnn.Transpose({0, 1, 4, 5, 2, 3});
    Contract(&ham_nn, {}, &id, {}, &extend_ham_nn_left);
    Contract(&id, {}, &ham_nn, {}, &extend_ham_nn_right);
    ham_tri_ = extend_ham_nnn + RealT(0.5) * extend_ham_nn_right + RealT(0.5) * extend_ham_nn_left;
  }

 private:
  void SetEvolveGate_(void) override {
    evolve_gate_nn_ = TaylorExpMatrix(RealT(this->update_para.tau), ham_nn_);
    evolve_gate_nn_half_ = TaylorExpMatrix(RealT(this->update_para.tau) / 2, ham_nn_);
    evolve_gate_tri_ = TaylorExpMatrix(RealT(this->update_para.tau), ham_tri_);
  }

  RealT SimpleUpdateSweep_(void) override;

  Tensor ham_nn_;
  Tensor ham_tri_;// Hamiltonian term on A-B-C 3-site,
    //with A-B and B-C are NN bonds, A-C is NNN bond.
  // ham_tri_ term contains half of J1 interaction on bond A-B and B-C,
  // while contain the whole of J2 interaction on bond A-C
  Tensor evolve_gate_nn_;
  Tensor evolve_gate_nn_half_;  // half of J1 interaction
  Tensor evolve_gate_tri_;
};

template<typename TenElemT, typename QNT>
typename SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>::RealT SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateSweep_(void) {
  Timer simple_update_sweep_timer("simple_update_sweep");
  SimpleUpdateTruncatePara para(this->update_para.Dmin, this->update_para.Dmax, this->update_para.Trunc_err);
  RealT e0 = 0.0;

  for (size_t col = 1; col < this->lx_; col++) {  //first row
    ProjectionRes<TenElemT>
        proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {0, col - 1}, HORIZONTAL, para);
    e0 += -std::log(proj_res.norm) / this->update_para.tau;
  }

  for (size_t row = 0; row < this->ly_ - 1; row++) {  // first and last column
    ProjectionRes<TenElemT>
        proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_half_, {row, 0}, VERTICAL, para);
    e0 += -std::log(proj_res.norm) / this->update_para.tau;
    proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_half_, {row, this->lx_ - 1}, VERTICAL, para);
    e0 += -std::log(proj_res.norm) / this->update_para.tau;
  }

  for (size_t col = 1; col < this->lx_; col++) {
    for (size_t row = 0; row < this->ly_ - 1; row++) {
      ProjectionRes<TenElemT>
          proj_res = this->peps_.LowerRightTriangleProject(evolve_gate_tri_, {row, col}, para);
      e0 += -std::log(proj_res.norm) / this->update_para.tau;

      RealT norm = this->peps_.LowerLeftTriangleProject(evolve_gate_tri_, {row, col - 1}, para);
      e0 += -std::log(norm) / this->update_para.tau;
    }
  }

  double sweep_time = simple_update_sweep_timer.Elapsed();
  auto [dmin, dmax] = this->peps_.GetMinMaxBondDim();
  std::cout << "Estimated E0 =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << e0
            << " Dmin/Dmax = " << std::setw(2) << std::right << dmin << "/" << std::setw(2) << std::left << dmax
            << " SweepTime = " << std::setw(8) << sweep_time
            << std::endl;

  return e0;
}
}

#endif //QLPEPS_VMC_PEPS_SQUARE_LATTICE_NNN_SIMPLE_UPDATE_H
