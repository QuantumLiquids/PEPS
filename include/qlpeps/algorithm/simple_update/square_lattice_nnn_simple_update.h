// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-09
*
* Description: GraceQ/VMC-SquareLatticePEPS project.
*              Simple Update for nearest-neighbor and next-nearest-neighbor interaction models in square lattice
*/

#ifndef GRACEQ_VMC_PEPS_SQUARE_LATTICE_NNN_SIMPLE_UPDATE_H
#define GRACEQ_VMC_PEPS_SQUARE_LATTICE_NNN_SIMPLE_UPDATE_H

#include "qlpeps/algorithm/simple_update/simple_update.h"

namespace qlpeps {

using namespace qlten;

template<typename TenElemT, typename QNT>
class SquareLatticeNNNSimpleUpdateExecutor : public SimpleUpdateExecutor<TenElemT, QNT> {
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
 public:
  SquareLatticeNNNSimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                       const PEPST &peps_initial,
                                       const Tensor &ham_nn,
                                       const Tensor &ham_tri) :
      SimpleUpdateExecutor<TenElemT, QNT>(update_para, peps_initial), \
          ham_nn_(ham_nn), ham_tri_(ham_tri) {}

 private:
  void SetEvolveGate_(void) override {
    evolve_gate_nn_ = TaylorExpMatrix(this->update_para.tau, ham_nn_);
    evolve_gate_nn_half_ = TaylorExpMatrix(this->update_para.tau / 2, ham_nn_);
    evolve_gate_tri_ = TaylorExpMatrix(this->update_para.tau, ham_tri_);
  }

  double SimpleUpdateSweep_(void) override;

  Tensor ham_nn_;
  Tensor ham_tri_;  // A-B-C 3-site, A-B and B-C are NN connect, A-C is NNN connect
  // A-B or B-C has half of J1 interaction, A-C has a J2 interaction
  Tensor evolve_gate_nn_;
  Tensor evolve_gate_nn_half_;  // half of J1 interaction
  Tensor evolve_gate_tri_;
};

template<typename TenElemT, typename QNT>
double SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateSweep_(void) {
  Timer simple_update_sweep_timer("simple_update_sweep");
  SimpleUpdateTruncatePara para(this->update_para.Dmin, this->update_para.Dmax, this->update_para.Trunc_err);
  double norm = 1.0;
  double e0 = 0.0;

  for (size_t col = 1; col < this->lx_; col++) {  //first row
    norm = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {0, col - 1}, HORIZONTAL, para);
    e0 += -std::log(norm) / this->update_para.tau;
  }

  for (size_t row = 0; row < this->ly_ - 1; row++) {  // first and last column
    norm = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_half_, {row, 0}, VERTICAL, para);
    e0 += -std::log(norm) / this->update_para.tau;
    norm = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_half_, {row, this->lx_ - 1}, VERTICAL, para);
    e0 += -std::log(norm) / this->update_para.tau;
  }

  for (size_t col = 1; col < this->lx_; col++) {
    for (size_t row = 0; row < this->ly_ - 1; row++) {
      norm = this->peps_.LowerRightTriangleProject(evolve_gate_tri_, {row, col}, para);
      e0 += -std::log(norm) / this->update_para.tau;

      norm = this->peps_.LowerLeftTriangleProject(evolve_gate_tri_, {row, col - 1}, para);
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

  return norm;
}
}

#endif //GRACEQ_VMC_PEPS_SQUARE_LATTICE_NNN_SIMPLE_UPDATE_H
