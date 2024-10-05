// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-28
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project.
*              Simple update implementation for
 *             uniform nearest-neighbor interaction models in square lattice
*/

#ifndef QLPEPS_VMC_PEPS_SQUARE_LATTICE_NN_SIMPLE_UPDATE_H
#define QLPEPS_VMC_PEPS_SQUARE_LATTICE_NN_SIMPLE_UPDATE_H

#include "qlpeps/algorithm/simple_update/simple_update.h"

namespace qlpeps {

using namespace qlten;

template<typename TenElemT, typename QNT>
class SquareLatticeNNSimpleUpdateExecutor : public SimpleUpdateExecutor<TenElemT, QNT> {
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
 public:
  SquareLatticeNNSimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                      const PEPST &peps_initial,
                                      const Tensor &ham_nn) :
      SimpleUpdateExecutor<TenElemT, QNT>(update_para, peps_initial), ham_nn_(ham_nn) {}

 private:
  void SetEvolveGate_(void) override {
    evolve_gate_nn_ = TaylorExpMatrix(this->update_para.tau, ham_nn_);
  }

  double SimpleUpdateSweep_(void) override;

  Tensor ham_nn_;
  Tensor evolve_gate_nn_;
};

template<typename TenElemT, typename QNT>
double SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateSweep_(void) {
  Timer simple_update_sweep_timer("simple_update_sweep");
  SimpleUpdateTruncatePara para(this->update_para.Dmin, this->update_para.Dmax, this->update_para.Trunc_err);
  TenElemT e0(0.0);
  double norm = 1.0;
  double middle_bond_trunc_err;
#ifdef QLPEPS_TIMING_MODE
  Timer vertical_nn_projection_timer("vertical_nn_projection");
#endif
  for (size_t col = 0; col < this->lx_; col++) {
    for (size_t row = 0; row < this->ly_ - 1; row++) {
      auto proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {row, col}, VERTICAL, para, ham_nn_);
      e0 += proj_res.e_loc.value();
      norm *= proj_res.norm;
      if (col == this->lx_ / 2 && row == this->ly_ / 2 - 1) {
        middle_bond_trunc_err = proj_res.trunc_err;
      }
    }
  }
#ifdef QLPEPS_TIMING_MODE
  std::cout << "\n";
  vertical_nn_projection_timer.PrintElapsed();
  Timer horizontal_nn_projection_timer("horizontal_nn_projection");
#endif
  for (size_t col = 0; col < this->lx_ - 1; col++) {
    for (size_t row = 0; row < this->ly_; row++) {
      auto proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {row, col}, HORIZONTAL, para, ham_nn_);
      e0 += proj_res.e_loc.value();
      norm *= proj_res.norm;
    }
  }

#ifdef QLPEPS_TIMING_MODE
  horizontal_nn_projection_timer.PrintElapsed();
#endif
  double sweep_time = simple_update_sweep_timer.Elapsed();
  std::cout << "lambda tensors in middle : " << std::endl;
  PrintLambda(this->peps_.lambda_vert({this->ly_ / 2, this->lx_ / 2}));
  PrintLambda(this->peps_.lambda_horiz({this->ly_ / 2, this->lx_ / 2}));
  auto [dmin, dmax] = this->peps_.GetMinMaxBondDim();
  std::cout << "Estimated E0 =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << e0
            << "Estimated En =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << -std::log(norm) / this->update_para.tau
            << " Dmin/Dmax = " << std::setw(2) << std::right << dmin << "/" << std::setw(2) << std::left << dmax
            << " TruncErr = " << std::setprecision(2) << std::scientific << middle_bond_trunc_err << std::fixed
            << " SweepTime = " << std::setw(8) << sweep_time
            << std::endl;
  return Real(e0);
}
}

#endif //QLPEPS_VMC_PEPS_SQUARE_LATTICE_NN_SIMPLE_UPDATE_H
