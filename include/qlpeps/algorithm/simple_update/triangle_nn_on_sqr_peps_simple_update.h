// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-28
*
* Description: QuantumLiquids/VMC-SquareLatticePEPS project.
*              Simple Update for nearest-neighbor interaction triangle lattice models in square lattice PEPS
*/

#ifndef QLPEPS_VMC_PEPS_TRIANGLE_NN_ON_SQR_PEPS_SIMPLE_UPDATE_H
#define QLPEPS_VMC_PEPS_TRIANGLE_NN_ON_SQR_PEPS_SIMPLE_UPDATE_H

#include "qlpeps/algorithm/simple_update/simple_update.h"

namespace qlpeps {

using namespace qlten;

template<typename TenElemT, typename QNT>
class TriangleNNModelSquarePEPSSimpleUpdateExecutor : public SimpleUpdateExecutor<TenElemT, QNT> {
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
 public:
  /**
   * @param ham_nn     nearest-neighbor interaction, involving bonds on upper and left edges of the lattice
   * @param ham_tri    three-site triangle interaction term
   */
  TriangleNNModelSquarePEPSSimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                                const PEPST &peps_initial,
                                                const Tensor &ham_nn,
                                                const Tensor &ham_tri) :
      SimpleUpdateExecutor<TenElemT, QNT>(update_para, peps_initial), \
          ham_nn_(ham_nn), ham_tri_(ham_tri) {}

 private:
  void SetEvolveGate_(void) override {
    evolve_gate_nn_ = TaylorExpMatrix(this->update_para.tau, ham_nn_);
    evolve_gate_tri_ = TaylorExpMatrix(this->update_para.tau, ham_tri_);
  }

  double SimpleUpdateSweep_(void) override;

  Tensor ham_nn_;
  Tensor ham_tri_;
  Tensor evolve_gate_nn_;
  Tensor evolve_gate_tri_;
};

template<typename TenElemT, typename QNT>
double TriangleNNModelSquarePEPSSimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateSweep_(void) {
  Timer simple_update_sweep_timer("simple_update_sweep");
  SimpleUpdateTruncatePara para(this->update_para.Dmin, this->update_para.Dmax, this->update_para.Trunc_err);
  double norm_a(1), norm_b(1);

  for (size_t row = 0; row < this->ly_ - 1; row++) {
    auto proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {row, 0}, VERTICAL, para);
    norm_a *= proj_res.norm;
  }
  for (size_t col = 1; col < this->lx_; col++) {
    auto proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {0, col - 1}, HORIZONTAL, para);
    norm_a *= proj_res.norm;
    for (size_t row = 0; row < this->ly_ - 1; row++) {
      auto proj_res = this->peps_.LowerRightTriangleProject(evolve_gate_tri_, {row, col}, para);
      norm_a *= proj_res.norm;
    }
  }

  double e_a = -std::log(norm_a) / this->update_para.tau;
  for (size_t col = 0; col < this->lx_ - 1; col++) {
    for (size_t row = 0; row < this->ly_ - 1; row++) {
      auto proj_res = this->peps_.UpperLeftTriangleProject(evolve_gate_tri_, {row, col}, para);
      norm_b *= proj_res.norm;
    }
    auto proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {this->ly_ - 1, col}, HORIZONTAL, para);
    norm_b *= proj_res.norm;
  }
  for (size_t row = 0; row < this->ly_ - 1; row++) {
    auto proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {row, this->lx_ - 1}, VERTICAL, para);
    norm_b *= proj_res.norm;
  }

  double e_b = -std::log(norm_b) / this->update_para.tau;
  double sweep_time = simple_update_sweep_timer.Elapsed();
  auto [dmin, dmax] = this->peps_.GetMinMaxBondDim();
  std::cout << "Estimated E0 =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << (e_a + e_b) / 2
            << " Delta E0 =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << std::fabs(e_a - e_b) / 2
            << " Dmin/Dmax = " << std::setw(2) << std::right << dmin << "/" << std::setw(2) << std::left << dmax
            << " SweepTime = " << std::setw(8) << sweep_time
            << std::endl;
  return (e_a + e_b) / 2;
}
}

#endif //QLPEPS_VMC_PEPS_TRIANGLE_NN_ON_SQR_PEPS_SIMPLE_UPDATE_H
