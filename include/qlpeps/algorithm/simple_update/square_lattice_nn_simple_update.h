// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-28
*
* Description: QuantumLiquids/PEPS project.
*              Simple update implementation for
 *             nearest-neighbor interaction models in square lattice,
 *             allowed additional on-site terms.
*/

#ifndef QLPEPS_ALGORITHM_SIMPLE_UPDATE_SQUARE_LATTICE_NN_SIMPLE_UPDATE_H
#define QLPEPS_ALGORITHM_SIMPLE_UPDATE_SQUARE_LATTICE_NN_SIMPLE_UPDATE_H

#include "qlpeps/algorithm/simple_update/simple_update.h"

namespace qlpeps {

using namespace qlten;

template<typename TenElemT, typename QNT>
class SquareLatticeNNSimpleUpdateExecutor : public SimpleUpdateExecutor<TenElemT, QNT> {
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
 public:
  /**
   *
   * @param update_para
   * @param peps_initial
   *
   * take transverse-field Ising as example, H = \sum_<i,j> S_i^z * S_j^z + h \sum_i S_i^x
   * where
   * @param ham_nn  S_i^z * S_j^z
   * @param ham_onsite  h * S_i^x
   */
  SquareLatticeNNSimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                      const PEPST &peps_initial,
                                      const Tensor &ham_nn,
                                      const Tensor &ham_onsite = Tensor()) :
      SimpleUpdateExecutor<TenElemT, QNT>(update_para, peps_initial), ham_two_site_term_(ham_nn),
      ham_on_site_terms_(this->ly_, this->lx_),
      horizontal_nn_ham_set_(this->ly_, this->lx_ - 1),
      vertical_nn_ham_set_(this->ly_ - 1, this->lx_),
      horizontal_nn_evolve_gate_set_(this->ly_, this->lx_ - 1),
      vertical_nn_evolve_gate_set_(this->ly_ - 1, this->lx_) {
    if (!ham_onsite.IsDefault())
      for (auto &ten : ham_on_site_terms_) {
        ten = ham_onsite;
      }
  }
  /**
   * The case containing non-uniform on-site terms, like pinning field.
   */
  SquareLatticeNNSimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                      const PEPST &peps_initial,
                                      const Tensor &ham_nn,
                                      const TenMatrix<Tensor> &ham_onsite_terms) :
      SimpleUpdateExecutor<TenElemT, QNT>(update_para, peps_initial), ham_two_site_term_(ham_nn),
      ham_on_site_terms_(ham_onsite_terms),
      horizontal_nn_ham_set_(this->ly_, this->lx_ - 1),
      vertical_nn_ham_set_(this->ly_ - 1, this->lx_),
      horizontal_nn_evolve_gate_set_(this->ly_, this->lx_ - 1),
      vertical_nn_evolve_gate_set_(this->ly_ - 1, this->lx_) {
    assert(ham_on_site_terms_.rows() == this->ly_);
    assert(ham_on_site_terms_.cols() == this->lx_);
  }
 private:
  void SetEvolveGate_(void) override;

  /**
   * @return h_ij = ham_two_site_term_ + h1 * ham_one_site_term_ * id + h2 * id * ham_one_site_term_
   */
  Tensor ConstructBondHamiltonian(const TenElemT h1, const Tensor &on_site_term1,
                                  const TenElemT h2, const Tensor &on_site_term2,
                                  const Tensor &id) const {
    Tensor term1, term2;
    Contract(&on_site_term1, {}, &id, {}, &term1);
    term1 *= h1;
    Contract(&id, {}, &on_site_term2, {}, &term2);
    term2 *= h2;
    return ham_two_site_term_ + term1 + term2;
  }

  Tensor ConstructEvolveOperator(const TenElemT h1, const Tensor &on_site_term1,
                                 const TenElemT h2, const Tensor &on_site_term2,
                                 const Tensor &id) const {
    return TaylorExpMatrix(this->update_para.tau,
                           ConstructBondHamiltonian(h1, on_site_term1, h2, on_site_term2, id));
  }

  double SimpleUpdateSweep_(void) override;

  Tensor ham_two_site_term_; //uniform bond term

  TenMatrix<Tensor> ham_on_site_terms_;  // on-site terms

  TenMatrix<Tensor> horizontal_nn_ham_set_;
  TenMatrix<Tensor> vertical_nn_ham_set_;

  TenMatrix<Tensor> horizontal_nn_evolve_gate_set_;
  TenMatrix<Tensor> vertical_nn_evolve_gate_set_;
};

template<typename TenElemT, typename QNT>
void SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT>::SetEvolveGate_() {
  if (ham_on_site_terms_(0, 0) == nullptr || ham_on_site_terms_(0, 0)->IsDefault()) {
    Tensor evolve_gate_nn = TaylorExpMatrix(this->update_para.tau, ham_two_site_term_);
    for (auto &ten : horizontal_nn_ham_set_) {
      ten = ham_two_site_term_;
    }
    for (auto &ten : vertical_nn_ham_set_) {
      ten = ham_two_site_term_;
    }
    for (auto &ten : horizontal_nn_evolve_gate_set_) {
      ten = evolve_gate_nn;
    }
    for (auto &ten : vertical_nn_evolve_gate_set_) {
      ten = evolve_gate_nn;
    }
  } else {// transverse-field Ising, Hubbard model, t-J + chemical potential
    //construct the on-site identity operator
    Tensor id(ham_on_site_terms_(0, 0)->GetIndexes()); // assume uniform hilbert space.
    if (Tensor::IsFermionic() && id.GetIndex(0).GetDir() != OUT) {
      std::cerr << "Index direction of on-site hamiltonian is unexpected." << std::endl;
    }
    for (size_t i = 0; i < id.GetShape()[0]; i++) {
      id({i, i}) = 1.0;
    }
    if (Tensor::IsFermionic()) {
      id.ActFermionPOps();
    }

    for (size_t col = 0; col < this->lx_ - 1; col++) {
      for (size_t row = 1; row < this->ly_ - 1; row++) {
        horizontal_nn_ham_set_({row, col}) = ConstructBondHamiltonian(0.25, ham_on_site_terms_({row, col}),
                                                                      0.25, ham_on_site_terms_({row, col + 1}),
                                                                      id);
        horizontal_nn_evolve_gate_set_({row, col}) = ConstructEvolveOperator(0.25, ham_on_site_terms_({row, col}),
                                                                             0.25, ham_on_site_terms_({row, col + 1}),
                                                                             id);
      }
    }
    for (size_t col = 1; col < this->lx_ - 1; col++) {
      for (size_t row = 0; row < this->ly_ - 1; row++) {
        vertical_nn_ham_set_({row, col}) = ConstructBondHamiltonian(0.25, ham_on_site_terms_({row, col}),
                                                                    0.25, ham_on_site_terms_({row + 1, col}),
                                                                    id);
        vertical_nn_evolve_gate_set_({row, col}) = ConstructEvolveOperator(0.25, ham_on_site_terms_({row, col}),
                                                                           0.25, ham_on_site_terms_({row + 1, col}),
                                                                           id);
      }
    }

    for (size_t col = 1; col < this->lx_ - 2; col++) {
      horizontal_nn_evolve_gate_set_({0, col}) = ConstructEvolveOperator(0.375, ham_on_site_terms_({0, col}),
                                                                         0.375, ham_on_site_terms_({0, col + 1}), id);
      horizontal_nn_evolve_gate_set_({this->ly_ - 1, col}) =
          ConstructEvolveOperator(0.375, ham_on_site_terms_({this->ly_ - 1, col}),
                                  0.375, ham_on_site_terms_({this->ly_ - 1, col + 1}), id);

      horizontal_nn_ham_set_({0, col}) = ConstructBondHamiltonian(0.375, ham_on_site_terms_({0, col}),
                                                                  0.375, ham_on_site_terms_({0, col + 1}), id);
      horizontal_nn_ham_set_({this->ly_ - 1, col}) =
          ConstructBondHamiltonian(0.375, ham_on_site_terms_({this->ly_ - 1, col}),
                                   0.375, ham_on_site_terms_({this->ly_ - 1, col + 1}), id);
    }

    for (size_t row = 1; row < this->ly_ - 2; row++) {
      vertical_nn_evolve_gate_set_({row, 0}) = ConstructEvolveOperator(0.375, ham_on_site_terms_({row, 0}),
                                                                       0.375, ham_on_site_terms_({row + 1, 0}), id);
      vertical_nn_evolve_gate_set_({row, this->lx_ - 1}) =
          ConstructEvolveOperator(0.375, ham_on_site_terms_({row, this->lx_ - 1}),
                                  0.375, ham_on_site_terms_({row + 1, this->lx_ - 1}), id);

      vertical_nn_ham_set_({row, 0}) = ConstructBondHamiltonian(0.375, ham_on_site_terms_({row, 0}),
                                                                0.375, ham_on_site_terms_({row + 1, 0}), id);
      vertical_nn_ham_set_({row, this->lx_ - 1}) =
          ConstructBondHamiltonian(0.375, ham_on_site_terms_({row, this->lx_ - 1}),
                                   0.375, ham_on_site_terms_({row + 1, this->lx_ - 1}), id);
    }

    //corner terms
    horizontal_nn_evolve_gate_set_({0, 0}) = ConstructEvolveOperator(0.5, ham_on_site_terms_({0, 0}),
                                                                     0.375, ham_on_site_terms_({0, 1}), id);
    horizontal_nn_evolve_gate_set_({this->ly_ - 1, 0}) =
        ConstructEvolveOperator(0.5, ham_on_site_terms_({this->ly_ - 1, 0}),
                                0.375, ham_on_site_terms_({this->ly_ - 1, 1}), id);
    vertical_nn_evolve_gate_set_({0, 0}) = ConstructEvolveOperator(0.5, ham_on_site_terms_({0, 0}),
                                                                   0.375, ham_on_site_terms_({1, 0}), id);
    vertical_nn_evolve_gate_set_({0, this->lx_ - 1}) =
        ConstructEvolveOperator(0.5, ham_on_site_terms_({0, this->lx_ - 1}),
                                0.375, ham_on_site_terms_({1, this->lx_ - 1}), id);

    horizontal_nn_ham_set_({0, 0}) = ConstructBondHamiltonian(0.5, ham_on_site_terms_({0, 0}),
                                                              0.375, ham_on_site_terms_({0, 1}), id);
    horizontal_nn_ham_set_({this->ly_ - 1, 0}) = ConstructBondHamiltonian(0.5,
                                                                          ham_on_site_terms_({this->ly_ - 1, 0}),
                                                                          0.375,
                                                                          ham_on_site_terms_({this->ly_ - 1, 1}),
                                                                          id);
    vertical_nn_ham_set_({0, 0}) = ConstructBondHamiltonian(0.5, ham_on_site_terms_({0, 0}),
                                                            0.375, ham_on_site_terms_({1, 0}), id);
    vertical_nn_ham_set_({0, this->lx_ - 1}) = ConstructBondHamiltonian(0.5,
                                                                        ham_on_site_terms_({0, this->lx_ - 1}),
                                                                        0.375,
                                                                        ham_on_site_terms_({1, this->lx_ - 1}),
                                                                        id);

    horizontal_nn_evolve_gate_set_({0, this->lx_ - 2}) =
        ConstructEvolveOperator(0.375, ham_on_site_terms_({0, this->lx_ - 2}),
                                0.5, ham_on_site_terms_({0, this->lx_ - 1}), id);
    vertical_nn_evolve_gate_set_({this->ly_ - 2, 0}) =
        ConstructEvolveOperator(0.375, ham_on_site_terms_({this->ly_ - 2, 0}),
                                0.5, ham_on_site_terms_({this->ly_ - 1, 0}), id);
    horizontal_nn_evolve_gate_set_({this->ly_ - 1, this->lx_ - 2}) =
        ConstructEvolveOperator(0.375, ham_on_site_terms_({this->ly_ - 1, this->lx_ - 2}),
                                0.5, ham_on_site_terms_({this->ly_ - 1, this->lx_ - 1}), id);
    vertical_nn_evolve_gate_set_({this->ly_ - 2, this->lx_ - 1}) =
        ConstructEvolveOperator(0.375, ham_on_site_terms_({this->ly_ - 2, this->lx_ - 1}),
                                0.5, ham_on_site_terms_({this->ly_ - 1, this->lx_ - 1}), id);

    horizontal_nn_ham_set_({0, this->lx_ - 2}) = ConstructBondHamiltonian(0.375,
                                                                          ham_on_site_terms_({0, this->lx_ - 2}),
                                                                          0.5,
                                                                          ham_on_site_terms_({0, this->lx_ - 1}),
                                                                          id);
    vertical_nn_ham_set_({this->ly_ - 2, 0}) = ConstructBondHamiltonian(0.375,
                                                                        ham_on_site_terms_({this->ly_ - 2, 0}),
                                                                        0.5,
                                                                        ham_on_site_terms_({this->ly_ - 1, 0}),
                                                                        id);
    horizontal_nn_ham_set_({this->ly_ - 1, this->lx_ - 2}) =
        ConstructBondHamiltonian(0.375, ham_on_site_terms_({this->ly_ - 1, this->lx_ - 2}),
                                 0.5, ham_on_site_terms_({this->ly_ - 1, this->lx_ - 1}), id);
    vertical_nn_ham_set_({this->ly_ - 2, this->lx_ - 1}) =
        ConstructBondHamiltonian(0.375, ham_on_site_terms_({this->ly_ - 2, this->lx_ - 1}),
                                 0.5, ham_on_site_terms_({this->ly_ - 1, this->lx_ - 1}), id);
  }
}

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
      SiteIdx upper_site{row, col};
      auto proj_res = this->peps_.NearestNeighborSiteProject(vertical_nn_evolve_gate_set_(upper_site),
                                                             upper_site,
                                                             VERTICAL,
                                                             para,
                                                             vertical_nn_ham_set_(upper_site));
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
      SiteIdx left_site = {row, col};
      auto proj_res = this->peps_.NearestNeighborSiteProject(horizontal_nn_evolve_gate_set_(left_site),
                                                             left_site,
                                                             HORIZONTAL,
                                                             para,
                                                             horizontal_nn_ham_set_(left_site));
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

#endif //QLPEPS_ALGORITHM_SIMPLE_UPDATE_SQUARE_LATTICE_NN_SIMPLE_UPDATE_H
