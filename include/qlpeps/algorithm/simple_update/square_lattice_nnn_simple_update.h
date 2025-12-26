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
                                       const Tensor &ham_nnn,
                                       const Tensor &ham_onsite = Tensor()) :
    SimpleUpdateExecutor<TenElemT, QNT>(update_para, peps_initial),
    ham_nn_(ham_nn), ham_nnn_(ham_nnn),
    ham_on_site_terms_(this->ly_, this->lx_),
    ham_upperright_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                        peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    ham_upperleft_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                        peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    ham_lowerright_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                        peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    ham_lowerleft_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                        peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    evolve_gate_upperright_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                                peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    evolve_gate_upperleft_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                               peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    evolve_gate_lowerright_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                                peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    evolve_gate_lowerleft_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                               peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1) {
    if (!ham_onsite.IsDefault())
    for (auto &ten : ham_on_site_terms_) {
      ten = ham_onsite;
    }
  }
  /**
   * The case containing non-uniform on-site terms, like pinning field.
   */
  SquareLatticeNNNSimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                                       const PEPST &peps_initial,
                                       const Tensor &ham_nn,
                                       const Tensor &ham_nnn,
                                       const TenMatrix<Tensor> &ham_onsite_terms) :
    SimpleUpdateExecutor<TenElemT, QNT>(update_para, peps_initial),
    ham_nn_(ham_nn), ham_nnn_(ham_nnn),
    ham_on_site_terms_(ham_onsite_terms),
    ham_upperright_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                        peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    ham_upperleft_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                        peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    ham_lowerright_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                        peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    ham_lowerleft_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                        peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    evolve_gate_upperright_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                                peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    evolve_gate_upperleft_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                               peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    evolve_gate_lowerright_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                                peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1),
    evolve_gate_lowerleft_tri_(peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->ly_ : this->ly_ - 1, 
                               peps_initial.GetBoundaryCondition() == BoundaryCondition::Periodic ? this->lx_ : this->lx_ - 1) {
      assert(ham_on_site_terms_.rows() == this->ly_);
      assert(ham_on_site_terms_.cols() == this->lx_);
    }

 private:
 void SetEvolveGate_(void) override;

  enum TriProjPOSITION {
    UpperRight = 0,
    UpperLeft,
    LowerRight,
    LowerLeft
  };

  /**
   * @return h_{ABC} = ham_{nnn}^A \otimes id \otimes ham_{nnn}^C
   *                 + id \otimes ham_{nn}^{BC} / bondtri\_horizontal
   *                 + ham_{nn}^{AB} \otimes id / bondtri\_vertical
   *                 + ham\_on\_site\_terms_A \otimes id \otimes id / site\_share\_tri_A
   *                 + id \otimes ham\_on\_site\_terms_B \otimes id / site\_share\_tri_B
   *                 + id \otimes id \otimes ham\_on\_site\_terms_C / site\_share\_tri_C
   *
   *         UpperLeft                      UpperRight                     LowerLeft                      LowerRight
   *      |             |                |             |                |             |                |             |
   *      |             |                |             |                |             |                |             |
   * -----B-------------A--------   -----A-------------B--------   -----A----------------------   -------------------A--------
   *      |             |                |             |                |             |                |             |
   *      |             |                |             |                |             |                |             |
   *      |             |                |             |                |             |                |             |
   * -----C----------------------   -------------------C--------   -----B-------------C--------   -----C-------------B--------
   *      |             |                |             |                |             |                |             |
   *      |             |                |             |                |             |                |             |
   * 
   */
  Tensor ConstructTriHamiltonian(const SiteIdx &site_b, const TriProjPOSITION triposition) const {
    Tensor tri_ham_nnn, tri_ham_nn_1, tri_ham_nn_2, hor_ham_nn, ver_ham_nn;
    std::vector<SiteIdx> tri_site(3);

    tri_site[1] = site_b;
    switch (triposition) {
      case UpperLeft: {
        tri_site[0] = {site_b[0], (site_b[1] + 1) % this->peps_.lambda_horiz.cols()};
        tri_site[2] = {(site_b[0] + 1) % this->peps_.lambda_vert.rows(), site_b[1]};
        break;
      }
      case UpperRight: {
        tri_site[0] = {site_b[0], (site_b[1] - 1 + this->peps_.lambda_horiz.cols()) % this->peps_.lambda_horiz.cols()};
        tri_site[2] = {(site_b[0] + 1) % this->peps_.lambda_vert.rows(), site_b[1]};
        break;
      }
      case LowerLeft: {
        tri_site[0] = {(site_b[0] - 1 + this->peps_.lambda_vert.rows()) % this->peps_.lambda_vert.rows(), site_b[1]};
        tri_site[2] = {(site_b[0] - 1 + this->peps_.lambda_vert.rows()) % this->peps_.lambda_vert.rows(), 
                       (site_b[1] + 1) % this->peps_.lambda_horiz.cols()};
        break;
      }
      case LowerRight: {
        tri_site[0] = {(site_b[0] - 1 + this->peps_.lambda_vert.rows()) % this->peps_.lambda_vert.rows(), site_b[1]};
        tri_site[2] = {site_b[0], (site_b[1] - 1 + this->peps_.lambda_horiz.cols()) % this->peps_.lambda_horiz.cols()};
        break;
      }
    }

    Tensor id = Eye<TenElemT>(ham_nn_.GetIndex(0)); // assume uniform hilbert space.
    if (Tensor::IsFermionic() && id.GetIndex(0).GetDir() != OUT) {
      std::cerr << "Index direction of identity operator is unexpected." << std::endl;
    }
    if (Tensor::IsFermionic()) {
      id.ActFermionPOps();
    }

    // NNN term
    /*      1             3             5              1             3             5
            |             |             |              |             |             |
            v             v             v              v             v             v
            |             |             |   Transpose  |             |             |
            A---ham_nnn---C-----------id(B) -------->  A------------id(B)----------C
            |             |             |              |             |             |
            v             v             v              v             v             v
            |             |             |              |             |             |
            0             2             4              0             2             4
     */
    Contract(&ham_nnn_, {}, &id, {}, &tri_ham_nnn);
    tri_ham_nnn *= RealT(0.5);
    tri_ham_nnn.Transpose({0, 1, 4, 5, 2, 3});

    const bool is_pbc = (this->peps_.GetBoundaryCondition() == BoundaryCondition::Periodic);

    // NN term
    Contract(&id, {}, &ham_nn_, {}, &tri_ham_nn_1);
    Contract(&ham_nn_, {}, &id, {}, &tri_ham_nn_2);
    size_t hor_bond_tri_num = 4; // Number of triangle shared the horizontal NN bond
    size_t ver_bond_tri_num = 4; // Number of triangle shared the vertical NN bond
    if (!is_pbc) {
      switch (triposition) {
        case UpperLeft: {
          if(site_b[0] == 0) hor_bond_tri_num = 2;
          if(site_b[1] == 0) ver_bond_tri_num = 2;
          hor_ham_nn = tri_ham_nn_2 * (RealT(1) / RealT(hor_bond_tri_num));
          ver_ham_nn = tri_ham_nn_1 * (RealT(1) / RealT(ver_bond_tri_num));
          break;
        }
        case UpperRight: {
          if(site_b[0] == 0) hor_bond_tri_num = 2;
          if(site_b[1] == this->lx_ - 1) ver_bond_tri_num = 2;
          hor_ham_nn = tri_ham_nn_2 * (RealT(1) / RealT(hor_bond_tri_num));
          ver_ham_nn = tri_ham_nn_1 * (RealT(1) / RealT(ver_bond_tri_num));
          break;
        }
        case LowerLeft: {
          if(site_b[0] == this->ly_ - 1) hor_bond_tri_num = 2;
          if(site_b[1] == 0) ver_bond_tri_num = 2;
          hor_ham_nn = tri_ham_nn_1 * (RealT(1) / RealT(hor_bond_tri_num));
          ver_ham_nn = tri_ham_nn_2 * (RealT(1) / RealT(ver_bond_tri_num));
          break;
        }
        case LowerRight: {
          if(site_b[0] == this->ly_ - 1) hor_bond_tri_num = 2;
          if(site_b[1] == this->lx_ - 1) ver_bond_tri_num = 2;
          hor_ham_nn = tri_ham_nn_1 * (RealT(1) / RealT(hor_bond_tri_num));
          ver_ham_nn = tri_ham_nn_2 * (RealT(1) / RealT(ver_bond_tri_num));
          break;
        }
      }
    }

     // on-site term
    if (ham_on_site_terms_(0, 0) == nullptr || ham_on_site_terms_(0, 0)->IsDefault()) {
      return tri_ham_nnn + hor_ham_nn + ver_ham_nn;
    } else {
      Tensor term1, term2, term3;
      Tensor tri_on_site1, tri_on_site2, tri_on_site3;

      std::vector<size_t> site_tri_num(3, 12); // Number of triangle shared the site
      if (!is_pbc) {
        for (size_t i = 0; i < tri_site.size(); i++) {
          const SiteIdx &site = tri_site[i];
          if(site[0] == 0 && site[1] > 0 && site[1] < this->lx_ - 1) 
            site_tri_num[i] = 6; // first row
          else if(site[0] == this->ly_ - 1 && site[1] > 0 && site[1] < this->lx_ - 1) 
            site_tri_num[i] = 6; // last row
          else if(site[1] == 0 && site[0] > 0 && site[0] < this->ly_ - 1) 
            site_tri_num[i] = 6; // first column
          else if(site[1] == this->lx_ - 1 && site[0] > 0 && site[0] < this->ly_ - 1) 
            site_tri_num[i] = 6; // last column
          else if(site[0] == 0 && (site[1] == 0 || site[1] == this->lx_ - 1))
            site_tri_num[i] = 3; // corner site on the first row
          else if(site[0] == this->ly_ - 1 && (site[1] == 0 || site[1] == this->lx_ - 1))
            site_tri_num[i] = 3; // corner site on the last row
        }
      }

      // site A
      const Tensor & on_site_term1 = ham_on_site_terms_({tri_site[0][0], tri_site[0][1]});
      Contract(&on_site_term1, {}, &id, {}, &term1);
      Contract(&term1, {}, &id, {}, &tri_on_site1);
      tri_on_site1 *=  RealT(1) / RealT(site_tri_num[0]);
      // site B
      const Tensor & on_site_term2 = ham_on_site_terms_({tri_site[1][0], tri_site[1][1]});
      Contract(&id, {}, &on_site_term2, {}, &term2);
      Contract(&term2, {}, &id, {}, &tri_on_site2);
      tri_on_site2 *=  RealT(1) / RealT(site_tri_num[1]);
      // site C
      const Tensor & on_site_term3 = ham_on_site_terms_({tri_site[2][0], tri_site[2][1]});
      Contract(&id, {}, &id, {}, &term3);
      Contract(&term3, {}, &on_site_term3, {}, &tri_on_site3);
      tri_on_site3 *= RealT(1) / RealT(site_tri_num[2]);

      return tri_ham_nnn + hor_ham_nn + ver_ham_nn
          + tri_on_site1 + tri_on_site2 + tri_on_site3;
    }
  }

  Tensor ConstructEvolveOperator(const SiteIdx &site_b, const TriProjPOSITION triposition) const {
    return TaylorExpMatrix(RealT(this->update_para.tau), ConstructTriHamiltonian(site_b, triposition));
  }

  RealT SimpleUpdateSweep_(void) override;

  Tensor ham_nn_;   // uniform NN interaction
  Tensor ham_nnn_;  // uniform NNN interaction
  TenMatrix<Tensor> ham_on_site_terms_;  // on-site terms

  TenMatrix<Tensor> ham_upperright_tri_;
  TenMatrix<Tensor> ham_upperleft_tri_;
  TenMatrix<Tensor> ham_lowerright_tri_;
  TenMatrix<Tensor> ham_lowerleft_tri_;
  // A-B-C 3-site, A-B and B-C are NN connect, A-C is NNN connect
  // A-B or B-C has J1 interaction divided by the triangle shared this bond
  // A-C has half of the J2 interaction
  // A, B and C also have on-site interaction

  TenMatrix<Tensor> evolve_gate_upperright_tri_;
  TenMatrix<Tensor> evolve_gate_upperleft_tri_;
  TenMatrix<Tensor> evolve_gate_lowerright_tri_;
  TenMatrix<Tensor> evolve_gate_lowerleft_tri_;
};

template<typename TenElemT, typename QNT>
void SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>::SetEvolveGate_(void) {
  const bool is_pbc = (this->peps_.GetBoundaryCondition() == BoundaryCondition::Periodic);
  const size_t hor_bond_limit = is_pbc ? this->lx_ : this->lx_ - 1;
  const size_t ver_bond_limit = is_pbc ? this->ly_ : this->ly_ - 1;

  
  for (size_t col = 0; col < hor_bond_limit; col++) {
    for (size_t row = 0; row < ver_bond_limit; row++) {
      ham_upperleft_tri_({row, col}) = ConstructTriHamiltonian({row, col}, UpperLeft);
      evolve_gate_upperleft_tri_({row, col}) = ConstructEvolveOperator({row, col}, UpperLeft);

      ham_upperright_tri_({row, col}) = ConstructTriHamiltonian({row, (col + 1) % this->peps_.lambda_horiz.cols()}, UpperRight);
      evolve_gate_upperright_tri_({row, col}) = ConstructEvolveOperator({row, (col + 1) % this->peps_.lambda_horiz.cols()}, UpperRight);

      ham_lowerleft_tri_({row, col}) = ConstructTriHamiltonian({(row + 1) % this->peps_.lambda_vert.rows(), col}, LowerLeft);
      evolve_gate_lowerleft_tri_({row, col}) = ConstructEvolveOperator({(row + 1) % this->peps_.lambda_vert.rows(), col}, LowerLeft);
      
      ham_lowerright_tri_({row, col}) = ConstructTriHamiltonian({(row + 1) % this->peps_.lambda_vert.rows(), (col + 1) % this->peps_.lambda_horiz.cols()}, LowerRight);
      evolve_gate_lowerright_tri_({row, col}) = ConstructEvolveOperator({(row + 1) % this->peps_.lambda_vert.rows(), (col + 1) % this->peps_.lambda_horiz.cols()}, LowerRight);
    }
  }
  
}

template<typename TenElemT, typename QNT>
typename SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>::RealT SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>::SimpleUpdateSweep_(void) {
  Timer simple_update_sweep_timer("simple_update_sweep");
  SimpleUpdateTruncatePara para(this->update_para.Dmin, this->update_para.Dmax, this->update_para.Trunc_err);
  TenElemT e0(0.0);
  RealT norm = 1.0;
  RealT max_trunc_err = 0.0;

  const bool is_pbc = (this->peps_.GetBoundaryCondition() == BoundaryCondition::Periodic);
  const size_t hor_bond_limit = is_pbc ? this->lx_ : this->lx_ - 1;
  const size_t ver_bond_limit = is_pbc ? this->ly_ : this->ly_ - 1;

  for (size_t col = 0; col < hor_bond_limit; col++) {
    for (size_t row = 0; row < ver_bond_limit; row++) {
      ProjectionRes<TenElemT>
          proj_res1 = this->peps_.UpperRightTriangleProject(evolve_gate_upperright_tri_({row, col}), 
                                                           {row, (col + 1) % this->peps_.lambda_horiz.cols()}, 
                                                            para, 
                                                            ham_upperright_tri_({row, col}));
      e0 += proj_res1.e_loc.value();
      norm *= proj_res1.norm;
      max_trunc_err = std::max(max_trunc_err, proj_res1.trunc_err);
      
      ProjectionRes<TenElemT>
          proj_res2 = this->peps_.UpperLeftTriangleProject(evolve_gate_upperleft_tri_({row, col}), 
                                                          {row, col}, 
                                                           para, 
                                                           ham_upperleft_tri_({row, col}));
      e0 += proj_res2.e_loc.value();
      norm *= proj_res2.norm;
      max_trunc_err = std::max(max_trunc_err, proj_res2.trunc_err);

      ProjectionRes<TenElemT>
          proj_res3 = this->peps_.LowerRightTriangleProject(evolve_gate_lowerright_tri_({row, col}), 
                                                           {row, (col + 1) % this->peps_.lambda_horiz.cols()}, 
                                                            para, 
                                                            ham_lowerright_tri_({row, col}));
      e0 += proj_res3.e_loc.value();
      norm *= proj_res3.norm;
      max_trunc_err = std::max(max_trunc_err, proj_res3.trunc_err);

      ProjectionRes<TenElemT>
          proj_res4 = this->peps_.LowerLeftTriangleProject(evolve_gate_lowerleft_tri_({row, col}), 
                                                          {row, col}, 
                                                           para, 
                                                           ham_lowerleft_tri_({row, col}));
      e0 += proj_res4.e_loc.value();
      norm *= proj_res4.norm;
      max_trunc_err = std::max(max_trunc_err, proj_res4.trunc_err);
    }
  }

  double sweep_time = simple_update_sweep_timer.Elapsed();
  auto [dmin, dmax] = this->peps_.GetMinMaxBondDim();
  std::cout << "Estimated E0 =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << e0
            << "Estimated En =" << std::setw(15) << std::setprecision(kEnergyOutputPrecision) << std::fixed
            << std::right << -std::log(norm) / this->update_para.tau
            << " Dmin/Dmax = " << std::setw(2) << std::right << dmin << "/" << std::setw(2) << std::left << dmax
            << " TruncErr = " << std::setprecision(2) << std::scientific << max_trunc_err << std::fixed
            << " SweepTime = " << std::setw(8) << sweep_time
            << std::endl;

  return qlmps::Real(e0);
}
}

#endif //QLPEPS_VMC_PEPS_SQUARE_LATTICE_NNN_SIMPLE_UPDATE_H
