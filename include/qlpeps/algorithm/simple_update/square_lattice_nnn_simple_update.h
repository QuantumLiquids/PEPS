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
    ham_lowerright_tri_(this->ly_ - 1, this->lx_),
    ham_lowerleft_tri_(this->ly_ - 1, this->lx_ - 1),
    evolve_gate_lowerright_tri_(this->ly_ - 1, this->lx_),
    evolve_gate_lowerleft_tri_(this->ly_ - 1, this->lx_ - 1) {
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
    ham_lowerright_tri_(this->ly_ - 1, this->lx_),
    ham_lowerleft_tri_(this->ly_ - 1, this->lx_ - 1),
    evolve_gate_lowerright_tri_(this->ly_ - 1, this->lx_),
    evolve_gate_lowerleft_tri_(this->ly_ - 1, this->lx_ - 1) {
      assert(ham_on_site_terms_.rows() == this->ly_);
      assert(ham_on_site_terms_.cols() == this->lx_);
    }

 private:
 void SetEvolveGate_(void) override;
  // void SetEvolveGate_(void) override {
  //   evolve_gate_nn_ = TaylorExpMatrix(this->update_para.tau, ham_nn_);
  //   evolve_gate_nn_half_ = TaylorExpMatrix(this->update_para.tau / 2, ham_nn_);
  //   evolve_gate_tri_ = TaylorExpMatrix(this->update_para.tau, ham_tri_);
  // }

  /**
   * @return h_{ABC} = ham_{nnn}^A \otimes id \otimes ham_{nnn}^C
   *                 + id \otimes ham_{nn}^{BC} / bondtri\_horizontal
   *                 + ham_{nn}^{AB} \otimes id / bondtri\_vertical
   *                 + ham\_on\_site\_terms_A \otimes id \otimes id / site\_share\_tri_A
   *                 + id \otimes ham\_on\_site\_terms_B \otimes id / site\_share\_tri_B
   *                 + id \otimes id \otimes ham\_on\_site\_terms_C / site\_share\_tri_C
   *
   *     LowerLeftTriangle                       LowerRightTriangle
   *      (leftright == 0)                         (leftright == 1)
   *      |             |                          |             |
   *      |             |                          |             |
   * -----A----------------------             -------------------A--------
   *      |             |                          |             |
   *      |             |                          |             |
   *      |             |                          |             |
   * -----B-------------C--------             -----C-------------B--------
   *      |             |                          |             |
   *      |             |                          |             |
   */
  Tensor ConstructTriHamiltonian(const SiteIdx &upper_site, const size_t leftright) const {
    Tensor extend_ham_nnn, extend_ham_nn_horizontal, extend_ham_nn_vertical;
    std::vector<SiteIdx> tri_site(3);

    tri_site[0] = upper_site;
    if (leftright == 0) { // LowerLeft
      tri_site[1] = {upper_site[0] + 1, upper_site[1]};
      tri_site[2] = {upper_site[0] + 1, upper_site[1] + 1};
    } else { // LowerRight
      tri_site[1] = {upper_site[0] + 1, upper_site[1]};
      tri_site[2] = {upper_site[0] + 1, upper_site[1] - 1};
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
    Contract(&ham_nnn_, {}, &id, {}, &extend_ham_nnn);
    extend_ham_nnn.Transpose({0, 1, 4, 5, 2, 3});

    // NN term
    size_t bondtri_horizontal = 2; // Number of triangle shared the horizontal NN bond
    size_t bondtri_vertical = 2; // Number of triangle shared the vertical NN bond
    if (leftright == 0 && upper_site[1] == 0) bondtri_vertical = 1; // LowerLeft, first column
    else if (leftright == 1 && upper_site[1] == this->lx_ - 1) bondtri_vertical = 1; // LowerRight, last column

    Contract(&id, {}, &ham_nn_, {}, &extend_ham_nn_horizontal);
    extend_ham_nn_horizontal *= RealT(1) / RealT(bondtri_horizontal);
    Contract(&ham_nn_, {}, &id, {}, &extend_ham_nn_vertical);
    extend_ham_nn_vertical *= RealT(1) / RealT(bondtri_vertical);

     // on-site term
    if (ham_on_site_terms_(0, 0) == nullptr || ham_on_site_terms_(0, 0)->IsDefault()) {
      return extend_ham_nnn + extend_ham_nn_horizontal + extend_ham_nn_vertical;
    } else {
      Tensor term1, term2, term3;
      Tensor extend_on_site1, extend_on_site2, extend_on_site3;

      std::vector<size_t> site_share_tri(3, 6); // Number of triangle shared the site
      for (size_t i = 0; i < tri_site.size(); i++) {
        const SiteIdx &site = tri_site[i];
        if(site[0] == 0 && site[1] > 0 && site[1] < this->lx_ - 1) 
          site_share_tri[i] = 2; // first row
        else if(site[1] == 0 && site[0] > 0 && site[0] < this->ly_ - 1) 
          site_share_tri[i] = 3; // first column
        else if(site[1] == this->lx_ - 1 && site[0] > 0 && site[0] < this->ly_ - 1) 
          site_share_tri[i] = 3; // last column
        else if(site[0] == 0 && (site[1] == 0 || site[1] == this->lx_ - 1))
          site_share_tri[i] = 1; // corner site on the first row
        else if(site[0] == this->ly_ - 1 && (site[1] == 0 || site[1] == this->lx_ - 1))
          site_share_tri[i] = 2; // corner site on the last row
      }

      // site A
      const Tensor & on_site_term1 = ham_on_site_terms_({tri_site[0][0], tri_site[0][1]});
      Contract(&on_site_term1, {}, &id, {}, &term1);
      Contract(&term1, {}, &id, {}, &extend_on_site1);
      extend_on_site1 *=  RealT(1) / RealT(site_share_tri[0]);
      // site B
      const Tensor & on_site_term2 = ham_on_site_terms_({tri_site[1][0], tri_site[1][1]});
      Contract(&id, {}, &on_site_term2, {}, &term2);
      Contract(&term2, {}, &id, {}, &extend_on_site2);
      extend_on_site2 *=  RealT(1) / RealT(site_share_tri[1]);
      // site C
      const Tensor & on_site_term3 = ham_on_site_terms_({tri_site[2][0], tri_site[2][1]});
      Contract(&id, {}, &id, {}, &term3);
      Contract(&term3, {}, &on_site_term3, {}, &extend_on_site3);
      extend_on_site3 *= RealT(1) / RealT(site_share_tri[2]);

      return extend_ham_nnn + extend_ham_nn_horizontal + extend_ham_nn_vertical
          + extend_on_site1 + extend_on_site2 + extend_on_site3;
    }
  }

  Tensor ConstructEvolveOperator(const SiteIdx &upper_site, const size_t leftright) const {
    return TaylorExpMatrix(RealT(this->update_para.tau), ConstructTriHamiltonian(upper_site, leftright));
  }

  RealT SimpleUpdateSweep_(void) override;

  Tensor ham_nn_;   // uniform NN interaction
  Tensor ham_nnn_;  // uniform NNN interaction
  TenMatrix<Tensor> ham_on_site_terms_;  // on-site terms

  TenMatrix<Tensor> ham_lowerright_tri_;
  TenMatrix<Tensor> ham_lowerleft_tri_;
  // A-B-C 3-site, A-B and B-C are NN connect, A-C is NNN connect
  // A-B or B-C has half of J1 interaction, A-C has a J2 interaction
  // A, B and C also have on-site interaction

  Tensor evolve_gate_nn_;
  TenMatrix<Tensor> evolve_gate_lowerright_tri_;
  TenMatrix<Tensor> evolve_gate_lowerleft_tri_;
};

template<typename TenElemT, typename QNT>
void SquareLatticeNNNSimpleUpdateExecutor<TenElemT, QNT>::SetEvolveGate_(void) {
  evolve_gate_nn_ = TaylorExpMatrix(RealT(this->update_para.tau), ham_nn_);
  
  for (size_t col = 0; col < this->lx_ - 1; col++) {
    for (size_t row = 0; row < this->ly_ - 1; row++) {
      ham_lowerright_tri_({row, col + 1}) = ConstructTriHamiltonian({row, col + 1}, 1);
      evolve_gate_lowerright_tri_({row, col + 1}) = ConstructEvolveOperator({row, col + 1}, 1);
      ham_lowerleft_tri_({row, col}) = ConstructTriHamiltonian({row, col}, 0);
      evolve_gate_lowerleft_tri_({row, col}) = ConstructEvolveOperator({row, col}, 0);
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

  for (size_t col = 0; col < this->lx_ - 1; col++) {  //first row
    ProjectionRes<TenElemT>
        proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_, {0, col}, HORIZONTAL, para, ham_nn_);
    e0 += proj_res.e_loc.value();
    norm *= proj_res.norm;
    max_trunc_err = std::max(max_trunc_err, proj_res.trunc_err);
    //e0 += -std::log(proj_res.norm) / this->update_para.tau;
  }

  // for (size_t row = 0; row < this->ly_ - 1; row++) {  // first and last column
  //   ProjectionRes<TenElemT>
  //       proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_half_, {row, 0}, VERTICAL, para);
  //   e0 += -std::log(proj_res.norm) / this->update_para.tau;
  //   proj_res = this->peps_.NearestNeighborSiteProject(evolve_gate_nn_half_, {row, this->lx_ - 1}, VERTICAL, para);
  //   e0 += -std::log(proj_res.norm) / this->update_para.tau;
  // }

  for (size_t col = 0; col < this->lx_ - 1; col++) {
    for (size_t row = 0; row < this->ly_ - 1; row++) {
      ProjectionRes<TenElemT>
          proj_res1 = this->peps_.LowerRightTriangleProject(evolve_gate_lowerright_tri_({row, col + 1}), {row, col + 1}, para, ham_lowerright_tri_({row, col + 1}));
      //e0 += -std::log(proj_res.norm) / this->update_para.tau;
      e0 += proj_res1.e_loc.value();
      norm *= proj_res1.norm;
      max_trunc_err = std::max(max_trunc_err, proj_res1.trunc_err);

      ProjectionRes<TenElemT>
          proj_res2 = this->peps_.LowerLeftTriangleProject(evolve_gate_lowerleft_tri_({row, col}), {row, col}, para, ham_lowerleft_tri_({row, col}));
      //e0 += -std::log(norm) / this->update_para.tau;
      e0 += proj_res2.e_loc.value();
      norm *= proj_res2.norm;
      max_trunc_err = std::max(max_trunc_err, proj_res2.trunc_err);
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
