/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-28
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 Triangle Heisenberg model on square PEPS
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/utility/observable_matrix.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps_contractor.h" //BMPSContractor

namespace qlpeps {
using namespace qlten;

/**
 * Spin-1/2 Heisenberg Model on Triangular Lattice using Square PEPS
 * 
 * Hamiltonian:
 * $$H = J \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j = J \sum_{\langle i,j \rangle} (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j)$$
 * 
 * where:
 * - Sum over all nearest-neighbor bonds on triangular lattice
 * - J: exchange coupling constant (J > 0 for antiferromagnetic)
 * - Triangular geometry mapped onto square PEPS representation
 * 
 * Bond structure:
 * - Horizontal bonds: (i,j) ↔ (i,j+1)
 * - Vertical bonds: (i,j) ↔ (i+1,j) 
 * - Diagonal bonds: (i,j) ↔ (i+1,j+1) [↘ direction]
 */
class SpinOneHalfTriHeisenbergSqrPEPS : public ModelEnergySolver<SpinOneHalfTriHeisenbergSqrPEPS>,
                                        public ModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS> {
 public:
  SpinOneHalfTriHeisenbergSqrPEPS(void) = default;

  using ModelEnergySolver<SpinOneHalfTriHeisenbergSqrPEPS>::CalEnergyAndHoles;
  using ModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS>::EvaluateObservables;
  using ModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS>::DescribeObservables;

  template<typename TenElemT, typename QNT, bool calchols>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  ) {
    using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
    BMPSContractor<TenElemT, QNT> &contractor = tps_sample->contractor;
    const Configuration &sample_config = tps_sample->config;
    const BMPSTruncateParams<RealT> &trunc_para = tps_sample->trun_para;
    return this->template CalEnergyAndHolesImpl<TenElemT, QNT, calchols>(split_index_tps,
                                                                         sample_config,
                                                                         sample_tn,
                                                                         contractor,
                                                                         trunc_para,
                                                                         hole_res,
                                                                         psi_list);
  }

  // Legacy SampleMeasureImpl removed under registry-only API

  template<typename TenElemT, typename QNT, bool calchols>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      const Configuration &config,
      TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const BMPSTruncateParams<typename RealTypeTrait<TenElemT>::type> &trunc_para,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  );

  // Registry API: preserve legacy measurement logic; emit energy, bond energies and correlations
  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample) {
    using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
    ObservableMap<TenElemT> out;
    std::vector<TenElemT> psi_list;

    auto &tn = tps_sample->tn;
    auto &contractor = tps_sample->contractor;
    const Configuration &config = tps_sample->config;
    const BMPSTruncateParams<RealT> &trunc_para = tps_sample->trun_para;
    const size_t ly = tn.rows();
    const size_t lx = tn.cols();

    // site-local: spin_z per site
    {
      std::vector<TenElemT> sz; sz.reserve(config.size());
      for (auto &c : config) { sz.push_back(static_cast<double>(c) - 0.5); }
      out["spin_z"] = std::move(sz);
    }

    // bond energies split by orientation on triangular mapping (h, v, diag_ur)
    ObservableMatrix<TenElemT> e_h;
    ObservableMatrix<TenElemT> e_v;
    ObservableMatrix<TenElemT> e_ur;
    if (lx > 1) e_h.Resize(ly, lx - 1);
    if (ly > 1) e_v.Resize(ly - 1, lx);
    if (lx > 1 && ly > 1) e_ur.Resize(ly - 1, lx - 1);
    TenElemT energy_total(0);

    // Horizontal scan (and collect psi along rows)
    contractor.GenerateBMPSApproach(tn, UP, trunc_para);
    for (size_t row = 0; row < ly; ++row) {
      contractor.InitBTen(tn, LEFT, row);
      contractor.GrowFullBTen(tn, RIGHT, row, 1, true);
      auto psi = contractor.Trace(tn, {row, 0}, HORIZONTAL);
      auto inv_psi = TenElemT(1.0) / psi;
      psi_list.push_back(psi);
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx s1{row, col};
        if (col + 1 < lx) {
          const SiteIdx s2{row, col + 1};
          TenElemT eb;
          if (config(s1) == config(s2)) {
            eb = TenElemT(0.25);
          } else {
            TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, s1, s2, HORIZONTAL,
                                                    (*split_index_tps)(s1)[config(s2)],
                                                    (*split_index_tps)(s2)[config(s1)]);
            eb = (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
          }
          if (e_h.size() != 0) {
            e_h(row, col) = eb;
          }
          energy_total += eb;
          contractor.BTenMoveStep(tn, RIGHT);
        }
      }
      if (row == ly / 2) {
        // Middle-row correlations: SzSz along row and XY channels via single-site flips
        {
          std::vector<TenElemT> szsz_row; szsz_row.reserve(lx / 2);
          SiteIdx site1{row, lx / 4};
          double sz1 = static_cast<double>(config(site1)) - 0.5;
          for (size_t i = 1; i <= lx / 2; ++i) {
            SiteIdx site2{row, lx / 4 + i};
            double sz2 = static_cast<double>(config(site2)) - 0.5;
            szsz_row.push_back(sz1 * sz2);
          }
          if (!szsz_row.empty()) out["SzSz_row"] = std::move(szsz_row);
        }

        // XY channels using legacy environment sequence
        {
          SiteIdx site1{row, lx / 4};
          std::vector<TenElemT> diag_corr(lx / 2, TenElemT(0));
          tn(site1) = (*split_index_tps)(site1)[1 - config(site1)]; // temporarily flip site1
          contractor.TruncateBTen(LEFT, lx / 4 + 1);
          contractor.GrowBTenStep(tn, LEFT);
          contractor.GrowFullBTen(tn, RIGHT, row, lx / 4 + 2, false);
          for (size_t i = 1; i <= lx / 2; ++i) {
            SiteIdx site2{row, lx / 4 + i};
            if (config(site2) == config(site1)) {
              diag_corr[i - 1] = TenElemT(0);
            } else {
              TenElemT psi_ex = contractor.ReplaceOneSiteTrace(tn, site2, (*split_index_tps)(site2)[1 - config(site2)], HORIZONTAL);
              diag_corr[i - 1] = ComplexConjugate(psi_ex * inv_psi);
            }
            contractor.BTenMoveStep(tn, RIGHT);
          }
          tn(site1) = (*split_index_tps)(site1)[config(site1)]; // revert

          std::vector<TenElemT> SmSp_row = diag_corr;
          std::vector<TenElemT> SpSm_row(diag_corr.size(), TenElemT(0));
          if (config(site1) == 0) {
            // In legacy path, when site1 is down, diag_corr corresponds to SpSm channel
            SpSm_row = diag_corr;
            std::fill(SmSp_row.begin(), SmSp_row.end(), TenElemT(0));
          }
          if (!SmSp_row.empty()) out["SmSp_row"] = std::move(SmSp_row);
          if (!SpSm_row.empty()) out["SpSm_row"] = std::move(SpSm_row);
        }
      }
      if (row + 1 < ly) {
        // Diagonal (↘) bonds contribute to triangular NN energy
        contractor.InitBTen2(tn, LEFT, row);
        contractor.GrowFullBTen2(tn, RIGHT, row, 2, true);
        auto psi2 = contractor.Trace(tn, {row, 0}, HORIZONTAL);
        auto inv_psi2 = TenElemT(1.0) / psi2;
        for (size_t col = 0; col + 1 < lx; ++col) {
          const SiteIdx ld{row + 1, col};
          const SiteIdx ru{row, col + 1};
          TenElemT eb;
          if (config(ld) == config(ru)) {
            eb = TenElemT(0.25);
          } else {
            TenElemT psi_ex = contractor.ReplaceNNNSiteTrace(tn, {row, col}, LEFTDOWN_TO_RIGHTUP, HORIZONTAL,
                                                     (*split_index_tps)(ld)[config(ru)],
                                                     (*split_index_tps)(ru)[config(ld)]);
            eb = (-0.25 + ComplexConjugate(psi_ex * inv_psi2) * 0.5);
          }
          if (e_ur.size() != 0) {
            e_ur(row, col) = eb;
          }
          energy_total += eb;
          if (col + 2 < lx) contractor.BTen2MoveStep(tn, RIGHT, row);
        }
        contractor.BMPSMoveStep(tn, DOWN, trunc_para);
      }
    }

    // Vertical scan
    contractor.GenerateBMPSApproach(tn, LEFT, trunc_para);
    for (size_t col = 0; col < lx; ++col) {
      contractor.InitBTen(tn, UP, col);
      contractor.GrowFullBTen(tn, DOWN, col, 2, true);
      auto psi = contractor.Trace(tn, {0, col}, VERTICAL);
      auto inv_psi = TenElemT(1.0) / psi;
      psi_list.push_back(psi);
      for (size_t row = 0; row + 1 < ly; ++row) {
        const SiteIdx s1{row, col};
        const SiteIdx s2{row + 1, col};
        TenElemT eb;
        if (config(s1) == config(s2)) {
          eb = TenElemT(0.25);
        } else {
          TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, s1, s2, VERTICAL,
                                                  (*split_index_tps)(s1)[config(s2)],
                                                  (*split_index_tps)(s2)[config(s1)]);
          eb = (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
        }
        if (e_v.size() != 0) {
          e_v(row, col) = eb;
        }
        energy_total += eb;
        if (row + 2 < ly) contractor.BTenMoveStep(tn, DOWN);
      }
      if (col + 1 < lx) contractor.BMPSMoveStep(tn, RIGHT, trunc_para);
    }

    out["energy"] = {energy_total};
    if (e_h.size() != 0) out["bond_energy_h"] = e_h.Extract();
    if (e_v.size() != 0) out["bond_energy_v"] = e_v.Extract();
    if (e_ur.size() != 0) out["bond_energy_ur"] = e_ur.Extract();
    // psi_list is not emitted via registry; Measurer computes PsiSummary separately

    // All-to-all SzSz packed as upper-triangular (i<=j)
    {
      const size_t N = config.size();
      auto full = CalculateSzAll2AllCorrelation_<TenElemT>(config); // length N*N
      if (!full.empty() && full.size() == N * N) {
        std::vector<TenElemT> packed;
        packed.reserve(N * (N + 1) / 2);
        for (size_t i = 0; i < N; ++i) {
          for (size_t j = i; j < N; ++j) {
            packed.push_back(full[i * N + j]);
          }
        }
        out["SzSz_all2all"] = std::move(packed);
      }
    }
    return out;
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    const size_t row_corr_len = lx / 2;
    const size_t bond_len_horizontal = ly * (lx > 0 ? lx - 1 : 0);
    const size_t bond_len_vertical = (ly > 0 ? ly - 1 : 0) * lx;
    const size_t bond_len_diag = (ly > 0 ? ly - 1 : 0) * (lx > 0 ? lx - 1 : 0);
    const size_t site_num = ly * lx;
    return {
        {"energy", "Total energy (scalar)", {}, {}},
        {"spin_z", "Local spin Sz per site (Ly,Lx)", {ly, lx}, {"y","x"}},
        {"SzSz_row", "Row SzSz correlations along middle row (flat)", {row_corr_len}, {"segment"}},
        {"SmSp_row", "Row Sm(i)Sp(j) along middle row (flat)", {row_corr_len}, {"segment"}},
        {"SpSm_row", "Row Sp(i)Sm(j) along middle row (flat)", {row_corr_len}, {"segment"}},
        {"bond_energy_h", "Bond energy on horizontal NN bonds (flat)", {bond_len_horizontal}, {"bond"}},
        {"bond_energy_v", "Bond energy on vertical NN bonds (flat)", {bond_len_vertical}, {"bond"}},
        {"bond_energy_ur", "Bond energy on diagonal (triangular, Up-Right) bonds (flat)", {bond_len_diag}, {"bond"}},
        {"SzSz_all2all", "All-to-all SzSz correlations (upper-tri packed)", {site_num * (site_num + 1) / 2}, {"pair_packed_upper_tri"}}
    };
  }

  // Legacy SampleMeasureImpl removed under registry-only API

 private:
  template<typename TenElemT>
  std::vector<TenElemT> CalculateSzAll2AllCorrelation_(
      const Configuration &config
  ) const;
};

template<typename TenElemT, typename QNT, bool calchols>
TenElemT SpinOneHalfTriHeisenbergSqrPEPS::
CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                      const qlpeps::Configuration &config,
                      TensorNetwork2D<TenElemT, QNT> &tn,
                      BMPSContractor<TenElemT, QNT> &contractor,
                      const qlpeps::BMPSTruncateParams<typename RealTypeTrait<TenElemT>::type> &trunc_para,
                      TensorNetwork2D<TenElemT, QNT> &hole_res,
                      std::vector<TenElemT> &psi_list) {
  TenElemT e(0);
  contractor.GenerateBMPSApproach(tn, UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    contractor.InitBTen(tn, LEFT, row);
    contractor.GrowFullBTen(tn, RIGHT, row, 1, true);
    auto psi = contractor.Trace(tn, {row, 0}, HORIZONTAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(contractor.PunchHole(tn, site1, HORIZONTAL));
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        if (config(site1) == config(site2)) {
          e += 0.25;
        } else {
          TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, HORIZONTAL,
                                                  (*split_index_tps)(site1)[config(site2)],
                                                  (*split_index_tps)(site2)[config(site1)]);
          e += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
        }
        contractor.BTenMoveStep(tn, RIGHT);
      }
    }
    if (row < tn.rows() - 1) {
      //calculate J2 energy
      contractor.InitBTen2(tn, LEFT, row);
      contractor.GrowFullBTen2(tn, RIGHT, row, 2, true);

      for (size_t col = 0; col < tn.cols() - 1; col++) {
        //Calculate diagonal energy contribution
        SiteIdx site1 = {row + 1, col}; //left-down
        SiteIdx site2 = {row, col + 1}; //right-up
        if (config(site1) == config(site2)) {
          e += 0.25;
        } else {
          TenElemT psi_ex = contractor.ReplaceNNNSiteTrace(tn, {row, col},
                                                   LEFTDOWN_TO_RIGHTUP,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          e += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
        }
        contractor.BTen2MoveStep(tn, RIGHT, row);
      }
      contractor.BMPSMoveStep(tn, DOWN, trunc_para);
    }
  }

  //Calculate vertical bond energy contribution
  contractor.GenerateBMPSApproach(tn, LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    contractor.InitBTen(tn, UP, col);
    contractor.GrowFullBTen(tn, DOWN, col, 2, true);
    auto psi = contractor.Trace(tn, {0, col}, VERTICAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        e += 0.25;
      } else {
        TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        e += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
      }
      if (row < tn.rows() - 2) {
        contractor.BTenMoveStep(tn, DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      contractor.BMPSMoveStep(tn, RIGHT, trunc_para);
    }
  }
  return e;
}

/* legacy SampleMeasureImpl fully removed under registry-only API */

template<typename TenElemT>
std::vector<TenElemT> SpinOneHalfTriHeisenbergSqrPEPS::CalculateSzAll2AllCorrelation_(const qlpeps::Configuration &config) const {
  std::vector<TenElemT> res;
  size_t N = config.size();
  res.reserve(N * N);
  for (auto &c1 : config) {
    for (auto &c2 : config) {
      if (c1 == c2) {
        res.push_back(0.25);
      } else {
        res.push_back(-0.25);
      }
    }
  }
  return res;
}
}//qlpeps



#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H
