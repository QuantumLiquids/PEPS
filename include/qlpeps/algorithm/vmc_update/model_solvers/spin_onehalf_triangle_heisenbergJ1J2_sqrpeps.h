/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-12-30
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 Triangle Heisenberg J1-J2 model on square PEPS
*/

#ifndef QLPEPS_QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H
#define QLPEPS_QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"    //ModelEnergySolver
#include "qlpeps/utility/helpers.h"                             //ComplexConjugate
namespace qlpeps {
using namespace qlten;

/**
 * Spin-1/2 J1-J2 Heisenberg Model on Triangular Lattice using Square PEPS
 * 
 * Hamiltonian:
 * $$H = J_1 \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j + J_2 \sum_{\langle\langle i,j \rangle\rangle} \vec{S}_i \cdot \vec{S}_j$$
 * $$  = J_1 \sum_{\langle i,j \rangle} (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j) + J_2 \sum_{\langle\langle i,j \rangle\rangle} (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j)$$
 * 
 * where:
 * - J_1: nearest-neighbor exchange coupling (triangular lattice bonds)
 * - J_2: next-nearest-neighbor exchange coupling (additional competing interactions)
 * - First sum: over all NN bonds in triangular coordination
 * - Second sum: over NNN bonds creating further-range correlations
 * 
 * Bond structure in triangular lattice on square PEPS:
 * J_1 bonds (nearest neighbors):
 * - Horizontal: (i,j) ↔ (i,j+1)
 * - Vertical: (i,j) ↔ (i+1,j)
 * - Diagonal: (i,j) ↔ (i+1,j+1) [↘ direction]
 * 
 * J_2 bonds (next-nearest neighbors):
 * - Skip-horizontal: (i,j) ↔ (i,j+2)  
 * - Skip-vertical: (i,j) ↔ (i+2,j)
 * - Long diagonal: (i,j) ↔ (i+2,j+1), (i+1,j) ↔ (i,j+2) [√5 distance]
 * 
 * This model exhibits rich physics including:
 * - Geometric frustration from triangular lattice
 * - Competing interactions between J_1 and J_2
 * - Possible exotic quantum phases (spin liquids, etc.)
 */
class SpinOneHalfTriJ1J2HeisenbergSqrPEPS : public ModelEnergySolver<SpinOneHalfTriJ1J2HeisenbergSqrPEPS>,
                                            public ModelMeasurementSolver<SpinOneHalfTriJ1J2HeisenbergSqrPEPS> {
 public:
  using ModelEnergySolver<SpinOneHalfTriJ1J2HeisenbergSqrPEPS>::CalEnergyAndHoles;
  using ModelMeasurementSolver<SpinOneHalfTriJ1J2HeisenbergSqrPEPS>::EvaluateObservables;
  using ModelMeasurementSolver<SpinOneHalfTriJ1J2HeisenbergSqrPEPS>::DescribeObservables;
  SpinOneHalfTriJ1J2HeisenbergSqrPEPS(void) = delete;

  SpinOneHalfTriJ1J2HeisenbergSqrPEPS(double j2) : j2_(j2) {}

  template<typename TenElemT, typename QNT, bool calchols = true>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  );

  // Legacy SampleMeasureImpl removed under registry-only API
  
  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample) {
    ObservableMap<TenElemT> out;
    std::vector<TenElemT> psi_list;

    auto &tn = tps_sample->tn;
    const Configuration &config = tps_sample->config;
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
    const size_t ly = tn.rows();
    const size_t lx = tn.cols();

    // site-local: spin_z per site
    {
      std::vector<TenElemT> sz; sz.reserve(config.size());
      for (auto &c : config) { sz.push_back(static_cast<double>(c) - 0.5); }
      out["spin_z"] = std::move(sz);
    }

    // J1 bond energies split by orientation; J2 only enter energy scalar
    std::vector<TenElemT> e_h; if (lx > 1) e_h.reserve(ly * (lx - 1));
    std::vector<TenElemT> e_v; if (ly > 1) e_v.reserve((ly - 1) * lx);
    std::vector<TenElemT> e_diag; if (lx > 1 && ly > 1) e_diag.reserve((ly - 1) * (lx - 1));
    TenElemT energy_total(0);    // accumulate J1 + j2_*J2
    TenElemT energy_j2_total(0); // accumulate J2 only, added with factor j2_

    // Horizontal scan (J1) + middle row correlations
    tn.GenerateBMPSApproach(UP, trunc_para);
    for (size_t row = 0; row < ly; ++row) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 1, true);
      auto psi = tn.Trace({row, 0}, HORIZONTAL);
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
            TenElemT psi_ex = tn.ReplaceNNSiteTrace(s1, s2, HORIZONTAL,
                                                    (*split_index_tps)(s1)[config(s2)],
                                                    (*split_index_tps)(s2)[config(s1)]);
            eb = (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
          }
          e_h.push_back(eb);
          energy_total += eb;
          tn.BTenMoveStep(RIGHT);
        }
      }

      if (row == ly / 2) {
        // SzSz middle row
        std::vector<TenElemT> szsz_row; szsz_row.reserve(lx / 2);
        SiteIdx site1{row, lx / 4};
        double sz1 = static_cast<double>(config(site1)) - 0.5;
        for (size_t i = 1; i <= lx / 2; ++i) {
          SiteIdx site2{row, lx / 4 + i};
          double sz2 = static_cast<double>(config(site2)) - 0.5;
          szsz_row.push_back(sz1 * sz2);
        }
        if (!szsz_row.empty()) out["SzSz_row"] = std::move(szsz_row);

        // XY channels (SmSp_row, SpSm_row)
        std::vector<TenElemT> diag_corr(lx / 2, TenElemT(0));
        tn(site1) = (*split_index_tps)(site1)[1 - config(site1)];
        tn.TruncateBTen(LEFT, lx / 4 + 1);
        tn.GrowBTenStep(LEFT);
        tn.GrowFullBTen(RIGHT, row, lx / 4 + 2, false);
        for (size_t i = 1; i <= lx / 2; ++i) {
          SiteIdx site2{row, lx / 4 + i};
          if (config(site2) == config(site1)) {
            diag_corr[i - 1] = TenElemT(0);
          } else {
            TenElemT psi_ex = tn.ReplaceOneSiteTrace(site2, (*split_index_tps)(site2)[1 - config(site2)], HORIZONTAL);
            diag_corr[i - 1] = ComplexConjugate(psi_ex * inv_psi);
          }
          tn.BTenMoveStep(RIGHT);
        }
        tn(site1) = (*split_index_tps)(site1)[config(site1)];
        std::vector<TenElemT> SmSp_row = diag_corr;
        std::vector<TenElemT> SpSm_row(diag_corr.size(), TenElemT(0));
        if (config(site1) == 0) {
          SpSm_row = diag_corr;
          std::fill(SmSp_row.begin(), SmSp_row.end(), TenElemT(0));
        }
        if (!SmSp_row.empty()) out["SmSp_row"] = std::move(SmSp_row);
        if (!SpSm_row.empty()) out["SpSm_row"] = std::move(SpSm_row);
      }

      if (row + 1 < ly) {
        // Triangular J1 diagonal (↘)
        tn.InitBTen2(LEFT, row);
        tn.GrowFullBTen2(RIGHT, row, 2, true);
        auto psi2 = tn.Trace({row, 0}, HORIZONTAL);
        auto inv_psi2 = TenElemT(1.0) / psi2;
        for (size_t col = 0; col + 1 < lx; ++col) {
          const SiteIdx ld{row + 1, col};
          const SiteIdx ru{row, col + 1};
          TenElemT eb;
          if (config(ld) == config(ru)) {
            eb = TenElemT(0.25);
          } else {
            TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col}, LEFTDOWN_TO_RIGHTUP, HORIZONTAL,
                                                     (*split_index_tps)(ld)[config(ru)],
                                                     (*split_index_tps)(ru)[config(ld)]);
            eb = (-0.25 + ComplexConjugate(psi_ex * inv_psi2) * 0.5);
          }
          e_diag.push_back(eb);
          energy_total += eb;
          if (col + 2 < lx) tn.BTen2MoveStep(RIGHT, row);
        }

        // J2 contribution on the same row window (left-up to right-down)
        for (size_t col = 0; col + 1 < lx; ++col) {
          SiteIdx s1{row, col};
          SiteIdx s2{row + 1, col + 1};
          if (config(s1) == config(s2)) {
            energy_j2_total += TenElemT(0.25);
          } else {
            TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col}, LEFTUP_TO_RIGHTDOWN, HORIZONTAL,
                                                     (*split_index_tps)(s1)[config(s2)],
                                                     (*split_index_tps)(s2)[config(s1)]);
            energy_j2_total += (-0.25 + ComplexConjugate(psi_ex * inv_psi2) * 0.5);
          }
        }
        tn.BMPSMoveStep(DOWN, trunc_para);
      }
    }

    // Vertical scan (J1)
    tn.GenerateBMPSApproach(LEFT, trunc_para);
    for (size_t col = 0; col < lx; ++col) {
      tn.InitBTen(UP, col);
      tn.GrowFullBTen(DOWN, col, 2, true);
      auto psi = tn.Trace({0, col}, VERTICAL);
      auto inv_psi = TenElemT(1.0) / psi;
      psi_list.push_back(psi);
      for (size_t row = 0; row + 1 < ly; ++row) {
        const SiteIdx s1{row, col};
        const SiteIdx s2{row + 1, col};
        TenElemT eb;
        if (config(s1) == config(s2)) {
          eb = TenElemT(0.25);
        } else {
          TenElemT psi_ex = tn.ReplaceNNSiteTrace(s1, s2, VERTICAL,
                                                  (*split_index_tps)(s1)[config(s2)],
                                                  (*split_index_tps)(s2)[config(s1)]);
          eb = (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
        }
        e_v.push_back(eb);
        energy_total += eb;
        if (row + 2 < ly) tn.BTenMoveStep(DOWN);
      }

      // J2 vertical window (sqrt(5) separation)
      if (col + 1 < lx) {
        tn.InitBTen2(UP, col);
        tn.GrowFullBTen2(DOWN, col, 3, true);
        for (size_t row = 0; row + 2 < ly; ++row) {
          const SiteIdx s1{row + 2, col};
          const SiteIdx s2{row, col + 1};
          if (config(s1) == config(s2)) {
            energy_j2_total += TenElemT(0.25);
          } else {
            TenElemT psi_ex = tn.ReplaceSqrt5DistTwoSiteTrace({row, col}, LEFTDOWN_TO_RIGHTUP, VERTICAL,
                                                              (*split_index_tps)(s1)[config(s2)],
                                                              (*split_index_tps)(s2)[config(s1)]);
            energy_j2_total += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
          }
          if (row + 3 < ly) tn.BTen2MoveStep(DOWN, col);
        }
      }
      if (col + 1 < lx) tn.BMPSMoveStep(RIGHT, trunc_para);
    }

    out["energy"] = {energy_total + j2_ * energy_j2_total};
    if (!e_h.empty()) out["bond_energy_h"] = std::move(e_h);
    if (!e_v.empty()) out["bond_energy_v"] = std::move(e_v);
    if (!e_diag.empty()) out["bond_energy_diag"] = std::move(e_diag);
    // psi_list is not emitted via registry; Measurer computes PsiSummary separately

    // All-to-all SzSz packed
    {
      const size_t N = config.size();
      auto full = this->template CalculateSzAll2AllCorrelation_<TenElemT>(config);
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
        {"bond_energy_diag", "Bond energy on diagonal (triangular) bonds (flat)", {bond_len_diag}, {"bond"}},
        {"SzSz_all2all", "All-to-all SzSz correlations (upper-tri packed)", {site_num * (site_num + 1) / 2}, {"pair_packed_upper_tri"}}
    };
  }
 private:
  template<typename TenElemT>
  std::vector<TenElemT> CalculateSzAll2AllCorrelation_(const qlpeps::Configuration &config) const;
  double j2_;
};

template<typename TenElemT, typename QNT, bool calchols>
TenElemT SpinOneHalfTriJ1J2HeisenbergSqrPEPS::
CalEnergyAndHolesImpl(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                      TensorNetwork2D<TenElemT, QNT> &hole_res,
                      std::vector<TenElemT> &psi_list) {
  TenElemT e1(0), e2(0); // energy in J1 and J2 bond respectively
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_list.push_back(tps_sample->amplitude);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL));
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        if (config(site1) == config(site2)) {
          e1 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, HORIZONTAL,
                                                  (*split_index_tps)(site1)[config(site2)],
                                                  (*split_index_tps)(site2)[config(site1)]);
          e1 += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
        }
        tn.BTenMoveStep(RIGHT);
      }
    }
    if (row < tn.rows() - 1) {
      tn.InitBTen2(LEFT, row);
      tn.GrowFullBTen2(RIGHT, row, 2, true);

      for (size_t col = 0; col < tn.cols() - 1; col++) {
        //Calculate diagonal J1 energy contribution
        SiteIdx site1 = {row + 1, col}; //left-down
        SiteIdx site2 = {row, col + 1}; //right-up
        if (config(site1) == config(site2)) {
          e1 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTDOWN_TO_RIGHTUP,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          e1 += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
        }
        //Calculate J2 contribution
        site1 = {row, col}; //left-top
        site2 = {row + 1, col + 1}; //right-bottom
        if (config(site1) == config(site2)) {
          e2 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceNNNSiteTrace({row, col},
                                                   LEFTUP_TO_RIGHTDOWN,
                                                   HORIZONTAL,
                                                   (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                   (*split_index_tps)(site2)[config(site1)]);
          e2 += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
        }

        if (col < tn.cols() - 2) {
          SiteIdx site1 = {row + 1, col}; //left-bottom
          SiteIdx site2 = {row, col + 2}; //right-top
          if (config(site1) == config(site2)) {
            e2 += 0.25;
          } else {
            TenElemT psi_ex = tn.ReplaceSqrt5DistTwoSiteTrace({row, col},
                                                              LEFTDOWN_TO_RIGHTUP,
                                                              HORIZONTAL,
                                                              (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                              (*split_index_tps)(site2)[config(site1)]);
            e2 += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
          }
        }
        tn.BTen2MoveStep(RIGHT, row);
      }
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    tps_sample->amplitude = tn.Trace({0, col}, VERTICAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_list.push_back(tps_sample->amplitude);
    //Calculate vertical bond energy contribution
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      if (config(site1) == config(site2)) {
        e1 += 0.25;
      } else {
        TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, VERTICAL,
                                                (*split_index_tps)(site1)[config(site2)],
                                                (*split_index_tps)(site2)[config(site1)]);
        e1 += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
      }
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.InitBTen2(UP, col);
      tn.GrowFullBTen2(DOWN, col, 3, true);
      //Calculate J2 energy contribution
      for (size_t row = 0; row < tn.rows() - 2; row++) {
        const SiteIdx site1 = {row + 2, col};
        const SiteIdx site2 = {row, col + 1};
        if (config(site1) == config(site2)) {
          e2 += 0.25;
        } else {
          TenElemT psi_ex = tn.ReplaceSqrt5DistTwoSiteTrace({row, col},
                                                            LEFTDOWN_TO_RIGHTUP,
                                                            VERTICAL,
                                                            (*split_index_tps)(site1)[config(site2)],  //the tensor at left
                                                            (*split_index_tps)(site2)[config(site1)]);
          e2 += (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
        }
        if ((int) row < (int) tn.rows() - 3) {
          tn.BTen2MoveStep(DOWN, col);
        }
      }
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  WaveFunctionAmplitudeConsistencyCheck(psi_list, 0.03);
  return e1 + j2_ * e2;
}

/* legacy SampleMeasureImpl fully removed under registry-only API */

template<typename TenElemT>
std::vector<TenElemT> SpinOneHalfTriJ1J2HeisenbergSqrPEPS::CalculateSzAll2AllCorrelation_(const qlpeps::Configuration &config) const {
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
#endif //QLPEPS_QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERGJ1J2_SQRPEPS_H
