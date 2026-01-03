/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-28
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 Triangle Heisenberg model on square PEPS
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H

#include <algorithm>

#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/structure_factor_measurement_mixin.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_xxz_model.h" // MeasureSpinOneHalfOffDiagOrderInRow
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps/bmps_contractor.h" //BMPSContractor

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
class SpinOneHalfTriHeisenbergSqrPEPS : 
    public SquareNNNModelEnergySolver<SpinOneHalfTriHeisenbergSqrPEPS>,
    public SquareNNNModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS>,
    public StructureFactorMeasurementMixin<SpinOneHalfTriHeisenbergSqrPEPS> {
 public:
  static constexpr bool requires_spin_sz_measurement = true;
  static constexpr bool requires_density_measurement = false;

  SpinOneHalfTriHeisenbergSqrPEPS(void) = default;

  using SquareNNNModelEnergySolver<SpinOneHalfTriHeisenbergSqrPEPS>::CalEnergyAndHoles;
  using SquareNNNModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS>::EvaluateObservables;
  using SquareNNNModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS>::DescribeObservables;
  using StructureFactorMeasurementMixin<SpinOneHalfTriHeisenbergSqrPEPS>::MeasureStructureFactor;

  // Implement interfaces required by SquareNNNModelMeasurementSolver
  [[nodiscard]] inline double CalSpinSzImpl(const size_t config_val) const { return double(config_val) - 0.5; }

  // Implement interfaces required by SquareNNNModelEnergySolver
  [[nodiscard]] inline double EvaluateTotalOnsiteEnergy(const Configuration &) const {
    return 0.0; // No onsite field in this model
  }

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(const SiteIdx site1, const SiteIdx site2,
                              const size_t c1, const size_t c2,
                              const BondOrientation orient,
                              TensorNetwork2D<TenElemT, QNT> &tn,
                              BMPSContractor<TenElemT, QNT> &contractor,
                              const std::vector<QLTensor<TenElemT, QNT>> &split_ts1,
                              const std::vector<QLTensor<TenElemT, QNT>> &split_ts2,
                              const TenElemT inv_psi) const {
    if (c1 == c2) {
      return TenElemT(0.25);
    } else {
      TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                                      split_ts1[c2],
                                                      split_ts2[c1]);
      return (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
    }
  }

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateNNNEnergy(const SiteIdx site1, const SiteIdx site2,
                             const size_t c1, const size_t c2,
                             const DIAGONAL_DIR dir,
                             TensorNetwork2D<TenElemT, QNT> &tn,
                             BMPSContractor<TenElemT, QNT> &contractor,
                             const std::vector<QLTensor<TenElemT, QNT>> &split_ts1,
                             const std::vector<QLTensor<TenElemT, QNT>> &split_ts2,
                             const TenElemT inv_psi) const {
    // Only calculate for the existing diagonal direction in triangular lattice (LeftDown to RightUp)
    if (dir != LEFTDOWN_TO_RIGHTUP) {
      return TenElemT(0);
    }

    if (c1 == c2) {
      return TenElemT(0.25);
    } else {
      // BondTraversalMixin provides (site1, site2) as the two endpoints and a diagonal direction.
      // For LEFTDOWN_TO_RIGHTUP, the contraction kernel expects the *left-up* corner of the plaquette.
      // See SquareSpinOneHalfXXZModelMixIn::EvaluateNNNEnergy for the same coordinate convention.
      const SiteIdx left_up_site{site2.row(), site1.col()};
      TenElemT psi_ex = contractor.ReplaceNNNSiteTrace(tn, left_up_site, dir, HORIZONTAL,
                                                       split_ts1[c2], split_ts2[c1]);
      return (-0.25 + ComplexConjugate(psi_ex * inv_psi) * 0.5);
    }
  }

  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample) {
    // Use the generic traversal (same as tJ/XXZ models) and only append cheap, config-based observables.
    ObservableMap<TenElemT> out =
        this->SquareNNNModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS>::EvaluateObservables(
            split_index_tps, tps_sample
        );
    
    // Expensive S+S- Structure Factor (controlled by flag)
    this->MeasureStructureFactor(tps_sample->tn, split_index_tps, tps_sample->contractor, tps_sample->config, out, tps_sample->trun_para);

    // Keep legacy public API: this model historically exposed only the interacting diagonal as bond_energy_ur.
    out.erase("bond_energy_dr");

    const auto &tn = tps_sample->tn;
    const auto &config = tps_sample->config;
    const size_t ly = tn.rows();
    const size_t lx = tn.cols();
    const size_t N = config.size();

    // Middle-row SzSz along row (purely config-based)
    if (lx > 0 && ly > 0) {
      const size_t row = ly / 2;
      std::vector<TenElemT> szsz_row;
      szsz_row.reserve(lx / 2);
      const SiteIdx site1{row, lx / 4};
      const double sz1 = CalSpinSzImpl(config(site1));
      for (size_t i = 1; i <= lx / 2; ++i) {
        const SiteIdx site2{row, lx / 4 + i};
        const double sz2 = CalSpinSzImpl(config(site2));
        szsz_row.push_back(static_cast<TenElemT>(sz1 * sz2));
      }
      if (!szsz_row.empty()) out["SzSz_row"] = std::move(szsz_row);
    }

    // All-to-all SzSz: directly compute upper-triangular (i <= j)
    std::vector<double> sz_vals;
    sz_vals.reserve(N);
    for (const auto &c : config) { sz_vals.push_back(CalSpinSzImpl(c)); }
    std::vector<TenElemT> szsz;
    szsz.reserve(N * (N + 1) / 2);
    for (size_t i = 0; i < N; ++i) {
      const double szi = sz_vals[i];
      for (size_t j = i; j < N; ++j) {
        szsz.push_back(static_cast<TenElemT>(szi * sz_vals[j]));
      }
    }
    out["SzSz_all2all"] = std::move(szsz);
    return out;
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    auto desc =
        this->SquareNNNModelMeasurementSolver<SpinOneHalfTriHeisenbergSqrPEPS>::DescribeObservables(ly, lx);
    // Keep legacy public API: remove bond_energy_dr metadata
    desc.erase(std::remove_if(desc.begin(), desc.end(),
                              [](const ObservableMeta &m) { return m.key == "bond_energy_dr"; }),
               desc.end());

    const size_t row_corr_len = lx / 2;
    const size_t site_num = ly * lx;
    desc.push_back({"SzSz_row", "Row SzSz correlations along middle row (flat)", {row_corr_len}, {"segment"}});
    desc.push_back({"SmSp_row", "Row Sm(i)Sp(j) along middle row (flat)", {row_corr_len}, {"segment"}});
    desc.push_back({"SpSm_row", "Row Sp(i)Sm(j) along middle row (flat)", {row_corr_len}, {"segment"}});
    desc.push_back({"SzSz_all2all", "All-to-all SzSz correlations (upper-tri packed)", {site_num * (site_num + 1) / 2},
                    {"pair_packed_upper_tri"}});
    desc.push_back({"SpSm_cross", "All-to-all SpSm structure factor correlations (sparse format)", {0}, {"y1", "x1", "y2", "x2", "val"}});
    return desc;
  }

  /**
   * @brief Get the tensor at site (row, col) for a given spin state.
   * Required by StructureFactorMeasurementMixin.
   */
  template<typename TenElemT, typename QNT>
  qlten::QLTensor<TenElemT, QNT> GetSiteTensor(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      size_t row, size_t col, size_t spin_val) const {
      return (*split_index_tps)({row, col})[spin_val];
  }

  template<typename TenElemT, typename QNT>
  void EvaluateOffDiagOrderInRow(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                                 TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
                                 const size_t row,
                                 const TenElemT inv_psi,
                                 ObservableMap<TenElemT> &out) const {
    auto &tn = tps_sample->tn;
    auto &contractor = tps_sample->contractor;
    const auto &config = tps_sample->config;
    const size_t lx = tn.cols();
    const size_t ly = tn.rows();
    if (row != ly / 2) {
      return;
    }

    // Reuse the generic off-diagonal measurement routine
    std::vector<TenElemT> diag_corr;
    diag_corr.reserve(lx / 2);
    MeasureSpinOneHalfOffDiagOrderInRow(split_index_tps, tn, contractor, diag_corr, inv_psi, config, row);

    // Split into S^-S^+ and S^+S^- channels based on site1 spin state
    const SiteIdx site1{row, lx / 4};
    std::vector<TenElemT> SmSp_row = diag_corr;
    std::vector<TenElemT> SpSm_row(diag_corr.size(), TenElemT(0));
    if (config(site1) == 0) {
      // When site1 is spin-down, diag_corr corresponds to S^+S^- channel
      SpSm_row = diag_corr;
      std::fill(SmSp_row.begin(), SmSp_row.end(), TenElemT(0));
    }
    if (!SmSp_row.empty()) out["SmSp_row"] = std::move(SmSp_row);
    if (!SpSm_row.empty()) out["SpSm_row"] = std::move(SpSm_row);
  }

};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_TRIANGLE_HEISENBERG_SQRPEPS_H
