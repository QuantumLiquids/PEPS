/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-12-15
*
* Description: QuantumLiquids/PEPS project.
* PBC version of transverse-field Ising model solver using TRGContractor.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_PBC_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_PBC_H

#include <complex>
#include <cmath>
#include <vector>

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/observable_matrix.h"                     // ObservableMatrix
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/two_dim_tn/tensor_network_2d/trg/trg_contractor.h"    // TRGContractor

namespace qlpeps {

/**
 * @brief Transverse-field Ising model on a square lattice (PBC, contracted by TRG).
 *
 * Hamiltonian convention (same as the OBC solver):
 * \f[
 *   H = -\sum_{\langle i,j\rangle} \sigma_i^z \sigma_j^z - h \sum_i \sigma_i^x.
 * \f]
 *
 * Notes on 2x2 PBC:
 * - If diagonal bonds are accumulated by "for every site connect to right/down neighbor",
 *   then for 2x2 the same physical bond is counted twice (the user explicitly wants this
 *   "same-as-larger-system" coding convention).
 *
 * Implementation status:
 * - Energy + gradient holes are supported for all supported PBC sizes using
 *   `TRGContractor::PunchAllHoles` batch API.
 * - Supported sizes: 2x2, 3x3, and N=2^k or 3*2^k periodic torus.
 */
class TransverseFieldIsingSquarePBC : public ModelEnergySolver<TransverseFieldIsingSquarePBC>,
                                      public ModelMeasurementSolver<TransverseFieldIsingSquarePBC> {
 public:
  TransverseFieldIsingSquarePBC(void) = delete;
  explicit TransverseFieldIsingSquarePBC(double h) : h_(h) {}

  using ModelEnergySolver::CalEnergyAndHoles;
  using ModelMeasurementSolver<TransverseFieldIsingSquarePBC>::EvaluateObservables;
  using ModelMeasurementSolver<TransverseFieldIsingSquarePBC>::DescribeObservables;

  template<typename TenElemT, typename QNT, bool calchols, typename ComponentT>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      ComponentT *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  ) {
    using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
    (void)RealT{};

    auto &tn = tps_sample->tn;
    auto &contractor = tps_sample->contractor;
    const auto &config = tps_sample->config;
    const auto &trunc_para = tps_sample->trun_para;
    const size_t lx = tn.cols();
    const size_t ly = tn.rows();

    // Enforce PBC semantics at the data level.
    if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) {
      throw std::invalid_argument("TransverseFieldIsingSquarePBC: requires periodic TensorNetwork2D.");
    }

    // Keep TRG truncation params in sync with component.
    // Contracting assumes: tn, contractor cache, and any cached amplitude are in sync
    // after each MC CommitTrial (and truncation params are unchanged).
    contractor.SetTruncateParams(trunc_para);
    // Init already performed in TPSWaveFunctionComponent.

    const TenElemT psi = contractor.Trace(tn);
    psi_list.push_back(psi);
    const TenElemT inv_psi = TenElemT(1.0) / psi;

    TenElemT energy_diag(0);
    // PBC diagonal zz energy: sum over all sites and connect to right and down neighbor.
    // For 2x2 this double-counts the same physical bond (intentionally).
    for (size_t row = 0; row < ly; ++row) {
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx s{row, col};
        const SiteIdx sr{row, (col + 1) % lx};
        const SiteIdx sd{(row + 1) % ly, col};
        energy_diag += (config(s) == config(sr)) ? TenElemT(-1) : TenElemT(+1);
        energy_diag += (config(s) == config(sd)) ? TenElemT(-1) : TenElemT(+1);
      }
    }

    // Batch compute all holes at once (much more efficient than per-site PunchHole)
    const auto all_holes = contractor.PunchAllHoles(tn);

    TenElemT energy_ex(0);
    for (size_t row = 0; row < ly; ++row) {
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx site{row, col};

        const auto& hole = all_holes(site);
        if constexpr (calchols) {
          hole_res(site) = Dag(hole);  // matches exact-sum evaluator convention: hole_dag encodes ∂_{θ*}Ψ*
        }

        const size_t s = config(site);
        const auto &alt_tensor = (*sitps)(site)[1 - s];

        qlten::QLTensor<TenElemT, QNT> out;
        qlten::Contract(&hole, {0, 1, 2, 3}, &alt_tensor, {0, 1, 2, 3}, &out);
        const TenElemT psi_ex = out();

        const TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
        energy_ex += (-h_) * ratio;
      }
    }

    return energy_diag + energy_ex;
  }

  template<typename TenElemT, typename QNT, typename ComponentT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      ComponentT *tps_sample
  ) {
    ObservableMap<TenElemT> out;

    auto &tn = tps_sample->tn;
    auto &contractor = tps_sample->contractor;
    const auto &config = tps_sample->config;
    const auto &trunc_para = tps_sample->trun_para;
    const size_t lx = tn.cols();
    const size_t ly = tn.rows();

    if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) {
      throw std::invalid_argument("TransverseFieldIsingSquarePBC: requires periodic TensorNetwork2D.");
    }

    contractor.SetTruncateParams(trunc_para);

    const TenElemT psi = contractor.Trace(tn);
    const TenElemT inv_psi = TenElemT(1.0) / psi;

    std::vector<TenElemT> psi_list;
    psi_list.reserve(1);
    psi_list.push_back(psi);

    // Local spin Sz
    std::vector<TenElemT> spin_z;
    spin_z.reserve(config.size());
    for (auto &s : config) {
      spin_z.push_back(static_cast<double>(s) - 0.5);
    }

    ObservableMatrix<TenElemT> sigma_x_mat;
    if (ly > 0 && lx > 0) {
      sigma_x_mat.Resize(ly, lx);
    }

    TenElemT energy_diag(0);
    for (size_t row = 0; row < ly; ++row) {
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx s{row, col};
        const SiteIdx sr{row, (col + 1) % lx};
        const SiteIdx sd{(row + 1) % ly, col};
        energy_diag += (config(s) == config(sr)) ? TenElemT(-1) : TenElemT(+1);
        energy_diag += (config(s) == config(sd)) ? TenElemT(-1) : TenElemT(+1);
      }
    }

    const auto all_holes = contractor.PunchAllHoles(tn);
    TenElemT energy_ex(0);
    for (size_t row = 0; row < ly; ++row) {
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx site{row, col};
        const size_t s = config(site);
        const auto &alt_tensor = (*sitps)(site)[1 - s];

        qlten::QLTensor<TenElemT, QNT> out_ten;
        const auto &hole = all_holes(site);
        qlten::Contract(&hole, {0, 1, 2, 3}, &alt_tensor, {0, 1, 2, 3}, &out_ten);
        const TenElemT psi_ex = out_ten();
        const TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);

        energy_ex += (-h_) * ratio;
        if (sigma_x_mat.size() != 0) {
          sigma_x_mat(site) = (h_ != 0.0) ? ratio : TenElemT(0);
        }
      }
    }

    std::vector<TenElemT> two_point;
    if (ly > 0 && lx > 0) {
      const size_t row = ly / 2;
      const SiteIdx site1{row, lx / 4};
      const double sz1 = config(site1) - 0.5;
      two_point.reserve(lx / 2);
      for (size_t i = 1; i <= lx / 2; ++i) {
        const SiteIdx site2{row, lx / 4 + i};
        const double sz2 = config(site2) - 0.5;
        two_point.push_back(sz1 * sz2);
      }
    }

    out["energy"] = {energy_diag + energy_ex};
    out["spin_z"] = std::move(spin_z);
    if (sigma_x_mat.size() != 0) out["sigma_x"] = sigma_x_mat.Extract();
    if (!two_point.empty()) out["SzSz_row"] = std::move(two_point);

    auto psi_summary = this->template ComputePsiSummary<TenElemT>(psi_list);
    this->template SetLastPsiSummary<TenElemT>(psi_summary.psi_mean, psi_summary.psi_rel_err);

    return out;
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    return {
        {"energy", "Total energy (scalar)", {}, {}},
        {"spin_z", "Local spin Sz per site (Ly,Lx)", {ly, lx}, {"y", "x"}},
        {"sigma_x", "Transverse magnetisation per site (Ly,Lx)", {ly, lx}, {"y", "x"}},
        {"SzSz_row", "SzSz correlations along middle row (flat)", {lx / 2}, {"segment"}}
    };
  }

 private:
  double h_;
};

} // namespace qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_PBC_H
