/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-01-25
*
* Description: QuantumLiquids/PEPS project.
* PBC version of spin-1/2 J1-J2 XXZ model solver using TRGContractor.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_PBC_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_PBC_H

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
 * @brief Spin-1/2 J1-J2 XXZ model on a square lattice (PBC, contracted by TRG).
 *
 * Hamiltonian:
 * \f[
 *   H = \sum_{\langle i,j \rangle} \left( J_{z1} S^z_i S^z_j + J_{xy1} (S^x_i S^x_j + S^y_i S^y_j) \right)
 *     + \sum_{\langle\langle i,j \rangle\rangle} \left( J_{z2} S^z_i S^z_j + J_{xy2} (S^x_i S^x_j + S^y_i S^y_j) \right)
 *     - h_{00} S^z_{00}
 * \f]
 *
 * For J1-only, set J2 = 0. In that case, the implementation skips NNN traversals
 * to keep TRG cost comparable to the pure NN model.
 */
class SquareSpinOneHalfJ1J2XXZModelPBC
    : public ModelEnergySolver<SquareSpinOneHalfJ1J2XXZModelPBC>,
      public ModelMeasurementSolver<SquareSpinOneHalfJ1J2XXZModelPBC> {
 public:
  SquareSpinOneHalfJ1J2XXZModelPBC(void) = delete;

  ///< Isotropic J1-J2 Heisenberg (J1=1, J2=j2)
  SquareSpinOneHalfJ1J2XXZModelPBC(double j2)
      : jz1_(1.0), jxy1_(1.0), jz2_(j2), jxy2_(j2), pinning00_(0.0) {}

  ///< Generic XXZ J1-J2 with pinning field at (0,0)
  SquareSpinOneHalfJ1J2XXZModelPBC(double jz1, double jxy1, double jz2, double jxy2, double pinning_field00)
      : jz1_(jz1), jxy1_(jxy1), jz2_(jz2), jxy2_(jxy2), pinning00_(pinning_field00) {}

  using ModelEnergySolver::CalEnergyAndHoles;
  using ModelMeasurementSolver<SquareSpinOneHalfJ1J2XXZModelPBC>::EvaluateObservables;
  using ModelMeasurementSolver<SquareSpinOneHalfJ1J2XXZModelPBC>::DescribeObservables;

  template<typename TenElemT, typename QNT, bool calchols, typename ComponentT>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      ComponentT *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  ) {
    auto &tn = tps_sample->tn;
    auto &contractor = tps_sample->contractor;
    const auto &config = tps_sample->config;
    const auto &trunc_para = tps_sample->trun_para;
    const size_t lx = tn.cols();
    const size_t ly = tn.rows();

    if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) {
      throw std::invalid_argument("SquareSpinOneHalfJ1J2XXZModelPBC: requires periodic TensorNetwork2D.");
    }

    contractor.SetTruncateParams(trunc_para);

    const TenElemT psi = contractor.Trace(tn);
    psi_list.push_back(psi);
    const TenElemT inv_psi = TenElemT(1.0) / psi;

    if constexpr (calchols) {
      const auto all_holes = contractor.PunchAllHoles(tn);
      for (size_t row = 0; row < ly; ++row) {
        for (size_t col = 0; col < lx; ++col) {
          const SiteIdx site{row, col};
          hole_res(site) = Dag(all_holes(site));
        }
      }
    }

    TenElemT energy(0);

    for (size_t row = 0; row < ly; ++row) {
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx site1{row, col};
        const SiteIdx site_right{row, (col + 1) % lx};
        const SiteIdx site_down{(row + 1) % ly, col};

        energy += EvaluateCouplingEnergy_<TenElemT, QNT>(
            site1, site_right, config, *sitps, contractor, inv_psi, jz1_, jxy1_);
        energy += EvaluateCouplingEnergy_<TenElemT, QNT>(
            site1, site_down, config, *sitps, contractor, inv_psi, jz1_, jxy1_);

        if (HasNNNInteraction_()) {
          const SiteIdx site_dr{(row + 1) % ly, (col + 1) % lx};
          const SiteIdx site_dl{(row + 1) % ly, (col + lx - 1) % lx};
          energy += EvaluateCouplingEnergy_<TenElemT, QNT>(
              site1, site_dr, config, *sitps, contractor, inv_psi, jz2_, jxy2_);
          energy += EvaluateCouplingEnergy_<TenElemT, QNT>(
              site1, site_dl, config, *sitps, contractor, inv_psi, jz2_, jxy2_);
        }
      }
    }

    energy += EvaluateTotalOnsiteEnergy_<TenElemT>(config);
    return energy;
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
      throw std::invalid_argument("SquareSpinOneHalfJ1J2XXZModelPBC: requires periodic TensorNetwork2D.");
    }

    contractor.SetTruncateParams(trunc_para);

    const TenElemT psi = contractor.Trace(tn);
    const TenElemT inv_psi = TenElemT(1.0) / psi;

    std::vector<TenElemT> psi_list;
    psi_list.reserve(1);
    psi_list.push_back(psi);

    std::vector<TenElemT> spin_z;
    spin_z.reserve(config.size());
    for (auto &s : config) {
      spin_z.push_back(static_cast<double>(s) - 0.5);
    }

    ObservableMatrix<TenElemT> e_h;
    ObservableMatrix<TenElemT> e_v;
    ObservableMatrix<TenElemT> e_dr;
    ObservableMatrix<TenElemT> e_ur;
    if (ly > 0 && lx > 0) {
      e_h.Resize(ly, lx);
      e_v.Resize(ly, lx);
      if (HasNNNInteraction_()) {
        e_dr.Resize(ly, lx);
        e_ur.Resize(ly, lx);
      }
    }

    TenElemT energy(0);
    for (size_t row = 0; row < ly; ++row) {
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx site1{row, col};

        const SiteIdx site_right{row, (col + 1) % lx};
        TenElemT eb_h = EvaluateCouplingEnergy_<TenElemT, QNT>(
            site1, site_right, config, *sitps, contractor, inv_psi, jz1_, jxy1_);
        if (e_h.size() != 0) {
          e_h(site1) = eb_h;
        }
        energy += eb_h;

        const SiteIdx site_down{(row + 1) % ly, col};
        TenElemT eb_v = EvaluateCouplingEnergy_<TenElemT, QNT>(
            site1, site_down, config, *sitps, contractor, inv_psi, jz1_, jxy1_);
        if (e_v.size() != 0) {
          e_v(site1) = eb_v;
        }
        energy += eb_v;

        if (HasNNNInteraction_()) {
          const SiteIdx site_dr{(row + 1) % ly, (col + 1) % lx};
          TenElemT eb_dr = EvaluateCouplingEnergy_<TenElemT, QNT>(
              site1, site_dr, config, *sitps, contractor, inv_psi, jz2_, jxy2_);
          if (e_dr.size() != 0) {
            e_dr(site1) = eb_dr;
          }
          energy += eb_dr;

          const SiteIdx site_dl{(row + 1) % ly, (col + lx - 1) % lx};
          TenElemT eb_ur = EvaluateCouplingEnergy_<TenElemT, QNT>(
              site1, site_dl, config, *sitps, contractor, inv_psi, jz2_, jxy2_);
          if (e_ur.size() != 0) {
            e_ur(site1) = eb_ur;
          }
          energy += eb_ur;
        }
      }
    }

    energy += EvaluateTotalOnsiteEnergy_<TenElemT>(config);

    out["energy"] = {energy};
    out["spin_z"] = std::move(spin_z);
    if (e_h.size() != 0) out["bond_energy_h"] = e_h.Extract();
    if (e_v.size() != 0) out["bond_energy_v"] = e_v.Extract();
    if (e_dr.size() != 0) out["bond_energy_dr"] = e_dr.Extract();
    if (e_ur.size() != 0) out["bond_energy_ur"] = e_ur.Extract();

    auto psi_summary = this->template ComputePsiSummary<TenElemT>(psi_list);
    this->template SetLastPsiSummary<TenElemT>(psi_summary.psi_mean, psi_summary.psi_rel_err);

    return out;
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    // NOTE: Compared to OBC NN/NNN solvers, long-range spin correlations are not implemented for PBC/TRG.
    std::vector<ObservableMeta> out = {
        {"energy", "Total energy (scalar)", {}, {}},
        {"spin_z", "Local spin Sz per site (Ly,Lx)", {ly, lx}, {"y", "x"}},
        {"bond_energy_h",
         "Bond energy on horizontal NN bonds (periodic wrap included)",
         {ly, lx}, {"bond_y", "bond_x"}},
        {"bond_energy_v",
         "Bond energy on vertical NN bonds (periodic wrap included)",
         {ly, lx}, {"bond_y", "bond_x"}},
    };
    if (HasNNNInteraction_()) {
      out.push_back({"bond_energy_dr",
                     "Bond energy on down-right NNN bonds (periodic wrap included)",
                     {ly, lx}, {"bond_y", "bond_x"}});
      out.push_back({"bond_energy_ur",
                     "Bond energy on down-left NNN bonds (periodic wrap included)",
                     {ly, lx}, {"bond_y", "bond_x"}});
    }
    return out;
  }

 private:
  double jz1_;
  double jxy1_;
  double jz2_;
  double jxy2_;
  double pinning00_;

  [[nodiscard]] bool HasNNNInteraction_() const {
    return (jz2_ != 0.0) || (jxy2_ != 0.0);
  }

  template<typename TenElemT>
  TenElemT EvaluateTotalOnsiteEnergy_(const Configuration &config) const {
    if (pinning00_ == 0.0) {
      return TenElemT(0);
    }
    const size_t c00 = config({0, 0});
    const double sz00 = static_cast<double>(c00) - 0.5;
    return TenElemT(-pinning00_ * sz00);
  }

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateCouplingEnergy_(
      const SiteIdx &site1,
      const SiteIdx &site2,
      const Configuration &config,
      const SplitIndexTPS<TenElemT, QNT> &sitps,
      TRGContractor<TenElemT, QNT> &contractor,
      const TenElemT &inv_psi,
      const double jz,
      const double jxy
  ) {
    using Tensor = qlten::QLTensor<TenElemT, QNT>;

    const size_t c1 = config(site1);
    const size_t c2 = config(site2);
    const double sz1 = static_cast<double>(c1) - 0.5;
    const double sz2 = static_cast<double>(c2) - 0.5;

    TenElemT energy_diag = TenElemT(jz * sz1 * sz2);
    TenElemT energy_offdiag(0);
    if (c1 != c2) {
      std::vector<std::pair<SiteIdx, Tensor>> replacements{
          {site1, sitps(site1)[c2]},
          {site2, sitps(site2)[c1]}
      };
      const TenElemT psi_ex = contractor.EvaluateReplacement(replacements);
      const TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
      energy_offdiag = TenElemT(0.5 * jxy) * ratio;
    }

    return energy_diag + energy_offdiag;
  }
};

}  // namespace qlpeps

#endif  // QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_SQUAREJ1J2_PBC_H
