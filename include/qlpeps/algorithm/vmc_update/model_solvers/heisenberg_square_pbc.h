/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-01-23
*
* Description: QuantumLiquids/PEPS project.
* PBC version of spin-1/2 Heisenberg (XXZ) model solver using TRGContractor.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_HEISENBERG_SQUARE_PBC_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_HEISENBERG_SQUARE_PBC_H

#include <complex>
#include <cmath>
#include <vector>

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/two_dim_tn/tensor_network_2d/trg/trg_contractor.h"    // TRGContractor

namespace qlpeps {

/**
 * @brief Spin-1/2 Heisenberg (XXZ) model on a square lattice (PBC, contracted by TRG).
 *
 * Hamiltonian:
 * \f[
 *   H = \sum_{\langle i,j \rangle} \left( J_z S^z_i S^z_j + J_{xy} (S^x_i S^x_j + S^y_i S^y_j) \right)
 * \f]
 *
 * where \f$S^{\alpha}_i\f$ are spin-1/2 operators.
 *
 * Using the identity \f$S^x_i S^x_j + S^y_i S^y_j = \frac{1}{2}(S^+_i S^-_j + S^-_i S^+_j)\f$,
 * the Hamiltonian can be rewritten as:
 * \f[
 *   H = \sum_{\langle i,j \rangle} \left( J_z S^z_i S^z_j + \frac{J_{xy}}{2} (S^+_i S^-_j + S^-_i S^+_j) \right)
 * \f]
 *
 * For isotropic Heisenberg model: J_z = J_xy = J.
 *
 * Implementation notes:
 * - Diagonal term \f$S^z_i S^z_j\f$: computed directly from configuration (\f$\pm 1/4\f$)
 * - Off-diagonal term \f$S^+_i S^-_j + S^-_i S^+_j\f$: requires computing amplitude ratio
 *   for the exchanged configuration using TRGContractor::BeginTrialWithReplacement
 *
 * @note The exchange term only contributes when the two sites have different spins.
 */
class HeisenbergSquarePBC : public ModelEnergySolver<HeisenbergSquarePBC>,
                            public ModelMeasurementSolver<HeisenbergSquarePBC> {
 public:
  HeisenbergSquarePBC(void) = delete;

  /// Isotropic Heisenberg model with J = 1
  HeisenbergSquarePBC(double j) : jz_(j), jxy_(j) {}

  /// XXZ model with separate Jz and Jxy couplings
  HeisenbergSquarePBC(double jz, double jxy) : jz_(jz), jxy_(jxy) {}

  using ModelEnergySolver::CalEnergyAndHoles;
  using ModelMeasurementSolver<HeisenbergSquarePBC>::EvaluateObservables;
  using ModelMeasurementSolver<HeisenbergSquarePBC>::DescribeObservables;

  template<typename TenElemT, typename QNT, bool calchols, typename ComponentT>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      ComponentT *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  ) {
    using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
    using Tensor = qlten::QLTensor<TenElemT, QNT>;
    (void)RealT{};

    auto &tn = tps_sample->tn;
    auto &contractor = tps_sample->contractor;
    const auto &config = tps_sample->config;
    const auto &trunc_para = tps_sample->trun_para;
    const size_t lx = tn.cols();
    const size_t ly = tn.rows();

    // Enforce PBC semantics at the data level.
    if (tn.GetBoundaryCondition() != BoundaryCondition::Periodic) {
      throw std::invalid_argument("HeisenbergSquarePBC: requires periodic TensorNetwork2D.");
    }

    // Keep TRG truncation params in sync with component.
    contractor.SetTruncateParams(trunc_para);
    contractor.Init(tn);

    const TenElemT psi = contractor.Trace(tn);
    psi_list.push_back(psi);
    const TenElemT inv_psi = TenElemT(1.0) / psi;

    // Batch compute all holes at once (for gradient computation)
    if constexpr (calchols) {
      const auto all_holes = contractor.PunchAllHoles(tn);
      for (size_t row = 0; row < ly; ++row) {
        for (size_t col = 0; col < lx; ++col) {
          const SiteIdx site{row, col};
          // Dag(hole) matches exact-sum evaluator convention: hole_dag encodes ∂_{θ*}Ψ*
          hole_res(site) = Dag(all_holes(site));
        }
      }
    }

    // Compute energy: iterate over all nearest-neighbor bonds
    // PBC: each site connects to right neighbor and down neighbor
    TenElemT energy(0);

    for (size_t row = 0; row < ly; ++row) {
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx site1{row, col};

        // Horizontal bond: site1 -- site1_right
        {
          const SiteIdx site2{row, (col + 1) % lx};
          energy += EvaluateBondEnergy_<TenElemT, QNT>(
              site1, site2, config, *sitps, contractor, inv_psi);
        }

        // Vertical bond: site1 -- site1_down
        {
          const SiteIdx site2{(row + 1) % ly, col};
          energy += EvaluateBondEnergy_<TenElemT, QNT>(
              site1, site2, config, *sitps, contractor, inv_psi);
        }
      }
    }

    return energy;
  }

 private:
  double jz_;   // Ising coupling (S^z S^z)
  double jxy_;  // XY coupling (S^x S^x + S^y S^y)

  /**
   * @brief Evaluate energy contribution from a single bond.
   *
   * For a bond (i, j):
   * - Diagonal: Jz * <σ|S^z_i S^z_j|σ> = Jz * (±1/4)
   * - Off-diagonal (only when σ(i) ≠ σ(j)):
   *   Jxy * <σ'|S^x S^x + S^y S^y|σ> * ψ(σ')/ψ(σ) = Jxy * (1/2) * ratio
   *
   * where σ' is the configuration with spins at i,j exchanged.
   */
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy_(
      const SiteIdx &site1,
      const SiteIdx &site2,
      const Configuration &config,
      const SplitIndexTPS<TenElemT, QNT> &sitps,
      TRGContractor<TenElemT, QNT> &contractor,
      const TenElemT &inv_psi
  ) {
    using Tensor = qlten::QLTensor<TenElemT, QNT>;

    const size_t c1 = config(site1);
    const size_t c2 = config(site2);

    // Spin values: 0 -> down (-1/2), 1 -> up (+1/2)
    const double sz1 = static_cast<double>(c1) - 0.5;
    const double sz2 = static_cast<double>(c2) - 0.5;

    // Diagonal term: Jz * S^z_i * S^z_j
    TenElemT energy_diag = TenElemT(jz_ * sz1 * sz2);

    // Off-diagonal term: only contributes when spins are different
    TenElemT energy_offdiag(0);
    if (c1 != c2) {
      // Exchange the spins: compute ψ(σ') where σ' has site1 and site2 swapped
      std::vector<std::pair<SiteIdx, Tensor>> replacements{
          {site1, sitps(site1)[c2]},  // site1 gets spin c2
          {site2, sitps(site2)[c1]}   // site2 gets spin c1
      };

      // Use EvaluateReplacement instead of BeginTrialWithReplacement:
      // - Pure read-only, no Trial object allocation
      // - No layer_updates saved (not needed for energy calculation)
      const TenElemT psi_ex = contractor.EvaluateReplacement(replacements);

      // Amplitude ratio: ψ*(σ')/ψ*(σ)
      const TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);

      // Off-diagonal contribution: Jxy/2 * (S^+_i S^-_j + S^-_i S^+_j)
      // Matrix element of (S^+ S^- + S^- S^+) is 1 when spins are exchanged
      energy_offdiag = TenElemT(0.5 * jxy_) * ratio;
    }

    return energy_diag + energy_offdiag;
  }
};

} // namespace qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_HEISENBERG_SQUARE_PBC_H
