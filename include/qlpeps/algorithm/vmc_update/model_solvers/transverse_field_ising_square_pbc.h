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
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/two_dim_tn/tensor_network_2d/trg_contractor.h"    // TRGContractor

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
 * - Energy + gradient holes are supported for 2x2 PBC using `TRGContractor::PunchHole` terminator.
 * - For larger sizes, `PunchHole` is not implemented yet in TRGContractor; thus gradient holes
 *   are not available.
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
    contractor.SetTruncateParams(trunc_para);
    contractor.Init(tn);

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

    TenElemT energy_ex(0);
    for (size_t row = 0; row < ly; ++row) {
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx site{row, col};

        // Hole (environment) for this site; TRG PunchHole currently only supports 2x2.
        const auto hole = contractor.PunchHole(tn, site);
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

 private:
  double h_;
};

} // namespace qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_PBC_H


