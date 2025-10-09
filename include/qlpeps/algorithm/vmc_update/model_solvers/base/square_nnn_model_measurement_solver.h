/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-06-10
*
* Description: QuantumLiquids/PEPS project. Measurement Solver Base for next-nearest-neighbor models on square lattice
*/


#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_SQUARE_NNN_MODEL_MEASUREMENT_SOLVERS_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_SQUARE_NNN_MODEL_MEASUREMENT_SOLVERS_H

#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/bond_traversal_mixin.h"

namespace qlpeps {

// SquareNNNModelMeasurementSolver: base for square-lattice models with (optional) NNN interactions.
// Provides registry-based EvaluateObservables; derived models supply bond evaluators and optional
// site-local fields. Psi is handled via BuildPsiList and the executor, not the registry. 

template<typename ModelType, bool has_nnn_interaction = true>
class SquareNNNModelMeasurementSolver
    : public ModelMeasurementSolver<SquareNNNModelMeasurementSolver<ModelType, has_nnn_interaction>> {
 public:
  using ModelMeasurementSolver<SquareNNNModelMeasurementSolver<ModelType, has_nnn_interaction>>::EvaluateObservables;
  using ModelMeasurementSolver<SquareNNNModelMeasurementSolver<ModelType, has_nnn_interaction>>::DescribeObservables;

  // Legacy SampleMeasureImpl removed under registry-only API

  // Minimal registry-based API. Avoid generic keys; expose concrete ones.
  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample
  ) {
    ObservableMap<TenElemT> out;
    auto &tn = tps_sample->tn;
    const Configuration &config = tps_sample->config;
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
    const size_t ly = tn.rows();
    const size_t lx = tn.cols();

    // Site-local observables
    auto *derived = static_cast<ModelType *>(this);
    if constexpr (ModelType::requires_spin_sz_measurement) {
      std::vector<TenElemT> sz; sz.reserve(config.size());
      for (auto &c : config) { sz.push_back(derived->CalSpinSzImpl(c)); }
      out["spin_z"] = std::move(sz);
    }
    if constexpr (ModelType::requires_density_measurement) {
      std::vector<TenElemT> ch; ch.reserve(config.size());
      for (auto &c : config) { ch.push_back(derived->CalDensityImpl(c)); }
      out["charge"] = std::move(ch);
    }

    // Accumulators for bond energies
    std::vector<TenElemT> e_h; if (lx > 1) e_h.reserve(ly * (lx - 1));
    std::vector<TenElemT> e_v; if (ly > 1) e_v.reserve((ly - 1) * lx);
    std::vector<TenElemT> e_diag; if constexpr (has_nnn_interaction) { if (lx > 1 && ly > 1) e_diag.reserve((ly - 1) * (lx - 1)); }
    TenElemT energy_bond_total = 0;

    // Bond measurement functor (NN)
    auto bond_measure_func = [&](const SiteIdx site1, const SiteIdx site2, const BondOrientation orient, const TenElemT inv_psi_row_or_col) {
      TenElemT eb;
      if constexpr (Index<QNT>::IsFermionic()) {
        std::optional<TenElemT> fermion_psi; fermion_psi = TenElemT(1.0) / inv_psi_row_or_col;
        eb = derived->EvaluateBondEnergy(site1, site2, config(site1), config(site2), orient, tn,
                                         (*split_index_tps)(site1), (*split_index_tps)(site2), fermion_psi);
      } else {
        eb = derived->EvaluateBondEnergy(site1, site2, config(site1), config(site2), orient, tn,
                                         (*split_index_tps)(site1), (*split_index_tps)(site2), inv_psi_row_or_col);
      }
      if (orient == HORIZONTAL) {
        e_h.push_back(eb);
      } else if (orient == VERTICAL) {
        e_v.push_back(eb);
      }
      energy_bond_total += eb;
    };

    // NNN link measurement functor (accumulate LEFTUP_TO_RIGHTDOWN to preserve legacy semantics)
    auto nnn_link_measure_func = [&](const SiteIdx site1, const SiteIdx site2, const DIAGONAL_DIR dir,
                                     const TenElemT inv_psi_row, std::optional<TenElemT> &fermion_psi) {
      if constexpr (!has_nnn_interaction) { return; }
      // Reconstruct psi for fermions if needed
      if constexpr (Index<QNT>::IsFermionic()) { fermion_psi = TenElemT(1.0) / inv_psi_row; }
      TenElemT eb;
      if constexpr (Index<QNT>::IsFermionic()) {
        eb = derived->EvaluateNNNEnergy(site1, site2, config(site1), config(site2), dir, tn,
                                        (*split_index_tps)(site1), (*split_index_tps)(site2), fermion_psi);
      } else {
        eb = derived->EvaluateNNNEnergy(site1, site2, config(site1), config(site2), dir, tn,
                                        (*split_index_tps)(site1), (*split_index_tps)(site2), inv_psi_row);
      }
      // Preserve old behavior: only record and sum LEFTUP_TO_RIGHTDOWN
      if (dir == LEFTUP_TO_RIGHTDOWN) {
        e_diag.push_back(eb);
        energy_bond_total += eb;
      }
    };

    // No off-diagonal long-range measurement at this layer
    auto off_diag_long_range_measure_func = [](const auto &, const auto &) {};

    std::vector<TenElemT> psi_list;
    psi_list.reserve(ly + lx);

    if constexpr (has_nnn_interaction) {
      BondTraversalMixin::TraverseAllBonds(tn, trunc_para, bond_measure_func, nnn_link_measure_func,
                                           off_diag_long_range_measure_func, psi_list);
    } else {
      // Pass nullptr to skip NNN traversal
      BondTraversalMixin::TraverseAllBonds(tn, trunc_para, bond_measure_func, nullptr,
                                           off_diag_long_range_measure_func, psi_list);
    }

    // Total energy = bond total + onsite; cache psi summary for Measurer to consume without re-traversal
    TenElemT energy_onsite = derived->EvaluateTotalOnsiteEnergy(config);
    out["energy"] = {energy_bond_total + energy_onsite};
    auto psi_summary = this->template ComputePsiSummary<TenElemT>(psi_list);
    this->template SetLastPsiSummary<TenElemT>(psi_summary.psi_mean, psi_summary.psi_rel_err);

    if (!e_h.empty()) out["bond_energy_h"] = std::move(e_h);
    if (!e_v.empty()) out["bond_energy_v"] = std::move(e_v);
    if constexpr (has_nnn_interaction) { if (!e_diag.empty()) out["bond_energy_diag"] = std::move(e_diag); }
    return out;
  }

  // Provide generic metadata with stable keys; shapes are model-dependent and may be left unspecified.
  std::vector<ObservableMeta> DescribeObservables() const {
    std::vector<ObservableMeta> out = {
        {"energy", "Total energy (scalar)", {}, {}}
    };
    if constexpr (ModelType::requires_spin_sz_measurement) {
      out.push_back({"spin_z", "Local spin Sz per site", {}, {}});
    }
    if constexpr (ModelType::requires_density_measurement) {
      out.push_back({"charge", "Local charge per site", {}, {}});
    }
    out.push_back({"bond_energy_h", "Bond energy on horizontal NN bonds", {}, {}});
    out.push_back({"bond_energy_v", "Bond energy on vertical NN bonds", {}, {}});
    if constexpr (has_nnn_interaction) {
      out.push_back({"bond_energy_diag", "Bond energy on diagonal NNN bonds", {}, {}});
    }
    return out;
  }


  //like Spin Sz & Charge order
  template<typename TenElemT>
  void MeasureDiagonalOneAndTwoPointFunctions(const Configuration &config, /* legacy removed */ void * = nullptr) {
    static_assert(
        !ModelType::requires_density_measurement
            || requires(ModelType m,
                        size_t
                        config) {{ m.CalDensityImpl(config) } -> std::convertible_to<QLTEN_Complex>; },
        "If requires_density_measurement is true, ModelType must implement CalDensityImpl correctly."
    );

    static_assert(
        !ModelType::requires_spin_sz_measurement
            || requires(ModelType m,
                        size_t
                        config) {{ m.CalSpinSzImpl(config) } -> std::convertible_to<QLTEN_Complex>; },
        "If requires_density_measurement is true, ModelType must implement CalDensityImpl correctly."

    );
    const size_t N = config.size();

    // Reserve space for measurements
    int num_measure_item = int(ModelType::requires_spin_sz_measurement) + int(ModelType::requires_density_measurement);
    (void)N; (void)num_measure_item;

    auto *derived = static_cast<ModelType *>(this);

    // Calculate density and density correlation, diagonal orders
    // usually invalid for spin model
    if constexpr (ModelType::requires_density_measurement) {
      for (auto &local_config : config) {
        (void)local_config;
      }
      // Calculate density-density correlations
      // move to postprocess to save memory. All the site-local fields will be dumped
      /*
      for (auto &config_i : config) {
        for (auto &config_j : config) {
          res.two_point_functions_loc.push_back(
              derived->CalDensityImpl(config_i) * derived->CalDensityImpl(config_j)
          );
        }
      }
       */
    }

    // Calculate spin-spin correlations if derived class has CalSpinSz
    // usually invalid for, like spinless fermion
    if constexpr (ModelType::requires_spin_sz_measurement) {
      // Calculate spin Sz for each site
      for (auto &local_config : config) {
        (void)local_config;
      }
//      for (auto &config_i : config) {
//        for (auto &config_j : config) {
//          res.two_point_functions_loc.push_back(
//              derived->CalSpinSzImpl(config_i) * derived->CalSpinSzImpl(config_j)
//          );
//        }
//      }
    }
  }
};//SquareNNNModelMeasurementSolver



}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_SQUARE_NNN_MODEL_MEASUREMENT_SOLVERS_H
