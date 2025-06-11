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

/**
 * SquareNNNModelMeasurementSolver is the base class to define general next-nearest-neighbor
 * model measurement solver on the square lattices,
 * work for the energy and order parameter measurements.
 * The CRTP technique is used to realize the Polymorphism.
 *
 * To defined the concrete model which inherit from SquareNNNModelMeasurementSolver,
 * the following member function with specific signature must to be defined:
 * template<typename TenElemT, typename QNT>
  [[nodiscard]] TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      const TenElemT inv_psi
  )

  template<typename TenElemT, typename QNT>
  [[nodiscard]] TenElemT EvaluateNNNEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const DIAGONAL_DIR diagonal_dir,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi // return value, used for check the accuracy
  );

  the following member functions can be defined if you want to perform the measurement, or relevant measure will be ignored if not define:
   - CalDensityImpl (work for, like fermion models)
   - CalSpinSz (work for, like t-J, Hubbard, and spin models)
   - EvaluateOffDiagOrderInRow (to evaluate the off-diagonal orders in specific row)
   - EvaluateBondSC (optional)
 *
 */

template<typename ModelType, bool has_nnn_interaction = true>
class SquareNNNModelMeasurementSolver : public ModelMeasurementSolver<SquareNNNModelMeasurementSolver<ModelType,
                                                                                                      has_nnn_interaction>> {
 public:
  using ModelMeasurementSolver<SquareNNNModelMeasurementSolver<ModelType, has_nnn_interaction>>::operator();

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> SampleMeasureImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      std::vector<TenElemT> &psi_list
  ) {
    ObservablesLocal<TenElemT> res;
    TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
    const Configuration &config = tps_sample->config;
    const size_t lx = tn.cols();
    const size_t N = config.size();

    this->MeasureDiagonalOneAndTwoPointFunctions(tps_sample->config, res);

    auto *derived = static_cast<ModelType *>(this);
    // Measure Energy
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;

    res.bond_energys_loc.reserve(2 * N); // horizontal and vertical bonds

    if constexpr (requires { ModelType::enable_sc_measurement; }) {
      res.two_point_functions_loc.reserve(4 * N);
    }

    auto bond_measure_func = [&, split_index_tps](const SiteIdx &site1,
                                                  const SiteIdx &site2,
                                                  const BondOrientation bond_orient,
                                                  const TenElemT &inv_psi) {
      TenElemT bond_energy;
      std::optional<TenElemT> fermion_psi;
      if constexpr (Index<QNT>::IsFermionic()) {
        bond_energy = derived->EvaluateBondEnergy(site1,
                                                  site2,
                                                  config(site1),
                                                  config(site2),
                                                  bond_orient,
                                                  tn,
                                                  (*split_index_tps)(site1),
                                                  (*split_index_tps)(site2),
                                                  fermion_psi);
      } else {
        bond_energy = derived->EvaluateBondEnergy(site1, site2, config(site1), config(site2), bond_orient,
                                                  tn, (*split_index_tps)(site1), (*split_index_tps)(site2), inv_psi);
      }
      res.bond_energys_loc.push_back(bond_energy);
      if constexpr (requires { ModelType::enable_sc_measurement; }) {
        auto sc_val = derived->EvaluateBondSC(site1, site2, config(site1), config(site2), bond_orient,
                                              tn, (*split_index_tps)(site1), (*split_index_tps)(site2), fermion_psi);
        res.two_point_functions_loc.emplace_back(sc_val.first);
        res.two_point_functions_loc.emplace_back(sc_val.second);
      }
    };

    auto nnn_link_measure_func = [&](const SiteIdx &site1,
                                     const SiteIdx &site2,
                                     const DIAGONAL_DIR diagonal_dir,
                                     const TenElemT &inv_psi,
                                     std::optional<TenElemT> &fermion_psi) {
      TenElemT nnn_energy;
      if constexpr (has_nnn_interaction) {
        if constexpr (Index<QNT>::IsFermionic()) {
          nnn_energy = derived->EvaluateNNNEnergy(site1,
                                                  site2,
                                                  config(site1),
                                                  config(site2),
                                                  diagonal_dir,
                                                  tn,
                                                  (*split_index_tps)(site1),
                                                  (*split_index_tps)(site2),
                                                  fermion_psi);
        } else {
          nnn_energy = derived->EvaluateNNNEnergy(site1, site2, config(site1), config(site2), diagonal_dir,
                                                  tn, (*split_index_tps)(site1), (*split_index_tps)(site2), inv_psi);
        }
        res.bond_energys_loc.push_back(nnn_energy);
      }
    };

    auto offdiag_long_order_func = [&, split_index_tps](const size_t row, const TenElemT &inv_psi) {
      if constexpr (requires {
        derived->EvaluateOffDiagOrderInRow(split_index_tps, tn, res.two_point_functions_loc, inv_psi, config, row);
      }) {
        derived->EvaluateOffDiagOrderInRow(split_index_tps, tn, res.two_point_functions_loc, inv_psi, config, row);
      }
    };
    if constexpr (has_nnn_interaction) {
      BondTraversalMixin::TraverseAllBonds(
          tps_sample->tn,
          tps_sample->trun_para,
          bond_measure_func,
          nnn_link_measure_func,
          offdiag_long_order_func,
          psi_list
      );
    } else {
      BondTraversalMixin::TraverseAllBonds(
          tps_sample->tn,
          tps_sample->trun_para,
          bond_measure_func,
          nullptr,
          offdiag_long_order_func,
          psi_list
      );
    }

    TenElemT energy_bond_total = std::reduce(res.bond_energys_loc.begin(), res.bond_energys_loc.end());

    TenElemT energy_onsite = static_cast<ModelType *>(this)->EvaluateTotalOnsiteEnergy(config);
    res.energy_loc = energy_bond_total + energy_onsite;
    return res;
  }

  //like Spin Sz & Charge order
  template<typename TenElemT>
  void MeasureDiagonalOneAndTwoPointFunctions(const Configuration &config, ObservablesLocal<TenElemT> &res) {
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
    res.one_point_functions_loc.reserve(N * num_measure_item);
    res.two_point_functions_loc.reserve(N * N * num_measure_item);

    auto *derived = static_cast<ModelType *>(this);

    // Calculate density and density correlation, diagonal orders
    // usually invalid for spin model
    if constexpr (ModelType::requires_density_measurement) {
      for (auto &local_config : config) {
        res.one_point_functions_loc.push_back(derived->CalDensityImpl(local_config));
      }
      // Calculate density-density correlations
      for (auto &config_i : config) {
        for (auto &config_j : config) {
          res.two_point_functions_loc.push_back(
              derived->CalDensityImpl(config_i) * derived->CalDensityImpl(config_j)
          );
        }
      }
    }

    // Calculate spin-spin correlations if derived class has CalSpinSz
    // usually invalid for, like spinless fermion
    if constexpr (ModelType::requires_spin_sz_measurement) {
      // Calculate spin Sz for each site
      for (auto &local_config : config) {
        res.one_point_functions_loc.push_back(derived->CalSpinSzImpl(local_config));
      }
      for (auto &config_i : config) {
        for (auto &config_j : config) {
          res.two_point_functions_loc.push_back(
              derived->CalSpinSzImpl(config_i) * derived->CalSpinSzImpl(config_j)
          );
        }
      }
    }
  }
};//SquareNNNModelMeasurementSolver



}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_SQUARE_NNN_MODEL_MEASUREMENT_SOLVERS_H
