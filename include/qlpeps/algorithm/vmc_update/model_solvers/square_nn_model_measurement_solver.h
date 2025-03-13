/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-20
*
* Description: QuantumLiquids/PEPS project. Energy and Measurement Solver Base for nearest-neighbor models on square lattice
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_MEASURE_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_MEASURE_SOLVER_H

#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_nn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/bond_traversal_mixin.h"

namespace qlpeps {

/**
 * SquareNNModelMeasurementSolver is the base class to define general nearest-neighbor
 * model measurement solver on the square lattices,
 * work for the energy and order parameter measurements.
 * The CRTP technique is used to realize the Polymorphism.
 *
 * To defined the concrete model which inherit from SquareNNModelMeasurementSolver,
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
  the following member functions can be defined if you want to perform the measurement, or relevant measure will be ignored if not define:
   - CalDensityImpl (work for, like fermion models)
   - CalSpinSz (work for, like t-J, Hubbard, and spin models)
   - EvaluateOffDiagOrderInRow (to evaluate the off-diagonal orders in specific row)
   - EvaluateBondSC (optional)
 *
 */
template<typename ModelType>
class SquareNNModelMeasurementSolver : public ModelMeasurementSolver<SquareNNModelMeasurementSolver<ModelType>> {
 public:
  using ModelMeasurementSolver<SquareNNModelMeasurementSolver<ModelType>>::operator();

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

    this->MeasureDiagonalOrder(tps_sample->config, res);

    auto *derived = static_cast<ModelType *>(this);
    // Measure Energy
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
    tn.GenerateBMPSApproach(UP, trunc_para);

    res.bond_energys_loc.reserve(2 * N); // horizontal and vertical bonds
    psi_list.reserve(tn.rows() + tn.cols());

    if constexpr (requires { ModelType::enable_sc_measurement; }) {
      res.two_point_functions_loc.reserve(4 * N);
    }

    BondTraversalMixin::TraverseAllBonds(
        tps_sample->tn,
        tps_sample->trun_para,
        [&, split_index_tps](const SiteIdx &site1,
                             const SiteIdx &site2,
                             const BondOrientation bond_orient,
                             const TenElemT &inv_psi) {
          TenElemT bond_energy;
          std::optional<TenElemT> fermion_psi;
          if constexpr (Index<QNT>::IsFermionic()) {
            bond_energy =
                static_cast<ModelType *>(this)->EvaluateBondEnergy(site1,
                                                                   site2,
                                                                   (config(site1)),
                                                                   (config(site2)),
                                                                   bond_orient,
                                                                   tn,
                                                                   (*split_index_tps)(site1),
                                                                   (*split_index_tps)(site2),
                                                                   fermion_psi);
          } else {
            bond_energy =
                static_cast<ModelType *>(this)->EvaluateBondEnergy(site1,
                                                                   site2,
                                                                   (config(site1)),
                                                                   (config(site2)),
                                                                   bond_orient,
                                                                   tn,
                                                                   (*split_index_tps)(site1),
                                                                   (*split_index_tps)(site2),
                                                                   inv_psi);
          }
          res.bond_energys_loc.push_back(bond_energy);
          if constexpr (requires { ModelType::enable_sc_measurement; }) {
            auto sc_val = static_cast<ModelType *>(this)->EvaluateBondSC(site1,
                                                                         site2,
                                                                         (config(site1)),
                                                                         (config(site2)),
                                                                         bond_orient,
                                                                         tn,
                                                                         (*split_index_tps)(site1),
                                                                         (*split_index_tps)(site2),
                                                                         fermion_psi);
            res.two_point_functions_loc.emplace_back(sc_val.first);
            res.two_point_functions_loc.emplace_back(sc_val.second);
          }
        },
        [&, split_index_tps](const size_t row, const TenElemT &inv_psi) {
          if constexpr (requires {
            derived->EvaluateOffDiagOrderInRow(split_index_tps,
                                               tn, res.two_point_functions_loc,
                                               inv_psi, config, row);
          }) {
            derived->EvaluateOffDiagOrderInRow(split_index_tps,
                                               tn, res.two_point_functions_loc,
                                               inv_psi, config, row);
          }
        },
        psi_list
    );

    TenElemT energy_bond_total = std::reduce(res.bond_energys_loc.begin(), res.bond_energys_loc.end());

    TenElemT energy_onsite = static_cast<ModelType *>(this)->EvaluateTotalOnsiteEnergy(config);
    res.energy_loc = energy_bond_total + energy_onsite;
    return res;
  }

  //like Spin Sz & Charge order
  template<typename TenElemT>
  void MeasureDiagonalOrder(const Configuration &config, ObservablesLocal<TenElemT> &res) {
    static_assert(
        !ModelType::requires_density_measurement
            || requires(ModelType m,
                        size_t config) {{ m.CalDensityImpl(config) } -> std::convertible_to<QLTEN_Complex>; },
        "If requires_density_measurement is true, ModelType must implement CalDensityImpl correctly."
    );

    static_assert(
        !ModelType::requires_spin_sz_measurement
            || requires(ModelType m,
                        size_t config) {{ m.CalSpinSzImpl(config) } -> std::convertible_to<QLTEN_Complex>; },
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
};//SquareNNModelMeasurementSolver

} // namespace qlpeps

#endif // QLPEPS_VMC_SQUARE_NN_FERMION_MEASURE_SOLVER_H 