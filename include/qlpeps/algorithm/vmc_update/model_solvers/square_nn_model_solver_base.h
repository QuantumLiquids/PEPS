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

namespace qlpeps {

/**
 * SquareNNModelSolverBase is the base class to define general nearest-neighbor models on the square lattices,
 * work for both energy & gradient evaluation and the order measurements.
 * The CRTP technique is used to realize the Polymorphism.
 *
 * To defined the concrete model which inherit from SquareNNModelSolverBase,
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
 *
 */
template<typename ModelType>
class SquareNNModelSolverBase : public ModelMeasurementSolver<SquareNNModelSolverBase<ModelType>>,
                                public SquareNNModelEnergySolver<ModelType> {
 public:
  using ModelMeasurementSolver<SquareNNModelSolverBase<ModelType>>::operator();
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
    TensorNetwork2D<TenElemT, QNT> hole_res(1, 1); //useless variable.
    for (size_t row = 0; row < tn.rows(); row++) {
      this->template CalHorizontalBondEnergyAndHolesSweepRowImpl<TenElemT, QNT, false>(row,
                                                                                       split_index_tps,
                                                                                       tps_sample,
                                                                                       hole_res,
                                                                                       res.bond_energys_loc,
                                                                                       psi_list);
      // Measure off-diagonal correlation, now only design the interface for boson/spin models
      if constexpr (requires {
        derived->EvaluateOffDiagOrderInRow(split_index_tps,
                                           tn, res.two_point_functions_loc,
                                           1.0 / psi_list.back(), config, row);
      }) {
        derived->EvaluateOffDiagOrderInRow(split_index_tps,
                                           tn, res.two_point_functions_loc,
                                           1.0 / psi_list.back(), config, row);
      }

      if (row < tn.rows() - 1) {
        tn.BMPSMoveStep(DOWN, trunc_para);
      }
    }

    this->CalVerticalBondEnergyImpl(split_index_tps, tps_sample, res.bond_energys_loc, psi_list);
    TenElemT energy_bond_total = std::reduce(res.bond_energys_loc.begin(), res.bond_energys_loc.end());

    TenElemT energy_onsite = static_cast<ModelType *>(this)->EvaluateTotalOnsiteEnergy(config);
    res.energy_loc = energy_bond_total + energy_onsite;
    return res;
  }

  //like Spin Sz & Charge order
  template<typename TenElemT>
  void MeasureDiagonalOrder(const Configuration &config, ObservablesLocal<TenElemT> &res) {
    const size_t N = config.size();

    // Reserve space for measurements
    res.one_point_functions_loc.reserve(N);
    res.two_point_functions_loc.reserve(N * N);

    auto *derived = static_cast<ModelType *>(this);

    // Calculate density and density correlation, diagonal orders
    // usually invalid for spin model
    if constexpr (requires { derived->CalDensity(config({0, 0})); }) {
      for (auto &local_config : config) {
        res.one_point_functions_loc.push_back(derived->CalDensity(local_config));
      }
      // Calculate density-density correlations
      for (auto &config_i : config) {
        for (auto &config_j : config) {
          res.two_point_functions_loc.push_back(
              derived->CalDensity(config_i) * derived->CalDensity(config_j)
          );
        }
      }
    }

    // Calculate spin-spin correlations if derived class has CalSpinSz
    // usually invalid for, like spinless fermion
    if constexpr (requires { derived->CalSpinSz(config({0, 0})); }) {
      // Calculate spin Sz for each site
      for (auto &local_config : config) {
        res.one_point_functions_loc.push_back(derived->CalSpinSz(local_config));
      }
      for (auto &config_i : config) {
        for (auto &config_j : config) {
          res.two_point_functions_loc.push_back(
              derived->CalSpinSz(config_i) * derived->CalSpinSz(config_j)
          );
        }
      }
    }
  }

  double CalDensity(const size_t config) const requires requires(ModelType m) { m.CalDensityImpl(config); } {
    auto *derived = static_cast<const ModelType *>(this);
    return derived->CalDensityImpl(config);
  }

  double CalSpinSz(const size_t config) const requires requires(ModelType m) { m.CalSpinSzImpl(config); } {
    auto *derived = static_cast<const ModelType *>(this);
    return derived->CalSpinSzImpl(config);
  }

};

} // namespace qlpeps

#endif // QLPEPS_VMC_SQUARE_NN_FERMION_MEASURE_SOLVER_H 