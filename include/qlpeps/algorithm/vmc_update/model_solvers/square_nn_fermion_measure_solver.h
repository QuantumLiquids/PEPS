/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-02-20
*
* Description: QuantumLiquids/PEPS project. Base Measurement Solver for nearest-neighbor fermion models on square lattice
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_MEASURE_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_NN_FERMION_MEASURE_SOLVER_H

#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_nn_energy_solver.h"

namespace qlpeps {

template<typename ModelType>
class SquareNNFermionMeasureSolver : public ModelMeasurementSolver<SquareNNFermionMeasureSolver<ModelType>> {
 public:
  using ModelMeasurementSolver<SquareNNFermionMeasureSolver<ModelType>>::operator();
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

    // Reserve space for measurements
    res.one_point_functions_loc.reserve(N);
    res.two_point_functions_loc.reserve(N * N);
    res.bond_energys_loc.reserve(2 * N); // horizontal and vertical bonds

    auto *derived = static_cast<ModelType *>(this);

    // Calculate density for each site
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

    // Calculate spin-spin correlations if derived class has CalSpinSz
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


    // Measure hopping terms using BMPS
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
    TenElemT inv_psi = 1.0 / tps_sample->amplitude;
    TenElemT energy(0);

    // Horizontal hopping measurements
    tn.GenerateBMPSApproach(UP, trunc_para);
    psi_list.reserve(tn.rows() + tn.cols());
    for (size_t row = 0; row < tn.rows(); row++) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 1, true);
      bool psi_added = false;
      for (size_t col = 0; col < tn.cols() - 1; col++) {
        const SiteIdx site1 = {row, col};
        const SiteIdx site2 = {row, col + 1};
        std::optional<TenElemT> psi;
        TenElemT bond_energy = derived->EvaluateBondEnergy(site1, site2,
                                                           config(site1), config(site2),
                                                           HORIZONTAL,
                                                           tn,
                                                           (*split_index_tps)(site1),
                                                           (*split_index_tps)(site2),
                                                           psi);
        res.bond_energys_loc.push_back(bond_energy);
        energy += bond_energy;
        if (!psi_added && psi.has_value()) {
          psi_list.push_back(psi.value());
          psi_added = true;
        }
        if (col < tn.cols() - 2) {
          tn.BTenMoveStep(RIGHT);
        }
      }
      if (row < tn.rows() - 1) {
        tn.BMPSMoveStep(DOWN, trunc_para);
      }
    }

    // Vertical hopping measurements  
    tn.GenerateBMPSApproach(LEFT, trunc_para);
    for (size_t col = 0; col < tn.cols(); col++) {
      tn.InitBTen(UP, col);
      tn.GrowFullBTen(DOWN, col, 2, true);
      bool psi_added = false;
      for (size_t row = 0; row < tn.rows() - 1; row++) {
        const SiteIdx site1 = {row, col};
        const SiteIdx site2 = {row + 1, col};
        std::optional<TenElemT> psi;
        TenElemT bond_energy = derived->EvaluateBondEnergy(site1, site2,
                                                           config(site1), config(site2),
                                                           VERTICAL,
                                                           tn,
                                                           (*split_index_tps)(site1),
                                                           (*split_index_tps)(site2),
                                                           psi);
        res.bond_energys_loc.push_back(bond_energy);
        energy += bond_energy;
        if (!psi_added && psi.has_value()) {
          psi_list.push_back(psi.value());
          psi_added = true;
        }
        if (row < tn.rows() - 2) {
          tn.BTenMoveStep(DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        tn.BMPSMoveStep(RIGHT, trunc_para);
      }
    }
    energy += static_cast<ModelType *>(this)->EvaluateTotalOnsiteEnergy(config);
    res.energy_loc = energy;

    return res;
  }

  double CalDensity(const size_t config) const {
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