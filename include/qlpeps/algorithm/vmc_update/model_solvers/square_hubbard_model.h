/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for the Hubbard model in square lattice
*/
#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_HUBBARD_MODEL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_HUBBARD_MODEL_H

#include "qlpeps/algorithm/vmc_update/model_solvers/square_nn_fermion_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_nn_fermion_measure_solver.h"
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate

namespace qlpeps {
using namespace qlten;

enum class HubbardSingleSiteState {
  DoubleOccupancy,  // 0
  SpinUp,           // 1
  SpinDown,         // 2
  Empty             // 3
};

double HubbardConfig2Density(const size_t config) {
  switch (HubbardSingleSiteState(config)) {
    case HubbardSingleSiteState::DoubleOccupancy:return 2;
    case HubbardSingleSiteState::SpinUp:return 1;
    case HubbardSingleSiteState::SpinDown:return 1;
    case HubbardSingleSiteState::Empty:return 0;
    default:std::cerr << "Expected configuration!" << std::endl;
      return -1;
  }
}

double HubbardConfig2Spinz(const size_t config) {
  switch (HubbardSingleSiteState(config)) {
    case HubbardSingleSiteState::DoubleOccupancy:return 0;
    case HubbardSingleSiteState::SpinUp:return 0.5;
    case HubbardSingleSiteState::SpinDown:return -0.5;
    case HubbardSingleSiteState::Empty:return 0;
    default:std::cerr << "Expected configuration!" << std::endl;
      return -1;
  }
}

/**
 * NN hopping + U term
 *
 * 0: double occupancy
 * 1: spin up
 * 2: spin down
 * 3: empty
 */

class SquareHubbardModel : public SquareNNFermionModelEnergySolver<SquareHubbardModel>,
                          public SquareNNFermionMeasureSolver<SquareHubbardModel> {
 public:
  using SquareNNFermionModelEnergySolver<SquareHubbardModel>::CalEnergyAndHoles;
  using SquareNNFermionMeasureSolver<SquareHubbardModel>::operator();
  SquareHubbardModel(void) = delete;
  SquareHubbardModel(double t, double U, double mu) : t_(t), U_(U), mu_(mu) {}

  ///< requirement from SquareNNFermionModelEnergySolver
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi // return value, used for check the accuracy
  );

  ///< requirement from SquareNNFermionModelEnergySolver
  ///< on-site repulsion and chemical potential terms
  double EvaluateTotalOnsiteEnergy(const Configuration &config) {
    double energy(0);
    size_t num_double_occupancy = 0;
    size_t num_single_occupancy = 0;
    for (auto &site_config : config) {
      if (HubbardSingleSiteState(site_config) == HubbardSingleSiteState::DoubleOccupancy) {
        num_double_occupancy++;
      } else if (HubbardSingleSiteState(site_config) == HubbardSingleSiteState::SpinUp
          || HubbardSingleSiteState(site_config) == HubbardSingleSiteState::SpinDown) {
        num_single_occupancy++;
      }
    }
    energy += U_ * double(num_double_occupancy);
    size_t electron_num = num_double_occupancy * 2 + num_single_occupancy;
    energy += (-mu_) * double(electron_num);
    return energy;
  }

  double CalDensityImpl(const size_t config) const {
    return HubbardConfig2Density(config);
  }

  double CalSpinSzImpl(const size_t config) const {
    return HubbardConfig2Spinz(config);
  }

 private:
  double t_;
  double U_;
  double mu_;
};


template<typename TenElemT, typename QNT>
TenElemT SquareHubbardModel::EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi
) {
  if (config1 == config2) {
    psi.reset();
    return 0;
  } else {
    psi = tn.Trace(site1, site2, orient);
    if ((HubbardSingleSiteState(config1) == HubbardSingleSiteState::Empty
        && (HubbardSingleSiteState(config2) == HubbardSingleSiteState::SpinUp
            || HubbardSingleSiteState(config2) == HubbardSingleSiteState::SpinDown))
        || (HubbardSingleSiteState(config2) == HubbardSingleSiteState::Empty
            && (HubbardSingleSiteState(config1) == HubbardSingleSiteState::SpinUp
                || HubbardSingleSiteState(config1) == HubbardSingleSiteState::SpinDown))) {
      //one electron case
      TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                              split_index_tps_on_site1[size_t(config2)],
                                              split_index_tps_on_site2[size_t(config1)]);
      TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
      return (-t_) * ratio;
    } else if (((HubbardSingleSiteState) config1 == HubbardSingleSiteState::DoubleOccupancy
        && ((HubbardSingleSiteState) config2 == HubbardSingleSiteState::SpinUp
            || (HubbardSingleSiteState) config2 == HubbardSingleSiteState::SpinDown))
        || ((HubbardSingleSiteState) config2 == HubbardSingleSiteState::DoubleOccupancy
            && ((HubbardSingleSiteState) config1 == HubbardSingleSiteState::SpinUp || (HubbardSingleSiteState) config1
                == HubbardSingleSiteState::SpinDown))) {
      //3 electrons case
      TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                              split_index_tps_on_site1[config2],
                                              split_index_tps_on_site2[config1]);
      TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
      return t_ * ratio;
    } else if ((HubbardSingleSiteState) config1 == HubbardSingleSiteState::SpinUp
        && (HubbardSingleSiteState) config2 == HubbardSingleSiteState::SpinDown) {
      TenElemT psi_ex1 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::Empty)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::DoubleOccupancy)]);
      TenElemT psi_ex2 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::DoubleOccupancy)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::Empty)]);
      TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi.value());
      TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi.value());
      return (-t_) * (ratio1 + ratio2);
    } else if ((HubbardSingleSiteState) config1 == HubbardSingleSiteState::SpinDown && (HubbardSingleSiteState) config2
        == HubbardSingleSiteState::SpinUp) {
      TenElemT psi_ex1 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::Empty)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::DoubleOccupancy)]);
      TenElemT psi_ex2 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::DoubleOccupancy)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::Empty)]);
      TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi.value());
      TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi.value());
      return t_ * (ratio1 + ratio2);
    } else { // |Double Occupancy, Empty> or |Empty, Double Occupancy>
      TenElemT psi_ex1 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::SpinUp)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::SpinDown)]);
      TenElemT psi_ex2 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::SpinDown)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::SpinUp)]);
      TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi.value());
      TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi.value());
      return -t_ * ratio1 + t_ * ratio2;
    }
  }
}
}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_HUBBARD_MODEL_H
