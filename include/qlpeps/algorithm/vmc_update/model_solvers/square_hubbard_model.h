/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for the Hubbard model in square lattice
*/
#ifndef QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_HUBBARD_MODEL_H
#define QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_HUBBARD_MODEL_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      //ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               //ComplexConjugate
namespace qlpeps {
using namespace qlten;

enum class HubbardSingleSiteState {
  DoubleOccupancy,  // 0
  SpinUp,           // 1
  SpinDown,         // 2
  Empty             // 3
};

size_t ElectronNum(HubbardSingleSiteState state) {
  if (state == HubbardSingleSiteState::DoubleOccupancy) {
    return 2;
  } else if (state == HubbardSingleSiteState::Empty) {
    return 0;
  } else {
    return 1;
  }
}

size_t ElectronSpin(HubbardSingleSiteState state) {
  if (state == HubbardSingleSiteState::DoubleOccupancy || state == HubbardSingleSiteState::Empty) {
    return 0;
  } else {
    return 1;
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
template<typename TenElemT, typename QNT>
class SquareHubbardModel : public ModelEnergySolver<TenElemT, QNT>, ModelMeasurementSolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SquareHubbardModel(void) = delete;
  SquareHubbardModel(double t, double U, double mu) : t_(t), U_(U), mu_(mu) {}

  template<typename WaveFunctionComponentType, bool calchols = true>
  TenElemT CalEnergyAndHoles(
      const SITPS *sitps,
      WaveFunctionComponentType *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res
  );

  template<typename WaveFunctionComponentType>
  ObservablesLocal<TenElemT> SampleMeasure(
      const SITPS *sitps,
      WaveFunctionComponentType *tps_sample
  );
 private:
  double t_;
  double U_;
  double mu_;
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
template<typename TenElemT, typename QNT>
TenElemT EvaluateBondHoppingEnergyForHubbardModel(
    const SiteIdx site1, const SiteIdx site2,
    const HubbardSingleSiteState config1, const HubbardSingleSiteState config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    double t
) {
  if (config1 == config2) {
    return 0;
  } else {
    TenElemT psi = tn.Trace(site1, site2, orient);
    if ((config1 == HubbardSingleSiteState::Empty
        && (config2 == HubbardSingleSiteState::SpinUp || config2 == HubbardSingleSiteState::SpinDown))
        || (config2 == HubbardSingleSiteState::Empty
            && (config1 == HubbardSingleSiteState::SpinUp || config1 == HubbardSingleSiteState::SpinDown))) {
      //one electron case
      TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                              split_index_tps_on_site1[size_t(config2)],
                                              split_index_tps_on_site2[size_t(config1)]);
      TenElemT ratio = ComplexConjugate(psi_ex / psi);
      return (-t) * ratio;
    } else if ((config1 == HubbardSingleSiteState::DoubleOccupancy
        && (config2 == HubbardSingleSiteState::SpinUp || config2 == HubbardSingleSiteState::SpinDown))
        || (config2 == HubbardSingleSiteState::DoubleOccupancy
            && (config1 == HubbardSingleSiteState::SpinUp || config1 == HubbardSingleSiteState::SpinDown))) {
      //3 electrons case
      TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                              split_index_tps_on_site1[config2],
                                              split_index_tps_on_site2[config1]);
      TenElemT ratio = ComplexConjugate(psi_ex / psi);
      return t * ratio;
    } else if (config1 == HubbardSingleSiteState::SpinUp && config2 == HubbardSingleSiteState::SpinDown) {
      TenElemT psi_ex1 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::Empty)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::DoubleOccupancy)]);
      TenElemT psi_ex2 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::DoubleOccupancy)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::Empty)]);
      TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi);
      TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi);
      return (-t) * (ratio1 + ratio2);
    } else if (config1 == HubbardSingleSiteState::SpinDown && config2 == HubbardSingleSiteState::SpinUp) {
      TenElemT psi_ex1 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::Empty)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::DoubleOccupancy)]);
      TenElemT psi_ex2 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::DoubleOccupancy)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::Empty)]);
      TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi);
      TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi);
      return t * (ratio1 + ratio2);
    } else { // |Double Occupancy, Empty> or |Empty, Double Occupancy>
      TenElemT psi_ex1 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::SpinUp)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::SpinDown)]);
      TenElemT psi_ex2 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                               split_index_tps_on_site1[size_t(HubbardSingleSiteState::SpinDown)],
                                               split_index_tps_on_site2[size_t(HubbardSingleSiteState::SpinUp)]);
      TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi);
      TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi);
      return -t * ratio1 + t * ratio2;
    }
  }
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType, bool calchols>
TenElemT SquareHubbardModel<TenElemT, QNT>::CalEnergyAndHoles(const SITPS *split_index_tps,
                                                              WaveFunctionComponentType *tps_sample,
                                                              TensorNetwork2D<TenElemT, QNT> &hole_res) {
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = WaveFunctionComponentType::trun_para;
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL)); // natural match to complex number wave-function case.
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        energy += EvaluateBondHoppingEnergyForHubbardModel(site1, site2,
                                                           HubbardSingleSiteState(config(site1)),
                                                           HubbardSingleSiteState(config(site2)),
                                                           HORIZONTAL,
                                                           tn,
                                                           (*split_index_tps)(site1), (*split_index_tps)(site2),
                                                           t_);
        tn.BTenMoveStep(RIGHT);
      }
    }
    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      energy += EvaluateBondHoppingEnergyForHubbardModel(site1, site2,
                                                         HubbardSingleSiteState(config(site1)),
                                                         HubbardSingleSiteState(config(site2)),
                                                         VERTICAL,
                                                         tn,
                                                         (*split_index_tps)(site1), (*split_index_tps)(site2),
                                                         t_);
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
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
  energy += num_double_occupancy * U_;
  size_t electron_num = num_double_occupancy * 2 + num_single_occupancy;
  energy += (-mu_) * electron_num;
  return energy;
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType>
ObservablesLocal<TenElemT> SquareHubbardModel<TenElemT, QNT>::SampleMeasure(
    const SITPS *split_index_tps,
    WaveFunctionComponentType *tps_sample
) {
  ObservablesLocal<TenElemT> res;
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const size_t lx = tn.cols(), ly = tn.rows();
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = WaveFunctionComponentType::trun_para.value();
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    for (size_t col = 0; col < tn.cols() - 1; col++) {
      //Calculate horizontal bond energy contribution
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row, col + 1};
      energy += EvaluateBondHoppingEnergyForHubbardModel(site1, site2,
                                                         HubbardSingleSiteState(config(site1)),
                                                         HubbardSingleSiteState(config(site2)),
                                                         HORIZONTAL,
                                                         tn,
                                                         (*split_index_tps)(site1), (*split_index_tps)(site2),
                                                         t_);
      tn.BTenMoveStep(RIGHT);
    }
    //measure correlation along the middle horizontal line
    if (row == tn.rows() / 2) {
      res.two_point_functions_loc.reserve(lx);
      SiteIdx site1 = {row, lx / 4};
      const size_t config1 = config(site1);

      // density correlation
      double n0 = HubbardConfig2Density(config1);
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        double n1 = HubbardConfig2Density(config(site2));
        res.two_point_functions_loc.push_back(n0 * n1);
      }
      // sz(i) * sz(j)
      double sz0 = HubbardConfig2Spinz(config1);
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        double sz1 = HubbardConfig2Spinz(config(site2));
        res.two_point_functions_loc.push_back(sz0 * sz1);
      }
    }

    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    tn.InitBTen(UP, col);
    tn.GrowFullBTen(DOWN, col, 2, true);
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      energy += EvaluateBondHoppingEnergyForHubbardModel(site1, site2,
                                                         HubbardSingleSiteState(config(site1)),
                                                         HubbardSingleSiteState(config(site2)),
                                                         VERTICAL,
                                                         tn,
                                                         (*split_index_tps)(site1), (*split_index_tps)(site2),
                                                         t_);
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
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
  energy += num_double_occupancy * U_;
  size_t electron_num = num_double_occupancy * 2 + num_single_occupancy;
  energy += (-mu_) * electron_num;
  res.energy_loc = energy;
  res.one_point_functions_loc.reserve(2 * tn.rows() * tn.cols());
  //charge
  for (auto &spin_config : config) {
    res.one_point_functions_loc.push_back(ElectronNum(HubbardSingleSiteState(spin_config)));
  }
  //spin
  for (auto &spin_config : config) {
    res.one_point_functions_loc.push_back(ElectronSpin(HubbardSingleSiteState(spin_config)));
  }
  return res;
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_HUBBARD_MODEL_H
