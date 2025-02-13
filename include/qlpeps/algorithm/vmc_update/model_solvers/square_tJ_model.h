/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for the t-J model in square lattice
*
* Hamiltonian :
 * H = \sum_{<i,j>,sigma} (c_{i,\sigma}^dag c_{j,\sigma} + h.c.)
 *    +\sum_{<i,j>} S_i \cdot S_j
 *    - \mu N
*/
#ifndef QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_TJ_MODEL_H
#define QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_TJ_MODEL_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      //ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               //ComplexConjugate
namespace qlpeps {
using namespace qlten;

enum class tJSingleSiteState {
  SpinUp,           // 0
  SpinDown,         // 1
  Empty             // 2
};

template<typename TenElemT, typename QNT>
class SquaretJModel : public ModelEnergySolver<TenElemT, QNT>, ModelMeasurementSolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SquaretJModel(void) = delete;
  explicit SquaretJModel(double t, double J, bool has_nn_term, double mu)
      : t_(t), J_(J), has_nn_term_(has_nn_term), mu_(mu) {}

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
  double J_;
  bool has_nn_term_;
  double mu_;

  const double tJ_wave_function_consistent_critical_bias = 1E-3;
};

double tJConfig2Density(const size_t config) {
  return (tJSingleSiteState(config) == tJSingleSiteState::Empty) ? 0.0 : 1.0;
}
double tJConfig2Spinz(const size_t config) {
  switch (tJSingleSiteState(config)) {
    case tJSingleSiteState::SpinUp: {
      return 0.5;
    }
    case tJSingleSiteState::SpinDown: {
      return -0.5;
    }
    case tJSingleSiteState::Empty: {
      return 0.0;
    }
    default: {
      std::cerr << "Unexpected config : " << config << " in t-J model" << std::endl;
      exit(1);
    }
  }
}
template<typename TenElemT, typename QNT>
TenElemT EvaluateBondEnergyFortJModel(
    const SiteIdx site1, const SiteIdx site2,
    const tJSingleSiteState config1, const tJSingleSiteState config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    double t, double J, bool has_nn_term,
    std::optional<TenElemT> &psi // return value, used for check the accuracy
) {
  if (config1 == config2) {
    psi.reset();
    if (config1 == tJSingleSiteState::Empty) {
      return 0.0;
    } else {
      return J * (0.25 - double(int(has_nn_term)) / 4.0); // sz * sz - 1/4 * n * n
    }
  } else {
    psi = tn.Trace(site1, site2, orient);
    if (psi == TenElemT(0)) [[unlikely]] {
      std::cerr << "Error: psi is 0. Division by 0 is not allowed." << std::endl;
      exit(EXIT_FAILURE);
    }

    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[size_t(config2)],
                                            split_index_tps_on_site2[size_t(config1)]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
    if (is_nan(ratio)) [[unlikely]] {
      std::cerr << "ratio is nan !" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (config1 == tJSingleSiteState::Empty || config2 == tJSingleSiteState::Empty) {
      // one site empty, the other site filled
      // only hopping energy contribution
      return (-t) * ratio;
    } else {
      // spin antiparallel
      // only spin interaction energy contribution
      return (-0.25 + ratio * 0.5 - int(has_nn_term) / 4.0) * J;
    }
  }
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType, bool calchols>
TenElemT SquaretJModel<TenElemT, QNT>::CalEnergyAndHoles(const SITPS *split_index_tps,
                                                         WaveFunctionComponentType *tps_sample,
                                                         TensorNetwork2D<TenElemT, QNT> &hole_res) {
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = WaveFunctionComponentType::trun_para.value();
  tn.GenerateBMPSApproach(UP, trunc_para);
  std::vector<TenElemT> psi_gather;
  psi_gather.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    bool psi_added = false;
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL)); // natural match to complex number wave-function case.
      }
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        std::optional<TenElemT> psi;
        energy += EvaluateBondEnergyFortJModel(site1, site2,
                                               tJSingleSiteState(config(site1)),
                                               tJSingleSiteState(config(site2)),
                                               HORIZONTAL,
                                               tn,
                                               (*split_index_tps)(site1), (*split_index_tps)(site2),
                                               t_, J_, has_nn_term_,
                                               psi);
        if (!psi_added && psi.has_value()) {
          psi_gather.push_back(psi.value());
          psi_added = true;
        }
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
    bool psi_added = false;
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      std::optional<TenElemT> psi;
      energy += EvaluateBondEnergyFortJModel(site1, site2,
                                             tJSingleSiteState(config(site1)),
                                             tJSingleSiteState(config(site2)),
                                             VERTICAL,
                                             tn,
                                             (*split_index_tps)(site1), (*split_index_tps)(site2),
                                             t_, J_, has_nn_term_, psi);
      if (!psi_added && psi.has_value()) {
        psi_gather.push_back(psi.value());
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
  if (mu_ != 0) {
    size_t ele_num(0);
    for (auto &spin : config) {
      if (spin != 2) {
        ele_num++;
      }
    }
    energy += -mu_ * double(ele_num);
  }
  WaveFunctionAmplitudeConsistencyCheck(psi_gather, tJ_wave_function_consistent_critical_bias);
  return energy;
}

template<typename TenElemT, typename QNT>
std::pair<TenElemT, TenElemT> EvaluateBondSingletPairFortJModel(const SiteIdx site1,
                                                                const SiteIdx site2,
                                                                const tJSingleSiteState config1,
                                                                const tJSingleSiteState config2,
                                                                const BondOrientation orient,
                                                                const TensorNetwork2D<TenElemT, QNT> &tn,
                                                                const std::vector<QLTensor<TenElemT,
                                                                                           QNT>> &split_index_tps_on_site1,
                                                                const std::vector<QLTensor<TenElemT,
                                                                                           QNT>> &split_index_tps_on_site2
) {
  TenElemT delta_dag, delta;
  if (config1 == tJSingleSiteState::Empty && config2 == tJSingleSiteState::Empty) {
    TenElemT psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex1 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                             split_index_tps_on_site1[tJSingleSiteState::SpinUp],
                                             split_index_tps_on_site2[tJSingleSiteState::SpinDown]);
    TenElemT psi_ex2 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                             split_index_tps_on_site1[tJSingleSiteState::SpinDown],
                                             split_index_tps_on_site2[tJSingleSiteState::SpinUp]);
    TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi);
    TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi);
    delta_dag = (ratio1 - ratio2) / std::sqrt(2);
    delta = 0;
    return std::make_pair(delta_dag, delta);
  } else if (config1 == tJSingleSiteState::SpinUp && config2 == tJSingleSiteState::SpinDown) {
    delta_dag = 0;
    TenElemT psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[tJSingleSiteState::Empty],
                                            split_index_tps_on_site2[tJSingleSiteState::Empty]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi);
    delta = -ratio / std::sqrt(2);
    return std::make_pair(delta_dag, delta);
  } else if (config1 == tJSingleSiteState::SpinDown && config2 == tJSingleSiteState::SpinUp) {
    delta_dag = 0;
    TenElemT psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[tJSingleSiteState::Empty],
                                            split_index_tps_on_site2[tJSingleSiteState::Empty]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi);
    delta = ratio / std::sqrt(2);
    return std::make_pair(delta_dag, delta);
  } else {
    return std::make_pair(TenElemT(0), TenElemT(0));
  }
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType>
ObservablesLocal<TenElemT> SquaretJModel<TenElemT, QNT>::SampleMeasure(
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
  std::vector<TenElemT> delta_dag, delta;
  delta_dag.reserve(lx * ly * 2); // bond singlet pair
  delta.reserve(lx * ly * 2);  // bond singlet pair
  std::vector<TenElemT> psi_gather;
  psi_gather.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    bool psi_added = false;
    for (size_t col = 0; col < tn.cols() - 1; col++) {
      //Calculate horizontal bond energy contribution
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row, col + 1};
      std::optional<TenElemT> psi;
      TenElemT bond_energy = EvaluateBondEnergyFortJModel(site1, site2,
                                                          tJSingleSiteState(config(site1)),
                                                          tJSingleSiteState(config(site2)),
                                                          HORIZONTAL,
                                                          tn,
                                                          (*split_index_tps)(site1), (*split_index_tps)(site2),
                                                          t_, J_, has_nn_term_, psi);
      res.bond_energys_loc.push_back(bond_energy);
      energy += bond_energy;
      if (!psi_added && psi.has_value()) {
        psi_gather.push_back(psi.value());
        psi_added = true;
      }
      tn.BTenMoveStep(RIGHT);
    }
    //measure correlation along the middle horizontal line
    if (row == tn.rows() / 2) {
      res.two_point_functions_loc.reserve(lx);
      SiteIdx site1 = {row, lx / 4};
      const size_t config1 = config(site1);

      // density correlation
      double n0 = tJConfig2Density(config1);
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        double n1 = tJConfig2Density(config(site2));
        res.two_point_functions_loc.push_back(n0 * n1);
      }
      // sz(i) * sz(j)
      double sz0 = tJConfig2Spinz(config1);
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        double sz1 = tJConfig2Spinz(config(site2));
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
    bool psi_added = false;
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      std::optional<TenElemT> psi;
      TenElemT bond_energy = EvaluateBondEnergyFortJModel(site1, site2,
                                                          tJSingleSiteState(config(site1)),
                                                          tJSingleSiteState(config(site2)),
                                                          VERTICAL,
                                                          tn,
                                                          (*split_index_tps)(site1), (*split_index_tps)(site2),
                                                          t_, J_, has_nn_term_, psi);
      res.bond_energys_loc.push_back(bond_energy);
      energy += bond_energy;
      if (!psi_added && psi.has_value()) {
        psi_gather.push_back(psi.value());
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
  assert(!is_nan(energy));
  if (mu_ != 0) {
    size_t ele_num(0);
    for (auto &spin : config) {
      ele_num += (spin != 2);
    }
    energy += -mu_ * double(ele_num);
  }
  res.energy_loc = energy;
  res.one_point_functions_loc.reserve(2 * tn.rows() * tn.cols());
  //charge
  for (const auto &spin_config : config) {
    res.one_point_functions_loc.push_back((spin_config == 2) ? 0.0 : 1.0);
  }
  for (const auto &spin_config : config) {
    if (spin_config == 0) {
      res.one_point_functions_loc.push_back(0.5); // spin up
    } else if (spin_config == 1) {
      res.one_point_functions_loc.push_back(-0.5);//spin down
    } else {
      res.one_point_functions_loc.push_back(0);   //empty state
    }
  }
  WaveFunctionAmplitudeConsistencyCheck(psi_gather, tJ_wave_function_consistent_critical_bias);
  return res;
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_TJ_MODEL_H
