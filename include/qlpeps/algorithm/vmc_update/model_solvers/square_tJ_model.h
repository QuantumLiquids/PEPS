/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for the t-J model in square lattice
*/
#ifndef QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_TJ_MODEL_H
#define QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_TJ_MODEL_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      //ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               //ComplexConjugate
namespace qlpeps {
using namespace qlten;

template<typename TenElemT, typename QNT>
class SquaretJModel : public ModelEnergySolver<TenElemT, QNT>, ModelMeasurementSolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SquaretJModel(void) = delete;
  SquaretJModel(double t, double J, bool has_nn_term) : t_(t), J_(J), has_nn_term_(has_nn_term) {}

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
};

double tJConfig2Density(const size_t config) {
  return (config == 2) ? 0.0 : 1.0;
}
double tJConfig2Spinz(const size_t config) {
  switch (config) {
    case 0: {
      return 0.5;
    }
    case 1: {
      return -0.5;
    }
    case 2: {
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
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    double t, double J, bool has_nn_term
) {
  if (config1 == config2) {
    if (config1 == 2) {
      // two empty state, no energy contribution
      return 0.0;
    } else {
      return J * (0.25 - int(has_nn_term) / 4.0); // sz * sz - 1/4 * n * n
    }
  } else {
    TenElemT psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[config2],
                                            split_index_tps_on_site2[config1]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi);
    if (config1 == 2 || config2 == 2) {
      // one site empty, the other site filled
      // only hopping energy contribution
      return (-t) * ratio;
    } else {
      // spin anti-parallel
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
  std::vector<double> psi_abs_gather;
  psi_abs_gather.reserve(tn.rows() + tn.cols());
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
        energy += EvaluateBondEnergyFortJModel(site1, site2,
                                               config(site1), config(site2),
                                               HORIZONTAL,
                                               tn,
                                               (*split_index_tps)(site1), (*split_index_tps)(site2),
                                               t_, J_, has_nn_term_);
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
      energy += EvaluateBondEnergyFortJModel(site1, site2,
                                             config(site1), config(site2),
                                             VERTICAL,
                                             tn,
                                             (*split_index_tps)(site1), (*split_index_tps)(site2),
                                             t_, J_, has_nn_term_);
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  return energy;
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
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    for (size_t col = 0; col < tn.cols() - 1; col++) {
      //Calculate horizontal bond energy contribution
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row, col + 1};
      TenElemT bond_energy = EvaluateBondEnergyFortJModel(site1, site2,
                                                          config(site1), config(site2),
                                                          HORIZONTAL,
                                                          tn,
                                                          (*split_index_tps)(site1), (*split_index_tps)(site2),
                                                          t_, J_, has_nn_term_);
      res.bond_energys_loc.push_back(bond_energy);
      energy += bond_energy;
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
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      TenElemT bond_energy = EvaluateBondEnergyFortJModel(site1, site2,
                                                          config(site1), config(site2),
                                                          VERTICAL,
                                                          tn,
                                                          (*split_index_tps)(site1), (*split_index_tps)(site2),
                                                          t_, J_, has_nn_term_);
      res.bond_energys_loc.push_back(bond_energy);
      energy += bond_energy;
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  assert(!is_nan(energy));
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
  return res;
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_TJ_MODEL_H
