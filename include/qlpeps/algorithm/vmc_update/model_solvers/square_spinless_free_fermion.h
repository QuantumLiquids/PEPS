/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for the spinless free-fermion in square lattice
*/
#ifndef QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION
#define QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      //ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               //ComplexConjugate

namespace qlpeps {
using namespace qlten;

/**
 * only include the nearest-neighbor hopping
 *  H = -t \sum_{<i,j>} c_i^dag c_j + h.c.
 *  with t = 1
 *
 * 0 for filled, 1 for empty
 */
template<typename TenElemT, typename QNT>
class SquareSpinlessFreeFermion : public ModelEnergySolver<TenElemT, QNT>, ModelMeasurementSolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SquareSpinlessFreeFermion(void) = default;

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
};

double SpinlessFermionConfig2Density(const size_t config) {
  return double(1 - config);
}

template<typename TenElemT, typename QNT>
TenElemT EvaluateBondEnergyForSpinlessFreeFermion(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2
) {
  if (config1 == config2) {
    return 0.0;
  } else {// one site empty, the other site filled
    TenElemT psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[config2],
                                            split_index_tps_on_site2[config1]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi);
    return -ratio;
  }
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType, bool calchols>
TenElemT SquareSpinlessFreeFermion<TenElemT, QNT>::CalEnergyAndHoles(const SITPS *split_index_tps,
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
        energy += EvaluateBondEnergyForSpinlessFreeFermion(site1, site2,
                                                           config(site1), config(site2),
                                                           HORIZONTAL,
                                                           tn,
                                                           (*split_index_tps)(site1), (*split_index_tps)(site2));
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
      energy += EvaluateBondEnergyForSpinlessFreeFermion(site1, site2,
                                                         config(site1), config(site2),
                                                         VERTICAL,
                                                         tn,
                                                         (*split_index_tps)(site1), (*split_index_tps)(site2));
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
ObservablesLocal<TenElemT> SquareSpinlessFreeFermion<TenElemT, QNT>::SampleMeasure(
    const SITPS *split_index_tps,
    WaveFunctionComponentType *tps_sample
) {
  ObservablesLocal<TenElemT> res;
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const size_t lx = tn.cols(), ly = tn.rows();
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = WaveFunctionComponentType::trun_para;
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    for (size_t col = 0; col < tn.cols() - 1; col++) {
      //Calculate horizontal bond energy contribution
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row, col + 1};
      energy += EvaluateBondEnergyForSpinlessFreeFermion(site1, site2,
                                                         config(site1), config(site2),
                                                         HORIZONTAL,
                                                         tn,
                                                         (*split_index_tps)(site1), (*split_index_tps)(site2));
      tn.BTenMoveStep(RIGHT);
    }
    //measure correlation along the middle horizontal line
    if (row == tn.rows() / 2) {
      res.two_point_functions_loc.reserve(lx);
      SiteIdx site1 = {row, lx / 4};
      const size_t config1 = config(site1);

      // density correlation
      double n0 = SpinlessFermionConfig2Density(config1);
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        double n1 = SpinlessFermionConfig2Density(config(site2));
        res.two_point_functions_loc.push_back(n0 * n1);
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
      energy += EvaluateBondEnergyForSpinlessFreeFermion(site1, site2,
                                                         config(site1), config(site2),
                                                         VERTICAL,
                                                         tn,
                                                         (*split_index_tps)(site1), (*split_index_tps)(site2));
      if (row < tn.rows() - 2) {
        tn.BTenMoveStep(DOWN);
      }
    }
    if (col < tn.cols() - 1) {
      tn.BMPSMoveStep(RIGHT, trunc_para);
    }
  }
  res.energy_loc = energy;
  res.one_point_functions_loc.reserve(tn.rows() * tn.cols());
  //charge
  for (auto &site_config : config) {
    res.one_point_functions_loc.push_back(SpinlessFermionConfig2Density(site_config));
  }
  return res;
}

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION
