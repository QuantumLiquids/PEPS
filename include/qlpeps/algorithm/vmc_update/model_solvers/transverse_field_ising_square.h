/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-08-02
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 FM transverse-field Ising model in square lattice
*/


#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      //ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               //ComplexConjugate

namespace qlpeps {
using namespace qlten;

/**
 * H = - sum_<i,j> sigma_i^z * sigma_j^z - h sum_i sigma_i^x
 * sigma^z & sigma^x are Pauli matrix with matrix element 1.
 */
template<typename TenElemT, typename QNT>
class TransverseIsingSquare : public ModelEnergySolver<TenElemT, QNT>, ModelMeasurementSolver<TenElemT, QNT> {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  TransverseIsingSquare(void) = delete;

  TransverseIsingSquare(double h) : h_(h) {}

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
  double h_;
};

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType, bool calchols>
TenElemT TransverseIsingSquare<TenElemT, QNT>::CalEnergyAndHoles(const SITPS *split_index_tps,
                                                                 WaveFunctionComponentType *tps_sample,
                                                                 TensorNetwork2D<TenElemT, QNT> &hole_res) {
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = WaveFunctionComponentType::trun_para.value();
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  std::vector<TenElemT> psi_gather;
  psi_gather.reserve(tn.rows() + tn.cols());
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    // update the amplitude so that the error of ratio of amplitude can reduce by cancellation.
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_gather.push_back(tps_sample->amplitude);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site1) = Dag(tn.PunchHole(site1, HORIZONTAL)); // natural match to complex number wave-function case.
      }
      //transverse-field terms
      TenElemT psi_ex = tn.ReplaceOneSiteTrace(site1,
                                               (*split_index_tps)(site1)[1 - config(site1)],
                                               HORIZONTAL);
      TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
      energy += (-h_) * ratio;
      //zz terms
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        energy += (config(site1) == config(site2)) ? -1 : 1;
        tn.BTenMoveStep(RIGHT);
      }
    }
    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  //vertical bond energy contribution
  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      energy += (config(site1) == config(site2)) ? -1 : 1;
    }
  }
  WaveFunctionAmplitudeConsistencyCheck(psi_gather, 0.01);
  return energy;
}

template<typename TenElemT, typename QNT>
template<typename WaveFunctionComponentType>
ObservablesLocal<TenElemT> TransverseIsingSquare<TenElemT, QNT>::SampleMeasure(
    const SITPS *split_index_tps,
    WaveFunctionComponentType *tps_sample
) {
  ObservablesLocal<TenElemT> res;
  TenElemT energy(0);
  TensorNetwork2D<TenElemT, QNT> &tn = tps_sample->tn;
  const size_t lx = tn.cols(), ly = tn.rows();
  res.bond_energys_loc.reserve(lx * ly * 2);
  res.two_point_functions_loc.reserve(lx / 2 * 3);
  const Configuration &config = tps_sample->config;
  const BMPSTruncatePara &trunc_para = WaveFunctionComponentType::trun_para.value();
  TenElemT inv_psi = 1.0 / (tps_sample->amplitude);
  tn.GenerateBMPSApproach(UP, trunc_para);
  std::vector<TenElemT> psi_gather;
  psi_gather.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < ly; row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    // update the amplitude so that the error of ratio of amplitude can reduce by cancellation.
    tps_sample->amplitude = tn.Trace({row, 0}, HORIZONTAL);
    inv_psi = 1.0 / tps_sample->amplitude;
    psi_gather.push_back(tps_sample->amplitude);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site1 = {row, col};
      //transverse-field terms
      TenElemT psi_ex = tn.ReplaceOneSiteTrace(site1,
                                               (*split_index_tps)(site1)[1 - config(site1)],
                                               HORIZONTAL);
      TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
      energy += (-h_) * ratio;
      //zz terms
      if (col < tn.cols() - 1) {
        //Calculate horizontal bond energy contribution
        const SiteIdx site2 = {row, col + 1};
        energy += (config(site1) == config(site2)) ? -1 : 1;
        tn.BTenMoveStep(RIGHT);
      }
    }
    if (row == tn.rows() / 2) { //measure correlation in the middle bonds
      SiteIdx site1 = {row, lx / 4};

      // sz(i) * sz(j)
      double sz1 = config(site1) - 0.5;
      for (size_t i = 1; i <= lx / 2; i++) {
        SiteIdx site2 = {row, lx / 4 + i};
        double sz2 = config(site2) - 0.5;
        res.two_point_functions_loc.push_back(sz1 * sz2);
      }
    }
    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }

  //vertical bond energy contribution
  tn.GenerateBMPSApproach(LEFT, trunc_para);
  for (size_t col = 0; col < tn.cols(); col++) {
    for (size_t row = 0; row < tn.rows() - 1; row++) {
      const SiteIdx site1 = {row, col};
      const SiteIdx site2 = {row + 1, col};
      energy += (config(site1) == config(site2)) ? -1 : 1;
    }
  }
  res.energy_loc = energy;
  res.one_point_functions_loc.reserve(tn.rows() * tn.cols());
  for (auto &spin_config : config) {
    res.one_point_functions_loc.push_back((double) spin_config - 0.5);
  }
  WaveFunctionAmplitudeConsistencyCheck(psi_gather, 0.01);
  return res;
}
}//qlpeps
#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_H
