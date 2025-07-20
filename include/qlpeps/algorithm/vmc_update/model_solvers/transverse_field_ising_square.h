/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-08-02
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 FM transverse-field Ising model in square lattice
*/


#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate

namespace qlpeps {
using namespace qlten;

/**
 * H = - sum_<i,j> sigma_i^z * sigma_j^z - h sum_i sigma_i^x
 * sigma^z & sigma^x are Pauli matrix with matrix element 1.
 */

class TransverseIsingSquare : public ModelEnergySolver<TransverseIsingSquare>,
                              public ModelMeasurementSolver<TransverseIsingSquare> {
 public:
  TransverseIsingSquare(void) = delete;

  TransverseIsingSquare(double h) : h_(h) {}
  using ModelEnergySolver::CalEnergyAndHoles;
  using ModelMeasurementSolver<TransverseIsingSquare>::operator();

  template<typename TenElemT, typename QNT, bool calchols>
  TenElemT CalEnergyAndHolesImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  ) {
    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
    const Configuration &sample_config = tps_sample->config;
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
    return this->template CalEnergyAndHolesImplParsed<TenElemT, QNT, calchols>(split_index_tps,
                                                                               sample_config,
                                                                               sample_tn,
                                                                               trunc_para,
                                                                               hole_res,
                                                                               psi_list);
  }

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> SampleMeasureImpl(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      std::vector<TenElemT> &psi_list
  ) {
    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
    const Configuration &sample_config = tps_sample->config;
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
    return this->SampleMeasureImpl(split_index_tps, sample_config, sample_tn, trunc_para, psi_list);
  }

  template<typename TenElemT, typename QNT, bool calchols>
  TenElemT CalEnergyAndHolesImplParsed(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      const Configuration &sample_config,
      TensorNetwork2D<TenElemT, QNT> &sample_tn,
      const BMPSTruncatePara &trunc_para,
      TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  );

  /** Diagonal terms, - sum_<i,j> sigma_i^z * sigma_j^z   **/
  template<typename TenElemT>
  TenElemT CalDiagTermEnergy(const Configuration &config) {
    double energy(0);

    //horizontal bond energy contribution
    for (size_t row = 0; row < config.rows(); row++) {
      for (size_t col = 0; col < config.cols() - 1; col++) {
        const SiteIdx site1 = {row, col};
        const SiteIdx site2 = {row, col + 1};
        energy += (config(site1) == config(site2)) ? -1 : 1;
      }
    }

    //vertical bond energy contribution
    for (size_t col = 0; col < config.cols(); col++) {
      for (size_t row = 0; row < config.rows() - 1; row++) {
        const SiteIdx site1 = {row, col};
        const SiteIdx site2 = {row + 1, col};
        energy += (config(site1) == config(site2)) ? -1 : 1;
      }
    }
    return TenElemT(energy);
  }

  /**
   * - h sigma_i^x
   * Assume the environment tensors for the site are
   * ready, with BMPS BondOrientation horizontal.
   *
   * @return
   */
  template<typename TenElemT, typename QNT>
  [[nodiscard]] TenElemT EvaluateOnSiteOffDiagEnergy(
      const SiteIdx site, const size_t config,
      const TensorNetwork2D<TenElemT, QNT> &tn, // environment info
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site,
      TenElemT inv_psi
  ) {
    TenElemT psi_ex = tn.ReplaceOneSiteTrace(site,
                                             split_index_tps_on_site[1 - config],
                                             HORIZONTAL);
    TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
    return (-h_) * ratio;
  }

  template<typename TenElemT, typename QNT>
  ObservablesLocal<TenElemT> SampleMeasureImpl(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      const Configuration &sample_config,
      TensorNetwork2D<TenElemT, QNT> &sample_tn,
      const BMPSTruncatePara &trunc_para,
      std::vector<TenElemT> &psi_list
  );
 private:
  double h_;
};

template<typename TenElemT, typename QNT, bool calchols>
TenElemT TransverseIsingSquare::CalEnergyAndHolesImplParsed(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                                                            const qlpeps::Configuration &config,
                                                            TensorNetwork2D<TenElemT, QNT> &tn,
                                                            const qlpeps::BMPSTruncatePara &trunc_para,
                                                            TensorNetwork2D<TenElemT, QNT> &hole_res,
                                                            std::vector<TenElemT> &psi_list) {
  TenElemT energy(0);
  psi_list.reserve(tn.rows() + tn.cols());
  tn.GenerateBMPSApproach(UP, trunc_para);
  for (size_t row = 0; row < tn.rows(); row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    // update the amplitude so that the error of ratio of amplitude can reduce by cancellation.
    auto psi = tn.Trace({row, 0}, HORIZONTAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site = {row, col};
      //Calculate the holes
      if constexpr (calchols) {
        hole_res(site) = Dag(tn.PunchHole(site, HORIZONTAL)); // natural match to complex number wave-function case.
      }
      //transverse-field terms
      energy += EvaluateOnSiteOffDiagEnergy(site, config(site), tn, (*split_index_tps)(site), inv_psi);
      if (col < tn.cols() - 1) {
        tn.BTenMoveStep(RIGHT);
      }
    }
    if (row < tn.rows() - 1) {
      tn.BMPSMoveStep(DOWN, trunc_para);
    }
  }
  energy += CalDiagTermEnergy<TenElemT>(config);

  return energy;
}

template<typename TenElemT, typename QNT>
ObservablesLocal<TenElemT> TransverseIsingSquare::SampleMeasureImpl(const SplitIndexTPS<TenElemT,
                                                                                        QNT> *split_index_tps,
                                                                    const qlpeps::Configuration &config,
                                                                    TensorNetwork2D<TenElemT, QNT> &tn,
                                                                    const qlpeps::BMPSTruncatePara &trunc_para,
                                                                    std::vector<TenElemT> &psi_list) {
  ObservablesLocal<TenElemT> res;
  TenElemT energy(0);
  const size_t lx = tn.cols(), ly = tn.rows();
  res.bond_energys_loc.reserve(lx * ly * 2);
  res.two_point_functions_loc.reserve(lx / 2 * 3);
  tn.GenerateBMPSApproach(UP, trunc_para);
  psi_list.reserve(tn.rows() + tn.cols());
  for (size_t row = 0; row < ly; row++) {
    tn.InitBTen(LEFT, row);
    tn.GrowFullBTen(RIGHT, row, 1, true);
    // update the amplitude so that the error of ratio of amplitude can reduce by cancellation.
    auto psi = tn.Trace({row, 0}, HORIZONTAL);
    auto inv_psi = 1.0 / psi;
    psi_list.push_back(psi);
    for (size_t col = 0; col < tn.cols(); col++) {
      const SiteIdx site = {row, col};
      //transverse-field terms
      energy += EvaluateOnSiteOffDiagEnergy(site, config(site), tn, (*split_index_tps)(site), inv_psi);
      //zz terms
      if (col < tn.cols() - 1) {
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

  energy += CalDiagTermEnergy<TenElemT>(config);
  res.energy_loc = energy;
  res.one_point_functions_loc.reserve(tn.rows() * tn.cols());
  for (auto &spin_config : config) {
    res.one_point_functions_loc.push_back((double) spin_config - 0.5);
  }
  return res;
}
}//qlpeps
#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_H
