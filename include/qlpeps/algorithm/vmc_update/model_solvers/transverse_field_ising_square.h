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
#include <complex>
#include <cmath>

namespace qlpeps {
using namespace qlten;

/**
 * H = - sum_<i,j> sigma_i^z * sigma_j^z - h sum_i sigma_i^x
 * sigma^z & sigma^x are Pauli matrix with matrix element 1.
 */

class TransverseFieldIsingSquare : public ModelEnergySolver<TransverseFieldIsingSquare>,
                              public ModelMeasurementSolver<TransverseFieldIsingSquare> {
 public:
  TransverseFieldIsingSquare(void) = delete;

  TransverseFieldIsingSquare(double h) : h_(h) {}
  using ModelEnergySolver::CalEnergyAndHoles;
  using ModelMeasurementSolver<TransverseFieldIsingSquare>::EvaluateObservables;
  using ModelMeasurementSolver<TransverseFieldIsingSquare>::DescribeObservables;

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

  // Legacy SampleMeasureImpl removed under registry-only API

  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample
  ) {
    ObservableMap<TenElemT> out;
    std::vector<TenElemT> psi_list;

    // Local references
    auto &tn = tps_sample->tn;
    const Configuration &config = tps_sample->config;
    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
    const size_t lx = tn.cols();
    const size_t ly = tn.rows();

    // Prepare environment
    tn.GenerateBMPSApproach(UP, trunc_para);

    // Accumulators
    TenElemT energy_ex(0); // off-diagonal part
    std::vector<TenElemT> sigma_x;
    sigma_x.reserve(lx * ly);
    std::vector<TenElemT> two_point;
    two_point.reserve(lx / 2 * 3);

    for (size_t row = 0; row < ly; ++row) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 1, true);
      // push psi at row-begin for consistency check
      auto psi = tn.Trace({row, 0}, HORIZONTAL);
      psi_list.push_back(psi);
      auto inv_psi = TenElemT(1.0) / psi;
      for (size_t col = 0; col < lx; ++col) {
        const SiteIdx site{row, col};
        TenElemT ex_term = EvaluateOnSiteOffDiagEnergy(site, config(site), tn, (*sitps)(site), inv_psi);
        energy_ex += ex_term;
        if (h_ != 0.0) {
          sigma_x.push_back((-ex_term) / static_cast<double>(h_));
        } else {
          sigma_x.push_back(TenElemT(0));
        }
        if (col < lx - 1) {
          tn.BTenMoveStep(RIGHT);
        }
      }
      if (row == ly / 2) {
        // simple SzSz along middle row
        SiteIdx site1{row, lx / 4};
        double sz1 = config(site1) - 0.5;
        for (size_t i = 1; i <= lx / 2; ++i) {
          SiteIdx site2{row, lx / 4 + i};
          double sz2 = config(site2) - 0.5;
          two_point.push_back(sz1 * sz2);
        }
      }
      if (row < ly - 1) {
        tn.BMPSMoveStep(DOWN, trunc_para);
      }
    }

    // Diagonal zz energy
    TenElemT energy_diag = CalDiagTermEnergy<TenElemT>(config);
    out["energy"] = {energy_ex + energy_diag};

    // spin_z (Ly x Lx)
    {
      std::vector<TenElemT> spin_z;
      spin_z.reserve(ly * lx);
      for (auto &s : config) { spin_z.push_back(static_cast<double>(s) - 0.5); }
      out["spin_z"] = std::move(spin_z);
    }

    if (!sigma_x.empty()) out["sigma_x"] = std::move(sigma_x);
    if (!two_point.empty()) out["SzSz_row"] = std::move(two_point);
    // psi_list is not emitted via registry; Measurer computes PsiSummary separately

    // Diagonal bond energies omitted by default; can be enabled if needed later
    return out;
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    return {
        {"energy", "Total energy (scalar)", {}, {}},
        {"spin_z", "Local spin Sz per site (Ly,Lx)", {ly, lx}, {"y","x"}},
        {"sigma_x", "Transverse magnetisation per site (Ly,Lx)", {ly, lx}, {"y","x"}},
        {"SzSz_row", "SzSz correlations along middle row (flat)", {lx / 2}, {"segment"}}
    };
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

  
 private:
  double h_;
};

template<typename TenElemT, typename QNT, bool calchols>
TenElemT TransverseFieldIsingSquare::CalEnergyAndHolesImplParsed(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
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

// Legacy SampleMeasureImpl(out-of-class) removed under registry-only API
}//qlpeps
#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_TRANSVERSE_FIELD_ISING_SQUARE_H
