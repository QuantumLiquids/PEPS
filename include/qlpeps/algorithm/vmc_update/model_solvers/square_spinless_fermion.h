/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for the spinless fermion in square lattice
*/
#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION

#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_model_measurement_solver.h"
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate

namespace qlpeps {
using namespace qlten;

/**
 * only include the nearest-neighbor hopping
 *  H = -t \sum_{<i,j>} c_i^dag c_j + h.c.
 *      -t_2 \sum_{<<i,j>>} c_i^dag c_j + h.c.
 *      +V \sum_{<i,j>} n_i^dag n_j
 *
 * 0 for filled, 1 for empty
 */

class SquareSpinlessFermion : public SquareNNNModelEnergySolver<SquareSpinlessFermion>,
                              public SquareNNNModelMeasurementSolver<SquareSpinlessFermion> {

 public:
  static constexpr bool requires_density_measurement = true;
  static constexpr bool requires_spin_sz_measurement = false;
  SquareSpinlessFermion(const double t, const double V) : t_(t), t2_(0), V_(V) {};
  SquareSpinlessFermion(const double t, const double t2, const double V) : t_(t), t2_(t2), V_(V) {};

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

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateNNNEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const DIAGONAL_DIR diagonal_dir,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi // return value, used for check the accuracy
  );

  // do not consider the chemical potential
  constexpr double EvaluateTotalOnsiteEnergy(const Configuration &config) { return 0; }

  double CalDensityImpl(const size_t config) const {
    return double(1 - config);
  }
 private:
  const double t_; // nearest-neighbor hopping amplitude
  const double t2_;// next-nearest-neighbor hopping amplitude
  const double V_; // nearest-neighbor density interaction, V * n_i * n_j
};

template<typename TenElemT, typename QNT>
TenElemT SquareSpinlessFermion::EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi
) {
  double n1 = this->CalDensityImpl(config1);
  double n2 = this->CalDensityImpl(config2);
  double e_intert = V_ * n1 * n2;
  if (config1 == config2) {
    psi.reset();
    return e_intert;
  } else {// one site empty, the other site filled
    psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[config2],
                                            split_index_tps_on_site2[config1]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
    return -t_ * ratio + e_intert;
  }
}

template<typename TenElemT, typename QNT>
TenElemT SquareSpinlessFermion::EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi // return value, used for check the accuracy
) {
  if (config1 == config2) {
    return 0;
  } else {// one site empty, the other site filled
    SiteIdx left_up_site;
    if (diagonal_dir == LEFTUP_TO_RIGHTDOWN) {
      left_up_site = site1;
    } else {
      left_up_site = {site2.row(), site1.col()};
    }

    if (!psi.has_value()) {
      psi = tn.ReplaceNNNSiteTrace(left_up_site,
                                   diagonal_dir,
                                   HORIZONTAL,
                                   split_index_tps_on_site1[size_t(config1)],
                                   split_index_tps_on_site2[size_t(config2)]);
    }
    TenElemT psi_ex = tn.ReplaceNNNSiteTrace(left_up_site,
                                             diagonal_dir,
                                             HORIZONTAL,
                                             split_index_tps_on_site1[size_t(config2)],
                                             split_index_tps_on_site2[size_t(config1)]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
    return -t2_ * ratio;
  }
}
}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_SPINLESS_FREE_FERMION
