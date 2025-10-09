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
 * Spinless Fermion Model on Square Lattice  
 * 
 * Hamiltonian:
 * $$H = -t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + c_j^\dagger c_i) 
 *       -t_2 \sum_{\langle\langle i,j \rangle\rangle} (c_i^\dagger c_j + c_j^\dagger c_i)
 *       + V \sum_{\langle i,j \rangle} n_i n_j$$
 * 
 * where:
 * - t: nearest-neighbor hopping amplitude
 * - t₂: next-nearest-neighbor hopping amplitude  
 * - V: nearest-neighbor density-density interaction
 * - c_i†/c_i: spinless fermion creation/annihilation operators at site i
 * - n_i = c_i† c_i: particle number operator (0 or 1 for fermions)
 * - ⟨i,j⟩: nearest-neighbor pairs (horizontal/vertical bonds)
 * - ⟨⟨i,j⟩⟩: next-nearest-neighbor pairs (diagonal bonds)
 * 
 * Single-site states (basis encoding):
 * - 0: occupied site |1⟩ (fermion present)
 * - 1: empty site |0⟩ (no fermion)
 * 
 * Physical interpretation:
 * - First term: kinetic energy allowing fermion hopping between NN sites
 * - Second term: kinetic energy (NNN hopping)  
 * - Third term: repulsive interaction between neighboring fermions
 * - No chemical potential term included (particle number not fixed)
 * 
 * Applications:
 * - Quantum dots and artificial lattices
 * - Cold atom systems in optical lattices
 * - Effective models for certain correlated materials
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

  std::vector<ObservableMeta> DescribeObservables() const {
    auto base = SquareNNNModelMeasurementSolver<SquareSpinlessFermion>::DescribeObservables();
    for (auto &meta : base) {
      if (meta.key == "charge" || meta.key == "spin_z") {
        meta.shape = {0, 0};
        meta.index_labels = {"y", "x"};
      }
      if (meta.key == "bond_energy_h" || meta.key == "bond_energy_v" || meta.key == "bond_energy_diag") {
        meta.index_labels = {"bond_id"};
      }
    }
    return base;
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
