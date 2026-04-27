/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-09-18
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for the t-J model in square lattice
*
* Hamiltonian :
 * H = -t\sum_{<i,j>,sigma} (c_{i,\sigma}^dag c_{j,\sigma} + h.c.)
 *     -t2\sum_{<<i,j>>,sigma} (c_{i,\sigma}^dag c_{j,\sigma} + h.c.)
 *     +J \sum_{<i,j>} (S_i \cdot S_j - 1/4 n_i \cdot n_j )
 *     - \mu N
 *
*/
#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_TJ_MODEL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_TJ_MODEL_H

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/singlet_pair_correlation_measurement_mixin.h"
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/vmc_basic/tj_single_site_state.h"

namespace qlpeps {
using namespace qlten;

namespace detail {
template<typename TenElemT, typename QNT>
std::pair<TenElemT, TenElemT> EvaluateBondSingletPairFortJModelWithPsi(
    const SiteIdx site1,
    const SiteIdx site2,
    const tJSingleSiteState config1,
    const tJSingleSiteState config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi);
}  // namespace detail

/**
 * t-J Model on Square Lattice with Extensions
 * 
 * Hamiltonian:
 * $$H = -t\sum_{\langle i,j\rangle,\sigma} (c_{i,\sigma}^\dagger c_{j,\sigma} + c_{j,\sigma}^\dagger c_{i,\sigma})$$
 * $$   -t_2\sum_{\langle\langle i,j\rangle\rangle,\sigma} (c_{i,\sigma}^\dagger c_{j,\sigma} + c_{j,\sigma}^\dagger c_{i,\sigma})$$  
 * $$   + J \sum_{\langle i,j\rangle} (\vec{S}_i \cdot \vec{S}_j - \frac{1}{4} n_i n_j)$$
 * $$   + V \sum_{\langle i,j\rangle} n_i n_j - \mu \sum_{i,\sigma} n_{i,\sigma}$$
 * An optional single-bond singlet pair pinning term
 * \f$\Delta(\hat{\Delta}_{ij}^{\dagger}+\hat{\Delta}_{ij})\f$ can be enabled
 * with SetSingletPairPinningField().
 * 
 * where:
 * - t: nearest-neighbor hopping amplitude
 * - t₂: next-nearest-neighbor hopping amplitude (extended t-J model)
 * - J: antiferromagnetic exchange coupling between neighboring spins
 * - V: nearest-neighbor density-density interaction
 * - μ: chemical potential
 * - c_{i,σ}†: creation operator with no double occupancy constraint
 * - n_i = n_{i,↑} + n_{i,↓}: total electron density at site i
 * - \vec{S}_i: spin-1/2 operator at site i
 * 
 * Single-site states (tJ basis):
 * - 0: spin up |↑⟩
 * - 1: spin down |↓⟩  
 * - 2: empty site |0⟩
 * Note: Double occupancy |↑↓⟩ is excluded by strong correlation constraint
 * 
 * Physical regime and motivation:
 * - Derived from Hubbard model in strong coupling limit (U >> t)
 * - Describes doped Mott insulators (e.g., cuprate superconductors)
 * - J ≈ 4t²/U from virtual hopping processes
 * - Captures essential physics: kinetic energy vs magnetic order
 * 
 * Special case: if V = J/4, the V term exactly cancels the density 
 * interaction in J term, yielding pure spin exchange for occupied sites.
 */
class SquaretJModelMixIn {
  struct SingletPairPinningField {
    SiteIdx site1;
    SiteIdx site2;
    BondOrientation orient;
    double delta;
  };

 public:
  static constexpr bool requires_density_measurement = true;
  static constexpr bool requires_spin_sz_measurement = true;
  inline static bool enable_sc_measurement = false;
  SquaretJModelMixIn(void) = delete;

  explicit SquaretJModelMixIn(double t, double t2, double J, double V, double mu)
      : t_(t), t2_(t2), J_(J), V_(V), mu_(mu) {}

  /**
   * Add a single NN-bond singlet pair source term to the Hamiltonian.
   *
   * This term breaks particle-number conservation, while preserving fermion
   * parity. Callers must avoid particle-number U(1) tensor symmetries and
   * number-conserving MC updaters at the application layer.
   */
  void SetSingletPairPinningField(SiteIdx site1, SiteIdx site2, double delta) {
    if (!std::isfinite(delta)) {
      throw std::invalid_argument("Singlet pair pinning delta must be finite.");
    }
    const auto dist = [](size_t a, size_t b) { return (a > b) ? (a - b) : (b - a); };
    const size_t row_dist = dist(site1.row(), site2.row());
    const size_t col_dist = dist(site1.col(), site2.col());
    if (row_dist + col_dist != 1) {
      throw std::invalid_argument("Singlet pair pinning field requires a nearest-neighbor bond.");
    }

    BondOrientation orient;
    if (row_dist == 0) {
      orient = HORIZONTAL;
      if (site2.col() < site1.col()) {
        std::swap(site1, site2);
      }
    } else {
      orient = VERTICAL;
      if (site2.row() < site1.row()) {
        std::swap(site1, site2);
      }
    }
    singlet_pair_pinning_ = SingletPairPinningField{site1, site2, orient, delta};
  }

  void ClearSingletPairPinningField() { singlet_pair_pinning_.reset(); }

  bool HasSingletPairPinningField() const { return singlet_pair_pinning_.has_value(); }

  ///< requirement from SquareNNNModelEnergySolver and SquareNNModelEnergySolver
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi // return value, used for check the accuracy
  );

  ///< requirement from SquareNNNModelEnergySolverJastrowDressed
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi, // return value, used for check the accuracy，
      const JastrowDress &jastrow_dress
  );

  ///< requirement from SquareNNNModelEnergySolver
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateNNNEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const DIAGONAL_DIR diagonal_dir,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi // return value, used for check the accuracy
  );

  ///< requirement from SquareNNNModelEnergySolver
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateNNNEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const DIAGONAL_DIR diagonal_dir,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi, // return value, used for check the accuracy
      const JastrowDress &jastrow_dress
  );

    ///< Here psi seems to be a redundant variable.
  template<typename TenElemT, typename QNT>
  std::pair<TenElemT, TenElemT> EvaluateBondSC(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi
  ) {
    return detail::EvaluateBondSingletPairFortJModelWithPsi(site1, site2,
                                                            tJSingleSiteState(config1),
                                                            tJSingleSiteState(config2),
                                                            orient, tn, contractor,
                                                            split_index_tps_on_site1,
                                                            split_index_tps_on_site2,
                                                            psi);
  }

  ///< requirement from SquareNNNModelEnergySolver
  double EvaluateTotalOnsiteEnergy(const Configuration &config) {
    ValidateSingletPairPinningBondInLattice_(config.rows(), config.cols());
    double energy = 0;
    if (mu_ != 0) {
      size_t ele_num(0);
      for (auto &spin : config) {
        if (spin != 2) {
          ele_num++;
        }
      }
      energy += -mu_ * double(ele_num);
    }
    return energy;
  }

  double CalDensityImpl(const size_t config) const {
    return (config == static_cast<size_t>(tJSingleSiteState::Empty)) ? 0.0 : 1.0;
  }

  double CalSpinSzImpl(const size_t config) const {
    return (config == static_cast<size_t>(tJSingleSiteState::Empty)) ? 0.0 : (config
        == static_cast<size_t>(tJSingleSiteState::SpinUp)) ? 0.5 : -0.5;
  }

 private:
  void ValidateSingletPairPinningBondInLattice_(const size_t rows, const size_t cols) const {
    if (!singlet_pair_pinning_.has_value()) {
      return;
    }
    const auto in_lattice = [rows, cols](const SiteIdx site) {
      return site.row() < rows && site.col() < cols;
    };
    if (!in_lattice(singlet_pair_pinning_->site1) || !in_lattice(singlet_pair_pinning_->site2)) {
      throw std::runtime_error("Singlet pair pinning bond is outside the evaluated lattice.");
    }
  }

  bool IsPinnedSingletPairBond_(const SiteIdx site1, const SiteIdx site2, const BondOrientation orient) const {
    return singlet_pair_pinning_.has_value()
        && singlet_pair_pinning_->orient == orient
        && singlet_pair_pinning_->site1 == site1
        && singlet_pair_pinning_->site2 == site2;
  }

  template<typename TenElemT, typename QNT>
  TenElemT EvaluateSingletPairPinningEnergy_(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi
  ) const {
    ValidateSingletPairPinningBondInLattice_(tn.rows(), tn.cols());
    if (!IsPinnedSingletPairBond_(site1, site2, orient)) {
      return TenElemT(0);
    }
    if constexpr (!Index<QNT>::IsFermionic()) {
      throw std::runtime_error("Singlet pair pinning field requires a fermionic tensor network.");
    } else {
      auto pair_source = detail::EvaluateBondSingletPairFortJModelWithPsi(site1, site2,
                                                                          tJSingleSiteState(config1),
                                                                          tJSingleSiteState(config2),
                                                                          orient, tn, contractor,
                                                                          split_index_tps_on_site1,
                                                                          split_index_tps_on_site2,
                                                                          psi);
      return TenElemT(singlet_pair_pinning_->delta) * (pair_source.first + pair_source.second);
    }
  }

  double t_;
  double t2_;
  double J_;
  double V_;
  double mu_;
  std::optional<SingletPairPinningField> singlet_pair_pinning_;
};

/**
 *  The variable psi is ONLY used for return. Any input will be covered by nullopt or a new value.
 *  The returned psi will be used to check the accuracy of psi get by boundary MPS contraction of tensor network.
 */
template<typename TenElemT, typename QNT>
TenElemT SquaretJModelMixIn::EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi // return value, used for check the accuracy
) {
  TenElemT energy;
  if (config1 == config2) {
    psi.reset();
    if (tJSingleSiteState(config1) == tJSingleSiteState::Empty) {
      energy = 0.0;
    } else { // both spin up or spin down,
      energy = V_; // sz * sz - 1/4 * n * n = 0 , n*n = 1
    }
  } else {
    psi = contractor.Trace(tn, site1, site2, orient);
    if (psi == TenElemT(0)) [[unlikely]] {
      std::cerr << "Error: psi is 0. Division by 0 is not allowed." << std::endl;
      exit(EXIT_FAILURE);
    }

    TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                            split_index_tps_on_site1[size_t(config2)],
                                            split_index_tps_on_site2[size_t(config1)]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
    if (is_nan(ratio)) [[unlikely]] {
      std::cerr << "ratio is nan !" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (tJSingleSiteState(config1) == tJSingleSiteState::Empty
        || tJSingleSiteState(config2) == tJSingleSiteState::Empty) {
      // one site empty, the other site filled
      // only hopping energy contribution
      energy = (-t_) * ratio;
    } else {
      // spin antiparallel
      // only spin interaction energy contribution
      energy = (-0.5 + ratio * 0.5) * J_ + V_; // J * (Sz * Sz - n * n /4) + J/2 * (S^+ * S^- * S^- * S^+) + V * n * n
    }
  }
  return energy + EvaluateSingletPairPinningEnergy_(site1, site2,
                                                    config1, config2,
                                                    orient, tn, contractor,
                                                    split_index_tps_on_site1,
                                                    split_index_tps_on_site2,
                                                    psi);
}

template<typename TenElemT, typename QNT>
TenElemT SquaretJModelMixIn::EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi, // return value, used for check the accuracy
    const JastrowDress &jastrow_dress
) {
  if (IsPinnedSingletPairBond_(site1, site2, orient)) {
    throw std::runtime_error(
        "Singlet pair pinning field is not implemented for Jastrow-dressed t-J wave functions.");
  }
  if (config1 == config2) {
    psi.reset();
    if (tJSingleSiteState(config1) == tJSingleSiteState::Empty) {
      return 0.0;
    } else { // both spin up or spin down,
      return V_; // sz * sz - 1/4 * n * n = 0 , n*n = 1
    }
  } else {
    psi = contractor.Trace(tn, site1, site2, orient);
    if (psi == TenElemT(0)) [[unlikely]] {
      std::cerr << "Error: psi is 0. Division by 0 is not allowed." << std::endl;
      exit(EXIT_FAILURE);
    }

    TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                            split_index_tps_on_site1[size_t(config2)],
                                            split_index_tps_on_site2[size_t(config1)]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
    if (is_nan(ratio)) [[unlikely]] {
      std::cerr << "ratio is nan !" << std::endl;
      exit(EXIT_FAILURE);
    }
    if (tJSingleSiteState(config1) == tJSingleSiteState::Empty
        || tJSingleSiteState(config2) == tJSingleSiteState::Empty) {
      // one site empty, the other site filled
      // only hopping energy contribution
      auto &jastrow = jastrow_dress.jastrow;
      auto &density_config = jastrow_dress.density_config;

      double field_site1 = jastrow.JastrowFieldAtSite(density_config, site1);
      double field_site2 = jastrow.JastrowFieldAtSite(density_config, site2);
      double jastrow_ratio;
      if (tJSingleSiteState(config1) == tJSingleSiteState::Empty) {
        jastrow_ratio = std::exp(field_site1 - field_site2);
      } else {
        jastrow_ratio = std::exp(field_site2 - field_site1);
      }
      return (-t_) * ratio * jastrow_ratio;
    } else {
      // spin antiparallel
      // only spin interaction energy contribution
      return (-0.5 + ratio * 0.5) * J_ + V_; // J * (Sz * Sz - n * n /4) + J/2 * (S^+ * S^- * S^- * S^+) + V * n * n
    }
  }
}

/**
 * t2 hopping energy
 *
 * @param psi  if containing value, we assume it is the correct value; if not, calculate it and store into it.
 * This logic is based on the operations outside the function:
 * psi will be reset as unavailable once boundary tensor was moved; And for the calculation in
 * the same plaquette, the psi can be reused for different diagonal bond direction.
 */
template<typename TenElemT, typename QNT>
TenElemT SquaretJModelMixIn::EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi // return value, used for check the accuracy
) {
  if (t2_ == 0) {
    return 0;
  }
  if (config1 == config2 ||
      (tJSingleSiteState(config1) != tJSingleSiteState::Empty
          && tJSingleSiteState(config2) != tJSingleSiteState::Empty)
      ) {
    return 0;
  }
  SiteIdx left_up_site;
  if (diagonal_dir == LEFTUP_TO_RIGHTDOWN) {
    left_up_site = site1;
  } else {
    left_up_site = {site2.row(), site1.col()};
  }
  if (!psi.has_value()) {
    psi = contractor.ReplaceNNNSiteTrace(tn, left_up_site,
                                 diagonal_dir,
                                 HORIZONTAL,
                                 split_index_tps_on_site1[size_t(config1)],
                                 split_index_tps_on_site2[size_t(config2)]);
  }
  TenElemT psi_ex = contractor.ReplaceNNNSiteTrace(tn, left_up_site,
                                           diagonal_dir,
                                           HORIZONTAL,
                                           split_index_tps_on_site1[size_t(config2)],
                                           split_index_tps_on_site2[size_t(config1)]);
  TenElemT ratio = ComplexConjugate(psi_ex / psi.value());
  return -t2_ * ratio;
}

template<typename TenElemT, typename QNT>
TenElemT SquaretJModelMixIn::EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi, // return value, used for check the accuracy
    const JastrowDress &jastrow_dress
) {
  if (t2_ == 0) {
    return 0;
  }
  if (config1 == config2 ||
      (tJSingleSiteState(config1) != tJSingleSiteState::Empty
          && tJSingleSiteState(config2) != tJSingleSiteState::Empty)
      ) {
    return 0;
  }
  SiteIdx left_up_site;
  if (diagonal_dir == LEFTUP_TO_RIGHTDOWN) {
    left_up_site = site1;
  } else {
    left_up_site = {site2.row(), site1.col()};
  }
  if (!psi.has_value()) {
    psi = contractor.ReplaceNNNSiteTrace(tn, left_up_site,
                                 diagonal_dir,
                                 HORIZONTAL,
                                 split_index_tps_on_site1[size_t(config1)],
                                 split_index_tps_on_site2[size_t(config2)]);
  }
  TenElemT psi_ex = contractor.ReplaceNNNSiteTrace(tn, left_up_site,
                                           diagonal_dir,
                                           HORIZONTAL,
                                           split_index_tps_on_site1[size_t(config2)],
                                           split_index_tps_on_site2[size_t(config1)]);
  TenElemT ratio = ComplexConjugate(psi_ex / psi.value());

  // Jastrow factor correction for NNN hopping
  auto &jastrow = jastrow_dress.jastrow;
  auto &density_config = jastrow_dress.density_config;
  double field_site1 = jastrow.JastrowFieldAtSite(density_config, site1);
  double field_site2 = jastrow.JastrowFieldAtSite(density_config, site2);
  double jastrow_ratio;
  if (tJSingleSiteState(config1) == tJSingleSiteState::Empty) {
    jastrow_ratio = std::exp(field_site1 - field_site2);
  } else {
    jastrow_ratio = std::exp(field_site2 - field_site1);
  }

  return -t2_ * ratio * jastrow_ratio;
}

/**
 * Evaluate the singlet pair operators (\Delta^\dagger, \Delta) for the t-J model.
 * \Delta^\dagger = \langle c_{i,\uparrow}^\dagger c_{j,\downarrow}^dagger - c_{i,\downarrow}^\dagger c_{j,\uparrow}^\dagger \rangle / \sqrt{2}
 * \Delta = \langle -c_{i,\uparrow} c_{j,\downarrow} + c_{i,\downarrow} c_{j,\uparrow} \rangle / \sqrt{2}
 */
template<typename TenElemT, typename QNT>
std::pair<TenElemT, TenElemT> EvaluateBondSingletPairFortJModel(const SiteIdx site1,
                                                                const SiteIdx site2,
                                                                const tJSingleSiteState config1,
                                                                const tJSingleSiteState config2,
                                                                const BondOrientation orient,
                                                                const TensorNetwork2D<TenElemT, QNT> &tn,
                                                                BMPSContractor<TenElemT, QNT> &contractor,
                                                                const std::vector<QLTensor<TenElemT,
                                                                                           QNT>> &split_index_tps_on_site1,
                                                                const std::vector<QLTensor<TenElemT,
                                                                                           QNT>> &split_index_tps_on_site2
) {
  std::optional<TenElemT> psi;
  return detail::EvaluateBondSingletPairFortJModelWithPsi(site1, site2, config1, config2, orient, tn, contractor,
                                                          split_index_tps_on_site1, split_index_tps_on_site2, psi);
}

namespace detail {
template<typename TenElemT, typename QNT>
std::pair<TenElemT, TenElemT> EvaluateBondSingletPairFortJModelWithPsi(
    const SiteIdx site1,
    const SiteIdx site2,
    const tJSingleSiteState config1,
    const tJSingleSiteState config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    BMPSContractor<TenElemT, QNT> &contractor,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi
) {
  TenElemT delta_dag, delta;
  auto trace = [&]() -> TenElemT {
    if (!psi.has_value()) {
      psi = contractor.Trace(tn, site1, site2, orient);
    }
    if (psi == TenElemT(0)) [[unlikely]] {
      throw std::runtime_error("EvaluateBondSingletPairFortJModel: psi is zero.");
    }
    return psi.value();
  };

  if (config1 == tJSingleSiteState::Empty && config2 == tJSingleSiteState::Empty) {
    TenElemT psi_val = trace();
    TenElemT psi_ex1 = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                             split_index_tps_on_site1[size_t(tJSingleSiteState::SpinUp)],
                                             split_index_tps_on_site2[size_t(tJSingleSiteState::SpinDown)]);
    TenElemT psi_ex2 = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                             split_index_tps_on_site1[size_t(tJSingleSiteState::SpinDown)],
                                             split_index_tps_on_site2[size_t(tJSingleSiteState::SpinUp)]);
    TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi_val);
    TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi_val);
    delta_dag = (ratio1 - ratio2) / std::sqrt(2);
    delta = 0;
    return std::make_pair(delta_dag, delta);
  } else if (config1 == tJSingleSiteState::SpinUp && config2 == tJSingleSiteState::SpinDown) {
    delta_dag = 0;
    TenElemT psi_val = trace();
    TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                            split_index_tps_on_site1[size_t(tJSingleSiteState::Empty)],
                                            split_index_tps_on_site2[size_t(tJSingleSiteState::Empty)]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi_val);
    delta = ratio / std::sqrt(2);
    return std::make_pair(delta_dag, delta);
  } else if (config1 == tJSingleSiteState::SpinDown && config2 == tJSingleSiteState::SpinUp) {
    delta_dag = 0;
    TenElemT psi_val = trace();
    TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                            split_index_tps_on_site1[size_t(tJSingleSiteState::Empty)],
                                            split_index_tps_on_site2[size_t(tJSingleSiteState::Empty)]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi_val);
    delta = -ratio / std::sqrt(2);
    return std::make_pair(delta_dag, delta);
  } else {
    return std::make_pair(TenElemT(0), TenElemT(0));
  }
}
}  // namespace detail

/*
 * Hamiltonian :
 * H = -t\sum_{<i,j>,sigma} (c_{i,\sigma}^dag c_{j,\sigma} + h.c.)
 *     +J \sum_{<i,j>} (S_i \cdot S_j - 1/4 n_i \cdot n_j )
 *     - \mu N
 *
*/
class SquaretJNNModel : public SquareNNModelEnergySolver<SquaretJNNModel>,
                        public SquareNNModelMeasurementSolver<SquaretJNNModel>,
                        public SquaretJModelMixIn,
                        public SingletPairCorrelationMixin<SquaretJNNModel> {
 public:
  using SquareNNModelMeasurementSolver<SquaretJNNModel>::DescribeObservables;
  explicit SquaretJNNModel(double t, double J, double mu) : SquaretJModelMixIn(t,
                                                                               0,
                                                                               J,
                                                                               0,
                                                                               mu) {}

  /**
   * @brief Evaluate all observables for the current sample.
   * 
   * This includes:
   * - Energy and standard t-J measurements (via base class)
   * - Singlet pair correlation ⟨Δ†Δ⟩ (if enabled)
   */
  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample) {
    // Use the generic registry traversal from base class
    ObservableMap<TenElemT> out =
        this->SquareNNModelMeasurementSolver<SquaretJNNModel>::EvaluateObservables(split_index_tps, tps_sample);

    // Measure singlet pair correlation if enabled
    if (this->IsSingletPairCorrelationEnabled()) {
      const auto& trunc_para = tps_sample->trun_para;
      // Grow DOWN BMPS stack to cover all target rows
      tps_sample->contractor.SetTruncateParams(trunc_para);
      tps_sample->contractor.GrowFullBMPS(tps_sample->tn, DOWN);
      this->MeasureSingletPairCorrelation(
          tps_sample->tn,
          split_index_tps,
          tps_sample->contractor,
          tps_sample->config,
          out);
    }

    return out;
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    auto base = this->SquareNNModelMeasurementSolver<SquaretJNNModel>::DescribeObservables(ly, lx);
    for (auto &meta : base) {
      if (meta.key == "spin_z" || meta.key == "charge") {
        meta.shape = {ly, lx};
        meta.index_labels = {"y", "x"};
      }
      if (meta.key == "bond_energy_h") {
        meta.shape = {ly, (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_v") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), lx};
        meta.index_labels = {"bond_y", "bond_x"};
      }
    }
    base.push_back({"SC_bond_singlet_h", "Bond singlet SC (horizontal) avg(conj(delta_dag), delta)", {ly, (lx > 0 ? lx - 1 : 0)}, {"bond_y", "bond_x"}, {}});
    base.push_back({"SC_bond_singlet_v", "Bond singlet SC (vertical) avg(conj(delta_dag), delta)", {(ly > 0 ? ly - 1 : 0), lx}, {"bond_y", "bond_x"}, {}});
    // Singlet pair correlation: values only, coordinate mapping generated automatically
    const auto& ref_bonds = this->GetSelectedRefBonds();
    size_t num_pairs = SingletPairCorrelationMixin<SquaretJNNModel>::ComputeNumSCPairs(ly, lx, ref_bonds);
    base.push_back({"SC_singlet_pair_corr",
        "Singlet pair correlation <Delta^dag Delta> values.",
        {num_pairs}, {"pair_idx"},
        [ref_bonds](size_t ly_, size_t lx_) {
          return SingletPairCorrelationMixin<SquaretJNNModel>::GenerateSCPairCorrCoordString(ly_, lx_, ref_bonds);
        }});
    return base;
  }
};

/*
* Hamiltonian :
* H = -t\sum_{<i,j>,sigma} (c_{i,\sigma}^dag c_{j,\sigma} + h.c.)
*     -t2\sum_{<<i,j>>,sigma} (c_{i,\sigma}^dag c_{j,\sigma} + h.c.)
*     +J \sum_{<i,j>} (S_i \cdot S_j - 1/4 n_i \cdot n_j )
*     - \mu N
*/
class SquaretJNNNModel : public SquareNNNModelEnergySolver<SquaretJNNNModel>,
                         public SquareNNNModelMeasurementSolver<SquaretJNNNModel>,
                         public SquaretJModelMixIn,
                         public SingletPairCorrelationMixin<SquaretJNNNModel> {
 public:
  using SquareNNNModelMeasurementSolver<SquaretJNNNModel>::DescribeObservables;
  explicit SquaretJNNNModel(double t, double t2, double J, double mu) : SquaretJModelMixIn(t,
                                                                                           t2,
                                                                                           J,
                                                                                           0,
                                                                                           mu) {}

  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample) {
    ObservableMap<TenElemT> out =
        this->SquareNNNModelMeasurementSolver<SquaretJNNNModel>::EvaluateObservables(split_index_tps, tps_sample);

    if (this->IsSingletPairCorrelationEnabled()) {
      const auto& trunc_para = tps_sample->trun_para;
      // Grow DOWN BMPS stack to cover all target rows
      tps_sample->contractor.SetTruncateParams(trunc_para);
      tps_sample->contractor.GrowFullBMPS(tps_sample->tn, DOWN);
      this->MeasureSingletPairCorrelation(
          tps_sample->tn,
          split_index_tps,
          tps_sample->contractor,
          tps_sample->config,
          out);
    }

    return out;
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    auto base = this->SquareNNNModelMeasurementSolver<SquaretJNNNModel>::DescribeObservables(ly, lx);
    for (auto &meta : base) {
      if (meta.key == "spin_z" || meta.key == "charge") {
        meta.shape = {ly, lx};
        meta.index_labels = {"y", "x"};
      }
      if (meta.key == "bond_energy_h") {
        meta.shape = {ly, (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_v") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), lx};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_dr") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_ur") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
    }
    base.push_back({"SC_bond_singlet_h", "Bond singlet SC (horizontal) avg(conj(delta_dag), delta)", {ly, (lx > 0 ? lx - 1 : 0)}, {"bond_y", "bond_x"}, {}});
    base.push_back({"SC_bond_singlet_v", "Bond singlet SC (vertical) avg(conj(delta_dag), delta)", {(ly > 0 ? ly - 1 : 0), lx}, {"bond_y", "bond_x"}, {}});
    const auto& ref_bonds = this->GetSelectedRefBonds();
    size_t num_pairs = SingletPairCorrelationMixin<SquaretJNNNModel>::ComputeNumSCPairs(ly, lx, ref_bonds);
    base.push_back({"SC_singlet_pair_corr",
        "Singlet pair correlation <Delta^dag Delta> values.",
        {num_pairs}, {"pair_idx"},
        [ref_bonds](size_t ly_, size_t lx_) {
          return SingletPairCorrelationMixin<SquaretJNNNModel>::GenerateSCPairCorrCoordString(ly_, lx_, ref_bonds);
        }});
    return base;
  }
};

/**
 * Hamiltonian :
 * H = -t\sum_{<i,j>,sigma} (c_{i,\sigma}^dag c_{j,\sigma} + h.c.)
 *     -t2\sum_{<<i,j>>,sigma} (c_{i,\sigma}^dag c_{j,\sigma} + h.c.)
 *     + J \sum_{<i,j>} (S_i \cdot S_j - 1/4 n_i \cdot n_j )
 *     + V \sum_{<i,j>} n_i n_j
 *     - \mu N
 *
 *  if V == J/4, V term can cancel density interaction in J term
 *
 *  Is it a good convention to include - J/4 n_i \cdot n_j  in the tJV model?
 */
class SquaretJVModel : public SquareNNNModelEnergySolver<SquaretJVModel>,
                       public SquareNNNModelMeasurementSolver<SquaretJVModel>,
                       public SquaretJModelMixIn,
                       public SingletPairCorrelationMixin<SquaretJVModel> {
 public:
  using SquareNNNModelMeasurementSolver<SquaretJVModel>::DescribeObservables;
  explicit SquaretJVModel(double t, double t2, double J, double V, double mu) : SquaretJModelMixIn(t,
                                                                                                   t2,
                                                                                                   J,
                                                                                                   V,
                                                                                                   mu) {}

  template<typename TenElemT, typename QNT>
  ObservableMap<TenElemT> EvaluateObservables(
      const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample) {
    ObservableMap<TenElemT> out =
        this->SquareNNNModelMeasurementSolver<SquaretJVModel>::EvaluateObservables(split_index_tps, tps_sample);

    if (this->IsSingletPairCorrelationEnabled()) {
      const auto& trunc_para = tps_sample->trun_para;
      // Grow DOWN BMPS stack to cover all target rows
      tps_sample->contractor.SetTruncateParams(trunc_para);
      tps_sample->contractor.GrowFullBMPS(tps_sample->tn, DOWN);
      this->MeasureSingletPairCorrelation(
          tps_sample->tn,
          split_index_tps,
          tps_sample->contractor,
          tps_sample->config,
          out);
    }

    return out;
  }

  std::vector<ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    auto base = this->SquareNNNModelMeasurementSolver<SquaretJVModel>::DescribeObservables(ly, lx);
    for (auto &meta : base) {
      if (meta.key == "spin_z" || meta.key == "charge") {
        meta.shape = {ly, lx};
        meta.index_labels = {"y", "x"};
      }
      if (meta.key == "bond_energy_h") {
        meta.shape = {ly, (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_v") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), lx};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_dr") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
      if (meta.key == "bond_energy_ur") {
        meta.shape = {(ly > 0 ? ly - 1 : 0), (lx > 0 ? lx - 1 : 0)};
        meta.index_labels = {"bond_y", "bond_x"};
      }
    }
    base.push_back({"SC_bond_singlet_h", "Bond singlet SC (horizontal) avg(conj(delta_dag), delta)", {ly, (lx > 0 ? lx - 1 : 0)}, {"bond_y", "bond_x"}, {}});
    base.push_back({"SC_bond_singlet_v", "Bond singlet SC (vertical) avg(conj(delta_dag), delta)", {(ly > 0 ? ly - 1 : 0), lx}, {"bond_y", "bond_x"}, {}});
    const auto& ref_bonds = this->GetSelectedRefBonds();
    size_t num_pairs = SingletPairCorrelationMixin<SquaretJVModel>::ComputeNumSCPairs(ly, lx, ref_bonds);
    base.push_back({"SC_singlet_pair_corr",
        "Singlet pair correlation <Delta^dag Delta> values.",
        {num_pairs}, {"pair_idx"},
        [ref_bonds](size_t ly_, size_t lx_) {
          return SingletPairCorrelationMixin<SquaretJVModel>::GenerateSCPairCorrCoordString(ly_, lx_, ref_bonds);
        }});
    return base;
  }
};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_TJ_MODEL_H
