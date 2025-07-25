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
 * with bool value  has_nn_term_  defining whether include  - 1/4 n_i \cdot n_j term
*/
#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_TJ_MODEL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SQUARE_TJ_MODEL_H

#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nnn_model_measurement_solver.h"
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate

namespace qlpeps {
using namespace qlten;

/*
 *  */
enum class tJSingleSiteState {
  SpinUp,           // 0
  SpinDown,         // 1
  Empty             // 2
};

class SquaretJModelMixIn {
 public:
  static constexpr bool requires_density_measurement = true;
  static constexpr bool requires_spin_sz_measurement = true;
  static const bool enable_sc_measurement;
  SquaretJModelMixIn(void) = delete;

  explicit SquaretJModelMixIn(double t, double t2, double J, bool has_nn_term, double mu)
      : t_(t), t2_(t2), J_(J), has_nn_term_(has_nn_term), mu_(mu) {}

  ///< requirement from SquareNNNModelEnergySolver and SquareNNModelEnergySolver
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

  ///< requirement from SquareNNNModelEnergySolver
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

  template<typename TenElemT, typename QNT>
  std::pair<TenElemT, TenElemT> EvaluateBondSC(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi
  ) {
    return EvaluateBondSingletPairFortJModel(site1, site2,
                                             tJSingleSiteState(config1),
                                             tJSingleSiteState(config2),
                                             orient, tn, split_index_tps_on_site1, split_index_tps_on_site2);
  }

  ///< requirement from SquareNNNModelEnergySolver
  double EvaluateTotalOnsiteEnergy(const Configuration &config) {
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
  double t_;
  double t2_;
  double J_;
  bool has_nn_term_;
  double mu_;
};

template<typename TenElemT, typename QNT>
TenElemT SquaretJModelMixIn::EvaluateBondEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const BondOrientation orient,
    const TensorNetwork2D<TenElemT, QNT> &tn,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
    const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
    std::optional<TenElemT> &psi // return value, used for check the accuracy
) {
  if (config1 == config2) {
    psi.reset();
    if (tJSingleSiteState(config1) == tJSingleSiteState::Empty) {
      return 0.0;
    } else {
      return J_ * (0.25 - double(int(has_nn_term_)) / 4.0); // sz * sz - 1/4 * n * n
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
    if (tJSingleSiteState(config1) == tJSingleSiteState::Empty
        || tJSingleSiteState(config2) == tJSingleSiteState::Empty) {
      // one site empty, the other site filled
      // only hopping energy contribution
      return (-t_) * ratio;
    } else {
      // spin antiparallel
      // only spin interaction energy contribution
      return (-0.25 + ratio * 0.5 - int(has_nn_term_) / 4.0) * J_;
    }
  }
}

/**
 * t2 hopping energy
 *
 * @param psi  if containing value, we assume it is the correct value; if not, calculate it and store into it.
 */
template<typename TenElemT, typename QNT>
TenElemT SquaretJModelMixIn::EvaluateNNNEnergy(
    const SiteIdx site1, const SiteIdx site2,
    const size_t config1, const size_t config2,
    const DIAGONAL_DIR diagonal_dir,
    const TensorNetwork2D<TenElemT, QNT> &tn,
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
                                                                const std::vector<QLTensor<TenElemT,
                                                                                           QNT>> &split_index_tps_on_site1,
                                                                const std::vector<QLTensor<TenElemT,
                                                                                           QNT>> &split_index_tps_on_site2
) {
  TenElemT delta_dag, delta;
  if (config1 == tJSingleSiteState::Empty && config2 == tJSingleSiteState::Empty) {
    TenElemT psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex1 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                             split_index_tps_on_site1[size_t(tJSingleSiteState::SpinUp)],
                                             split_index_tps_on_site2[size_t(tJSingleSiteState::SpinDown)]);
    TenElemT psi_ex2 = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                             split_index_tps_on_site1[size_t(tJSingleSiteState::SpinDown)],
                                             split_index_tps_on_site2[size_t(tJSingleSiteState::SpinUp)]);
    TenElemT ratio1 = ComplexConjugate(psi_ex1 / psi);
    TenElemT ratio2 = ComplexConjugate(psi_ex2 / psi);
    delta_dag = (ratio1 - ratio2) / std::sqrt(2);
    delta = 0;
    return std::make_pair(delta_dag, delta);
  } else if (config1 == tJSingleSiteState::SpinUp && config2 == tJSingleSiteState::SpinDown) {
    delta_dag = 0;
    TenElemT psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[size_t(tJSingleSiteState::Empty)],
                                            split_index_tps_on_site2[size_t(tJSingleSiteState::Empty)]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi);
    delta = ratio / std::sqrt(2);
    return std::make_pair(delta_dag, delta);
  } else if (config1 == tJSingleSiteState::SpinDown && config2 == tJSingleSiteState::SpinUp) {
    delta_dag = 0;
    TenElemT psi = tn.Trace(site1, site2, orient);
    TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                            split_index_tps_on_site1[size_t(tJSingleSiteState::Empty)],
                                            split_index_tps_on_site2[size_t(tJSingleSiteState::Empty)]);
    TenElemT ratio = ComplexConjugate(psi_ex / psi);
    delta = -ratio / std::sqrt(2);
    return std::make_pair(delta_dag, delta);
  } else {
    return std::make_pair(TenElemT(0), TenElemT(0));
  }
}

class SquaretJNNModel : public SquareNNModelEnergySolver<SquaretJNNModel>,
                        public SquareNNModelMeasurementSolver<SquaretJNNModel>,
                        public SquaretJModelMixIn {
 public:
  explicit SquaretJNNModel(double t, double J, bool has_nn_term, double mu) : SquaretJModelMixIn(t,
                                                                                                 0,
                                                                                                 J,
                                                                                                 has_nn_term,
                                                                                                 mu) {};
};

class SquaretJNNNModel : public SquareNNNModelEnergySolver<SquaretJNNNModel>,
                         public SquareNNNModelMeasurementSolver<SquaretJNNNModel>,
                         public SquaretJModelMixIn {
 public:
  explicit SquaretJNNNModel(double t, double t2, double J, bool has_nn_term, double mu) : SquaretJModelMixIn(t,
                                                                                                             t2,
                                                                                                             J,
                                                                                                             has_nn_term,
                                                                                                             mu) {};
};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_PEPS_MODEL_SOLVERS_SQUARE_TJ_MODEL_H
