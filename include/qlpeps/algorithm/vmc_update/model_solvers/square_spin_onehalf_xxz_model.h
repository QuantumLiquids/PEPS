/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-09-12
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver for spin-1/2 AFM Heisenberg model in square lattice
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_HEISENBERG_SQUARE_H

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"      // ModelEnergySolver
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h" // ModelMeasurementSolver
#include "qlpeps/utility/helpers.h"                               // ComplexConjugate
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_energy_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_model_measurement_solver.h"

namespace qlpeps {
using namespace qlten;

///< assume the boundary MPS has formed before run the function
template<typename TenElemT, typename QNT>
void MeasureSpinOneHalfOffDiagOrderInRow(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                                         TensorNetwork2D<TenElemT, QNT> &tn,
                                         std::vector<TenElemT> &two_point_functions_loc,
                                         double inv_psi,
                                         const Configuration &config,
                                         size_t &row) {
  const size_t lx = tn.cols();
  SiteIdx site1 = {row, lx / 4};
  std::vector<TenElemT> off_diag_corr(lx / 2);// sp(i) * sm(j) or sm(i) * sp(j), the valid channel
  tn.UpdateSiteTensor(site1, 1 - config(site1), *split_index_tps, true);
  //temporally change, and also trucated the left boundary tensor
  tn.GrowBTenStep(LEFT); // left boundary tensor just across Lx/4
  tn.GrowFullBTen(RIGHT, row, lx / 4 + 2, false); //environment for Lx/4 + 1 site
  for (size_t i = 1; i <= lx / 2; i++) {
    SiteIdx site2 = {row, lx / 4 + i};
    //sm(i) * sp(j) + sp(j) * sm(i)
    if (config(site2) == config(site1)) {
      off_diag_corr[i - 1] = 0.0;
    } else {
      TenElemT psi_ex = tn.ReplaceOneSiteTrace(site2, (*split_index_tps)(site2)[1 - config(site2)], HORIZONTAL);
      off_diag_corr[i - 1] = (ComplexConjugate(psi_ex * inv_psi));
    }
    tn.BTenMoveStep(RIGHT);
  }
  tn.UpdateSiteTensor(site1, config(site1), *split_index_tps, true);
  // change back

  two_point_functions_loc.insert(two_point_functions_loc.end(),
                                 off_diag_corr.begin(),
                                 off_diag_corr.end());

}

///< h_nn = jz * Sz * Sz + jxy * (Sx * Sx + Sy * Sy),
///< h_nnn = jz2 * Sz * Sz + jxy2 * (Sx * Sx + Sy * Sy)
class SquareSpinOneHalfXXZModelMixIn {
 public:
  static constexpr bool requires_density_measurement = false;
  static constexpr bool requires_spin_sz_measurement = true;

  SquareSpinOneHalfXXZModelMixIn(double jz, double jxy, double jz2, double jxy2, double pinning_field_00)
      : jz_(jz), jxy_(jxy), jz2_(jz2), jxy2_(jxy2), pinning00_(pinning_field_00) {};

  template<typename TenElemT, typename QNT>
  [[nodiscard]] TenElemT EvaluateBondEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const BondOrientation orient,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      const TenElemT inv_psi
  ) {
    if (config1 == config2) {
      return 0.25 * jz_;
    } else {
      TenElemT psi_ex = tn.ReplaceNNSiteTrace(site1, site2, orient,
                                              split_index_tps_on_site1[config2],
                                              split_index_tps_on_site2[config1]);
      TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
      return (-0.25 * jz_ + ratio * 0.5 * jxy_);
    }
  }

  template<typename TenElemT, typename QNT>
  [[nodiscard]] TenElemT EvaluateNNNEnergy(
      const SiteIdx site1, const SiteIdx site2,
      const size_t config1, const size_t config2,
      const DIAGONAL_DIR diagonal_dir,
      const TensorNetwork2D<TenElemT, QNT> &tn,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      const TenElemT inv_psi
  ) {
    if (config1 == config2) {
      return 0.25 * jz2_;
    } else {
      SiteIdx left_up_site;
      if (diagonal_dir == LEFTUP_TO_RIGHTDOWN) {
        left_up_site = site1;
      } else {
        left_up_site = {site2.row(), site1.col()};
      }
      TenElemT psi_ex = tn.ReplaceNNNSiteTrace(left_up_site,
                                               diagonal_dir,
                                               HORIZONTAL,
                                               split_index_tps_on_site1[config2],
                                               split_index_tps_on_site2[config1]);
      TenElemT ratio = ComplexConjugate(psi_ex * inv_psi);
      return (-0.25 * jz2_ + ratio * 0.5 * jxy2_);
    }
  }

  [[nodiscard]] inline double CalSpinSzImpl(const size_t config) const { return double(config) - 0.5; }

  [[nodiscard]] inline double EvaluateTotalOnsiteEnergy(const Configuration &config) const {
    return -pinning00_ * CalSpinSzImpl(config({0, 0}));
  }

  template<typename TenElemT, typename QNT>
  inline void EvaluateOffDiagOrderInRow(const SplitIndexTPS<TenElemT, QNT> *split_index_tps,
                                        TensorNetwork2D<TenElemT, QNT> &tn,
                                        std::vector<TenElemT> &two_point_function_loc,
                                        double inv_psi,
                                        const Configuration &config,
                                        size_t &row) const {
    MeasureSpinOneHalfOffDiagOrderInRow(split_index_tps, tn, two_point_function_loc, inv_psi, config, row);
  }

 private:
  const double jz_; // NN zz coupling,
  const double jxy_; // NN xx and yy coupling
  const double jz2_;  // NNN zz coupling
  const double jxy2_; // NNN xy coupling
  const double pinning00_;
};

/**
 *  $$H = sum_<i,j> (J_z * S^z_i \cdot S^z_j + J_{xy} * ( S^x_i \cdot S^x_j  +  S^y_i \cdot S^y_j ))- h_{00} * S^z_{00}$$
 * $S^{\alpha}_i$ are spin-1/2 operator, h_{00} is the pinning field in corner.
 */
class SquareSpinOneHalfXXZModel
    : public SquareNNModelEnergySolver<SquareSpinOneHalfXXZModel>,
      public SquareNNModelMeasurementSolver<SquareSpinOneHalfXXZModel>,
      public SquareSpinOneHalfXXZModelMixIn {
 public:
  ///< Isotropic Heisenberg model with J = 1 and no pinning field
  SquareSpinOneHalfXXZModel(void) : SquareSpinOneHalfXXZModelMixIn(1.0, 1.0, 0.0, 0.0, 0.0) {}

  SquareSpinOneHalfXXZModel(double jz, double jxy, double pinning00) :
      SquareSpinOneHalfXXZModelMixIn(jz, jxy, 0.0, 0.0, pinning00) {}
};

}//qlpeps




#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_SPIN_ONEHALF_HEISENBERG_SQUARE_H
