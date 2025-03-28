/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2025-03-13
 *
 * Description: QuantumLiquids/PEPS project.
 * Bond traversal Mixin Class
*/


#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BOND_TRAVERSAL_MIXIN_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BOND_TRAVERSAL_MIXIN_H

#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"    //TensorNetwork2D

namespace qlpeps {

class BondTraversalMixin {
 public:
  ///< BondMeasureFunc
  /**
   * @tparam BondMeasureFunc : measure energy or some order parameter
   * @tparam OffDiagLongRangeMeasureFunc
   */
  template<typename TenElemT, typename QNT, typename BondMeasureFunc, typename OffDiagLongRangeMeasureFunc>
  static void TraverseAllBonds(
      TensorNetwork2D<TenElemT, QNT> &tn,
      const BMPSTruncatePara &trunc_para,
      BondMeasureFunc &&bond_measure_func,
      OffDiagLongRangeMeasureFunc &&off_diag_long_range_measure_func,
      std::vector<TenElemT> &psi_list // gather the wave function amplitude
  ) {
    tn.GenerateBMPSApproach(UP, trunc_para);
    // Horizontal bond Transverse
    for (size_t row = 0; row < tn.rows(); ++row) {
      tn.InitBTen(LEFT, row);
      tn.GrowFullBTen(RIGHT, row, 1, true);

      auto psi = tn.Trace({row, 0}, HORIZONTAL);
      if (psi == TenElemT(0)) [[unlikely]] {
        throw std::runtime_error("Wavefunction amplitude is near zero, causing division by zero.");
      }
      TenElemT inv_psi = 1.0 / psi; // now only useful for boson case
      psi_list.push_back(psi);

      for (size_t col = 0; col < tn.cols() - 1; ++col) {
        const SiteIdx site1{row, col};
        const SiteIdx site2{row, col + 1};
        bond_measure_func(site1, site2, HORIZONTAL, inv_psi);
        tn.BTenMoveStep(RIGHT);
      }
      // measure the off-diagonal long-range order correlations, like S^+ * S-
      if constexpr (!std::is_same_v<OffDiagLongRangeMeasureFunc, std::nullptr_t>) {
        off_diag_long_range_measure_func(row, inv_psi);
      }
      if (row < tn.rows() - 1) {
        tn.BMPSMoveStep(DOWN, trunc_para);
      }
    }

    // Vertical bond transverse
    tn.GenerateBMPSApproach(LEFT, trunc_para);
    for (size_t col = 0; col < tn.cols(); col++) {
      tn.InitBTen(UP, col);
      tn.GrowFullBTen(DOWN, col, 2, true);

      auto psi = tn.Trace({0, col}, VERTICAL);
      if (psi == TenElemT(0)) [[unlikely]] {
        throw std::runtime_error("Wavefunction amplitude is near zero, causing division by zero.");
      }
      TenElemT inv_psi = 1.0 / psi;// only useful for Bosonic case
      psi_list.push_back(psi);

      for (size_t row = 0; row < tn.rows() - 1; row++) {
        const SiteIdx site1{row, col};
        const SiteIdx site2{row + 1, col};
        bond_measure_func(site1, site2, VERTICAL, inv_psi);
        if (row < tn.rows() - 2) {
          tn.BTenMoveStep(DOWN);
        }
      }
      if (col < tn.cols() - 1) {
        tn.BMPSMoveStep(RIGHT, trunc_para);
      }
    }
  }
};
}

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_SOLVERS_BOND_TRAVERSAL_MIXIN_H
