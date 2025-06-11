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

  /**
   * Traverse all NN bonds in the 2D tensor network, and evaluate the bond energy and some order parameters.
   * 
   * @tparam BondMeasureFunc : measure energy or some order parameter
   * @tparam OffDiagLongRangeMeasureFunc
   * 
   * @param tn The tensor network to traverse
   * @param trunc_para The truncation parameters for BMPS
   * @param bond_measure_func functor to measure bond energy or order parameters
   * @param off_diag_long_range_measure_func functor to measure off-diagonal long-range order parameters along horizontal direction. Set to nullptr if not needed.
   */
  template<typename TenElemT, typename QNT, typename BondMeasureFunc, typename NNNLinkMeasureFunc, typename OffDiagLongRangeMeasureFunc>
  static void TraverseAllBonds(
      TensorNetwork2D<TenElemT, QNT> &tn,
      const BMPSTruncatePara &trunc_para,
      BondMeasureFunc &&bond_measure_func,
      NNNLinkMeasureFunc &&nnn_link_measure_func,
      OffDiagLongRangeMeasureFunc &&off_diag_long_range_measure_func,
      std::vector<TenElemT> &psi_list // gather the wave function amplitudes
  ) {
    psi_list.reserve(tn.rows() + tn.cols());
    TraverseHorizontalBonds(tn, trunc_para, std::forward<BondMeasureFunc>(bond_measure_func),
                            std::forward<NNNLinkMeasureFunc>(nnn_link_measure_func),
                            std::forward<OffDiagLongRangeMeasureFunc>(off_diag_long_range_measure_func), psi_list);
    TraverseVerticalBonds(tn, trunc_para, std::forward<BondMeasureFunc>(bond_measure_func), psi_list);
  }

  /**
   * Traverse all horizontal NN bonds and NNN links by horizontal MPS in the 2D tensor network, and evaluate the bond energy and some order parameters.
   * 
   * @tparam BondMeasureFunc : measure energy or some order parameter
   * @tparam OffDiagLongRangeMeasureFunc
   */
  template<typename TenElemT, typename QNT, typename BondMeasureFunc, typename NNNLinkMeasureFunc, typename OffDiagLongRangeMeasureFunc>
  static void TraverseHorizontalBonds(
      TensorNetwork2D<TenElemT, QNT> &tn,
      const BMPSTruncatePara &trunc_para,
      BondMeasureFunc &&bond_measure_func,
      NNNLinkMeasureFunc &&nnn_link_measure_func,
      OffDiagLongRangeMeasureFunc &&off_diag_long_range_measure_func,
      std::vector<TenElemT> &psi_list // gather the wave function amplitudes
  ) {
    tn.GenerateBMPSApproach(UP, trunc_para);
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
      if constexpr (!std::is_same_v<NNNLinkMeasureFunc, std::nullptr_t>) {
        if (row < tn.rows() - 1) {
          tn.InitBTen2(LEFT, row);
          tn.GrowFullBTen2(RIGHT, row, 2, true);
          for (size_t col = 0; col < tn.cols() - 1; col++) {
            std::optional<TenElemT> fermion_psi; // only used for fermion model
            SiteIdx site1 = {row, col};
            SiteIdx site2 = {row + 1, col + 1};
            nnn_link_measure_func(site1, site2, LEFTUP_TO_RIGHTDOWN, inv_psi, fermion_psi);

            site1 = {row + 1, col}; //left-down
            site2 = {row, col + 1}; //right-up
            nnn_link_measure_func(site1, site2, LEFTDOWN_TO_RIGHTUP, inv_psi, fermion_psi);
            tn.BTen2MoveStep(RIGHT, row);
          }
        }
      }
      if constexpr (!std::is_same_v<OffDiagLongRangeMeasureFunc, std::nullptr_t>) {
        off_diag_long_range_measure_func(row, inv_psi);
      }
      if (row < tn.rows() - 1) {
        tn.BMPSMoveStep(DOWN, trunc_para);
      }
    }
  }

  /**
   * Traverse all vertical NN bonds in the 2D tensor network, and evaluate the bond energy and some order parameters.
   * 
   * @tparam BondMeasureFunc : measure energy or some order parameter
   */
  template<typename TenElemT, typename QNT, typename BondMeasureFunc>
  static void TraverseVerticalBonds(
      TensorNetwork2D<TenElemT, QNT> &tn,
      const BMPSTruncatePara &trunc_para,
      BondMeasureFunc &&bond_measure_func,
      std::vector<TenElemT> &psi_list // gather the wave function amplitudes
  ) {
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
