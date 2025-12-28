/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-08-02
*
* Description: QuantumLiquids/PEPS project. Model Energy Solver base class.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H

#include <complex>
#include <string>
#include <type_traits>

#include "qlten/qlten.h"
#include "qlpeps/vmc_basic/wave_function_component.h"  // TPSWaveFunctionComponent
#include "qlpeps/algorithm/vmc_update/psi_consistency.h"

namespace qlpeps {

/**
 * @brief ModelEnergySolver is a base class used for calculating the energy and gradient hole
 *        during the optimization of the Tensor Product State (TPS).
 *        The class rewrite the operator() so that it works as a functor which calculate
 *        the energy and gradient info upon specific Monte-Carlo samples
 *        It uses CRTP (Curiously Recurring Template Pattern).
 *
 * In the inherited class, the function CalEnergyAndHolesImpl should be defined to evaluate
 * the energy and holes of 2D tensor-network (without additional divide on psi).
 * @tparam ConcreteModelSolver the derived class
 */
template<typename ConcreteModelSolver>
class ModelEnergySolver {
 public:
  ModelEnergySolver(void) = default;

  void SetPsiConsistencyWarningParams(const PsiConsistencyWarningParams &p) {
    // Solver does not print warnings; this only affects how much of psi_list we cache
    // for executor-level warning messages.
    psi_list_max_print_elems_ = p.max_print_elems;
  }

  template<typename TenElemT>
  PsiConsistencySummary<TenElemT> GetLastPsiConsistencySummary(void) const {
    PsiConsistencySummary<TenElemT> out{};
    if (!last_psi_valid_) {
      out.psi_mean = TenElemT(0);
      out.psi_rel_err = 0.0;
      return out;
    }
    TenElemT mean = static_cast<TenElemT>(last_psi_mean_re_);
    if constexpr (!std::is_same_v<TenElemT, double>) {
      mean = TenElemT(last_psi_mean_re_, last_psi_mean_im_);
    }
    out.psi_mean = mean;
    out.psi_rel_err = last_psi_rel_err_;
    return out;
  }

  const std::string &GetLastPsiListTruncated(void) const { return last_psi_list_trunc_; }
  /**
   *
   * @tparam calchols   whether calculate the gradient hole sample data and return in hole_res
   * @param sitps       the TPS wave function
   * @param tps_sample  the wave function component
   * @param hole_res    the gradient hole sample data, valid only when calchols==true
   * @return  evaluated total energy in current Monte Carlo samples
   */
  template<typename TenElemT, typename QNT, bool calchols, typename WaveFunctionComponentT>
  TenElemT CalEnergyAndHoles(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      WaveFunctionComponentT *tps_sample,
      TensorNetwork2D<TenElemT, QNT> &hole_res  // the return value
  ) {
    std::vector<TenElemT> psi_list;

//    TensorNetwork2D<TenElemT, QNT> &sample_tn = tps_sample->tn;
//    const Configuration &sample_config = tps_sample->config;
//    const BMPSTruncatePara &trunc_para = tps_sample->trun_para;
//    TenElemT energy = static_cast<ConcreteModelSolver *>(this)->template CalEnergyAndHolesImpl<calchols>(sitps,
//                                                                                                         sample_config,
//                                                                                                         sample_tn,
//                                                                                                         trunc_para,
//                                                                                                         hole_res,
//                                                                                                         psi_list);
    TenElemT energy = static_cast<ConcreteModelSolver *>(this)
                          ->template CalEnergyAndHolesImpl<TenElemT, QNT, calchols>(sitps, tps_sample, hole_res, psi_list);

    UpdatePsiConsistencyCache_(psi_list);
    return energy;
  }

  template<typename TenElemT, typename QNT, typename WaveFunctionComponentT>
  TenElemT CalEnergy(
      const SplitIndexTPS<TenElemT, QNT> *sitps,
      WaveFunctionComponentT *tps_sample
  ) {
    TensorNetwork2D<TenElemT, QNT> hole_res(1, 1);
    return CalEnergyAndHoles<TenElemT, QNT, false>(sitps, tps_sample, hole_res);
  }

 private:
  template<typename TenElemT>
  void UpdatePsiConsistencyCache_(const std::vector<TenElemT> &psi_list) const {
    const auto s = ComputePsiConsistencySummaryAligned(psi_list);
    if constexpr (std::is_same_v<TenElemT, double>) {
      last_psi_mean_re_ = s.psi_mean;
      last_psi_mean_im_ = 0.0;
    } else {
      last_psi_mean_re_ = static_cast<double>(std::real(s.psi_mean));
      last_psi_mean_im_ = static_cast<double>(std::imag(s.psi_mean));
    }
    last_psi_rel_err_ = s.psi_rel_err;
    last_psi_list_trunc_ = FormatPsiListTruncated(psi_list, psi_list_max_print_elems_);
    last_psi_valid_ = true;
  }

  size_t psi_list_max_print_elems_ = 8;

  // Last-sample cache (type-agnostic)
  mutable bool last_psi_valid_ = false;
  mutable double last_psi_mean_re_ = 0.0;
  mutable double last_psi_mean_im_ = 0.0;
  mutable double last_psi_rel_err_ = 0.0;
  mutable std::string last_psi_list_trunc_;
};

}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_MODEL_ENERGY_SOLVER_H
