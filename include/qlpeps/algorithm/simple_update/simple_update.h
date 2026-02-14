// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-20
*
* Description: QuantumLiquids/PEPS project. Abstract class for simple update.
*/

#ifndef QLPEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H
#define QLPEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H

#include <optional>

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"    //SquareLatticePEPS

namespace qlpeps {


struct SimpleUpdatePara {
  struct AdvancedStopConfig {
    double energy_abs_tol;
    double energy_rel_tol;
    double lambda_rel_tol;
    size_t patience;
    size_t min_steps;
  };

  size_t steps;
  double tau;         // Step length

  size_t Dmin;
  size_t Dmax;        // Bond dimension
  double Trunc_err;   // Truncation error
  std::optional<AdvancedStopConfig> advanced_stop;

  SimpleUpdatePara(size_t steps, double tau, size_t Dmin, size_t Dmax, double Trunc_err)
      : steps(steps), tau(tau), Dmin(Dmin), Dmax(Dmax), Trunc_err(Trunc_err), advanced_stop(std::nullopt) {}

  SimpleUpdatePara(size_t steps, double tau, size_t Dmin, size_t Dmax, double Trunc_err,
                   const AdvancedStopConfig &advanced_stop_config)
      : steps(steps), tau(tau), Dmin(Dmin), Dmax(Dmax), Trunc_err(Trunc_err), advanced_stop(advanced_stop_config) {}

  static SimpleUpdatePara Advanced(size_t steps, double tau, size_t Dmin, size_t Dmax, double Trunc_err,
                                   double energy_abs_tol, double energy_rel_tol, double lambda_rel_tol,
                                   size_t patience, size_t min_steps) {
    return SimpleUpdatePara(
        steps, tau, Dmin, Dmax, Trunc_err,
        AdvancedStopConfig{energy_abs_tol, energy_rel_tol, lambda_rel_tol, patience, min_steps});
  }
};

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> TaylorExpMatrix(const typename qlten::RealTypeTrait<TenElemT>::type tau, const QLTensor<TenElemT, QNT> &ham);

/** SimpleUpdateExecutor
 * abstract class for execution on simple update in SquareLatticePEPS
 */
template<typename TenElemT, typename QNT>
class SimpleUpdateExecutor : public Executor {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  enum class StopReason {
    kNotRun = 0,
    kMaxSteps,
    kAdvancedConverged
  };
  struct RunSummary {
    bool converged = false;
    StopReason stop_reason = StopReason::kNotRun;
    size_t executed_steps = 0;
    std::optional<RealT> final_energy = std::nullopt;
  };

  SimpleUpdateExecutor(const SimpleUpdatePara &update_para,
                       const PEPST &peps_initial);

  void ResetStepLenth(double tau) {
    update_para.tau = tau;
    SetEvolveGate_();
  }

  void Execute(void) override;

  const PEPST &GetPEPS(void) const {
    return peps_;
  }

  bool DumpResult(std::string path, bool release_mem) {
    return peps_.Dump(path, release_mem);
  }

  RealT GetEstimatedEnergy(void) const {
    if (estimated_energy_.has_value()) {
      return estimated_energy_.value();
    } else {
      std::cout << "No estimated energy value!" << std::endl;
      return 0.0;
    }
  }

  const RunSummary &GetLastRunSummary(void) const {
    return last_run_summary_;
  }

  bool LastRunConverged(void) const {
    return last_run_summary_.converged;
  }

  size_t LastRunExecutedSteps(void) const {
    return last_run_summary_.executed_steps;
  }

  SimpleUpdatePara update_para;
 protected:

  virtual void SetEvolveGate_(void) = 0;

  // return the estimated energy
  virtual RealT SimpleUpdateSweep_(void) = 0;

  const size_t lx_;
  const size_t ly_;
  PEPST peps_;

  std::optional<RealT> estimated_energy_;
  RunSummary last_run_summary_;
};

}//qlpeps

#include "qlpeps/algorithm/simple_update/simple_update_impl.h"

#endif //QLPEPS_ALGORITHM_SIMPLE_UPDATE_SIMPLE_UPDATE_H
