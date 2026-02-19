// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-06-21
*
* Description: QuantumLiquids/PEPS project. Loop update executor class.
* Reference: [1] PRB 102, 075147 (2020), "Loop update for iPEPS in 2D".
*
*/


#ifndef QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_H
#define QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_H

#include <functional>
#include <optional>
#include <vector>

#include "qlpeps/algorithm/simple_update/simple_update.h"
#include "qlpeps/two_dim_tn/framework/duomatrix.h"        //DuoMatrix

namespace qlpeps {

using namespace qlten;

template<typename TenT>
using LoopGates = std::array<TenT, 4>;

/// @brief Type of imaginary-time evolution gate used in loop update.
///
/// Determines how the energy is estimated from the overlap after projection:
/// - kFirstOrder: gate ≈ 1 - tau*H. Energy = (1 - overlap) / tau.
/// - kExponential: gate ≈ exp(-tau*H). Energy = -ln(overlap) / tau.
enum class LoopGateType {
  kFirstOrder,   ///< Gate is first-order Taylor expansion: 1 - tau*H
  kExponential   ///< Gate is (approximate) exponential: exp(-tau*H)
};

/// @brief Per-step metrics collected during loop update execution.
template<typename RealT>
struct LoopUpdateStepMetrics {
  size_t step_index;                   ///< Zero-based step index within the current Execute() call
  double tau;                          ///< Trotter step length used for this step
  RealT estimated_e0;                  ///< Sum of local energies (E0 estimate)
  RealT estimated_en;                  ///< Norm-based energy estimate: -log(norm)/tau
  std::optional<RealT> trunc_err;      ///< Representative truncation error for this sweep.
                                       ///< nullopt when the executor does not report truncation error.
  double elapsed_sec;                  ///< Wall-clock time for this sweep in seconds
  bool bond_dim_changed;               ///< Whether bond dimensions changed compared to previous step
};

/// @brief Parameters for LoopUpdateExecutor.
///
/// Bundles truncation parameters, step count, Trotter step length,
/// and gate type into a single configuration struct.
struct LoopUpdatePara {
  struct AdvancedStopConfig {
    double energy_abs_tol;
    double energy_rel_tol;
    double lambda_rel_tol;
    size_t patience;
    size_t min_steps;
  };

  LoopUpdateTruncatePara truncate_para;
  size_t steps;
  double tau;
  LoopGateType gate_type;
  std::optional<AdvancedStopConfig> advanced_stop;

  /// @brief Optional per-step observer callback. Called after each sweep with step metrics.
  /// If not set, no callback is invoked (zero overhead).
  /// @note The callback uses `LoopUpdateStepMetrics<double>` (not `RealT`) because
  /// `LoopUpdatePara` is a non-templated struct and cannot depend on `RealT`.
  /// `double` is the natural external-consumption precision; when `RealT` is `float`,
  /// values are losslessly widened. This is a deliberate design choice.
  std::optional<std::function<void(const LoopUpdateStepMetrics<double>&)>> step_observer;

  /// @brief When true, emit one machine-readable line per step to stdout.
  bool emit_machine_readable_metrics = false;

  LoopUpdatePara(const LoopUpdateTruncatePara &truncate_para,
                 size_t steps,
                 double tau,
                 LoopGateType gate_type = LoopGateType::kFirstOrder)
      : truncate_para(truncate_para), steps(steps), tau(tau), gate_type(gate_type),
        advanced_stop(std::nullopt), step_observer(std::nullopt),
        emit_machine_readable_metrics(false) {}

  LoopUpdatePara(const LoopUpdateTruncatePara &truncate_para,
                 size_t steps,
                 double tau,
                 const AdvancedStopConfig &advanced_stop_config,
                 LoopGateType gate_type = LoopGateType::kFirstOrder)
      : truncate_para(truncate_para), steps(steps), tau(tau), gate_type(gate_type),
        advanced_stop(advanced_stop_config), step_observer(std::nullopt),
        emit_machine_readable_metrics(false) {}

  static LoopUpdatePara Advanced(const LoopUpdateTruncatePara &truncate_para,
                                 size_t steps,
                                 double tau,
                                 double energy_abs_tol,
                                 double energy_rel_tol,
                                 double lambda_rel_tol,
                                 size_t patience,
                                 size_t min_steps,
                                 LoopGateType gate_type = LoopGateType::kFirstOrder) {
    return LoopUpdatePara(
        truncate_para, steps, tau,
        AdvancedStopConfig{energy_abs_tol, energy_rel_tol, lambda_rel_tol, patience, min_steps},
        gate_type);
  }
};

/// @brief Executor for the loop update algorithm on square-lattice PEPS.
///
/// The loop update applies imaginary-time evolution gates arranged in 2x2
/// plaquette loops, following the full-environment truncation scheme of
/// Ref. [1] PRB 102, 075147 (2020).
template<typename TenElemT, typename QNT>
class LoopUpdateExecutor : public Executor {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  using LoopGatesT = LoopGates<Tensor>;
  using StepMetrics = LoopUpdateStepMetrics<RealT>;

  enum class StopReason {
    kNotRun = 0,
    kMaxSteps,
    kAdvancedConverged
  };

  struct RunSummary {
    bool converged = false;
    StopReason stop_reason = StopReason::kNotRun;
    size_t executed_steps = 0;
    std::optional<RealT> final_energy = std::nullopt;        // kept for backward compat (= final_estimated_e0)
    std::optional<RealT> final_estimated_e0 = std::nullopt;  ///< Final E0 estimate
    std::optional<RealT> final_estimated_en = std::nullopt;  ///< Final En estimate
  };

  /// @brief Data returned by each loop sweep, carrying all per-sweep observables.
  struct SweepResult {
    RealT estimated_e0;                ///< Sum of local energies
    RealT estimated_en;                ///< Norm-based energy: -log(norm)/tau
    std::optional<RealT> trunc_err;    ///< Representative truncation error.
                                       ///< nullopt if the executor does not track truncation error.
    double elapsed_sec;                ///< Wall time for the sweep
    size_t dmin;                       ///< Minimum bond dimension after sweep
    size_t dmax;                       ///< Maximum bond dimension after sweep
  };

  LoopUpdateExecutor(const LoopUpdatePara &para,
                     const DuoMatrix<LoopGatesT> &evolve_gates,
                     const PEPST &peps_initial);

  [[deprecated("Use the LoopUpdatePara constructor instead")]]
  LoopUpdateExecutor(const LoopUpdateTruncatePara &truncate_para,
                     const size_t steps,
                     const double tau,
                     const DuoMatrix<LoopGatesT> &evolve_gates,
                     const PEPST &peps_initial);

  void Execute(void) override;

  const PEPST &GetPEPS(void) const {
    return peps_;
  }

  double GetEstimatedEnergy(void) const {
    return estimated_energy_;
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

  /// @brief Get the per-step metrics collected during the last Execute() call.
  const std::vector<StepMetrics> &GetStepMetrics(void) const {
    return step_metrics_;
  }

  bool DumpResult(std::string path, bool release_mem) {
    return peps_.Dump(path, release_mem);
  }

 private:

  SweepResult LoopUpdateSweep_(void);

  /**
   *
   * @param site: the coordinate of the left-upper site in the loop
   * @return
   */
  std::pair<double, double> UpdateOneLoop(const SiteIdx &site,
                                          const LoopUpdateTruncatePara &para,
                                          const bool print_time);

  const size_t lx_;
  const size_t ly_;

  LoopUpdatePara para_;

  /** The set of the evolve gates
  *  where each gate form a loop and includes 4 tensors, and
  *  represents the local imaginary evolve gate exp(-\tau * h).
  *  Sizes of the DuoMatrix:
  *    - OBC: (Ly-1) rows by (Lx-1) columns
  *    - PBC: Ly rows by Lx columns (requires even Lx and Ly)
  *
  *  The order of the 4 tensors in one loop is accord to Ref. [1] Fig. 2 (a)
  *
  *  And the orders of the legs in MPO tensors are
  *
  *        2                  2
  *        |                  |
  *        |                  |
  *  0---[gate 0]---3  0---[gate 1]---3
  *        |                  |
  *        |                  |
  *        1                  1
  *  so on and so forth for gate 2 and 3. Note the leg 3 of gate 0 are connected to
  *  the leg 0 of gate 1, so on so forth.
  *  And the legs 1 of the gates are connected to PEPS physical legs.
  *  (The diagrams here are upside down from the Fig. in the Ref. [1].)
  *
  */
  DuoMatrix<LoopGatesT> evolve_gates_;
  Tensor id_nn_;    ///< Nearest-neighbor identity operator. @todo Re-enable canonicalization sweeps.
  double estimated_energy_ = 0.0;
  RunSummary last_run_summary_;
  std::vector<StepMetrics> step_metrics_;

  PEPST peps_;
};

}//qlpeps

#include "qlpeps/algorithm/loop_update/loop_update_impl.h"
#endif //QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_H
