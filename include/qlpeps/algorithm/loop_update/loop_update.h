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

/// @brief Parameters for LoopUpdateExecutor.
///
/// Bundles truncation parameters, step count, Trotter step length,
/// and gate type into a single configuration struct.
///
/// @todo Add AdvancedStopConfig for automatic convergence detection.
/// @todo Add step_observer callback for machine-readable per-step metrics.
/// @todo Add emit_machine_readable_metrics flag.
struct LoopUpdatePara {
  LoopUpdateTruncatePara truncate_para;
  size_t steps;
  double tau;
  LoopGateType gate_type;

  LoopUpdatePara(const LoopUpdateTruncatePara &truncate_para,
                 size_t steps,
                 double tau,
                 LoopGateType gate_type = LoopGateType::kFirstOrder)
      : truncate_para(truncate_para), steps(steps), tau(tau), gate_type(gate_type) {}
};

/// @brief Executor for the loop update algorithm on square-lattice PEPS.
///
/// The loop update applies imaginary-time evolution gates arranged in 2x2
/// plaquette loops, following the full-environment truncation scheme of
/// Ref. [1] PRB 102, 075147 (2020).
///
/// @todo Return a SweepResult struct from each sweep (analogous to SimpleUpdateExecutor).
/// @todo Add RunSummary with convergence detection support.
/// @todo Add per-step metrics collection and observer callback.
template<typename TenElemT, typename QNT>
class LoopUpdateExecutor : public Executor {
  using Tensor = QLTensor<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;
  using LoopGatesT = LoopGates<Tensor>;
 public:
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

  bool DumpResult(std::string path, bool release_mem) {
    return peps_.Dump(path, release_mem);
  }

 private:

  double LoopUpdateSweep_(void);

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

  PEPST peps_;
};

}//qlpeps

#include "qlpeps/algorithm/loop_update/loop_update_impl.h"
#endif //QLPEPS_ALGORITHM_LOOP_UPDATE_LOOP_UPDATE_H
