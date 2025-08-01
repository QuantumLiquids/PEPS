// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. VMC PEPS Optimizer Executor parameters structure.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_PARAMS_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_PARAMS_H

#include <string>
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"
#include "qlpeps/vmc_basic/configuration.h"

namespace qlpeps {
struct VMCPEPSOptimizerParams {
  OptimizerParams optimizer_params;
  MonteCarloParams mc_params;
  PEPSParams peps_params;

  VMCPEPSOptimizerParams() = default;

  VMCPEPSOptimizerParams(const OptimizerParams &opt_params,
                         const MonteCarloParams &mc_params,
                         const PEPSParams &peps_params)
    : optimizer_params(opt_params), mc_params(mc_params), peps_params(peps_params) {
  }

  operator BMPSTruncatePara() const { return peps_params.truncate_para; }
  operator MonteCarloParams() const { return mc_params; }
  operator PEPSParams() const { return peps_params; }
  operator OptimizerParams() const { return optimizer_params; }
  //  operator VMCOptimizePara() const {
  //    return VMCOptimizePara(peps_params.truncate_para, mc_params.num_samples,
  //                           mc_params.num_warmup_sweeps, mc_params.sweeps_between_samples,
  //                           mc_params.alternative_init_config,
  //                           optimizer_params.core_params.step_lengths,
  //                           optimizer_params.update_scheme, optimizer_params.cg_params,
  //                           peps_params.wavefunction_path);
  //  }
};
} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_PARAMS_H
