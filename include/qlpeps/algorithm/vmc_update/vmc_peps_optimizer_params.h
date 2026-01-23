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
#include "monte_carlo_peps_params.h"               // MonteCarloParams, PEPSParams  
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"
#include "qlpeps/vmc_basic/configuration.h"
#include "qlpeps/consts.h"                          // kTpsPathBase
#include "qlpeps/algorithm/vmc_update/psi_consistency.h"

namespace qlpeps {
struct VMCPEPSOptimizerParams {
  OptimizerParams optimizer_params;
  MonteCarloParams mc_params;
  PEPSParams peps_params;
  std::string tps_dump_base_name;  ///< Base name for TPS dump files
  std::string tps_dump_path;  ///< Path for dumping optimized TPS (empty = no dump)

  // psi(S) consistency warning controls (per-rank). Applied by the optimizer/evaluator.
  RuntimeParams runtime_params;

  VMCPEPSOptimizerParams() : tps_dump_base_name(kTpsPathBase), tps_dump_path("./") {}

  VMCPEPSOptimizerParams(const OptimizerParams &opt_params,
                         const MonteCarloParams &mc_params,
                         const PEPSParams &peps_params,
                         const std::string &tps_dump_path = "./")
    : optimizer_params(opt_params), mc_params(mc_params), 
      peps_params(peps_params), tps_dump_base_name(kTpsPathBase), 
      tps_dump_path(tps_dump_path) {
  }

  // Explicit accessors - no implicit conversions
  auto GetTruncatePara() const { return peps_params.truncate_para; }
  auto GetTRGTruncatePara() const { return peps_params.trg_truncate_para; }
  const MonteCarloParams& GetMCParams() const { return mc_params; }
  const PEPSParams& GetPEPSParams() const { return peps_params; }
  const OptimizerParams& GetOptimizerParams() const { return optimizer_params; }

};
} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_PARAMS_H
