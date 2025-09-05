// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project.
*              Simple public API wrappers for VMC optimization and measurement
*              that avoid verbose template arguments at call sites.
*/

#ifndef QLPEPS_API_VMC_API_H
#define QLPEPS_API_VMC_API_H

#include <memory>
#include <string>

#include <mpi.h>
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"

namespace qlpeps {

/**
 * @brief One-call VMC PEPS optimization wrapper.
 *
 * This function eliminates explicit template parameters in user code.
 * All template types are deduced from function arguments. In particular,
 * the Monte Carlo updater type is deduced from the tag object `mc_updater`.
 *
 * The function constructs the optimizer executor, runs Execute(), and returns
 * the owning pointer so callers can inspect results (e.g. energy trajectory,
 * best state, etc.).
 *
 * Example usage:
 *
 *   auto exec = VmcOptimize(params, sitps, comm, solver, HeatBathUpdater{});
 *
 * Input contract (MPI):
 * - Collective: all ranks in `comm` call this function in same form.
 * - Parameters: every rank must pass a valid data with identical
 *   logical content across ranks. Do NOT pass placeholders include `sitps`. 
 * - Environment: MPI is initialized; `comm` is valid and not `MPI_COMM_NULL`.
 *
 * @tparam TenElemT Tensor element type (deduced from sitps)
 * @tparam QNT Quantum number type (deduced from sitps)
 * @tparam MonteCarloSweepUpdater MC sweep updater strategy (deduced from mc_updater)
 * @tparam EnergySolver Model energy solver (deduced from solver)
 *
 * @param params Unified optimizer parameters
 * @param sitps Split-index TPS initial state
 * @param comm MPI communicator
 * @param solver Energy solver instance
 * @param mc_updater Tag object to deduce MC updater type (e.g., `HeatBathUpdater{}`)
 *
 * @return std::unique_ptr to the finished executor
 */
template <typename TenElemT,
          typename QNT,
          typename MonteCarloSweepUpdater,
          typename EnergySolver>
inline std::unique_ptr<VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>>
VmcOptimize(const VMCPEPSOptimizerParams &params,
            const SplitIndexTPS<TenElemT, QNT> &sitps,
            const MPI_Comm &comm,
            const EnergySolver &solver,
            MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater{}) {
  using ExecT = VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>;
  auto executor = std::make_unique<ExecT>(params, sitps, comm, solver, std::move(mc_updater));
  executor->Execute();
  return executor;
}

/**
 * @brief One-call Monte Carlo PEPS measurement wrapper.
 *
 * This function eliminates explicit template parameters in user code.
 * All template types are deduced from function arguments. The Monte Carlo
 * updater type is deduced from the tag object `mc_updater`.
 *
 * The function constructs the measurement executor, runs Execute(), and
 * returns the owning pointer so callers can inspect results.
 *
 * Example usage:
 *
 *   auto meas = MonteCarloMeasure(sitps, measure_params, comm, solver, HeatBathUpdater{});
 *
 * Input contract (MPI):
 * - Collective: all ranks in `comm` call this function in same form.
 * - Parameters: every rank must pass a valid data with identical
 *   logical content across ranks. Do NOT pass placeholders. 
 * - Environment: MPI is initialized; `comm` is valid and not `MPI_COMM_NULL`.
 *
 * @tparam TenElemT Tensor element type (deduced from sitps)
 * @tparam QNT Quantum number type (deduced from sitps)
 * @tparam MonteCarloSweepUpdater MC sweep updater strategy (deduced from mc_updater)
 * @tparam MeasurementSolver Model measurement solver (deduced from solver)
 *
 * @param sitps Split-index TPS state
 * @param measurement_params Unified measurement parameters
 * @param comm MPI communicator
 * @param solver Measurement solver instance
 * @param mc_updater Tag object to deduce MC updater type (e.g., `HeatBathUpdater{}`)
 *
 * @return std::unique_ptr to the finished measurer
 */
template <typename TenElemT,
          typename QNT,
          typename MonteCarloSweepUpdater,
          typename MeasurementSolver>
inline std::unique_ptr<MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>>
MonteCarloMeasure(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  const MCMeasurementParams &measurement_params,
                  const MPI_Comm &comm,
                  const MeasurementSolver &solver = MeasurementSolver{},
                  MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater{}) {
  using MeasT = MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>;
  auto measurer = std::make_unique<MeasT>(sitps, measurement_params, comm, solver, std::move(mc_updater));
  measurer->Execute();
  return measurer;
}

} // namespace qlpeps

#endif // QLPEPS_API_VMC_API_H


