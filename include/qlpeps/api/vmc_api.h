// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project.
*              Simple public API wrappers for VMC optimization and measurement.
*
* Note (2026-01): OBC(BMPS) vs PBC(TRG) is inferred from SplitIndexTPS boundary condition,
* and cross-checked against PEPSParams. The wrappers return backend-agnostic result structs
* instead of executor pointers to keep a single return type.
*/

#ifndef QLPEPS_API_VMC_API_H
#define QLPEPS_API_VMC_API_H

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>
#include "qlten/qlten.h"
#include "qlpeps/optimizer/spike_detection.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"

namespace qlpeps {

template <typename TenElemT, typename QNT>
struct VmcOptimizeResult {
  SplitIndexTPS<TenElemT, QNT> state;
  SplitIndexTPS<TenElemT, QNT> best_state;
  double min_energy = 0.0;
  std::vector<TenElemT> energy_trajectory;
  std::vector<double> energy_error_trajectory;
  std::vector<double> gradient_norms;
  SpikeStatistics spike_stats;   ///< Spike detection statistics for the run
};

template <typename TenElemT>
using ObservableRegistry =
    std::unordered_map<std::string, std::pair<std::vector<TenElemT>, std::vector<double>>>;

template <typename TenElemT, typename QNT>
struct MonteCarloMeasureResult {
  std::pair<TenElemT, double> energy;
  ObservableRegistry<TenElemT> registry;
  std::string dump_path;
};

inline void CrossCheckBackendOrThrow_(BoundaryCondition bc, const PEPSParams &peps_params, const char *api_name) {
  if (bc == BoundaryCondition::Periodic) {
    if (!peps_params.IsPBC()) {
      throw std::invalid_argument(std::string(api_name) +
                                  ": SITPS is PBC but PEPSParams does not hold TRG truncation params.");
    }
  } else {
    if (peps_params.IsPBC()) {
      throw std::invalid_argument(std::string(api_name) +
                                  ": SITPS is OBC but PEPSParams holds TRG truncation params.");
    }
  }
}

/**
 * @brief One-call VMC PEPS optimization wrapper.
 *
 * OBC vs PBC is inferred from `sitps.GetBoundaryCondition()` and cross-checked against `params.peps_params`.
 * The executor is constructed, executed, and a backend-agnostic result struct is returned.
 */
template <typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
inline VmcOptimizeResult<TenElemT, QNT>
VmcOptimize(const VMCPEPSOptimizerParams &params,
            const SplitIndexTPS<TenElemT, QNT> &sitps,
            const MPI_Comm &comm,
            const EnergySolver &solver,
            MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater{}) {
  const BoundaryCondition bc = sitps.GetBoundaryCondition();
  CrossCheckBackendOrThrow_(bc, params.peps_params, "VmcOptimize");

  constexpr bool kSupportsBMPS =
      MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT, BMPSContractor>;
  constexpr bool kSupportsTRG =
      MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT, TRGContractor>;

  if (bc == BoundaryCondition::Periodic) {
    if constexpr (kSupportsTRG) {
      using ExecT = VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, TRGContractor>;
      auto executor = std::make_unique<ExecT>(params, sitps, comm, solver, std::move(mc_updater));
      executor->Execute();
      VmcOptimizeResult<TenElemT, QNT> out;
      out.state = executor->GetState();
      out.best_state = executor->GetBestState();
      out.min_energy = executor->GetMinEnergy();
      out.energy_trajectory = executor->GetEnergyTrajectory();
      out.energy_error_trajectory = executor->GetEnergyErrorTrajectory();
      out.gradient_norms = executor->GetGradientNorms();
      out.spike_stats = executor->GetSpikeStatistics();
      return out;
    } else {
      throw std::invalid_argument("VmcOptimize: PBC requested but template args do not support TRG/PBC.");
    }
  } else {
    if constexpr (kSupportsBMPS) {
      using ExecT = VMCPEPSOptimizer<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver>;
      auto executor = std::make_unique<ExecT>(params, sitps, comm, solver, std::move(mc_updater));
      executor->Execute();
      VmcOptimizeResult<TenElemT, QNT> out;
      out.state = executor->GetState();
      out.best_state = executor->GetBestState();
      out.min_energy = executor->GetMinEnergy();
      out.energy_trajectory = executor->GetEnergyTrajectory();
      out.energy_error_trajectory = executor->GetEnergyErrorTrajectory();
      out.gradient_norms = executor->GetGradientNorms();
      out.spike_stats = executor->GetSpikeStatistics();
      return out;
    } else {
      throw std::invalid_argument("VmcOptimize: OBC requested but template args do not support BMPS/OBC.");
    }
  }
}

/**
 * @brief One-call Monte Carlo PEPS measurement wrapper.
 *
 * OBC vs PBC is inferred from `sitps.GetBoundaryCondition()` and cross-checked against `measurement_params.peps_params`.
 * The measurer is constructed, executed, and aggregated stats are returned.
 */
template <typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename MeasurementSolver>
inline MonteCarloMeasureResult<TenElemT, QNT>
MonteCarloMeasure(const SplitIndexTPS<TenElemT, QNT> &sitps,
                  const MCMeasurementParams &measurement_params,
                  const MPI_Comm &comm,
                  const MeasurementSolver &solver = MeasurementSolver{},
                  MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater{}) {
  const BoundaryCondition bc = sitps.GetBoundaryCondition();
  CrossCheckBackendOrThrow_(bc, measurement_params.peps_params, "MonteCarloMeasure");

  constexpr bool kSupportsBMPS =
      MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT, BMPSContractor>;
  constexpr bool kSupportsTRG =
      MonteCarloSweepUpdaterConcept<MonteCarloSweepUpdater, TenElemT, QNT, TRGContractor>;

  MonteCarloMeasureResult<TenElemT, QNT> out;
  out.dump_path = measurement_params.measurement_data_dump_path;

  if (bc == BoundaryCondition::Periodic) {
    if constexpr (kSupportsTRG) {
      using MeasT = MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver, TRGContractor>;
      auto measurer = std::make_unique<MeasT>(sitps, measurement_params, comm, solver, std::move(mc_updater));
      measurer->Execute();
      out.energy = measurer->OutputEnergy();
      out.registry = measurer->ObservableRegistry();
      return out;
    } else {
      throw std::invalid_argument("MonteCarloMeasure: PBC requested but template args do not support TRG/PBC.");
    }
  } else {
    if constexpr (kSupportsBMPS) {
      using MeasT = MCPEPSMeasurer<TenElemT, QNT, MonteCarloSweepUpdater, MeasurementSolver>;
      auto measurer = std::make_unique<MeasT>(sitps, measurement_params, comm, solver, std::move(mc_updater));
      measurer->Execute();
      out.energy = measurer->OutputEnergy();
      out.registry = measurer->ObservableRegistry();
      return out;
    } else {
      throw std::invalid_argument("MonteCarloMeasure: OBC requested but template args do not support BMPS/OBC.");
    }
  }
}

} // namespace qlpeps

#endif // QLPEPS_API_VMC_API_H
