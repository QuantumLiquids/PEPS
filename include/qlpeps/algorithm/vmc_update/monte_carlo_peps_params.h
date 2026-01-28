// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: QuantumLiquids/PEPS project. Variational Monte-Carlo PEPS parameters structure.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H

#include <string>                                    // std::string
#include <stdexcept>                                // std::runtime_error
#include <variant>                                  // std::variant
#include "qlpeps/vmc_basic/configuration.h"       // Configuration
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"  // BMPSTruncateParams
#include "qlpeps/two_dim_tn/tensor_network_2d/trg/trg_contractor.h"  // TRGTruncateParams
#include "qlpeps/algorithm/vmc_update/psi_consistency.h"

namespace qlpeps {
/**
 * @struct MonteCarloParams
 * @brief Parameters for Monte Carlo sampling.
 *
 * - num_samples: Number of Monte Carlo samples (\f$N_{MC}\f$).
 * - num_warmup_sweeps: Number of warm-up sweeps before sampling.
 * - sweeps_between_samples: Number of sweeps between successive samples.
 * - initial_config: User-provided initial configuration.
 * - is_warmed_up: Whether the initial configuration is already warmed up.
 * - config_dump_path: Path for dumping final configuration (empty = no dump).
 *                     Used by both VMC optimizer and measurement executors.
 *                     This is useful because the final configuration can be used as the warmed-up initial configuration for the next measurement.
 */
struct MonteCarloParams {
  size_t num_samples; // Number of Monte Carlo samples
  size_t num_warmup_sweeps; // Warm-up sweeps before sampling starts
  size_t sweeps_between_samples; // Sweeps between successive samples
  Configuration initial_config; // User-provided initial configuration
  bool is_warmed_up; // Whether initial configuration is warmed up
  std::string config_dump_path; // Path for dumping final configuration (empty = no dump)

  MonteCarloParams() = default;

  MonteCarloParams(size_t samples,
                   size_t warmup_sweeps,
                   size_t sweeps_between,
                   const Configuration &config,
                   bool is_warmed_up,
                   const std::string &config_dump_path = "")
    : num_samples(samples), num_warmup_sweeps(warmup_sweeps),
      sweeps_between_samples(sweeps_between), initial_config(config),
      is_warmed_up(is_warmed_up), config_dump_path(config_dump_path) {
  }

  /**
   * @brief Construct MonteCarloParams by loading configuration from file
   * 
   * Convenience constructor that loads configuration from file.
   * By default assumes loaded configs are warmed up (typical use case).
   * 
   * @param samples Number of Monte Carlo samples
   * @param warmup_sweeps Number of warm-up sweeps  
   * @param sweeps_between Sweeps between successive samples
   * @param config_file_path Path to configuration file to load
   * @param warmed_up Whether to treat loaded config as warmed up (default: true)
   * @param config_dump_path Path for dumping final configuration
   * @throws std::runtime_error if file doesn't exist or cannot be loaded
   */
  MonteCarloParams(size_t samples,
                   size_t warmup_sweeps,
                   size_t sweeps_between,
                   const std::string &config_file_path,
                   bool warmed_up = true,
                   const std::string &config_dump_path = "")
    : num_samples(samples), num_warmup_sweeps(warmup_sweeps),
      sweeps_between_samples(sweeps_between), is_warmed_up(warmed_up),
      config_dump_path(config_dump_path) {
    bool success = initial_config.Load(config_file_path, 0);
    if (!success) {
      throw std::runtime_error("Failed to load configuration from: " + config_file_path);
    }
  }
};

/**
 * @struct PEPSParams
 * @brief Parameters for PEPS calculations.
 * 
 * Simplified structure - user handles TPS I/O separately.
 * 
 * Use the appropriate constructor based on boundary condition:
 * - OBC: PEPSParams(BMPSTruncateParams) for BMPS contraction
 * - PBC: PEPSParams(TRGTruncateParams) for TRG contraction
 *
 * The stored variant is used to infer OBC vs PBC in higher-level APIs.
 */
struct PEPSParams {
  using BMPSParams = BMPSTruncateParams<qlten::QLTEN_Double>;
  using TRGParams = TRGTruncateParams<qlten::QLTEN_Double>;
  std::variant<BMPSParams, TRGParams> trunc_params;

  PEPSParams() = delete;

  /// OBC: BMPS contraction parameters
  explicit PEPSParams(const BMPSParams &trunc_para)
    : trunc_params(trunc_para) {
  }

  /// PBC: TRG contraction parameters
  explicit PEPSParams(const TRGParams &trg_para)
    : trunc_params(trg_para) {
  }

  bool IsOBC() const { return std::holds_alternative<BMPSParams>(trunc_params); }
  bool IsPBC() const { return std::holds_alternative<TRGParams>(trunc_params); }

  void SetBMPSParams(const BMPSParams &trunc_para) { trunc_params = trunc_para; }
  void SetTRGParams(const TRGParams &trg_para) { trunc_params = trg_para; }

  const BMPSParams &GetBMPSParams() const {
    if (!IsOBC()) {
      throw std::runtime_error("PEPSParams: requested BMPS params but TRG params are set.");
    }
    return std::get<BMPSParams>(trunc_params);
  }

  const TRGParams &GetTRGParams() const {
    if (!IsPBC()) {
      throw std::runtime_error("PEPSParams: requested TRG params but BMPS params are set.");
    }
    return std::get<TRGParams>(trunc_params);
  }
};

// Legacy VMCOptimizePara removed - use VMCPEPSOptimizerParams instead

/**
 * @struct MCMeasurementParams
 * @brief Unified parameters for Monte Carlo measurement.
 * 
 * Clean composition of Monte Carlo and PEPS parameters.
 * User explicitly provides all data - no magic, no guessing.
 */
struct MCMeasurementParams {
  MonteCarloParams mc_params;
  PEPSParams peps_params;
  std::string measurement_data_dump_path;  ///< Path for dumping measurement results (empty = current dir)
  RuntimeParams runtime_params; ///< Applied by MCPEPSMeasurer

  MCMeasurementParams() = delete;

  MCMeasurementParams(const MonteCarloParams &mc_params,
                      const PEPSParams &peps_params,
                      const std::string &measurement_data_dump_path = "./")
    : mc_params(mc_params), peps_params(peps_params), 
      measurement_data_dump_path(measurement_data_dump_path) {
    // Preserve historical default for measurement warnings.
    runtime_params.psi_consistency.threshold = 1e-4;
  }

  // Explicit accessors - no implicit conversions
  BMPSTruncateParams<qlten::QLTEN_Double> GetTruncatePara() const {
    return peps_params.GetBMPSParams();
  }
  TRGTruncateParams<qlten::QLTEN_Double> GetTRGTruncatePara() const {
    return peps_params.GetTRGParams();
  }
  const MonteCarloParams& GetMCParams() const {
    return mc_params;
  }
  const PEPSParams& GetPEPSParams() const {
    return peps_params;
  }
};
} //qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H
