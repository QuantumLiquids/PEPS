// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: QuantumLiquids/PEPS project. Variational Monte-Carlo PEPS parameters structure.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H

#include <vector>
#include "qlpeps/consts.h"                        //kTpsPath
#include "qlpeps/vmc_basic/configuration.h"  //Configuration
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"  //BMPSTruncatePara
#include "qlpeps/optimizer/optimizer_params.h"    //ConjugateGradientParams

namespace qlpeps {
/**
 * @struct MonteCarloParams
 * @brief Parameters for Monte Carlo sampling.
 *
 * - num_samples: Number of Monte Carlo samples (\f$N_{MC}\f$).
 * - num_warmup_sweeps: Number of warm-up sweeps before sampling.
 * - sweeps_between_samples: Number of sweeps between successive samples.
 * - wavefunction_path: Path to the wavefunction file.
 * - init_config: Initial configuration for the Markov chain.
 */
struct MonteCarloParams {
  size_t num_samples; // Number of Monte Carlo samples
  size_t num_warmup_sweeps; // Warm-up sweeps before sampling starts
  size_t sweeps_between_samples; // Sweeps between successive samples

  std::string config_path; // Path to load initial warmed-up configuration
  Configuration alternative_init_config; // Alternative initial configuration if no warmed-up configuration in disk

  MonteCarloParams() = default;

  MonteCarloParams(size_t samples,
                   size_t warmup_sweeps,
                   size_t sweeps_between,
                   const std::string &config_path,
                   const Configuration &alt_config)
    : num_samples(samples), num_warmup_sweeps(warmup_sweeps),
      sweeps_between_samples(sweeps_between), config_path(config_path),
      alternative_init_config(alt_config) {
  }
};

struct PEPSParams {
  BMPSTruncatePara truncate_para;
  std::string wavefunction_path;

  PEPSParams() = default;

  PEPSParams(const BMPSTruncatePara &trunc_para, const std::string &wavefunction_path)
    : truncate_para(trunc_para), wavefunction_path(wavefunction_path) {
  }
};

// Legacy VMCOptimizePara removed - use VMCPEPSOptimizerParams instead

/**
 * @struct MCMeasurementPara
 * @brief Parameters for Monte Carlo measurement.
 *
 * - bmps_trunc_para: Truncation parameters for boundary MPS.
 * - mc_samples: Number of Monte Carlo samples.
 * - mc_warm_up_sweeps: Number of warm-up sweeps.
 * - mc_sweeps_between_sample: Number of sweeps between samples.
 * - init_config: Initial configuration for MC sampling.
 * - wavefunction_path: Path to the wavefunction file.
 */
struct MCMeasurementPara {
  BMPSTruncatePara bmps_trunc_para;
  size_t mc_samples;
  size_t mc_warm_up_sweeps;
  size_t mc_sweeps_between_sample;
  Configuration init_config;
  std::string wavefunction_path;

  MCMeasurementPara(void) = default;

  MCMeasurementPara(BMPSTruncatePara trunc_para, size_t samples, size_t warm_up_sweeps,
                    size_t mc_sweeps_between_sample,
                    const std::vector<size_t> &occupancy,
                    const size_t rows, const size_t cols,
                    const std::string &wavefunction_path = kTpsPath) :
      bmps_trunc_para(trunc_para), mc_samples(samples),
      mc_warm_up_sweeps(warm_up_sweeps),
      mc_sweeps_between_sample(mc_sweeps_between_sample),
      init_config(rows, cols),
      wavefunction_path(wavefunction_path) {
    init_config.Random(occupancy);
  }

  MCMeasurementPara(BMPSTruncatePara trunc_para, size_t samples, size_t warm_up_sweeps,
                    size_t mc_sweeps_between_sample,
                    const Configuration &init_config,
                    const std::string &wavefunction_path = kTpsPath) :
      bmps_trunc_para(trunc_para), mc_samples(samples),
      mc_warm_up_sweeps(warm_up_sweeps),
      mc_sweeps_between_sample(mc_sweeps_between_sample),
      init_config(init_config),
      wavefunction_path(wavefunction_path) {}

  operator BMPSTruncatePara() const {
    return bmps_trunc_para;
  }
  operator MonteCarloParams() const {
    return {mc_samples, mc_warm_up_sweeps, mc_sweeps_between_sample, wavefunction_path, init_config};
  }
  operator PEPSParams() const {
    return {bmps_trunc_para, wavefunction_path};
  }
};
} //qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H
