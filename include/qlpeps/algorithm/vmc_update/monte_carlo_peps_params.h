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

namespace qlpeps {

enum WAVEFUNCTION_UPDATE_SCHEME {
  StochasticGradient,                     //0
  RandomStepStochasticGradient,           //1
  StochasticReconfiguration,              //2
  RandomStepStochasticReconfiguration,    //3
  NormalizedStochasticReconfiguration,    //4
  RandomGradientElement,                  //5
  BoundGradientElement,                   //6
  GradientLineSearch,                     //7
  NaturalGradientLineSearch               //8
};

// Function to convert enum to string
std::string WavefunctionUpdateSchemeString(WAVEFUNCTION_UPDATE_SCHEME scheme) {
  switch (scheme) {
    case StochasticGradient:return "StochasticGradient";
    case RandomStepStochasticGradient:return "RandomStepStochasticGradient";
    case StochasticReconfiguration:return "StochasticReconfiguration";
    case RandomStepStochasticReconfiguration:return "RandomStepStochasticReconfiguration";
    case NormalizedStochasticReconfiguration:return "NormalizedStochasticReconfiguration";
    case RandomGradientElement:return "RandomGradientElement";
    case BoundGradientElement:return "BoundGradientElement";
    case GradientLineSearch:return "GradientLineSearch";
    case NaturalGradientLineSearch:return "NaturalGradientLineSearch";
    default:return "Unknown scheme";
  }
}

///< Conjugate gradient parameters used in Stochastic Reconfiguration update PEPS
struct ConjugateGradientParams {
  size_t max_iter;
  double tolerance;
  int residue_restart_step;
  double diag_shift;

  ConjugateGradientParams(void) = default;

  ConjugateGradientParams(size_t max_iter, double tolerance, int residue_restart_step, double diag_shift)
      : max_iter(max_iter), tolerance(tolerance), residue_restart_step(residue_restart_step), diag_shift(diag_shift) {}
};

const std::vector<WAVEFUNCTION_UPDATE_SCHEME> stochastic_reconfiguration_method({StochasticReconfiguration,
                                                                                 RandomStepStochasticReconfiguration,
                                                                                 NormalizedStochasticReconfiguration,
                                                                                 NaturalGradientLineSearch});

struct MonteCarloParams {
  size_t num_samples;                // Number of Monte Carlo samples
  size_t num_warmup_sweeps;          // Warm-up sweeps before sampling starts
  size_t sweeps_between_samples;     // Sweeps between successive samples

  std::string config_path;           // Path to load initial warmed-up configuration
  Configuration alternative_init_config;  // Alternative initial configuration if no warmed-up configuration in disk
};

struct PEPSParams {
  BMPSTruncatePara truncate_para;
  std::string wavefunction_path;
};

struct VMCOptimizePara {
  VMCOptimizePara(void) = default;

  VMCOptimizePara(BMPSTruncatePara trunc_para, size_t samples, size_t warm_up_sweeps,
                  size_t mc_sweeps_between_sample,
                  const std::vector<size_t> &occupancy,
                  const size_t rows, const size_t cols,
                  const std::vector<double> &step_lens,
                  const WAVEFUNCTION_UPDATE_SCHEME update_scheme,
                  const ConjugateGradientParams &cg_params = ConjugateGradientParams(),
                  const std::string &wavefunction_path = kTpsPath) :
      bmps_trunc_para(trunc_para), mc_samples(samples),
      mc_warm_up_sweeps(warm_up_sweeps),
      mc_sweeps_between_sample(mc_sweeps_between_sample),
      init_config(rows, cols),
      step_lens(step_lens),
      update_scheme(update_scheme),
      wavefunction_path(wavefunction_path), cg_params(cg_params) {
    init_config.Random(occupancy);
  }

  VMCOptimizePara(BMPSTruncatePara trunc_para, size_t samples, size_t warm_up_sweeps,
                  size_t mc_sweeps_between_sample,
                  const Configuration &init_config,
                  const std::vector<double> &step_lens,
                  const WAVEFUNCTION_UPDATE_SCHEME update_scheme,
                  const ConjugateGradientParams &cg_params = ConjugateGradientParams(),
                  const std::string &wavefunction_path = kTpsPath) :
      bmps_trunc_para(trunc_para), mc_samples(samples),
      mc_warm_up_sweeps(warm_up_sweeps),
      mc_sweeps_between_sample(mc_sweeps_between_sample),
      init_config(init_config),
      step_lens(step_lens),
      update_scheme(update_scheme),
      wavefunction_path(wavefunction_path), cg_params(cg_params) {}

  operator BMPSTruncatePara() const {
    return bmps_trunc_para;
  }
  operator MonteCarloParams() const {
    return {mc_samples, mc_warm_up_sweeps, mc_sweeps_between_sample, wavefunction_path, init_config};
  }
  operator PEPSParams() const {
    return {bmps_trunc_para, wavefunction_path};
  }


  BMPSTruncatePara bmps_trunc_para; // Truncation error and bond dimension for compressing boundary MPS

  //Monte-Carlo parameters
  size_t mc_samples;
  size_t mc_warm_up_sweeps;
  size_t mc_sweeps_between_sample;

  Configuration init_config;

  std::vector<double> step_lens;
  WAVEFUNCTION_UPDATE_SCHEME update_scheme;
  std::string wavefunction_path;
  std::optional<ConjugateGradientParams> cg_params;
};

struct MCMeasurementPara {
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

  BMPSTruncatePara bmps_trunc_para;

  size_t mc_samples;
  size_t mc_warm_up_sweeps;
  size_t mc_sweeps_between_sample;

  Configuration init_config;
  std::string wavefunction_path;
};
}//qlpeps

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H
