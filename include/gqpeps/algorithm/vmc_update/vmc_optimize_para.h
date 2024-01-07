// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: GraceQ/VMC-PEPS project. Variational Monte-Carlo PEPS parameters structure.
*/


#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H

#include <vector>
#include "gqpeps/consts.h"                        //kTpsPath
#include "gqpeps/two_dim_tn/tps/configuration.h"  //Configuration
#include "gqpeps/ond_dim_tn/boundary_mps/bmps.h"  //BMPSTruncatePara

namespace gqpeps {

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

const std::vector<WAVEFUNCTION_UPDATE_SCHEME> stochastic_reconfiguration_method({StochasticReconfiguration,
                                                                                 RandomStepStochasticReconfiguration,
                                                                                 NormalizedStochasticReconfiguration,
                                                                                 NaturalGradientLineSearch});

enum MC_SWEEP_SCHEME {  // 5th Jan, 2024 note : useless definition
  SequentiallyNNSiteFlip,
  CompressedLatticeKagomeLocalUpdate
};


struct VMCOptimizePara {
  VMCOptimizePara(void) = default;

  VMCOptimizePara(BMPSTruncatePara trunc_para, size_t samples, size_t warm_up_sweeps,
                  size_t mc_sweeps_between_sample,
                  const std::vector<size_t> &occupancy,
                  const size_t rows, const size_t cols,
                  const std::vector<double> &step_lens,
                  const WAVEFUNCTION_UPDATE_SCHEME update_scheme = StochasticGradient,
                  const std::string wavefunction_path = kTpsPath) :
      bmps_trunc_para(trunc_para), mc_samples(samples),
      mc_warm_up_sweeps(warm_up_sweeps),
      mc_sweeps_between_sample(mc_sweeps_between_sample),
      init_config(rows, cols),
      step_lens(step_lens),
      update_scheme(update_scheme),
      wavefunction_path(wavefunction_path) {
    init_config.Random(occupancy);
  }

  VMCOptimizePara(double truncErr, size_t Dmin, size_t Dmax, CompressMPSScheme compress_mps_scheme,
                  size_t samples, size_t warm_up_sweeps,
                  size_t mc_sweeps_between_sample,
                  const std::vector<size_t> &occupancy,
                  const size_t rows, const size_t cols,
                  const std::vector<double> &step_lens,
                  const WAVEFUNCTION_UPDATE_SCHEME update_scheme = StochasticGradient,
                  const std::string wavefunction_path = kTpsPath)
      : VMCOptimizePara(BMPSTruncatePara(Dmin, Dmax, truncErr, compress_mps_scheme), samples,
                        warm_up_sweeps, mc_sweeps_between_sample, occupancy, rows, cols,
                        step_lens, update_scheme, wavefunction_path) {}

  VMCOptimizePara(BMPSTruncatePara trunc_para, size_t samples, size_t warm_up_sweeps,
                  size_t mc_sweeps_between_sample,
                  const Configuration &init_config,
                  const std::vector<double> &step_lens,
                  const WAVEFUNCTION_UPDATE_SCHEME update_scheme = StochasticGradient,
                  const std::string wavefunction_path = kTpsPath) :
      bmps_trunc_para(trunc_para), mc_samples(samples),
      mc_warm_up_sweeps(warm_up_sweeps),
      mc_sweeps_between_sample(mc_sweeps_between_sample),
      init_config(init_config),
      step_lens(step_lens),
      update_scheme(update_scheme),
      wavefunction_path(wavefunction_path) {}

  VMCOptimizePara(double truncErr, size_t Dmin, size_t Dmax, CompressMPSScheme compress_mps_scheme,
                  size_t samples, size_t warm_up_sweeps,
                  size_t mc_sweeps_between_sample,
                  const Configuration &init_config,
                  const std::vector<double> &step_lens,
                  const WAVEFUNCTION_UPDATE_SCHEME update_scheme = StochasticGradient,
                  const std::string wavefunction_path = kTpsPath)
      : VMCOptimizePara(BMPSTruncatePara(Dmin, Dmax, truncErr, compress_mps_scheme), samples,
                        warm_up_sweeps, mc_sweeps_between_sample, init_config,
                        step_lens, update_scheme, wavefunction_path) {}

  operator BMPSTruncatePara() const {
    return bmps_trunc_para;
  }

  BMPSTruncatePara bmps_trunc_para; // Truncation Error and bond dimensionts for compressing boundary MPS

  //MC parameters
  size_t mc_samples;
  size_t mc_warm_up_sweeps;
  size_t mc_sweeps_between_sample;

//  // e.g. In spin model, how many spin up sites and how many spin down sites.
//  std::vector<size_t> occupancy_num;

  Configuration init_config;

  std::vector<double> step_lens;
  WAVEFUNCTION_UPDATE_SCHEME update_scheme;
  std::string wavefunction_path;

  MC_SWEEP_SCHEME mc_sweep_scheme = SequentiallyNNSiteFlip;
};

}//gqpeps

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_VMC_OPTIMIZE_PARA_H
