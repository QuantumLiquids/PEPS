// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: GraceQ/VMC-PEPS project. Variational Monte-Carlo PEPS executor.
*/

#ifndef GQPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H
#define GQPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H

#include "boost/mpi.hpp"                            //boost::mpi

#include "gqpeps/two_dim_tn/tps/tps.h"              // TPS
#include "gqpeps/two_dim_tn/tps/split_index_tps.h"  //SplitIndexTPS

#include "gqpeps/algorithm/vmc_update/vmc_optimize_para.h"  //VMCOptimizePara

namespace gqpeps {
using namespace gqten;

///< For stochastic reconfiguration
struct ConjugateGradientParams {
  size_t max_iter;
  double tolerance;
  int residue_restart_step;
  double diag_shift;

  ConjugateGradientParams(size_t max_iter, double tolerance, int residue_restart_step, double diag_shift)
      : max_iter(max_iter), tolerance(tolerance), residue_restart_step(residue_restart_step), diag_shift(diag_shift) {}
};

/**
 * Finite-size PEPS optimization executor in Variational Monte-Carlo method.
 *
 * @tparam TenElemT wavefunctional elementary type, real or complex
 * @tparam QNT quantum number type
 * @tparam EnergySolver Energy solver, corresponding to the model
 * @tparam WaveFunctionComponentType the derived class of WaveFunctionComponent, control the monte carlo sweep method
 */
template<typename TenElemT, typename QNT, typename EnergySolver, typename WaveFunctionComponentType>
class VMCPEPSExecutor : public Executor {
 public:
  using Tensor = GQTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;

  VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                  const TPST &tps_init,
                  const boost::mpi::communicator &world,
                  const EnergySolver &solver = EnergySolver());

  VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                  const SITPST &sitpst_init,
                  const boost::mpi::communicator &world,
                  const EnergySolver &solver = EnergySolver());

  //Load Data from path
  VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                  const size_t ly, const size_t lx,
                  const boost::mpi::communicator &world,
                  const EnergySolver &solver = EnergySolver());

  void Execute(void) override;

  void LoadTenData(void);

  void LoadTenData(const std::string &tps_path);

  void DumpData(const bool release_mem = false);

  void DumpData(const std::string &tps_path, const bool release_mem = false);

  VMCOptimizePara optimize_para;

  ConjugateGradientParams cg_params = ConjugateGradientParams(100, 1e-8, 20, 1e-2);
 protected:
  // Level 1 Member Functions
  void WarmUp_(void);

  void LineSearchOptimizeTPS_(void);
  void IterativeOptimizeTPS_(void);

  void Measure_(void);

  // Level 2 Member Functions
  void PrintExecutorInfo_(void);
  void ReserveSamplesDataSpace_(void);

  void IterativeOptimizeTPSStep_(const size_t iter);
  void LineSearch_(const SITPST &search_dir,
                   const std::vector<double> &strides);

  // Level 3 Member Functions
  void UpdateTPSByVecAndSynchronize_(const VMCPEPSExecutor::SITPST &grad, double step_len);
  void BoundGradElementUpdateTPS_(VMCPEPSExecutor::SITPST &grad, double step_len);
  std::pair<size_t, double> StochReconfigUpdateTPS_(const VMCPEPSExecutor::SITPST &grad,
                                                    double step_len,
                                                    const SITPST &init_guess,
                                                    const bool normalize_natural_grad);

  // Lowest Level Member functions who could directly change data
  ///< functions who cloud directly act on sample data
  TenElemT SampleEnergy_(void);
  void SampleEnergyAndHols_(void);
  void ClearEnergyAndHoleSamples_(void);

  ///< statistic and gradient operation functions
  ///< return the gradient;
  SITPST GatherStatisticEnergyAndGrad_(void);
  void GradientRandElementSign_();
  size_t CalcNaturalGradient_(const VMCPEPSExecutor::SITPST &grad, const SITPST &init_guess);

  std::vector<double> MCSweep_(void);
  // Input Data Region
  const boost::mpi::communicator world_;

  size_t lx_; //cols
  size_t ly_; //rows

  EnergySolver energy_solver_;

  //Runtime Data Region
  SITPST split_index_tps_;  //also can be input/output
  bool warm_up_;
  bool stochastic_reconfiguration_update_class_;
  WaveFunctionComponentType tps_sample_;

  std::vector<TenElemT> energy_samples_;
  ///<outside vector indices corresponding to the local hilbert space basis
//  DuoMatrix<std::vector<std::vector<Tensor *> >> gten_samples_;
//  DuoMatrix<std::vector<std::vector<Tensor *> >> g_times_energy_samples_;

  ///< vector index corresponding to the samples.
  std::vector<SITPST> gten_samples_; //useful for stochastic reconfiguration

  SITPST gten_sum_; // the holes * psi^(-1)
  SITPST gten_ave_; // average of gten_sum_;
  SITPST g_times_energy_sum_;

  SITPST grad_;
  SITPST natural_grad_;
  std::vector<double> grad_norm_;

  //Output/Dump Data Region
  std::vector<TenElemT> energy_trajectory_;
  std::vector<TenElemT> energy_error_traj_;
};

}//gqpeps;

#include "gqpeps/algorithm/vmc_update/vmc_peps_impl.h"

#endif //GQPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H
