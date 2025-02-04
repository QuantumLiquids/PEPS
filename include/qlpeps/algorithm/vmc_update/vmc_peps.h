// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: QuantumLiquids/PEPS project. Variational Monte-Carlo PEPS executor.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H

#include "qlpeps/two_dim_tn/tps/tps.h"              // TPS
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"  //SplitIndexTPS

#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"  //VMCOptimizePara
#include "monte_carlo_peps_base.h"

namespace qlpeps {
using namespace qlten;

/**
 * Finite-size PEPS optimization executor in Variational Monte-Carlo method.
 *
 * @tparam TenElemT wavefunctional elementary type, real or complex
 * @tparam QNT quantum number type
 * @tparam WaveFunctionComponentType the derived class of WaveFunctionComponent, defines the monte carlo sweep method
 * @tparam EnergySolver Energy solver, define evaluation of the model energy and holes in PEPS
 */
template<typename TenElemT, typename QNT, typename WaveFunctionComponentType, typename EnergySolver>
class VMCPEPSExecutor : public MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType> {
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType>::comm_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType>::mpi_size_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType>::rank_;

  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType>::split_index_tps_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType>::tps_sample_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType>::ly_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, WaveFunctionComponentType>::lx_;

  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;
 public:

  VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                  const TPST &tps_init,
                  const MPI_Comm &comm,
                  const EnergySolver &solver = EnergySolver());

  VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                  const SITPST &sitpst_init,
                  const MPI_Comm &comm,
                  const EnergySolver &solver = EnergySolver());

  //Load Data from path
  VMCPEPSExecutor(const VMCOptimizePara &optimize_para,
                  const size_t ly, const size_t lx,
                  const MPI_Comm &comm,
                  const EnergySolver &solver = EnergySolver());

  void Execute(void) override;

  const SITPST &GetState(void) const { return split_index_tps_; }

  void DumpData(const bool release_mem = false);

  void DumpData(const std::string &tps_path, const bool release_mem = false);

  VMCOptimizePara optimize_para;
 protected:
  // Level 1 Member Functions

  void LineSearchOptimizeTPS_(void);
  void IterativeOptimizeTPS_(void);

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
  void NormalizeTPS_(void);

  // Lowest Level Member functions who could directly change data
  ///< functions who cloud directly act on sample data
  TenElemT SampleEnergy_(void);
  void SampleEnergyAndHols_(void);
  void ClearEnergyAndHoleSamples_(void);

  ///< statistic and gradient operation functions
  ///< return [energy, gradient];
  std::pair<TenElemT, SITPST> GatherStatisticEnergyAndGrad_(void);
  void GradientRandElementSign_();
  size_t CalcNaturalGradient_(const VMCPEPSExecutor::SITPST &grad, const SITPST &init_guess);

  bool AcceptanceRateCheck(const std::vector<double> &) const;

  EnergySolver energy_solver_;

  //Runtime Data Region
  bool stochastic_reconfiguration_update_class_;

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

  double en_min_;

  //Output/Dump Data Region
  SITPST tps_lowest_; //lowest energy tps
  std::vector<TenElemT> energy_trajectory_;
  std::vector<double> energy_error_traj_;
};

}//qlpeps;

#include "qlpeps/algorithm/vmc_update/vmc_peps_impl.h"

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H
