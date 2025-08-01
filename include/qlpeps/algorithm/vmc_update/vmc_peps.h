// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: QuantumLiquids/PEPS project. Variational Monte-Carlo PEPS executor.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H

#include "qlpeps/two_dim_tn/tps/tps.h"                            // TPS
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"                //SplitIndexTPS
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"  //VMCOptimizePara
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_base.h"

namespace qlpeps {
using namespace qlten;

/**
 * Finite-size PEPS optimization executor in Variational Monte-Carlo method.
 *
 * @tparam TenElemT wavefunctional elementary type, real or complex
 * @tparam QNT quantum number type
 * @tparam MonteCarloSweepUpdater functor to define the monte carlo sweep update strategies, find more info in MonteCarloPEPSBaseExecutor
 * @tparam EnergySolver functor to define the evaluation of the model energy and holes upon wave function component of PEPS
 * 
 * @details The EnergySolver functor should inherit from ModelEnergySolver and implement CalEnergyAndHolesImpl method with signature:
 * template<typename TenElemT, typename QNT, bool calchols>
 * TenElemT CalEnergyAndHolesImpl(
 *     const SplitIndexTPS<TenElemT, QNT>*,
 *     TPSWaveFunctionComponent<TenElemT, QNT>*,
 *     TensorNetwork2D<TenElemT, QNT>&,
 *     std::vector<TenElemT>&
 * )
 * 
 * Built-in energy solvers in model_solvers/:
 * - TransverseIsingSquare: Transverse field Ising model on square lattice
 * - SquareSpinOneHalfXXZModel: Spin-1/2 AFM Heisenberg model on square lattice
 * - SquareSpinOneHalfJ1J2XXZModel: Spin-1/2 J1-J2 Heisenberg model on square lattice
 * - SpinOneHalfTriHeisenbergSqrPEPS: Triangular Spin-1/2 AFM Heisenberg model on square lattice PEPS
 * - SpinOneHalfTriJ1J2HeisenbergSqrPEPS: Triangular Spin-1/2 J1-J2 Heisenberg model on square lattice PEPS
 * - SquaretJModel: t-J model on square lattice
 * - SquareSpinlessFermion: Spinless free fermion model on square lattice
 * - SquareHubbardModel: Hubbard model on square lattice
 * 
 * Below class may be helpful for the implementation of the EnergySolver:
 * - SquareNNModelEnergySolver: Base class for nearest-neighbor fermion models on square lattice
 * 
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
class VMCPEPSExecutor : public MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater> {
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::comm_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::mpi_size_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::rank_;

  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::split_index_tps_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::tps_sample_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::ly_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::lx_;

  using Tensor = QLTensor<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using WaveFunctionComponentT = TPSWaveFunctionComponent<TenElemT, QNT>;

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

  const SITPST &GetState(void) const noexcept { return split_index_tps_; }

  void DumpData(const bool release_mem = false);
  void DumpData(const std::string &tps_path, const bool release_mem = false);

  // Add getter for optimization parameters
  const VMCOptimizePara& GetOptimizePara() const noexcept { return optimize_para; }
  
  // Add getter for current energy
  double GetCurrentEnergy() const noexcept { 
    return energy_trajectory_.empty() ? std::numeric_limits<double>::max() : 
           Real(energy_trajectory_.back()); 
  }
  
  // Add getter for minimum energy
  double GetMinEnergy() const noexcept { return en_min_; }

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

  std::mt19937 random_engine_{std::random_device{}()};
};

}//qlpeps;

#include "qlpeps/algorithm/vmc_update/vmc_peps_impl.h"

#endif //QLPEPS_ALGORITHM_VMC_UPDATE_VMC_UPDATE_H
