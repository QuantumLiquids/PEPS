// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. VMC PEPS executor using the optimizer.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_H

#include <vector>
#include <memory>
#include <functional>
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_base.h"
#include "qlpeps/optimizer/optimizer.h"
// Removed circular include - vmc_peps.h is deprecated
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"

namespace qlpeps {

/**
 * @brief VMC PEPS optimization executor with clear separation from the optimizer.
 *
 * Design: drop-in replacement of the legacy VMCPEPSExecutor. This class owns Monte Carlo
 * sampling and MPI gathering/broadcast, while the Optimizer owns update logic.
 *
 * Math (consistent with tutorials):
 * - Wavefunction and weights: \f$\Psi(S;\theta),\; w_{\text{raw}}(S)=|\Psi(S)|^2,\; w = w_{\text{raw}}/Z\,.\f$
 * - Local energy: \f$ E_{\mathrm{loc}}(S) = \sum_{S'} \dfrac{\Psi^*(S')}{\Psi^*(S)}\, \langle S'|H|S\rangle. \f$
 * - Log-derivative: \f$ O_i^*(S) = \dfrac{\partial \ln \Psi^*(S)}{\partial \theta_i^*}. \f$
 * - Energy and complex gradient: \f$ E = \langle E_{\mathrm{loc}} \rangle,\; \partial E/\partial \theta_i^*
 *   = \langle E_{\mathrm{loc}}^* O_i^* \rangle - E^* \langle O_i^* \rangle. \f$
 *
 * Accumulators during MC sampling:
 * - \f$\sum O_i^*\Rightarrow\f$ `Ostar_sum_`
 * - \f$\sum E_{\mathrm{loc}}^* O_i^*\Rightarrow\f$ `ELocConj_Ostar_sum_`
 * Master rank computes the final gradient; SR uses `Ostar_samples_` and `Ostar_mean_`.
 *
 * MPI behavior:
 * - MC sampling and local quantity evaluation on all ranks;
 * - Gradient gathering and parameter updates on master only;
 * - SR system solving on all ranks; single state broadcast in the evaluator, final broadcast on completion.
 *
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 * @tparam MonteCarloSweepUpdater MC sweep updater strategy
 * @tparam EnergySolver Model energy solver
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
class VMCPEPSOptimizerExecutor : public MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater> {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using OptimizerT = Optimizer<TenElemT, QNT>;
  using BaseExecutor = MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>;

  using WaveFunctionComponentT = TPSWaveFunctionComponent<TenElemT, QNT>;

  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::comm_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::mpi_size_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::rank_;

  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::split_index_tps_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::tps_sample_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::ly_;
  using MonteCarloPEPSBaseExecutor<TenElemT, QNT, MonteCarloSweepUpdater>::lx_;

  /**
   * @brief Constructor with explicit TPS provided by user.
   * 
   * User provides all data explicitly - no hidden file loading.
   * This constructor gives users complete control over the input data.
   * 
   * @param params Unified optimizer parameters (Optimizer + MC + PEPS)
   * @param sitpst_init Split-index TPS provided by user
   * @param comm MPI communicator
   * @param solver Energy solver for optimization
   */
  VMCPEPSOptimizerExecutor(const VMCPEPSOptimizerParams &params,
                           const SITPST &sitpst_init,
                           const MPI_Comm &comm,
                           const EnergySolver &solver);

  /**
   * @brief Static factory function to create optimizer executor by loading TPS from file path.
   * 
   * Convenience factory for users who have TPS data stored on disk.
   * This is the recommended approach when starting optimization from saved TPS.
   
   * 
   * @param params Unified optimizer parameters (must contain valid initial_config)
   * @param tps_path Path to TPS data files on disk
   * @param comm MPI communicator
   * @param solver Energy solver for optimization
   * @return Unique pointer to the created optimizer executor
   * 
   * @note The initial_config in params.mc_params must be properly sized to determine lattice dimensions
   * @note This factory automatically loads TPS from disk and initializes the optimization system
   * 
   * Usage:
   *   auto executor = VMCPEPSOptimizerExecutor::CreateByLoadingTPS(params, tps_path, comm, solver);
   */
  static std::unique_ptr<VMCPEPSOptimizerExecutor> 
  CreateByLoadingTPS(const VMCPEPSOptimizerParams& params,
                     const std::string& tps_path,
                     const MPI_Comm& comm,
                     const EnergySolver& solver);

  // Main execution method
  void Execute(void) override;

  // Data access methods - matching VMCPEPSExecutor interface
  const SITPST &GetState() const noexcept { return this->split_index_tps_; }
  const SITPST &GetOptimizedState() const { return this->split_index_tps_; }
  const SITPST &GetBestState() const { return tps_lowest_; }
  double GetMinEnergy() const noexcept { return en_min_; }
  double GetCurrentEnergy() const noexcept {
    return energy_trajectory_.empty() ? std::numeric_limits<double>::max() :
           Real(energy_trajectory_.back());
  }
  const std::vector<TenElemT> &GetEnergyTrajectory() const { return energy_trajectory_; }
  const std::vector<double> &GetEnergyErrorTrajectory() const { return energy_error_traj_; }
  const std::vector<double> &GetGradientNorms() const { return grad_norm_; }

  const VMCPEPSOptimizerParams &GetParams() const noexcept { return params_; }

  // Data dumping methods
  void DumpData(const bool release_mem = false);
  void DumpData(const std::string &tps_base_name, const bool release_mem = false);

  // Optimizer access for advanced usage
  OptimizerT &GetOptimizer() { return optimizer_; }
  const OptimizerT &GetOptimizer() const { return optimizer_; }

  // Callback customization
  void SetOptimizationCallback(const typename OptimizerT::OptimizationCallback &callback) {
    optimization_callback_ = callback;
  }

  // Energy evaluator customization
  void SetEnergyEvaluator(std::function<std::tuple<TenElemT, SITPST, double>(const SITPST &)> evaluator) {
    custom_energy_evaluator_ = evaluator;
  }

 protected:
  // Monte Carlo sampling methods
  /**
   * @brief Compute local energy and hole tensors for the current configuration,
   * and accumulate O^* and E_loc^* O^*.
   *
   * Mapping:
   * - E_loc from EnergySolver::CalEnergyAndHoles();
   * - O^*(S): boson uses inverse_amplitude * holes; fermion uses CalGTenForFermionicTensors(...);
   * - Accumulators: Ostar_sum_ += O^*, ELocConj_Ostar_sum_ += E_loc^* · O^*;
   * - SR: append per-sample O^*(S) to Ostar_samples_ when enabled.
   */
  void SampleEnergyAndHoles_(void);
  TenElemT SampleEnergy_(void);
  void ClearEnergyAndHoleSamples_(void);

  // Statistics gathering
  /**
   * @brief Gather energy and gradient over MPI and return (energy, gradient, energy_error).
   *
   * - energy: average over ranks and broadcast;
   * - gradient: computed on master as ⟨E_loc^* O^*⟩ − E^* ⟨O^*⟩;
   * - energy_error: valid on master only.
   */
  std::tuple<TenElemT, SITPST, double> GatherStatisticEnergyAndGrad_(void);

  // Acceptance rate checking
  bool AcceptanceRateCheck(const std::vector<double> &accept_rate) const;

  // Data dumping helpers
  void DumpVecData_(const std::string &path, const std::vector<TenElemT> &data);
  void DumpVecDataDouble_(const std::string &path, const std::vector<double> &data);

 private:
  VMCPEPSOptimizerParams params_;  // New parameter structure
  EnergySolver energy_solver_;

  // Optimizer instance
  OptimizerT optimizer_;

  // Optimization callback
  typename OptimizerT::OptimizationCallback optimization_callback_;

  // Custom energy evaluator (optional)
  std::function<std::tuple<TenElemT, SITPST, double>(const SITPST &)> custom_energy_evaluator_;

  // Data storage
  std::vector<TenElemT> energy_samples_;
  std::vector<TenElemT> energy_trajectory_;
  std::vector<double> energy_error_traj_;
  std::vector<double> grad_norm_;

  // Gradient calculation storage
  /**
   * @brief Accumulator of O_i^* over MC samples (Σ O_i^*)
   *
   * Mathematical mapping: O_i^*(S) = ∂ ln Ψ^*(S) / ∂ θ_i^*. Under MC sampling,
   * averaging by the number of samples yields ⟨O^*⟩.
   */
  SITPST Ostar_sum_;
  /**
   * @brief Accumulator of E_loc^* · O_i^* over MC samples (Σ E_loc^* O_i^*)
   *
   * Used to compute the complex gradient via
   * ∂E/∂θ_i^* = ⟨E_loc^* O_i^*⟩ − E^* ⟨O_i^*⟩.
   */
  SITPST ELocConj_Ostar_sum_;
  /**
   * @brief Final gradient tensor (valid on master after gather)
   */
  SITPST grad_;
  /**
   * @brief Mean of O_i^* under MC sampling: ⟨O^*⟩
   */
  SITPST Ostar_mean_;

  // Best state tracking
  double en_min_;
  SITPST tps_lowest_;

  // Stochastic reconfiguration storage
  /**
   * @brief Per-sample O^*(S) tensors for SR S-matrix construction
   */
  std::vector<SITPST> Ostar_samples_;
  bool stochastic_reconfiguration_update_class_;

  // Current energy error for optimizer access
  double current_energy_error_;

  // Helper methods
  void ReserveSamplesDataSpace_(void);
  void PrintExecutorInfo_(void);
  void ValidateState_(const SITPST &state);


  // CRITICAL: Helper method to ensure wavefunction component consistency
  // This encapsulates the intertwined relationship between split_index_tps_ and tps_sample_
  void UpdateWavefunctionComponent_();

  // Default energy evaluator
  std::tuple<TenElemT, SITPST, double> DefaultEnergyEvaluator_(const SITPST &state);
};

} // namespace qlpeps

#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_impl.h"

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_H 