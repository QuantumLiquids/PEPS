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
using namespace qlten;

/**
 * @brief Elegant VMC PEPS executor that uses the optimizer for clean separation of concerns
 * 
 * This executor is designed as a drop-in replacement for VMCPEPSExecutor, providing
 * the same interface while using the optimizer for all optimization logic. It offers
 * better modularity, testability, and maintainability.
 * 
 * MPI Behavior:
 * - Monte Carlo sampling is performed on all ranks in parallel
 * - Gradient calculation is performed on all ranks and gathered to master
 * - State updates (gradient descent, stochastic reconfiguration) are performed only on master rank
 * - Updated states are broadcast to all ranks to maintain synchronization
 * - Stochastic reconfiguration equation solving involves all cores working together
 * 
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 * @tparam MonteCarloSweepUpdater Monte Carlo updater type
 * @tparam EnergySolver Energy solver type
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
   * @brief Constructor with TPS loaded from file path.
   * 
   * Convenience constructor for users who have TPS data stored on disk.
   * This is the recommended approach when starting optimization from saved TPS.
   * The lattice dimensions (ly, lx) are automatically inferred from the
   * initial configuration size: ly = initial_config.rows(), lx = initial_config.cols().
   * 
   * @param params Unified optimizer parameters (must contain valid initial_config)
   * @param tps_path Path to TPS data files on disk
   * @param comm MPI communicator
   * @param solver Energy solver for optimization
   * 
   * @note The initial_config in params.mc_params must be properly sized to determine lattice dimensions
   * @note This constructor automatically loads TPS from disk and initializes the optimization system
   * 
   * @todo REFACTOR PROPOSAL - Replace with static factory function for better design:
   * @todo   static std::unique_ptr<VMCPEPSOptimizerExecutor> 
   * @todo   CreateByLoadingTPS(const VMCPEPSOptimizerParams& params,
   * @todo                      const std::string& tps_path,
   * @todo                      const MPI_Comm& comm,
   * @todo                      const EnergySolver& solver);
   * @todo
   * @todo This follows the same refactor proposal as the base class - using static factory
   * @todo functions instead of multiple constructors for better single-responsibility design.
   * @todo See base class MonteCarloPEPSBaseExecutor for detailed rationale.
   */
  VMCPEPSOptimizerExecutor(const VMCPEPSOptimizerParams &params,
                           const std::string &tps_path,
                           const MPI_Comm &comm,
                           const EnergySolver &solver);

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
  void SampleEnergyAndHoles_(void);
  TenElemT SampleEnergy_(void);
  void ClearEnergyAndHoleSamples_(void);

  // Statistics gathering
  std::tuple<TenElemT, SITPST, double> GatherStatisticEnergyAndGrad_(void);

  // Acceptance rate checking
  bool AcceptanceRateCheck(const std::vector<double> &accept_rate) const;

  // Data dumping helpers
  void DumpVecData(const std::string &path, const std::vector<TenElemT> &data);
  void DumpVecDataDouble(const std::string &path, const std::vector<double> &data);

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
  SITPST gten_sum_;
  SITPST g_times_energy_sum_;
  SITPST grad_;
  SITPST gten_ave_;

  // Best state tracking
  double en_min_;
  SITPST tps_lowest_;

  // Stochastic reconfiguration storage
  std::vector<SITPST> gten_samples_;
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