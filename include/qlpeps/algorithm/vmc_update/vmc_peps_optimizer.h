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
#include "qlten/qlten.h"
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_engine.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/algorithm/vmc_update/mc_energy_grad_evaluator.h"

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
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver,
         template<typename, typename> class ContractorT = BMPSContractor>
class VMCPEPSOptimizer : public qlten::Executor {
 public:
  using Tensor = QLTensor<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using OptimizerT = Optimizer<TenElemT, QNT>;
  using WaveFunctionComponentT = TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, ContractorT>;

  /**
   * @brief Constructor with explicit TPS provided by user.
   * 
   * User provides all data explicitly - no hidden file loading.
   * This constructor gives users complete control over the input data.
   * 
   * @param params Unified optimizer parameters (Optimizer + MC + PEPS)
   * @param sitpst_init Split-index TPS provided by user.
   *        MPI semantics: this constructor does NOT broadcast the state.
   *        All MPI ranks must receive a complete and valid `SITPST` before
   *        construction, because warm-up and internal initialization happen
   *        immediately on each rank. The state will be broadcast later inside
   *        the energy/gradient evaluator per evaluation, and the final
   *        optimized state is broadcast at the end of `Execute()`.
   * @param comm MPI communicator
   * @param solver Energy solver for optimization
   */
  VMCPEPSOptimizer(const VMCPEPSOptimizerParams &params,
                           const SITPST &sitpst_init,
                           const MPI_Comm &comm,
                           const EnergySolver &solver,
                           MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater());

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
   *   auto executor = VMCPEPSOptimizer::CreateByLoadingTPS(params, tps_path, comm, solver);
   */
  static std::unique_ptr<VMCPEPSOptimizer>
  CreateByLoadingTPS(const VMCPEPSOptimizerParams& params,
                     const std::string& tps_path,
                     const MPI_Comm& comm,
                     const EnergySolver& solver,
                     MonteCarloSweepUpdater mc_updater = MonteCarloSweepUpdater());

  // Main execution method
  void Execute(void) override;

  // Data access methods - matching VMCPEPSExecutor interface
  const SITPST &GetState() const noexcept { return monte_carlo_engine_.State(); }
  const SITPST &GetOptimizedState() const { return monte_carlo_engine_.State(); }
  const SITPST &GetBestState() const { return tps_lowest_; }
  double GetMinEnergy() const noexcept { return en_min_; }
  double GetCurrentEnergy() const noexcept {
    return energy_trajectory_.empty() ? std::numeric_limits<double>::max() :
           std::real(energy_trajectory_.back());
  }
  const std::vector<TenElemT> &GetEnergyTrajectory() const { return energy_trajectory_; }
  const std::vector<double> &GetEnergyErrorTrajectory() const { return energy_error_traj_; }
  const std::vector<double> &GetGradientNorms() const { return grad_norm_; }
  const SpikeStatistics &GetSpikeStatistics() const { return spike_stats_; }

  const VMCPEPSOptimizerParams &GetParams() const noexcept { return params_; }

  // Data dumping methods
  /**
   * @brief Dump TPS and energy trajectories.
   *
   * Historical note: older versions also dumped per-rank energy samples as
   * "energy/energy_sample<rank>" for debugging. During the ongoing refactor,
   * we found this feature is rarely needed in practice and it is temporarily
   * disabled. If required in the future, it can be reintroduced behind a
   * debug switch in the energy evaluator.
   */
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
  // Monte Carlo sampling helpers have been migrated into MCEnergyGradEvaluator

  // Data dumping helpers
  /**
   * @deprecated Legacy binary dump of energy trajectory.
   *             Will be removed in a future release. Prefer CSV output at
   *             `energy/energy_trajectory.csv`.
   */
  void DumpVecData_(const std::string &path, const std::vector<TenElemT> &data);
  /**
   * @deprecated Legacy binary dump of double vectors (e.g., energy errors).
   *             Will be removed in a future release. Prefer CSV output at
   *             `energy/energy_trajectory.csv`.
   */
  void DumpVecDataDouble_(const std::string &path, const std::vector<double> &data);

 private:
  MonteCarloEngine<TenElemT, QNT, MonteCarloSweepUpdater, ContractorT> monte_carlo_engine_;
  VMCPEPSOptimizerParams params_;  // New parameter structure
  EnergySolver energy_solver_;

  // Optimizer instance
  OptimizerT optimizer_;

  // Optimization callback
  typename OptimizerT::OptimizationCallback optimization_callback_;

  // Custom energy evaluator (optional)
  std::function<std::tuple<TenElemT, SITPST, double>(const SITPST &)> custom_energy_evaluator_;

  // Data storage
  std::vector<TenElemT> energy_trajectory_;
  std::vector<double> energy_error_traj_;
  std::vector<double> grad_norm_;

  // Gradient calculation storage
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

  // Spike stats cached from OptimizationResult (optimizer clears its copy in ClearUp)
  SpikeStatistics spike_stats_;

  // Current energy error for optimizer access
  double current_energy_error_;

  // Helper methods
  void ReserveSamplesDataSpace_(void);
  void PrintExecutorInfo_(void);
  void ValidateState_(const SITPST &state);
  
  
  // Note: Wavefunction component refresh is now handled by MonteCarloEngine::RefreshWavefunctionComponent.

  // Default energy evaluator
  std::tuple<TenElemT, SITPST, double> DefaultEnergyEvaluator_(const SITPST &state);

  // Persistent Monte-Carlo energy+gradient evaluator for buffer reuse
  std::unique_ptr<MCEnergyGradEvaluator<TenElemT, QNT, MonteCarloSweepUpdater, EnergySolver, ContractorT>>
      energy_grad_evaluator_;
};

} // namespace qlpeps

#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_impl.h"

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_VMC_PEPS_OPTIMIZER_H 
