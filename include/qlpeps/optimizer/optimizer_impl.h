// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Implementation for optimizer.
*
* CRITICAL MPI ARCHITECTURE: This implementation follows strict responsibility separation
*
 * 【CORE PRINCIPLE】: "Clear MPI responsibility: Optimizer handles algorithm MPI, Energy evaluator owns STATE distribution."
*
* This design eliminates triple redundant broadcasts (3→1 per iteration) and follows 
* Linus Torvalds' "good taste" philosophy: "Good taste is about eliminating special 
* cases and making the normal case work correctly."
*
 * 【RESPONSIBILITY BOUNDARIES】:
 * - Optimizer:        Algorithm computation (SR uses MPI for CG solving), NO state broadcasts
 * - Energy Evaluator: State distribution owner, broadcasts states for Monte Carlo sampling  
 * - VMC Executor:     Final synchronization guarantee at optimization completion
*
 * 【MPI FLOW PER ITERATION】:
 * 1. Optimizer updates state (master rank only)        ← NO state broadcast
 * 2. SR algorithm: Distributed CG solving              ← Optimizer internal MPI
 * 3. Energy evaluator receives state (master only)     ← NO state broadcast yet
 * 4. Energy evaluator broadcasts for MC sampling       ← SINGLE state broadcast
 * 5. Distributed Monte Carlo execution                 ← All ranks
 * 6. Energy evaluator gathers gradients to master      ← Standard gather
 * 7. Loop continues...
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H

#include <iomanip>
#include <algorithm>
#include <complex>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/utility/helpers.h"
#include "qlpeps/utility/filesystem_utils.h"
#include "qlten/utility/timer.h"

namespace qlpeps {
using qlten::Timer;

template<typename TenElemT, typename QNT>
Optimizer<TenElemT, QNT>::Optimizer(const OptimizerParams &params,
                                    const MPI_Comm &comm,
                                    int rank,
                                    int mpi_size)
    : params_(params), comm_(comm), rank_(rank), mpi_size_(mpi_size),
      adagrad_initialized_(false),
      sgd_momentum_initialized_(false),
      adam_timestep_(0),
      adam_initialized_(false),
      ema_energy_(params.spike_recovery_params.ema_window),
      ema_error_(params.spike_recovery_params.ema_window),
      ema_grad_norm_(params.spike_recovery_params.ema_window),
      ema_ngrad_norm_(params.spike_recovery_params.ema_window),
      prev_energy_(0.0),
      has_prev_state_(false) {
}

template<typename TenElemT, typename QNT>
double Optimizer<TenElemT, QNT>::GetCurrentLearningRate(size_t iteration, double current_energy) const {
  double learning_rate = params_.base_params.learning_rate;  // Default/base rate
  
  if (params_.base_params.lr_scheduler) {
    learning_rate = params_.base_params.lr_scheduler->GetLearningRate(iteration, current_energy);
  }
  
  return learning_rate;
}

/**
 * @brief Iterative optimization with MPI support
 * 
 * MPI Behavior Documentation:
 * - INPUT: initial_state valid ONLY on master rank (optimizer manages distribution)
 * - energy_evaluator: Takes state from master rank, returns (energy, gradient, energy_error) where:
 *   * energy: Valid on ALL ranks (broadcast by energy_evaluator)
 *   * gradient: Valid ONLY on master rank (gathered by energy_evaluator)  
 *   * energy_error: Valid ONLY on master rank
 * - All algorithm updates are performed only on master rank
 * - Energy evaluator internally broadcasts state for Monte Carlo sampling
 * - Optimization statistics (energy_trajectory, etc.) are tracked only on master rank
 * 
 * @param initial_state Initial TPS state (valid ONLY on master rank)
 * @param energy_evaluator Function returning (energy, gradient, energy_error)
 * @param callback Optional callback for monitoring progress
 * @param Ostar_samples O^*(S) samples for stochastic reconfiguration (if used)
 * @param Ostar_mean Average O^* for stochastic reconfiguration (if used)
 * @return Optimization result with final state valid on all ranks
 */
template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::OptimizationResult
Optimizer<TenElemT, QNT>::IterativeOptimize(
    const WaveFunctionT &initial_state,
    std::function<std::tuple<TenElemT, WaveFunctionT, double>(const WaveFunctionT &)> energy_evaluator,
    const OptimizationCallback &callback,
    const std::vector<WaveFunctionT> *Ostar_samples,
    const WaveFunctionT *Ostar_mean) {

  OptimizationResult result;
  result.optimized_state = initial_state;
  result.converged = false;
  result.total_iterations = 0;

  WaveFunctionT current_state = initial_state;
  WaveFunctionT best_state = initial_state;
  double best_energy = std::numeric_limits<double>::max();

  // Initialize for stochastic reconfiguration if needed
  WaveFunctionT sr_init_guess;
  if (params_.IsAlgorithm<StochasticReconfigurationParams>()) {
    sr_init_guess = WaveFunctionT(initial_state.rows(), initial_state.cols(), initial_state.PhysicalDim());
  }

  // Advanced stopping criteria tracking
  double previous_energy = std::numeric_limits<double>::max();
  size_t iterations_without_improvement = 0;  // plateau counter

  // Spike recovery config aliases
  const auto &spike_cfg = params_.spike_recovery_params;
  const auto &ckpt_cfg = params_.checkpoint_params;
  const bool is_sr = params_.IsAlgorithm<StochasticReconfigurationParams>();
  const bool is_sgd = params_.IsAlgorithm<SGDParams>();
  const bool is_lbfgs = params_.IsAlgorithm<LBFGSParams>();
  const auto &initial_selector_cfg = params_.base_params.initial_step_selector;
  const bool initial_selector_enabled = initial_selector_cfg.enabled;
  const auto &periodic_selector_cfg = params_.base_params.periodic_step_selector;
  const bool periodic_selector_enabled = periodic_selector_cfg.enabled;
  const bool any_selector_enabled = initial_selector_enabled || periodic_selector_enabled;
  constexpr double kSelectorLateSigma = 1.0;

  if (any_selector_enabled) {
    if (!(is_sgd || is_sr)) {
      throw std::invalid_argument(
          "Step selectors only support SGD and SR in IterativeOptimize");
    }
    if (params_.base_params.lr_scheduler) {
      throw std::invalid_argument(
          "Step selectors cannot be used together with lr_scheduler in v1");
    }
  }

  if (initial_selector_enabled && initial_selector_cfg.max_line_search_steps == 0) {
    throw std::invalid_argument("Initial step selector max_line_search_steps must be > 0");
  }
  if (periodic_selector_enabled) {
    if (periodic_selector_cfg.every_n_steps == 0) {
      throw std::invalid_argument("Auto step selector every_n_steps must be > 0");
    }
    if (periodic_selector_cfg.phase_switch_ratio < 0.0 ||
        periodic_selector_cfg.phase_switch_ratio > 1.0) {
      throw std::invalid_argument(
          "Auto step selector phase_switch_ratio must be within [0, 1]");
    }
  }
  if (any_selector_enabled && params_.base_params.learning_rate <= 0.0) {
    throw std::invalid_argument("Step selectors require a positive base learning_rate");
  }

  double selector_base_lr = params_.base_params.learning_rate;

  if (is_lbfgs && rank_ == qlten::hp_numeric::kMPIMasterRank) {
    ResetLBFGSState_();
  }

  size_t total_iterations_performed = 0;
  bool done = false;
  for (size_t iter = 0; iter < params_.base_params.max_iterations && !done; ++iter) {
    bool step_accepted = false;
    size_t attempts = 0;
    bool lbfgs_history_updated_this_iter = false;
    std::deque<LBFGSHistoryPair> lbfgs_iter_start_history;
    WaveFunctionT lbfgs_iter_start_anchor_state;
    WaveFunctionT lbfgs_iter_start_anchor_gradient;
    bool lbfgs_iter_start_has_anchor = false;

    if (is_lbfgs && rank_ == qlten::hp_numeric::kMPIMasterRank) {
      lbfgs_iter_start_history = lbfgs_history_;
      lbfgs_iter_start_anchor_state = lbfgs_anchor_state_;
      lbfgs_iter_start_anchor_gradient = lbfgs_anchor_gradient_;
      lbfgs_iter_start_has_anchor = lbfgs_has_anchor_;
    }

    auto restore_lbfgs_iter_start_snapshot = [&]() {
      if (!(is_lbfgs && rank_ == qlten::hp_numeric::kMPIMasterRank)) {
        return;
      }
      lbfgs_history_ = lbfgs_iter_start_history;
      lbfgs_anchor_state_ = lbfgs_iter_start_anchor_state;
      lbfgs_anchor_gradient_ = lbfgs_iter_start_anchor_gradient;
      lbfgs_has_anchor_ = lbfgs_iter_start_has_anchor;
      lbfgs_history_updated_this_iter = false;
    };

    while (!step_accepted) {
      // === A: Evaluate energy and gradient ===
      Timer energy_eval_timer("energy_evaluation");
      auto [current_energy, current_gradient, current_error] = energy_evaluator(current_state);
      double energy_eval_time = energy_eval_timer.Elapsed();

      double grad_norm = 0.0;
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        grad_norm = std::sqrt(current_gradient.NormSquare());
      }

      if (is_lbfgs && !lbfgs_history_updated_this_iter &&
          rank_ == qlten::hp_numeric::kMPIMasterRank) {
        const auto &lbfgs_params = params_.GetAlgorithmParams<LBFGSParams>();
        UpdateLBFGSHistoryFromAnchor_(current_state, current_gradient, lbfgs_params);
        lbfgs_history_updated_this_iter = true;
      }

      // === B: S1/S2 detection (before any trajectory or state update) ===
      // Guard uses iter > 0 (not ema_.IsInitialized()) so ALL ranks enter together for MPI_Bcast.
      if (spike_cfg.enable_auto_recover && iter > 0) {
        SpikeSignal signal = SpikeSignal::kNone;
        SpikeAction action = SpikeAction::kAccept;
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          signal = DetectS1S2_(current_error, grad_norm);
          if (signal != SpikeSignal::kNone) {
            action = DecideAction_(signal, attempts, iter);
          }
        }
        action = BroadcastAction_(action);
        if (action == SpikeAction::kResample || action == SpikeAction::kRollback) {
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            double thresh = (signal == SpikeSignal::kS1_ErrorbarSpike)
                ? spike_cfg.factor_err * ema_error_.Mean()
                : spike_cfg.factor_grad * ema_grad_norm_.Mean();
            double val = (signal == SpikeSignal::kS1_ErrorbarSpike) ? current_error : grad_norm;
            LogSpikeEvent_(iter, attempts, signal, action, val, thresh);
            if (action == SpikeAction::kRollback && has_prev_state_) {
              // MPI invariant: only master rank holds the authoritative current_state.
              // Non-master ranks' current_state is stale but harmless — the energy_evaluator
              // broadcasts the master's state at the start of each evaluation call.
              current_state = prev_accepted_state_;
              has_prev_state_ = false;
              if (is_lbfgs) {
                RestoreLBFGSSnapshot_();
                lbfgs_history_updated_this_iter = false;
              }
            }
          }
          if (action == SpikeAction::kResample) {
            restore_lbfgs_iter_start_snapshot();
            attempts++;
          } else {
            attempts = 0;
          }
          continue;
        }
        if (action == SpikeAction::kAcceptWithWarning && rank_ == qlten::hp_numeric::kMPIMasterRank) {
          double thresh = (signal == SpikeSignal::kS1_ErrorbarSpike)
              ? spike_cfg.factor_err * ema_error_.Mean()
              : spike_cfg.factor_grad * ema_grad_norm_.Mean();
          double val = (signal == SpikeSignal::kS1_ErrorbarSpike) ? current_error : grad_norm;
          LogSpikeEvent_(iter, attempts, signal, action, val, thresh);
        }
      }

      // === C: Compute learning rate and optimization update ===
      double learning_rate = any_selector_enabled
          ? selector_base_lr
          : GetCurrentLearningRate(iter, (iter == 0) ? std::real(current_energy) : previous_energy);
      double effective_learning_rate = learning_rate;

      Timer update_timer("optimization_update");

      // Gradient preprocessing (clipping) for first-order methods
      WaveFunctionT preprocessed_gradient = current_gradient;
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        if (params_.IsFirstOrder()) {
          if (params_.base_params.clip_value && *(params_.base_params.clip_value) > 0.0) {
            preprocessed_gradient.ElementWiseClipTo(*(params_.base_params.clip_value));
          }
          if (params_.base_params.clip_norm && *(params_.base_params.clip_norm) > 0.0) {
            preprocessed_gradient.ClipByGlobalNorm(*(params_.base_params.clip_norm));
          }
        }
      }

      const bool initial_selector_triggered = initial_selector_enabled && (iter == 0);
      const bool periodic_selector_triggered =
          periodic_selector_enabled &&
          (iter % periodic_selector_cfg.every_n_steps == 0) &&
          !initial_selector_triggered;
      const bool selector_triggered = initial_selector_triggered || periodic_selector_triggered;
      bool use_precomputed_sr_direction = false;
      WaveFunctionT sr_precomputed_direction;
      size_t sr_precomputed_iters = 0;
      double sr_precomputed_natural_grad_norm = 0.0;

      if (selector_triggered) {
        int deterministic_error_missing = 0;
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          const bool require_error_bar =
              (initial_selector_triggered && !initial_selector_cfg.enable_in_deterministic) ||
              (periodic_selector_triggered && !periodic_selector_cfg.enable_in_deterministic);
          deterministic_error_missing = (require_error_bar && current_error <= 0.0) ? 1 : 0;
        }
        if (mpi_size_ > 1) {
          HANDLE_MPI_ERROR(::MPI_Bcast(&deterministic_error_missing, 1, MPI_INT,
                                       qlten::hp_numeric::kMPIMasterRank, comm_));
        }
        if (deterministic_error_missing == 1) {
          if (initial_selector_triggered) {
            throw std::invalid_argument(
                "Initial step selector requires positive energy error in MC mode. "
                "Set enable_in_deterministic=true to allow deterministic evaluators.");
          }
          throw std::invalid_argument(
              "Auto step selector requires positive energy error in MC mode. "
              "Set enable_in_deterministic=true to allow deterministic evaluators.");
        }

        std::vector<double> candidate_etas;
        if (initial_selector_triggered) {
          candidate_etas.reserve(initial_selector_cfg.max_line_search_steps);
          for (size_t i = 1; i <= initial_selector_cfg.max_line_search_steps; ++i) {
            candidate_etas.push_back(selector_base_lr * static_cast<double>(i));
          }
        } else {
          const double candidate_eta = selector_base_lr;
          const double candidate_half_eta = selector_base_lr * 0.5;
          if (candidate_half_eta <= 0.0) {
            throw std::invalid_argument(
                "Auto step selector requires positive candidate step sizes");
          }
          candidate_etas.push_back(candidate_eta);
          candidate_etas.push_back(candidate_half_eta);
        }

        auto evaluate_candidate = [&](const WaveFunctionT &trial_state) {
          // v1 keeps the existing evaluator contract (energy, gradient, error).
          // Gradient is intentionally ignored for selector trials; consider an
          // energy-only evaluator path in v2 for expensive MC evaluations.
          // Trial evaluations intentionally bypass spike detection and EMA/trajectory
          // updates. Only the accepted main-path step participates in S1-S4 logic.
          auto [trial_energy, trial_gradient, trial_error] = energy_evaluator(trial_state);
          (void)trial_gradient;
          const double trial_energy_real = std::real(trial_energy);
          int local_failure_flags = 0;
          if (!std::isfinite(trial_energy_real)) {
            local_failure_flags |= 0x1;
          }
          if (rank_ == qlten::hp_numeric::kMPIMasterRank &&
              !std::isfinite(trial_error)) {
            local_failure_flags |= 0x2;
          }

          // Keep failure handling rank-consistent: all ranks observe the same
          // selector failure before any later collective communication.
          int global_failure_flags = local_failure_flags;
          if (mpi_size_ > 1) {
            HANDLE_MPI_ERROR(::MPI_Allreduce(&local_failure_flags, &global_failure_flags, 1,
                                             MPI_INT, MPI_BOR, comm_));
          }
          if (global_failure_flags != 0) {
            if ((global_failure_flags & 0x1) != 0 && (global_failure_flags & 0x2) != 0) {
              throw std::runtime_error(
                  "Step selector candidate evaluation produced non-finite energy and energy error");
            }
            if ((global_failure_flags & 0x1) != 0) {
              throw std::runtime_error(
                  "Step selector candidate evaluation produced non-finite energy");
            }
            throw std::runtime_error(
                "Step selector candidate evaluation produced non-finite energy error");
          }
          return std::make_pair(trial_energy_real, trial_error);
        };

        struct CandidateEvalResult {
          double eta;
          double energy;
          double error;
        };
        std::vector<CandidateEvalResult> candidate_results;
        candidate_results.reserve(candidate_etas.size());

        if (is_sgd) {
          const auto &sgd_params = params_.GetAlgorithmParams<SGDParams>();
          for (const double eta : candidate_etas) {
            WaveFunctionT trial_state =
                SGDPreviewUpdate_(current_state, preprocessed_gradient, eta, sgd_params);
            auto [trial_energy, trial_error] = evaluate_candidate(trial_state);
            candidate_results.push_back(CandidateEvalResult{eta, trial_energy, trial_error});
          }
        } else if (is_sr) {
          const auto &sr_params = params_.GetAlgorithmParams<StochasticReconfigurationParams>();
          if (Ostar_samples == nullptr || Ostar_mean == nullptr) {
            throw std::invalid_argument(
                "Auto step selector for SR requires Ostar_samples and Ostar_mean");
          }
          auto [natural_grad, cg_iters] = CalculateNaturalGradient(
              current_gradient, *Ostar_samples, *Ostar_mean, sr_init_guess);
          sr_precomputed_direction = std::move(natural_grad);
          sr_precomputed_iters = cg_iters;
          sr_precomputed_natural_grad_norm = std::sqrt(sr_precomputed_direction.NormSquare());
          use_precomputed_sr_direction = true;

          auto make_sr_trial = [&](double eta) {
            WaveFunctionT trial_state = current_state;
            double applied_eta = eta;
            if (sr_params.normalize_update && sr_precomputed_natural_grad_norm > 0.0) {
              applied_eta /= sr_precomputed_natural_grad_norm;
            }
            if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
              trial_state += (-applied_eta) * sr_precomputed_direction;
            }
            return trial_state;
          };

          for (const double eta : candidate_etas) {
            WaveFunctionT trial_state = make_sr_trial(eta);
            auto [trial_energy, trial_error] = evaluate_candidate(trial_state);
            candidate_results.push_back(CandidateEvalResult{eta, trial_energy, trial_error});
          }
        }

        double selected_eta = candidate_results.front().eta;
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          if (initial_selector_triggered) {
            double best_energy_for_selection = candidate_results.front().energy;
            for (const auto &candidate : candidate_results) {
              if (candidate.energy < best_energy_for_selection) {
                best_energy_for_selection = candidate.energy;
                selected_eta = candidate.eta;
              }
            }
          } else {
            const double energy_eta = candidate_results[0].energy;
            const double error_eta = candidate_results[0].error;
            const double energy_half_eta = candidate_results[1].energy;
            const double error_half_eta = candidate_results[1].error;
            const double phase_boundary = periodic_selector_cfg.phase_switch_ratio *
                static_cast<double>(params_.base_params.max_iterations);
            const bool early_phase = static_cast<double>(iter) < phase_boundary;
            if (early_phase) {
              if (energy_half_eta < energy_eta) {
                selected_eta = candidate_results[1].eta;
              }
            } else {
              const double improvement = energy_eta - energy_half_eta;
              const double threshold = kSelectorLateSigma * std::max(error_eta, error_half_eta);
              if (improvement > threshold) {
                selected_eta = candidate_results[1].eta;
              }
            }
          }
        }
        if (mpi_size_ > 1) {
          HANDLE_MPI_ERROR(::MPI_Bcast(&selected_eta, 1, MPI_DOUBLE,
                                       qlten::hp_numeric::kMPIMasterRank, comm_));
        }
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          if (initial_selector_triggered && selected_eta != selector_base_lr) {
            std::cout << "[INITIAL_STEP] Iter 0: selected step size "
                      << selector_base_lr << " -> " << selected_eta << std::endl;
          }
          if (periodic_selector_triggered && selected_eta < selector_base_lr) {
            std::cout << "[PERIODIC_STEP] Iter " << iter << ": step size halved "
                      << selector_base_lr << " -> " << selected_eta << std::endl;
          }
        }
        if (initial_selector_triggered) {
          selector_base_lr = selected_eta;
        } else {
          selector_base_lr = std::min(selector_base_lr, selected_eta);
        }
        learning_rate = selector_base_lr;
        effective_learning_rate = learning_rate;
      }

      WaveFunctionT updated_state;
      size_t sr_iterations = 0;
      double sr_natural_grad_norm = 0.0;

      // Single algorithm dispatch with correct learning rate
      std::visit([&](const auto& algo_params) {
        using T = std::decay_t<decltype(algo_params)>;

        if constexpr (std::is_same_v<T, SGDParams>) {
          updated_state = SGDUpdate(current_state, preprocessed_gradient, learning_rate, algo_params);
        }
        else if constexpr (std::is_same_v<T, StochasticReconfigurationParams>) {
          if (selector_triggered && use_precomputed_sr_direction) {
            double applied_step = learning_rate;
            if (algo_params.normalize_update && sr_precomputed_natural_grad_norm > 0.0) {
              applied_step /= sr_precomputed_natural_grad_norm;
            }
            updated_state = current_state;
            if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
              updated_state += (-applied_step) * sr_precomputed_direction;
            }
            sr_natural_grad_norm = sr_precomputed_natural_grad_norm;
            sr_iterations = sr_precomputed_iters;
            // Keep alignment with existing SR path: updated_state is meaningful
            // on master rank only at this point; this assignment intentionally
            // follows that same rank-local convention.
            sr_init_guess = updated_state;
          } else {
            if (Ostar_samples == nullptr || Ostar_mean == nullptr) {
              throw std::invalid_argument(
                  "SR requires Ostar_samples and Ostar_mean");
            }
            auto sr_result = StochasticReconfigurationUpdate(
                current_state, current_gradient,
                *Ostar_samples, *Ostar_mean,
                learning_rate, sr_init_guess, algo_params.normalize_update);
            updated_state = std::get<0>(sr_result);
            sr_natural_grad_norm = std::get<1>(sr_result);
            sr_iterations = std::get<2>(sr_result);
            sr_init_guess = std::get<0>(sr_result);
          }
        }
        else if constexpr (std::is_same_v<T, AdaGradParams>) {
          updated_state = AdaGradUpdate(current_state, preprocessed_gradient, learning_rate);
        }
        else if constexpr (std::is_same_v<T, AdamParams>) {
          updated_state = AdamUpdate(current_state, preprocessed_gradient, learning_rate, algo_params);
        }
        else if constexpr (std::is_same_v<T, LBFGSParams>) {
          WaveFunctionT search_direction = ComputeLBFGSSearchDirection_(current_gradient, algo_params);

          double dphi0 = 0.0;
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            dphi0 = RealInnerProduct_(current_gradient, search_direction);
          }
          if (mpi_size_ > 1) {
            HANDLE_MPI_ERROR(::MPI_Bcast(&dphi0, 1, MPI_DOUBLE,
                                         qlten::hp_numeric::kMPIMasterRank, comm_));
          }
          if (dphi0 >= 0.0) {
            if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
              ++lbfgs_descent_reset_count_;
              ResetLBFGSState_();
              search_direction = (-1.0) * current_gradient;
              dphi0 = RealInnerProduct_(current_gradient, search_direction);
            }
            if (mpi_size_ > 1) {
              HANDLE_MPI_ERROR(::MPI_Bcast(&dphi0, 1, MPI_DOUBLE,
                                           qlten::hp_numeric::kMPIMasterRank, comm_));
            }
          }

          if (algo_params.step_mode == LBFGSStepMode::kFixed) {
            auto [next_state, used_step] = ApplyFixedLBFGSStep_(current_state, search_direction, learning_rate);
            updated_state = std::move(next_state);
            effective_learning_rate = used_step;
          } else {
            if (mpi_size_ > 1) {
              qlpeps::MPI_Bcast(search_direction, comm_, qlten::hp_numeric::kMPIMasterRank);
            }
            auto [wolfe_status, next_state, used_step] = StrongWolfeLBFGSStep_(
                current_state, search_direction, energy_evaluator, learning_rate, algo_params,
                std::real(current_energy), dphi0);
            if (wolfe_status == WolfeStepStatus::kAccepted) {
              updated_state = std::move(next_state);
              effective_learning_rate = used_step;
            } else if (algo_params.allow_fallback_to_fixed_step) {
              const double fallback_lr = std::max(algo_params.min_step,
                                                  learning_rate * algo_params.fallback_fixed_step_scale);
              auto [fallback_state, used_step] = ApplyFixedLBFGSStep_(current_state, search_direction, fallback_lr);
              updated_state = std::move(fallback_state);
              effective_learning_rate = used_step;
              if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
                std::cerr << "[LBFGS][WARN] Strong Wolfe failed within max_eval="
                          << algo_params.max_eval
                          << ", fallback to fixed step alpha=" << std::scientific
                          << effective_learning_rate << std::endl;
              }
            } else {
              throw std::runtime_error(
                  "L-BFGS strong-Wolfe line search failed. "
                  "Set allow_fallback_to_fixed_step=true for explicit fixed-step fallback.");
            }
          }
        }
        else {
          throw std::runtime_error("Unsupported algorithm type for iterative optimization");
        }
      }, params_.algorithm_params);

      double update_time = update_timer.Elapsed();

      // === D: S3 detection (SR-only, after CG solve) ===
      // Known limitation: if this were extended to non-SR algorithms (AdaGrad/Adam),
      // a RESAMPLE here would cause the algorithm's accumulator state (e.g., AdaGrad's
      // sum-of-squared-gradients, Adam's moment estimates) to be double-updated, since
      // algorithm dispatch (section C) already ran. Currently safe because S3 is gated
      // on is_sr, and SR's CG solver has no persistent accumulator state.
      // See RFC 2025-09-01 "Known Limitations" section.
      // Guard uses iter > 0 (not ema_.IsInitialized()) so ALL ranks enter together for MPI_Bcast.
      if (is_sr && spike_cfg.enable_auto_recover && iter > 0) {
        SpikeSignal signal = SpikeSignal::kNone;
        SpikeAction action = SpikeAction::kAccept;
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          signal = DetectS3_(sr_natural_grad_norm, sr_iterations);
          if (signal != SpikeSignal::kNone) {
            action = DecideAction_(signal, attempts, iter);
          }
        }
        action = BroadcastAction_(action);
        if (action == SpikeAction::kResample || action == SpikeAction::kRollback) {
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            LogSpikeEvent_(iter, attempts, signal, action, sr_natural_grad_norm,
                           spike_cfg.factor_ngrad * ema_ngrad_norm_.Mean());
            if (action == SpikeAction::kRollback && has_prev_state_) {
              current_state = prev_accepted_state_;
              has_prev_state_ = false;
              if (is_lbfgs) {
                RestoreLBFGSSnapshot_();
                lbfgs_history_updated_this_iter = false;
              }
            }
          }
          if (action == SpikeAction::kResample) {
            restore_lbfgs_iter_start_snapshot();
            attempts++;
          } else {
            attempts = 0;
          }
          continue;
        }
        if (action == SpikeAction::kAcceptWithWarning && rank_ == qlten::hp_numeric::kMPIMasterRank) {
          LogSpikeEvent_(iter, attempts, signal, action, sr_natural_grad_norm,
                         spike_cfg.factor_ngrad * ema_ngrad_norm_.Mean());
        }
      }

      // === D2: S4 detection (rollback, opt-in) ===
      // Guard uses iter > 0 (not ema_.IsInitialized()) so ALL ranks enter together for MPI_Bcast.
      if (spike_cfg.enable_rollback && iter > 0) {
        SpikeSignal signal = SpikeSignal::kNone;
        SpikeAction action = SpikeAction::kAccept;
        if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
          signal = DetectS4_(std::real(current_energy));
          if (signal != SpikeSignal::kNone) {
            action = SpikeAction::kRollback;
          }
        }
        action = BroadcastAction_(action);
        if (action == SpikeAction::kRollback) {
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            LogSpikeEvent_(iter, attempts, signal, action,
                           std::real(current_energy),
                           ema_energy_.Mean() + spike_cfg.sigma_k * ema_energy_.Std());
            if (has_prev_state_) {
              // MPI invariant: only master rank holds the authoritative current_state.
              // Non-master ranks' current_state is stale but harmless — the energy_evaluator
              // broadcasts the master's state at the start of each evaluation call.
              current_state = prev_accepted_state_;
              has_prev_state_ = false;
              if (is_lbfgs) {
                RestoreLBFGSSnapshot_();
                lbfgs_history_updated_this_iter = false;
              }
            }
          }
          attempts = 0;
          continue;
        }
      }

      // === E: ACCEPT — update EMA trackers (only on accepted steps) ===
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        ema_energy_.Update(std::real(current_energy));
        ema_error_.Update(current_error);
        ema_grad_norm_.Update(grad_norm);
        if (is_sr) {
          ema_ngrad_norm_.Update(sr_natural_grad_norm);
        }
        prev_energy_ = std::real(current_energy);
      }

      // === F: Trajectories, best-state, stopping, state update, log, callback ===
      result.energy_trajectory.push_back(current_energy);
      result.learning_rate_trajectory.push_back(effective_learning_rate);
      if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
        result.energy_error_trajectory.push_back(current_error);
        result.gradient_norms.push_back(grad_norm);
      }

      total_iterations_performed = iter + 1;

      // Update best state
      if (std::real(current_energy) < best_energy) {
        best_energy = std::real(current_energy);
        best_state = current_state;
        result.min_energy = best_energy;
        iterations_without_improvement = 0;
        if (callback.on_best_state_found) {
          callback.on_best_state_found(best_state, best_energy);
        }
      } else {
        iterations_without_improvement++;
      }

      // Advanced stopping criteria (skip first iteration)
      if (iter > 0) {
        double current_energy_real = std::real(current_energy);
        bool should_stop = ShouldStop(current_energy_real, previous_energy, grad_norm, iterations_without_improvement);
        if (should_stop) {
          if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
            LogOptimizationStep(iter, current_energy_real, current_error, grad_norm,
                               effective_learning_rate, current_accept_rates_, sr_iterations, sr_natural_grad_norm,
                               energy_eval_time, update_time);
            result.converged = true;
          }
          step_accepted = true;
          done = true;
          break;
        }
        previous_energy = current_energy_real;
      } else {
        previous_energy = std::real(current_energy);
      }

      // Save prev state for S4 rollback BEFORE applying update
      if (spike_cfg.enable_rollback && rank_ == qlten::hp_numeric::kMPIMasterRank) {
        prev_accepted_state_ = current_state;
        has_prev_state_ = true;
        if (is_lbfgs) {
          SaveLBFGSSnapshot_();
        }
      }

      if (is_lbfgs && rank_ == qlten::hp_numeric::kMPIMasterRank) {
        lbfgs_anchor_state_ = current_state;
        lbfgs_anchor_gradient_ = current_gradient;
        lbfgs_has_anchor_ = true;
      }

      // Apply the state update.
      // Note: updated_state is only meaningful on master rank (algorithm dispatch
      // computes on master). Non-master ranks rely on energy_evaluator to receive
      // the updated state via MPI broadcast on the next evaluation call.
      current_state = updated_state;

      // Log progress
      LogOptimizationStep(iter, std::real(current_energy), current_error, grad_norm,
                          effective_learning_rate, current_accept_rates_, sr_iterations, sr_natural_grad_norm,
                          energy_eval_time, update_time);

      if (callback.on_iteration) {
        callback.on_iteration(iter, std::real(current_energy), current_error, grad_norm);
      }

      if (callback.should_stop && callback.should_stop(iter, std::real(current_energy), current_error)) {
        step_accepted = true;
        done = true;
        break;
      }

      // === G: Checkpoint ===
      if (ckpt_cfg.IsEnabled() && iter > 0 && (iter % ckpt_cfg.every_n_steps == 0)) {
        SaveCheckpoint_(current_state, iter, result.energy_trajectory, result.energy_error_trajectory);
      }

      step_accepted = true;
    } // end while (!step_accepted)
  } // end for

  result.optimized_state = best_state;
  result.final_energy = best_energy;
  result.total_iterations = total_iterations_performed;
  result.spike_stats = spike_stats_;

  // Ensure the final state is valid
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // Check that the state has the correct dimensions
    if (result.optimized_state.rows() != initial_state.rows() ||
        result.optimized_state.cols() != initial_state.cols()) {
      throw std::runtime_error("Optimized state has incorrect dimensions");
    }
  }

  // Clean up optimization state
  ClearUp();

  return result;
}

/**
 * @brief Unified SGD implementation with momentum and Nesterov support
 * 
 * ELEGANT UNIFICATION: Single implementation handles all SGD variants:
 * - momentum = 0.0: reduces to vanilla SGD (direct inline update)
 * - momentum > 0.0 + nesterov = false: standard momentum SGD
 * - momentum > 0.0 + nesterov = true: Nesterov accelerated gradient
 * 
 * MPI RESPONSIBILITY: Follows the same "no state broadcast" contract as vanilla SGD
 * - gradient: Valid ONLY on master rank (gathered by energy_evaluator)
 * - velocity_: Maintained ONLY on master rank (algorithm state isolation)
 * - Update: Performed ONLY on master rank 
 * - Broadcast: Energy evaluator's sole responsibility
 * 
 * Algorithm (unified mathematical formulation):
 * v_{t+1} = μ * v_t + g_t
 * θ_{t+1} = (1 - αλ) θ_t - α * (μ * v_{t+1} + g_t)  [if Nesterov, decoupled L2 (λ)]
 * θ_{t+1} = (1 - αλ) θ_t - α * v_{t+1}              [if standard momentum, decoupled L2]
 * 
 * Special case (μ=0): v_{t+1} = g_t → both reduce to θ_{t+1} = θ_t - α * g_t
 * This case uses direct inline update for simplicity.
 */
template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::WaveFunctionT
Optimizer<TenElemT, QNT>::SGDUpdate(const WaveFunctionT &current_state,
                                   const WaveFunctionT &gradient,
                                   double learning_rate,
                                   const SGDParams &params) {
  
  WaveFunctionT updated_state = current_state;
  
  // MPI VERIFICATION: Only master rank processes gradients and algorithm state
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // LAZY INITIALIZATION: Initialize velocity on first use (master rank only)
    if (!sgd_momentum_initialized_) {
      velocity_ = WaveFunctionT(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      // Default construction gives zero tensors - perfect for initial velocity
      sgd_momentum_initialized_ = true;
      
      // DEBUG: Velocity state exists only on master rank
      // Non-master ranks never initialize or access velocity_
    }
    
    // Apply decoupled L2 weight decay (AdamW-style) to parameters
    if (params.weight_decay > 0.0) {
      const double decay_factor = 1.0 - learning_rate * params.weight_decay;
      updated_state *= decay_factor;
    }

    if (params.momentum > 0.0) {
      // Momentum SGD: maintain velocity state
      // v_{t+1} = μ * v_t + g_t
      velocity_ = params.momentum * velocity_ + gradient;
      
      if (params.nesterov) {
        // Nesterov: θ_{t+1} = θ_t - α * (μ * v_{t+1} + g_t)
        WaveFunctionT nesterov_update = params.momentum * velocity_ + gradient;
        updated_state += (-learning_rate) * nesterov_update;
      } else {
        // Standard momentum: θ_{t+1} = θ_t - α * v_{t+1}
        updated_state += (-learning_rate) * velocity_;
      }
    } else {
      // Vanilla SGD: direct inline update (already inside master-only block)
      updated_state += (-learning_rate) * gradient;
    }
  }
  
  // CRITICAL MPI DESIGN: Do NOT broadcast here!
  // 
  // RESPONSIBILITY SEPARATION: Energy evaluator owns all state distribution for Monte Carlo sampling.
  return updated_state;
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::WaveFunctionT
Optimizer<TenElemT, QNT>::SGDPreviewUpdate_(const WaveFunctionT &current_state,
                                            const WaveFunctionT &gradient,
                                            double learning_rate,
                                            const SGDParams &params) const {
  WaveFunctionT updated_state = current_state;
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    if (params.weight_decay > 0.0) {
      const double decay_factor = 1.0 - learning_rate * params.weight_decay;
      updated_state *= decay_factor;
    }

    if (params.momentum > 0.0) {
      // Preview keeps velocity_ immutable by construction. With momentum enabled,
      // this evaluates candidates from the same velocity snapshot.
      WaveFunctionT velocity_for_preview =
          sgd_momentum_initialized_ ? velocity_ : WaveFunctionT(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      velocity_for_preview = params.momentum * velocity_for_preview + gradient;
      if (params.nesterov) {
        WaveFunctionT nesterov_update = params.momentum * velocity_for_preview + gradient;
        updated_state += (-learning_rate) * nesterov_update;
      } else {
        updated_state += (-learning_rate) * velocity_for_preview;
      }
    } else {
      updated_state += (-learning_rate) * gradient;
    }
  }
  return updated_state;
}

template<typename TenElemT, typename QNT>
std::pair<typename Optimizer<TenElemT, QNT>::WaveFunctionT, size_t>
Optimizer<TenElemT, QNT>::CalculateNaturalGradient(
    const WaveFunctionT &gradient,
    const std::vector<WaveFunctionT> &Ostar_samples,
    const WaveFunctionT &Ostar_mean,
    const WaveFunctionT &init_guess) {

  // Get CG parameters from StochasticReconfigurationParams
  const auto& sr_params = params_.GetAlgorithmParams<StochasticReconfigurationParams>();
  const ConjugateGradientParams &cg_params = sr_params.cg_params;

  // Create S-matrix for stochastic reconfiguration
  WaveFunctionT *pOstar_mean = nullptr;
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    pOstar_mean = const_cast<WaveFunctionT *>(&Ostar_mean);
  }

  SRSMatrix s_matrix(const_cast<std::vector<WaveFunctionT> *>(&Ostar_samples), pOstar_mean, mpi_size_);
  s_matrix.diag_shift = cg_params.diag_shift;

  auto cg_result = ConjugateGradientSolver(
      s_matrix, gradient, init_guess,
      cg_params.max_iter, cg_params.relative_tolerance,
      cg_params.residue_restart_step, comm_,
      cg_params.absolute_tolerance);

  // Broadcast convergence status from master to all ranks for coordinated error handling.
  // Slave ranks hold a placeholder (converged=false) that does not reflect the actual solve.
  int converged_flag = cg_result.converged ? 1 : 0;
  ::MPI_Bcast(&converged_flag, 1, MPI_INT, qlten::hp_numeric::kMPIMasterRank, comm_);

  if (!converged_flag) {
    std::string msg = "CG solver did not converge in SR natural gradient computation.";
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      msg += " iterations=" + std::to_string(cg_result.iterations)
           + " residual_norm=" + std::to_string(cg_result.residual_norm);
    }
    throw std::runtime_error(msg);
  }

  return {std::move(cg_result.x), cg_result.iterations};
}

template<typename TenElemT, typename QNT>
std::tuple<typename Optimizer<TenElemT, QNT>::WaveFunctionT, double, size_t>
Optimizer<TenElemT, QNT>::StochasticReconfigurationUpdate(
    const WaveFunctionT &current_state,
    const WaveFunctionT &gradient,
    const std::vector<WaveFunctionT> &Ostar_samples,
    const WaveFunctionT &Ostar_mean,
    double learning_rate,
    const WaveFunctionT &init_guess,
    bool normalize) {

  // Calculate natural gradient using stochastic reconfiguration
  // This involves solving the SR equation which should be done by all cores together
  auto [natural_gradient, cg_iterations] = CalculateNaturalGradient(
      gradient, Ostar_samples, Ostar_mean, init_guess);

  double natural_grad_norm = std::sqrt(natural_gradient.NormSquare());

  if (normalize && natural_grad_norm > 0.0) {
    learning_rate /= natural_grad_norm;
  }

  // Apply the update using the natural gradient
  WaveFunctionT updated_state = current_state;
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    updated_state += (-learning_rate) * natural_gradient;
  }

  return {updated_state, natural_grad_norm, cg_iterations};
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::WaveFunctionT
Optimizer<TenElemT, QNT>::AdaGradUpdate(const WaveFunctionT &current_state,
                                        const WaveFunctionT &gradient,
                                        double learning_rate) {
  // MPI VERIFICATION: Only master rank processes gradients and algorithm state
  WaveFunctionT updated_state = current_state;
  
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // Get AdaGrad parameters from the algorithm params
    const auto& adagrad_params = params_.GetAlgorithmParams<AdaGradParams>();
    
    // LAZY INITIALIZATION: Initialize AdaGrad state on first use (master rank only)
    if (!adagrad_initialized_) {
      accumulated_gradients_ = WaveFunctionT(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      // Initialize with small values to avoid division by zero
      for (size_t row = 0; row < accumulated_gradients_.rows(); ++row) {
        for (size_t col = 0; col < accumulated_gradients_.cols(); ++col) {
          for (size_t i = 0; i < accumulated_gradients_({row, col}).size(); ++i) {
            accumulated_gradients_({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
            QNT div = gradient({row, col})[i].Div();
            accumulated_gradients_({row, col})[i].Fill(div, adagrad_params.initial_accumulator_value);
            // Note here for fermionic tensor, positive and negative are relatively define, 
            // In the AdaGrad, we fix the order of indexes to make it make sense.
          }
        }
      }
      adagrad_initialized_ = true;
      
      // DEBUG: AdaGrad state exists only on master rank
      // Non-master ranks never initialize or access accumulated_gradients_
    }

    // Update accumulated gradients: G_k = G_{k-1} + |gradient|^2
    WaveFunctionT squared_gradient = ElementWiseSquaredNorm(gradient);
    accumulated_gradients_ += squared_gradient;

    // Compute adaptive learning rates: 1/sqrt(G_k) for |G_k| > epsilon
    WaveFunctionT adaptive_rates = ElementWiseInverse(ElementWiseSqrt(accumulated_gradients_), adagrad_params.epsilon);

    // Apply AdaGrad update: θ_{k+1} = θ_k - η * adaptive_rates * gradient
    for (size_t row = 0; row < current_state.rows(); ++row) {
      for (size_t col = 0; col < current_state.cols(); ++col) {
        for (size_t i = 0; i < current_state({row, col}).size(); ++i) {
          // Compute adaptive step: learning_rate * adaptive_rate * gradient
          Tensor adaptive_step = ElementWiseMultiply(adaptive_rates({row, col})[i], gradient({row, col})[i]) * learning_rate;

          // Update state: θ_{k+1} = θ_k - adaptive_step
          updated_state({row, col})[i] += (-adaptive_step);
        }
      }
    }
  }

  // CRITICAL MPI DESIGN: Do NOT broadcast here!
  // Energy evaluator owns all state distribution for Monte Carlo sampling.
  return updated_state;
}

/**
 * @brief Adam optimizer with bias-corrected moment estimates
 * 
 * MPI CONTRACT:
 * - gradient: Valid ONLY on master rank (gathered by energy_evaluator)
 * - first_moment_, second_moment_, adam_timestep_: Master rank ONLY
 * - updated_state: Valid ONLY on master rank
 * - State broadcast: Energy evaluator's sole responsibility (NOT here)
 * 
 * Algorithm (AdamW-style decoupled weight decay):
 *   m_t = β₁ m_{t-1} + (1-β₁) g_t
 *   v_t = β₂ v_{t-1} + (1-β₂) g_t²
 *   m̂_t = m_t / (1 - β₁^t)
 *   v̂_t = v_t / (1 - β₂^t)
 *   θ_{t+1} = (1 - αλ) θ_t - α m̂_t / (√v̂_t + ε)
 */
template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::WaveFunctionT
Optimizer<TenElemT, QNT>::AdamUpdate(const WaveFunctionT &current_state,
                                     const WaveFunctionT &gradient,
                                     double learning_rate,
                                     const AdamParams &params) {
  WaveFunctionT updated_state = current_state;
  
  // MPI VERIFICATION: Only master rank processes gradients and algorithm state
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // LAZY INITIALIZATION on first use (master rank only)
    if (!adam_initialized_) {
      first_moment_ = WaveFunctionT(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      second_moment_ = WaveFunctionT(gradient.rows(), gradient.cols(), gradient.PhysicalDim());
      // Initialize with zeros
      for (size_t row = 0; row < first_moment_.rows(); ++row) {
        for (size_t col = 0; col < first_moment_.cols(); ++col) {
          for (size_t i = 0; i < first_moment_({row, col}).size(); ++i) {
            first_moment_({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
            second_moment_({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
            QNT div = gradient({row, col})[i].Div();
            first_moment_({row, col})[i].Fill(div, 0.0);
            second_moment_({row, col})[i].Fill(div, 0.0);
          }
        }
      }
      adam_timestep_ = 0;
      adam_initialized_ = true;
    }
    
    // Increment timestep
    adam_timestep_++;
    
    // Apply decoupled L2 weight decay (AdamW-style) to parameters
    if (params.weight_decay > 0.0) {
      const double decay_factor = 1.0 - learning_rate * params.weight_decay;
      updated_state *= decay_factor;
    }
    
    // Update biased first moment estimate: m_t = β₁ m_{t-1} + (1-β₁) g_t
    first_moment_ = params.beta1 * first_moment_ + (1.0 - params.beta1) * gradient;
    
    // Update biased second moment estimate: v_t = β₂ v_{t-1} + (1-β₂) |g_t|²
    WaveFunctionT squared_gradient = ElementWiseSquaredNorm(gradient);
    second_moment_ = params.beta2 * second_moment_ + (1.0 - params.beta2) * squared_gradient;
    
    // Compute bias correction factors
    double bias_correction1 = 1.0 - std::pow(params.beta1, adam_timestep_);
    double bias_correction2 = 1.0 - std::pow(params.beta2, adam_timestep_);
    
    // Compute bias-corrected estimates
    WaveFunctionT m_hat = (1.0 / bias_correction1) * first_moment_;
    WaveFunctionT v_hat = (1.0 / bias_correction2) * second_moment_;
    
    // Compute update: -α * m̂_t / (√v̂_t + ε)
    WaveFunctionT sqrt_v_hat = ElementWiseSqrt(v_hat);
    WaveFunctionT denom = ElementWiseInverse(sqrt_v_hat, params.epsilon);  // 1/(√v̂ + ε)
    
    for (size_t row = 0; row < current_state.rows(); ++row) {
      for (size_t col = 0; col < current_state.cols(); ++col) {
        for (size_t i = 0; i < current_state({row, col}).size(); ++i) {
          Tensor update = ElementWiseMultiply(denom({row, col})[i], m_hat({row, col})[i]) * learning_rate;
          updated_state({row, col})[i] += (-update);
        }
      }
    }
  }
  
  // CRITICAL MPI DESIGN: Do NOT broadcast here!
  // Energy evaluator owns all state distribution for Monte Carlo sampling.
  return updated_state;
}

template<typename TenElemT, typename QNT>
double Optimizer<TenElemT, QNT>::RealInnerProduct_(const WaveFunctionT& lhs, const WaveFunctionT& rhs) const {
  return std::real(lhs * rhs);
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::ResetLBFGSState_() {
  lbfgs_history_.clear();
  lbfgs_anchor_state_ = WaveFunctionT();
  lbfgs_anchor_gradient_ = WaveFunctionT();
  lbfgs_has_anchor_ = false;
  lbfgs_skip_curvature_count_ = 0;
  lbfgs_damping_count_ = 0;
  lbfgs_descent_reset_count_ = 0;
  lbfgs_prev_snapshot_ = LBFGSSnapshot{};
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::SaveLBFGSSnapshot_() {
  if (rank_ != qlten::hp_numeric::kMPIMasterRank) {
    return;
  }
  lbfgs_prev_snapshot_.valid = true;
  lbfgs_prev_snapshot_.history = lbfgs_history_;
  lbfgs_prev_snapshot_.anchor_state = lbfgs_anchor_state_;
  lbfgs_prev_snapshot_.anchor_gradient = lbfgs_anchor_gradient_;
  lbfgs_prev_snapshot_.has_anchor = lbfgs_has_anchor_;
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::RestoreLBFGSSnapshot_() {
  if (rank_ != qlten::hp_numeric::kMPIMasterRank) {
    return;
  }
  if (!lbfgs_prev_snapshot_.valid) {
    ResetLBFGSState_();
    return;
  }
  lbfgs_history_ = lbfgs_prev_snapshot_.history;
  lbfgs_anchor_state_ = lbfgs_prev_snapshot_.anchor_state;
  lbfgs_anchor_gradient_ = lbfgs_prev_snapshot_.anchor_gradient;
  lbfgs_has_anchor_ = lbfgs_prev_snapshot_.has_anchor;
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::UpdateLBFGSHistoryFromAnchor_(
    const WaveFunctionT& current_state,
    const WaveFunctionT& current_gradient,
    const LBFGSParams& params) {
  if (rank_ != qlten::hp_numeric::kMPIMasterRank) {
    return;
  }
  if (!lbfgs_has_anchor_) {
    return;
  }
  if (params.history_size == 0) {
    throw std::invalid_argument("LBFGS history_size must be > 0");
  }

  WaveFunctionT s = current_state - lbfgs_anchor_state_;
  WaveFunctionT y = current_gradient - lbfgs_anchor_gradient_;

  double curvature = RealInnerProduct_(s, y);
  if (curvature <= params.min_curvature && params.use_damping) {
    const double yy = RealInnerProduct_(y, y);
    if (yy > std::numeric_limits<double>::epsilon()) {
      const double delta = params.min_curvature - curvature + 1e-15;
      // Heuristic curvature lift: this nudges Re(<s, y>) upward but does not
      // enforce exact equality to min_curvature after one correction.
      y += (delta / yy) * s;
      curvature = RealInnerProduct_(s, y);
      ++lbfgs_damping_count_;
    }
  }

  if (curvature <= params.min_curvature) {
    ++lbfgs_skip_curvature_count_;
    return;
  }

  LBFGSHistoryPair pair;
  pair.s = std::move(s);
  pair.y = std::move(y);
  pair.rho = 1.0 / curvature;
  lbfgs_history_.push_back(std::move(pair));
  while (lbfgs_history_.size() > params.history_size) {
    lbfgs_history_.pop_front();
  }
}

template<typename TenElemT, typename QNT>
typename Optimizer<TenElemT, QNT>::WaveFunctionT
Optimizer<TenElemT, QNT>::ComputeLBFGSSearchDirection_(const WaveFunctionT& gradient,
                                                       const LBFGSParams& params) {
  WaveFunctionT search_direction = (-1.0) * gradient;
  if (rank_ != qlten::hp_numeric::kMPIMasterRank) {
    return search_direction;
  }
  if (params.history_size == 0) {
    throw std::invalid_argument("LBFGS history_size must be > 0");
  }

  if (!lbfgs_history_.empty()) {
    WaveFunctionT q = gradient;
    std::vector<double> alpha(lbfgs_history_.size(), 0.0);
    for (size_t i = lbfgs_history_.size(); i-- > 0;) {
      alpha[i] = lbfgs_history_[i].rho * RealInnerProduct_(lbfgs_history_[i].s, q);
      q += (-alpha[i]) * lbfgs_history_[i].y;
    }

    double gamma = 1.0;
    const auto &last = lbfgs_history_.back();
    const double yy = RealInnerProduct_(last.y, last.y);
    const double sy = RealInnerProduct_(last.s, last.y);
    if (yy > std::numeric_limits<double>::epsilon() && sy > params.min_curvature) {
      gamma = sy / yy;
    }

    WaveFunctionT r = gamma * q;
    for (size_t i = 0; i < lbfgs_history_.size(); ++i) {
      const double beta = lbfgs_history_[i].rho * RealInnerProduct_(lbfgs_history_[i].y, r);
      r += (alpha[i] - beta) * lbfgs_history_[i].s;
    }
    search_direction = (-1.0) * r;
  }

  if (params.max_direction_norm > 0.0) {
    const double norm = std::sqrt(search_direction.NormSquare());
    if (norm > params.max_direction_norm && norm > 0.0) {
      search_direction *= (params.max_direction_norm / norm);
    }
  }

  return search_direction;
}

template<typename TenElemT, typename QNT>
std::pair<typename Optimizer<TenElemT, QNT>::WaveFunctionT, double>
Optimizer<TenElemT, QNT>::ApplyFixedLBFGSStep_(const WaveFunctionT& current_state,
                                                const WaveFunctionT& search_direction,
                                                double learning_rate) {
  const double alpha = std::max(0.0, learning_rate);
  WaveFunctionT updated_state = current_state;
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    updated_state += alpha * search_direction;
  }
  return {std::move(updated_state), alpha};
}

template<typename TenElemT, typename QNT>
std::tuple<typename Optimizer<TenElemT, QNT>::WolfeStepStatus,
           typename Optimizer<TenElemT, QNT>::WaveFunctionT,
           double>
Optimizer<TenElemT, QNT>::StrongWolfeLBFGSStep_(
    const WaveFunctionT& current_state,
    const WaveFunctionT& search_direction,
    const std::function<std::tuple<TenElemT, WaveFunctionT, double>(const WaveFunctionT&)>& energy_evaluator,
    double learning_rate,
    const LBFGSParams& params,
    double phi0,
    double dphi0) {
  if (params.wolfe_c1 <= 0.0 || params.wolfe_c1 >= 1.0 ||
      params.wolfe_c2 <= params.wolfe_c1 || params.wolfe_c2 >= 1.0) {
    throw std::invalid_argument("LBFGS strong-Wolfe parameters must satisfy 0 < c1 < c2 < 1");
  }
  if (params.max_eval == 0) {
    throw std::invalid_argument("LBFGS max_eval must be > 0 for strong-Wolfe mode");
  }
  if (params.tolerance_grad < 0.0) {
    throw std::invalid_argument("LBFGS tolerance_grad must be >= 0 for strong-Wolfe mode");
  }
  if (dphi0 >= 0.0) {
    return {WolfeStepStatus::kFailed, current_state, 0.0};
  }
  const double curvature_threshold = std::max(-params.wolfe_c2 * dphi0, params.tolerance_grad);

  auto bcast_double = [&](double* value) {
    if (mpi_size_ > 1) {
      HANDLE_MPI_ERROR(::MPI_Bcast(value, 1, MPI_DOUBLE, qlten::hp_numeric::kMPIMasterRank, comm_));
    }
  };

  size_t eval_count = 0;
  auto eval_alpha = [&](double alpha, double* phi, double* dphi, WaveFunctionT* trial_state) {
    *trial_state = current_state;
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      *trial_state += alpha * search_direction;
    }
    auto [energy_trial, grad_trial, err_trial] = energy_evaluator(*trial_state);
    (void)err_trial;
    if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
      *phi = std::real(energy_trial);
      *dphi = RealInnerProduct_(grad_trial, search_direction);
    }
    bcast_double(phi);
    bcast_double(dphi);
    ++eval_count;
  };

  auto zoom = [&](double alo,
                  double phi_alo,
                  double ahi,
                  double phi_ahi) -> std::tuple<WolfeStepStatus, WaveFunctionT, double> {
    (void)phi_ahi;
    while (eval_count < params.max_eval) {
      const double alpha = 0.5 * (alo + ahi);
      WaveFunctionT trial_state;
      double phi = 0.0;
      double dphi = 0.0;
      eval_alpha(alpha, &phi, &dphi, &trial_state);

      if ((phi > phi0 + params.wolfe_c1 * alpha * dphi0) || (phi >= phi_alo)) {
        ahi = alpha;
      } else {
        if (std::abs(dphi) <= curvature_threshold) {
          return {WolfeStepStatus::kAccepted, std::move(trial_state), alpha};
        }
        if (dphi * (ahi - alo) >= 0.0) {
          ahi = alo;
        }
        alo = alpha;
        phi_alo = phi;
      }

      if (std::abs(ahi - alo) <= params.tolerance_change) {
        break;
      }
    }
    return {WolfeStepStatus::kFailed, current_state, 0.0};
  };

  double alpha_prev = 0.0;
  double phi_prev = phi0;
  double alpha = std::clamp(std::max(learning_rate, params.min_step), params.min_step, params.max_step);
  if (alpha <= 0.0) {
    alpha = params.min_step;
  }

  for (size_t outer = 0; eval_count < params.max_eval; ++outer) {
    WaveFunctionT trial_state;
    double phi = 0.0;
    double dphi = 0.0;
    eval_alpha(alpha, &phi, &dphi, &trial_state);

    if ((phi > phi0 + params.wolfe_c1 * alpha * dphi0) ||
        (outer > 0 && phi >= phi_prev)) {
      return zoom(alpha_prev, phi_prev, alpha, phi);
    }
    if (std::abs(dphi) <= curvature_threshold) {
      return {WolfeStepStatus::kAccepted, std::move(trial_state), alpha};
    }
    if (dphi >= 0.0) {
      return zoom(alpha, phi, alpha_prev, phi_prev);
    }

    alpha_prev = alpha;
    phi_prev = phi;
    const double next_alpha = std::min(alpha * 2.0, params.max_step);
    if (next_alpha - alpha <= params.tolerance_change) {
      break;
    }
    alpha = next_alpha;
  }
  return {WolfeStepStatus::kFailed, current_state, 0.0};
}

// Element-wise helpers are now defined on SplitIndexTPS; Optimizer no longer owns them.



template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::LogOptimizationStep(size_t iteration,
                                                   double energy,
                                                   double energy_error,
                                                   double gradient_norm,
                                                   double learning_rate,
                                                   const std::vector<double> &accept_rates,
                                                   size_t sr_iterations,
                                                   double sr_natural_grad_norm,
                                                   double energy_eval_time,
                                                   double update_time) {
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    std::cout << "Iter " << std::setw(4) << iteration;
    if (learning_rate > 0.0) {
      std::cout << "LR = " << std::setw(9) << std::scientific << std::setprecision(1) << learning_rate;
    }
    std::cout << "E0 = " << std::setw(14) << std::fixed << std::setprecision(6) << energy;
    if (energy_error > 0.0) {
      std::cout << " ± " << std::setw(10) << std::scientific << std::setprecision(2) << energy_error;
    }
    std::cout << "Grad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << gradient_norm;

    // Print acceptance rates if available
    if (!accept_rates.empty()) {
      std::cout << "Accept rate = [";
      for (double rate : accept_rates) {
        std::cout << std::setw(5) << std::fixed << std::setprecision(2) << rate;
      }
      std::cout << "]";
    }

    // Print SR information if available
    if (params_.IsAlgorithm<StochasticReconfigurationParams>() && sr_iterations > 0) {
      std::cout << "SRSolver Iter = " << std::setw(4) << sr_iterations;
      std::cout << "NGrad norm = " << std::setw(9) << std::scientific << std::setprecision(1) << sr_natural_grad_norm;
    }

    // Print timing information
    double total_time = energy_eval_time + update_time;
    std::cout << " EvalT = " << std::setw(8) << std::fixed << std::setprecision(2) << energy_eval_time << "s";
    std::cout << " UpdateT = " << std::setw(8) << std::fixed << std::setprecision(2) << update_time << "s";
    std::cout << " TotT = " << std::setw(8) << std::fixed << std::setprecision(2) << total_time << "s";

    std::cout << std::endl;
  }
}

template<typename TenElemT, typename QNT>
bool Optimizer<TenElemT, QNT>::ShouldStop(double current_energy, double previous_energy,
                                          double gradient_norm, size_t iterations_without_improvement) {
  // EFFICIENT APPROACH: Only master rank evaluates stopping criteria, then broadcasts decision
  bool should_stop = false;
  
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    // Check gradient norm convergence
    if (gradient_norm < params_.base_params.gradient_tolerance) {
      std::cout << "Stopping: Gradient norm converged (" << std::scientific << std::setprecision(1) << gradient_norm
                << " < " << std::scientific << std::setprecision(1) << params_.base_params.gradient_tolerance
                << ")" << std::endl;
      should_stop = true;
    }

    // Check energy convergence
    if (std::abs(current_energy - previous_energy) < params_.base_params.energy_tolerance) {
      std::cout << "Stopping: Energy converged (|ΔE| = " << std::scientific << std::setprecision(1)
                << std::abs(current_energy - previous_energy)
                << " < " << std::scientific << std::setprecision(1) << params_.base_params.energy_tolerance
                << ")" << std::endl;
      should_stop = true;
    }

    // Check plateau detection
    if (iterations_without_improvement >= params_.base_params.plateau_patience) {
      std::cout << "Stopping: No improvement for " << params_.base_params.plateau_patience
                << " iterations (plateau detected)" << std::endl;
      should_stop = true;
    }
  }

  // Single-rank fast path: avoid calling MPI collectives when mpi_size_ == 1.
  // This keeps Optimizer usable in non-MPI test binaries and simplifies debugging.
  if (mpi_size_ == 1) return should_stop;

  // Broadcast the stopping decision to all ranks
  int stop_flag = should_stop ? 1 : 0;
  HANDLE_MPI_ERROR(::MPI_Bcast(&stop_flag, 1, MPI_INT, qlten::hp_numeric::kMPIMasterRank, comm_));
  
  return stop_flag == 1;
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::ClearUp() {
  // Clear AdaGrad state
  if (adagrad_initialized_) {
    accumulated_gradients_ = WaveFunctionT();
    adagrad_initialized_ = false;
  }

  // Clear SGD momentum state
  if (sgd_momentum_initialized_) {
    velocity_ = WaveFunctionT();
    sgd_momentum_initialized_ = false;
  }

  // Clear Adam state
  if (adam_initialized_) {
    first_moment_ = WaveFunctionT();
    second_moment_ = WaveFunctionT();
    adam_timestep_ = 0;
    adam_initialized_ = false;
  }

  // Clear L-BFGS state
  if (params_.IsAlgorithm<LBFGSParams>() &&
      rank_ == qlten::hp_numeric::kMPIMasterRank) {
    ResetLBFGSState_();
  }

  // Clear spike detection state
  ema_energy_.Reset();
  ema_error_.Reset();
  ema_grad_norm_.Reset();
  ema_ngrad_norm_.Reset();
  prev_energy_ = 0.0;

  // Write CSV trigger log if configured
  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    const auto &csv_path = params_.spike_recovery_params.log_trigger_csv_path;
    if (!csv_path.empty() && !spike_stats_.event_log.empty()) {
      std::ofstream csv(csv_path);
      if (csv) {
        csv << "step,attempt,signal,action,value,threshold\n";
        for (const auto &ev : spike_stats_.event_log) {
          csv << ev.step << "," << ev.attempt << ","
              << SignalName(ev.signal) << "," << ActionName(ev.action) << ","
              << std::scientific << std::setprecision(6) << ev.value << ","
              << ev.threshold << "\n";
        }
      }
    }
  }

  // Clear spike statistics (after CSV has been written above)
  spike_stats_ = SpikeStatistics{};
  current_accept_rates_.clear();

  // Clear rollback state
  prev_accepted_state_ = WaveFunctionT();
  has_prev_state_ = false;
}

// =============================================================================
// Spike detection methods
// =============================================================================

template<typename TenElemT, typename QNT>
SpikeSignal Optimizer<TenElemT, QNT>::DetectS1S2_(double error, double grad_norm) const {
  const auto &cfg = params_.spike_recovery_params;
  // S1: error bar spike
  if (ema_error_.IsInitialized() && ema_error_.Mean() > 0.0) {
    if (error > cfg.factor_err * ema_error_.Mean()) {
      return SpikeSignal::kS1_ErrorbarSpike;
    }
  }
  // S2: gradient norm spike
  if (ema_grad_norm_.IsInitialized() && ema_grad_norm_.Mean() > 0.0) {
    if (grad_norm > cfg.factor_grad * ema_grad_norm_.Mean()) {
      return SpikeSignal::kS2_GradientNormSpike;
    }
  }
  return SpikeSignal::kNone;
}

template<typename TenElemT, typename QNT>
SpikeSignal Optimizer<TenElemT, QNT>::DetectS3_(double ngrad_norm, size_t sr_iters) const {
  const auto &cfg = params_.spike_recovery_params;
  // Check natural gradient norm anomaly against EMA
  if (ema_ngrad_norm_.IsInitialized() && ema_ngrad_norm_.Mean() > 0.0) {
    if (ngrad_norm > cfg.factor_ngrad * ema_ngrad_norm_.Mean()) {
      return SpikeSignal::kS3_NaturalGradientAnomaly;
    }
  }
  // Suspiciously few CG iterations — independent trigger (no ngrad threshold).
  // When CG converges in very few iterations the solver may have "succeeded"
  // trivially on a near-singular system, so this alone signals trouble.
  if (sr_iters <= cfg.sr_min_iters_suspicious) {
    return SpikeSignal::kS3_NaturalGradientAnomaly;
  }
  return SpikeSignal::kNone;
}

template<typename TenElemT, typename QNT>
SpikeSignal Optimizer<TenElemT, QNT>::DetectS4_(double energy) const {
  const auto &cfg = params_.spike_recovery_params;
  if (!ema_energy_.IsInitialized()) return SpikeSignal::kNone;
  // Only trigger on upward spikes: energy increased AND exceeds EMA + k*sigma
  double delta = energy - ema_energy_.Mean();
  if (delta > 0.0 && ema_energy_.Std() > 0.0) {
    if (delta > cfg.sigma_k * ema_energy_.Std()) {
      return SpikeSignal::kS4_EMAEnergySpikeUpward;
    }
  }
  return SpikeSignal::kNone;
}

template<typename TenElemT, typename QNT>
SpikeAction Optimizer<TenElemT, QNT>::DecideAction_(SpikeSignal signal, size_t attempts, size_t step) const {
  if (signal == SpikeSignal::kNone) return SpikeAction::kAccept;

  const auto &cfg = params_.spike_recovery_params;

  // For S4, always rollback (caller decides to invoke this)
  if (signal == SpikeSignal::kS4_EMAEnergySpikeUpward) {
    return SpikeAction::kRollback;
  }

  // S1/S2/S3: try resample if under retry limit
  if (attempts < cfg.redo_mc_max_retries) {
    return SpikeAction::kResample;
  }

  // Retries exhausted: try rollback if enabled and we have a prev state
  if (cfg.enable_rollback && has_prev_state_ && step > 0) {
    return SpikeAction::kRollback;
  }

  // Last resort: accept with warning
  return SpikeAction::kAcceptWithWarning;
}

template<typename TenElemT, typename QNT>
SpikeAction Optimizer<TenElemT, QNT>::BroadcastAction_(SpikeAction action) {
  if (mpi_size_ == 1) return action;

  int action_int = static_cast<int>(action);
  HANDLE_MPI_ERROR(::MPI_Bcast(&action_int, 1, MPI_INT, qlten::hp_numeric::kMPIMasterRank, comm_));
  return static_cast<SpikeAction>(action_int);
}

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::LogSpikeEvent_(
    size_t step, size_t attempt, SpikeSignal signal,
    SpikeAction action, double value, double threshold) {
  // Console log (master rank only — caller guarantees this)
  std::cout << "[SPIKE] Step " << step << " attempt " << attempt
            << " | " << SignalName(signal)
            << " | value=" << std::scientific << std::setprecision(3) << value
            << " threshold=" << std::scientific << std::setprecision(3) << threshold
            << " | -> " << ActionName(action) << std::endl;

  // Update statistics
  if (action == SpikeAction::kResample) spike_stats_.total_resamples++;
  else if (action == SpikeAction::kRollback) spike_stats_.total_rollbacks++;
  else if (action == SpikeAction::kAcceptWithWarning) spike_stats_.total_forced_accepts++;

  // Record event for optional CSV output
  spike_stats_.event_log.push_back({step, attempt, signal, action, value, threshold});
}

// =============================================================================
// Checkpoint
// =============================================================================

template<typename TenElemT, typename QNT>
void Optimizer<TenElemT, QNT>::SaveCheckpoint_(
    const WaveFunctionT &state, size_t step,
    const std::vector<TenElemT> &energy_traj,
    const std::vector<double> &error_traj) {
  const auto &cfg = params_.checkpoint_params;
  if (!cfg.IsEnabled()) return;

  std::string dir = cfg.base_path + "/step_" + std::to_string(step);

  if (rank_ == qlten::hp_numeric::kMPIMasterRank) {
    EnsureDirectoryExists(dir);
    // Save TPS (Dump is non-const due to optional release_mem; const_cast is safe here)
    const_cast<WaveFunctionT &>(state).Dump(dir);

    // Write metadata
    std::string meta_path = dir + "/checkpoint_meta.txt";
    std::ofstream meta(meta_path);
    if (meta) {
      meta << "step " << step << "\n";
      if (!energy_traj.empty()) {
        meta << "energy " << std::setprecision(17) << std::real(energy_traj.back()) << "\n";
      }
      if (!error_traj.empty()) {
        meta << "error " << std::setprecision(17) << error_traj.back() << "\n";
      }
    }

    // Snapshot trajectory CSV
    std::string csv_path = dir + "/trajectory_snapshot.csv";
    std::ofstream csv(csv_path);
    if (csv) {
      csv << "iteration,energy,energy_error\n";
      for (size_t i = 0; i < energy_traj.size(); ++i) {
        csv << i << "," << std::setprecision(17) << std::real(energy_traj[i]) << ","
            << (i < error_traj.size() ? error_traj[i] : 0.0) << "\n";
      }
    }

    std::cout << "[CHECKPOINT] Saved step " << step << " to " << dir << std::endl;
  }

  // All ranks wait for master I/O to complete
  if (mpi_size_ > 1) {
    HANDLE_MPI_ERROR(::MPI_Barrier(comm_));
  }
}

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_OPTIMIZER_IMPL_H 
