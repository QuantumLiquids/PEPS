// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-27
*
* Description: QuantumLiquids/PEPS project. Independent Monte-Carlo based
*              energy and gradient evaluator (no optimizer coupling).
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_MC_ENERGY_GRAD_EVALUATOR_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_MC_ENERGY_GRAD_EVALUATOR_H

#include <optional>
#include <vector>
#include <iostream>
#include <cmath>
#include "mpi.h"

#include "qlten/qlten.h"
#include "qlten/framework/hp_numeric/mpi_fun.h"

#include "qlpeps/utility/helpers.h"
#include "qlpeps/vmc_basic/statistics_tensor.h"
#include "qlpeps/vmc_basic/monte_carlo_tools/statistics.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_engine.h"
#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"

namespace qlpeps {

/**
 * @brief Independent Monte-Carlo energy and gradient evaluator.
 *
 * Responsibilities:
 * - Given a split-index TPS state and a MonteCarloEngine, perform MC sampling
 *   to estimate energy (mean and standard error) and gradient tensors.
 * - Optionally collect SR buffers (O* samples and mean O*).
 * - Encapsulate MPI broadcast/reduction semantics in a single, testable unit.
 *
 * Non-responsibilities:
 * - Does not own model/Hamiltonian logic (delegates to EnergySolver).
 * - Does not manage optimizer logic or parameter updates.
 *
 * Numeric stability and normalization policy:
 * - Evaluate() DOES NOT normalize the wavefunction/state. Plain gradients are not
 *   scale-invariant, therefore implicit normalization would rescale gradients and
 *   can interfere with optimizers (e.g., LBFGS, line-search methods).
 * - Users should provide a well-scaled wavefunction. In our VMC flow,
 *   MonteCarloEngine normalizes during initialization and after WarmUp() so that
 *   the amplitude stays O(1). If you use this evaluator independently, adopt a
 *   similar safeguard: perform amplitude sanity checks and, if necessary, apply
 *   a global rescaling outside Evaluate().
 * - As a light-weight guard, Evaluate() emits a warning when |amplitude| lies
 *   outside [sqrt(min_double), sqrt(max_double)], following TPS policy.
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
class MCEnergyGradEvaluator {
 public:
  using Tensor = qlten::QLTensor<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using WaveFunctionComponentT = TPSWaveFunctionComponent<TenElemT, QNT>;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;

  struct Result {
    TenElemT energy;                 // Broadcast to all ranks
    double energy_error;             // Valid on master rank; 0 on others
    SITPST gradient;                 // Valid on master rank
    double gradient_norm;            // Valid on master rank
    std::vector<double> accept_rates_avg; // Per-update-type average acceptance across samples
    std::optional<SITPST> Ostar_mean;     // Present when collect_sr_buffers==true
    std::vector<SITPST> Ostar_samples;    // Present when collect_sr_buffers==true
  };

  /**
   * @param engine  Reference to MC engine (state holder and sampler)
   * @param solver  Reference to model energy solver
   * @param comm    MPI communicator
   * @param collect_sr_buffers  Whether to collect SR buffers (O* samples and mean O*)
   */
  MCEnergyGradEvaluator(MonteCarloEngine<TenElemT, QNT, MonteCarloSweepUpdater> &engine,
                        EnergySolver &solver,
                        const MPI_Comm &comm,
                        bool collect_sr_buffers = false)
      : engine_(engine), solver_(solver), comm_(comm), collect_sr_buffers_(collect_sr_buffers) {}

  /**
   * @brief Reserve internal buffers for repeated evaluations (best-effort).
   *
   * Note: Physical dimensions can vary per site, so only coarse-grained
   * reservations (e.g. sample counts) are effective here. Fine-grained tensor
   * buffers are still shaped during Evaluate() when exact sizes are known.
   *
   * Historical: This function supersedes the pre-allocation that used to live
   * in VMCPEPSOptimizer (for O* accumulators and per-sample buffers).
   */
  void ReserveBuffers(size_t /*ly*/, size_t /*lx*/, size_t sample_count) {
    reserved_samples_ = sample_count;
  }

  /**
   * @brief Evaluate energy and gradient for a given state using Monte Carlo sampling.
   *
   * Semantics:
   * - Master rank assigns state (if different), broadcasts to all ranks.
   * - Engine rebuilds the wavefunction component; no normalization is performed here.
   * - Performs MC sampling for engine.MCParams().num_samples samples.
   * - Reduces energy and gradient across MPI; returns Result.
   *
   * Per-sample computation (migrated from VMCPEPSOptimizer):
   * - Compute local energy E_loc(S) and holes via solver.
   * - Define O^*(S) = ∂ ln Ψ^*(S) / ∂θ^*; for bosonic case use
   *   inverse_amplitude * holes; for fermionic case use CalGTenForFermionicTensors.
   * - Accumulate Σ O^* and Σ E_loc^* O^* over samples.
   *
   * Gradient (complex parameters):
   *   ∂E/∂θ^* = ⟨E_loc^* O^*⟩ − E^* ⟨O^*⟩,
   * where complex conjugate is applied to E_loc and the mean energy E.
   * We implement this by computing Ostar_mean = Σ O^* / N and
   * grad = Σ E_loc^* O^* / N + (-E)^* · Ostar_mean, and then performing MPI means
   * per tensor component. Fermionic parity operations are applied to grad at the end.
   *
   * SR buffers:
   * - When collect_sr_buffers==true, per-sample O^*(S) tensors and their mean are
   *   returned in Result (move semantics), enabling S-matrix construction with
   *   zero extra copies.
   *
   * Energy samples:
   * - Per-rank raw energy samples are not returned by default. Historically they
   *   were dumped for debugging from VMCPEPSOptimizer; this has been disabled
   *   during refactor and can be reintroduced here behind a debug switch if needed.
   *
   * Error estimation:
   * - The reported energy error is a coarse reference estimated via sqrt(N)-binning
   *   (MeanAndBinnedErrorSqrtNUniformBin). Optimizer logic does not depend on this
   *   error value. For more accurate error bars, use MCPEPSMeasurer (to be implemented).
   */
  Result Evaluate(const SITPST &state) {
    // Assign state on master only to avoid spurious self-assignment
    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
      if (&state != &engine_.State()) {
        engine_.AssignState(state);
      }
    }

    // Broadcast the updated state to all ranks
    MPI_Bcast(engine_.State(), engine_.Comm());

    // Refresh wavefunction component. Do NOT normalize here to avoid rescaling gradients.
    engine_.RefreshWavefunctionComponent();
    // Sanity-check wavefunction amplitude magnitude following TPS component policy
    if (!engine_.WavefuncComp().IsAmplitudeSquareLegal()) {
      std::cout << "Warning (rank " << engine_.Rank() << "): wavefunction amplitude magnitude "
                << std::scientific << std::abs(engine_.WavefuncComp().GetAmplitude())
                << " is outside sqrt(numeric_limits) bounds." << std::endl;
    }

    // Local accumulators
    const size_t ly = engine_.Ly();
    const size_t lx = engine_.Lx();
    const size_t sample_num = engine_.MCParams().num_samples;

    std::vector<TenElemT> energy_samples;
    energy_samples.reserve(std::max(sample_num, reserved_samples_));

    SITPST Ostar_sum(ly, lx);
    SITPST ELocConj_Ostar_sum(ly, lx);
    SITPST grad(ly, lx);
    std::vector<SITPST> Ostar_samples; // SR optional
    if (collect_sr_buffers_) {
      Ostar_samples.reserve(std::max(sample_num, reserved_samples_));
    }

    // Pre-allocate containers with correct shapes
    for (size_t row = 0; row < ly; row++) {
      for (size_t col = 0; col < lx; col++) {
        const size_t dim = engine_.State()({row, col}).size();
        Ostar_sum({row, col}) = std::vector<Tensor>(dim);
        for (size_t compt = 0; compt < dim; compt++) {
          Ostar_sum({row, col})[compt] = Tensor(engine_.State()({row, col})[compt].GetIndexes());
        }
        ELocConj_Ostar_sum({row, col}) = Ostar_sum({row, col});
        grad({row, col}) = std::vector<Tensor>(dim);
      }
    }

    // Monte Carlo sampling loop
    std::vector<double> accept_rates_accum;
    // Reuse holes buffer across samples to avoid per-iteration allocation
    TensorNetwork2D<TenElemT, QNT> holes(ly, lx);
    for (size_t sweep = 0; sweep < sample_num; sweep++) {
      std::vector<double> accept_rates = engine_.StepSweep();
      if (sweep == 0) {
        accept_rates_accum = accept_rates;
      } else {
        for (size_t i = 0; i < accept_rates_accum.size(); i++) {
          accept_rates_accum[i] += accept_rates[i];
        }
      }

      // Compute local energy and holes (holes is overwritten each sample)
      TenElemT local_energy = solver_.template CalEnergyAndHoles<TenElemT, QNT, true>(&engine_.State(),
                                                                                       &engine_.WavefuncComp(),
                                                                                       holes);
      TenElemT local_energy_conjugate = ComplexConjugate(local_energy);
      TenElemT inverse_amplitude = ComplexConjugate(1.0 / engine_.WavefuncComp().GetAmplitude());
      energy_samples.push_back(local_energy);

      // Accumulate O* and E_loc^* O*
      // O* = d ln(psi*) / d theta*
      std::optional<SITPST> Ostar_sample_opt;
      if (collect_sr_buffers_) {
        Ostar_sample_opt.emplace(ly, lx, engine_.State().PhysicalDim());
      }
      for (size_t row = 0; row < ly; row++) {
        for (size_t col = 0; col < lx; col++) {
          const size_t basis_index = engine_.WavefuncComp().GetConfiguration({row, col});

          Tensor Ostar_tensor;
          if constexpr (Tensor::IsFermionic()) {
            Ostar_tensor = CalGTenForFermionicTensors<TenElemT, QNT>(holes({row, col}), engine_.WavefuncComp().tn({row, col}));
          } else {
            Ostar_tensor = inverse_amplitude * holes({row, col});
          }

          Ostar_sum({row, col})[basis_index] += Ostar_tensor;
          ELocConj_Ostar_sum({row, col})[basis_index] += local_energy_conjugate * Ostar_tensor;

          if (collect_sr_buffers_) {
            (*Ostar_sample_opt)({row, col})[basis_index] = Ostar_tensor;
          }
        }
      }
      if (collect_sr_buffers_) {
        Ostar_samples.emplace_back(std::move(*Ostar_sample_opt));
      }
    }

    // Average acceptance rates
    std::vector<double> accept_rates_avg = accept_rates_accum;
    for (double &rate : accept_rates_avg) { rate /= double(sample_num); }

    // Downstream: check acceptance anomalies locally (with global max via MPI)
    AcceptanceRateCheck_(accept_rates_avg);

    // Energy reduction across MPI using binning for error estimation
    auto [energy, en_err] = MeanAndBinnedErrorSqrtNUniformBin(energy_samples, comm_);
    qlten::hp_numeric::MPI_Bcast(&energy, 1, qlten::hp_numeric::kMPIMasterRank, comm_);

    // Gradient estimation and MPI mean per tensor component
    SITPST Ostar_mean(ly, lx);
    Ostar_mean = Ostar_sum * (RealT(1.0) / RealT(sample_num));
    grad = ELocConj_Ostar_sum * (RealT(1.0) / RealT(sample_num)) + ComplexConjugate(-energy) * Ostar_mean;

    for (size_t row = 0; row < ly; row++) {
      for (size_t col = 0; col < lx; col++) {
        const size_t phy_dim = grad({row, col}).size();
        for (size_t compt = 0; compt < phy_dim; compt++) {
          grad({row, col})[compt] = MPIMeanTensor(grad({row, col})[compt], comm_);
          if (collect_sr_buffers_) {
            Ostar_mean({row, col})[compt] = MPIMeanTensor(Ostar_mean({row, col})[compt], comm_);
          }
        }
      }
    }
    grad.ActFermionPOps();

    double grad_norm = 0.0;
    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
      grad_norm = grad.NormSquare();
    }

    Result res;
    res.energy = energy;
    double en_err_out = en_err;
    DetectEnergyErrorAnomaly_(en_err_out, sample_num, energy_samples);
    res.energy_error = (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) ? en_err_out : 0.0;
    res.gradient = std::move(grad);
    res.gradient_norm = grad_norm;
    res.accept_rates_avg = std::move(accept_rates_avg);
    if (collect_sr_buffers_) {
      res.Ostar_mean = std::move(Ostar_mean);
      res.Ostar_samples = std::move(Ostar_samples);
    }
    return res;
  }

 private:
  /**
   * @brief Check acceptance rates against global maxima and print anomalies.
   *
   * Implementation mirrors the previous Optimizer-level check, but is now
   * local to the evaluator to keep MC responsibilities together.
   */
  bool AcceptanceRateCheck_(const std::vector<double> &accept_rate) const {
    if (accept_rate.empty()) { return false; }
    bool too_small = false;
    std::vector<double> global_max(accept_rate.size());
    HANDLE_MPI_ERROR(::MPI_Allreduce(accept_rate.data(),
                                     global_max.data(),
                                     static_cast<int>(accept_rate.size()),
                                     MPI_DOUBLE,
                                     MPI_MAX,
                                     comm_));
    for (size_t i = 0; i < accept_rate.size(); i++) {
      if (accept_rate[i] < 0.5 * global_max[i]) {
        too_small = true;
        std::cout << "Process " << engine_.Rank() << ": Acceptance rate[" << i
                  << "] = " << accept_rate[i] << " is too small compared to global max "
                  << global_max[i] << std::endl;
      }
    }
    return too_small;
  }

  MonteCarloEngine<TenElemT, QNT, MonteCarloSweepUpdater> &engine_;
  EnergySolver &solver_;
  const MPI_Comm &comm_;
  bool collect_sr_buffers_;
  size_t reserved_samples_ = 0;

  /**
   * @brief Detect and report anomalies when the energy error is infinite.
   *
   * - If expected_total_bins > 1 (given sqrt(N) binning), an infinite error is
   *   considered anomalous. Master prints summary diagnostics; all ranks scan a
   *   few local samples for non-finite values and print them.
   */
  void DetectEnergyErrorAnomaly_(double en_err_out,
                                 size_t samples_per_rank,
                                 const std::vector<TenElemT> &energy_samples) const {
    if (!std::isinf(en_err_out)) { return; }
    const size_t bin_size = std::max<size_t>(1, static_cast<size_t>(std::sqrt(samples_per_rank)));
    const size_t local_bins = samples_per_rank / bin_size;
    const size_t expected_total_bins = engine_.MpiSize() * local_bins;

    if (engine_.Rank() == qlten::hp_numeric::kMPIMasterRank && expected_total_bins > 1) {
      std::cerr << "Energy error is infinite. expected_total_bins=" << expected_total_bins
                << ", mpi_size=" << engine_.MpiSize()
                << ", samples_per_rank=" << samples_per_rank
                << ", bin_size=" << bin_size
                << ". This indicates an anomaly (e.g., NaNs or identical samples)."
                << std::endl;
    }
    if (expected_total_bins <= 1) { return; }

    size_t printed = 0;
    bool has_nonfinite = false;
    for (size_t i = 0; i < energy_samples.size() && printed < 5; ++i) {
      const auto &e = energy_samples[i];
      const bool finite = std::isfinite(static_cast<double>(std::real(e))) &&
                          std::isfinite(static_cast<double>(std::imag(e)));
      if (!finite) {
        has_nonfinite = true;
        std::cerr << "Process " << engine_.Rank() << ": non-finite energy_samples[" << i << "] = " << e << std::endl;
        ++printed;
      }
    }
    if (!has_nonfinite && engine_.Rank() == qlten::hp_numeric::kMPIMasterRank) {
      std::cerr << "No non-finite samples detected locally; consider checking bin counts/gather consistency across ranks." << std::endl;
    }
  }
};

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_MC_ENERGY_GRAD_EVALUATOR_H


