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
#include <tuple>
#include <vector>
#include <type_traits>
#include <iostream>
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
 */
template<typename TenElemT, typename QNT, typename MonteCarloSweepUpdater, typename EnergySolver>
class MCEnergyGradEvaluator {
 public:
  using Tensor = qlten::QLTensor<TenElemT, QNT>;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using WaveFunctionComponentT = TPSWaveFunctionComponent<TenElemT, QNT>;

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
   * @brief Evaluate energy and gradient for a given state using Monte Carlo sampling.
   *
   * Semantics:
   * - Master rank assigns state (if different), broadcasts to all ranks.
   * - Engine rebuilds wavefunction component and normalizes to O(1) amplitude.
   * - Performs MC sampling for engine.MCParams().num_samples samples.
   * - Reduces energy and gradient across MPI; returns Result.
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

    // Update wavefunction component and normalize amplitude globally
    engine_.UpdateWavefunctionComponent();
    engine_.NormalizeStateOrder1();

    // Local accumulators
    const size_t ly = engine_.Ly();
    const size_t lx = engine_.Lx();
    const size_t sample_num = engine_.MCParams().num_samples;

    std::vector<TenElemT> energy_samples;
    energy_samples.reserve(sample_num);

    SITPST Ostar_sum(ly, lx);
    SITPST ELocConj_Ostar_sum(ly, lx);
    SITPST grad(ly, lx);
    std::optional<SITPST> Ostar_mean_opt;
    std::vector<SITPST> Ostar_samples; // SR optional

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
    for (size_t sweep = 0; sweep < sample_num; sweep++) {
      std::vector<double> accept_rates = engine_.StepSweep();
      if (sweep == 0) {
        accept_rates_accum = accept_rates;
      } else {
        for (size_t i = 0; i < accept_rates_accum.size(); i++) {
          accept_rates_accum[i] += accept_rates[i];
        }
      }

      // Compute local energy and holes
      TensorNetwork2D<TenElemT, QNT> holes(ly, lx);
      TenElemT local_energy = solver_.template CalEnergyAndHoles<TenElemT, QNT, true>(&engine_.State(),
                                                                                       &engine_.WavefuncComp(),
                                                                                       holes);
      TenElemT local_energy_conjugate = ComplexConjugate(local_energy);
      TenElemT inverse_amplitude = ComplexConjugate(1.0 / engine_.WavefuncComp().GetAmplitude());
      energy_samples.push_back(local_energy);

      // Accumulate O* and E_loc^* O*
      // O* = d ln(psi*) / d theta*
      SITPST Ostar_sample(ly, lx, engine_.State().PhysicalDim());
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
            Ostar_sample({row, col})[basis_index] = Ostar_tensor;
          }
        }
      }
      if (collect_sr_buffers_) {
        Ostar_samples.emplace_back(Ostar_sample);
      }
    }

    // Average acceptance rates
    std::vector<double> accept_rates_avg = accept_rates_accum;
    for (double &rate : accept_rates_avg) { rate /= double(sample_num); }

    // Downstream: check acceptance anomalies locally (with global max via MPI)
    AcceptanceRateCheck_(accept_rates_avg);

    // Energy reduction across MPI
    TenElemT en_self = Mean(energy_samples);
    auto [energy, en_err] = GatherStatisticSingleData(en_self, comm_);
    qlten::hp_numeric::MPI_Bcast(&energy, 1, qlten::hp_numeric::kMPIMasterRank, comm_);

    // Gradient estimation and MPI mean per tensor component
    SITPST Ostar_mean(ly, lx);
    Ostar_mean = Ostar_sum * (1.0 / sample_num);
    grad = ELocConj_Ostar_sum * (1.0 / sample_num) + ComplexConjugate(-energy) * Ostar_mean;

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
    // TODO(energy-error): If MPI size > 1 but en_err is infinite, this indicates an abnormal state.
    // Potential causes include: NaN energies on some ranks, degenerate identical inputs,
    // or mismatched MPI gathers. We should add diagnostics (rank-wise finite checks on
    // en_self and energy_samples) and consider failing fast or emitting a structured log.
    // Current behavior: keep en_err finite in single-process; sanitize infinities to 0 for stability.
    if (engine_.MpiSize() == 1 || std::isinf(en_err_out)) { en_err_out = 0.0; }
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
};

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_MC_ENERGY_GRAD_EVALUATOR_H


