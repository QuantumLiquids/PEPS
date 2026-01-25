// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 2026-01-24
 *
 * Description: Performance benchmark for TRG PBC operations in VMC.
 * This test measures the time for key operations:
 * - TRGContractor: Init, Trace, PunchAllHoles, EvaluateReplacement, BeginTrial/CommitTrial
 * - MC Sweep: per-bond update time and acceptance rate
 * - Energy calculation: CalEnergyAndHoles breakdown
 */

#include "gtest/gtest.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/trg/trg_contractor.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_j1j2_xxz_pbc.h"

using namespace qlten;
using namespace qlpeps;

using qlten::Timer;

// Use double for benchmarking (consistent with typical VMC runs)
using TenElemT = QLTEN_Double;
using QNT = qlten::special_qn::TrivialRepQN;
using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;

namespace {

// Helper to print benchmark header
void PrintBenchmarkHeader(const std::string& title, size_t n, size_t D) {
  std::cout << "\n====== " << title << " (" << n << "x" << n << ", D=" << D << ") ======\n" << std::endl;
}

// Helper to print timing result
void PrintTiming(const std::string& name, double time_ms, size_t repeat = 1, const std::string& suffix = "") {
  std::cout << "  " << std::left << std::setw(24) << name << ": "
            << std::fixed << std::setprecision(3) << std::setw(10) << time_ms << " ms";
  if (repeat > 1) {
    std::cout << " (avg over " << repeat << ")";
  }
  if (!suffix.empty()) {
    std::cout << " " << suffix;
  }
  std::cout << std::endl;
}

// Build a synthetic PBC SplitIndexTPS for benchmarking
SplitIndexTPS<TenElemT, QNT> BuildSyntheticSITPS(size_t n, size_t D) {
  SplitIndexTPS<TenElemT, QNT> sitps(n, n, /*phy_dim=*/2, BoundaryCondition::Periodic);

  const IndexT virt_out({QNSctT(QNT(), D)}, TenIndexDirType::OUT);

  // Build edge indices
  std::vector<std::vector<IndexT>> hx_out(n, std::vector<IndexT>(n, virt_out));
  std::vector<std::vector<IndexT>> vy_out(n, std::vector<IndexT>(n, virt_out));

  std::mt19937 rng(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<double> dist(0.1, 1.0);

  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      const IndexT idx_l = InverseIndex(hx_out[r][(c + n - 1) % n]);
      const IndexT idx_r = hx_out[r][c];
      const IndexT idx_u = InverseIndex(vy_out[(r + n - 1) % n][c]);
      const IndexT idx_d = vy_out[r][c];

      for (size_t phys = 0; phys < 2; ++phys) {
        Tensor T({idx_l, idx_d, idx_r, idx_u});
        // Fill with random values
        for (size_t l = 0; l < D; ++l) {
          for (size_t d = 0; d < D; ++d) {
            for (size_t rr = 0; rr < D; ++rr) {
              for (size_t u = 0; u < D; ++u) {
                T({l, d, rr, u}) = TenElemT(dist(rng));
              }
            }
          }
        }
        sitps({r, c})[phys] = std::move(T);
      }
    }
  }

  return sitps;
}

// Build TensorNetwork2D from SplitIndexTPS and configuration
TensorNetwork2D<TenElemT, QNT> BuildTNFromConfig(
    const SplitIndexTPS<TenElemT, QNT>& sitps,
    const Configuration& config) {
  return TensorNetwork2D<TenElemT, QNT>(sitps, config);
}

}  // namespace

// ============================================================================
// Test 1: TRG Contractor Operations Benchmark
// ============================================================================
TEST(TRGPBCBenchmark, ContractorOperations_4x4) {
  constexpr size_t n = 4;
  constexpr size_t D = 4;  // Small D for quick benchmark; scale up for real profiling
  constexpr size_t REPEAT = 3;

  PrintBenchmarkHeader("TRG Contractor Operations", n, D);

  auto sitps = BuildSyntheticSITPS(n, D);

  // Create checkerboard Neel configuration (half up, half down)
  Configuration config(n, n);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      config({r, c}) = (r + c) % 2;
    }
  }

  auto tn = BuildTNFromConfig(sitps, config);

  TRGContractor<TenElemT, QNT> trg(n, n);
  TRGTruncateParams<RealT> trunc_params(/*d_min=*/1, /*d_max=*/D, /*trunc_error=*/1e-10);
  trg.SetTruncateParams(trunc_params);

  std::cout << "TRG Contractor Operations:\n";

  // Benchmark Init + Trace (cold cache) - create fresh contractor each time
  double trace_cold_total = 0.0;
  for (size_t i = 0; i < REPEAT; ++i) {
    TRGContractor<TenElemT, QNT> trg_fresh(n, n);
    trg_fresh.SetTruncateParams(trunc_params);
    trg_fresh.Init(tn);
    Timer trace_timer("Trace_cold");
    auto Z = trg_fresh.Trace(tn);
    trace_cold_total += trace_timer.Elapsed() * 1000.0;
    (void)Z;
  }
  PrintTiming("Init+Trace() [cold]", trace_cold_total / REPEAT, REPEAT);

  // Now use the pre-created trg for warm cache tests
  trg.Init(tn);
  trg.Trace(tn);  // Initialize cache

  // Benchmark Trace with warm cache
  double trace_warm_total = 0.0;
  for (size_t i = 0; i < REPEAT; ++i) {
    Timer trace_warm_timer("Trace_warm");
    auto Z = trg.Trace(tn);
    trace_warm_total += trace_warm_timer.Elapsed() * 1000.0;
    (void)Z;
  }
  PrintTiming("Trace() [warm cache]", trace_warm_total / REPEAT, REPEAT);

  // Benchmark PunchAllHoles (requires warm cache)
  double punch_total = 0.0;
  for (size_t i = 0; i < REPEAT; ++i) {
    Timer punch_timer("PunchAllHoles");
    auto holes = trg.PunchAllHoles(tn);
    punch_total += punch_timer.Elapsed() * 1000.0;
    (void)holes;
  }
  PrintTiming("PunchAllHoles()", punch_total / REPEAT, REPEAT);

  // Benchmark EvaluateReplacement (single site, requires warm cache)
  SiteIdx site{1, 2};
  size_t alt_config = 1 - config(site);
  std::vector<std::pair<SiteIdx, Tensor>> replacements{{site, sitps(site)[alt_config]}};

  double eval_total = 0.0;
  for (size_t i = 0; i < REPEAT; ++i) {
    Timer eval_timer("EvaluateReplacement");
    auto psi = trg.EvaluateReplacement(replacements);
    eval_total += eval_timer.Elapsed() * 1000.0;
    (void)psi;
  }
  PrintTiming("EvaluateReplacement() [1 site]", eval_total / REPEAT, REPEAT);

  // Benchmark EvaluateReplacement (2 sites - typical for NN exchange)
  SiteIdx site2{1, 3};
  std::vector<std::pair<SiteIdx, Tensor>> replacements2{
      {site, sitps(site)[alt_config]},
      {site2, sitps(site2)[1 - config(site2)]}
  };

  double eval2_total = 0.0;
  for (size_t i = 0; i < REPEAT; ++i) {
    Timer eval2_timer("EvaluateReplacement_2site");
    auto psi = trg.EvaluateReplacement(replacements2);
    eval2_total += eval2_timer.Elapsed() * 1000.0;
    (void)psi;
  }
  PrintTiming("EvaluateReplacement() [2 sites]", eval2_total / REPEAT, REPEAT);

  std::cout << std::endl;
}

// ============================================================================
// Test 2: TRG Trial Operations Benchmark
// ============================================================================
TEST(TRGPBCBenchmark, TrialOperations_4x4) {
  constexpr size_t n = 4;
  constexpr size_t D = 4;
  constexpr size_t REPEAT = 3;

  PrintBenchmarkHeader("TRG Trial Operations (MC Update Simulation)", n, D);

  auto sitps = BuildSyntheticSITPS(n, D);

  Configuration config(n, n);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      config({r, c}) = (r + c) % 2;
    }
  }

  auto tn = BuildTNFromConfig(sitps, config);

  TRGContractor<TenElemT, QNT> trg(n, n);
  TRGTruncateParams<RealT> trunc_params(/*d_min=*/1, /*d_max=*/D, /*trunc_error=*/1e-10);
  trg.SetTruncateParams(trunc_params);
  trg.Init(tn);
  trg.Trace(tn);  // Initialize cache

  std::cout << "Trial Operations (simulating MC updates):\n";

  // Benchmark BeginTrialWithReplacement
  SiteIdx site1{1, 2};
  SiteIdx site2{1, 3};
  std::vector<std::pair<SiteIdx, Tensor>> replacements{
      {site1, sitps(site1)[1 - config(site1)]},
      {site2, sitps(site2)[1 - config(site2)]}
  };

  double begin_total = 0.0;
  for (size_t i = 0; i < REPEAT; ++i) {
    Timer begin_timer("BeginTrial");
    auto trial = trg.BeginTrialWithReplacement(replacements);
    begin_total += begin_timer.Elapsed() * 1000.0;
    (void)trial;
  }
  PrintTiming("BeginTrialWithReplacement()", begin_total / REPEAT, REPEAT);

  // Benchmark CommitTrial
  double commit_total = 0.0;
  for (size_t i = 0; i < REPEAT; ++i) {
    // Create fresh trial each time
    auto trial = trg.BeginTrialWithReplacement(replacements);
    Timer commit_timer("CommitTrial");
    trg.CommitTrial(std::move(trial));
    commit_total += commit_timer.Elapsed() * 1000.0;

    // Toggle config and replacements for next iteration
    config(site1) = 1 - config(site1);
    config(site2) = 1 - config(site2);
    replacements = {
        {site1, sitps(site1)[1 - config(site1)]},
        {site2, sitps(site2)[1 - config(site2)]}
    };
  }
  PrintTiming("CommitTrial()", commit_total / REPEAT, REPEAT);

  // Benchmark full trial cycle (Begin + Commit)
  double cycle_total = 0.0;
  for (size_t i = 0; i < REPEAT; ++i) {
    Timer cycle_timer("TrialCycle");
    auto trial = trg.BeginTrialWithReplacement(replacements);
    trg.CommitTrial(std::move(trial));
    cycle_total += cycle_timer.Elapsed() * 1000.0;

    config(site1) = 1 - config(site1);
    config(site2) = 1 - config(site2);
    replacements = {
        {site1, sitps(site1)[1 - config(site1)]},
        {site2, sitps(site2)[1 - config(site2)]}
    };
  }
  PrintTiming("Begin + Commit cycle", cycle_total / REPEAT, REPEAT);

  std::cout << std::endl;
}

// ============================================================================
// Test 3: MC Sweep Benchmark
// ============================================================================
TEST(TRGPBCBenchmark, MCSweep_4x4) {
  constexpr size_t n = 4;
  constexpr size_t D = 4;
  constexpr size_t NUM_SWEEPS = 2;

  PrintBenchmarkHeader("MC Sweep (PBC NN Exchange)", n, D);

  auto sitps = BuildSyntheticSITPS(n, D);

  // Neel configuration
  Configuration config(n, n);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      config({r, c}) = (r + c) % 2;
    }
  }

  TRGTruncateParams<RealT> trunc_params(/*d_min=*/1, /*d_max=*/D, /*trunc_error=*/1e-10);
  TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, TRGContractor> comp(sitps, config, trunc_params);

  MCUpdateSquareNNExchangePBC updater;

  std::cout << "MC Sweep Statistics:\n";

  std::vector<double> sweep_times;
  std::vector<double> accept_rates_all;
  size_t total_bonds = n * n * 2;

  for (size_t sweep = 0; sweep < NUM_SWEEPS; ++sweep) {
    std::vector<double> accept_rates;

    Timer sweep_timer("Sweep");
    updater(sitps, comp, accept_rates);
    double sweep_time_ms = sweep_timer.Elapsed() * 1000.0;

    sweep_times.push_back(sweep_time_ms);
    accept_rates_all.push_back(accept_rates[0]);

    std::cout << "  Sweep " << sweep + 1 << ": "
              << std::fixed << std::setprecision(2) << sweep_time_ms << " ms, "
              << "accept rate = " << std::setprecision(4) << accept_rates[0] * 100.0 << "%"
              << std::endl;
  }

  // Summary statistics
  double avg_sweep_time = 0.0;
  double avg_accept_rate = 0.0;
  for (size_t i = 0; i < NUM_SWEEPS; ++i) {
    avg_sweep_time += sweep_times[i];
    avg_accept_rate += accept_rates_all[i];
  }
  avg_sweep_time /= NUM_SWEEPS;
  avg_accept_rate /= NUM_SWEEPS;

  std::cout << "\nSummary (" << total_bonds << " bonds per sweep):\n";
  PrintTiming("Avg sweep time", avg_sweep_time, NUM_SWEEPS);
  PrintTiming("Per-bond time", avg_sweep_time / total_bonds, NUM_SWEEPS);
  std::cout << "  " << std::left << std::setw(24) << "Avg acceptance rate" << ": "
            << std::fixed << std::setprecision(2) << avg_accept_rate * 100.0 << " %" << std::endl;

  std::cout << std::endl;
}

// ============================================================================
// Test 4: Energy Calculation Benchmark (CalEnergyAndHoles)
// ============================================================================
TEST(TRGPBCBenchmark, EnergyCalculation_4x4) {
  constexpr size_t n = 4;
  constexpr size_t D = 4;
  constexpr size_t REPEAT = 2;

  PrintBenchmarkHeader("Energy Calculation (J1-J2 XXZ PBC, J2=0)", n, D);

  auto sitps = BuildSyntheticSITPS(n, D);

  // Neel configuration
  Configuration config(n, n);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      config({r, c}) = (r + c) % 2;
    }
  }

  TRGTruncateParams<RealT> trunc_params(/*d_min=*/1, /*d_max=*/D, /*trunc_error=*/1e-10);
  TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, TRGContractor> comp(sitps, config, trunc_params);

  SquareSpinOneHalfJ1J2XXZModelPBC model(1.0, 1.0, 0.0, 0.0, 0.0);

  std::cout << "Energy Calculation (CalEnergyAndHoles):\n";

  TensorNetwork2D<TenElemT, QNT> holes(n, n);
  std::vector<double> energy_times;

  for (size_t i = 0; i < REPEAT; ++i) {
    Timer energy_timer("CalEnergyAndHoles");
    TenElemT energy = model.template CalEnergyAndHoles<TenElemT, QNT, true>(&sitps, &comp, holes);
    double time_ms = energy_timer.Elapsed() * 1000.0;

    energy_times.push_back(time_ms);

    std::cout << "  Iteration " << i + 1 << ": " << std::fixed << std::setprecision(2)
              << time_ms << " ms, E = " << std::setprecision(6) << energy << std::endl;
  }

  // Count off-diagonal bonds (different spins)
  size_t off_diag_bonds = 0;
  size_t total_bonds = n * n * 2;  // horizontal + vertical
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      // Horizontal
      if (config({r, c}) != config({r, (c + 1) % n})) {
        off_diag_bonds++;
      }
      // Vertical
      if (config({r, c}) != config({(r + 1) % n, c})) {
        off_diag_bonds++;
      }
    }
  }

  double avg_time = 0.0;
  for (auto t : energy_times) avg_time += t;
  avg_time /= REPEAT;

  std::cout << "\nSummary:\n";
  PrintTiming("Avg CalEnergyAndHoles", avg_time, REPEAT);
  std::cout << "  " << std::left << std::setw(24) << "Off-diagonal bonds" << ": "
            << off_diag_bonds << " / " << total_bonds << std::endl;
  PrintTiming("Estimated per EvalRepl", avg_time / std::max(off_diag_bonds, size_t(1)), 1,
              "(if dominated by EvalRepl)");

  std::cout << std::endl;
}

// ============================================================================
// Test 5: Full VMC Iteration Cost Estimation
// ============================================================================
TEST(TRGPBCBenchmark, IterationCostEstimate_4x4) {
  constexpr size_t n = 4;
  constexpr size_t D = 4;
  constexpr size_t NUM_SAMPLES = 3;  // Small number for quick benchmark

  PrintBenchmarkHeader("VMC Iteration Cost Estimate", n, D);

  auto sitps = BuildSyntheticSITPS(n, D);

  Configuration config(n, n);
  for (size_t r = 0; r < n; ++r) {
    for (size_t c = 0; c < n; ++c) {
      config({r, c}) = (r + c) % 2;
    }
  }

  TRGTruncateParams<RealT> trunc_params(/*d_min=*/1, /*d_max=*/D, /*trunc_error=*/1e-10);
  TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, TRGContractor> comp(sitps, config, trunc_params);

  MCUpdateSquareNNExchangePBC updater;
  SquareSpinOneHalfJ1J2XXZModelPBC model(1.0, 1.0, 0.0, 0.0, 0.0);

  std::cout << "Simulating " << NUM_SAMPLES << " MC samples...\n\n";

  double total_sweep_time = 0.0;
  double total_energy_time = 0.0;
  std::vector<double> accept_rates_all;

  TensorNetwork2D<TenElemT, QNT> holes(n, n);

  for (size_t sample = 0; sample < NUM_SAMPLES; ++sample) {
    // MC sweep
    std::vector<double> accept_rates;
    Timer sweep_timer("Sweep");
    updater(sitps, comp, accept_rates);
    total_sweep_time += sweep_timer.Elapsed() * 1000.0;
    accept_rates_all.push_back(accept_rates[0]);

    // Energy calculation
    Timer energy_timer("Energy");
    auto energy = model.template CalEnergyAndHoles<TenElemT, QNT, true>(&sitps, &comp, holes);
    total_energy_time += energy_timer.Elapsed() * 1000.0;
    (void)energy;
  }

  double avg_sweep = total_sweep_time / NUM_SAMPLES;
  double avg_energy = total_energy_time / NUM_SAMPLES;
  double avg_accept = 0.0;
  for (auto r : accept_rates_all) avg_accept += r;
  avg_accept /= NUM_SAMPLES;

  std::cout << "Per-sample breakdown:\n";
  PrintTiming("MC Sweep", avg_sweep, NUM_SAMPLES);
  PrintTiming("CalEnergyAndHoles", avg_energy, NUM_SAMPLES);
  PrintTiming("Total per sample", avg_sweep + avg_energy, NUM_SAMPLES);

  std::cout << "\nProjected iteration cost (100 samples):\n";
  double projected_100 = (avg_sweep + avg_energy) * 100.0 / 1000.0;  // Convert to seconds
  std::cout << "  Total time: " << std::fixed << std::setprecision(1) << projected_100 << " s\n";
  std::cout << "  Avg acceptance rate: " << std::setprecision(2) << avg_accept * 100.0 << " %\n";

  std::cout << "\nBreakdown percentage:\n";
  double total = avg_sweep + avg_energy;
  std::cout << "  MC Sweep:           " << std::fixed << std::setprecision(1)
            << (avg_sweep / total * 100.0) << " %\n";
  std::cout << "  CalEnergyAndHoles:  " << std::fixed << std::setprecision(1)
            << (avg_energy / total * 100.0) << " %\n";

  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  return RUN_ALL_TESTS();
}
