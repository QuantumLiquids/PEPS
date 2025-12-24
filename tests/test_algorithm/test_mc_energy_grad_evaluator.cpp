// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-27
*
* Description: QuantumLiquids/PEPS project. Exact vs MC evaluator test for
*              2x2 spinless free fermion (t=1, t2=0) — compares energy and gradient.

two issues: 1. mpirun -n 4 ./complex can pass. But potential trouble is the phase difference when benchmark grad.
2. The results will change, random seeds is not fixed. MonteCarloSweepUpdaterBase, Configuration::Random(...).
We will fix them after fixed some other issues.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/algorithm/vmc_update/mc_energy_grad_evaluator.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"

#include <unordered_map>
#include <string>
#include <sstream>
#include <cstdint>

#include "../test_mpi_env.h"

#include <filesystem>
#include <iostream>
#include <cmath>

using namespace qlten;
using namespace qlpeps;

// A test-local cached EnergySolver wrapper for SquareSpinlessFermion.
// Cache key is the configuration flattened to a string; value stores E_loc and holes.
template<typename TenElemT, typename QNT>
class CachedSpinlessFermionSolver {
 public:
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;
  using WaveFuncComp = TPSWaveFunctionComponent<TenElemT, QNT>;
  using HoleTN = TensorNetwork2D<TenElemT, QNT>;

  explicit CachedSpinlessFermionSolver(double t, double t2, double V) : base_(t, t2, V) {}

  template<typename TE = TenElemT, typename QN = QNT, bool calchols>
  TE CalEnergyAndHoles(const SITPST *sitps, WaveFuncComp *tps_sample, HoleTN &hole_res) {
    const std::string key = KeyFromConfig_(tps_sample->config);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      if constexpr (calchols) {
        hole_res = it->second.holes;
      }
      return it->second.energy;
    }

    TE energy = base_.template CalEnergyAndHoles<TE, QN, calchols>(sitps, tps_sample, hole_res);
    if constexpr (calchols) {
      CacheValue v{energy, hole_res};
      cache_.emplace(key, std::move(v));
    } else {
      // Not expected in this test, but keep compatible: compute holes to cache as well.
      HoleTN holes_tmp(hole_res.rows(), hole_res.cols());
      TE energy2 = base_.template CalEnergyAndHoles<TE, QN, true>(sitps, tps_sample, holes_tmp);
      (void)energy2; // energy2 should equal energy up to numerical noise
      CacheValue v{energy, std::move(holes_tmp)};
      cache_.emplace(key, std::move(v));
    }
    return energy;
  }

 private:
  struct CacheValue {
    TenElemT energy;
    HoleTN holes;
  };

  static std::string KeyFromConfig_(const Configuration &config) {
    std::ostringstream os;
    for (size_t r = 0; r < config.rows(); ++r) {
      for (size_t c = 0; c < config.cols(); ++c) {
        os << config({r, c}) << ',';
      }
      os << ';';
    }
    return os.str();
  }

  SquareSpinlessFermion base_;
  std::unordered_map<std::string, CacheValue> cache_;
};

// A test-local cached MC updater for NN exchange.
// Cache key: flattened configuration + bond (site1, site2, dir). Value: psi_b.
template<typename TenElemT, typename QNT>
class CachedMCUpdateSquareNNExchange : public MonteCarloSweepUpdaterBase<> {
 public:
  // Deterministically seed RNG per MPI rank (multi-chain sampling).
  //
  // Each rank runs an independent Markov chain. We aggregate estimates across ranks via MPI.
  // Reproducibility across runs is achieved by a fixed base seed + a deterministic rank mix.
  CachedMCUpdateSquareNNExchange() : MonteCarloSweepUpdaterBase<>() {
    int rank_tmp = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_tmp);
    // splitmix64-style mixing via a simple golden-ratio multiplier xor base
    const uint64_t base_seed = 0xC0FFEE1234ULL;
    const uint64_t mixed = base_seed ^ (0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(rank_tmp));
    this->random_engine_.seed(static_cast<unsigned int>(mixed));
  }
  template<typename TE = TenElemT, typename QN = QNT>
  void operator()(const SplitIndexTPS<TE, QN> &sitps,
                  TPSWaveFunctionComponent<TE, QN> &tps_component,
                  std::vector<double> &accept_rates) {
    if (!precomputed_) {
      PrecomputeAllPermutations_(sitps, tps_component);
      precomputed_ = true;
    }
    // Global update among all cached configurations using Non-detailed-balance MCMC
    const std::string cur_key = KeyFromConfig_(tps_component.config);
    auto it_idx = key_to_index_.find(cur_key);
    if (it_idx == key_to_index_.end()) {
      throw std::runtime_error("Current configuration not found in precomputed table.");
    }
    size_t init_idx = it_idx->second;
    size_t final_idx = SuwaTodoStateUpdate(init_idx, weights_, this->random_engine_);
    bool changed = (final_idx != init_idx);
    if (changed) {
      // Rebuild the wavefunction component at the chosen configuration (deep rebuild of TN/env).
      //
      // IMPORTANT:
      // TPSWaveFunctionComponent now owns a Contractor with internal caches. Directly overwriting
      // `tn`/`config` without reinitializing the contractor will leave caches stale and can cause
      // inconsistent amplitudes / biased energies. Always use the component API to keep them in sync.
      const Configuration &new_cfg = all_configs_[final_idx];
      tps_component.ReplaceGlobalConfig(sitps, new_cfg);
    }
    accept_rates = {changed ? 1.0 : 0.0};
  }

 private:
  template<typename TE = TenElemT, typename QN = QNT>
  void PrecomputeAllPermutations_(const SplitIndexTPS<TE, QN> &sitps,
                                  const TPSWaveFunctionComponent<TE, QN> &tps_component) {
    const size_t Lx = sitps.cols();
    const size_t Ly = sitps.rows();
    // Flatten current config and sort to get multiset baseline
    std::vector<size_t> vec;
    vec.reserve(Lx * Ly);
    for (size_t r = 0; r < Ly; ++r) {
      for (size_t c = 0; c < Lx; ++c) {
        vec.push_back(tps_component.config({r, c}));
      }
    }
    std::sort(vec.begin(), vec.end());

    // Enumerate all unique permutations
    do {
      // Construct Configuration from vector
      Configuration cfg(Ly, Lx);
      for (size_t i = 0; i < vec.size(); ++i) {
        const size_t r = i / Lx;
        const size_t c = i % Lx;
        cfg({r, c}) = vec[i];
      }
      // Build an isolated wavefunction component to evaluate amplitude without touching the live one
      TPSWaveFunctionComponent<TE, QN> tmp(sitps, cfg, tps_component.trun_para);
      const std::string key = KeyFromConfig_(cfg);
      key_to_index_[key] = all_configs_.size();
      all_configs_.push_back(cfg);
      amps_.push_back(tmp.amplitude);
      weights_.push_back(std::norm(tmp.amplitude));
    } while (std::next_permutation(vec.begin(), vec.end()));

    // Ensure the current state's amplitude is consistent in table as a sanity check path
    const std::string cur_key = KeyFromConfig_(tps_component.config);
    if (key_to_index_.find(cur_key) == key_to_index_.end()) {
      key_to_index_[cur_key] = all_configs_.size();
      all_configs_.push_back(tps_component.config);
      amps_.push_back(tps_component.amplitude);
      weights_.push_back(std::norm(tps_component.amplitude));
    }
  }

  static std::string KeyFromConfig_(const Configuration &config) {
    std::ostringstream os;
    for (size_t r = 0; r < config.rows(); ++r) {
      for (size_t c = 0; c < config.cols(); ++c) {
        os << config({r, c}) << ',';
      }
      os << ';';
    }
    return os.str();
  }

  bool precomputed_ = false;
  std::unordered_map<std::string, size_t> key_to_index_;
  std::vector<Configuration> all_configs_;
  std::vector<TenElemT> amps_;
  std::vector<double> weights_;
};

struct SpinlessFermionExactVsMCTest : MPITest {
  using QNT = qlten::special_qn::fZ2QN;
  using TenElemT = TEN_ELEM_TYPE;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;

  size_t Lx = 2;
  size_t Ly = 2;
  size_t Dpeps = 4;
  double t = 1.0;
  double t2 = 0.0;

  // Build a random Z2-symmetric SITPS with OBC: boundary bonds dim 1 (even), interior bonds have sectors {even:1, odd:1}
  SITPST BuildRandomZ2OBCSITPS(size_t d_virtual = 2) {
    // physical components follow SquareSpinlessFermion convention:
    // phys=0 -> |1> (occupied, odd parity), phys=1 -> |0> (empty, even parity)
    SITPST tps(Ly, Lx, 2);
    auto even_sct = QNSctT(QNT(0), 1);
    auto odd_sct = QNSctT(QNT(1), 1);
    qlten::SetRandomSeed(1); // This seed make each amplitude component of wave function uniform (in the sense of order).(for real tensors.)
    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        // Construct four virtual indices with OBC
        IndexT left = (col == 0)
                        ? IndexT({even_sct}, TenIndexDirType::IN)
                        : IndexT({even_sct, odd_sct}, TenIndexDirType::IN);
        IndexT down = (row == Ly - 1)
                        ? IndexT({even_sct}, TenIndexDirType::OUT)
                        : IndexT({even_sct, odd_sct}, TenIndexDirType::OUT);
        IndexT right = (col == Lx - 1)
                         ? IndexT({even_sct}, TenIndexDirType::OUT)
                         : IndexT({even_sct, odd_sct}, TenIndexDirType::OUT);
        IndexT up = (row == 0)
                      ? IndexT({even_sct}, TenIndexDirType::IN)
                      : IndexT({even_sct, odd_sct}, TenIndexDirType::IN);
        // Fermionic parity legs (1-dim), choose by physical state per convention:
        // phys=0 (|1>): odd; phys=1 (|0>): even
        IndexT parity_even = IndexT({even_sct}, TenIndexDirType::IN);
        IndexT parity_odd = IndexT({odd_sct}, TenIndexDirType::IN);

        for (size_t phys = 0; phys < 2; ++phys) {
          const IndexT &parity = (phys == 0) ? parity_odd : parity_even;
          Tensor tensor({left, down, right, up, parity});
          tensor.Random(QNT(0)); // even parity random init
          tps({row, col})[phys] = tensor;
        }
      }
    }
    // Broadcast to all ranks
    qlpeps::MPI_Bcast(tps, comm, qlten::hp_numeric::kMPIMasterRank);
    return tps;
  }
};

TEST_F(SpinlessFermionExactVsMCTest, EnergyAndGradientMatch) {
  // 1) Build a small random Z2-symmetric SITPS (D=2 with even/odd sectors), OBC boundaries dim 1, and set model
  SITPST sitps = BuildRandomZ2OBCSITPS(2);
  SquareSpinlessFermion model(t, t2, 0);

  // 2) Exact enumeration over all half-filling configurations (two occupied, two empty)
  auto trun_para = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, Dpeps * Dpeps, 0.0);
  std::vector<Configuration> all_configs = GenerateAllPermutationConfigs({2, 2}, Lx, Ly);
  auto [energy_exact, grad_exact, err_exact] = ExactSumEnergyEvaluatorMPI<SquareSpinlessFermion, TenElemT, QNT>(
    sitps,
    all_configs,
    trun_para,
    model,
    Ly,
    Lx,
    comm,
    qlten::hp_numeric::kMPIMasterRank,
    1);

  // 3) MC evaluator with moderate samples (fast + stable)
  Configuration half_filling(Ly, Lx);
  half_filling({0, 0}) = 1;
  half_filling({0, 1}) = 0;
  half_filling({1, 0}) = 0;
  half_filling({1, 1}) = 1;
  // Use smaller samples to keep Debug run fast; can be increased later
  MonteCarloParams mc_params(10000, 1, 1, half_filling, false);
  PEPSParams peps_params(trun_para);
  MonteCarloEngine<TenElemT, QNT, CachedMCUpdateSquareNNExchange<TenElemT, QNT>> engine(sitps, mc_params, peps_params, comm);
  // Use cached solver to accelerate repeated configurations in MC.
  CachedSpinlessFermionSolver<TenElemT, QNT> cached_model(t, t2, 0);
  MCEnergyGradEvaluator<TenElemT, QNT, CachedMCUpdateSquareNNExchange<TenElemT, QNT>, CachedSpinlessFermionSolver<TenElemT, QNT>> evaluator(
    engine,
    cached_model,
    comm,
    false);
  auto mc_res = evaluator.Evaluate(sitps);

  // 4) Assertions
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    double e_mc = std::real(mc_res.energy);
    double e_ex = std::real(energy_exact);
    double e_sigma = mc_res.energy_error;
    EXPECT_TRUE(std::isfinite(e_sigma));
    EXPECT_TRUE(e_sigma >= 0);
    double e_tol =  3.0 * e_sigma;
    EXPECT_NEAR(e_mc, e_ex, e_tol);

    // Gradient comparison: normalize both to unit norm and compare direction
    double grad_ex_norm = grad_exact.NormSquare();
    double grad_mc_norm = mc_res.gradient.NormSquare();
    double grad_ex_abs = std::sqrt(grad_ex_norm);
    double grad_mc_abs = std::sqrt(grad_mc_norm);

    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      std::cout << "[debug] grad_ex_abs=" << grad_ex_abs
          << ", grad_mc_abs=" << grad_mc_abs << std::endl;
    }
    if (grad_ex_abs < 1e-6 || grad_mc_abs < 1e-6) {
      if (rank == qlten::hp_numeric::kMPIMasterRank) {
        std::cout << "[debug] branch=near-zero-abs" << std::endl;
      }
      // If either is near-zero, compare absolute difference directly
      auto grad_diff_abs = std::sqrt((grad_exact - mc_res.gradient).NormSquare());
      EXPECT_LT(grad_diff_abs, 0.05);
    } else {
      if (rank == qlten::hp_numeric::kMPIMasterRank) {
        std::cout << "[debug] branch=unit-direction" << std::endl;
      }
      // Unit-normalize and compare direction (scale-free)
      grad_exact.NormalizeAllSite();
      mc_res.gradient.NormalizeAllSite();
      auto grad_dir_diff = grad_exact - mc_res.gradient;
      double dir_err = std::sqrt(grad_dir_diff.NormSquare());
      // For 2x2 system with 10k samples, a conservative direction error tolerance
      EXPECT_LT(dir_err, 0.3);
    }
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
