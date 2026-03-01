// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-28
*
* Description: Equivalence test: SR and MinSR should produce the same natural
*              gradient (up to numerical tolerance) on a small fermionic system.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/mc_energy_grad_evaluator.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <complex>

#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

namespace {

// ---------------------------------------------------------------------------
// Helpers copied from test_fermion_mc_sr_golden.cpp for self-contained build.
// ---------------------------------------------------------------------------

template<typename TenElemT, typename QNT>
class CachedSpinlessFermionSolver {
 public:
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using WaveFuncComp = TPSWaveFunctionComponent<TenElemT, QNT>;
  using HoleTN = TensorNetwork2D<TenElemT, QNT>;

  explicit CachedSpinlessFermionSolver(double t, double t2, double V) : base_(t, t2, V) {}

  template<typename TE = TenElemT, typename QN = QNT, bool calchols = true>
  TE CalEnergyAndHoles(const SITPST *sitps, WaveFuncComp *tps_sample, HoleTN &hole_res) {
    const std::string key = KeyFromConfig_(tps_sample->config);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      if constexpr (calchols) { hole_res = it->second.holes; }
      return it->second.energy;
    }
    TE energy = base_.template CalEnergyAndHoles<TE, QN, calchols>(sitps, tps_sample, hole_res);
    if constexpr (calchols) {
      cache_.emplace(key, CacheValue{energy, hole_res});
    } else {
      HoleTN holes_tmp(hole_res.rows(), hole_res.cols());
      (void) base_.template CalEnergyAndHoles<TE, QN, true>(sitps, tps_sample, holes_tmp);
      cache_.emplace(key, CacheValue{energy, std::move(holes_tmp)});
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
  std::map<std::string, CacheValue> cache_;
};

template<typename TenElemT, typename QNT>
class CachedMCUpdateSquareNNExchange : public MonteCarloSweepUpdaterBase<> {
 public:
  CachedMCUpdateSquareNNExchange() : MonteCarloSweepUpdaterBase<>() {
    int rank_tmp = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_tmp);
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
    const std::string cur_key = KeyFromConfig_(tps_component.config);
    auto it_idx = key_to_index_.find(cur_key);
    if (it_idx == key_to_index_.end()) {
      throw std::runtime_error("Current configuration not found in precomputed table.");
    }
    const size_t init_idx = it_idx->second;
    const size_t final_idx = SuwaTodoStateUpdate(init_idx, weights_, this->random_engine_);
    const bool changed = (final_idx != init_idx);
    if (changed) {
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
    std::vector<size_t> vec;
    vec.reserve(Lx * Ly);
    for (size_t r = 0; r < Ly; ++r) {
      for (size_t c = 0; c < Lx; ++c) {
        vec.push_back(tps_component.config({r, c}));
      }
    }
    std::sort(vec.begin(), vec.end());
    do {
      Configuration cfg(Ly, Lx);
      for (size_t i = 0; i < vec.size(); ++i) {
        cfg({i / Lx, i % Lx}) = vec[i];
      }
      TPSWaveFunctionComponent<TE, QN> tmp(sitps, cfg, tps_component.trun_para);
      const std::string key = KeyFromConfig_(cfg);
      key_to_index_[key] = all_configs_.size();
      all_configs_.push_back(cfg);
      weights_.push_back(std::norm(tmp.amplitude));
    } while (std::next_permutation(vec.begin(), vec.end()));
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
  std::map<std::string, size_t> key_to_index_;
  std::vector<Configuration> all_configs_;
  std::vector<double> weights_;
};

template<typename SITPST, typename TenElemT>
std::complex<double> WeightedProbeInnerProduct(const SITPST &x) {
  SITPST probe = x;
  for (size_t row = 0; row < probe.rows(); ++row) {
    for (size_t col = 0; col < probe.cols(); ++col) {
      const size_t phy_dim = probe({row, col}).size();
      for (size_t i = 0; i < phy_dim; ++i) {
        auto &ten = probe({row, col})[i];
        if (ten.IsDefault()) { continue; }
        const double base = 0.012 * static_cast<double>((row + 1) * 11 + (col + 1) * 5 + (i + 1) * 2);
        if constexpr (std::is_same_v<TenElemT, QLTEN_Complex>) {
          const double imag = 0.0025 * static_cast<double>((row + 1) + (i + 1));
          ten *= TenElemT(base, imag);
        } else {
          ten *= TenElemT(base);
        }
      }
    }
  }
  const TenElemT ip = x * probe;
  return {std::real(ip), std::imag(ip)};
}

} // namespace

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

struct SRvsMinSREquivalenceTest : MPITest {
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = qlten::special_qn::fZ2QN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;

  static constexpr size_t kLy = 2;
  static constexpr size_t kLx = 2;
  static constexpr size_t kSamples = 8;

  SITPST BuildRandomZ2OBCSITPS() const {
    using IndexT = Index<QNT>;
    using QNSctT = QNSector<QNT>;
    SITPST tps(kLy, kLx, 2);
    const auto even_sct = QNSctT(QNT(0), 1);
    const auto odd_sct = QNSctT(QNT(1), 1);
    qlten::SetRandomSeed(1);
    for (size_t row = 0; row < kLy; ++row) {
      for (size_t col = 0; col < kLx; ++col) {
        IndexT left = (col == 0) ? IndexT({even_sct}, TenIndexDirType::IN)
                                 : IndexT({even_sct, odd_sct}, TenIndexDirType::IN);
        IndexT down = (row == kLy - 1) ? IndexT({even_sct}, TenIndexDirType::OUT)
                                       : IndexT({even_sct, odd_sct}, TenIndexDirType::OUT);
        IndexT right = (col == kLx - 1) ? IndexT({even_sct}, TenIndexDirType::OUT)
                                        : IndexT({even_sct, odd_sct}, TenIndexDirType::OUT);
        IndexT up = (row == 0) ? IndexT({even_sct}, TenIndexDirType::IN)
                               : IndexT({even_sct, odd_sct}, TenIndexDirType::IN);
        IndexT parity_even({even_sct}, TenIndexDirType::IN);
        IndexT parity_odd({odd_sct}, TenIndexDirType::IN);
        for (size_t phys = 0; phys < 2; ++phys) {
          const IndexT &parity = (phys == 0) ? parity_odd : parity_even;
          Tensor ten({left, down, right, up, parity});
          ten.Random(QNT(0));
          tps({row, col})[phys] = ten;
        }
      }
    }
    qlpeps::MPI_Bcast(tps, comm, qlten::hp_numeric::kMPIMasterRank);
    return tps;
  }
};

TEST_F(SRvsMinSREquivalenceTest, NaturalGradientEquivalence) {
  // --- 1. Evaluate energy, gradient, O* samples, and energy samples ---
  SITPST sitps = BuildRandomZ2OBCSITPS();
  CachedSpinlessFermionSolver<TenElemT, QNT> cached_model(/*t=*/1.0, /*t2=*/0.0, /*V=*/0.0);

  Configuration half_filling(kLy, kLx);
  half_filling({0, 0}) = 1;
  half_filling({0, 1}) = 0;
  half_filling({1, 0}) = 0;
  half_filling({1, 1}) = 1;

  auto trun_para = BMPSTruncateParams<qlten::QLTEN_Double>::SVD(/*D_min=*/4, /*D_max=*/16, /*trunc_error=*/0.0);
  MonteCarloParams mc_params(/*samples=*/kSamples, /*warmup_sweeps=*/1, /*sweeps_between=*/1, half_filling, false);
  PEPSParams peps_params(trun_para);
  MonteCarloEngine<TenElemT, QNT, CachedMCUpdateSquareNNExchange<TenElemT, QNT>> engine(
      sitps, mc_params, peps_params, comm);

  MCEnergyGradEvaluator<TenElemT, QNT, CachedMCUpdateSquareNNExchange<TenElemT, QNT>,
                        CachedSpinlessFermionSolver<TenElemT, QNT>>
      evaluator(engine, cached_model, comm, /*collect_sr_buffers=*/true);
  auto eval_res = evaluator.Evaluate(sitps);
  ASSERT_TRUE(eval_res.Ostar_mean.has_value());
  const size_t samples_per_rank = engine.SamplesPerRank();
  ASSERT_EQ(eval_res.Ostar_samples.size(), samples_per_rank);
  ASSERT_EQ(eval_res.energy_samples.size(), samples_per_rank);

  // --- 2. SR natural gradient (all ranks participate for CG MPI) ---
  ConjugateGradientParams cg_params{.max_iter = 100, .relative_tolerance = 1e-14,
                                    .residual_recompute_interval = 8};
  StochasticReconfigurationParams sr_algo_params{.cg_params = cg_params, .diag_shift = 0.0};
  OptimizerParams sr_params = OptimizerFactory::CreateStochasticReconfigurationAdvanced(
      /*max_iterations=*/1, /*energy_tolerance=*/1e-30, /*gradient_tolerance=*/1e-30,
      /*plateau_patience=*/1, sr_algo_params, /*learning_rate=*/0.1);
  Optimizer<TenElemT, QNT> sr_optimizer(sr_params, comm, rank, mpi_size);

  SITPST init_guess = eval_res.gradient * TenElemT(0.0);
  auto [sr_nat_grad, sr_cg_iters, sr_cg_resid] = sr_optimizer.CalculateNaturalGradient(
      eval_res.gradient, eval_res.Ostar_samples, eval_res.Ostar_mean.value(), init_guess);
  (void)sr_cg_iters;
  (void)sr_cg_resid;

  // --- 3. MinSR update (all ranks participate for ring exchange + eigensolve) ---
  const double learning_rate = 0.1;
  MinSRParams minsr_alg_params;
  minsr_alg_params.r_pinv = 0.0;
  minsr_alg_params.a_pinv = 0.0;
  minsr_alg_params.soft_cutoff = false; // hard cutoff = exact inverse

  OptimizerParams::BaseParams minsr_base(/*max_iterations=*/1, /*energy_tolerance=*/1e-30,
                                         /*gradient_tolerance=*/1e-30, /*plateau_patience=*/1,
                                         /*learning_rate=*/learning_rate);
  OptimizerParams minsr_params(minsr_base, minsr_alg_params);
  Optimizer<TenElemT, QNT> minsr_optimizer(minsr_params, comm, rank, mpi_size);

  auto [minsr_state, minsr_ngrad_norm] = minsr_optimizer.MinSRUpdate(
      sitps, eval_res.gradient, eval_res.Ostar_samples,
      eval_res.Ostar_mean.value(), eval_res.energy_samples,
      eval_res.energy, learning_rate, minsr_alg_params);

  // --- 4. Compare on master rank ---
  if (rank != qlten::hp_numeric::kMPIMasterRank) { return; }

  // Extract MinSR natural gradient: nat_grad = (old - new) / lr
  SITPST minsr_nat_grad = sitps + TenElemT(-1.0) * minsr_state;
  minsr_nat_grad *= TenElemT(1.0 / learning_rate);

  // Compare norms
  const double sr_ngrad_norm = std::sqrt(sr_nat_grad.NormSquare());
  // minsr_ngrad_norm returned directly by MinSRUpdate
  const double norm_rel_err = std::abs(sr_ngrad_norm - minsr_ngrad_norm)
                              / std::max(sr_ngrad_norm, 1e-15);

  // Compare via probe inner products (more sensitive to element-wise differences)
  auto sr_probe = WeightedProbeInnerProduct<SITPST, TenElemT>(sr_nat_grad);
  auto minsr_probe = WeightedProbeInnerProduct<SITPST, TenElemT>(minsr_nat_grad);

  // Compute difference norm for a direct element-wise comparison
  SITPST diff = sr_nat_grad + TenElemT(-1.0) * minsr_nat_grad;
  const double diff_norm = std::sqrt(diff.NormSquare());
  const double rel_diff = diff_norm / std::max(sr_ngrad_norm, 1e-15);

  // Print for diagnostics
  std::cout << "[SR-vs-MinSR] SR nat_grad_norm = " << sr_ngrad_norm
            << ", MinSR nat_grad_norm = " << minsr_ngrad_norm
            << ", norm_rel_err = " << norm_rel_err << std::endl;
  std::cout << "[SR-vs-MinSR] SR probe = (" << sr_probe.real() << ", " << sr_probe.imag()
            << "), MinSR probe = (" << minsr_probe.real() << ", " << minsr_probe.imag() << ")" << std::endl;
  std::cout << "[SR-vs-MinSR] diff_norm = " << diff_norm
            << ", rel_diff = " << rel_diff << std::endl;
  std::cout << "[SR-vs-MinSR] SR CG iters = " << sr_cg_iters << std::endl;

  // Assertions: SR and MinSR should agree to high precision
  constexpr double kTol = 1e-6;
  EXPECT_LT(norm_rel_err, kTol)
      << "Natural gradient norms differ: SR=" << sr_ngrad_norm
      << " MinSR=" << minsr_ngrad_norm;
  EXPECT_LT(rel_diff, kTol)
      << "Natural gradient element-wise relative difference: " << rel_diff;
  EXPECT_NEAR(sr_probe.real(), minsr_probe.real(),
              kTol * std::max(std::abs(sr_probe.real()), 1e-10))
      << "Probe inner product real parts differ";
  EXPECT_NEAR(sr_probe.imag(), minsr_probe.imag(),
              kTol * std::max(std::abs(sr_probe.imag()), 1e-10))
      << "Probe inner product imaginary parts differ";
}

// ---------------------------------------------------------------------------
// Dispatch logic tests: exercise MinSREigenSolveDispatch directly
// ---------------------------------------------------------------------------

/// Build a small known positive-definite symmetric matrix and RHS for dispatch tests.
/// Uses a diagonally-dominant construction: T_ij = 1/(1 + |i-j|), then T_ii += N.
/// This ensures all eigenvalues are positive and well-conditioned.
template<typename TenElemT>
struct DispatchTestData {
  size_t ns_global;
  size_t ns_local;
  std::vector<TenElemT> T_full;      // ns_global x ns_global, row-major
  std::vector<TenElemT> T_rowblock;  // ns_local x ns_global, row-major
  std::vector<TenElemT> rhs;         // length ns_global
};

template<typename TenElemT>
DispatchTestData<TenElemT> BuildDispatchTestData(int rank, int mpi_size) {
  // Total matrix size = 4 * mpi_size (so each rank owns 4 rows)
  const size_t ns_local = 4;
  const size_t ns_global = ns_local * static_cast<size_t>(mpi_size);

  // Build full symmetric matrix on all ranks (deterministic, same everywhere)
  std::vector<TenElemT> T_full(ns_global * ns_global);
  for (size_t i = 0; i < ns_global; ++i) {
    for (size_t j = 0; j < ns_global; ++j) {
      double val = 1.0 / (1.0 + std::abs(static_cast<double>(i) - static_cast<double>(j)));
      if (i == j) val += static_cast<double>(ns_global); // diagonal dominance
      if constexpr (std::is_same_v<TenElemT, std::complex<double>>) {
        // Add small imaginary off-diagonal part for Hermitian matrix
        double imag = (i != j) ? 0.01 * (static_cast<double>(i) - static_cast<double>(j)) : 0.0;
        T_full[i * ns_global + j] = TenElemT(val, imag);
        T_full[j * ns_global + i] = TenElemT(val, -imag); // Hermitian
      } else {
        T_full[i * ns_global + j] = val;
      }
    }
  }

  // Extract this rank's row block
  const size_t row_start = static_cast<size_t>(rank) * ns_local;
  std::vector<TenElemT> T_rowblock(ns_local * ns_global);
  for (size_t i = 0; i < ns_local; ++i) {
    for (size_t j = 0; j < ns_global; ++j) {
      T_rowblock[i * ns_global + j] = T_full[(row_start + i) * ns_global + j];
    }
  }

  // RHS: deterministic vector
  std::vector<TenElemT> rhs(ns_global);
  for (size_t i = 0; i < ns_global; ++i) {
    if constexpr (std::is_same_v<TenElemT, std::complex<double>>) {
      rhs[i] = TenElemT(std::sin(0.3 * static_cast<double>(i + 1)),
                         std::cos(0.7 * static_cast<double>(i + 1)));
    } else {
      rhs[i] = std::sin(0.3 * static_cast<double>(i + 1));
    }
  }

  return {ns_global, ns_local, std::move(T_full), std::move(T_rowblock), std::move(rhs)};
}

struct DispatchTest : MPITest {
  using TenElemT = TEN_ELEM_TYPE;
};

/// kReplicated mode always uses Path B (replicated LAPACK) and produces a valid result.
TEST_F(DispatchTest, ReplicatedModeProducesValidResult) {
  auto data = BuildDispatchTestData<TenElemT>(rank, mpi_size);
  const double r_pinv = 1e-12;
  const double a_pinv = 0.0;
  const bool soft_cutoff = true;

  auto y = MinSREigenSolveDispatch<TenElemT>(
      data.T_rowblock.data(), data.ns_local, data.ns_global,
      data.rhs, comm, r_pinv, a_pinv, soft_cutoff,
      MinSRSolverMode::kReplicated);

  ASSERT_EQ(y.size(), data.ns_global);
  // y should be non-trivial (not all zeros)
  double y_norm = 0.0;
  for (const auto &val : y) { y_norm += std::norm(val); }
  EXPECT_GT(y_norm, 0.0) << "Dispatch returned zero vector";
}

/// kDistributed mode throws when ScaLAPACK is not compiled in;
/// produces correct result (matching Path B) when ScaLAPACK IS available.
TEST_F(DispatchTest, DistributedModeDispatch) {
  auto data = BuildDispatchTestData<TenElemT>(rank, mpi_size);
#ifndef QLPEPS_HAS_SCALAPACK
  EXPECT_THROW(
      MinSREigenSolveDispatch<TenElemT>(
          data.T_rowblock.data(), data.ns_local, data.ns_global,
          data.rhs, comm, 1e-12, 0.0, true,
          MinSRSolverMode::kDistributed),
      std::runtime_error);
#else
  const double r_pinv = 1e-12;
  const double a_pinv = 0.0;
  const bool soft_cutoff = true;

  // Path B reference
  auto y_replicated = MinSREigenSolveDispatch<TenElemT>(
      data.T_rowblock.data(), data.ns_local, data.ns_global,
      data.rhs, comm, r_pinv, a_pinv, soft_cutoff,
      MinSRSolverMode::kReplicated);

  // Path A (ScaLAPACK)
  auto y_distributed = MinSREigenSolveDispatch<TenElemT>(
      data.T_rowblock.data(), data.ns_local, data.ns_global,
      data.rhs, comm, r_pinv, a_pinv, soft_cutoff,
      MinSRSolverMode::kDistributed);

  ASSERT_EQ(y_replicated.size(), y_distributed.size());
  for (size_t i = 0; i < y_replicated.size(); ++i) {
    // Different eigensolvers (LAPACK vs ScaLAPACK) so allow small numerical difference
    EXPECT_NEAR(std::abs(y_distributed[i] - y_replicated[i]), 0.0,
                1e-10 * std::max(std::abs(y_replicated[i]), 1e-15))
        << "Path A and Path B differ at index " << i;
  }
#endif
}

/// kAuto mode falls back to Path B when ScaLAPACK is absent (or Ns <= 5000).
/// Result must match kReplicated exactly (same code path).
TEST_F(DispatchTest, AutoModeFallsBackToPathB) {
  auto data = BuildDispatchTestData<TenElemT>(rank, mpi_size);
  const double r_pinv = 1e-12;
  const double a_pinv = 0.0;
  const bool soft_cutoff = true;

  auto y_replicated = MinSREigenSolveDispatch<TenElemT>(
      data.T_rowblock.data(), data.ns_local, data.ns_global,
      data.rhs, comm, r_pinv, a_pinv, soft_cutoff,
      MinSRSolverMode::kReplicated);

  auto y_auto = MinSREigenSolveDispatch<TenElemT>(
      data.T_rowblock.data(), data.ns_local, data.ns_global,
      data.rhs, comm, r_pinv, a_pinv, soft_cutoff,
      MinSRSolverMode::kAuto);

  ASSERT_EQ(y_replicated.size(), y_auto.size());
  for (size_t i = 0; i < y_replicated.size(); ++i) {
    EXPECT_EQ(y_replicated[i], y_auto[i])
        << "kAuto and kReplicated differ at index " << i;
  }
}

/// Verify pseudo-inverse cutoff modes: hard cutoff with zero tolerance = exact inverse.
/// T * y should equal rhs (for a well-conditioned matrix).
TEST_F(DispatchTest, HardCutoffExactInverse) {
  auto data = BuildDispatchTestData<TenElemT>(rank, mpi_size);

  // Hard cutoff with zero tolerance → exact inverse of all eigenvalues
  auto y = MinSREigenSolveDispatch<TenElemT>(
      data.T_rowblock.data(), data.ns_local, data.ns_global,
      data.rhs, comm, /*r_pinv=*/0.0, /*a_pinv=*/0.0, /*soft_cutoff=*/false,
      MinSRSolverMode::kReplicated);

  // Verify: T * y ≈ rhs (using full T on all ranks)
  const size_t N = data.ns_global;
  double residual_norm = 0.0;
  double rhs_norm = 0.0;
  for (size_t i = 0; i < N; ++i) {
    TenElemT Ty_i(0);
    for (size_t j = 0; j < N; ++j) {
      Ty_i += data.T_full[i * N + j] * y[j];
    }
    residual_norm += std::norm(Ty_i - data.rhs[i]);
    rhs_norm += std::norm(data.rhs[i]);
  }
  residual_norm = std::sqrt(residual_norm);
  rhs_norm = std::sqrt(rhs_norm);
  const double rel_residual = residual_norm / std::max(rhs_norm, 1e-15);

  if (rank == 0) {
    std::cout << "[Dispatch] Hard cutoff residual: ||T*y - rhs|| / ||rhs|| = "
              << rel_residual << std::endl;
  }
  EXPECT_LT(rel_residual, 1e-10)
      << "T * y should closely approximate rhs with exact inverse";
}

/// Path A exact-inverse test: T * y should equal rhs when ScaLAPACK is available.
#ifdef QLPEPS_HAS_SCALAPACK
TEST_F(DispatchTest, HardCutoffExactInverseDistributed) {
  auto data = BuildDispatchTestData<TenElemT>(rank, mpi_size);

  auto y = MinSREigenSolveDispatch<TenElemT>(
      data.T_rowblock.data(), data.ns_local, data.ns_global,
      data.rhs, comm, /*r_pinv=*/0.0, /*a_pinv=*/0.0, /*soft_cutoff=*/false,
      MinSRSolverMode::kDistributed);

  const size_t N = data.ns_global;
  double residual_norm = 0.0;
  double rhs_norm = 0.0;
  for (size_t i = 0; i < N; ++i) {
    TenElemT Ty_i(0);
    for (size_t j = 0; j < N; ++j) {
      Ty_i += data.T_full[i * N + j] * y[j];
    }
    residual_norm += std::norm(Ty_i - data.rhs[i]);
    rhs_norm += std::norm(data.rhs[i]);
  }
  residual_norm = std::sqrt(residual_norm);
  rhs_norm = std::sqrt(rhs_norm);
  const double rel_residual = residual_norm / std::max(rhs_norm, 1e-15);

  if (rank == 0) {
    std::cout << "[Dispatch/PathA] Hard cutoff residual: ||T*y - rhs|| / ||rhs|| = "
              << rel_residual << std::endl;
  }
  EXPECT_LT(rel_residual, 1e-10)
      << "ScaLAPACK path: T * y should closely approximate rhs with exact inverse";
}
#endif

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
