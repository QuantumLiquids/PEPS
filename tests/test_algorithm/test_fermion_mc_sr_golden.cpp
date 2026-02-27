// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-11
*
* Description: Golden regression test for fermionic MC gradient + SR natural gradient.
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

template<typename SITPST, typename TenElemT, typename QNT>
std::complex<double> RandomProbeInnerProduct(const SITPST &x, unsigned int seed) {
  SITPST probe = x;
  qlten::SetRandomSeed(seed);
  for (size_t row = 0; row < probe.rows(); ++row) {
    for (size_t col = 0; col < probe.cols(); ++col) {
      const size_t phy_dim = probe({row, col}).size();
      for (size_t i = 0; i < phy_dim; ++i) {
        auto &ten = probe({row, col})[i];
        if (ten.IsDefault()) { continue; }
        ten.Random(QNT(0));
      }
    }
  }
  const TenElemT ip = x * probe;
  return {std::real(ip), std::imag(ip)};
}

template<typename TenElemT>
void PrintCurrentValues(const char *label,
                        const TenElemT &energy,
                        double grad_norm,
                        const std::complex<double> &grad_probe,
                        const std::complex<double> &grad_probe_rand1,
                        const std::complex<double> &grad_probe_rand2,
                        double nat_grad_norm,
                        const std::complex<double> &nat_grad_probe,
                        const std::complex<double> &nat_grad_probe_rand1,
                        const std::complex<double> &nat_grad_probe_rand2,
                        size_t cg_iters) {
  std::cout << "[GOLDEN-CANDIDATE][" << label << "] "
            << "energy=(" << std::real(energy) << "," << std::imag(energy) << ") "
            << "grad_norm=" << grad_norm << " "
            << "grad_probe=(" << grad_probe.real() << "," << grad_probe.imag() << ") "
            << "grad_probe_rand1=(" << grad_probe_rand1.real() << "," << grad_probe_rand1.imag() << ") "
            << "grad_probe_rand2=(" << grad_probe_rand2.real() << "," << grad_probe_rand2.imag() << ") "
            << "nat_grad_norm=" << nat_grad_norm << " "
            << "nat_grad_probe=(" << nat_grad_probe.real() << "," << nat_grad_probe.imag() << ") "
            << "nat_grad_probe_rand1=(" << nat_grad_probe_rand1.real() << "," << nat_grad_probe_rand1.imag() << ") "
            << "nat_grad_probe_rand2=(" << nat_grad_probe_rand2.real() << "," << nat_grad_probe_rand2.imag() << ") "
            << "cg_iters=" << cg_iters
            << std::endl;
}

} // namespace

struct FermionMCSRGoldenTest : MPITest {
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

TEST_F(FermionMCSRGoldenTest, MCEnergyGradientAndSRGolden) {
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
  ASSERT_EQ(eval_res.Ostar_samples.size(), kSamples);
  if (rank != qlten::hp_numeric::kMPIMasterRank) { return; }

  ConjugateGradientParams cg_params(/*max_iter=*/32, /*relative_tolerance=*/1e-5, /*residue_restart_step=*/8, /*diag_shift=*/1e-3);
  OptimizerParams sr_params = OptimizerFactory::CreateStochasticReconfigurationAdvanced(
      /*max_iterations=*/1, /*energy_tolerance=*/1e-30, /*gradient_tolerance=*/1e-30,
      /*plateau_patience=*/1, cg_params, /*learning_rate=*/0.1);
  Optimizer<TenElemT, QNT> optimizer(sr_params, comm, rank, mpi_size);

  SITPST init_guess = eval_res.gradient * TenElemT(0.0);
  auto [nat_grad, cg_iters] = optimizer.CalculateNaturalGradient(
      eval_res.gradient, eval_res.Ostar_samples, eval_res.Ostar_mean.value(), init_guess);

  const double grad_norm = eval_res.gradient.NormSquare();
  const std::complex<double> grad_probe = WeightedProbeInnerProduct<SITPST, TenElemT>(eval_res.gradient);
  const std::complex<double> grad_probe_rand1 =
      RandomProbeInnerProduct<SITPST, TenElemT, QNT>(eval_res.gradient, /*seed=*/1337U);
  const std::complex<double> grad_probe_rand2 =
      RandomProbeInnerProduct<SITPST, TenElemT, QNT>(eval_res.gradient, /*seed=*/424242U);
  const double nat_grad_norm = nat_grad.NormSquare();
  const std::complex<double> nat_grad_probe = WeightedProbeInnerProduct<SITPST, TenElemT>(nat_grad);
  const std::complex<double> nat_grad_probe_rand1 =
      RandomProbeInnerProduct<SITPST, TenElemT, QNT>(nat_grad, /*seed=*/7331U);
  const std::complex<double> nat_grad_probe_rand2 =
      RandomProbeInnerProduct<SITPST, TenElemT, QNT>(nat_grad, /*seed=*/900001U);
  constexpr bool kPrintGolden = false;
  if (kPrintGolden) {
    PrintCurrentValues("mc+sr",
                       eval_res.energy,
                       grad_norm,
                       grad_probe,
                       grad_probe_rand1,
                       grad_probe_rand2,
                       nat_grad_norm,
                       nat_grad_probe,
                       nat_grad_probe_rand1,
                       nat_grad_probe_rand2,
                       cg_iters);
  }

  // Golden values are captured from the current implementation before refactor.
  if constexpr (std::is_same_v<TenElemT, QLTEN_Double>) {
    constexpr double kEnergyReal = 0.6763170558260847;
    constexpr double kEnergyImag = 0.0;
    constexpr double kGradNorm = 10.430271873006413;
    constexpr double kGradProbeReal = 3.1712271031834121;
    constexpr double kGradProbeImag = 0.0;
    constexpr double kGradProbeRand1Real = 0.85021927432696276;
    constexpr double kGradProbeRand1Imag = 0.0;
    constexpr double kGradProbeRand2Real = -0.7299325373645108;
    constexpr double kGradProbeRand2Imag = 0.0;
    constexpr double kNatGradNorm = 0.47405741880181806;
    constexpr double kNatGradProbeReal = 0.15357967746815987;
    constexpr double kNatGradProbeImag = 0.0;
    constexpr double kNatGradProbeRand1Real = 0.42345437198122371;
    constexpr double kNatGradProbeRand1Imag = 0.0;
    constexpr double kNatGradProbeRand2Real = 0.078505173196347616;
    constexpr double kNatGradProbeRand2Imag = 0.0;
    constexpr size_t kCGIters = 3;
    EXPECT_NEAR(std::real(eval_res.energy), kEnergyReal, 1e-9);
    EXPECT_NEAR(std::imag(eval_res.energy), kEnergyImag, 1e-12);
    EXPECT_NEAR(grad_norm, kGradNorm, 1e-8);
    EXPECT_NEAR(grad_probe.real(), kGradProbeReal, 1e-8);
    EXPECT_NEAR(grad_probe.imag(), kGradProbeImag, 1e-12);
    EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real, 1e-8);
    EXPECT_NEAR(grad_probe_rand1.imag(), kGradProbeRand1Imag, 1e-8);
    EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real, 1e-8);
    EXPECT_NEAR(grad_probe_rand2.imag(), kGradProbeRand2Imag, 1e-8);
    EXPECT_NEAR(nat_grad_norm, kNatGradNorm, 1e-8);
    EXPECT_NEAR(nat_grad_probe.real(), kNatGradProbeReal, 1e-8);
    EXPECT_NEAR(nat_grad_probe.imag(), kNatGradProbeImag, 1e-12);
    EXPECT_NEAR(nat_grad_probe_rand1.real(), kNatGradProbeRand1Real, 1e-8);
    EXPECT_NEAR(nat_grad_probe_rand1.imag(), kNatGradProbeRand1Imag, 1e-8);
    EXPECT_NEAR(nat_grad_probe_rand2.real(), kNatGradProbeRand2Real, 1e-8);
    EXPECT_NEAR(nat_grad_probe_rand2.imag(), kNatGradProbeRand2Imag, 1e-8);
    EXPECT_EQ(cg_iters, kCGIters);
  } else {
    constexpr double kEnergyReal = -0.24725045025627607;
    constexpr double kEnergyImag = -0.051112577883968571;
    constexpr double kGradNorm = 5.7343207658246129;
    constexpr double kGradProbeReal = 1.6762821167546227;
    constexpr double kGradProbeImag = 0.039798421537668147;
    constexpr double kGradProbeRand1Real = 1.3214871877027676;
    constexpr double kGradProbeRand1Imag = 1.3270548260829671;
    constexpr double kGradProbeRand2Real = 0.64293983466084448;
    constexpr double kGradProbeRand2Imag = 0.89090276250115463;
    constexpr double kNatGradNorm = 1.3733177098359004;
    constexpr double kNatGradProbeReal = 0.4109995632211681;
    constexpr double kNatGradProbeImag = 0.0096634784101417917;
    constexpr double kNatGradProbeRand1Real = 0.41021773922907417;
    constexpr double kNatGradProbeRand1Imag = -0.2306855400810065;
    constexpr double kNatGradProbeRand2Real = 0.57092656940884234;
    constexpr double kNatGradProbeRand2Imag = -0.035644329416899795;
    constexpr size_t kCGIters = 2;
    EXPECT_NEAR(std::real(eval_res.energy), kEnergyReal, 1e-9);
    EXPECT_NEAR(std::imag(eval_res.energy), kEnergyImag, 1e-9);
    EXPECT_NEAR(grad_norm, kGradNorm, 1e-8);
    EXPECT_NEAR(grad_probe.real(), kGradProbeReal, 1e-8);
    EXPECT_NEAR(grad_probe.imag(), kGradProbeImag, 1e-8);
    EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real, 1e-8);
    EXPECT_NEAR(grad_probe_rand1.imag(), kGradProbeRand1Imag, 1e-8);
    EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real, 1e-8);
    EXPECT_NEAR(grad_probe_rand2.imag(), kGradProbeRand2Imag, 1e-8);
    EXPECT_NEAR(nat_grad_norm, kNatGradNorm, 1e-8);
    EXPECT_NEAR(nat_grad_probe.real(), kNatGradProbeReal, 1e-8);
    EXPECT_NEAR(nat_grad_probe.imag(), kNatGradProbeImag, 1e-8);
    EXPECT_NEAR(nat_grad_probe_rand1.real(), kNatGradProbeRand1Real, 1e-8);
    EXPECT_NEAR(nat_grad_probe_rand1.imag(), kNatGradProbeRand1Imag, 1e-8);
    EXPECT_NEAR(nat_grad_probe_rand2.real(), kNatGradProbeRand2Real, 1e-8);
    EXPECT_NEAR(nat_grad_probe_rand2.imag(), kNatGradProbeRand2Imag, 1e-8);
    EXPECT_EQ(cg_iters, kCGIters);
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
