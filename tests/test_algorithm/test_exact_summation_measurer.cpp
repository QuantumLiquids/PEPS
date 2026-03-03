/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-03-02
*
* Description: QuantumLiquids/PEPS project.
* Exact summation measurer tests — golden-regress all observables
* from EvaluateObservables() for 4 OBC models on 2x2 lattice.
*
* Models tested:
*  1. Spinless free fermion (t=1, t2=0, V=0)
*  2. Heisenberg (J=1)
*  3. Transverse-field Ising (h=1)
*  4. t-J (t=1, J=0.3, V=J/4)
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_measurer.h"
#include "../test_mpi_env.h"
#include <cmath>
#include <limits>
#include <set>

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::fZ2QN;
using qlten::special_qn::TrivialRepQN;

double Calculate2x2OBCTransverseIsingEnergy(double J, double h) {
  const double pi = std::acos(-1.0);
  std::vector<double> k_values = {pi / 4.0, 3.0 * pi / 4.0};
  double ground_state_energy = 0.0;
  for (auto k : k_values) {
    const double epsilon_k = 2.0 * std::sqrt(J * J + h * h - 2.0 * J * h * std::cos(k));
    ground_state_energy -= epsilon_k;
  }
  return ground_state_energy;
}

// ============================================================================
// Helper: print observable map for golden capture
// ============================================================================
template<typename TenElemT>
void PrintObservableMap(const ObservableMap<TenElemT> &obs, const std::string &prefix) {
  // Sort keys for deterministic output
  std::vector<std::string> keys;
  for (const auto &[key, values] : obs) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  for (const auto &key : keys) {
    const auto &values = obs.at(key);
    std::cout << "[GOLDEN][" << prefix << "] " << key << " (" << values.size() << " values):";
    for (size_t i = 0; i < values.size(); ++i) {
      if constexpr (std::is_same_v<TenElemT, QLTEN_Double>) {
        std::cout << "\n  [" << i << "] = " << std::setprecision(16) << values[i];
      } else {
        std::cout << "\n  [" << i << "] = (" << std::setprecision(16)
                  << values[i].real() << ", " << values[i].imag() << ")";
      }
    }
    std::cout << std::endl;
  }
}

// ============================================================================
// Helpers: assert observable values match golden
// ============================================================================

/// Assert real parts match expected golden values.
template<typename TenElemT>
void AssertObservableNear(const ObservableMap<TenElemT> &obs,
                          const std::string &key,
                          const std::vector<double> &expected,
                          double tol,
                          const std::string &context) {
  auto it = obs.find(key);
  ASSERT_NE(it, obs.end()) << context << ": missing key '" << key << "'";
  const auto &values = it->second;
  ASSERT_EQ(values.size(), expected.size())
      << context << ": key '" << key << "' size mismatch";
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(std::real(values[i]), expected[i], tol)
        << context << ": Re key '" << key << "' [" << i << "]";
  }
}

/// Assert imaginary parts are near zero (for exact summation with real Hamiltonians).
template<typename TenElemT>
void AssertObservableImagNearZero(const ObservableMap<TenElemT> &obs,
                                  const std::string &key,
                                  double tol,
                                  const std::string &context) {
  if constexpr (std::is_same_v<TenElemT, QLTEN_Double>) {
    return;  // No imaginary part for real types
  } else {
    auto it = obs.find(key);
    ASSERT_NE(it, obs.end()) << context << ": missing key '" << key << "'";
    for (size_t i = 0; i < it->second.size(); ++i) {
      EXPECT_NEAR(std::imag(it->second[i]), 0.0, tol)
          << context << ": Im key '" << key << "' [" << i << "] expected near-zero";
    }
  }
}

template<typename TenElemT>
void AssertObservableKeySet(const ObservableMap<TenElemT> &obs,
                            const std::set<std::string> &expected_keys,
                            const std::string &context) {
  std::set<std::string> actual_keys;
  for (const auto &[key, _] : obs) {
    actual_keys.insert(key);
  }
  EXPECT_EQ(actual_keys, expected_keys) << context << ": observable key set mismatch";
}

template<typename TenElemT>
double SumObservableValues(const ObservableMap<TenElemT> &obs,
                           const std::string &key,
                           const std::string &context) {
  auto it = obs.find(key);
  if (it == obs.end()) {
    ADD_FAILURE() << context << ": missing key '" << key << "'";
    return std::numeric_limits<double>::quiet_NaN();
  }
  double sum = 0.0;
  for (const auto &v : it->second) {
    sum += std::real(v);
  }
  return sum;
}

template<typename TenElemT>
void AssertObservableMapNear(const ObservableMap<TenElemT> &actual,
                             const ObservableMap<TenElemT> &expected,
                             double tol,
                             const std::string &context) {
  std::set<std::string> expected_keys;
  for (const auto &[key, _] : expected) {
    expected_keys.insert(key);
  }
  AssertObservableKeySet(actual, expected_keys, context);
  for (const auto &[key, expected_values] : expected) {
    auto it = actual.find(key);
    ASSERT_NE(it, actual.end()) << context << ": missing key '" << key << "'";
    ASSERT_EQ(it->second.size(), expected_values.size())
        << context << ": key '" << key << "' size mismatch";
    for (size_t i = 0; i < expected_values.size(); ++i) {
      EXPECT_NEAR(std::real(it->second[i]), std::real(expected_values[i]), tol)
          << context << ": Re key '" << key << "' [" << i << "]";
      if constexpr (!std::is_same_v<TenElemT, QLTEN_Double>) {
        EXPECT_NEAR(std::imag(it->second[i]), std::imag(expected_values[i]), tol)
            << context << ": Im key '" << key << "' [" << i << "]";
      }
    }
  }
}

TEST(ExactSummationMeasurerHelperTest, GenerateAllBinaryConfigsRejectsTooManySites) {
  const size_t bit_width = std::numeric_limits<size_t>::digits;
  EXPECT_THROW((void)GenerateAllBinaryConfigs(bit_width, 1), std::invalid_argument);
}

// ============================================================================
// 1. Spinless Fermion Test
// ============================================================================
struct SpinlessFermionMeasurerTest : public MPITest {
  using SITPST = SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>;

  size_t Lx = 2, Ly = 2;
  double t = 1.0, t2 = 0.0, V = 0.0;
  SITPST sitps = SITPST(Ly, Lx);
  std::vector<Configuration> all_configs;

  SITPST LoadTPS(double t2_value, bool lowest_state) {
    const std::string suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>
                               ? (lowest_state ? "_doublelowest" : "_double_from_simple_update")
                               : (lowest_state ? "_complexlowest" : "_complex_from_simple_update");
    const std::string path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                             "spinless_fermion_tps_t2_" + std::to_string(t2_value) + suffix;
    SITPST target_sitps(Ly, Lx);
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!target_sitps.Load(path)) {
        throw std::runtime_error("Failed to load: " + path);
      }
    }
    qlpeps::MPI_Bcast(target_sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return target_sitps;
  }

  void SetUp() override {
    MPITest::SetUp();
    sitps = LoadTPS(/*t2_value=*/0.0, /*lowest_state=*/false);

    // Generate all C(4,2)=6 half-filling configs
    std::vector<size_t> cfg = {0, 0, 1, 1};
    do { all_configs.push_back(Vec2Config(cfg, Lx, Ly)); }
    while (std::next_permutation(cfg.begin(), cfg.end()));
  }
};

TEST_F(SpinlessFermionMeasurerTest, SimpleUpdateState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);
  SquareSpinlessFermion model(t, t2, V);

  auto obs = ExactSumMeasurerMPI<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN>(
      sitps, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr bool kPrintGolden = false;
    if (kPrintGolden) {
      PrintObservableMap(obs, "SpinlessFermion_SU");
    }

    constexpr double kTol = 1e-10;
    constexpr double kImagTol = 1e-10;
    const std::string ctx = "SpinlessFermion";
    AssertObservableKeySet(
        obs,
        {"energy", "charge", "bond_energy_h", "bond_energy_v", "bond_energy_dr", "bond_energy_ur"},
        ctx);
    const double energy_sum = SumObservableValues(obs, "energy", ctx);
    const double bond_sum = SumObservableValues(obs, "bond_energy_h", ctx) +
                            SumObservableValues(obs, "bond_energy_v", ctx) +
                            SumObservableValues(obs, "bond_energy_dr", ctx) +
                            SumObservableValues(obs, "bond_energy_ur", ctx);
    EXPECT_NEAR(energy_sum, bond_sum, kTol) << ctx << ": energy/bond decomposition mismatch";
    EXPECT_NEAR(SumObservableValues(obs, "charge", ctx), 2.0, kTol)
        << ctx << ": total charge mismatch";

    if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      AssertObservableNear(obs, "energy", {-1.98218053854462}, kTol, ctx);
      AssertObservableNear(obs, "charge",
          {0.438249672027182, 0.561398937748767, 0.4304163527242686, 0.5699350374997822}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_h",
          {-0.9919642003163891, -0.9897945508819544}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_v",
          {-0.01384822509079633, 0.0134264377445201}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_dr", {0.0}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_ur", {0.0}, kTol, ctx);
    } else {
      AssertObservableNear(obs, "energy", {-1.98218053854462}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
      AssertObservableNear(obs, "charge",
          {0.4382496720271818, 0.5613989377487671, 0.4304163527242685, 0.5699350374997826}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "charge", kImagTol, ctx);
      AssertObservableNear(obs, "bond_energy_h",
          {-0.9919642003163892, -0.9897945508819544}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "bond_energy_h", kImagTol, ctx);
      AssertObservableNear(obs, "bond_energy_v",
          {-0.01384822509079676, 0.01342643774452053}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "bond_energy_v", kImagTol, ctx);
      AssertObservableNear(obs, "bond_energy_dr", {0.0}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_ur", {0.0}, kTol, ctx);
    }
  }
}

TEST_F(SpinlessFermionMeasurerTest, RejectsEmptyConfigList) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  const auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);
  SquareSpinlessFermion model(t, t2, V);
  std::vector<Configuration> empty_configs;
  auto invoke_empty = [&]() {
    (void)ExactSumMeasurerMPI<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN>(
        sitps, empty_configs, trun_para, model, Ly, Lx, comm, /*rank=*/0, /*mpi_size=*/1);
  };
  EXPECT_THROW(invoke_empty(), std::invalid_argument);
}

TEST_F(SpinlessFermionMeasurerTest, SingleConfigParallelMatchesSerialReference) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  const auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);
  SquareSpinlessFermion model(t, t2, V);
  const std::vector<Configuration> one_config = {all_configs.front()};

  auto obs_parallel = ExactSumMeasurerMPI<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN>(
      sitps, one_config, trun_para, model, Ly, Lx, comm, rank, mpi_size);
  auto obs_serial = ExactSumMeasurerMPI<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN>(
      sitps, one_config, trun_para, model, Ly, Lx, comm, /*rank=*/0, /*mpi_size=*/1);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr double kTol = 1e-12;
    AssertObservableMapNear(obs_parallel, obs_serial, kTol, "SpinlessFermion_OneConfig_MPI");
  }
}

TEST_F(SpinlessFermionMeasurerTest, SimpleUpdateStateT2Sweep) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  const auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);

  const std::vector<double> t2_list = {2.1, 0.0, -2.5};
  constexpr double kTol = 1e-8;
  constexpr double kImagTol = 1e-10;
  const std::set<std::string> expected_keys = {
      "energy", "charge", "bond_energy_h", "bond_energy_v", "bond_energy_dr", "bond_energy_ur"};

  for (size_t i = 0; i < t2_list.size(); ++i) {
    const double t2_cur = t2_list[i];
    auto sitps_cur = LoadTPS(t2_cur, /*lowest_state=*/false);
    SquareSpinlessFermion model(t, t2_cur, V);
    auto obs = ExactSumMeasurerMPI<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN>(
        sitps_cur, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      const std::string ctx = "SpinlessFermion_SimpleUpdate_t2=" + std::to_string(t2_cur);
      AssertObservableKeySet(obs, expected_keys, ctx);
      if constexpr (!std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
        AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
      }
      const double energy_sum = SumObservableValues(obs, "energy", ctx);
      EXPECT_TRUE(std::isfinite(energy_sum)) << ctx << ": non-finite energy";
      const double bond_sum = SumObservableValues(obs, "bond_energy_h", ctx) +
                              SumObservableValues(obs, "bond_energy_v", ctx) +
                              SumObservableValues(obs, "bond_energy_dr", ctx) +
                              SumObservableValues(obs, "bond_energy_ur", ctx);
      EXPECT_NEAR(energy_sum, bond_sum, kTol) << ctx << ": energy/bond decomposition mismatch";
      EXPECT_NEAR(SumObservableValues(obs, "charge", ctx), 2.0, kTol)
          << ctx << ": total charge mismatch";
    }
  }
}

TEST_F(SpinlessFermionMeasurerTest, LowestStateT2SweepEnergy) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  const auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);

  const std::vector<double> t2_list = {2.1, 0.0, -2.5};
  constexpr double kTol = 1e-8;
  constexpr double kImagTol = 1e-10;
  const std::set<std::string> expected_keys = {
      "energy", "charge", "bond_energy_h", "bond_energy_v", "bond_energy_dr", "bond_energy_ur"};

  for (double t2_cur : t2_list) {
    auto sitps_cur = LoadTPS(t2_cur, /*lowest_state=*/true);
    SquareSpinlessFermion model(t, t2_cur, V);
    auto obs = ExactSumMeasurerMPI<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN>(
        sitps_cur, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      const std::string ctx = "SpinlessFermion_Lowest_t2=" + std::to_string(t2_cur);
      AssertObservableKeySet(obs, expected_keys, ctx);
      if constexpr (!std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
        AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
      }
      const double energy_sum = SumObservableValues(obs, "energy", ctx);
      EXPECT_TRUE(std::isfinite(energy_sum)) << ctx << ": non-finite energy";
      const double bond_sum = SumObservableValues(obs, "bond_energy_h", ctx) +
                              SumObservableValues(obs, "bond_energy_v", ctx) +
                              SumObservableValues(obs, "bond_energy_dr", ctx) +
                              SumObservableValues(obs, "bond_energy_ur", ctx);
      EXPECT_NEAR(energy_sum, bond_sum, kTol) << ctx << ": energy/bond decomposition mismatch";
      EXPECT_NEAR(SumObservableValues(obs, "charge", ctx), 2.0, kTol)
          << ctx << ": total charge mismatch";
    }
  }
}

TEST_F(SpinlessFermionMeasurerTest, LowestStateObservables) {
  // Use t2=2.1 to lift the ground-state degeneracy of the t2=0 free fermion.
  // At t2=0 the single-particle spectrum is {-2,0,0,2}; the two zero-energy
  // levels make the half-filled GS degenerate, so per-site observables are
  // ambiguous. Nonzero t2 splits these levels, giving a unique GS.
  // ED (QuSpin): E=-4.2, charge=0.5 uniform, NN bond=0, NNN bond=-2.1.
  constexpr double kTestT2 = 2.1;
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  const auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);
  auto lowest_sitps = LoadTPS(/*t2_value=*/kTestT2, /*lowest_state=*/true);
  SquareSpinlessFermion model(t, kTestT2, V);

  auto obs = ExactSumMeasurerMPI<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN>(
      lowest_sitps, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr double kTol = 6e-8;
    constexpr double kObsTol = 1e-5;
    constexpr double kImagTol = 1e-10;
    const std::string ctx = "SpinlessFermion_Lowest_t2=2.1";
    AssertObservableKeySet(
        obs,
        {"energy", "charge", "bond_energy_h", "bond_energy_v", "bond_energy_dr", "bond_energy_ur"},
        ctx);
    // ED energy and observable checks (QuSpin ground-state references)
    AssertObservableNear(obs, "energy", {-4.2}, kTol, ctx);
    if constexpr (!std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
    }
    AssertObservableNear(obs, "charge", {0.5, 0.5, 0.5, 0.5}, kObsTol, ctx);
    AssertObservableNear(obs, "bond_energy_h", {0.0, 0.0}, kObsTol, ctx);
    AssertObservableNear(obs, "bond_energy_v", {0.0, 0.0}, kObsTol, ctx);
    AssertObservableNear(obs, "bond_energy_dr", {-kTestT2}, kObsTol, ctx);
    AssertObservableNear(obs, "bond_energy_ur", {-kTestT2}, kObsTol, ctx);
    EXPECT_NEAR(SumObservableValues(obs, "charge", ctx), 2.0, kTol)
        << ctx << ": total charge mismatch";
    const double energy_sum = SumObservableValues(obs, "energy", ctx);
    const double bond_sum = SumObservableValues(obs, "bond_energy_h", ctx) +
                            SumObservableValues(obs, "bond_energy_v", ctx) +
                            SumObservableValues(obs, "bond_energy_dr", ctx) +
                            SumObservableValues(obs, "bond_energy_ur", ctx);
    EXPECT_NEAR(energy_sum, bond_sum, kTol) << ctx << ": energy/bond decomposition mismatch";
  }
}

// ============================================================================
// 2. Heisenberg Test
// ============================================================================
struct HeisenbergMeasurerTest : public MPITest {
  using SITPST = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>;

  size_t Lx = 2, Ly = 2;
  double J = 1.0;
  SITPST sitps = SITPST(Ly, Lx);
  std::vector<Configuration> all_configs;

  SITPST LoadTPS(bool lowest_state) {
    const std::string suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>
                               ? (lowest_state ? "_doublelowest" : "_double_from_simple_update")
                               : (lowest_state ? "_complexlowest" : "_complex_from_simple_update");
    const std::string path = std::string(TEST_SOURCE_DIR) + "/test_data/" + "heisenberg_tps" + suffix;
    SITPST target_sitps(Ly, Lx);
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!target_sitps.Load(path)) {
        throw std::runtime_error("Failed to load: " + path);
      }
    }
    qlpeps::MPI_Bcast(target_sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return target_sitps;
  }

  void SetUp() override {
    MPITest::SetUp();
    sitps = LoadTPS(/*lowest_state=*/false);

    std::vector<size_t> cfg = {0, 0, 1, 1};
    do { all_configs.push_back(Vec2Config(cfg, Lx, Ly)); }
    while (std::next_permutation(cfg.begin(), cfg.end()));
  }
};

TEST_F(HeisenbergMeasurerTest, SimpleUpdateState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  SquareSpinOneHalfXXZModelOBC model(J, J, 0);

  auto obs = ExactSumMeasurerMPI<SquareSpinOneHalfXXZModelOBC, TEN_ELEM_TYPE, TrivialRepQN>(
      sitps, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr bool kPrintGolden = false;
    if (kPrintGolden) {
      PrintObservableMap(obs, "Heisenberg_SU");
    }

    constexpr double kTol = 1e-10;
    constexpr double kImagTol = 1e-10;
    const std::string ctx = "Heisenberg";
    AssertObservableKeySet(
        obs,
        {"energy", "spin_z", "bond_energy_h", "bond_energy_v", "SzSz_all2all", "SmSp_row", "SpSm_row"},
        ctx);
    const double energy_sum = SumObservableValues(obs, "energy", ctx);
    const double bond_sum = SumObservableValues(obs, "bond_energy_h", ctx) +
                            SumObservableValues(obs, "bond_energy_v", ctx);
    EXPECT_NEAR(energy_sum, bond_sum, kTol) << ctx << ": energy/bond decomposition mismatch";

    if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      AssertObservableNear(obs, "energy", {-1.995212787934525}, kTol, ctx);
      AssertObservableNear(obs, "spin_z",
          {-0.02429911059357229, 0.02429911059357257, 0.02429911059357496, -0.02429911059357522}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_h",
          {-0.5361159973934749, -0.5361159973934747}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_v",
          {-0.4614903965737877, -0.4614903965737878}, kTol, ctx);
      AssertObservableNear(obs, "SzSz_all2all",
          {0.25, -0.1805756774232915, -0.1559777039119346, 0.08655338133522604,
           0.25, 0.08655338133522604, -0.1559777039119346, 0.25,
           -0.1805756774232915, 0.25}, kTol, ctx);
      AssertObservableNear(obs, "SmSp_row", {-0.3555403199701831}, kTol, ctx);
      AssertObservableNear(obs, "SpSm_row", {-0.3555403199701833}, kTol, ctx);
    } else {
      AssertObservableNear(obs, "energy", {-1.995212787934525}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
      AssertObservableNear(obs, "spin_z",
          {-0.02429911059357229, 0.02429911059357256, 0.02429911059357492, -0.02429911059357518}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "spin_z", kImagTol, ctx);
      AssertObservableNear(obs, "bond_energy_h",
          {-0.5361159973934748, -0.5361159973934746}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "bond_energy_h", kImagTol, ctx);
      AssertObservableNear(obs, "bond_energy_v",
          {-0.4614903965737877, -0.4614903965737876}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "bond_energy_v", kImagTol, ctx);
      AssertObservableNear(obs, "SzSz_all2all",
          {0.25, -0.1805756774232915, -0.1559777039119345, 0.08655338133522601,
           0.25, 0.08655338133522601, -0.1559777039119345, 0.25,
           -0.1805756774232915, 0.25}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "SzSz_all2all", kImagTol, ctx);
      AssertObservableNear(obs, "SmSp_row", {-0.3555403199701831}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "SmSp_row", kImagTol, ctx);
      AssertObservableNear(obs, "SpSm_row", {-0.3555403199701834}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "SpSm_row", kImagTol, ctx);
    }
  }
}

TEST_F(HeisenbergMeasurerTest, LowestStateEnergy) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  const auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  auto lowest_sitps = LoadTPS(/*lowest_state=*/true);
  SquareSpinOneHalfXXZModelOBC model(J, J, 0);

  auto obs = ExactSumMeasurerMPI<SquareSpinOneHalfXXZModelOBC, TEN_ELEM_TYPE, TrivialRepQN>(
      lowest_sitps, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr double kTol = 6e-8;
    constexpr double kImagTol = 1e-10;
    constexpr double kObsTol = 1e-5;
    const std::string ctx = "Heisenberg_Lowest";
    AssertObservableKeySet(
        obs,
        {"energy", "spin_z", "bond_energy_h", "bond_energy_v", "SzSz_all2all", "SmSp_row", "SpSm_row"},
        ctx);
    AssertObservableNear(obs, "energy", {-2.0 * J}, kTol, ctx);
    if constexpr (!std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
    }
    // ED observable checks (QuSpin ground-state references)
    // spin_z is zero by singlet symmetry; TPS may soft-break this,
    // so use a looser tolerance (wavefunction error is linear, not quadratic).
    constexpr double kSymTol = 5e-4;
    AssertObservableNear(obs, "spin_z", {0.0, 0.0, 0.0, 0.0}, kSymTol, ctx);
    AssertObservableNear(obs, "bond_energy_h", {-0.5, -0.5}, kObsTol, ctx);
    AssertObservableNear(obs, "bond_energy_v", {-0.5, -0.5}, kObsTol, ctx);
    AssertObservableNear(obs, "SzSz_all2all",
        {0.25, -1.0 / 6.0, -1.0 / 6.0, 1.0 / 12.0,
         0.25, 1.0 / 12.0, -1.0 / 6.0, 0.25,
         -1.0 / 6.0, 0.25},
        kObsTol, ctx);
    AssertObservableNear(obs, "SmSp_row", {-1.0 / 3.0}, kObsTol, ctx);
    AssertObservableNear(obs, "SpSm_row", {-1.0 / 3.0}, kObsTol, ctx);
  }
}

// ============================================================================
// 3. Transverse-field Ising Test
// ============================================================================
struct TFIMMeasurerTest : public MPITest {
  using SITPST = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>;

  size_t Lx = 2, Ly = 2;
  double h = 1.0;
  SITPST sitps = SITPST(Ly, Lx);
  std::vector<Configuration> all_configs;

  SITPST LoadTPS(bool lowest_state) {
    const std::string suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>
                               ? (lowest_state ? "_doublelowest" : "_double_from_simple_update")
                               : (lowest_state ? "_complexlowest" : "_complex_from_simple_update");
    const std::string path = std::string(TEST_SOURCE_DIR) + "/test_data/" + "transverse_ising_tps" + suffix;
    SITPST target_sitps(Ly, Lx);
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!target_sitps.Load(path)) {
        throw std::runtime_error("Failed to load: " + path);
      }
    }
    qlpeps::MPI_Bcast(target_sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return target_sitps;
  }

  void SetUp() override {
    MPITest::SetUp();
    sitps = LoadTPS(/*lowest_state=*/false);

    // Generate all 2^4 = 16 configs (TFIM has no conserved Sz)
    all_configs = GenerateAllBinaryConfigs(Lx, Ly);
  }
};

TEST_F(TFIMMeasurerTest, SimpleUpdateState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  TransverseFieldIsingSquareOBC model(h);

  auto obs = ExactSumMeasurerMPI<TransverseFieldIsingSquareOBC, TEN_ELEM_TYPE, TrivialRepQN>(
      sitps, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr bool kPrintGolden = false;
    if (kPrintGolden) {
      PrintObservableMap(obs, "TFIM_SU");
    }

    constexpr double kTol = 1e-10;
    constexpr double kImagTol = 1e-10;
    const std::string ctx = "TFIM";
    AssertObservableKeySet(obs, {"energy", "spin_z", "sigma_x", "SzSz_row"}, ctx);

    if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      AssertObservableNear(obs, "energy", {-5.199919952278064}, kTol, ctx);
      AssertObservableNear(obs, "spin_z",
          {-0.1966962357606675, -0.1966962357606675, -0.1966962357606674, -0.1966962357606675}, kTol, ctx);
      AssertObservableNear(obs, "sigma_x",
          {0.6378595085591251, 0.6378595085591244, 0.6378595085591242, 0.6378595085591238}, kTol, ctx);
      AssertObservableNear(obs, "SzSz_row", {0.1690025253051581}, kTol, ctx);
    } else {
      AssertObservableNear(obs, "energy", {-5.199919952278063}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
      AssertObservableNear(obs, "spin_z",
          {-0.1966962357606671, -0.1966962357606672, -0.1966962357606669, -0.1966962357606671}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "spin_z", kImagTol, ctx);
      AssertObservableNear(obs, "sigma_x",
          {0.6378595085591251, 0.6378595085591244, 0.6378595085591247, 0.637859508559124}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "sigma_x", kImagTol, ctx);
      AssertObservableNear(obs, "SzSz_row", {0.169002525305158}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "SzSz_row", kImagTol, ctx);
    }
  }
}

TEST_F(TFIMMeasurerTest, LowestStateEnergy) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  const auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  auto lowest_sitps = LoadTPS(/*lowest_state=*/true);
  TransverseFieldIsingSquareOBC model(h);

  auto obs = ExactSumMeasurerMPI<TransverseFieldIsingSquareOBC, TEN_ELEM_TYPE, TrivialRepQN>(
      lowest_sitps, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr double kTol = 6e-8;
    constexpr double kImagTol = 1e-10;
    constexpr double kObsTol = 1e-5;
    const std::string ctx = "TFIM_Lowest";
    AssertObservableKeySet(obs, {"energy", "spin_z", "sigma_x", "SzSz_row"}, ctx);
    AssertObservableNear(obs, "energy", {Calculate2x2OBCTransverseIsingEnergy(1.0, h)}, kTol, ctx);
    if constexpr (!std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
    }
    // ED observable checks (QuSpin ground-state references)
    AssertObservableNear(obs, "spin_z", {0.0, 0.0, 0.0, 0.0}, kObsTol, ctx);
    AssertObservableNear(obs, "sigma_x",
        {0.6532814824381878, 0.6532814824381877,
         0.6532814824381881, 0.6532814824381881},
        kObsTol, ctx);
    AssertObservableNear(obs, "SzSz_row", {0.163320370609547}, kObsTol, ctx);
  }
}

// ============================================================================
// 4. t-J Test
// ============================================================================
struct tJMeasurerTest : public MPITest {
  using SITPST = SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>;

  size_t Lx = 2, Ly = 2;
  double t = 1.0, J = 0.3, V = 0.075, mu = 0.0;
  SITPST sitps = SITPST(Ly, Lx);
  std::vector<Configuration> all_configs;

  SITPST LoadTPS(bool lowest_state) {
    const std::string suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>
                               ? (lowest_state ? "_doublelowest" : "_double_from_simple_update")
                               : (lowest_state ? "_complexlowest" : "_complex_from_simple_update");
    const std::string path = std::string(TEST_SOURCE_DIR) + "/test_data/" + "tj_model_tps" + suffix;
    SITPST target_sitps(Ly, Lx);
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!target_sitps.Load(path)) {
        throw std::runtime_error("Failed to load: " + path);
      }
    }
    qlpeps::MPI_Bcast(target_sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return target_sitps;
  }

  void SetUp() override {
    MPITest::SetUp();
    sitps = LoadTPS(/*lowest_state=*/false);

    // Generate permutations of {0, 1, 2, 2} = 12 configs
    std::vector<size_t> cfg = {0, 1, 2, 2};
    std::sort(cfg.begin(), cfg.end());
    do { all_configs.push_back(Vec2Config(cfg, Lx, Ly)); }
    while (std::next_permutation(cfg.begin(), cfg.end()));
  }
};

TEST_F(tJMeasurerTest, SimpleUpdateState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(4, 4, 0);
  SquaretJVModel model(t, 0, J, V, mu);

  auto obs = ExactSumMeasurerMPI<SquaretJVModel, TEN_ELEM_TYPE, fZ2QN>(
      sitps, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr bool kPrintGolden = false;
    if (kPrintGolden) {
      PrintObservableMap(obs, "tJ_SU");
    }

    constexpr double kTol = 1e-10;
    constexpr double kImagTol = 1e-10;
    const std::string ctx = "tJ";
    AssertObservableKeySet(
        obs,
        {"energy", "spin_z", "charge", "bond_energy_h", "bond_energy_v", "bond_energy_dr", "bond_energy_ur"},
        ctx);
    const double energy_sum = SumObservableValues(obs, "energy", ctx);
    const double bond_sum = SumObservableValues(obs, "bond_energy_h", ctx) +
                            SumObservableValues(obs, "bond_energy_v", ctx) +
                            SumObservableValues(obs, "bond_energy_dr", ctx) +
                            SumObservableValues(obs, "bond_energy_ur", ctx);
    EXPECT_NEAR(energy_sum, bond_sum, kTol) << ctx << ": energy/bond decomposition mismatch";
    EXPECT_NEAR(SumObservableValues(obs, "charge", ctx), 2.0, kTol)
        << ctx << ": total charge mismatch";

    if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      AssertObservableNear(obs, "energy", {-2.780081873851862}, kTol, ctx);
      AssertObservableNear(obs, "spin_z",
          {0.1245081089104045, -0.1245081089104108, 0.1339804006435313, -0.133980400643525}, kTol, ctx);
      AssertObservableNear(obs, "charge",
          {0.4980707820017161, 0.4980707820017206, 0.5019292179982832, 0.5019292179982803}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_h",
          {-0.6939789513147783, -0.6793556507433933}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_v",
          {-0.7033736358968448, -0.7033736358968449}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_dr", {0.0}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_ur", {0.0}, kTol, ctx);
    } else {
      AssertObservableNear(obs, "energy", {-2.780081873851861}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
      AssertObservableNear(obs, "spin_z",
          {0.1245081089104045, -0.1245081089104109, 0.1339804006435314, -0.133980400643525}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "spin_z", kImagTol, ctx);
      AssertObservableNear(obs, "charge",
          {0.4980707820017161, 0.4980707820017206, 0.5019292179982832, 0.5019292179982802}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "charge", kImagTol, ctx);
      AssertObservableNear(obs, "bond_energy_h",
          {-0.693978951314778, -0.6793556507433933}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "bond_energy_h", kImagTol, ctx);
      AssertObservableNear(obs, "bond_energy_v",
          {-0.7033736358968449, -0.7033736358968449}, kTol, ctx);
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "bond_energy_v", kImagTol, ctx);
      AssertObservableNear(obs, "bond_energy_dr", {0.0}, kTol, ctx);
      AssertObservableNear(obs, "bond_energy_ur", {0.0}, kTol, ctx);
    }
  }
}

TEST_F(tJMeasurerTest, LowestStateEnergy) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  const auto trun_para = BMPSTruncateParams<RealT>::SVD(4, 4, 0);
  auto lowest_sitps = LoadTPS(/*lowest_state=*/true);
  SquaretJVModel model(t, 0, J, V, mu);

  auto obs = ExactSumMeasurerMPI<SquaretJVModel, TEN_ELEM_TYPE, fZ2QN>(
      lowest_sitps, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    constexpr double kTol = 6e-8;
    constexpr double kImagTol = 1e-10;
    constexpr double kObsTol = 1e-5;
    const std::string ctx = "tJ_Lowest";
    AssertObservableKeySet(
        obs,
        {"energy", "spin_z", "charge", "bond_energy_h", "bond_energy_v", "bond_energy_dr", "bond_energy_ur"},
        ctx);
    AssertObservableNear(obs, "energy", {-2.9431635706137875}, kTol, ctx);
    if constexpr (!std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      AssertObservableImagNearZero<TEN_ELEM_TYPE>(obs, "energy", kImagTol, ctx);
    }
    // ED observable checks (QuSpin ground-state references)
    // spin_z and charge are zero/uniform by symmetry; TPS may soft-break this.
    constexpr double kSymTol = 5e-4;
    AssertObservableNear(obs, "spin_z", {0.0, 0.0, 0.0, 0.0}, kSymTol, ctx);
    AssertObservableNear(obs, "charge", {0.5, 0.5, 0.5, 0.5}, kSymTol, ctx);
    AssertObservableNear(obs, "bond_energy_h",
        {-0.7357908926534471, -0.7357908926534471}, kObsTol, ctx);
    AssertObservableNear(obs, "bond_energy_v",
        {-0.7357908926534469, -0.7357908926534469}, kObsTol, ctx);
    AssertObservableNear(obs, "bond_energy_dr", {0.0}, kObsTol, ctx);
    AssertObservableNear(obs, "bond_energy_ur", {0.0}, kObsTol, ctx);
  }
}

int main(int argc, char *argv[]) {
  testing::AddGlobalTestEnvironment(new MPIEnvironment);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  return RUN_ALL_TESTS();
}
