/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-03-28.
*
 * Description: QuantumLiquids/PEPS project. Regression tests for t-J Model solvers.
 *
 * Test strategy:
 * 1. Load pre-computed TPS data (6x6 t-J model with 2 holes, J/t=0.3, D=8)
 * 2. Use fixed random seed for reproducible MC sampling
 * 3. Smoke test: verify measurement runs without crashing
 * 4. Regression: verify energy matches expected value from VMC optimization
 *
 * Test data generated from VMC optimization (~300 iterations):
 * - System: 6x6 square lattice, OBC
 * - Holes: 2 (doping ~ 5.6%)
 * - Parameters: t=1.0, J=0.3, μ=0
 * - Symmetry: fU1QN (fermion U(1), particle number conservation, no spin symmetry)
 * - Final energy: ~ -14.76
 *
 * Full measurement reference (from cluster run with high statistics):
 * - Energy: -14.7581 ± 0.00128
 * - Charge: 6x6 matrix, edges ~0.99, center ~0.79 (holes concentrated in center)
 * - Spin_z: 6x6 matrix, ~0 with small AFM fluctuations (|Sz| < 0.04)
 * Reference data location: finite-size_PEPS_tJ/data/tJfU1_stats6x6Hole2J0.3D8-8/
 */

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_tJ_model.h"

#include "../test_mpi_env.h"
#include <filesystem>
#include <iomanip>

using namespace qlten;
using namespace qlpeps;

// Test uses fU1QN (fermion U(1) symmetry) to match t-J TPS data
using TenElemT = QLTEN_Double;
using QNT = special_qn::fU1QN;

/**
 * @brief Fixture for t-J Model Measurer regression testing.
 *
 * Uses 6x6 t-J TPS data with 2 holes, J/t=0.3, D=8.
 * All tests use fixed random seed and deterministic configuration for reproducibility.
 */
class tJModelMeasurerTest : public MPITest {
 protected:
  // Lattice and TPS parameters
  static constexpr size_t Ly = 6;
  static constexpr size_t Lx = 6;
  static constexpr size_t Dpeps = 8;
  static constexpr size_t N = Ly * Lx;

  // Model parameters
  static constexpr double t = 1.0;
  static constexpr double J = 0.3;
  static constexpr double mu = 0.0;

  // Particle numbers
  static constexpr size_t num_hole = 2;
  static constexpr size_t num_up = 17;
  static constexpr size_t num_down = 17;

  // Random seed for reproducible MC sampling
  static constexpr unsigned int MC_SEED = 42;

  // Deterministic regression value (seed=42, config0, 10 samples, 10 warmup)
  // Full measurement reference: -14.7581 ± 0.00128 (from tJfU1_stats6x6Hole2J0.3D8-8)
  static constexpr double EXPECTED_ENERGY = -14.74320489110316;
  static constexpr double ENERGY_TOLERANCE = 1e-8;  // Deterministic: floating-point tolerance only

  // Paths
  std::string tps_path;
  std::string output_dir;

  void SetUp() override {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);

    tps_path = std::string(TEST_SOURCE_DIR) + "/test_data/tps_tJ_6x6Hole2_J0.3_D8_fU1";
    output_dir = "test_tJ_model_measurer_output";

    if (rank == hp_numeric::kMPIMasterRank) {
      std::filesystem::create_directories(output_dir);
    }
    MPI_Barrier(comm);
  }

  /**
   * @brief Create initial configuration from saved file.
   *
   * Format: Ly rows, each with Lx integers (0=up, 1=down, 2=empty)
   */
  Configuration LoadConfiguration(const std::string &config_file) const {
    Configuration config(Ly, Lx);
    std::ifstream ifs(config_file);
    if (!ifs) {
      throw std::runtime_error("Cannot open configuration file: " + config_file);
    }
    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        size_t val;
        ifs >> val;
        config({row, col}) = val;
      }
    }
    return config;
  }

  /**
   * @brief Create a valid t-J configuration with specified particle numbers.
   *
   * Configuration encoding:
   * - 0: spin up |↑⟩
   * - 1: spin down |↓⟩
   * - 2: empty site |0⟩ (hole)
   */
  Configuration CreatetJConfiguration() const {
    Configuration config(Ly, Lx);

    // Fill with alternating spins, then place holes
    size_t up_count = 0, down_count = 0, hole_count = 0;
    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        if (hole_count < num_hole && (row == Ly - 1) && (col < num_hole)) {
          // Place holes at bottom-left corner
          config({row, col}) = 2;
          hole_count++;
        } else if ((row + col) % 2 == 0 && up_count < num_up) {
          config({row, col}) = 0;  // spin up
          up_count++;
        } else if (down_count < num_down) {
          config({row, col}) = 1;  // spin down
          down_count++;
        } else {
          config({row, col}) = 0;  // fill remaining with spin up
          up_count++;
        }
      }
    }
    return config;
  }
};

/**
 * @brief Smoke test for t-J model energy measurement.
 *
 * Verifies:
 * 1. MCPEPSMeasurer can be created and executed
 * 2. Energy is computed without crashing
 * 3. Energy is negative (bound state for antiferromagnetic coupling)
 */
TEST_F(tJModelMeasurerTest, EnergyMeasurementSmoke) {
  using Model = SquaretJNNModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  auto sitps = SplitIndexTPS<TenElemT, QNT>(Ly, Lx);
  sitps.Load(tps_path);

  // Load configuration from saved file
  Configuration config = LoadConfiguration(tps_path + "/configuration0");

  // Verify configuration particle counts
  size_t count_up = 0, count_down = 0, count_hole = 0;
  for (size_t row = 0; row < Ly; ++row) {
    for (size_t col = 0; col < Lx; ++col) {
      size_t state = config({row, col});
      if (state == 0) count_up++;
      else if (state == 1) count_down++;
      else count_hole++;
    }
  }

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "[tJModelMeasurerTest] Configuration: "
              << count_up << " up, " << count_down << " down, "
              << count_hole << " holes" << std::endl;
  }

  EXPECT_EQ(count_hole, num_hole) << "Configuration should have " << num_hole << " holes";
  EXPECT_EQ(count_up + count_down + count_hole, N) << "Total sites should be " << N;

  // Setup MC parameters with minimal sampling for smoke test
  MonteCarloParams mc_params(5, 5, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, 2 * Dpeps, 1e-15));
  MCMeasurementParams para(mc_params, peps_params);

  // Create model solver
  Model tJ_model(t, J, mu);

  // Create MC updater with fixed seed
  MCUpdater mc_updater(MC_SEED);

  // Create measurer
  auto executor = new MCPEPSMeasurer<TenElemT, QNT, MCUpdater, Model>(
      sitps, para, comm, tJ_model, mc_updater
  );
  ASSERT_NE(executor, nullptr) << "Failed to create MCPEPSMeasurer";

  // Execute measurement
  executor->Execute();

  // Get energy
  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "[tJModelMeasurerTest] Energy = " << std::real(energy)
              << " +/- " << en_err << std::endl;

    // Basic sanity checks
    EXPECT_LT(std::real(energy), 0.0)
        << "Energy should be negative for antiferromagnetic t-J model";

    // Energy should be in reasonable range (not completely wrong)
    EXPECT_GT(std::real(energy), -50.0)
        << "Energy should not be unreasonably negative";
  }

  delete executor;
}

/**
 * @brief Regression test for t-J model energy.
 *
 * This test ensures reproducibility and catches unintended changes.
 * Uses more samples for better statistics.
 */
TEST_F(tJModelMeasurerTest, EnergyRegression) {
  using Model = SquaretJNNModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  auto sitps = SplitIndexTPS<TenElemT, QNT>(Ly, Lx);
  sitps.Load(tps_path);

  // Load configuration from saved file
  Configuration config = LoadConfiguration(tps_path + "/configuration0");

  // Limited samples for fast regression test (fixed seed ensures determinism)
  MonteCarloParams mc_params(10, 10, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, 2 * Dpeps, 1e-15));
  MCMeasurementParams para(mc_params, peps_params);

  Model tJ_model(t, J, mu);
  MCUpdater mc_updater(MC_SEED);

  auto executor = new MCPEPSMeasurer<TenElemT, QNT, MCUpdater, Model>(
      sitps, para, comm, tJ_model, mc_updater
  );
  ASSERT_NE(executor, nullptr);

  executor->Execute();

  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << std::setprecision(15);
    std::cout << "[tJModelMeasurerTest] Regression Energy = " << std::real(energy)
              << " +/- " << en_err << std::endl;
    std::cout << "[tJModelMeasurerTest] Expected = " << EXPECTED_ENERGY
              << " +/- " << ENERGY_TOLERANCE << std::endl;

    // Regression check: energy should match expected value within tolerance
    EXPECT_NEAR(std::real(energy), EXPECTED_ENERGY, ENERGY_TOLERANCE)
        << "Energy regression check failed";
  }

  delete executor;
}

/**
 * @brief Test local observables (spin_z and charge).
 *
 * Verifies:
 * 1. Observable registry contains expected keys
 * 2. Observables have correct dimensions
 * 3. Charge sum matches particle number
 * 4. Spin_z values are bounded in [-0.5, 0.5]
 */
TEST_F(tJModelMeasurerTest, LocalObservables) {
  using Model = SquaretJNNModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  auto sitps = SplitIndexTPS<TenElemT, QNT>(Ly, Lx);
  sitps.Load(tps_path);

  Configuration config = LoadConfiguration(tps_path + "/configuration0");

  MonteCarloParams mc_params(5, 5, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, 2 * Dpeps, 1e-15));
  MCMeasurementParams para(mc_params, peps_params);

  Model tJ_model(t, J, mu);
  MCUpdater mc_updater(MC_SEED);

  auto executor = new MCPEPSMeasurer<TenElemT, QNT, MCUpdater, Model>(
      sitps, para, comm, tJ_model, mc_updater
  );
  ASSERT_NE(executor, nullptr);

  executor->Execute();

  const auto& registry = executor->ObservableRegistry();

  if (rank == hp_numeric::kMPIMasterRank) {
    // List available observables
    std::cout << "[tJModelMeasurerTest] Available observables:" << std::endl;
    for (const auto& [key, data] : registry) {
      std::cout << "  " << key << ": values=" << data.first.size()
                << ", errors=" << data.second.size() << std::endl;
    }

    // Check spin_z
    auto spin_it = registry.find("spin_z");
    ASSERT_NE(spin_it, registry.end()) << "spin_z must be present";

    const auto& [spin_values, spin_errors] = spin_it->second;
    EXPECT_EQ(spin_values.size(), N) << "spin_z should have N=" << N << " values";

    // Spin_z should be bounded
    for (size_t i = 0; i < spin_values.size(); ++i) {
      double sz = std::real(spin_values[i]);
      EXPECT_GE(sz, -0.5) << "spin_z[" << i << "] should be >= -0.5";
      EXPECT_LE(sz, 0.5) << "spin_z[" << i << "] should be <= 0.5";
    }

    // Check charge
    auto charge_it = registry.find("charge");
    ASSERT_NE(charge_it, registry.end()) << "charge must be present";

    const auto& [charge_values, charge_errors] = charge_it->second;
    EXPECT_EQ(charge_values.size(), N) << "charge should have N=" << N << " values";

    // Total charge should equal number of electrons
    double total_charge = 0.0;
    for (const auto& q : charge_values) {
      total_charge += std::real(q);
    }
    double expected_electrons = static_cast<double>(num_up + num_down);
    std::cout << "[tJModelMeasurerTest] Total charge = " << total_charge
              << ", expected = " << expected_electrons << std::endl;

    // Allow some Monte Carlo variance
    EXPECT_NEAR(total_charge, expected_electrons, 1.0)
        << "Total charge should approximately equal number of electrons";
  }

  delete executor;
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
