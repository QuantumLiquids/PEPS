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
#include <sstream>

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

/**
 * @brief Smoke test for singlet pair correlation measurement.
 *
 * Verifies:
 * 1. MeasureSingletPairCorrelation runs without crashing
 * 2. Output "SC_singlet_pair_corr" is present in registry
 * 3. Output structure is correct (multiples of 7: ref_y, ref_x, ref_orient, tgt_y, tgt_x, tgt_orient, val)
 */
TEST_F(tJModelMeasurerTest, DISABLED_SingletPairCorrelationSmoke) {
  using Model = SquaretJNNModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  auto sitps = SplitIndexTPS<TenElemT, QNT>(Ly, Lx);
  sitps.Load(tps_path);

  // Manually construct configuration with horizontal hole pair at (2,2)-(2,3)
  // This is configuration17 transposed, ensuring selection rules are satisfied
  // from the first sample (adjacent holes form valid reference bond)
  Configuration config(Ly, Lx);
  std::vector<std::vector<size_t>> cfg = {
    {1, 0, 0, 1, 0, 1},
    {0, 1, 1, 0, 1, 0},
    {1, 0, 2, 2, 0, 1},  // holes at (2,2) and (2,3) - horizontal pair!
    {0, 1, 1, 0, 1, 0},
    {1, 0, 1, 0, 0, 1},
    {0, 1, 0, 1, 1, 0}
  };
  for (size_t y = 0; y < Ly; ++y) {
    for (size_t x = 0; x < Lx; ++x) {
      config({y, x}) = cfg[y][x];
    }
  }

  // Setup MC parameters with minimal sampling for smoke test
  MonteCarloParams mc_params(5, 5, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, 2 * Dpeps, 1e-15));
  MCMeasurementParams para(mc_params, peps_params);

  // Create model with singlet pair correlation enabled
  Model tJ_model(t, J, mu);
  tJ_model.SetEnableSingletPairCorrelation(true);

  MCUpdater mc_updater(MC_SEED);

  auto executor = new MCPEPSMeasurer<TenElemT, QNT, MCUpdater, Model>(
      sitps, para, comm, tJ_model, mc_updater
  );
  ASSERT_NE(executor, nullptr) << "Failed to create MCPEPSMeasurer";

  executor->Execute();

  auto [energy, en_err] = executor->OutputEnergy();
  const auto& registry = executor->ObservableRegistry();

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "[tJModelMeasurerTest] SC Energy = " << std::real(energy)
              << " +/- " << en_err << std::endl;

    // Energy should still be valid
    EXPECT_LT(std::real(energy), 0.0) << "Energy should be negative";

    // List available observables
    std::cout << "[tJModelMeasurerTest] Available observables with SC:" << std::endl;
    for (const auto& [key, data] : registry) {
      std::cout << "  " << key << ": values=" << data.first.size()
                << ", errors=" << data.second.size() << std::endl;
    }

    // Check SC_singlet_pair_corr output (new format: values only)
    auto it = registry.find("SC_singlet_pair_corr");
    if (it != registry.end()) {
      const auto& [values, errors] = it->second;
      std::cout << "[tJModelMeasurerTest] SC_singlet_pair_corr data size: " << values.size() << std::endl;

      // Expected number of pairs for 6x6 lattice (horizontal targets only for now):
      // Horizontal ref bonds: (Ly-1) * (Lx-1) = 5 * 5 = 25
      // For each ref at y1, horizontal targets at y2 > y1: (Ly-1-y1) * (Lx-1)
      // Total: (Lx-1)^2 * sum_{k=1}^{Ly-1} k = 25 * 15 = 375
      constexpr size_t expected_pairs = 375;
      EXPECT_EQ(values.size(), expected_pairs)
          << "SC_singlet_pair_corr should have exactly " << expected_pairs << " values";

      // Verify using ComputeNumSCPairs
      size_t computed_pairs = SingletPairCorrelationMixin<SquaretJNNModel>::ComputeNumSCPairs(Ly, Lx);
      EXPECT_EQ(computed_pairs, expected_pairs) << "ComputeNumSCPairs should return " << expected_pairs;

      // Print non-zero correlations with coordinate mapping
      if (values.size() > 0) {
        auto mapping = SingletPairCorrelationMixin<SquaretJNNModel>::GenerateSCPairCorrIndexMapping(Ly, Lx);
        EXPECT_EQ(mapping.size(), values.size()) << "Mapping size should equal values size";

        std::cout << "[tJModelMeasurerTest] Non-zero SC correlations (ref_y,ref_x,orient -> tgt_y,tgt_x,orient: val):" << std::endl;
        size_t nonzero_printed = 0;
        for (size_t i = 0; i < values.size() && nonzero_printed < 10; ++i) {
          if (std::abs(values[i]) > 1e-15) {
            const auto& m = mapping[i];
            std::cout << "  (" << m[0] << "," << m[1] << "," << m[2] << ") -> ("
                      << m[3] << "," << m[4] << "," << m[5] << "): " 
                      << std::scientific << std::setprecision(6) << values[i] 
                      << std::defaultfloat << std::endl;
            nonzero_printed++;
          }
        }
        if (nonzero_printed == 0) {
          std::cout << "  (none found)" << std::endl;
        }
      }
    } else {
      // Might not be present if no valid bond pairs exist in this configuration
      std::cout << "[tJModelMeasurerTest] SC_singlet_pair_corr not found in registry" << std::endl;
      // For a t-J model with 2 holes, we expect SOME valid pairs
      FAIL() << "SC_singlet_pair_corr should be present with singlet pair correlation enabled";
    }

    // Verify output file generation
    const std::string stats_dir = "./stats/";
    
    // Check SC_singlet_pair_corr.csv exists and has correct format
    {
      std::ifstream ifs(stats_dir + "SC_singlet_pair_corr.csv");
      ASSERT_TRUE(ifs.good()) << "SC_singlet_pair_corr.csv should be generated";
      
      std::string header;
      std::getline(ifs, header);
      EXPECT_EQ(header, "index,mean,stderr") << "CSV header should be 'index,mean,stderr'";
      
      size_t data_lines = 0;
      std::string line;
      while (std::getline(ifs, line)) {
        if (!line.empty()) data_lines++;
      }
      
      constexpr size_t expected_pairs = 375;
      EXPECT_EQ(data_lines, expected_pairs) 
          << "SC_singlet_pair_corr.csv should have " << expected_pairs << " data lines";
    }
    
    // Check SC_singlet_pair_corr_coords.txt exists and matches expected format
    {
      std::ifstream ifs(stats_dir + "SC_singlet_pair_corr_coords.txt");
      ASSERT_TRUE(ifs.good()) << "SC_singlet_pair_corr_coords.txt should be generated";
      
      size_t data_lines = 0;
      std::string line;
      while (std::getline(ifs, line)) {
        if (!line.empty() && line[0] != '#') data_lines++;
      }
      
      constexpr size_t expected_pairs = 375;
      EXPECT_EQ(data_lines, expected_pairs) 
          << "SC_singlet_pair_corr_coords.txt should have " << expected_pairs << " data lines";
    }
    
    std::cout << "[tJModelMeasurerTest] Output file verification: PASSED" << std::endl;
  }

  delete executor;
}

/**
 * @brief Unit test for coordinate mapping generation.
 *
 * Verifies:
 * 1. GenerateSCPairCorrCoordString() produces correct format
 * 2. Number of lines matches ComputeNumSCPairs()
 * 3. coord_generator is set in DescribeObservables()
 */
TEST_F(tJModelMeasurerTest, CoordinateMappingGeneration) {
  using Model = SquaretJNNModel;

  // Test GenerateSCPairCorrCoordString output format
  std::string coord_str = SingletPairCorrelationMixin<Model>::GenerateSCPairCorrCoordString(Ly, Lx);
  
  // Count non-header lines
  size_t line_count = 0;
  std::istringstream iss(coord_str);
  std::string line;
  while (std::getline(iss, line)) {
    if (!line.empty() && line[0] != '#') {
      line_count++;
    }
  }
  
  size_t expected_pairs = SingletPairCorrelationMixin<Model>::ComputeNumSCPairs(Ly, Lx);
  EXPECT_EQ(line_count, expected_pairs) 
      << "Coordinate string should have " << expected_pairs << " data lines";

  // Verify first line format (after header)
  iss.clear();
  iss.str(coord_str);
  std::getline(iss, line);  // Skip header
  std::getline(iss, line);  // First data line
  EXPECT_EQ(line, "0 0 0 0 1 0 0") << "First line should be index 0 with ref(0,0,0)->tgt(1,0,0)";

  // Verify coord_generator is set in DescribeObservables
  Model tJ_model(t, J, mu);
  tJ_model.SetEnableSingletPairCorrelation(true);
  auto metas = tJ_model.DescribeObservables(Ly, Lx);
  
  bool found_sc_corr = false;
  for (const auto& meta : metas) {
    if (meta.key == "SC_singlet_pair_corr") {
      found_sc_corr = true;
      EXPECT_TRUE(static_cast<bool>(meta.coord_generator)) 
          << "SC_singlet_pair_corr should have coord_generator set";
      
      // Verify coord_generator produces same output
      if (meta.coord_generator) {
        std::string generated = meta.coord_generator(Ly, Lx);
        EXPECT_EQ(generated, coord_str) 
            << "coord_generator should produce same output as GenerateSCPairCorrCoordString";
      }
      break;
    }
  }
  EXPECT_TRUE(found_sc_corr) << "SC_singlet_pair_corr should be in DescribeObservables";
}

/**
 * @brief Unit test for singlet pair correlation selection rules.
 *
 * Verifies the selection rule logic:
 * - Reference bond: both sites must be empty (state = 2)
 * - Target bond: must be (up,down) or (down,up) pair
 * 
 * This is a simple logic test that doesn't require full tensor contraction.
 */
TEST_F(tJModelMeasurerTest, DISABLED_SingletPairCorrelationSelectionRules) {
  // Create a test configuration with adjacent holes at (0,0)-(0,1)
  Configuration config(Ly, Lx);
  size_t up_count = 0, down_count = 0;
  for (size_t row = 0; row < Ly; ++row) {
    for (size_t col = 0; col < Lx; ++col) {
      if (row == 0 && col < 2) {
        config({row, col}) = 2;  // empty
      } else if ((row + col) % 2 == 0 && up_count < num_up) {
        config({row, col}) = 0;  // spin up
        up_count++;
      } else if (down_count < num_down) {
        config({row, col}) = 1;  // spin down
        down_count++;
      } else {
        config({row, col}) = 0;
        up_count++;
      }
    }
  }

  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "[tJModelMeasurerTest] Selection rules test:" << std::endl;
    
    // Verify ref bond (0,0)-(0,1) is valid (both empty)
    bool ref_valid = (config({0, 0}) == 2 && config({0, 1}) == 2);
    std::cout << "  Ref bond (0,0)-(0,1) both empty: " << (ref_valid ? "YES" : "NO") << std::endl;
    EXPECT_TRUE(ref_valid) << "Ref bond should have both sites empty";
    
    // Verify ref bond (0,1)-(0,2) is NOT valid (one empty, one up)
    bool ref_invalid = (config({0, 1}) == 2 && config({0, 2}) == 2);
    std::cout << "  Ref bond (0,1)-(0,2) both empty: " << (ref_invalid ? "YES" : "NO") << std::endl;
    EXPECT_FALSE(ref_invalid) << "Only (0,0)-(0,1) should be valid ref bond";
    
    // Count valid target bonds in row 1
    size_t valid_targets = 0;
    for (size_t x = 0; x < Lx - 1; ++x) {
      size_t s1 = config({1, x});
      size_t s2 = config({1, x + 1});
      bool is_up_down = (s1 == 0 && s2 == 1);
      bool is_down_up = (s1 == 1 && s2 == 0);
      if (is_up_down || is_down_up) {
        valid_targets++;
        std::cout << "  Target bond (1," << x << ")-(1," << x+1 << "): " 
                  << (is_up_down ? "up-down" : "down-up") << std::endl;
      }
    }
    EXPECT_GT(valid_targets, 0) << "Should have at least one valid target bond in row 1";
    std::cout << "  Valid target bonds in row 1: " << valid_targets << std::endl;
  }
}

/**
 * @brief Regression test for singlet pair correlation with fixed seed.
 *
 * This test ensures reproducibility and catches unintended changes.
 * Uses deterministic seed and configuration for exact reproducibility.
 *
 * NOTE: This test is disabled by default (prefix DISABLED_) because it takes ~4 min.
 * Run explicitly with: --gtest_also_run_disabled_tests --gtest_filter="*SingletPairCorrelationRegression*"
 *
 * TODO: Use a dataset with strong superconducting correlation for meaningful regression test.
 *       Current configuration0 may not have adjacent holes, leading to sparse SC correlations.
 */
TEST_F(tJModelMeasurerTest, DISABLED_SingletPairCorrelationRegression) {
  using Model = SquaretJNNModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  auto sitps = SplitIndexTPS<TenElemT, QNT>(Ly, Lx);
  sitps.Load(tps_path);

  Configuration config = LoadConfiguration(tps_path + "/configuration0");

  // 10 samples for regression (same as EnergyRegression)
  MonteCarloParams mc_params(10, 10, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, 2 * Dpeps, 1e-15));
  MCMeasurementParams para(mc_params, peps_params);

  Model tJ_model(t, J, mu);
  tJ_model.SetEnableSingletPairCorrelation(true);

  MCUpdater mc_updater(MC_SEED);

  auto executor = new MCPEPSMeasurer<TenElemT, QNT, MCUpdater, Model>(
      sitps, para, comm, tJ_model, mc_updater
  );
  ASSERT_NE(executor, nullptr);

  executor->Execute();

  const auto& registry = executor->ObservableRegistry();

  if (rank == hp_numeric::kMPIMasterRank) {
    auto it = registry.find("SC_singlet_pair_corr");
    ASSERT_NE(it, registry.end()) << "SC_singlet_pair_corr must be present";

    const auto& [values, errors] = it->second;

    // Check size is exactly as expected (values only, no coordinates)
    constexpr size_t expected_pairs = 375;
    EXPECT_EQ(values.size(), expected_pairs) << "SC_singlet_pair_corr should have exactly num_pairs values";

    // Print summary statistics for regression baseline
    double sum_abs = 0.0;
    size_t nonzero_count = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      double val = std::abs(values[i]);
      sum_abs += val;
      if (val > 1e-15) nonzero_count++;
    }
    std::cout << std::setprecision(15);
    std::cout << "[tJModelMeasurerTest] SC regression: sum_abs=" << sum_abs 
              << ", nonzero_count=" << nonzero_count << "/" << expected_pairs << std::endl;

    // Most correlations should be zero due to selection rules
    // (ref needs empty-empty, target needs up-down or down-up)
    // With 2 holes in a 36-site lattice, valid pairs are rare
    // But we expect SOME non-zero correlations when holes happen to be adjacent
    EXPECT_LT(nonzero_count, expected_pairs / 2) 
        << "Most correlations should be zero due to selection rules";

    // TODO: Add numerical regression baseline values once the implementation is stable.
    //       Run this test once to get baseline values:
    //         ./tests/test_tJ_model_solver --gtest_also_run_disabled_tests \
    //             --gtest_filter="*SingletPairCorrelationRegression*"
    //       Then add assertions like:
    //         constexpr double expected_sum_abs = XXX.XXXXXXXXXXXX;
    //         EXPECT_NEAR(sum_abs, expected_sum_abs, 1e-10);
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
