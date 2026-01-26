/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2026-01-03
 *
 * Description: Regression tests for SquareSpinOneHalfXXZModelOBC Measurer.
 * 
 * This file contains regression tests for MCPEPSMeasurer with SquareSpinOneHalfXXZModelOBC,
 * covering energy, local observables, and structure factor measurements.
 * 
 * Test strategy:
 * 1. Load pre-computed TPS data (4x4 Heisenberg)
 * 2. Use fixed random seed for reproducible MC sampling
 * 3. Smoke test: verify output structure (non-empty, correct dimensions)
 * 4. Regression: verify results match expected values
 */

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/square_spin_onehalf_xxz_obc.h"

#include "../test_mpi_env.h"
#include <cstdlib>
#include <filesystem>
#include <iomanip>

using namespace qlten;
using namespace qlpeps;

// Test uses TrivialRepQN (no symmetry) to match Heisenberg TPS data
using TenElemT = QLTEN_Double;
using QNT = special_qn::TrivialRepQN;

/**
 * @brief Fixture for SquareSpinOneHalfXXZModelOBC Measurer regression testing.
 * 
 * Uses 4x4 Heisenberg TPS data from slow_tests/test_data.
 * All tests use fixed random seed and deterministic configuration for reproducibility.
 */
class SquareXXZMeasurerTest : public MPITest {
 protected:
  // Lattice and TPS parameters
  static constexpr size_t Ly = 4;
  static constexpr size_t Lx = 4;
  static constexpr size_t Dpeps = 8;
  
  // Model parameters (Heisenberg: Jxy = Jz = J)
  static constexpr double J = 1.0;
  
  // Random seed for reproducible MC sampling
  static constexpr unsigned int MC_SEED = 42;
  
  // Paths
  std::string tps_path;
  std::string output_dir;
  
  void SetUp() override {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);
    
    tps_path = std::string(TEST_SOURCE_DIR) + "/slow_tests/test_data/tps_square_heisenberg4x4D8Double";
    output_dir = "test_square_xxz_measurer_output";
    
    if (rank == hp_numeric::kMPIMasterRank) {
      std::filesystem::create_directories(output_dir);
    }
    MPI_Barrier(comm);
  }
  
  /**
   * @brief Create a deterministic checkerboard configuration.
   */
  Configuration CreateCheckerboardConfig() const {
    Configuration config(Ly, Lx);
    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        config({row, col}) = (row + col) % 2;
      }
    }
    return config;
  }
};

/**
 * @brief Smoke test for Structure Factor measurement.
 * 
 * Verifies:
 * 1. MeasureStructureFactor runs without crashing
 * 2. Output "SpSm_cross" is non-empty when there are valid spin pairs
 * 3. Output structure is correct (multiples of 5: y1, x1, y2, x2, val)
 */
TEST_F(SquareXXZMeasurerTest, StructureFactorSmoke) {
  using Model = SquareSpinOneHalfXXZModelOBC;
  using MCUpdater = MCUpdateSquareNNExchange;
  
  auto sitps = SplitIndexTPS<TenElemT, QNT>(Ly, Lx);
  sitps.Load(tps_path);
  
  Configuration config = CreateCheckerboardConfig();
  
  // Verify configuration
  size_t count_up = 0, count_down = 0;
  for (size_t row = 0; row < Ly; ++row) {
    for (size_t col = 0; col < Lx; ++col) {
      if (config({row, col}) == 0) count_down++;
      else count_up++;
    }
  }
  
  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "[SquareXXZMeasurerTest] Configuration: " << count_down << " down, " 
              << count_up << " up spins" << std::endl;
  }
  
  EXPECT_EQ(count_up, count_down) << "Checkerboard should have equal up and down spins";
  
  // Setup MC parameters with minimal sampling for smoke test
  MonteCarloParams mc_params(5, 5, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, 2 * Dpeps, 1e-15));
  MCMeasurementParams para(mc_params, peps_params);
  
  // Create model with structure factor measurement enabled
  Model heisenberg_model(J, J, 0);
  heisenberg_model.SetEnableStructureFactor(true);
  
  // Create MC updater with fixed seed
  MCUpdater mc_updater(MC_SEED);
  
  // Create measurer using loaded TPS
  auto executor = new MCPEPSMeasurer<TenElemT, QNT, MCUpdater, Model>(
      sitps, para, comm, heisenberg_model, mc_updater
  );
  ASSERT_NE(executor, nullptr) << "Failed to create MCPEPSMeasurer";
  
  // Execute measurement
  executor->Execute();
  
  // Get energy as sanity check
  auto [energy, en_err] = executor->OutputEnergy();
  
  // Get observable registry for structure factor check
  const auto& registry = executor->ObservableRegistry();
  
  if (rank == hp_numeric::kMPIMasterRank) {
    std::cout << "[SquareXXZMeasurerTest] Energy = " << std::real(energy) 
              << " +/- " << en_err << std::endl;
    
    // Heisenberg 4x4 ground state energy ~ -9.189
    // Allow large tolerance for smoke test with only 5 samples
    EXPECT_LT(std::real(energy), 0.0) << "Energy should be negative for antiferromagnetic Heisenberg";
    
    // List available observables
    std::cout << "[SquareXXZMeasurerTest] Available observables:" << std::endl;
    for (const auto& [key, data] : registry) {
      std::cout << "  " << key << ": values=" << data.first.size() 
                << ", errors=" << data.second.size() << std::endl;
    }
    
    // Check SpSm_cross output in registry
    auto it = registry.find("SpSm_cross");
    if (it != registry.end()) {
      const auto& [values, errors] = it->second;
      std::cout << "[SquareXXZMeasurerTest] SpSm_cross data size: " << values.size() << std::endl;
      
      // Each correlation entry has 5 values: y1, x1, y2, x2, val
      EXPECT_EQ(values.size() % 5, 0) << "SpSm_cross should have multiples of 5 elements";
      
      size_t num_correlations = values.size() / 5;
      std::cout << "[SquareXXZMeasurerTest] Number of SpSm correlations: " << num_correlations << std::endl;
      
      // Non-zero correlations expected (half up, half down spins)
      EXPECT_GT(num_correlations, 0) << "Should have at least one SpSm correlation";
      
      // Print first few correlations for regression test values
      if (num_correlations > 0) {
        std::cout << "[SquareXXZMeasurerTest] First correlations (y1,x1,y2,x2,val):" << std::endl;
        for (size_t i = 0; i < std::min(num_correlations, size_t(5)); ++i) {
          size_t base = i * 5;
          std::cout << "  (" << size_t(std::real(values[base])) << "," << size_t(std::real(values[base+1])) 
                    << ") -> (" << size_t(std::real(values[base+2])) << "," << size_t(std::real(values[base+3])) 
                    << "): " << values[base+4] << std::endl;
        }
      }
    } else {
      std::cout << "[SquareXXZMeasurerTest] WARNING: SpSm_cross not found in registry" << std::endl;
      // This is expected if cross-row correlations are not aggregated in registry
    }
  }
  
  delete executor;
}

/**
 * @brief Regression test for Structure Factor with fixed seed.
 * 
 * This test ensures reproducibility and catches unintended changes.
 * With the fix to record ALL pairs (including zeros), vector length is now
 * deterministic: for 4x4 lattice, pairs with y2 > y1:
 *   - Total pairs = sum_{y1=0}^{Ly-2} Lx * (Ly-1-y1) * Lx
 *   - = Lx^2 * sum_{k=1}^{Ly-1} k = Lx^2 * Ly*(Ly-1)/2
 *   - = 16 * 6 = 96 pairs, each with 5 values = 480 elements
 */
TEST_F(SquareXXZMeasurerTest, StructureFactorRegression) {
  using Model = SquareSpinOneHalfXXZModelOBC;
  using MCUpdater = MCUpdateSquareNNExchange;
  
  auto sitps = SplitIndexTPS<TenElemT, QNT>(Ly, Lx);
  sitps.Load(tps_path);
  
  Configuration config = CreateCheckerboardConfig();
  
  // 5 samples, 5 warmup sweeps, 1 sweep between samples
  // Fixed seed ensures reproducible MC evolution
  MonteCarloParams mc_params(5, 5, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, 2 * Dpeps, 1e-15));
  MCMeasurementParams para(mc_params, peps_params);
  
  Model heisenberg_model(J, J, 0);
  heisenberg_model.SetEnableStructureFactor(true);
  
  MCUpdater mc_updater(MC_SEED);
  
  auto executor = new MCPEPSMeasurer<TenElemT, QNT, MCUpdater, Model>(
      sitps, para, comm, heisenberg_model, mc_updater
  );
  ASSERT_NE(executor, nullptr);
  
  executor->Execute();
  
  auto [energy, en_err] = executor->OutputEnergy();
  const auto& registry = executor->ObservableRegistry();
  
  if (rank == hp_numeric::kMPIMasterRank) {
    // Regression assertions for energy
    EXPECT_NEAR(std::real(energy), -9.22, 0.01) << "Energy regression check";
    
    // Regression assertions for SpSm_cross
    auto it = registry.find("SpSm_cross");
    ASSERT_NE(it, registry.end()) << "SpSm_cross must be present";
    
    const auto& [values, errors] = it->second;
    
    // Expected size: 96 pairs × 5 values = 480 elements
    // 96 = Lx^2 * Ly*(Ly-1)/2 = 16 * 6 = 96
    constexpr size_t expected_num_pairs = 96;
    constexpr size_t expected_size = expected_num_pairs * 5;
    EXPECT_EQ(values.size(), expected_size) << "SpSm_cross size regression check";
    
    // Regression check: compare all 96 correlation values (every 5th element starting from index 4)
    // Expected values obtained from deterministic run with seed=42 and fixed checkerboard config.
    // Note: these values depend on the structure-factor contraction truncation parameters
    // (now taken from tps_sample->trun_para), so re-baseline if that policy changes.
    const std::vector<double> expected_corr = {
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      8.03066952633268716e-02,
      0.00000000000000000e+00,
      -1.32760398440958305e-03,
      -2.65520796881916956e-03,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      -2.23898892249996045e-04,
      -2.94384850275200745e-04,
      4.43982046534908948e-04,
      -1.69688180365574397e-04,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      1.03676086183806214e-02,
      0.00000000000000000e+00,
      -1.68181608721474363e-01,
      -3.36363217442948892e-01,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      9.57678713096372968e-04,
      -9.58586526251396152e-05,
      -3.57444358737392577e-05,
      -2.04020155726445643e-04,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      -1.35780733619570385e-02,
      3.99840061292929122e-02,
      -9.50078371876302008e-03,
      6.77918538979805224e-02,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      -1.93839211368095760e-03,
      0.00000000000000000e+00,
      -5.94231937740026224e-04,
      -4.09724853993283754e-03,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      -3.61715338979970574e-02,
      1.02487407380379897e-01,
      -4.10384771646527674e-02,
      3.30013230446646461e-02,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      0.00000000000000000e+00,
      -2.23154685548629514e-03,
      0.00000000000000000e+00,
      -4.15025480396104371e-04,
      -5.26757498422884439e-04,
      -4.73960059143751278e-01,
      8.77766916220446836e-02,
      -6.64819801061862503e-02,
      6.51233602512299586e-02,
      5.54578481629943165e-02,
      -1.26286218401435169e-01,
      6.80757544662411174e-02,
      -2.72722880478375104e-02,
      -8.14476347480973806e-02,
      9.35789435133548009e-02,
      -4.03841386053227336e-01,
      1.31771921887982896e-01,
      1.52637448080098249e-02,
      -2.44963201749073059e-02,
      5.84935955320620843e-02,
      -1.08010160023499502e-01
    };
    
    ASSERT_EQ(expected_corr.size(), expected_num_pairs) 
        << "Expected correlation vector size mismatch";
    
    // Helper for re-baselining: dump the current correlation values as a C++ initializer list.
    // Usage:
    //   QLPEPS_DUMP_XXZ_SPSM_CROSS_GOLDEN=1 ctest -R test_square_xxz_measurer -V
    if (const char* dump_env = std::getenv("QLPEPS_DUMP_XXZ_SPSM_CROSS_GOLDEN");
        dump_env != nullptr && std::string(dump_env) != "0") {
      std::vector<double> corr_now;
      corr_now.reserve(expected_num_pairs);
      for (size_t i = 0; i < expected_num_pairs; ++i) {
        corr_now.push_back(std::real(values[i * 5 + 4]));
      }
      std::cout << std::setprecision(17);
      std::cout << "XXZ SpSm_cross golden (size=" << corr_now.size() << "):\n";
      std::cout << "{\n";
      for (size_t i = 0; i < corr_now.size(); ++i) {
        std::cout << "  " << corr_now[i] << (i + 1 == corr_now.size() ? "" : ",") << "\n";
      }
      std::cout << "}\n";
    }

    // Extract correlation values (every 5th element starting at index 4)
    constexpr double tol = 1e-10;
    for (size_t i = 0; i < expected_num_pairs; ++i) {
      size_t val_idx = i * 5 + 4;  // 5th element of each tuple
      EXPECT_NEAR(std::real(values[val_idx]), expected_corr[i], tol)
          << "SpSm_cross correlation[" << i << "] mismatch at y1=" 
          << std::real(values[i*5]) << ",x1=" << std::real(values[i*5+1])
          << " -> y2=" << std::real(values[i*5+2]) << ",x2=" << std::real(values[i*5+3]);
    }
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
