// SPDX-License-Identifier: LGPL-3.0-only

/*
 * Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
 * Creation Date: 2025-12-30
 *
 * Description: QuantumLiquids/PEPS project. Tests for MonteCarloEngine class.
 *
 * Current test coverage:
 * - Configuration rescue functionality when some configurations lead to invalid amplitudes
 * - More tests can be added here for other MonteCarloEngine features
 *
 * Configuration rescue test strategy:
 * We construct a TPS as a superposition of two product states:
 *   |ψ⟩ = |ψ₁⟩ + ε|ψ₂⟩
 * where:
 * - |ψ₁⟩ = Neel state {{0,1},{1,0}}, amplitude O(1) for Neel config
 * - |ψ₂⟩ = Anti-Neel state {{1,0},{0,1}}, amplitude O(1) for anti-Neel config
 * - ε ~ 1e-300 (tiny coefficient)
 *
 * After the wave function sum (via virtual bond direct sum):
 * - Neel configuration has amplitude O(1) (from |ψ₁⟩ component) -> VALID
 * - Anti-Neel configuration has amplitude ~ε (from |ψ₂⟩ component) -> INVALID
 * - Other Sz=0 configurations have amplitude 0 -> INVALID
 *
 * The rescue mechanism should detect invalid configurations and broadcast
 * a valid configuration from another MPI rank.
 */

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_engine.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/two_dim_tn/peps/square_lattice_peps.h"
#include <filesystem>

#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

TEST(MonteCarloEngineGuards, PBCRequiresTRGContractor) {
  using QNT = special_qn::TrivialRepQN;
  using TenElemT = QLTEN_Double;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  SITPST sitps(2, 2, BoundaryCondition::Periodic);
  Configuration init_cfg(2, 2);
  MonteCarloParams mc_params(/*samples=*/1, /*warmup_sweeps=*/0, /*sweeps_between=*/1, init_cfg, /*is_warmed_up=*/true);
  PEPSParams peps_params(
      TRGTruncateParams<qlten::QLTEN_Double>::SVD(/*d_min=*/1, /*d_max=*/2, /*trunc_error=*/0.0));

  EXPECT_THROW((MonteCarloEngine<TenElemT, QNT, MCUpdateSquareNNExchange>(
                    sitps, mc_params, peps_params, MPI_COMM_WORLD)),
               std::invalid_argument);
}

/**
 * @brief Test fixture for configuration rescue tests using WaveFunctionSum
 * 
 * Constructs a 2x2 spin-1/2 lattice (no U1 symmetry for simplicity).
 * The TPS is a superposition of two product states with different coefficients.
 */
struct TestConfigurationRescueSum : MPITest {
  using QNT = special_qn::TrivialRepQN;
  using TenElemT = QLTEN_Double;
  using Tensor = QLTensor<TenElemT, QNT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using TPST = TPS<TenElemT, QNT>;
  using SplitTPST = SplitIndexTPS<TenElemT, QNT>;
  using PEPST = SquareLatticePEPS<TenElemT, QNT>;

  static constexpr size_t Lx = 2;
  static constexpr size_t Ly = 2;
  static constexpr size_t Dpeps = 3;  // Bond dimension after WaveFunctionSum: 1 + 1 + 1 = 3

  // Physical index: spin-1/2 with 2 states (0=up, 1=down)
  IndexT phy_idx = IndexT({QNSctT(QNT(), 2)}, OUT);

  void SetUp() {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);
  }

  /**
   * @brief Create a PEPS initialized to a product state
   * 
   * @param config 2D configuration specifying the physical state at each site
   * @return PEPS in the specified product state
   */
  PEPST CreateProductStatePEPS(const std::vector<std::vector<size_t>>& config) {
    PEPST peps(phy_idx, Ly, Lx, BoundaryCondition::Open);
    auto config_copy = config;  // Initial() takes non-const reference
    peps.Initial(config_copy);
    return peps;
  }

  /**
   * @brief Create a superposition TPS: |ψ⟩ = |ψ₁⟩ + β|ψ₂⟩ + 0|ψ₃⟩
   * 
   * |ψ₁⟩ is the Neel state: {{0, 1}, {1, 0}} (up-down-down-up)
   * |ψ₂⟩ is the anti-Neel state: {{1, 0}, {0, 1}} (down-up-up-down)
   * |ψ₃⟩ is a third Sz=0 state: {{0, 0}, {1, 1}} (up-up-down-down) with coefficient 0
   * 
   * This tests two failure modes:
   * - Config {{1,0},{0,1}} with small β: amplitude is tiny but blocks exist
   * - Config {{0,0},{1,1}} with β=0: blocks exist but amplitude is exactly zero
   * - Config {{1,1},{0,0}}: NOT in wavefunction at all, no blocks exist
   * 
   * @param beta Coefficient for the anti-Neel state (small = makes it invalid)
   * @return SplitIndexTPS representing the superposition
   */
  SplitTPST CreateSuperpositionTPS(TenElemT beta = 1e-200) {
    // Configuration for Neel state: {{0, 1}, {1, 0}}
    std::vector<std::vector<size_t>> neel_config = {{0, 1}, {1, 0}};
    
    // Configuration for anti-Neel state: {{1, 0}, {0, 1}}
    std::vector<std::vector<size_t>> anti_neel_config = {{1, 0}, {0, 1}};
    
    // Configuration for third Sz=0 state: {{0, 0}, {1, 1}}
    std::vector<std::vector<size_t>> third_config = {{0, 0}, {1, 1}};
    
    // Create PEPS for each state
    PEPST peps_neel = CreateProductStatePEPS(neel_config);
    PEPST peps_anti_neel = CreateProductStatePEPS(anti_neel_config);
    PEPST peps_third = CreateProductStatePEPS(third_config);
    
    // Convert to TPS
    TPST tps_neel = peps_neel.ToTPS();
    TPST tps_anti_neel = peps_anti_neel.ToTPS();
    TPST tps_third = peps_third.ToTPS();
    
    // Create superposition: |ψ⟩ = 1.0 * |Neel⟩ + β * |anti-Neel⟩ + 0 * |third⟩
    // The third state has coefficient 0: blocks exist but amplitude is zero
    std::vector<TPST> tps_list = {tps_neel, tps_anti_neel, tps_third};
    std::vector<TenElemT> coeffs = {TenElemT(1.0), beta, TenElemT(0.0)};
    TPST tps_sum = WaveFunctionSum(tps_list, coeffs);
    
    // Convert to SplitIndexTPS
    return SplitTPST::FromTPS(tps_sum);
  }

  /**
   * @brief Create the "good" Neel configuration (has O(1) amplitude)
   * Config: {{0, 1}, {1, 0}} -> Sz = 0
   */
  Configuration CreateNeelConfig() {
    Configuration config(Ly, Lx);
    config({0, 0}) = 0;  // up
    config({0, 1}) = 1;  // down
    config({1, 0}) = 1;  // down
    config({1, 1}) = 0;  // up
    return config;
  }

  /**
   * @brief Create the anti-Neel configuration (has ~β amplitude, invalid when β is tiny)
   * Config: {{1, 0}, {0, 1}} -> Sz = 0
   */
  Configuration CreateAntiNeelConfig() {
    Configuration config(Ly, Lx);
    config({0, 0}) = 1;  // down
    config({0, 1}) = 0;  // up
    config({1, 0}) = 0;  // up
    config({1, 1}) = 1;  // down
    return config;
  }

  /**
   * @brief Create a third Sz=0 configuration that is NOT in the wavefunction
   * Config: {{0, 0}, {1, 1}} -> Sz = 0, but not in |ψ₁⟩ or |ψ₂⟩
   * This configuration has ZERO amplitude (completely outside wavefunction support).
   */
  Configuration CreateThirdSz0Config() {
    Configuration config(Ly, Lx);
    config({0, 0}) = 0;  // up
    config({0, 1}) = 0;  // up
    config({1, 0}) = 1;  // down
    config({1, 1}) = 1;  // down
    return config;
  }

  /**
   * @brief Create a fourth Sz=0 configuration that is NOT in the wavefunction
   * Config: {{1, 1}, {0, 0}} -> Sz = 0, but not in |ψ₁⟩ or |ψ₂⟩
   * This configuration has ZERO amplitude (completely outside wavefunction support).
   */
  Configuration CreateFourthSz0Config() {
    Configuration config(Ly, Lx);
    config({0, 0}) = 1;  // down
    config({0, 1}) = 1;  // down
    config({1, 0}) = 0;  // up
    config({1, 1}) = 0;  // up
    return config;
  }
};

/**
 * @brief Test rescue with strict threshold (1e-100) - all 3 invalid ranks rescued
 * 
 * Scenario (4 MPI ranks) with strict threshold 1e-100:
 * - Rank 0: Neel config {{0,1},{1,0}} -> amplitude O(1), VALID
 * - Rank 1: Anti-Neel config {{1,0},{0,1}} -> amplitude ~1e-300 < 1e-100, INVALID
 * - Rank 2: Third Sz=0 config {{0,0},{1,1}} -> construction fails, INVALID
 * - Rank 3: Fourth Sz=0 config {{1,1},{0,0}} -> construction fails, INVALID
 * 
 * Expected: Ranks 1, 2, and 3 should be rescued by rank 0's configuration.
 */
TEST_F(TestConfigurationRescueSum, Rescue3of4_Threshold1e100) {
  if (mpi_size < 4) {
    GTEST_SKIP() << "This test requires at least 4 MPI ranks";
  }

  auto sitps = CreateSuperpositionTPS(1e-300);

  // Assign different configurations based on rank
  Configuration config(Ly, Lx);
  if (rank == 0) {
    config = CreateNeelConfig();        // Valid: amplitude O(1)
  } else if (rank == 1) {
    config = CreateAntiNeelConfig();    // Invalid with strict threshold
  } else if (rank == 2) {
    config = CreateThirdSz0Config();    // Invalid: not in wavefunction
  } else {
    config = CreateFourthSz0Config();   // Invalid: not in wavefunction
  }

  MonteCarloParams mc_params(5, 2, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<QLTEN_Double>(
      Dpeps, 2 * Dpeps, 1e-15,
      CompressMPSScheme::SVD_COMPRESS,
      std::make_optional<double>(1e-14),
      std::make_optional<size_t>(10)));

  // Use strict threshold: 1e-100
  ConfigurationRescueParams rescue_params;
  rescue_params.amplitude_min_threshold = 1e-100;
  EXPECT_TRUE(rescue_params.enabled);

  // Create engine - this triggers rescue for ranks 1, 2, and 3
  MonteCarloEngine<TenElemT, QNT, MCUpdateSquareNNExchange> engine(
      sitps, mc_params, peps_params, comm, MCUpdateSquareNNExchange(), rescue_params);

  // After rescue, all ranks should have valid amplitude
  const auto& wfc = engine.WavefuncComp();
  EXPECT_TRUE(CheckWaveFunctionAmplitudeValidity(
      wfc,
      rescue_params.amplitude_min_threshold,
      rescue_params.amplitude_max_threshold))
      << "Rank " << rank << " should have valid amplitude after rescue";

  // All ranks should have the Neel configuration (rescued from rank 0)
  Configuration expected_config = CreateNeelConfig();
  EXPECT_EQ(wfc.config, expected_config)
      << "Rank " << rank << " should have been rescued to Neel config";

  if (rank == 0) {
    std::cout << "✓ Rescue3of4: All 4 ranks now have Neel configuration" << std::endl;
  }
}

/**
 * @brief Test rescue with permissive threshold (DBL_MIN) - only construction failures rescued
 * 
 * Scenario (4 MPI ranks) with permissive threshold DBL_MIN:
 * - Rank 0: Neel config -> amplitude O(1), VALID
 * - Rank 1: Anti-Neel config -> amplitude ~1e-300 > DBL_MIN, VALID (not rescued)
 * - Rank 2: Third Sz=0 config -> construction fails, INVALID (rescued)
 * - Rank 3: Fourth Sz=0 config -> construction fails, INVALID (rescued)
 * 
 * Expected: Only ranks 2 and 3 should be rescued.
 */
TEST_F(TestConfigurationRescueSum, Rescue2of4_DefaultThreshold) {
  if (mpi_size < 4) {
    GTEST_SKIP() << "This test requires at least 4 MPI ranks";
  }

  auto sitps = CreateSuperpositionTPS(1e-300);

  // Assign different configurations based on rank
  Configuration config(Ly, Lx);
  if (rank == 0) {
    config = CreateNeelConfig();        // Valid: amplitude O(1)
  } else if (rank == 1) {
    config = CreateAntiNeelConfig();    // Valid with permissive threshold (amplitude > DBL_MIN)
  } else if (rank == 2) {
    config = CreateThirdSz0Config();    // Invalid: not in wavefunction
  } else {
    config = CreateFourthSz0Config();   // Invalid: not in wavefunction
  }

  MonteCarloParams mc_params(5, 2, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<QLTEN_Double>(
      Dpeps, 2 * Dpeps, 1e-15,
      CompressMPSScheme::SVD_COMPRESS,
      std::make_optional<double>(1e-14),
      std::make_optional<size_t>(10)));

  // Use permissive threshold: DBL_MIN (default)
  ConfigurationRescueParams rescue_params;
  // amplitude_min_threshold defaults to std::numeric_limits<double>::min()
  EXPECT_TRUE(rescue_params.enabled);

  // Create engine - only ranks 2 and 3 should be rescued
  MonteCarloEngine<TenElemT, QNT, MCUpdateSquareNNExchange> engine(
      sitps, mc_params, peps_params, comm, MCUpdateSquareNNExchange(), rescue_params);

  const auto& wfc = engine.WavefuncComp();

  // After rescue, amplitude should be valid
  EXPECT_TRUE(CheckWaveFunctionAmplitudeValidity(
      wfc,
      rescue_params.amplitude_min_threshold,
      rescue_params.amplitude_max_threshold))
      << "Rank " << rank << " should have valid amplitude";

  // Rank 0: original Neel config
  // Rank 1: original Anti-Neel config (NOT rescued with permissive threshold)
  // Rank 2, 3: rescued to Neel config
  if (rank == 0) {
    EXPECT_EQ(wfc.config, CreateNeelConfig());
  } else if (rank == 1) {
    EXPECT_EQ(wfc.config, CreateAntiNeelConfig())
        << "Rank 1 should NOT be rescued with permissive threshold";
  } else {
    EXPECT_EQ(wfc.config, CreateNeelConfig())
        << "Rank " << rank << " should be rescued to Neel config";
  }

  if (rank == 0) {
    std::cout << "✓ Rescue2of4: Rank 1 keeps Anti-Neel, Ranks 2,3 rescued" << std::endl;
  }
}

/**
 * @brief Test that all-valid configurations don't trigger rescue
 */
TEST_F(TestConfigurationRescueSum, Rescue0of4_AllValid) {
  // Use equal coefficients so both configs have O(1) amplitude
  auto sitps = CreateSuperpositionTPS(1.0);

  Configuration config = CreateNeelConfig();

  MonteCarloParams mc_params(5, 2, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<QLTEN_Double>(
      Dpeps, 2 * Dpeps, 1e-15,
      CompressMPSScheme::SVD_COMPRESS,
      std::make_optional<double>(1e-14),
      std::make_optional<size_t>(10)));

  ConfigurationRescueParams rescue_params;

  MonteCarloEngine<TenElemT, QNT, MCUpdateSquareNNExchange> engine(
      sitps, mc_params, peps_params, comm, MCUpdateSquareNNExchange(), rescue_params);

  const auto& wfc = engine.WavefuncComp();
  EXPECT_TRUE(CheckWaveFunctionAmplitudeValidity(
      wfc,
      rescue_params.amplitude_min_threshold,
      rescue_params.amplitude_max_threshold));
  EXPECT_GT(std::abs(wfc.amplitude), 0.1);  // Should have O(1) amplitude
}

/**
 * @brief Test warmup works correctly after rescue
 */
TEST_F(TestConfigurationRescueSum, WarmupAfterRescue2of4) {
  if (mpi_size < 4) {
    GTEST_SKIP() << "This test requires at least 4 MPI ranks";
  }

  auto sitps = CreateSuperpositionTPS(1e-300);

  // Same setup as RescueFourRanks
  Configuration config(Ly, Lx);
  if (rank == 0) {
    config = CreateNeelConfig();
  } else if (rank == 1) {
    config = CreateAntiNeelConfig();
  } else if (rank == 2) {
    config = CreateThirdSz0Config();
  } else {
    config = CreateFourthSz0Config();
  }

  MonteCarloParams mc_params(5, 3, 1, config, false);
  PEPSParams peps_params(BMPSTruncateParams<QLTEN_Double>(
      Dpeps, 2 * Dpeps, 1e-15,
      CompressMPSScheme::SVD_COMPRESS,
      std::make_optional<double>(1e-14),
      std::make_optional<size_t>(10)));

  ConfigurationRescueParams rescue_params;

  MonteCarloEngine<TenElemT, QNT, MCUpdateSquareNNExchange> engine(
      sitps, mc_params, peps_params, comm, MCUpdateSquareNNExchange(), rescue_params);

  // After rescue with default permissive threshold (DBL_MIN):
  // - Rank 0: Neel config (original, valid)
  // - Rank 1: Anti-Neel config (NOT rescued because amplitude ~1e-300 > DBL_MIN)
  // - Rank 2, 3: rescued to Neel config
  const auto& wfc = engine.WavefuncComp();
  if (rank == 1) {
    EXPECT_EQ(wfc.config, CreateAntiNeelConfig())
        << "Rank 1 should NOT be rescued with default permissive threshold";
  } else {
    EXPECT_EQ(wfc.config, CreateNeelConfig())
        << "Rank " << rank << " should have Neel config (original or rescued)";
  }

  // Warmup should succeed after rescue
  int warmup_result = engine.WarmUp();
  EXPECT_EQ(warmup_result, 0) << "Warmup should succeed after rescue";

  // Amplitude should remain valid according to rescue_params thresholds
  // Note: We use CheckWaveFunctionAmplitudeValidity with the same threshold as rescue,
  // not IsAmplitudeSquareLegal() which has a stricter threshold (sqrt(DBL_MIN) vs DBL_MIN).
  // For rank 1 with amplitude ~1e-300: 1e-300 > DBL_MIN (valid) but < sqrt(DBL_MIN) (~1.5e-154).
  EXPECT_TRUE(CheckWaveFunctionAmplitudeValidity(
      engine.WavefuncComp(),
      rescue_params.amplitude_min_threshold,
      rescue_params.amplitude_max_threshold))
      << "Rank " << rank << " should have valid amplitude after warmup";
}

/**
 * @brief Verify amplitude magnitudes for different configurations
 * 
 * This test verifies our test setup is correct:
 * - Neel config {{0,1},{1,0}} should have O(1) amplitude
 * - Anti-Neel config {{1,0},{0,1}} should have ~β amplitude (very small)
 * - Third/Fourth Sz=0 configs should throw exception (not in wavefunction)
 */
TEST_F(TestConfigurationRescueSum, VerifyAmplitudeMagnitudes) {
  const TenElemT beta = 1e-100;  // Not too small to avoid underflow issues
  auto sitps = CreateSuperpositionTPS(beta);

  PEPSParams peps_params(BMPSTruncateParams<QLTEN_Double>(
      Dpeps, 2 * Dpeps, 1e-15,
      CompressMPSScheme::SVD_COMPRESS,
      std::make_optional<double>(1e-14),
      std::make_optional<size_t>(10)));

  // Use default thresholds for amplitude validity checks
  const double amp_min = std::numeric_limits<double>::min();
  const double amp_max = std::numeric_limits<double>::max();

  // Test Neel configuration amplitude (should be O(1))
  {
    Configuration neel_config = CreateNeelConfig();
    TPSWaveFunctionComponent<TenElemT, QNT> wfc(sitps, neel_config, peps_params.GetBMPSParams());
    double amp_mag = std::abs(wfc.amplitude);
    
    if (rank == 0) {
      std::cout << "Neel config {{0,1},{1,0}} amplitude: " << amp_mag << std::endl;
    }
    EXPECT_GT(amp_mag, 0.1) << "Neel config should have O(1) amplitude";
    EXPECT_TRUE(CheckWaveFunctionAmplitudeValidity(wfc, amp_min, amp_max));
  }

  // Test anti-Neel configuration amplitude (should be ~beta)
  {
    Configuration anti_neel_config = CreateAntiNeelConfig();
    TPSWaveFunctionComponent<TenElemT, QNT> wfc(sitps, anti_neel_config, peps_params.GetBMPSParams());
    double amp_mag = std::abs(wfc.amplitude);
    
    if (rank == 0) {
      std::cout << "Anti-Neel config {{1,0},{0,1}} amplitude: " << amp_mag << std::endl;
    }
    // The amplitude should be very small (proportional to beta)
    EXPECT_LT(amp_mag, 1e-50) << "Anti-Neel config should have tiny amplitude";
  }

  // Test third Sz=0 configuration - should throw exception (not in wavefunction)
  {
    Configuration third_config = CreateThirdSz0Config();
    auto throw_third = [&]() {
      TPSWaveFunctionComponent<TenElemT, QNT> wfc(sitps, third_config, peps_params.GetBMPSParams());
    };
    EXPECT_THROW(throw_third(), std::runtime_error) 
        << "Third config should throw exception (outside wavefunction support)";
    
    if (rank == 0) {
      std::cout << "Third Sz=0 config {{0,0},{1,1}}: correctly throws exception" << std::endl;
    }
  }

  // Test fourth Sz=0 configuration - should also throw exception
  {
    Configuration fourth_config = CreateFourthSz0Config();
    auto throw_fourth = [&]() {
      TPSWaveFunctionComponent<TenElemT, QNT> wfc(sitps, fourth_config, peps_params.GetBMPSParams());
    };
    EXPECT_THROW(throw_fourth(), std::runtime_error) 
        << "Fourth config should throw exception (outside wavefunction support)";
    
    if (rank == 0) {
      std::cout << "Fourth Sz=0 config {{1,1},{0,0}}: correctly throws exception" << std::endl;
    }
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  return RUN_ALL_TESTS();
}
