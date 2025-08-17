/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: QuantumLiquids/PEPS project. Fast unittests for Monte-Carlo Measurement for 2x2 PEPS systems.
* This test covers multiple models (Heisenberg, Transverse Ising, Spinless Fermions) with quick execution.
*
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurement.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"

#include "../test_mpi_env.h"
#include "../utilities.h"
using namespace qlten;
using namespace qlpeps;

#if TEN_ELEM_TYPE_NUM == 1
std::string data_type_in_file_name = "Double";
#elif TEN_ELEM_TYPE_NUM == 2
std::string data_type_in_file_name = "Complex";
#else
#error "Unexpected TEN_ELEM_TYPE_NUM"
#endif

/**
 * @brief Get the correct TPS data path based on tensor element type
 * @param base_name Base name of the TPS data (e.g., "heisenberg_tps")
 * @return Full path to the TPS data directory
 */
std::string GetTPSDataPath(const std::string &base_name) {
  if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
    return "test_algorithm/test_data/" + base_name + "_doublelowest";
  } else if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Complex>) {
    return "test_algorithm/test_data/" + base_name + "_complexlowest";
  } else {
    return "test_algorithm/test_data/" + base_name + "_unknownlowest";
  }
}

#include <filesystem>

/**
 * @brief Computes the ground state energy of spinless free fermions
 *        on a 2x2 OBC square lattice with NN and NNN hopping.
 */
double Calculate2x2OBCSpinlessFreeFermionEnergy(double t, double t2) {
  std::vector<double> k_values = {0.0, M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0};
  std::vector<double> single_particle_energies;
  for (auto k : k_values) {
    double energy = -2.0 * t * std::cos(k) - t2 * std::cos(2.0 * k);
    single_particle_energies.push_back(energy);
  }
  double ground_state_energy = 0.0;
  for (auto energy : single_particle_energies) {
    ground_state_energy += (double) (energy < 0) * energy;
  }
  return ground_state_energy;
}

/**
 * @brief The ground state energy of spin-1/2 Heisenberg model
 *        on a 2x2 square lattice with OBC boundary conditions.
 */
double Calculate2x2HeisenbergEnergy(double J) {
  return -2 * J;
}

/**
 * @brief Computes the exact ground state energy of transverse Ising model
 *        on a 2x2 OBC square lattice.
 */
double Calculate2x2OBCTransverseIsingEnergy(double J, double h) {
  std::vector<double> k_values = {M_PI / 4.0, 3.0 * M_PI / 4.0};
  double ground_state_energy = 0.0;
  for (auto k : k_values) {
    double epsilon_k = 2.0 * std::sqrt(J * J + h * h - 2.0 * J * h * std::cos(k));
    ground_state_energy -= epsilon_k;
  }
  return ground_state_energy;
}

// Test fixture for boson systems (Heisenberg, Transverse Ising)
struct Test2x2MCPEPSBoson : MPITest {
  using QNT = qlten::special_qn::TrivialRepQN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;
  using TenElemT = TEN_ELEM_TYPE;

  size_t Lx = 2;
  size_t Ly = 2;
  size_t N = Lx * Ly;
  size_t Dpeps = 4; // Smaller bond dimension for fast testing

  void SetUp(void) {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);
  }
};

// Test fixture for fermion systems (Spinless Fermion, t-J)
struct Test2x2MCPEPSFermion : MPITest {
  using QNT = qlten::special_qn::fZ2QN;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QNSctVecT = QNSectorVec<QNT>;
  using TenElemT = TEN_ELEM_TYPE;
  using TPSSampleFlipT = MCUpdateSquareNNExchange;

  size_t Lx = 2;
  size_t Ly = 2;
  size_t N = Lx * Ly;
  size_t Dpeps = 4; // Smaller bond dimension for fast testing

  void SetUp(void) {
    MPITest::SetUp();
    qlten::hp_numeric::SetTensorManipulationThreads(1);
  }
};

// Test Heisenberg model
TEST_F(Test2x2MCPEPSBoson, HeisenbergModel) {
  using Model = SquareSpinOneHalfXXZModel;

  double J = 1.0;
  double energy_exact = Calculate2x2HeisenbergEnergy(J);

  // Monte Carlo measurement parameters
  // Use a configuration compatible with the TPS data generation
  // The TPS was generated with alternating spins {{0, 1}, {1, 0}}
  Configuration compatible_config(2, 2);
  compatible_config({0, 0}) = 0; // up
  compatible_config({0, 1}) = 1; // down  
  compatible_config({1, 0}) = 1; // down
  compatible_config({1, 1}) = 0; // up
  // This gives us 2 up spins and 2 down spins: [0, 1, 1, 0]

  // TODO: DESIGN PROBLEM - Configuration read/write paths are the same!
  // MonteCarloMeasurementExecutor writes config files to wavefunction_path,
  // which pollutes the source directory. Need to separate read/write paths.
  // Future: Let user control config/TPS read/write instead of automatic behavior.
  
  // TEMPORARY FIX: Use test output directory for wavefunction_path to avoid pollution
  std::string source_tps_path = (std::filesystem::path(TEST_SOURCE_DIR) / GetTPSDataPath("heisenberg_tps")).string();
  std::string working_tps_path = GetTestOutputPath("mc_peps_measure", "heisenberg_tps_work");
  
  // Copy source TPS data to working directory
  if (std::filesystem::exists(source_tps_path)) {
    std::filesystem::copy(source_tps_path, working_tps_path,
                          std::filesystem::copy_options::overwrite_existing |
                          std::filesystem::copy_options::recursive);
  }

  MCMeasurementPara para = MCMeasurementPara(
    BMPSTruncatePara(Dpeps,
                     2 * Dpeps,
                     1e-15,
                     CompressMPSScheme::SVD_COMPRESS,
                     std::make_optional<double>(1e-14),
                     std::make_optional<size_t>(10)),
    50,
    50,
    1,
    // Use compatible configuration instead of random
    compatible_config,
    // Use working directory (safe for writes) instead of source directory
    working_tps_path);

  Model heisenberg_model(J, J, 0);

  auto executor = new MonteCarloMeasurementExecutor<TenElemT, QNT, MCUpdateSquareNNExchange, Model>(
    para,
    Ly,
    Lx,
    comm,
    heisenberg_model);

  // Check that configuration has exactly 2 up and 2 down spins
  size_t count_0 = 0, count_1 = 0;
  for (size_t row = 0; row < 2; row++) {
    for (size_t col = 0; col < 2; col++) {
      if (executor->GetCurrentConfiguration()({row, col}) == 0) count_0++;
      else if (executor->GetCurrentConfiguration()({row, col}) == 1) count_1++;
    }
  }
  EXPECT_EQ(count_0, 2); // Exactly 2 up spins
  EXPECT_EQ(count_1, 2); // Exactly 2 down spins

  executor->Execute();

  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == kMPIMasterRank) {
    EXPECT_NEAR(Real(energy), energy_exact, 0.01); // Relaxed tolerance for fast test
  }

  delete executor;
}

// Test Transverse Ising model
TEST_F(Test2x2MCPEPSBoson, TransverseIsingModel) {
  using Model = TransverseIsingSquare;

  double J = 1.0;
  double h = 1.0;
  double energy_exact = Calculate2x2OBCTransverseIsingEnergy(J, h);

  // TODO: Same design problem - separate read/write paths needed
  // Load TPS data from file and copy to working directory
  std::string source_tps_path = (std::filesystem::path(TEST_SOURCE_DIR) /
    GetTPSDataPath("transverse_ising_tps")).string();
  std::string working_tps_path = GetTestOutputPath("mc_peps_measure", "transverse_ising_tps_work");
  
  // Copy source TPS data to working directory
  if (std::filesystem::exists(source_tps_path)) {
    std::filesystem::copy(source_tps_path, working_tps_path,
                          std::filesystem::copy_options::overwrite_existing |
                          std::filesystem::copy_options::recursive);
  }
  std::string tps_path = working_tps_path;

  SplitIndexTPS<TenElemT, QNT> sitps(Ly, Lx);
  sitps.Load(tps_path);

  // Monte Carlo measurement parameters
  MCMeasurementPara para = MCMeasurementPara(
    BMPSTruncatePara(Dpeps,
                     2 * Dpeps,
                     1e-15,
                     CompressMPSScheme::SVD_COMPRESS,
                     std::make_optional<double>(1e-14),
                     std::make_optional<size_t>(10)),
    50,
    50,
    1,
    // Reduced samples for fast testing
    std::vector<size_t>(2, N / 2),
    Ly,
    Lx);

  Model transverse_ising_model(h);

  auto executor = new MonteCarloMeasurementExecutor<TenElemT, QNT, MCUpdateSquareNNFullSpaceUpdate, Model>(
    para,
    sitps,
    comm,
    transverse_ising_model);

  executor->Execute();
  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == kMPIMasterRank) {
    // For single process, use tight tolerance; for MPI, use error estimate
    if (mpi_size == 1) {
      EXPECT_NEAR(Real(energy), energy_exact, 0.01); // Tight tolerance for single process
    } else {
      EXPECT_NEAR(Real(energy), energy_exact, 3 * std::abs(en_err)); // Use error estimate for MPI
    }
  }

  delete executor;
}

// Test Spinless Fermion model
TEST_F(Test2x2MCPEPSFermion, SpinlessFermionModel) {
  using Model = SquareSpinlessFermion;

  double t = 1.0;
  double t2 = 0.0; // No NNN hopping for simplicity
  double energy_exact = Calculate2x2OBCSpinlessFreeFermionEnergy(t, t2);

  // Monte Carlo measurement parameters
  // Use a configuration compatible with the TPS data generation
  // For spinless fermion: 0=empty, 1=occupied, half-filling means 2 occupied sites
  Configuration compatible_config(2, 2);
  compatible_config({0, 0}) = 1; // occupied
  compatible_config({0, 1}) = 0; // empty  
  compatible_config({1, 0}) = 0; // empty
  compatible_config({1, 1}) = 1; // occupied
  // This gives us 2 occupied sites and 2 empty sites: [1, 0, 0, 1]

  MCMeasurementPara para = MCMeasurementPara(
    BMPSTruncatePara(Dpeps,
                     2 * Dpeps,
                     1e-15,
                     CompressMPSScheme::SVD_COMPRESS,
                     std::make_optional<double>(1e-14),
                     std::make_optional<size_t>(10)),
    50,
    50,
    1,
    // Use compatible configuration instead of random
    compatible_config,
    // Load TPS data from path (generated by test_exact_sum_optimization.cpp)
    // TODO: Same design problem - copy to working directory
    [&]() {
      std::string source = (std::filesystem::path(TEST_SOURCE_DIR) / GetTPSDataPath("spinless_fermion_tps_t2_0.000000")).string();
      std::string working = GetTestOutputPath("mc_peps_measure", "spinless_fermion_work");
      if (std::filesystem::exists(source)) {
        std::filesystem::copy(source, working,
                              std::filesystem::copy_options::overwrite_existing |
                              std::filesystem::copy_options::recursive);
      }
      return working;
    }());

  Model fermion_model(t, t2, 0);

  auto executor = new MonteCarloMeasurementExecutor<TenElemT, QNT, MCUpdateSquareNNExchange, Model>(
    para,
    Ly,
    Lx,
    comm,
    fermion_model);

  executor->Execute();
  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == kMPIMasterRank) {
    EXPECT_NEAR(Real(energy), energy_exact, 0.01); // Relaxed tolerance for fast test
  }

  delete executor;
}

// Test t-J model
TEST_F(Test2x2MCPEPSFermion, TJModel) {
  using Model = SquaretJVModel;

  double t = 1.0;
  double J = 0.3;
  double mu = 0.0;
  double energy_exact = -2.9431635706137875; // Pre-calculated exact energy

  // Monte Carlo measurement parameters
  // Use a configuration compatible with the TPS data generation
  // For t-J model: 0=up, 1=down, 2=empty, configuration should be [2, 2, 0, 1]
  Configuration compatible_config(2, 2);
  compatible_config({0, 0}) = 2; // empty
  compatible_config({0, 1}) = 2; // empty  
  compatible_config({1, 0}) = 0; // up
  compatible_config({1, 1}) = 1; // down
  // This gives us 1 up, 1 down, 2 empty sites: [2, 2, 0, 1]

  MCMeasurementPara para = MCMeasurementPara(
    BMPSTruncatePara(Dpeps,
                     2 * Dpeps,
                     1e-15,
                     CompressMPSScheme::SVD_COMPRESS,
                     std::make_optional<double>(1e-14),
                     std::make_optional<size_t>(10)),
    50,
    50,
    1,
    // Use compatible configuration instead of random
    compatible_config,
    // Load TPS data from path (generated by test_exact_sum_optimization.cpp)
    // TODO: Same design problem - copy to working directory
    [&]() {
      std::string source = (std::filesystem::path(TEST_SOURCE_DIR) / GetTPSDataPath("tj_model_tps")).string();
      std::string working = GetTestOutputPath("mc_peps_measure", "tj_model_work");
      if (std::filesystem::exists(source)) {
        std::filesystem::copy(source, working,
                              std::filesystem::copy_options::overwrite_existing |
                              std::filesystem::copy_options::recursive);
      }
      return working;
    }());

  Model tj_model(t, 0, J, J / 4, mu);

  auto executor = new MonteCarloMeasurementExecutor<TenElemT, QNT, MCUpdateSquareNNExchange, Model>(
    para,
    Ly,
    Lx,
    comm,
    tj_model);

  executor->Execute();
  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == kMPIMasterRank) {
    // For single process, use tight tolerance; for MPI, use error estimate
    if (mpi_size == 1) {
      EXPECT_NEAR(Real(energy), energy_exact, 0.01); // Tight tolerance for single process
    } else {
      EXPECT_NEAR(Real(energy), energy_exact, 3 * std::abs(en_err)); // Use error estimate for MPI
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
