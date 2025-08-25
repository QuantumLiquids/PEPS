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
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_measurer.h"
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
    return "test_data/" + base_name + "_doublelowest";
  } else if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Complex>) {
    return "test_data/" + base_name + "_complexlowest";
  } else {
    return "test_data/" + base_name + "_unknownlowest";
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

  // Path separation problem SOLVED by refactor - user controls all dump paths explicitly
  std::string source_tps_path = (std::filesystem::path(TEST_SOURCE_DIR) / GetTPSDataPath("heisenberg_tps")).string();
  std::string output_dir = GetTestOutputPath("mc_peps_measure", "results");
  
  // Ensure subdirectories exist for dump paths
  std::filesystem::create_directories(output_dir);
  
  // Create unified parameter structure with explicit dump path control  
  MonteCarloParams mc_params(50, 50, 1, compatible_config, false, output_dir + "/final_config");  // explicit config dump path
  PEPSParams peps_params(BMPSTruncatePara(Dpeps,
                                          2 * Dpeps,
                                          1e-15,
                                          CompressMPSScheme::SVD_COMPRESS,
                                          std::make_optional<double>(1e-14),
                                          std::make_optional<size_t>(10)));
  MCMeasurementParams para(mc_params, peps_params, output_dir + "/measurement_data");  // explicit measurement dump path

  Model heisenberg_model(J, J, 0);

  auto executor = MCPEPSMeasurer<TenElemT, QNT, MCUpdateSquareNNExchange, Model>::CreateByLoadingTPS(
    source_tps_path,   // TPS path - convenient factory loads automatically
    para,              // Unified parameters
    comm,
    heisenberg_model).release(); // Convert unique_ptr to raw pointer for compatibility

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

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
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

  // Path separation problem SOLVED by refactor - use convenient file loading pattern
  std::string source_tps_path = (std::filesystem::path(TEST_SOURCE_DIR) / GetTPSDataPath("transverse_ising_tps")).string();
  std::string output_dir = GetTestOutputPath("mc_peps_measure", "transverse_ising_results");

  // Create random configuration explicitly
  Configuration random_config(Ly, Lx);
  random_config.Random(std::vector<size_t>(2, N / 2));  // Reduced samples for fast testing
  
  // Create unified parameter structure with explicit dump path control
  MonteCarloParams mc_params(50, 50, 1, random_config, false, output_dir + "/final_config");  // explicit config dump path
  PEPSParams peps_params(BMPSTruncatePara(Dpeps,
                                          2 * Dpeps,
                                          1e-15,
                                          CompressMPSScheme::SVD_COMPRESS,
                                          std::make_optional<double>(1e-14),
                                          std::make_optional<size_t>(10)));
  MCMeasurementParams para(mc_params, peps_params, output_dir + "/measurement_data");  // explicit measurement dump path

  Model transverse_ising_model(h);

  auto executor = MCPEPSMeasurer<TenElemT, QNT, MCUpdateSquareNNFullSpaceUpdate, Model>::CreateByLoadingTPS(
    source_tps_path,   // TPS path - convenient factory loads automatically
    para,              // Unified parameters  
    comm,
    transverse_ising_model).release(); // Convert unique_ptr to raw pointer for compatibility

  executor->Execute();
  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
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

  // Path separation problem SOLVED by refactor - use convenient file loading pattern
  std::string source_tps_path = (std::filesystem::path(TEST_SOURCE_DIR) / GetTPSDataPath("spinless_fermion_tps_t2_0.000000")).string();
  std::string output_dir = GetTestOutputPath("mc_peps_measure", "spinless_fermion_results");
  
  // Create unified parameter structure with explicit dump path control
  MonteCarloParams mc_params(50, 50, 1, compatible_config, false, output_dir + "/final_config");  // explicit config dump path
  PEPSParams peps_params(BMPSTruncatePara(Dpeps,
                                          2 * Dpeps,
                                          1e-15,
                                          CompressMPSScheme::SVD_COMPRESS,
                                          std::make_optional<double>(1e-14),
                                          std::make_optional<size_t>(10)));
  MCMeasurementParams para(mc_params, peps_params, output_dir + "/measurement_data");  // explicit measurement dump path

  Model fermion_model(t, t2, 0);

  auto executor = MCPEPSMeasurer<TenElemT, QNT, MCUpdateSquareNNExchange, Model>::CreateByLoadingTPS(
    source_tps_path,   // TPS path - convenient factory loads automatically
    para,              // Unified parameters
    comm,
    fermion_model).release(); // Convert unique_ptr to raw pointer for compatibility

  executor->Execute();
  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
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

  // Path separation problem SOLVED by refactor - use convenient file loading pattern
  std::string source_tps_path = (std::filesystem::path(TEST_SOURCE_DIR) / GetTPSDataPath("tj_model_tps")).string();
  std::string output_dir = GetTestOutputPath("mc_peps_measure", "tj_model_results");
  
  // Create unified parameter structure with explicit dump path control
  MonteCarloParams mc_params(50, 50, 1, compatible_config, false, output_dir + "/final_config");  // explicit config dump path
  PEPSParams peps_params(BMPSTruncatePara(Dpeps,
                                          2 * Dpeps,
                                          1e-15,
                                          CompressMPSScheme::SVD_COMPRESS,
                                          std::make_optional<double>(1e-14),
                                          std::make_optional<size_t>(10)));
  MCMeasurementParams para(mc_params, peps_params, output_dir + "/measurement_data");  // explicit measurement dump path

  Model tj_model(t, 0, J, J / 4, mu);

  auto executor = MCPEPSMeasurer<TenElemT, QNT, MCUpdateSquareNNExchange, Model>::CreateByLoadingTPS(
    source_tps_path,   // TPS path - convenient factory loads automatically
    para,              // Unified parameters
    comm,
    tj_model).release(); // Convert unique_ptr to raw pointer for compatibility

  executor->Execute();
  auto [energy, en_err] = executor->OutputEnergy();

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
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
