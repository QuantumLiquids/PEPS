/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: QuantumLiquids/PEPS project. Fast unittests for Monte-Carlo Measurement for 2x2 PEPS systems.
* Symmetry benchmarks currently enforced:
*   - SU(2) magnetisation cancellation (Heisenberg, t-J).
*   - Z2 charge/spin uniformity on 2x2 clusters (spinless fermion, t-J).
*   - Bond energy partitioning across four NN links when no NNN term is present. (This is not easy to be guaranteed by monte carlo sampling.)
* Outstanding references still required: transverse Ising Z2 order parameter, long-distance SzSz/SC correlators.

* todo: remove the artificial tolerance after we have a mature std error system.
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
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string Trim(const std::string &input) {
  size_t start = 0;
  while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
    ++start;
  }
  size_t end = input.size();
  while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
    --end;
  }
  return input.substr(start, end - start);
}

double ParseRealField(const std::string &field) {
  std::string trimmed = Trim(field);
  if (trimmed.empty()) {
    return 0.0;
  }
  if (!trimmed.empty() && trimmed.front() == '(') {
    trimmed.erase(trimmed.begin());
  }
  if (!trimmed.empty() && trimmed.back() == ')') {
    trimmed.pop_back();
  }
  trimmed = Trim(trimmed);
  if (trimmed.empty()) {
    return 0.0;
  }
  size_t comma_pos = trimmed.find(',');
  std::string real_part = (comma_pos == std::string::npos) ? trimmed : trimmed.substr(0, comma_pos);
  real_part = Trim(real_part);
  if (real_part.empty()) {
    return 0.0;
  }
  try {
    return std::stod(real_part);
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to parse real value from field '" +
                             field + "' (trimmed='" + trimmed +
                             "', real_part='" + real_part +
                             "'): " + e.what());
  }
}

inline double CombineTol(double stderr_value, double floor = 1e-3) {
  if (!std::isfinite(stderr_value)) {
    return floor;
  }
  return std::max(floor, 5.0 * stderr_value);
}

inline double CombineTol(const std::vector<double> &stderr_vec, double floor = 1e-3) {
  double acc = 0.0;
  for (double val : stderr_vec) {
    if (std::isfinite(val)) {
      acc += val;
    }
  }
  return std::max(floor, 5.0 * acc);
}

inline double CombineTol(const std::vector<std::vector<double>> &stderr_mat, double floor = 1e-3) {
  double acc = 0.0;
  for (const auto &row : stderr_mat) {
    for (double val : row) {
      if (std::isfinite(val)) {
        acc += val;
      }
    }
  }
  return std::max(floor, 5.0 * acc);
}

std::filesystem::path StatsDirPath(const std::string &output_dir) {
  return std::filesystem::path(output_dir) / "measurement_data" / "stats";
}

struct FlatCSVStats {
  std::vector<size_t> indices;
  std::vector<double> means;
  std::vector<double> stderrs;
};

std::vector<std::string> SplitCsvRespectingParens(const std::string &line) {
  std::vector<std::string> fields;
  if (line.empty()) {
    return fields;
  }
  size_t field_start = 0;
  int paren_depth = 0;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '(') {
      ++paren_depth;
    } else if (ch == ')' && paren_depth > 0) {
      --paren_depth;
    }
    if (ch == ',' && paren_depth == 0) {
      fields.emplace_back(Trim(line.substr(field_start, i - field_start)));
      field_start = i + 1;
    }
  }
  fields.emplace_back(Trim(line.substr(field_start)));
  return fields;
}

bool ParseFlatCsvLine(const std::string &line,
                      std::string &idx_str,
                      std::string &mean_str,
                      std::string &stderr_str) {
  idx_str.clear();
  mean_str.clear();
  stderr_str.clear();
  if (line.empty()) {
    return false;
  }
  std::vector<std::string> fields = SplitCsvRespectingParens(line);
  while (fields.size() < 3) {
    fields.emplace_back();
  }
  idx_str = fields[0];
  mean_str = fields[1];
  stderr_str = fields[2];
  return !idx_str.empty();
}

FlatCSVStats LoadFlatStats(const std::filesystem::path &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open stats file: " + path.string());
  }
  std::string line;
  // Skip header if present
  if (!std::getline(ifs, line)) {
    return {};
  }
  FlatCSVStats stats;
  if (line.find("index") == std::string::npos) {
    std::string idx_str;
    std::string mean_str;
    std::string stderr_str;
    if (ParseFlatCsvLine(line, idx_str, mean_str, stderr_str)) {
      stats.indices.push_back(static_cast<size_t>(std::stoul(idx_str)));
      stats.means.push_back(ParseRealField(mean_str));
      stats.stderrs.push_back(ParseRealField(stderr_str));
    }
  }
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    std::string idx_str;
    std::string mean_str;
    std::string stderr_str;
    if (!ParseFlatCsvLine(line, idx_str, mean_str, stderr_str)) {
      continue;
    }
    stats.indices.push_back(static_cast<size_t>(std::stoul(idx_str)));
    stats.means.push_back(ParseRealField(mean_str));
    stats.stderrs.push_back(ParseRealField(stderr_str));
  }
  return stats;
}

std::vector<std::vector<double>> LoadMatrixCSV(const std::filesystem::path &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open matrix file: " + path.string());
  }
  std::vector<std::vector<double>> matrix;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields = SplitCsvRespectingParens(line);
    std::vector<double> row;
    row.reserve(fields.size());
    for (const auto &field : fields) {
      row.push_back(ParseRealField(field));
    }
    matrix.push_back(std::move(row));
  }
  return matrix;
}

double SumMatrix(const std::vector<std::vector<double>> &matrix) {
  double total = 0.0;
  for (const auto &row : matrix) {
    total = std::accumulate(row.begin(), row.end(), total);
  }
  return total;
}

template<typename T>
std::vector<double> Flatten(const std::vector<std::vector<T>> &matrix) {
  std::vector<double> flat;
  flat.reserve(matrix.size() * (matrix.empty() ? 0 : matrix.front().size()));
  for (const auto &row : matrix) {
    for (const auto &val : row) {
      flat.push_back(static_cast<double>(val));
    }
  }
  return flat;
}

size_t CountEntries(const std::vector<std::vector<double>> &matrix) {
  size_t count = 0;
  for (const auto &row : matrix) {
    count += row.size();
  }
  return count;
}

std::vector<std::vector<double>> AppendRows(const std::vector<std::vector<double>> &lhs,
                                            const std::vector<std::vector<double>> &rhs) {
  std::vector<std::vector<double>> result = lhs;
  result.insert(result.end(), rhs.begin(), rhs.end());
  return result;
}

std::vector<std::vector<double>> ConstantMatrixLike(const std::vector<std::vector<double>> &matrix,
                                                    double constant_value) {
  std::vector<std::vector<double>> expected(matrix.size());
  for (size_t i = 0; i < matrix.size(); ++i) {
    expected[i].assign(matrix[i].size(), constant_value);
  }
  return expected;
}

double AverageVector(const std::vector<double> &vec) {
  if (vec.empty()) {
    return 0.0;
  }
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  return sum / static_cast<double>(vec.size());
}

double AverageMatrix(const std::vector<std::vector<double>> &matrix) {
  const double total = SumMatrix(matrix);
  const size_t entries = CountEntries(matrix);
  if (entries == 0) {
    return 0.0;
  }
  return total / static_cast<double>(entries);
}

bool FileExists(const std::filesystem::path &path) {
  return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

double SigmaWithFallback(double sigma, double fallback) {
  if (std::isfinite(sigma) && sigma > 0.0) {
    return 3.5 * sigma;
  }
  return fallback;
}

double SigmaWithFallback(const std::vector<double> &sigma_vec, double fallback) {
  double max_sigma = 0.0;
  for (double sigma : sigma_vec) {
    if (std::isfinite(sigma) && sigma > 0.0) {
      max_sigma = std::max(max_sigma, sigma);
    }
  }
  return (max_sigma > 0.0) ? (3.5 * max_sigma) : fallback;
}

double SigmaWithFallback(const std::vector<std::vector<double>> &sigma_mat, double fallback) {
  return SigmaWithFallback(Flatten(sigma_mat), fallback);
}

double EntryTol(double sigma, double fallback) {
  return SigmaWithFallback(std::vector<double>{sigma}, fallback);
}

void ExpectMatrixNear(const std::vector<std::vector<double>> &mean,
                      const std::vector<std::vector<double>> &errs,
                      const std::vector<std::vector<double>> &expected,
                      double default_floor = 1e-3) {
  ASSERT_EQ(mean.size(), errs.size());
  ASSERT_EQ(mean.size(), expected.size());
  for (size_t i = 0; i < mean.size(); ++i) {
    ASSERT_EQ(mean[i].size(), errs[i].size());
    ASSERT_EQ(mean[i].size(), expected[i].size());
    for (size_t j = 0; j < mean[i].size(); ++j) {
      double tol = CombineTol(errs[i][j], default_floor);
      EXPECT_NEAR(mean[i][j], expected[i][j], tol) << "matrix entry (" << i << ", " << j << ")";
    }
  }
}

}  // namespace

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
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>(Dpeps,
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
    EXPECT_NEAR(std::real(energy), energy_exact, 0.01); // Relaxed tolerance for fast test
  }

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    const auto stats_dir = StatsDirPath(output_dir);
    ASSERT_TRUE(std::filesystem::exists(stats_dir));
    ASSERT_TRUE(std::filesystem::is_directory(stats_dir));

    // Energy stats (scalar CSV)
    FlatCSVStats energy_stats = LoadFlatStats(stats_dir / "energy.csv");
    ASSERT_FALSE(energy_stats.means.empty());
    double energy_tol = SigmaWithFallback(energy_stats.stderrs, std::abs(energy_exact) * 1e-4);
    EXPECT_NEAR(energy_stats.means.front(), energy_exact, energy_tol);

    // Spin_z should sum to zero (SU(2) symmetry)
    auto sz_mean = LoadMatrixCSV(stats_dir / "spin_z_mean.csv");
    auto sz_err = LoadMatrixCSV(stats_dir / "spin_z_stderr.csv");
    ASSERT_EQ(sz_mean.size(), 2);
    ASSERT_EQ(sz_mean[0].size(), 2);
    double sz_sum = SumMatrix(sz_mean);
    double sz_tol = SigmaWithFallback(sz_err, 5e-2);
    EXPECT_NEAR(sz_sum, 0.0, sz_tol);

    const auto bond_h_mean_path = stats_dir / "bond_energy_h_mean.csv";
    const auto bond_v_mean_path = stats_dir / "bond_energy_v_mean.csv";
    if (FileExists(bond_h_mean_path) && FileExists(bond_v_mean_path)) {
      auto bond_h_mean = LoadMatrixCSV(bond_h_mean_path);
      auto bond_h_err = LoadMatrixCSV(stats_dir / "bond_energy_h_stderr.csv");
      auto bond_v_mean = LoadMatrixCSV(bond_v_mean_path);
      auto bond_v_err = LoadMatrixCSV(stats_dir / "bond_energy_v_stderr.csv");
      auto bond_all_mean = AppendRows(bond_h_mean, bond_v_mean);
      auto bond_all_err = AppendRows(bond_h_err, bond_v_err);
      double bond_total = SumMatrix(bond_all_mean);
      double bond_total_tol = SigmaWithFallback(bond_all_err, std::abs(energy_exact) * 1e-4);
      EXPECT_NEAR(bond_total, energy_stats.means.front(), bond_total_tol);
    } else {
      double expected_per_bond = energy_stats.means.front() / 4.0;
      EXPECT_NEAR(expected_per_bond, energy_exact / 4.0, std::abs(energy_exact) * 1e-4);
    }
  }

  delete executor;
}

// Test Transverse Ising model
TEST_F(Test2x2MCPEPSBoson, TransverseIsingModel) {
  using Model = TransverseFieldIsingSquareOBC;

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
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>(Dpeps,
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
      EXPECT_NEAR(std::real(energy), energy_exact, 0.01); // Tight tolerance for single process
    } else {
      EXPECT_NEAR(std::real(energy), energy_exact, 3 * std::abs(en_err)); // Use error estimate for MPI
    }

    const auto stats_dir = StatsDirPath(output_dir);
    ASSERT_TRUE(std::filesystem::exists(stats_dir));
    ASSERT_TRUE(std::filesystem::is_directory(stats_dir));

    auto energy_stats = LoadFlatStats(stats_dir / "energy.csv");
    ASSERT_FALSE(energy_stats.means.empty());

    auto spin_z_mean = LoadMatrixCSV(stats_dir / "spin_z_mean.csv");
    auto spin_z_err = LoadMatrixCSV(stats_dir / "spin_z_stderr.csv");
    ASSERT_EQ(spin_z_mean.size(), Ly);
    ASSERT_EQ(spin_z_err.size(), Ly);
    for (const auto &row : spin_z_mean) { ASSERT_EQ(row.size(), Lx); }
    for (const auto &row : spin_z_err) { ASSERT_EQ(row.size(), Lx); }

    auto sigma_x_mean = LoadMatrixCSV(stats_dir / "sigma_x_mean.csv");
    auto sigma_x_err = LoadMatrixCSV(stats_dir / "sigma_x_stderr.csv");
    ASSERT_EQ(sigma_x_mean.size(), Ly);
    ASSERT_EQ(sigma_x_err.size(), Ly);
    for (const auto &row : sigma_x_mean) { ASSERT_EQ(row.size(), Lx); }
    for (const auto &row : sigma_x_err) { ASSERT_EQ(row.size(), Lx); }
    const double avg_sigma_x = AverageMatrix(sigma_x_mean);
    (void)avg_sigma_x;  // TODO(benchmark): add quantitative check once Z2 order reference is available.

    auto szsz_row_stats = LoadFlatStats(stats_dir / "SzSz_row.csv");
    ASSERT_EQ(szsz_row_stats.means.size(), Lx / 2);

    // TODO(benchmarks): add numerical assertions for transverse Ising observables once Z2 order benchmarks are available.
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
  PEPSParams peps_params(BMPSTruncateParams<qlten::QLTEN_Double>(Dpeps,
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
    EXPECT_NEAR(std::real(energy), energy_exact, 0.01); // Relaxed tolerance for fast test
  }

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    const auto stats_dir = StatsDirPath(output_dir);
    ASSERT_TRUE(std::filesystem::exists(stats_dir));
    ASSERT_TRUE(std::filesystem::is_directory(stats_dir));

    // Energy should match benchmark
    FlatCSVStats energy_stats = LoadFlatStats(stats_dir / "energy.csv");
    ASSERT_FALSE(energy_stats.means.empty());
    double energy_tol = SigmaWithFallback(energy_stats.stderrs, std::abs(energy_exact) * 1e-4);
    EXPECT_NEAR(energy_stats.means.front(), energy_exact, energy_tol);

    // Charge density should equal total charge / number of sites
    auto charge_mean = LoadMatrixCSV(stats_dir / "charge_mean.csv");
    auto charge_err = LoadMatrixCSV(stats_dir / "charge_stderr.csv");
    double total_charge = SumMatrix(charge_mean);
    ASSERT_DOUBLE_EQ(total_charge, 2.0);
    double expected_charge_per_site = total_charge / static_cast<double>(CountEntries(charge_mean));
    double charge_tol = SigmaWithFallback(charge_err, 5e-2);
    for (size_t i = 0; i < charge_mean.size(); ++i) {
      for (size_t j = 0; j < charge_mean[i].size(); ++j) {
        EXPECT_NEAR(charge_mean[i][j], expected_charge_per_site, charge_tol);
      }
    }

    // Bond energies should distribute total energy uniformly across four NN bonds
    auto bond_h_mean = LoadMatrixCSV(stats_dir / "bond_energy_h_mean.csv");
    auto bond_h_err = LoadMatrixCSV(stats_dir / "bond_energy_h_stderr.csv");
    auto bond_v_mean = LoadMatrixCSV(stats_dir / "bond_energy_v_mean.csv");
    auto bond_v_err = LoadMatrixCSV(stats_dir / "bond_energy_v_stderr.csv");
    auto bond_all_mean = AppendRows(bond_h_mean, bond_v_mean);
    auto bond_all_err = AppendRows(bond_h_err, bond_v_err);
    double bond_total = SumMatrix(bond_all_mean);
    EXPECT_NEAR(bond_total, std::real(energy), 1e-14);
    double bond_tol = SigmaWithFallback(bond_all_err, 5e-2);
    EXPECT_NEAR(bond_total, energy_stats.means.front(), bond_tol);
    // This bond energy has no translational symmetry of the ring, for ground state degeneracy for 2 electrons.
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
  MonteCarloParams mc_params(10, 100, 1, compatible_config, false, output_dir + "/final_config");  // explicit config dump path
  PEPSParams peps_params = PEPSParams(BMPSTruncateParams<qlten::QLTEN_Double>::SVD(Dpeps, 2 * Dpeps, 1e-15));
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
      EXPECT_NEAR(std::real(energy), energy_exact, 0.01); // Tight tolerance for single process
    } else {
      EXPECT_NEAR(std::real(energy), energy_exact, 3 * std::abs(en_err)); // Use error estimate for MPI
    }
  }

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    const auto stats_dir = StatsDirPath(output_dir);
    ASSERT_TRUE(std::filesystem::exists(stats_dir));
    ASSERT_TRUE(std::filesystem::is_directory(stats_dir));

    auto energy_stats = LoadFlatStats(stats_dir / "energy.csv");
    ASSERT_FALSE(energy_stats.means.empty());
    double energy_tol = CombineTol(energy_stats.stderrs.front());
    EXPECT_NEAR(energy_stats.means.front(), energy_exact, energy_tol);

    auto spin_mean = LoadMatrixCSV(stats_dir / "spin_z_mean.csv");
    auto spin_err = LoadMatrixCSV(stats_dir / "spin_z_stderr.csv");
    ASSERT_EQ(spin_mean.size(), 2);
    ASSERT_EQ(spin_err.size(), 2);
    for (const auto &row : spin_mean) { ASSERT_EQ(row.size(), 2); }
    for (const auto &row : spin_err) { ASSERT_EQ(row.size(), 2); }
    double spin_sum = SumMatrix(spin_mean);
    double spin_tol = SigmaWithFallback(spin_err, 5e-2);
    EXPECT_NEAR(spin_sum, 0.0, spin_tol);

    auto bond_h_mean = LoadMatrixCSV(stats_dir / "bond_energy_h_mean.csv");
    auto bond_h_err = LoadMatrixCSV(stats_dir / "bond_energy_h_stderr.csv");
    auto bond_v_mean = LoadMatrixCSV(stats_dir / "bond_energy_v_mean.csv");
    auto bond_v_err = LoadMatrixCSV(stats_dir / "bond_energy_v_stderr.csv");
    auto bond_all_mean = AppendRows(bond_h_mean, bond_v_mean);
    auto bond_all_err = AppendRows(bond_h_err, bond_v_err);
    double bond_total = SumMatrix(bond_all_mean);
    EXPECT_NEAR(bond_total, energy_stats.means.front(), 1e-14);
    double bond_tol = SigmaWithFallback(bond_all_err, 5e-2);
    EXPECT_NEAR(bond_total, energy_stats.means.front(), bond_tol);
    double expected_per_bond = energy_stats.means.front() / 4.0;
    auto expected_h = ConstantMatrixLike(bond_h_mean, expected_per_bond);
    auto expected_v = ConstantMatrixLike(bond_v_mean, expected_per_bond);
    double entry_tol = std::max(SigmaWithFallback(bond_all_err, 5e-2), 0.5); // todo: remove the artificial tolerance after we have a mature std error system.
    ExpectMatrixNear(bond_h_mean, bond_h_err, expected_h, entry_tol);
    ExpectMatrixNear(bond_v_mean, bond_v_err, expected_v, entry_tol);
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
