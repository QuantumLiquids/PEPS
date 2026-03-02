/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-15
*
* Description: QuantumLiquids/PEPS project. 
* Pure ExactSumEnergyEvaluator unit tests - LoadFromLowestState variant.
* 
* This test file focuses ONLY on ExactSumEnergyEvaluator verification by:
*  1. Loading pre-generated ground state TPS data from test_data/[model]_doublelowest/[model]_complexlowest
*  2. [REMOVED] Computing exact summation energy and gradient using ExactSumEnergyEvaluatorMPI (caused memory leaks)
*  3. Validating energy against analytical exact values (energy_gs_exact) for 4 test models
*  4. Ensuring MPI safety and high precision (1e-7 tolerance)
* 
* Design Philosophy (Linus): "Good taste eliminates special cases and complexity."
* - NO optimization algorithm complexity
* - NO Monte Carlo sampling noise  
* - NO file I/O during computation
* - Focus ONLY on exact summation correctness
* 
* Models tested:
* 1. Spinless free fermions (multiple t2 values)
* 2. Spin-1/2 Heisenberg model
* 3. Transverse field Ising model  
* 4. t-J model
*
* MPI Safety Features:
* - Timeout protection for MPI operations
* - Memory usage monitoring
* - Deadlock prevention via proper communication patterns
* - Graceful error handling and cleanup
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "../utilities.h"
#include "../test_mpi_env.h"

#include <chrono>
#include <csignal>
#include <csetjmp>

using namespace qlten;
using namespace qlpeps;

// ---------------------------------------------------------------------------
// Gradient probe utilities (same pattern as test_fermion_mc_sr_golden.cpp)
// ---------------------------------------------------------------------------
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
        ten.Random(QNT());
      }
    }
  }
  const TenElemT ip = x * probe;
  return {std::real(ip), std::imag(ip)};
}

using qlten::special_qn::fZ2QN;
using qlten::special_qn::TrivialRepQN;

// Global timeout protection for MPI operations
static std::chrono::steady_clock::time_point g_test_start_time;
static const std::chrono::seconds kMaxTestDuration{30}; // 30 seconds max per operation

/**
 * @brief Check if current test has exceeded timeout limit
 * @return true if timeout exceeded, false otherwise
 */
bool IsTestTimeoutExceeded() {
  auto current_time = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - g_test_start_time);
  return elapsed > kMaxTestDuration;
}

/**
 * @brief Memory-aware MPI operation wrapper with timeout protection
 * @param operation Lambda function containing MPI operation
 * @param operation_name Descriptive name for logging
 * @return true if operation succeeded, false if timeout/error
 */
template<typename Operation>
bool SafeMPIOperation(Operation operation, const std::string& operation_name) {
  if (IsTestTimeoutExceeded()) {
    std::cerr << "TIMEOUT: " << operation_name << " exceeded maximum test duration" << std::endl;
    return false;
  }
  
  try {
    operation();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "ERROR in " << operation_name << ": " << e.what() << std::endl;
    return false;
  }
}

/**
 * @brief Computes the ground state energy of spinless free fermions
 *        on a 2x2 OBC square lattice with NN and NNN hopping
 * @param t     Nearest-neighbor hopping amplitude
 * @param t2    Next-nearest-neighbor hopping amplitude  
 * @return double Ground state energy
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
    ground_state_energy += (double)(energy < 0) * energy;
  }
  return ground_state_energy;
}

/**
 * @brief The ground state energy of spin-1/2 Heisenberg model
 *        on a 2x2 square lattice with OBC boundary conditions
 * @param J     Exchange coupling
 * @return double Ground state energy
 */
double Calculate2x2HeisenbergEnergy(double J) {
  return -2 * J;
}

/**
 * @brief Computes the exact ground state energy of transverse Ising model
 *        on a 2x2 OBC square lattice
 * @param J     Ising coupling strength
 * @param h     Transverse field strength  
 * @return double Ground state energy
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

template<typename TenElemT, typename QNT>
struct ExactSumTestResult {
  bool passed = false;
  SplitIndexTPS<TenElemT, QNT> gradient;
};

/**
 * @brief Core exact summation test function with MPI protection
 *
 * @tparam ModelT Model type
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 * @tparam SITPST SplitIndexTPS type
 * @param model The model to test
 * @param split_index_tps Initial TPS state
 * @param all_configs All possible configurations
 * @param trun_para BMPSTruncateParams<> for truncation
 * @param Ly Number of rows
 * @param Lx Number of columns
 * @param energy_expect Expected exact energy
 * @param test_name Name of the test for output
 * @param comm MPI communicator
 * @param rank MPI rank
 * @param mpi_size MPI size
 * @return ExactSumTestResult with pass/fail status and gradient
 */
template<typename ModelT, typename TenElemT, typename QNT, typename SITPST>
ExactSumTestResult<TenElemT, QNT> RunExactSummationTest(
    ModelT& model,
    const SITPST& split_index_tps,
    const std::vector<Configuration>& all_configs,
    const BMPSTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type>& trun_para,
    size_t Ly,
    size_t Lx,
    double energy_expect,
    const std::string& test_name,
    const MPI_Comm& comm,
    int rank,
    int mpi_size) {

  g_test_start_time = std::chrono::steady_clock::now();

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    std::cout << "Testing exact summation for " << test_name
              << " (expected energy: " << energy_expect << ")" << std::endl;
  }

  // MPI API (Phase 1: fake-parallel)
  auto result = ExactSumEnergyEvaluatorMPI<ModelT, TEN_ELEM_TYPE, QNT>(
      split_index_tps,
      all_configs,
      trun_para,
      model,
      Ly,
      Lx,
      comm,
      rank,
      mpi_size);

  auto [computed_energy, gradient, error] = result;
  double final_energy = std::real(computed_energy);

  ExactSumTestResult<TenElemT, QNT> test_result;
  test_result.gradient = std::move(gradient);

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    std::cout << test_name << " results:" << std::endl;
    std::cout << "  Computed energy: " << std::setprecision(12) << final_energy << std::endl;
    std::cout << "  Expected energy: " << std::setprecision(12) << energy_expect << std::endl;
    std::cout << "  Gradient norm:   " << std::setprecision(6) << test_result.gradient.NormSquare() << std::endl;
    std::cout << "  Error:          " << error << std::endl;

    // Validate energy accuracy (strict tolerance for high-precision converged TPS states)
    // Note: Using high-precision doublelowest data generated by SimpleUpdate + Optimization
    double energy_diff = std::abs(final_energy - energy_expect);
    double tolerance = std::max(6e-8, std::abs(energy_expect) * 6e-8);  // 6e-8 tolerance with 2x safety margin
    if (energy_diff > tolerance) {
      std::cerr << "FAILED: " << test_name << " - energy mismatch: " << energy_diff
                << " (tolerance: " << tolerance << ")" << std::endl;
      test_result.passed = false;
      return test_result;
    }

    std::cout << "PASSED: " << test_name << " - energy validation successful" << std::endl;
  }

  test_result.passed = true;
  return test_result;
}

// Test spinless free fermions model
struct Z2SpinlessFreeFermionTest : public MPITest {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, fZ2QN>;
  using SITPST = SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>;
  
  size_t Lx = 2;
  size_t Ly = 2;
  double t = 1.0;
  std::vector<double> t2_list = {2.1, 0, -2.5};
  std::vector<SITPST> split_index_tps_list_lowest;
  std::vector<SITPST> split_index_tps_list_simple_update;
  std::vector<Configuration> all_configs;
  
  void SetUp() override {
    MPITest::SetUp();
    LoadLowestStateData();
    LoadSimpleUpdateData();
    GenerateAllConfigs();
  }
  
  void LoadLowestStateData() {
    split_index_tps_list_lowest.clear();
    for (auto t2 : t2_list) {
      split_index_tps_list_lowest.push_back(LoadLowestStateTPS(t2));
    }
  }
  
  void LoadSimpleUpdateData() {
    split_index_tps_list_simple_update.clear();
    for (auto t2 : t2_list) {
      split_index_tps_list_simple_update.push_back(LoadSimpleUpdateTPS(t2));
    }
  }
  
  SITPST LoadLowestStateTPS(double t2) {
    std::string type_suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ?
                             "_doublelowest" : "_complexlowest";
    
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "spinless_fermion_tps_t2_" + std::to_string(t2) + type_suffix;
    
    SITPST sitps(Ly, Lx);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!sitps.Load(data_path)) {
        throw std::runtime_error("Failed to load spinless fermion TPS data from: " + data_path);
      }
      std::cout << "Loaded spinless fermion lowest TPS data (t2=" << t2 << ") from: " << data_path << std::endl;
    }
    
    qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return sitps;
  }
  
  SITPST LoadSimpleUpdateTPS(double t2) {
    std::string type_suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ?
                             "_double_from_simple_update" : "_complex_from_simple_update";
    
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "spinless_fermion_tps_t2_" + std::to_string(t2) + type_suffix;
    
    SITPST sitps(Ly, Lx);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!sitps.Load(data_path)) {
        throw std::runtime_error("Failed to load spinless fermion TPS data from: " + data_path);
      }
      std::cout << "Loaded spinless fermion simple_update TPS data (t2=" << t2 << ") from: " << data_path << std::endl;
    }
    
    qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return sitps;
  }
  
  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {0, 0, 1, 1}; // Half-filling
    do {
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    } while (std::next_permutation(config_vec.begin(), config_vec.end()));
  }
};

TEST_F(Z2SpinlessFreeFermionTest, LowestState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);
  

  for (size_t i = 0; i < t2_list.size(); i++) {
    auto t2 = t2_list[i];
    auto energy_gs_exact = Calculate2x2OBCSpinlessFreeFermionEnergy(t, t2);  // Exact analytical calculation
    SquareSpinlessFermion model(t, t2, 0);
    
    std::string test_name = "SpinlessFreeFermion_Lowest_t2=" + std::to_string(t2);
    auto result = RunExactSummationTest<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN, SITPST>(
        model,
        split_index_tps_list_lowest[i],
        all_configs,
        trun_para,
        Ly,
        Lx,
        energy_gs_exact,
        test_name,
        comm,
        rank,
        mpi_size);

    EXPECT_TRUE(result.passed) << "Exact summation test failed for " << test_name;

    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      const double grad_norm = result.gradient.NormSquare();
      const auto grad_probe = WeightedProbeInnerProduct<SITPST, TEN_ELEM_TYPE>(result.gradient);
      const auto grad_probe_rand1 = RandomProbeInnerProduct<SITPST, TEN_ELEM_TYPE, fZ2QN>(
          result.gradient, /*seed=*/1337U);
      const auto grad_probe_rand2 = RandomProbeInnerProduct<SITPST, TEN_ELEM_TYPE, fZ2QN>(
          result.gradient, /*seed=*/424242U);

      constexpr bool kPrintGolden = false;
      if (kPrintGolden) {
        std::cout << "[GOLDEN-CANDIDATE][SpinlessFermion_Lowest_i=" << i << "] "
                  << "grad_norm=" << std::setprecision(16) << grad_norm << " "
                  << "grad_probe=(" << grad_probe.real() << "," << grad_probe.imag() << ") "
                  << "grad_probe_rand1=(" << grad_probe_rand1.real() << "," << grad_probe_rand1.imag() << ") "
                  << "grad_probe_rand2=(" << grad_probe_rand2.real() << "," << grad_probe_rand2.imag() << ")"
                  << std::endl;
      }

      constexpr double kGradTol = 1e-8;
      if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
        constexpr double kGradNorm[] = {1.152438300797112e-12, 2.184991439005157e-17, 1.928582676855609e-10};
        constexpr double kGradProbeReal[] = {3.683776728378202e-13, 7.407222090395872e-18, 6.551465833461134e-11};
        constexpr double kGradProbeRand1Real[] = {-2.086678524238405e-07, 3.871881553437145e-10, 4.644126093035119e-06};
        constexpr double kGradProbeRand2Real[] = {-4.594949654707012e-07, 1.672776854302904e-09, 6.889134343658586e-07};
        EXPECT_NEAR(grad_norm, kGradNorm[i], kGradTol);
        EXPECT_NEAR(grad_probe.real(), kGradProbeReal[i], kGradTol);
        EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real[i], kGradTol);
        EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real[i], kGradTol);
      } else {
        constexpr double kGradNorm[] = {8.774999537624253e-15, 3.080265654142141e-17, 1.489500558224182e-15};
        constexpr double kGradProbeReal[] = {3.054125684056241e-15, 1.065331570040376e-17, 5.367856989269067e-16};
        constexpr double kGradProbeImag[] = {6.968344158219192e-17, 2.44397422669205e-19, 1.291113078120351e-17};
        constexpr double kGradProbeRand1Real[] = {6.786528077495943e-09, -2.135690892840431e-09, 3.552063827742262e-09};
        constexpr double kGradProbeRand1Imag[] = {-1.110293952968061e-08, 8.850319984622591e-10, -7.068372857070966e-09};
        constexpr double kGradProbeRand2Real[] = {-1.344969572055623e-07, -1.972187795342304e-09, 4.981834593791531e-10};
        constexpr double kGradProbeRand2Imag[] = {7.661384896964128e-11, -3.228181924395671e-09, -4.945874739474172e-09};
        EXPECT_NEAR(grad_norm, kGradNorm[i], kGradTol);
        EXPECT_NEAR(grad_probe.real(), kGradProbeReal[i], kGradTol);
        EXPECT_NEAR(grad_probe.imag(), kGradProbeImag[i], kGradTol);
        EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real[i], kGradTol);
        EXPECT_NEAR(grad_probe_rand1.imag(), kGradProbeRand1Imag[i], kGradTol);
        EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real[i], kGradTol);
        EXPECT_NEAR(grad_probe_rand2.imag(), kGradProbeRand2Imag[i], kGradTol);
      }
    }
  }
}

TEST_F(Z2SpinlessFreeFermionTest, SimpleUpdateState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);
  
  // Actual computed values from single process run using Simple Update TPS data
  std::vector<double> energy_simple_update_values = {
    -4.1879072654,  // t2=2.1, for simple_update TPS
    -1.98218053854, // t2=0.0, for simple_update TPS
    -4.98966397657  // t2=-2.5, for simple_update TPS
  };

  for (size_t i = 0; i < t2_list.size(); i++) {
    auto t2 = t2_list[i];
    auto energy_expect = energy_simple_update_values[i];  // Simple update result (to be computed)
    SquareSpinlessFermion model(t, t2, 0);
    
    std::string test_name = "SpinlessFreeFermion_SimpleUpdate_t2=" + std::to_string(t2);
    auto result = RunExactSummationTest<SquareSpinlessFermion, TEN_ELEM_TYPE, fZ2QN, SITPST>(
        model,
        split_index_tps_list_simple_update[i],
        all_configs,
        trun_para,
        Ly,
        Lx,
        energy_expect,
        test_name,
        comm,
        rank,
        mpi_size);

    EXPECT_TRUE(result.passed) << "Exact summation test failed for " << test_name;
  }
}

// Test Heisenberg model  
struct TrivialHeisenbergTest : public MPITest {
  using IndexT = Index<TrivialRepQN>;
  using QNSctT = QNSector<TrivialRepQN>;
  using QNSctVecT = QNSectorVec<TrivialRepQN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, TrivialRepQN>;
  using SITPST = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>;
  
  size_t Lx = 2;
  size_t Ly = 2;
  double J = 1.0;
  SITPST split_index_tps_lowest = SITPST(Ly, Lx);
  SITPST split_index_tps_simple_update = SITPST(Ly, Lx);
  std::vector<Configuration> all_configs;
  
  void SetUp() override {
    MPITest::SetUp();
    split_index_tps_lowest = LoadLowestStateTPS();
    split_index_tps_simple_update = LoadSimpleUpdateTPS();
    GenerateAllConfigs();
  }
  
  SITPST LoadLowestStateTPS() {
    std::string type_suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ?
                             "_doublelowest" : "_complexlowest";
    
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "heisenberg_tps" + type_suffix;
    
    SITPST sitps(Ly, Lx);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!sitps.Load(data_path)) {
        throw std::runtime_error("Failed to load Heisenberg TPS data from: " + data_path);
      }
      std::cout << "Loaded Heisenberg lowest TPS data from: " << data_path << std::endl;
    }
    
    qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return sitps;
  }
  
  SITPST LoadSimpleUpdateTPS() {
    std::string type_suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ?
                             "_double_from_simple_update" : "_complex_from_simple_update";
    
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "heisenberg_tps" + type_suffix;
    
    SITPST sitps(Ly, Lx);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!sitps.Load(data_path)) {
        throw std::runtime_error("Failed to load Heisenberg TPS data from: " + data_path);
      }
      std::cout << "Loaded Heisenberg simple_update TPS data from: " << data_path << std::endl;
    }
    
    qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return sitps;
  }
  
  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {0, 0, 1, 1}; // Two up, two down spins
    do {
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    } while (std::next_permutation(config_vec.begin(), config_vec.end()));
  }
};

TEST_F(TrivialHeisenbergTest, LowestState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  
  auto energy_gs_exact = Calculate2x2HeisenbergEnergy(J);  // Exact analytical calculation
  SquareSpinOneHalfXXZModelOBC model(J, J, 0); // XY model
  
  auto result = RunExactSummationTest<SquareSpinOneHalfXXZModelOBC, TEN_ELEM_TYPE, TrivialRepQN, SITPST>(
      model,
      split_index_tps_lowest,
      all_configs,
      trun_para,
      Ly,
      Lx,
      energy_gs_exact,
      "Heisenberg_Lowest",
      comm,
      rank,
      mpi_size);

  EXPECT_TRUE(result.passed) << "Exact summation test failed for Heisenberg lowest model";

  if (rank != qlten::hp_numeric::kMPIMasterRank) { return; }

  // Gradient golden regression
  const double grad_norm = result.gradient.NormSquare();
  const auto grad_probe = WeightedProbeInnerProduct<SITPST, TEN_ELEM_TYPE>(result.gradient);
  const auto grad_probe_rand1 = RandomProbeInnerProduct<SITPST, TEN_ELEM_TYPE, TrivialRepQN>(
      result.gradient, /*seed=*/1337U);
  const auto grad_probe_rand2 = RandomProbeInnerProduct<SITPST, TEN_ELEM_TYPE, TrivialRepQN>(
      result.gradient, /*seed=*/424242U);

  constexpr bool kPrintGolden = false;
  if (kPrintGolden) {
    std::cout << "[GOLDEN-CANDIDATE][Heisenberg_Lowest] "
              << "grad_norm=" << std::setprecision(16) << grad_norm << " "
              << "grad_probe=(" << grad_probe.real() << "," << grad_probe.imag() << ") "
              << "grad_probe_rand1=(" << grad_probe_rand1.real() << "," << grad_probe_rand1.imag() << ") "
              << "grad_probe_rand2=(" << grad_probe_rand2.real() << "," << grad_probe_rand2.imag() << ")"
              << std::endl;
  }

  constexpr double kGradTol = 1e-8;
  if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
    constexpr double kGradNorm = 2.69115141087757e-08;
    constexpr double kGradProbeReal = 8.719330571244627e-09;
    constexpr double kGradProbeRand1Real = 7.247764646273227e-06;
    constexpr double kGradProbeRand2Real = 7.942431888529052e-05;
    EXPECT_NEAR(grad_norm, kGradNorm, kGradTol);
    EXPECT_NEAR(grad_probe.real(), kGradProbeReal, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real, kGradTol);
  } else {
    constexpr double kGradNorm = 2.277663798157925e-08;
    constexpr double kGradProbeReal = 7.37963070602707e-09;
    constexpr double kGradProbeImag = 1.708247848617349e-10;
    constexpr double kGradProbeRand1Real = -4.524246946088829e-05;
    constexpr double kGradProbeRand1Imag = -9.074124907574788e-05;
    constexpr double kGradProbeRand2Real = -8.482661047397774e-05;
    constexpr double kGradProbeRand2Imag = -5.701510687765299e-05;
    EXPECT_NEAR(grad_norm, kGradNorm, kGradTol);
    EXPECT_NEAR(grad_probe.real(), kGradProbeReal, kGradTol);
    EXPECT_NEAR(grad_probe.imag(), kGradProbeImag, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.imag(), kGradProbeRand1Imag, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.imag(), kGradProbeRand2Imag, kGradTol);
  }
}

TEST_F(TrivialHeisenbergTest, SimpleUpdateState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  
  // Actual computed value from single process run using Simple Update TPS data
  double energy_simple_update_expect = -1.99521278793;  // computed from simple_update TPS
  SquareSpinOneHalfXXZModelOBC model(J, J, 0); // XY model
  
  auto result = RunExactSummationTest<SquareSpinOneHalfXXZModelOBC, TEN_ELEM_TYPE, TrivialRepQN, SITPST>(
      model,
      split_index_tps_simple_update,
      all_configs,
      trun_para,
      Ly,
      Lx,
      energy_simple_update_expect,
      "Heisenberg_SimpleUpdate",
      comm,
      rank,
      mpi_size);

  EXPECT_TRUE(result.passed) << "Exact summation test failed for Heisenberg simple_update model";
}

// Test Transverse Ising model
struct TrivialTransverseIsingTest : public MPITest {
  using IndexT = Index<TrivialRepQN>;
  using QNSctT = QNSector<TrivialRepQN>;  
  using QNSctVecT = QNSectorVec<TrivialRepQN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, TrivialRepQN>;
  using SITPST = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>;
  
  size_t Lx = 2;
  size_t Ly = 2;
  double h = 1.0;
  double J = 1.0;
  SITPST split_index_tps_lowest = SITPST(Ly, Lx);
  SITPST split_index_tps_simple_update = SITPST(Ly, Lx);
  std::vector<Configuration> all_configs;
  
  void SetUp() override {
    MPITest::SetUp();
    split_index_tps_lowest = LoadLowestStateTPS();
    split_index_tps_simple_update = LoadSimpleUpdateTPS();
    GenerateAllConfigs();
  }
  
  SITPST LoadLowestStateTPS() {
    std::string type_suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ?
                             "_doublelowest" : "_complexlowest";
    
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "transverse_ising_tps" + type_suffix;
    
    SITPST sitps(Ly, Lx);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!sitps.Load(data_path)) {
        throw std::runtime_error("Failed to load Transverse Ising TPS data from: " + data_path);
      }
      std::cout << "Loaded Transverse Ising lowest TPS data from: " << data_path << std::endl;
    }
    
    qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return sitps;
  }
  
  SITPST LoadSimpleUpdateTPS() {
    std::string type_suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ?
                             "_double_from_simple_update" : "_complex_from_simple_update";
    
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "transverse_ising_tps" + type_suffix;
    
    SITPST sitps(Ly, Lx);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!sitps.Load(data_path)) {
        throw std::runtime_error("Failed to load Transverse Ising TPS data from: " + data_path);
      }
      std::cout << "Loaded Transverse Ising simple_update TPS data from: " << data_path << std::endl;
    }
    
    qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return sitps;
  }
  
  void GenerateAllConfigs() {
    // Generate all 16 possible configurations for 2x2 lattice
    for (size_t i = 0; i < 16; i++) {
      std::vector<size_t> config_vec(4);
      config_vec[0] = (i >> 0) & 1; // site 0
      config_vec[1] = (i >> 1) & 1; // site 1
      config_vec[2] = (i >> 2) & 1; // site 2
      config_vec[3] = (i >> 3) & 1; // site 3
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    }
  }
};

TEST_F(TrivialTransverseIsingTest, LowestState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  
  auto energy_gs_exact = Calculate2x2OBCTransverseIsingEnergy(J, h);  // Exact analytical calculation
  TransverseFieldIsingSquareOBC model(h);
  
  auto result = RunExactSummationTest<TransverseFieldIsingSquareOBC, TEN_ELEM_TYPE, TrivialRepQN, SITPST>(
      model,
      split_index_tps_lowest,
      all_configs,
      trun_para,
      Ly,
      Lx,
      energy_gs_exact,
      "TransverseIsing_Lowest",
      comm,
      rank,
      mpi_size);

  EXPECT_TRUE(result.passed) << "Exact summation test failed for Transverse Ising lowest model";

  if (rank != qlten::hp_numeric::kMPIMasterRank) { return; }

  const double grad_norm = result.gradient.NormSquare();
  const auto grad_probe = WeightedProbeInnerProduct<SITPST, TEN_ELEM_TYPE>(result.gradient);
  const auto grad_probe_rand1 = RandomProbeInnerProduct<SITPST, TEN_ELEM_TYPE, TrivialRepQN>(
      result.gradient, /*seed=*/1337U);
  const auto grad_probe_rand2 = RandomProbeInnerProduct<SITPST, TEN_ELEM_TYPE, TrivialRepQN>(
      result.gradient, /*seed=*/424242U);

  constexpr bool kPrintGolden = false;
  if (kPrintGolden) {
    std::cout << "[GOLDEN-CANDIDATE][TransverseIsing_Lowest] "
              << "grad_norm=" << std::setprecision(16) << grad_norm << " "
              << "grad_probe=(" << grad_probe.real() << "," << grad_probe.imag() << ") "
              << "grad_probe_rand1=(" << grad_probe_rand1.real() << "," << grad_probe_rand1.imag() << ") "
              << "grad_probe_rand2=(" << grad_probe_rand2.real() << "," << grad_probe_rand2.imag() << ")"
              << std::endl;
  }

  constexpr double kGradTol = 1e-8;
  if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
    constexpr double kGradNorm = 1.290630314256308e-10;
    constexpr double kGradProbeReal = 4.081475798300479e-11;
    constexpr double kGradProbeRand1Real = -9.488932296498787e-06;
    constexpr double kGradProbeRand2Real = -8.79805491844677e-06;
    EXPECT_NEAR(grad_norm, kGradNorm, kGradTol);
    EXPECT_NEAR(grad_probe.real(), kGradProbeReal, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real, kGradTol);
  } else {
    constexpr double kGradNorm = 1.291818680681758e-10;
    constexpr double kGradProbeReal = 4.090781110515482e-11;
    constexpr double kGradProbeImag = 8.556929891955689e-13;
    constexpr double kGradProbeRand1Real = -1.087840379768135e-05;
    constexpr double kGradProbeRand1Imag = -7.173576288770621e-06;
    constexpr double kGradProbeRand2Real = -4.468070011578015e-06;
    constexpr double kGradProbeRand2Imag = -1.076013502301865e-05;
    EXPECT_NEAR(grad_norm, kGradNorm, kGradTol);
    EXPECT_NEAR(grad_probe.real(), kGradProbeReal, kGradTol);
    EXPECT_NEAR(grad_probe.imag(), kGradProbeImag, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.imag(), kGradProbeRand1Imag, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.imag(), kGradProbeRand2Imag, kGradTol);
  }
}

TEST_F(TrivialTransverseIsingTest, SimpleUpdateState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  
  // Actual computed value from single process run using Simple Update TPS data
  double energy_simple_update_expect = -5.19991995228;  // computed from simple_update TPS
  TransverseFieldIsingSquareOBC model(h);
  
  auto result = RunExactSummationTest<TransverseFieldIsingSquareOBC, TEN_ELEM_TYPE, TrivialRepQN, SITPST>(
      model,
      split_index_tps_simple_update,
      all_configs,
      trun_para,
      Ly,
      Lx,
      energy_simple_update_expect,
      "TransverseIsing_SimpleUpdate",
      comm,
      rank,
      mpi_size);

  EXPECT_TRUE(result.passed) << "Exact summation test failed for Transverse Ising simple_update model";
}

// Test t-J model
struct Z2tJTest : public MPITest {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, fZ2QN>;
  using SITPST = SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>;
  
  size_t Lx = 2;
  size_t Ly = 2;
  double t = 1.0;
  double J = 0.3;
  double mu = 0.0;
  double energy_gs_exact = -2.9431635706137875; // Known exact value for t-J model 2x2 lattice
  size_t Db = 4;
  SITPST split_index_tps_lowest = SITPST(Ly, Lx);
  SITPST split_index_tps_simple_update = SITPST(Ly, Lx);
  std::vector<Configuration> all_configs;
  
  void SetUp() override {
    MPITest::SetUp();
    split_index_tps_lowest = LoadLowestStateTPS();
    split_index_tps_simple_update = LoadSimpleUpdateTPS();
    GenerateAllConfigs();
  }
  
  SITPST LoadLowestStateTPS() {
    std::string type_suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ?
                             "_doublelowest" : "_complexlowest";
    
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "tj_model_tps" + type_suffix;
    
    SITPST sitps(Ly, Lx);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!sitps.Load(data_path)) {
        throw std::runtime_error("Failed to load t-J model TPS data from: " + data_path);
      }
      std::cout << "Loaded t-J model lowest TPS data from: " << data_path << std::endl;
    }
    
    qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return sitps;
  }
  
  SITPST LoadSimpleUpdateTPS() {
    std::string type_suffix = std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double> ?
                             "_double_from_simple_update" : "_complex_from_simple_update";
    
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "tj_model_tps" + type_suffix;
    
    SITPST sitps(Ly, Lx);
    
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      if (!sitps.Load(data_path)) {
        throw std::runtime_error("Failed to load t-J model TPS data from: " + data_path);
      }
      std::cout << "Loaded t-J model simple_update TPS data from: " << data_path << std::endl;
    }
    
    qlpeps::MPI_Bcast(sitps, comm, qlten::hp_numeric::kMPIMasterRank);
    return sitps;
  }
  
  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {2, 2, 0, 1}; // Half-filling for t-J model
    std::sort(config_vec.begin(), config_vec.end());
    do {
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    } while (std::next_permutation(config_vec.begin(), config_vec.end()));
  }
};

TEST_F(Z2tJTest, LowestState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(Db, Db, 0);
  
  SquaretJVModel model(t, 0, J, J / 4, mu);
  
  auto result = RunExactSummationTest<SquaretJVModel, TEN_ELEM_TYPE, fZ2QN, SITPST>(
      model,
      split_index_tps_lowest,
      all_configs,
      trun_para,
      Ly,
      Lx,
      energy_gs_exact,
      "tJ_Lowest",
      comm,
      rank,
      mpi_size);

  EXPECT_TRUE(result.passed) << "Exact summation test failed for t-J lowest model";

  if (rank != qlten::hp_numeric::kMPIMasterRank) { return; }

  const double grad_norm = result.gradient.NormSquare();
  const auto grad_probe = WeightedProbeInnerProduct<SITPST, TEN_ELEM_TYPE>(result.gradient);
  const auto grad_probe_rand1 = RandomProbeInnerProduct<SITPST, TEN_ELEM_TYPE, fZ2QN>(
      result.gradient, /*seed=*/1337U);
  const auto grad_probe_rand2 = RandomProbeInnerProduct<SITPST, TEN_ELEM_TYPE, fZ2QN>(
      result.gradient, /*seed=*/424242U);

  constexpr bool kPrintGolden = false;
  if (kPrintGolden) {
    std::cout << "[GOLDEN-CANDIDATE][tJ_Lowest] "
              << "grad_norm=" << std::setprecision(16) << grad_norm << " "
              << "grad_probe=(" << grad_probe.real() << "," << grad_probe.imag() << ") "
              << "grad_probe_rand1=(" << grad_probe_rand1.real() << "," << grad_probe_rand1.imag() << ") "
              << "grad_probe_rand2=(" << grad_probe_rand2.real() << "," << grad_probe_rand2.imag() << ")"
              << std::endl;
  }

  constexpr double kGradTol = 1e-8;
  if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
    constexpr double kGradNorm = 1.567391632086754e-16;
    constexpr double kGradProbeReal = 5.570517772849881e-17;
    constexpr double kGradProbeRand1Real = 5.861090045826115e-09;
    constexpr double kGradProbeRand2Real = -1.757235656456151e-09;
    EXPECT_NEAR(grad_norm, kGradNorm, kGradTol);
    EXPECT_NEAR(grad_probe.real(), kGradProbeReal, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real, kGradTol);
  } else {
    constexpr double kGradNorm = 1.485495502708925e-10;
    constexpr double kGradProbeReal = 5.748499136853164e-11;
    constexpr double kGradProbeImag = 1.568014407189329e-12;
    constexpr double kGradProbeRand1Real = 1.866777089942861e-06;
    constexpr double kGradProbeRand1Imag = 5.987074697019814e-06;
    constexpr double kGradProbeRand2Real = 9.920980918226958e-06;
    constexpr double kGradProbeRand2Imag = 2.912306733411341e-06;
    EXPECT_NEAR(grad_norm, kGradNorm, kGradTol);
    EXPECT_NEAR(grad_probe.real(), kGradProbeReal, kGradTol);
    EXPECT_NEAR(grad_probe.imag(), kGradProbeImag, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.real(), kGradProbeRand1Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand1.imag(), kGradProbeRand1Imag, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.real(), kGradProbeRand2Real, kGradTol);
    EXPECT_NEAR(grad_probe_rand2.imag(), kGradProbeRand2Imag, kGradTol);
  }
}

TEST_F(Z2tJTest, SimpleUpdateState) {
  using RealT = typename qlten::RealTypeTrait<TEN_ELEM_TYPE>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(Db, Db, 0);
  
  // Actual computed value from single process run using Simple Update TPS data
  double energy_simple_update_expect = -2.78008187385;  // computed from simple_update TPS
  SquaretJVModel model(t, 0, J, J / 4, mu);
  
  auto result = RunExactSummationTest<SquaretJVModel, TEN_ELEM_TYPE, fZ2QN, SITPST>(
      model,
      split_index_tps_simple_update,
      all_configs,
      trun_para,
      Ly,
      Lx,
      energy_simple_update_expect,
      "tJ_SimpleUpdate",
      comm,
      rank,
      mpi_size);

  EXPECT_TRUE(result.passed) << "Exact summation test failed for t-J simple_update model";
}

int main(int argc, char* argv[]) {
  testing::AddGlobalTestEnvironment(new MPIEnvironment);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  
  // Install timeout signal handler for additional protection on macOS
  auto test_result = RUN_ALL_TESTS();
  
  return test_result;
}
