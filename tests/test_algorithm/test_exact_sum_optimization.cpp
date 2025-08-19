/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Refactored: 2025-08-18
*
* Description: QuantumLiquids/PEPS project. 
* Pure Optimizer unit tests using exact summation for deterministic gradient computation.
* 
* This test file focuses ONLY on Optimizer algorithm verification by:
*  1. Using ExactSumEnergyEvaluator for deterministic gradient computation (no Monte Carlo noise)
*  2. Testing Optimizer directly without VMCPEPSOptimizerExecutor overhead
*  3. Verifying convergence for different optimization algorithms (AdaGrad, SGD, etc.)
*  4. Testing model solvers: t-J, spinless free fermions, Heisenberg, Transverse-field Ising
* 
* Design Philosophy (Linus): "Good taste eliminates special cases and complexity."
* - NO Monte Carlo sampling complexity
* - NO state normalization overhead  
* - NO data collection noise
* - NO file I/O operations
* - Focus ONLY on optimization algorithm correctness
*/


#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nnn_simple_update.h"
#include "../utilities.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::fZ2QN;
using qlten::special_qn::TrivialRepQN;

/**
 * @brief Generate type-specific path for TPS data in test output directory
 * @param base_name Base name for the TPS data (e.g., "heisenberg_tps")
 * @return Path in test output directory with type suffix (Double or Complex)
 */
std::string GetTypeSpecificPath(const std::string& base_name) {
  std::string type_suffix;
  if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
    type_suffix = "_double";
  } else if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Complex>) {
    type_suffix = "_complex";
  } else {
    // This should never be reached due to the CMake configuration
    type_suffix = "_unknown";
  }
  
  // Extract just the model name from the base_name (remove "test_algorithm/test_data/" prefix if present)
  std::string model_name = base_name;
  size_t pos = model_name.find_last_of('/');
  if (pos != std::string::npos) {
    model_name = model_name.substr(pos + 1);
  }
  
  return GetTestOutputPath("exact_sum_optimization", model_name + type_suffix + "lowest");
}

struct SimpleUpdateTestParams {
  size_t Ly;
  size_t Lx;
  size_t D;
  double Tau0;
  size_t Steps;
};

/**
 * @brief Computes the ground state energy of spinless free fermions
 *        on a 2x2 OBC square lattice with NN and NNN hopping, equivalent
 *        to a 4-site PBC chain.
 *
 * @param t     Nearest-neighbor hopping amplitude
 * @param t2    Next-nearest-neighbor hopping amplitude
 * @return double Ground state energy
 */
double Calculate2x2OBCSpinlessFreeFermionEnergy(double t, double t2) {
  std::vector<double> k_values = {0.0, M_PI / 2.0, M_PI, 3.0 * M_PI / 2.0}; //momentum in 4-site PBC chain
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
 *        on a 2x2 square lattice with obc boundary conditions.
 *      --> Equivalent to 1D PBC 4-site.
 *
 * How to obtained it analytically?
 * U1 symmetry restrict to 6-dim basis,
 * translational symmetry (assume momentum 0) further restrict to 2-dim basis.
 * The Hamiltonian matrix in these basis
 * (0, sqrt(2);
 *  sqrt(2), -1).
 * Then the ground state energy is -2.
 *
 * @param J     Exchange coupling
 * @return double Ground state energy per site
 */
double Calculate2x2HeisenbergEnergy(double J) {
  return -2 * J;
}

/**
 * @brief Computes the exact ground state energy of transverse Ising model
 *        on a 2x2 OBC square lattice using Jordan-Wigner and Fourier-Bogoliubov transformation.
 *        This is the analytical solution from the image.
 *
 * @param J     Ising coupling strength
 * @param h     Transverse field strength
 * @return double Ground state energy
 */
double Calculate2x2OBCTransverseIsingEnergy(double J, double h) {
  // After Jordan-Wigner transformation and Fourier-Bogoliubov transformation:
  // H = \sum_k \left[ \epsilon_k \left( \gamma_k^\dagger \gamma_k - \frac{1}{2} \right) \right]
  // where \epsilon_k = 2 \sqrt{J^2 + h^2 - 2 J h \cos k}

  // For N=4 sites in the even-parity (ground-state) sector, allowed k-modes are:
  // k = \pm \frac{\pi}{4}, \pm \frac{3\pi}{4}

  // The exact ground-state energy is:
  // E_0 = - \frac{1}{2} \sum_k \epsilon_k = - \sum_{k>0} \epsilon_k

  std::vector<double> k_values = {M_PI / 4.0, 3.0 * M_PI / 4.0}; // k > 0 modes

  double ground_state_energy = 0.0;
  for (auto k : k_values) {
    double epsilon_k = 2.0 * std::sqrt(J * J + h * h - 2.0 * J * h * std::cos(k));
    ground_state_energy -= epsilon_k; // - \sum_{k>0} \epsilon_k
  }

  return ground_state_energy;
}

/**
 * @brief Common test runner for exact summation optimization with VMC optimizer
 * 
 * @tparam ModelT Model type
 * @tparam TenElemT Tensor element type
 * @tparam QNT Quantum number type
 * @tparam SITPST SplitIndexTPS type
 * @tparam MCUpdater Monte Carlo updater type
 * @param model The model to test
 * @param split_index_tps Initial TPS state
 * @param all_configs All possible configurations
 * @param trun_para BMPSTruncatePara for truncation
 * @param Ly Number of rows
 * @param Lx Number of columns
 * @param energy_exact Expected exact energy
 * @param optimize_para Optimization parameters
 * @param test_name Name of the test for output
 * @return double Final energy after optimization
 */
/**
 * @brief Pure Optimizer algorithm test using exact summation (NO VMCPEPSOptimizerExecutor)
 * 
 * This function tests ONLY the Optimizer algorithm by:
 * - Directly calling Optimizer.IterativeOptimize()
 * - Using deterministic exact gradient computation
 * - Eliminating ALL Monte Carlo noise and overhead
 * 
 * Design Philosophy: "Good taste is about eliminating special cases"
 * - No Monte Carlo sampling complexity
 * - No state normalization operations
 * - No data collection overhead
 * - No file I/O noise
 * - Focus purely on optimization algorithm correctness
 */
template<typename ModelT, typename TenElemT, typename QNT, typename SITPST>
double RunPureOptimizerTest(
    ModelT &model,
    SITPST &split_index_tps,
    const std::vector<Configuration> &all_configs,
    const BMPSTruncatePara &trun_para,
    size_t Ly, size_t Lx,
    double energy_exact,
    const qlpeps::OptimizerParams &optimizer_params,
    const std::string &test_name) {

  // Create pure Optimizer (NO executor overhead)
  int rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  Optimizer<TenElemT, QNT> optimizer(optimizer_params, MPI_COMM_WORLD, rank, mpi_size);

  // MPI-AWARE exact summation energy evaluator with PROPER state broadcasting
  auto energy_evaluator = [&](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    // CRITICAL FIX: Create local state copy for MPI broadcasting
    SITPST local_state = state;
    
    // CRITICAL MPI SYNC: Broadcast state to all ranks (matches DefaultEnergyEvaluator_ behavior)
    // This ensures all ranks process the same updated state from the optimizer
    MPI_Bcast(local_state, MPI_COMM_WORLD);
    
    // All ranks now use the synchronized state for exact computation
    auto [energy, gradient, error] =
        ExactSumEnergyEvaluator(local_state, all_configs, trun_para, model, Ly, Lx);
    return {energy, gradient, error};
  };

  // Simple monitoring callback
  typename Optimizer<TenElemT, QNT>::OptimizationCallback callback;
  callback.on_iteration =
      [&energy_exact, &test_name](size_t iteration, double energy, double energy_error, double gradient_norm) {
        std::cout << test_name << " - step: " << iteration
                  << " E0=" << std::setw(14) << std::fixed
                  << std::setprecision(kEnergyOutputPrecision) << energy
                  << " ||grad||=" << std::setw(8) << std::fixed
                  << std::setprecision(kEnergyOutputPrecision) << gradient_norm;
        if (energy_exact != 0.0) {
          std::cout << " exact=" << energy_exact;
        }
        std::cout << std::endl;
      };

  // PURE optimization - NO external complexity
  auto result = optimizer.IterativeOptimize(split_index_tps, energy_evaluator, callback);

  // Verify algorithm convergence
  double final_energy = std::real(result.final_energy);
  if (energy_exact != 0.0) {
    EXPECT_GE(final_energy, energy_exact - 1E-10);
    EXPECT_NEAR(final_energy, energy_exact, 1e-5);
  }

  return final_energy;
}

struct Z2SpinlessFreeFermionTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<TEN_ELEM_TYPE, fZ2QN>;
  size_t Lx = 2;
  size_t Ly = 2;

  std::vector<SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>> split_index_tps_list;
  double t = 1.0;
  std::vector<double> t2_list = {2.1, 0, -2.5};
  // available t2: (-inf, -2] U {0} U [2, inf), these value make sure the ground state particle number is even (=2).
  fZ2QN qn0 = fZ2QN(0);
  // |ket>
  IndexT loc_phy_ket = IndexT({QNSctT(fZ2QN(1), 1),  // |1> occupied
                               QNSctT(fZ2QN(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  // <bra|
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  std::vector<Configuration> all_configs;
  void SetUp(void) {
    GenerateAllSimpleUpdateResults();
    GenerateAllConfigs();
  }

  void GenerateAllSimpleUpdateResults() {
    for (auto t2 : t2_list) {
      split_index_tps_list.push_back(SimpleUpdate(t2));
    }
  }

  SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> SimpleUpdate(double t2) {
    SquareLatticePEPS<TEN_ELEM_TYPE, fZ2QN> peps0(loc_phy_ket, Ly, Lx);

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    //half-filling
    size_t n_int = 1;
    for (size_t y = 0; y < Ly; y++) {
      for (size_t x = 0; x < Lx; x++) {
        activates[y][x] = n_int % 2;
        n_int++;
      }
    }
    peps0.Initial(activates);

    // NN Hamiltonian
    DTensor ham_nn = DTensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
    ham_nn({1, 0, 1, 0}) = -t;
    ham_nn({0, 1, 0, 1}) = -t;
    ham_nn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention

    // NNN Hamiltonian (same structure as NN but with factor t2/t)
    DTensor ham_nnn = DTensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
    ham_nnn({1, 0, 1, 0}) = -t2;
    ham_nnn({0, 1, 0, 1}) = -t2;
    ham_nnn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention

    SimpleUpdatePara update_para(30, 0.1, 4, 4, 1e-15);
    SimpleUpdateExecutor<TEN_ELEM_TYPE, fZ2QN>
        *su_exe = new SquareLatticeNNNSimpleUpdateExecutor<TEN_ELEM_TYPE, fZ2QN>(update_para, peps0,
                                                                               ham_nn, ham_nnn);
    su_exe->Execute();
    auto peps = su_exe->GetPEPS();
    auto tps = TPS<TEN_ELEM_TYPE, fZ2QN>(peps);
    delete su_exe;
    SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> sitps = SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>(tps);
    sitps.ScaleMaxAbsForAllSite(1.0);
    return sitps;
  }

  //half-filling
  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {0, 0, 1, 1};
    do {
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    } while (std::next_permutation(config_vec.begin(),
                                   config_vec.end()));
    // Generates the next lexicographical permutation
  }
};

TEST_F(Z2SpinlessFreeFermionTools, ExactSumGradientOptWithVMCOptimizer) {
  auto trun_para =
      BMPSTruncatePara(8, 8, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = SquareSpinlessFermion;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = fZ2QN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;


  for (size_t i = 0; i < t2_list.size(); i++) {
    auto t2 = t2_list[i];
    auto energy_exact = Calculate2x2OBCSpinlessFreeFermionEnergy(t, t2);
    Model spinless_fermion_model(t, t2, 0);

    // Use the corresponding simple update result for this t2 value
    auto& split_index_tps = split_index_tps_list[i];

    // Test the exact energy evaluator on the initial state
    auto [initial_energy, initial_gradient, initial_error] = ExactSumEnergyEvaluator(
        split_index_tps, all_configs, trun_para, spinless_fermion_model, Ly, Lx);

    std::cout << "Initial energy: " << initial_energy << ", Expected: " << energy_exact << std::endl;
    std::cout << "Initial gradient norm: " << initial_gradient.NormSquare() << std::endl;

    // Pure Optimizer parameters - adjusted for better convergence
    // Increased patience and more conservative step size for stable convergence
    qlpeps::OptimizerParams::BaseParams base_params(200, 1e-15, 1e-15, 100, 0.1);
    qlpeps::AdaGradParams adagrad_params(1e-10, 0.0);
    qlpeps::OptimizerParams opt_params(base_params, adagrad_params);

    // Pure algorithm test (NO VMCPEPSOptimizerExecutor overhead)
    std::string test_name = "SpinlessFreeFermion_t2=" + std::to_string(t2);
    RunPureOptimizerTest<Model, TenElemT, QNT, SITPST>(
        spinless_fermion_model, split_index_tps, all_configs, trun_para,
        Ly, Lx, energy_exact, opt_params, test_name);
  }
}

// Add Heisenberg test with trivial quantum numbers
struct TrivialHeisenbergTools : public testing::Test {
  using IndexT = Index<TrivialRepQN>;
  using QNSctT = QNSector<TrivialRepQN>;
  using QNSctVecT = QNSectorVec<TrivialRepQN>;

  using DTensor = QLTensor<TEN_ELEM_TYPE, TrivialRepQN>;
  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> split_index_tps = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>(Ly, Lx);
  double J = 1.0;
  TrivialRepQN qn0 = TrivialRepQN();
  // |ket> - spin-1/2 system
  IndexT loc_phy_ket = IndexT({QNSctT(TrivialRepQN(), 2)}, TenIndexDirType::OUT);
  // <bra|
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  std::vector<Configuration> all_configs;
  void SetUp(void) {
    split_index_tps = SimpleUpdate();
    GenerateAllConfigs();
  }

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> SimpleUpdate(void) {
    SquareLatticePEPS<TEN_ELEM_TYPE, TrivialRepQN> peps0(loc_phy_ket, Ly, Lx);

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    // Initialize with alternating spins
    activates = {{0, 1}, {1, 0}};
    peps0.Initial(activates);

    DTensor ham_nn = DTensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
    // Heisenberg interaction: S_i \cdot S_j
    ham_nn({0, 1, 0, 1}) = 0.5 * J;    // |up down> -> |down up>
    ham_nn({1, 0, 1, 0}) = 0.5 * J;    // |down up> -> |up down>
    ham_nn({0, 1, 1, 0}) = -0.25 * J;   // |up down> -> |up down>
    ham_nn({1, 0, 0, 1}) = -0.25 * J;   // |down up> -> |down down>
    ham_nn({0, 0, 0, 0}) = 0.25 * J;    // |up up> -> |up up>
    ham_nn({1, 1, 1, 1}) = 0.25 * J;    // |down down> -> |down down>
    ham_nn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention

    SimpleUpdatePara update_para(10, 0.1, 1, 4, 1e-10);
    SimpleUpdateExecutor<TEN_ELEM_TYPE, TrivialRepQN>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TEN_ELEM_TYPE, TrivialRepQN>(update_para, peps0,
                                                                                      ham_nn);
    su_exe->Execute();
    auto peps = su_exe->GetPEPS();
    auto tps = TPS<TEN_ELEM_TYPE, TrivialRepQN>(peps);
    delete su_exe;
    SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> sitps = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>(tps);
    sitps.ScaleMaxAbsForAllSite(1.0);
    return sitps;
  }

  // Generate all possible spin configurations
  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {0, 0, 1, 1}; // Two up, two down spins
    do {
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    } while (std::next_permutation(config_vec.begin(),
                                   config_vec.end()));
    // Generates the next lexicographical permutation
  }
};

TEST_F(TrivialHeisenbergTools, ExactSumGradientOptWithVMCOptimizer) {
  auto trun_para =
      BMPSTruncatePara(1, 8, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = SquareSpinOneHalfXXZModel;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;


  auto energy_exact = Calculate2x2HeisenbergEnergy(J);
  Model heisenberg_model(J, J, 0); // Jx = Jy = J, Jz = 0 (XY model)

  // Pure Optimizer parameters (NO Monte Carlo complexity)
  qlpeps::OptimizerParams opt_params = qlpeps::OptimizerFactory::CreateAdaGradAdvanced(100, 1e-15, 1e-30, 20, 1.0, 1e-8, 0.0);

  // Pure algorithm test (NO VMCPEPSOptimizerExecutor overhead)
  std::string test_name = "Heisenberg_Trivial";
  RunPureOptimizerTest<Model, TenElemT, QNT, SITPST>(
      heisenberg_model, split_index_tps, all_configs, trun_para,
      Ly, Lx, energy_exact, opt_params, test_name);
}

// Add Transverse Ising test with trivial quantum numbers
struct TrivialTransverseIsingTools : public testing::Test {
  using IndexT = Index<TrivialRepQN>;
  using QNSctT = QNSector<TrivialRepQN>;
  using QNSctVecT = QNSectorVec<TrivialRepQN>;

  using DTensor = QLTensor<TEN_ELEM_TYPE, TrivialRepQN>;
  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> split_index_tps = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>(Ly, Lx);
  double h = 1.0;
  double J = 1.0;
  TrivialRepQN qn0 = TrivialRepQN();
  // |ket> - spin-1/2 system
  IndexT loc_phy_ket = IndexT({QNSctT(TrivialRepQN(), 2)}, TenIndexDirType::OUT);
  // <bra|
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  std::vector<Configuration> all_configs;
  void SetUp(void) {
    split_index_tps = SimpleUpdate();
    GenerateAllConfigs();
  }

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> SimpleUpdate(void) {
    SquareLatticePEPS<TEN_ELEM_TYPE, TrivialRepQN> peps0(loc_phy_ket, Ly, Lx);

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    // Initialize with all spins up
    activates = {{0, 0}, {0, 0}};
    peps0.Initial(activates);

    DTensor ham_nn = DTensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
    // Transverse field Ising interaction: -J \sigma_i^z \sigma_j^z - h \sigma_i^x
    // ZZ
    ham_nn({0, 0, 0, 0}) = -J;    // |up up> -> |up up>
    ham_nn({1, 1, 1, 1}) = -J;    // |down down> -> |down down>
    ham_nn({0, 1, 1, 0}) = J;     // |up down> -> |up down>
    ham_nn({1, 0, 0, 1}) = J;     // |down up> -> |down up>
    // site 1 X/2
    ham_nn({0, 1, 1, 1}) = -0.5 * h; // site 1 X
    ham_nn({0, 0, 0, 1}) = -0.5 * h;
    ham_nn({1, 1, 1, 0}) = -0.5 * h;
    ham_nn({1, 0, 0, 0}) = -0.5 * h;
    // site 2 X/2
    ham_nn({1, 0, 1, 1}) = -0.5 * h;
    ham_nn({1, 1, 0, 1}) = -0.5 * h;
    ham_nn({0, 0, 1, 0}) = -0.5 * h;
    ham_nn({0, 1, 0, 0}) = -0.5 * h;
    ham_nn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention

    SimpleUpdatePara update_para(10, 0.1, 1, 4, 1e-10);
    SimpleUpdateExecutor<TEN_ELEM_TYPE, TrivialRepQN>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TEN_ELEM_TYPE, TrivialRepQN>(update_para, peps0,
                                                                                      ham_nn);
    su_exe->Execute();
    auto peps = su_exe->GetPEPS();
    auto tps = TPS<TEN_ELEM_TYPE, TrivialRepQN>(peps);
    delete su_exe;
    SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> sitps = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>(tps);
    sitps.ScaleMaxAbsForAllSite(1.0);
    return sitps;
  }

  // Generate all possible spin configurations
  void GenerateAllConfigs() {
    // Generate all 16 possible configurations for 2x2 lattice
    for (size_t i = 0; i < 16; i++) {
      std::vector<size_t> config_vec(4);
      config_vec[0] = (i >> 0) & 1;  // site 0
      config_vec[1] = (i >> 1) & 1;  // site 1  
      config_vec[2] = (i >> 2) & 1;  // site 2
      config_vec[3] = (i >> 3) & 1;  // site 3
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    }
  }
};

TEST_F(TrivialTransverseIsingTools, ExactSumGradientOptWithVMCOptimizer) {
  auto trun_para =
      BMPSTruncatePara(1, 8, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = TransverseIsingSquare;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;


  auto energy_exact = Calculate2x2OBCTransverseIsingEnergy(J, h);
  Model transverse_ising_model(h);

  // Pure Optimizer parameters (NO Monte Carlo complexity)
  qlpeps::OptimizerParams opt_params = qlpeps::OptimizerFactory::CreateAdaGradAdvanced(100, 1e-15, 1e-30, 20, 0.05, 1e-8, 0.0);

  // Pure algorithm test (NO VMCPEPSOptimizerExecutor overhead)
  std::string test_name = "TransverseIsing_Trivial";
  RunPureOptimizerTest<Model, TenElemT, QNT, SITPST>(
      transverse_ising_model, split_index_tps, all_configs, trun_para,
      Ly, Lx, energy_exact, opt_params, test_name);
}

struct Z2tJTools : public testing::Test {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using QNSctVecT = QNSectorVec<fZ2QN>;

  using DTensor = QLTensor<TEN_ELEM_TYPE, fZ2QN>;
  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> split_index_tps = SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>(Ly, Lx);
  double t = 1.0;
  double J = 0.3;
  double mu = 0.0;
  double energy_exact = -2.9431635706137875;
  size_t Db = 4;
  fZ2QN qn0 = fZ2QN(0);
  // |ket>
  IndexT loc_phy_ket = IndexT({QNSctT(fZ2QN(1), 2), // |up>, |down>
                               QNSctT(fZ2QN(0), 1)}, // |0> empty state
                              TenIndexDirType::IN
  );
  // <bra|
  IndexT loc_phy_bra = InverseIndex(loc_phy_ket);

  std::vector<Configuration> all_configs;
  void SetUp(void) {
    split_index_tps = SimpleUpdate();
    GenerateAllConfigs();
  }

  SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> SimpleUpdate(void) {
    SquareLatticePEPS<TEN_ELEM_TYPE, fZ2QN> peps0(loc_phy_ket, Ly, Lx);

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx));
    activates = {{2, 2}, {0, 1}};
    peps0.Initial(activates);

    DTensor dham_tj_nn = DTensor({loc_phy_ket, loc_phy_ket, loc_phy_bra, loc_phy_bra});//site: i-j-j-i (i<j)
    dham_tj_nn({2, 0, 2, 0}) = -t;
    dham_tj_nn({2, 1, 2, 1}) = -t;
    dham_tj_nn({0, 2, 0, 2}) = -t;
    dham_tj_nn({1, 2, 1, 2}) = -t;

    dham_tj_nn({0, 0, 0, 0}) = 0.25 * J;    //FM, diagonal element
    dham_tj_nn({1, 1, 1, 1}) = 0.25 * J;    //FM, diagonal element
    dham_tj_nn({0, 1, 1, 0}) = -0.25 * J;   //AFM,diagonal element
    dham_tj_nn({1, 0, 0, 1}) = -0.25 * J;   //AFM,diagonal element
    dham_tj_nn({0, 1, 0, 1}) = 0.5 * J;     //off diagonal element
    dham_tj_nn({1, 0, 1, 0}) = 0.5 * J;     //off diagonal element

    dham_tj_nn.Transpose({3, 0, 2, 1}); // transpose indices order for consistent with simple update convention


    SimpleUpdatePara update_para(100, 0.1, 1, Db, 1e-10);
    SimpleUpdateExecutor<TEN_ELEM_TYPE, fZ2QN>
        *su_exe = new SquareLatticeNNSimpleUpdateExecutor<TEN_ELEM_TYPE, fZ2QN>(update_para, peps0,
                                                                               dham_tj_nn);
    su_exe->Execute();
    auto peps = su_exe->GetPEPS();
    auto tps = TPS<TEN_ELEM_TYPE, fZ2QN>(peps);
    delete su_exe;
    SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> sitps = SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>(tps);
    return sitps;
  }

  //half-filling
  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {2, 2, 0, 1};
    std::sort(config_vec.begin(), config_vec.end());
    do {
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    } while (std::next_permutation(config_vec.begin(),
                                   config_vec.end()));
    // Generates the next lexicographical permutation
  }
};

TEST_F(Z2tJTools, ExactSumGradientOptWithVMCOptimizer) {
  auto trun_para =
      BMPSTruncatePara(Db,
                       Db,
                       0,
                       CompressMPSScheme::SVD_COMPRESS,
                       std::optional<double>(),
                       std::optional<size_t>());
  using Model = SquaretJVModel;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = fZ2QN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;


  Model tj_model(t, 0, J, J / 4, mu);

  // Pure Optimizer parameters - adjusted for better convergence
  // Increased patience and reduced step size for more stable convergence  
  qlpeps::OptimizerParams opt_params = qlpeps::OptimizerFactory::CreateAdaGradAdvanced(500, 1e-15, 1e-30, 50, 0.1, 1e-8, 0.0);

  // Pure algorithm test (NO VMCPEPSOptimizerExecutor overhead)
  std::string test_name = "tJ_Model";
  RunPureOptimizerTest<Model, TenElemT, QNT, SITPST>(
      tj_model, split_index_tps, all_configs, trun_para,
      Ly, Lx, energy_exact, opt_params, test_name);
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
} 