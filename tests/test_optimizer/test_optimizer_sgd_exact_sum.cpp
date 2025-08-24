/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Refactored: 2025-08-24
*
* Description: QuantumLiquids/PEPS project.
* Pure Optimizer unit tests using exact summation for deterministic gradient computation.
* This file targets SGD (with optional momentum) on four small models.
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/optimizer/lr_schedulers.h"
#include "qlpeps/optimizer/optimizer.h"
#include "../utilities.h"
#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::fZ2QN;
using qlten::special_qn::TrivialRepQN;

// Analytical helpers (copied from AdaGrad exact-sum test)
static double Calculate2x2OBCSpinlessFreeFermionEnergy(double t, double t2) {
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

static double Calculate2x2HeisenbergEnergy(double J) { return -2 * J; }

static double Calculate2x2OBCTransverseIsingEnergy(double J, double h) {
  std::vector<double> k_values = {M_PI / 4.0, 3.0 * M_PI / 4.0};
  double ground_state_energy = 0.0;
  for (auto k : k_values) {
    double epsilon_k = 2.0 * std::sqrt(J * J + h * h - 2.0 * J * h * std::cos(k));
    ground_state_energy -= epsilon_k;
  }
  return ground_state_energy;
}

// Common pure-optimizer runner (copied and adapted)
template<typename ModelT, typename TenElemT, typename QNT, typename SITPST>
double RunPureOptimizerSGD(
  ModelT &model,
  SITPST &split_index_tps,
  const std::vector<Configuration> &all_configs,
  const BMPSTruncatePara &trun_para,
  size_t Ly,
  size_t Lx,
  double energy_exact,
  const qlpeps::OptimizerParams &optimizer_params,
  const std::string &test_name,
  const MPI_Comm &comm,
  int rank,
  int mpi_size) {
  Optimizer<TenElemT, QNT> optimizer(optimizer_params, comm, rank, mpi_size);

  auto energy_evaluator = [&](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    auto [energy, gradient, error] = ExactSumEnergyEvaluatorMPI<ModelT, TenElemT, QNT>(
        state, all_configs, trun_para, model, Ly, Lx, comm, rank, mpi_size);
    return {energy, gradient, error};
  };

  typename Optimizer<TenElemT, QNT>::OptimizationCallback callback;
  callback.on_iteration = [&energy_exact, &test_name, rank](size_t iteration, double energy, double error, double gradnorm) {
    if (rank == kMPIMasterRank) {
      std::cout << test_name << " - step: " << iteration
                << " E0=" << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision) << energy
                << " ||grad||=" << std::setw(8) << std::fixed << std::setprecision(kEnergyOutputPrecision) << gradnorm;
      if (energy_exact != 0.0) { std::cout << " exact=" << energy_exact; }
      std::cout << std::endl;
    }
  };

  auto result = optimizer.IterativeOptimize(split_index_tps, energy_evaluator, callback);

  double final_energy = std::real(result.final_energy);
  if (rank == kMPIMasterRank && energy_exact != 0.0) {
    EXPECT_GE(final_energy, energy_exact - 1E-10);
    EXPECT_NEAR(final_energy, energy_exact, 1e-5);
  }
  return final_energy;
}

// ===== Spinless free fermion (Z2) =====
struct Z2SpinlessFreeFermionSGDTools : public MPITest {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, fZ2QN>;

  size_t Lx = 2;
  size_t Ly = 2;

  std::vector<SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> > split_index_tps_list;
  double t = 1.0;
  std::vector<double> t2_list = {2.1, 0, -2.5};
  std::vector<Configuration> all_configs;

  void SetUp(void) override {
    LoadAllPreGeneratedResults();
    GenerateAllConfigs();
  }

  SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> LoadPreGeneratedTPS(double t2) {
    std::string type_suffix;
    if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) {
      type_suffix = "_double_from_simple_update";
    } else if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Complex>) {
      type_suffix = "_complex_from_simple_update";
    }
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" +
                           "spinless_fermion_tps_t2_" + std::to_string(t2) + type_suffix;
    SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> sitps(Ly, Lx);
    if (this->rank == 0) {
      bool success = sitps.Load(data_path);
      if (!success) { throw std::runtime_error("Failed to load: " + data_path); }
      std::cout << "Loaded spinless fermion TPS (t2=" << t2 << ") from: " << data_path << std::endl;
    }
    if (this->mpi_size > 1) { qlpeps::MPI_Bcast(sitps, this->comm, 0); }
    return sitps;
  }

  void LoadAllPreGeneratedResults() {
    for (auto t2 : t2_list) { split_index_tps_list.push_back(LoadPreGeneratedTPS(t2)); }
  }

  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {0, 0, 1, 1};
    do { all_configs.push_back(Vec2Config(config_vec, Lx, Ly)); } while (std::next_permutation(config_vec.begin(), config_vec.end()));
  }
};

TEST_F(Z2SpinlessFreeFermionSGDTools, ExactSumGradientOptWithSGD) {
  auto trun_para = BMPSTruncatePara(8, 8, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = SquareSpinlessFermion;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = fZ2QN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  for (size_t i = 0; i < t2_list.size(); i++) {
    auto t2 = t2_list[i];
    auto energy_exact = Calculate2x2OBCSpinlessFreeFermionEnergy(t, t2);
    Model model(t, t2, 0);
    auto &split_index_tps = split_index_tps_list[i];

    // Faster convergence: StepLR + Nesterov, reduced iterations and patience
    auto scheduler = std::make_unique<qlpeps::StepLR>(0.05, 80, 0.3);
    qlpeps::OptimizerParams::BaseParams base_params(180, 1e-15, 1e-15, 60, 0.05, std::move(scheduler));
    qlpeps::SGDParams sgd_params(0.9, true);
    qlpeps::OptimizerParams opt_params(base_params, sgd_params);

    std::string test_name = std::string("SGD_SpinlessFreeFermion_t2=") + std::to_string(t2);
    RunPureOptimizerSGD<Model, TenElemT, QNT, SITPST>(
      model, split_index_tps, all_configs, trun_para, Ly, Lx, energy_exact, opt_params, test_name, this->comm, this->rank, this->mpi_size);
  }
}

// ===== Heisenberg (Trivial) =====
struct TrivialHeisenbergSGDTools : public MPITest {
  using IndexT = Index<TrivialRepQN>;
  using QNSctT = QNSector<TrivialRepQN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, TrivialRepQN>;

  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> split_index_tps = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>(Ly, Lx);
  double J = 1.0;
  std::vector<Configuration> all_configs;

  void SetUp(void) override {
    split_index_tps = LoadPreGeneratedTPS();
    GenerateAllConfigs();
  }

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> LoadPreGeneratedTPS(void) {
    std::string type_suffix;
    if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) { type_suffix = "_double_from_simple_update"; }
    else if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Complex>) { type_suffix = "_complex_from_simple_update"; }
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" + "heisenberg_tps" + type_suffix;
    SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> sitps(Ly, Lx);
    if (this->rank == 0) {
      bool success = sitps.Load(data_path);
      if (!success) { throw std::runtime_error("Failed to load: " + data_path); }
      std::cout << "Loaded Heisenberg TPS from: " << data_path << std::endl;
    }
    if (this->mpi_size > 1) { qlpeps::MPI_Bcast(sitps, this->comm, 0); }
    return sitps;
  }

  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {0, 0, 1, 1};
    do { all_configs.push_back(Vec2Config(config_vec, Lx, Ly)); } while (std::next_permutation(config_vec.begin(), config_vec.end()));
  }
};

TEST_F(TrivialHeisenbergSGDTools, ExactSumGradientOptWithSGD) {
  auto trun_para = BMPSTruncatePara(1, 8, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = SquareSpinOneHalfXXZModel;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  auto energy_exact = Calculate2x2HeisenbergEnergy(J);
  Model model(J, J, 0);

  // Faster convergence: StepLR + Nesterov, reduced iterations and patience
  auto scheduler = std::make_unique<qlpeps::StepLR>(0.05, 60, 0.4);
  qlpeps::OptimizerParams::BaseParams base_params(120, 1e-15, 1e-30, 45, 0.05, std::move(scheduler));
  qlpeps::SGDParams sgd_params(0.9, true);
  qlpeps::OptimizerParams opt_params(base_params, sgd_params);

  std::string test_name = "SGD_Heisenberg_Trivial";
  RunPureOptimizerSGD<Model, TenElemT, QNT, SITPST>(
    model, split_index_tps, all_configs, trun_para, Ly, Lx, energy_exact, opt_params, test_name, this->comm, this->rank, this->mpi_size);
}

// ===== Transverse Ising (Trivial) =====
struct TrivialTransverseIsingSGDTools : public MPITest {
  using IndexT = Index<TrivialRepQN>;
  using QNSctT = QNSector<TrivialRepQN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, TrivialRepQN>;

  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> split_index_tps = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>(Ly, Lx);
  double h = 1.0;
  double J = 1.0;
  std::vector<Configuration> all_configs;

  void SetUp(void) override {
    split_index_tps = LoadPreGeneratedTPS();
    GenerateAllConfigs();
  }

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> LoadPreGeneratedTPS(void) {
    std::string type_suffix;
    if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) { type_suffix = "_double_from_simple_update"; }
    else if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Complex>) { type_suffix = "_complex_from_simple_update"; }
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" + "transverse_ising_tps" + type_suffix;
    SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> sitps(Ly, Lx);
    if (this->rank == 0) {
      bool success = sitps.Load(data_path);
      if (!success) { throw std::runtime_error("Failed to load: " + data_path); }
      std::cout << "Loaded Transverse Ising TPS from: " << data_path << std::endl;
    }
    if (this->mpi_size > 1) { qlpeps::MPI_Bcast(sitps, this->comm, 0); }
    return sitps;
  }

  void GenerateAllConfigs() {
    for (size_t i = 0; i < 16; i++) {
      std::vector<size_t> config_vec(4);
      config_vec[0] = (i >> 0) & 1;
      config_vec[1] = (i >> 1) & 1;
      config_vec[2] = (i >> 2) & 1;
      config_vec[3] = (i >> 3) & 1;
      all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
    }
  }
};

TEST_F(TrivialTransverseIsingSGDTools, ExactSumGradientOptWithSGD) {
  auto trun_para = BMPSTruncatePara(1, 8, 1e-16, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = TransverseIsingSquare;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  auto energy_exact = Calculate2x2OBCTransverseIsingEnergy(J, h);
  Model model(h);

  // Faster convergence: StepLR + Nesterov, reduced iterations and patience
  auto scheduler = std::make_unique<qlpeps::StepLR>(0.03, 60, 0.4);
  qlpeps::OptimizerParams::BaseParams base_params(120, 1e-15, 1e-30, 40, 0.03, std::move(scheduler));
  qlpeps::SGDParams sgd_params(0.9, true);
  qlpeps::OptimizerParams opt_params(base_params, sgd_params);

  std::string test_name = "SGD_TransverseIsing_Trivial";
  RunPureOptimizerSGD<Model, TenElemT, QNT, SITPST>(
    model, split_index_tps, all_configs, trun_para, Ly, Lx, energy_exact, opt_params, test_name, this->comm, this->rank, this->mpi_size);
}

// ===== t-J model (Z2) =====
struct Z2tJSGDTools : public MPITest {
  using IndexT = Index<fZ2QN>;
  using QNSctT = QNSector<fZ2QN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, fZ2QN>;

  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> split_index_tps = SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN>(Ly, Lx);
  double t = 1.0;
  double J = 0.3;
  double mu = 0.0;
  double energy_exact = -2.9431635706137875;
  size_t Db = 4;
  std::vector<Configuration> all_configs;

  void SetUp(void) override {
    split_index_tps = LoadPreGeneratedTPS();
    GenerateAllConfigs();
  }

  SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> LoadPreGeneratedTPS(void) {
    std::string type_suffix;
    if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Double>) { type_suffix = "_double_from_simple_update"; }
    else if constexpr (std::is_same_v<TEN_ELEM_TYPE, QLTEN_Complex>) { type_suffix = "_complex_from_simple_update"; }
    std::string data_path = std::string(TEST_SOURCE_DIR) + "/test_data/" + "tj_model_tps" + type_suffix;
    SplitIndexTPS<TEN_ELEM_TYPE, fZ2QN> sitps(Ly, Lx);
    if (this->rank == 0) {
      bool success = sitps.Load(data_path);
      if (!success) { throw std::runtime_error("Failed to load: " + data_path); }
      std::cout << "Loaded t-J TPS data from: " << data_path << std::endl;
    }
    if (this->mpi_size > 1) { qlpeps::MPI_Bcast(sitps, this->comm, 0); }
    return sitps;
  }

  void GenerateAllConfigs() {
    std::vector<size_t> config_vec = {2, 2, 0, 1};
    std::sort(config_vec.begin(), config_vec.end());
    do { all_configs.push_back(Vec2Config(config_vec, Lx, Ly)); } while (std::next_permutation(config_vec.begin(), config_vec.end()));
  }
};

TEST_F(Z2tJSGDTools, ExactSumGradientOptWithSGD) {
  auto trun_para = BMPSTruncatePara(Db, Db, 0, CompressMPSScheme::SVD_COMPRESS, std::optional<double>(), std::optional<size_t>());
  using Model = SquaretJVModel;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = fZ2QN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  Model model(t, 0, J, J / 4, mu);

  // Plateau-aware smoke test
  auto scheduler = std::make_unique<qlpeps::PlateauLR>(0.01, 0.5, 30, 1e-6);
  qlpeps::OptimizerParams::BaseParams base_params(700, 1e-15, 1e-30, 350, 0.01, std::move(scheduler));
  qlpeps::SGDParams sgd_params(0.9, false);
  qlpeps::OptimizerParams opt_params(base_params, sgd_params);

  std::string test_name = "SGD_tJ_Model";
  RunPureOptimizerSGD<Model, TenElemT, QNT, SITPST>(
    model, split_index_tps, all_configs, trun_para, Ly, Lx, energy_exact, opt_params, test_name, this->comm, this->rank, this->mpi_size);
}

int main(int argc, char *argv[]) {
  testing::AddGlobalTestEnvironment(new MPIEnvironment);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  return RUN_ALL_TESTS();
}
