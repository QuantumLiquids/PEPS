// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-06
* Refactored: 2026-02-12
*
* Description: QuantumLiquids/PEPS project.
* Pure Optimizer unit tests using exact summation for deterministic gradient computation.
* This file targets L-BFGS optimizer on four small models (strong-Wolfe mode).
*/

#include "gtest/gtest.h"
#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/optimizer/lr_schedulers.h"
#include "qlpeps/optimizer/optimizer.h"
#include <cmath>
#include "../utilities.h"
#include "../test_mpi_env.h"

using namespace qlten;
using namespace qlpeps;

using qlten::special_qn::fZ2QN;
using qlten::special_qn::TrivialRepQN;

namespace {

OptimizerParams CreateStrongWolfeLBFGSParams(size_t max_iterations,
                                             size_t plateau_patience,
                                             double learning_rate) {
  LBFGSParams lbfgs(/*hist=*/10,
                    /*tol_grad=*/1e-8,
                    /*tol_change=*/1e-10,
                    /*max_eval=*/128,
                    /*step_mode=*/LBFGSStepMode::kStrongWolfe,
                    /*wolfe_c1=*/1e-4,
                    /*wolfe_c2=*/0.99,
                    /*min_step=*/1e-8,
                    /*max_step=*/1.0,
                    /*min_curvature=*/1e-12,
                    /*use_damping=*/true,
                    /*max_direction_norm=*/1e3,
                    /*allow_fallback_to_fixed_step=*/false,
                    /*fallback_fixed_step_scale=*/0.2);
  return OptimizerFactory::CreateLBFGSAdvanced(
      max_iterations, /*energy_tolerance=*/1e-12, /*gradient_tolerance=*/1e-6,
      plateau_patience, learning_rate, lbfgs);
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> CreateScalarState(double value) {
  SplitIndexTPS<TenElemT, QNT> s(1, 1, 1);
  Index<QNT> v_out({QNSector<QNT>(QNT(), 1)}, OUT);
  Index<QNT> v_in = InverseIndex(v_out);
  s({0, 0})[0] = QLTensor<TenElemT, QNT>({v_in, v_out, v_out, v_in});
  s({0, 0})[0].Fill(QNT(), value);
  return s;
}

template<typename TenElemT, typename QNT>
double ExtractScalarValue(const SplitIndexTPS<TenElemT, QNT>& s) {
  return s({0, 0})[0].GetMaxAbs();
}

template<typename TenElemT, typename QNT>
SplitIndexTPS<TenElemT, QNT> BuildGradientLike(const SplitIndexTPS<TenElemT, QNT>& state, double grad_value) {
  SplitIndexTPS<TenElemT, QNT> grad = state;
  grad({0, 0})[0].Fill(QNT(), grad_value);
  return grad;
}

}  // namespace

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

// Common pure-optimizer runner for L-BFGS (strong Wolfe)
template<typename ModelT, typename TenElemT, typename QNT, typename SITPST>
double RunPureOptimizerLBFGS(
  ModelT &model,
  SITPST &split_index_tps,
  const std::vector<Configuration> &all_configs,
  const BMPSTruncateParams<typename qlten::RealTypeTrait<TenElemT>::type> &trun_para,
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
    if (rank == qlten::hp_numeric::kMPIMasterRank) {
      std::cout << test_name << " - step: " << iteration
                << " E0=" << std::setw(14) << std::fixed << std::setprecision(kEnergyOutputPrecision) << energy
                << " ||grad||=" << std::setw(8) << std::fixed << std::setprecision(kEnergyOutputPrecision) << gradnorm;
      if (energy_exact != 0.0) { std::cout << " exact=" << energy_exact; }
      std::cout << std::endl;
    }
  };

  auto result = optimizer.IterativeOptimize(split_index_tps, energy_evaluator, callback);

  double final_energy = std::real(result.final_energy);
  if (rank == qlten::hp_numeric::kMPIMasterRank && energy_exact != 0.0) {
    EXPECT_GE(final_energy, energy_exact - 1E-10);
    EXPECT_NEAR(final_energy, energy_exact, 1e-5);
  }
  return final_energy;
}

// ===== Spinless free fermion (Z2) =====
struct Z2SpinlessFreeFermionLBFGSTools : public MPITest {
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
    MPITest::SetUp();
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

TEST_F(Z2SpinlessFreeFermionLBFGSTools, ExactSumGradientOptWithLBFGS) {
  using Model = SquareSpinlessFermion;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = fZ2QN;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(8, 8, 1e-16);
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  for (size_t i = 0; i < t2_list.size(); i++) {
    auto t2 = t2_list[i];
    auto energy_exact = Calculate2x2OBCSpinlessFreeFermionEnergy(t, t2);
    Model model(t, t2, 0);
    auto &split_index_tps = split_index_tps_list[i];

    // L-BFGS (strong-Wolfe) deterministic setup
    qlpeps::OptimizerParams opt_params = CreateStrongWolfeLBFGSParams(
        /*max_iterations=*/140, /*plateau_patience=*/80, /*learning_rate=*/0.05);

    std::string test_name = std::string("LBFGS_SpinlessFreeFermion_t2=") + std::to_string(t2);
    RunPureOptimizerLBFGS<Model, TenElemT, QNT, SITPST>(
      model, split_index_tps, all_configs, trun_para, Ly, Lx, energy_exact, opt_params, test_name, this->comm, this->rank, this->mpi_size);
  }
}

// ===== Heisenberg (Trivial) =====
struct TrivialHeisenbergLBFGSTools : public MPITest {
  using IndexT = Index<TrivialRepQN>;
  using QNSctT = QNSector<TrivialRepQN>;
  using DTensor = QLTensor<TEN_ELEM_TYPE, TrivialRepQN>;

  size_t Lx = 2;
  size_t Ly = 2;

  SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN> split_index_tps = SplitIndexTPS<TEN_ELEM_TYPE, TrivialRepQN>(Ly, Lx);
  double J = 1.0;
  std::vector<Configuration> all_configs;

  void SetUp(void) override {
    MPITest::SetUp();
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

TEST_F(TrivialHeisenbergLBFGSTools, ExactSumGradientOptWithLBFGS) {
  using Model = SquareSpinOneHalfXXZModelOBC;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = TrivialRepQN;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  auto energy_exact = Calculate2x2HeisenbergEnergy(J);
  Model model(J, J, 0);

  // L-BFGS (strong-Wolfe) deterministic setup
  qlpeps::OptimizerParams opt_params = CreateStrongWolfeLBFGSParams(
      /*max_iterations=*/80, /*plateau_patience=*/60, /*learning_rate=*/0.1);

  std::string test_name = "LBFGS_Heisenberg_Trivial";
  RunPureOptimizerLBFGS<Model, TenElemT, QNT, SITPST>(
    model, split_index_tps, all_configs, trun_para, Ly, Lx, energy_exact, opt_params, test_name, this->comm, this->rank, this->mpi_size);
}

// ===== Transverse Ising (Trivial) =====
struct TrivialTransverseIsingLBFGSTools : public MPITest {
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
    MPITest::SetUp();
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

TEST_F(TrivialTransverseIsingLBFGSTools, ExactSumGradientOptWithLBFGS) {
  using Model = TransverseFieldIsingSquareOBC;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = TrivialRepQN;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(1, 8, 1e-16);
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  auto energy_exact = Calculate2x2OBCTransverseIsingEnergy(J, h);
  Model model(h);

  // L-BFGS (strong-Wolfe) deterministic setup
  qlpeps::OptimizerParams opt_params = CreateStrongWolfeLBFGSParams(
      /*max_iterations=*/140, /*plateau_patience=*/80, /*learning_rate=*/0.05);

  std::string test_name = "LBFGS_TransverseIsing_Trivial";
  RunPureOptimizerLBFGS<Model, TenElemT, QNT, SITPST>(
    model, split_index_tps, all_configs, trun_para, Ly, Lx, energy_exact, opt_params, test_name, this->comm, this->rank, this->mpi_size);
}

// ===== t-J model (Z2) =====
struct Z2tJLBFGSTools : public MPITest {
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
    MPITest::SetUp();
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

TEST_F(Z2tJLBFGSTools, ExactSumGradientOptWithLBFGS) {
  using Model = SquaretJVModel;
  using TenElemT = TEN_ELEM_TYPE;
  using QNT = fZ2QN;
  using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
  auto trun_para = BMPSTruncateParams<RealT>::SVD(Db, Db, 0);
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  Model model(t, 0, J, J / 4, mu);

  // L-BFGS (strong-Wolfe) deterministic setup - adjusted for t-J model convergence
  qlpeps::OptimizerParams opt_params = CreateStrongWolfeLBFGSParams(
      /*max_iterations=*/200, /*plateau_patience=*/120, /*learning_rate=*/0.03);

  std::string test_name = "LBFGS_tJ_Model";
  RunPureOptimizerLBFGS<Model, TenElemT, QNT, SITPST>(
    model, split_index_tps, all_configs, trun_para, Ly, Lx, energy_exact, opt_params, test_name, this->comm, this->rank, this->mpi_size);
}

TEST(OptimizerLBFGSStrongWolfeBehavior, SatisfiesStrongWolfeOnConvexQuadratic) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  LBFGSParams lbfgs(/*hist=*/5,
                    /*tol_grad=*/1e-12,
                    /*tol_change=*/1e-14,
                    /*max_eval=*/64,
                    /*step_mode=*/LBFGSStepMode::kStrongWolfe,
                    /*wolfe_c1=*/1e-4,
                    /*wolfe_c2=*/0.9,
                    /*min_step=*/1e-8,
                    /*max_step=*/1.0,
                    /*min_curvature=*/1e-12,
                    /*use_damping=*/true,
                    /*max_direction_norm=*/1e3,
                    /*allow_fallback_to_fixed_step=*/false,
                    /*fallback_fixed_step_scale=*/0.2);
  OptimizerParams params = OptimizerFactory::CreateLBFGSAdvanced(
      /*max_iterations=*/2,
      /*energy_tolerance=*/0.0,
      /*gradient_tolerance=*/0.0,
      /*plateau_patience=*/10,
      /*learning_rate=*/0.5,
      lbfgs);

  Optimizer<TenElemT, QNT> opt(params, MPI_COMM_SELF, 0, 1);
  SITPST init = CreateScalarState<TenElemT, QNT>(1.0);

  auto evaluator = [](const SITPST& state) -> std::tuple<TenElemT, SITPST, double> {
    const double x = ExtractScalarValue<TenElemT, QNT>(state);
    const double g = x - 2.0;                  // f(x) = 0.5 * (x-2)^2
    const double e = 0.5 * (x - 2.0) * (x - 2.0);
    SITPST grad = BuildGradientLike<TenElemT, QNT>(state, g);
    return {TenElemT(e), std::move(grad), 0.0};
  };

  std::vector<double> best_x;
  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
  cb.on_best_state_found = [&best_x](const SITPST& state, double) {
    best_x.push_back(ExtractScalarValue<TenElemT, QNT>(state));
  };

  (void)opt.IterativeOptimize(init, evaluator, cb);

  ASSERT_GE(best_x.size(), 2u);
  const double x0 = best_x[0];
  const double x1 = best_x[1];
  const double g0 = x0 - 2.0;
  const double d0 = -g0;
  ASSERT_NEAR(d0, 1.0, 1e-12);  // first L-BFGS step is steepest descent

  const double alpha = (x1 - x0) / d0;
  const double phi0 = 0.5 * (x0 - 2.0) * (x0 - 2.0);
  const double dphi0 = g0 * d0;
  const double phi_alpha = 0.5 * (x1 - 2.0) * (x1 - 2.0);
  const double dphi_alpha = (x1 - 2.0) * d0;

  EXPECT_LE(phi_alpha, phi0 + lbfgs.wolfe_c1 * alpha * dphi0 + 1e-12);
  EXPECT_LE(std::abs(dphi_alpha), -lbfgs.wolfe_c2 * dphi0 + 1e-12);
}

TEST(OptimizerLBFGSStrongWolfeBehavior, ToleranceGradFloorCanRelaxCurvatureAcceptance) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  auto evaluator = [](const SITPST& state) -> std::tuple<TenElemT, SITPST, double> {
    const double x = ExtractScalarValue<TenElemT, QNT>(state);
    const double g = x - 2.0;  // f(x) = 0.5 * (x - 2)^2
    const double e = 0.5 * g * g;
    SITPST grad = BuildGradientLike<TenElemT, QNT>(state, g);
    return {TenElemT(e), std::move(grad), 0.0};
  };

  auto make_params = [](double tol_grad) {
    LBFGSParams lbfgs(/*hist=*/5,
                      /*tol_grad=*/tol_grad,
                      /*tol_change=*/1e-14,
                      /*max_eval=*/1,
                      /*step_mode=*/LBFGSStepMode::kStrongWolfe,
                      /*wolfe_c1=*/1e-4,
                      /*wolfe_c2=*/0.9,
                      /*min_step=*/0.05,
                      /*max_step=*/0.05,
                      /*min_curvature=*/1e-12,
                      /*use_damping=*/true,
                      /*max_direction_norm=*/1e3,
                      /*allow_fallback_to_fixed_step=*/false,
                      /*fallback_fixed_step_scale=*/0.2);
    return OptimizerFactory::CreateLBFGSAdvanced(
        /*max_iterations=*/1,
        /*energy_tolerance=*/0.0,
        /*gradient_tolerance=*/0.0,
        /*plateau_patience=*/5,
        /*learning_rate=*/0.05,
        lbfgs);
  };

  {
    Optimizer<TenElemT, QNT> strict_opt(make_params(/*tol_grad=*/0.0), MPI_COMM_SELF, 0, 1);
    SITPST init = CreateScalarState<TenElemT, QNT>(2.0 + 1e-8);
    typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
    EXPECT_THROW((void)strict_opt.IterativeOptimize(init, evaluator, cb), std::runtime_error);
  }

  {
    Optimizer<TenElemT, QNT> relaxed_opt(make_params(/*tol_grad=*/1e-16), MPI_COMM_SELF, 0, 1);
    SITPST init = CreateScalarState<TenElemT, QNT>(2.0 + 1e-8);
    typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
    auto result = relaxed_opt.IterativeOptimize(init, evaluator, cb);
    EXPECT_FALSE(result.energy_trajectory.empty());
    EXPECT_TRUE(std::isfinite(std::real(result.final_energy)));
  }
}

TEST(OptimizerLBFGSStrongWolfeBehavior, FailureThrowsUnlessFallbackEnabled) {
  using TenElemT = QLTEN_Double;
  using QNT = qlten::special_qn::TrivialRepQN;
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  auto evaluator = [](const SITPST& state) -> std::tuple<TenElemT, SITPST, double> {
    const double x = ExtractScalarValue<TenElemT, QNT>(state);
    const double g = 2.0 * x;  // f(x) = x^2
    const double e = x * x;
    SITPST grad = BuildGradientLike<TenElemT, QNT>(state, g);
    return {TenElemT(e), std::move(grad), 0.0};
  };

  LBFGSParams strict_wolfe(/*hist=*/5,
                           /*tol_grad=*/1e-12,
                           /*tol_change=*/1e-14,
                           /*max_eval=*/1,
                           /*step_mode=*/LBFGSStepMode::kStrongWolfe,
                           /*wolfe_c1=*/1e-4,
                           /*wolfe_c2=*/0.9,
                           /*min_step=*/0.1,
                           /*max_step=*/10.0,
                           /*min_curvature=*/1e-12,
                           /*use_damping=*/true,
                           /*max_direction_norm=*/1e3,
                           /*allow_fallback_to_fixed_step=*/false,
                           /*fallback_fixed_step_scale=*/0.05);
  OptimizerParams throw_params = OptimizerFactory::CreateLBFGSAdvanced(
      /*max_iterations=*/1, /*energy_tolerance=*/0.0, /*gradient_tolerance=*/0.0,
      /*plateau_patience=*/5, /*learning_rate=*/10.0, strict_wolfe);

  {
    Optimizer<TenElemT, QNT> opt(throw_params, MPI_COMM_SELF, 0, 1);
    SITPST init = CreateScalarState<TenElemT, QNT>(1.0);
    typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
    EXPECT_THROW((void)opt.IterativeOptimize(init, evaluator, cb), std::runtime_error);
  }

  strict_wolfe.allow_fallback_to_fixed_step = true;
  OptimizerParams fallback_params = OptimizerFactory::CreateLBFGSAdvanced(
      /*max_iterations=*/2, /*energy_tolerance=*/0.0, /*gradient_tolerance=*/0.0,
      /*plateau_patience=*/5, /*learning_rate=*/10.0, strict_wolfe);
  Optimizer<TenElemT, QNT> fallback_opt(fallback_params, MPI_COMM_SELF, 0, 1);
  SITPST init = CreateScalarState<TenElemT, QNT>(1.0);
  typename Optimizer<TenElemT, QNT>::OptimizationCallback cb;
  auto result = fallback_opt.IterativeOptimize(init, evaluator, cb);
  EXPECT_FALSE(result.energy_trajectory.empty());
  EXPECT_TRUE(std::isfinite(std::real(result.final_energy)));
}

int main(int argc, char *argv[]) {
  testing::AddGlobalTestEnvironment(new MPIEnvironment);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  return RUN_ALL_TESTS();
}
