// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Tests for the VMC PEPS optimizer executor.
TODO: actually test the VMC PEPS optimizer executor for 2 by 2 systems, as test_mc_peps_measure.cpp does.
*/

#include <gtest/gtest.h>
#include <mpi.h>
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/consts.h"
#include "qlten/qlten.h"
#include "qlpeps/vmc_basic/wave_function_component.h"
#include "../test_mpi_env.h"
#include "../utilities.h"

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;
using TPST = TPS<TenElemT, QNT>;
using SITPST = SplitIndexTPS<TenElemT, QNT>;

class VMCPEPSOptimizerUnitTest : public MPITest {
 protected:
  size_t Lx = 4;  // 4x4 system
  size_t Ly = 4;
  size_t D = 8;   // Bond dimension

  // Helper function to initialize TPS with Open Boundary Conditions (OBC)
  void InitializeTPSWithOBC(SITPST &tps, size_t Ly, size_t Lx) {
    for (size_t row = 0; row < Ly; ++row) {
      for (size_t col = 0; col < Lx; ++col) {
        for (size_t i = 0; i < 2; ++i) {  // 2 physical states (0 and 1)
          // Create tensor with 4 indices: (left, down, right, up) = (in, out, out, in)
          std::vector<IndexT> indices;

          // Left index (incoming)
          size_t left_dim = (col == 0) ? 1 : D;  // OBC: left boundary has dim 1
          indices.push_back(IndexT({QNSctT(QNT(), left_dim)}, TenIndexDirType::IN));

          // Down index (outgoing)
          size_t down_dim = (row == Ly - 1) ? 1 : D;  // OBC: bottom boundary has dim 1
          indices.push_back(IndexT({QNSctT(QNT(), down_dim)}, TenIndexDirType::OUT));

          // Right index (outgoing)
          size_t right_dim = (col == Lx - 1) ? 1 : D;  // OBC: right boundary has dim 1
          indices.push_back(IndexT({QNSctT(QNT(), right_dim)}, TenIndexDirType::OUT));

          // Up index (incoming)
          size_t up_dim = (row == 0) ? 1 : D;  // OBC: top boundary has dim 1
          indices.push_back(IndexT({QNSctT(QNT(), up_dim)}, TenIndexDirType::IN));

          Tensor tensor(indices);

          tensor.Random(QNT());

          tps({row, col})[i] = tensor;
        }
      }
    }
  }

  QNT qn0 = QNT();
  IndexT pb_out;
  IndexT pb_in;

  VMCPEPSOptimizerParams optimize_para;
  std::string test_data_path;

  void SetUp() override {
    MPITest::SetUp();

    // Set up physical indices
    pb_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
    pb_in = InverseIndex(pb_out);

    // Set up test data paths: read reference data from source, write results to build directory
    std::string reference_data_path;
    std::string output_data_subdir;
#if TEN_ELEM_TYPE_NUM == 1
    reference_data_path = std::string(TEST_SOURCE_DIR) + "/slow_tests/test_data/tps_square_heisenberg4x4D8Double";
    output_data_subdir = "tps_square_heisenberg4x4D8Double";
#elif TEN_ELEM_TYPE == QLTEN_Complex
    reference_data_path = std::string(TEST_SOURCE_DIR) + "/slow_tests/test_data/tps_square_heisenberg4x4D8Complex";
    output_data_subdir = "tps_square_heisenberg4x4D8Complex";
#else
#error "Unexpected TEN_ELEM_TYPE_NUM"
#endif

    // Create output directory and copy reference data
    test_data_path = GetTestOutputPath("vmc_peps_optimizer", output_data_subdir);
    
    // Copy reference TPS data to output directory so optimizer can read and modify it
    if (std::filesystem::exists(reference_data_path)) {
      std::filesystem::copy(reference_data_path, test_data_path, 
                          std::filesystem::copy_options::overwrite_existing | 
                          std::filesystem::copy_options::recursive);
    }

    // Set up VMC parameters using new structure
    ConjugateGradientParams cg_params(100, 1e-5, 10, 0.01);
    OptimizerParams opt_params = OptimizerFactory::CreateStochasticReconfigurationAdvanced(
        10, 1e-15, 1e-30, 20, cg_params, 0.1);
    Configuration random_config(Ly, Lx);
    std::vector<size_t> occupancy = {Ly * Lx / 2, Ly * Lx / 2};  // Equal number of 0s and 1s
    random_config.Random(occupancy);
    MonteCarloParams mc_params(10, 10, 1, "", random_config);
    PEPSParams peps_params(BMPSTruncatePara(4, 8, 1e-15,
                                            CompressMPSScheme::SVD_COMPRESS,
                                            std::make_optional<double>(1e-14),
                                            std::make_optional<size_t>(10)), test_data_path);
    optimize_para = VMCPEPSOptimizerParams(opt_params, mc_params, peps_params);
  }
};

// Test VMC PEPS Optimizer Executor Construction with TPS loading
TEST_F(VMCPEPSOptimizerUnitTest, ConstructorWithTPS) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  // Create a simple TPS in memory for testing
  SITPST test_tps(Ly, Lx, 2);

  // Initialize with simple data
  InitializeTPSWithOBC(test_tps, Ly, Lx);

  // Test constructor with TPS in memory
  auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, test_tps, comm, model);

  EXPECT_EQ(executor->GetParams().mc_params.alternative_init_config.rows(), Ly);
  EXPECT_EQ(executor->GetParams().mc_params.alternative_init_config.cols(), Lx);

  delete executor;
}

// Test VMC PEPS Optimizer Parameter Validation
TEST_F(VMCPEPSOptimizerUnitTest, ParameterValidation) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  // Create a simple TPS in memory for testing
  SITPST test_tps(Ly, Lx, 2);

  // Initialize with simple data
  InitializeTPSWithOBC(test_tps, Ly, Lx);

  // Test with invalid parameters - negative learning rate should be handled gracefully
  VMCPEPSOptimizerParams invalid_para = optimize_para;
  invalid_para.optimizer_params.base_params.learning_rate = -1.0;  // Invalid negative learning rate

  // The constructor should handle invalid learning rate gracefully, not throw
  EXPECT_NO_THROW(
      (void) (new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
          invalid_para, test_tps, comm, model)));
}

// Test VMC PEPS Optimizer State Management
TEST_F(VMCPEPSOptimizerUnitTest, StateManagement) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, Ly, Lx, comm, model);

  // Test initial energy
  auto initial_energy = executor->GetCurrentEnergy();
  EXPECT_EQ(initial_energy, std::numeric_limits<double>::max());

  // Test min energy - should be initialized to max double, not 0
  auto min_energy = executor->GetMinEnergy();
  EXPECT_EQ(min_energy, std::numeric_limits<double>::max());  // Should be initialized to max

  delete executor;
}

// Test VMC PEPS Optimizer Optimization Schemes
TEST_F(VMCPEPSOptimizerUnitTest, OptimizationSchemes) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  // Test different optimization algorithms
  std::vector<std::string> algorithm_names = {
      "StochasticReconfiguration",
      "SGD",
      "AdaGrad"
  };

  for (const auto& name : algorithm_names) {
    VMCPEPSOptimizerParams scheme_para = optimize_para;
    
    // Update optimizer params for each algorithm type
    if (name == "StochasticReconfiguration") {
      ConjugateGradientParams cg_params(100, 1e-5, 10, 0.01);
      scheme_para.optimizer_params = OptimizerFactory::CreateStochasticReconfigurationAdvanced(10, 1e-15, 1e-30, 20, cg_params, 0.1);
    } else if (name == "SGD") {
      OptimizerParams::BaseParams base_params(10, 1e-15, 1e-15, 20, 0.1);
      SGDParams sgd_params(0.0, false);
      scheme_para.optimizer_params = OptimizerParams(base_params, sgd_params);
    } else if (name == "AdaGrad") {
      scheme_para.optimizer_params = OptimizerFactory::CreateAdaGradAdvanced(10, 1e-15, 1e-30, 20, 0.1, 1e-8, 0.0);
    }

    auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
        scheme_para, Ly, Lx, comm, model);

    // Test that the correct algorithm type is configured
    if (name == "StochasticReconfiguration") {
      EXPECT_TRUE(executor->GetParams().optimizer_params.IsAlgorithm<StochasticReconfigurationParams>());
    } else if (name == "SGD") {
      EXPECT_TRUE(executor->GetParams().optimizer_params.IsAlgorithm<SGDParams>());
    } else if (name == "AdaGrad") {
      EXPECT_TRUE(executor->GetParams().optimizer_params.IsAlgorithm<AdaGradParams>());
    }

    delete executor;
  }
}

// Test VMC PEPS Optimizer with Different Models
TEST_F(VMCPEPSOptimizerUnitTest, DifferentModels) {
  using MCUpdater = MCUpdateSquareNNExchange;

  // Test with different energy solvers
  {
    using Model = SquareSpinOneHalfXXZModel;
    Model model;

    auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
        optimize_para, Ly, Lx, comm, model);
    delete executor;
  }

  {
    using Model = SquareSpinOneHalfJ1J2XXZModel;
    Model model(1.0, 1.0, 0.2, 0.2, 0.0);

    auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
        optimize_para, Ly, Lx, comm, model);
    delete executor;
  }
}

// Test VMC PEPS Optimizer with Different MC Updaters
TEST_F(VMCPEPSOptimizerUnitTest, DifferentMCUpdaters) {
  using Model = SquareSpinOneHalfXXZModel;
  Model model;

  // Test with different MC updaters
  {
    using MCUpdater = MCUpdateSquareNNExchange;
    auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
        optimize_para, Ly, Lx, comm, model);
    delete executor;
  }

  {
    using MCUpdater = MCUpdateSquareNNFullSpaceUpdate;
    auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
        optimize_para, Ly, Lx, comm, model);
    delete executor;
  }
}

// Test VMC PEPS Optimizer Data Dumping
TEST_F(VMCPEPSOptimizerUnitTest, DataDumping) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, Ly, Lx, comm, model);

  // Test DumpData without path
  EXPECT_NO_THROW(executor->DumpData(false));

  // Test DumpData with path
  EXPECT_NO_THROW(executor->DumpData("test_vmc_optimizer_dump", false));

  delete executor;
}

// Test VMC PEPS Optimizer BMPSTruncatePara
TEST_F(VMCPEPSOptimizerUnitTest, BMPSTruncatePara) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  // Test different BMPSTruncatePara configurations
  VMCPEPSOptimizerParams para = optimize_para;
  para.peps_params.truncate_para = BMPSTruncatePara(2, 4, 1e-10,
                                                    CompressMPSScheme::VARIATION1Site,
                                                    std::make_optional<double>(1e-9),
                                                    std::make_optional<size_t>(5));

  auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
      para, Ly, Lx, comm, model);

  EXPECT_EQ(executor->GetParams().peps_params.truncate_para.D_min, 2);
  EXPECT_EQ(executor->GetParams().peps_params.truncate_para.D_max, 4);
  EXPECT_EQ(executor->GetParams().peps_params.truncate_para.compress_scheme,
            CompressMPSScheme::VARIATION1Site);

  delete executor;
}

// Test VMC PEPS Optimizer Interface Compatibility
TEST_F(VMCPEPSOptimizerUnitTest, InterfaceCompatibility) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, Ly, Lx, comm, model);

  // Test that all interface methods exist and work
  EXPECT_NO_THROW(executor->GetState());
  EXPECT_NO_THROW(executor->GetOptimizedState());
  EXPECT_NO_THROW(executor->GetBestState());
  EXPECT_NO_THROW(executor->GetMinEnergy());
  EXPECT_NO_THROW(executor->GetMinEnergy());
  EXPECT_NO_THROW(executor->GetCurrentEnergy());
  EXPECT_NO_THROW(executor->GetOptimizer());

  delete executor;
}

// Test VMC PEPS Optimizer Callback System
TEST_F(VMCPEPSOptimizerUnitTest, CallbackSystem) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, Ly, Lx, comm, model);

  // Test callback setting
  typename VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>::OptimizerT::OptimizationCallback callback;
  callback.on_iteration = [](size_t iteration, double energy, double energy_error, double gradient_norm) {
    // Callback should be called during optimization
  };

  EXPECT_NO_THROW(executor->SetOptimizationCallback(callback));

  delete executor;
}

// Test VMC PEPS Optimizer Custom Energy Evaluator
TEST_F(VMCPEPSOptimizerUnitTest, CustomEnergyEvaluator) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, Ly, Lx, comm, model);

  // Test custom energy evaluator setting
  auto custom_evaluator = [](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
    return {0.0, state, 0.0}; // Simple mock evaluator
  };

  EXPECT_NO_THROW(executor->SetEnergyEvaluator(custom_evaluator));

  delete executor;
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  hp_numeric::SetTensorManipulationThreads(1);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
} 