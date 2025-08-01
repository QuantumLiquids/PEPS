// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-01-27
*
* Description: QuantumLiquids/PEPS project. Tests for the VMC PEPS optimizer executor.
*/

#include <gtest/gtest.h>
#include <mpi.h>
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/algorithm/vmc_update/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/consts.h"
#include "qlten/qlten.h"
#include "qlpeps/vmc_basic/wave_function_component.h"
#include "../test_mpi_env.h"

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

    // Set up test data path based on data type using CMake-defined source directory
#if TEN_ELEM_TYPE_NUM == 1
    test_data_path = std::string(TEST_SOURCE_DIR) + "/test_algorithm_boson/test_data/tps_square_heisenberg4x4D8Double";
#elif TEN_ELEM_TYPE == QLTEN_Complex
    test_data_path = std::string(TEST_SOURCE_DIR) + "/test_algorithm_boson/test_data/tps_square_heisenberg4x4D8Complex";
#else
#error "Unexpected TEN_ELEM_TYPE_NUM"
#endif

    // Set up VMC parameters using new structure
    OptimizerParams opt_params = OptimizerParams::CreateStochasticReconfiguration(
        {0.1}, ConjugateGradientParams(100, 1e-5, 10, 0.01), 10);
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

  // Test with invalid parameters - empty step lengths should be handled gracefully
  VMCPEPSOptimizerParams invalid_para = optimize_para;
  invalid_para.optimizer_params.core_params.step_lengths.clear();  // Empty step lengths

  // The constructor should handle empty step lengths gracefully, not throw
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

  // Test different optimization schemes
  std::vector<WAVEFUNCTION_UPDATE_SCHEME> schemes = {
      StochasticReconfiguration,
      StochasticGradient,
      NaturalGradientLineSearch,
      GradientLineSearch
  };

  for (auto scheme : schemes) {
    VMCPEPSOptimizerParams scheme_para = optimize_para;
    scheme_para.optimizer_params.update_scheme = scheme;

    auto executor = new VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model>(
        scheme_para, Ly, Lx, comm, model);

    EXPECT_EQ(executor->GetParams().optimizer_params.update_scheme, scheme);

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