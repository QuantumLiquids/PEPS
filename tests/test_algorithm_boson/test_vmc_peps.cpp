// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-07-25
*
* Description: QuantumLiquids/PEPS project. Unit tests for VMC Optimization in PEPS.
*/

#include "gtest/gtest.h"

#include "qlten/qlten.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/algorithm/vmc_update/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "../test_mpi_env.h"
#include <filesystem>

using namespace qlten;
using namespace qlpeps;

using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;
using TPST = TPS<TenElemT, QNT>;
using SITPST = SplitIndexTPS<TenElemT, QNT>;

class VMCPEPSUnitTest : public MPITest {
 protected:
  size_t Lx = 4;  // Changed to 4x4 system
  size_t Ly = 4;
  size_t D = 8;   // Bond dimension from test data

  QNT qn0 = QNT();
  IndexT pb_out;
  IndexT pb_in;

  VMCOptimizePara optimize_para;
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

    // Debug print to see what path is being used
#if TEN_ELEM_TYPE_NUM == 1
    std::cout << "DEBUG: TEN_ELEM_TYPE = QLTEN_Double" << std::endl;
#elif TEN_ELEM_TYPE_NUM == 2
    std::cout << "DEBUG: TEN_ELEM_TYPE = QLTEN_Complex" << std::endl;
#else
    std::cout << "DEBUG: TEN_ELEM_TYPE = UNKNOWN" << std::endl;
#endif
    std::cout << "DEBUG: TEST_SOURCE_DIR = " << TEST_SOURCE_DIR << std::endl;
    std::cout << "DEBUG: test_data_path = " << test_data_path << std::endl;

    // Set up VMC parameters
    optimize_para = VMCOptimizePara(
        BMPSTruncatePara(4, 8, 1e-15,
                         CompressMPSScheme::SVD_COMPRESS,
                         std::make_optional<double>(1e-14),
                         std::make_optional<size_t>(10)),
        10, 10, 1,  // Reduced for unit tests
        std::vector<size_t>(2, Lx * Ly / 2),
        Ly, Lx,
        std::vector<double>(5, 0.1),  // Reduced for unit tests
        StochasticReconfiguration,
        ConjugateGradientParams(10, 1e-4, 5, 0.01),
        test_data_path);  // Use the test data path
  }
};

// Test VMC PEPS Executor Construction with TPS loading
TEST_F(VMCPEPSUnitTest, ConstructorWithTPS) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  // Test constructor with TPS loading from test data
  auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, Ly, Lx, comm, model);

  EXPECT_EQ(executor->GetOptimizePara().init_config.rows(), Ly);
  EXPECT_EQ(executor->GetOptimizePara().init_config.cols(), Lx);

  delete executor;
}

// Test VMC PEPS Parameter Validation
TEST_F(VMCPEPSUnitTest, ParameterValidation) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  // Test with invalid parameters - empty step lengths should be handled gracefully
  VMCOptimizePara invalid_para = optimize_para;
  invalid_para.step_lens.clear();  // Empty step lengths

  // The constructor should handle empty step_lens gracefully, not throw
  EXPECT_NO_THROW(
      (void)(new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
          invalid_para, Ly, Lx, comm, model)));
}

// Test VMC PEPS State Management
TEST_F(VMCPEPSUnitTest, StateManagement) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, Ly, Lx, comm, model);

  // Test initial energy
  auto initial_energy = executor->GetCurrentEnergy();
  EXPECT_EQ(initial_energy, std::numeric_limits<double>::max());

  // Test min energy - should be initialized to max double, not 0
  auto min_energy = executor->GetMinEnergy();
  EXPECT_EQ(min_energy, std::numeric_limits<double>::max());  // Should be initialized to max

  delete executor;
}

// Test VMC PEPS Optimization Schemes
TEST_F(VMCPEPSUnitTest, OptimizationSchemes) {
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
    VMCOptimizePara scheme_para = optimize_para;
    scheme_para.update_scheme = scheme;

    auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
        scheme_para, Ly, Lx, comm, model);

    EXPECT_EQ(executor->GetOptimizePara().update_scheme, scheme);

    delete executor;
  }
}

// Test VMC PEPS with Different Models
TEST_F(VMCPEPSUnitTest, DifferentModels) {
  using MCUpdater = MCUpdateSquareNNExchange;

  // Test with different energy solvers
  {
    using Model = SquareSpinOneHalfXXZModel;
    Model model;

    auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
        optimize_para, Ly, Lx, comm, model);
    delete executor;
  }

  {
    using Model = SquareSpinOneHalfJ1J2XXZModel;
    Model model(1.0, 1.0, 0.2, 0.2, 0.0);

    auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
        optimize_para, Ly, Lx, comm, model);
    delete executor;
  }
}

// Test VMC PEPS with Different MC Updaters
TEST_F(VMCPEPSUnitTest, DifferentMCUpdaters) {
  using Model = SquareSpinOneHalfXXZModel;
  Model model;

  // Test with different MC updaters
  {
    using MCUpdater = MCUpdateSquareNNExchange;
    auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
        optimize_para, Ly, Lx, comm, model);
    delete executor;
  }

  {
    using MCUpdater = MCUpdateSquareNNFullSpaceUpdate;
    auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
        optimize_para, Ly, Lx, comm, model);
    delete executor;
  }
}

// Test VMC PEPS Data Dumping
TEST_F(VMCPEPSUnitTest, DataDumping) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
      optimize_para, Ly, Lx, comm, model);

  // Test DumpData without path
  EXPECT_NO_THROW(executor->DumpData(false));

  // Test DumpData with path
  EXPECT_NO_THROW(executor->DumpData("test_vmc_dump", false));

  delete executor;
}

// Test VMC PEPS BMPSTruncatePara
TEST_F(VMCPEPSUnitTest, BMPSTruncatePara) {
  using Model = SquareSpinOneHalfXXZModel;
  using MCUpdater = MCUpdateSquareNNExchange;

  Model model;

  // Test different BMPSTruncatePara configurations
  VMCOptimizePara para = optimize_para;
  para.bmps_trunc_para = BMPSTruncatePara(2, 4, 1e-10,
                                          CompressMPSScheme::VARIATION1Site,
                                          std::make_optional<double>(1e-9),
                                          std::make_optional<size_t>(5));

  auto executor = new VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model>(
      para, Ly, Lx, comm, model);

  EXPECT_EQ(executor->GetOptimizePara().bmps_trunc_para.D_min, 2);
  EXPECT_EQ(executor->GetOptimizePara().bmps_trunc_para.D_max, 4);
  EXPECT_EQ(executor->GetOptimizePara().bmps_trunc_para.compress_scheme,
            CompressMPSScheme::VARIATION1Site);

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
