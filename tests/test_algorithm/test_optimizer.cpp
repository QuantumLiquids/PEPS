// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-07-30
*
* Description: QuantumLiquids/PEPS project. Tests for the optimizer.
*/

#include <gtest/gtest.h>
#include <mpi.h>
#include <sstream>
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/consts.h"
#include "qlten/qlten.h"

namespace qlpeps {
using namespace qlten;

class OptimizerTest : public ::testing::Test {
  protected:
    using TenElemT = double;
    using QNT = qlten::special_qn::U1QN;
    using SITPST = SplitIndexTPS<TenElemT, QNT>;
    using OptimizerT = Optimizer<TenElemT, QNT>;
    using Tensor = QLTensor<TenElemT, QNT>;
    using IndexT = Index<QNT>;
    using QNSctT = QNSector<QNT>;

    static constexpr size_t Ly = 2;
    static constexpr size_t Lx = 2;
    static constexpr size_t D = 2;

    MPI_Comm comm_;
    int rank_;
    int mpi_size_;

    OptimizerParams test_params_;
    SITPST test_tps_;

    void SetUp() override {
      comm_ = MPI_COMM_WORLD;
      MPI_Comm_rank(comm_, &rank_);
      MPI_Comm_size(comm_, &mpi_size_);

      // Disable output from non-master ranks
      if (rank_ != kMPIMasterRank) {
        std::cout.setstate(std::ios_base::failbit);
      }

      // Set up test parameters with realistic convergence criteria for testing
      // Mock evaluator changes energy by ~0.01 per iteration, so set reasonable tolerances
      test_params_ = OptimizerParams(
        OptimizerParams::BaseParams(1000, 1e-6, 1e-6, 20, {0.1, 0.05, 0.01}),
        StochasticGradient);

      // Create a simple test TPS
      test_tps_ = CreateRandTestTPS();
    }

    void TearDown() override {
      if (rank_ != kMPIMasterRank) {
        std::cout.clear();
      }
    }

    SITPST CreateRandTestTPS() {
      SITPST tps(Ly, Lx, D);

      // Create simple tensors for testing
      for (size_t row = 0; row < Ly; ++row) {
        for (size_t col = 0; col < Lx; ++col) {
          for (size_t i = 0; i < D; ++i) {
            // Create a simple tensor with some non-zero elements
            QNT qn0 = QNT();
            IndexT index0 = IndexT({QNSctT(qn0, 2)}, TenIndexDirType::IN);
            IndexT index1 = IndexT({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
            IndexT index2 = IndexT({QNSctT(qn0, 2)}, TenIndexDirType::OUT);
            IndexT index3 = IndexT({QNSctT(qn0, 2)}, TenIndexDirType::IN);

            Tensor tensor({index0, index1, index2, index3});
            tensor.Random(qn0);
            tps({row, col})[i] = tensor;
          }
        }
      }

      return tps;
    }

    // Mock energy evaluator for testing (with error support)
    std::tuple<TenElemT, SITPST, double> MockEnergyEvaluator(const SITPST &state) {
      static size_t call_count = 0;
      call_count++;

      TenElemT energy = 0.0;
      double error = 0.0;

      for (size_t row = 0; row < state.rows(); ++row) {
        for (size_t col = 0; col < state.cols(); ++col) {
          for (size_t i = 0; i < state({row, col}).size(); ++i) {
            double norm = state({row, col})[i].GetMaxAbs();
            energy += norm * norm;
            error += norm * 0.1; // Mock error estimate
          }
        }
      }

      // Simple decreasing energy for convergence testing
      energy -= call_count * 0.01; // Energy decreases by 0.01 per iteration

      // Mock gradient (not related to actual energy function)
      SITPST gradient = 0.01 * state; // a small gradient so that have small effect on energy improvement.
      // if use gradient = state, then the gradient will converge sometime as the gradient of cost function |state|^2.

      // // Scale gradient down as optimization progresses (simulating convergence)
      // double gradient_scale = std::max(0.00001, 1.0 - call_count * 0.1);
      // gradient *= gradient_scale;

      return {energy, gradient, error};
    }

    // Helper function to get max absolute value of a SplitIndexTPS
    double GetMaxAbs(const SITPST &tps) {
      double max_abs = 0.0;
      for (size_t row = 0; row < tps.rows(); ++row) {
        for (size_t col = 0; col < tps.cols(); ++col) {
          for (size_t i = 0; i < tps({row, col}).size(); ++i) {
            max_abs = std::max(max_abs, tps({row, col})[i].GetMaxAbs());
          }
        }
      }
      return max_abs;
    }
};

// Test optimizer construction
TEST_F(OptimizerTest, Construction) {
  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);
  EXPECT_TRUE(true); // Should not throw
}

// Test static methods
TEST_F(OptimizerTest, StaticMethods) {
  EXPECT_TRUE(OptimizerT::UsesStochasticReconfiguration(StochasticReconfiguration));
  EXPECT_TRUE(OptimizerT::UsesStochasticReconfiguration(RandomStepStochasticReconfiguration));
  EXPECT_TRUE(OptimizerT::UsesStochasticReconfiguration(NormalizedStochasticReconfiguration));
  EXPECT_FALSE(OptimizerT::UsesStochasticReconfiguration(StochasticGradient));
  EXPECT_FALSE(OptimizerT::UsesStochasticReconfiguration(RandomGradientElement));

  EXPECT_TRUE(OptimizerT::IsLineSearchScheme(GradientLineSearch));
  EXPECT_TRUE(OptimizerT::IsLineSearchScheme(NaturalGradientLineSearch));
  EXPECT_FALSE(OptimizerT::IsLineSearchScheme(StochasticGradient));
  EXPECT_FALSE(OptimizerT::IsLineSearchScheme(RandomGradientElement));
}

// Test gradient update
TEST_F(OptimizerTest, GradientUpdate) {
  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  SITPST initial_state = test_tps_;
  SITPST gradient = test_tps_; // Use same structure for gradient

  // Fill gradient with some values
  for (size_t row = 0; row < gradient.rows(); ++row) {
    for (size_t col = 0; col < gradient.cols(); ++col) {
      for (size_t i = 0; i < gradient({row, col}).size(); ++i) {
        QNT qn0 = QNT();
        gradient({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
        gradient({row, col})[i].Random(qn0);
      }
    }
  }

  double step_length = 0.1;
  SITPST updated_state = optimizer.UpdateTPSByGradient(initial_state, gradient, step_length);

  // Check that the state was updated
  EXPECT_NE(GetMaxAbs(updated_state), GetMaxAbs(initial_state));
}

// Test line search optimization
TEST_F(OptimizerTest, LineSearchOptimization) {
  // Use gradient line search for simplicity
  test_params_ =
      OptimizerParams(OptimizerParams::BaseParams(1000, 1e-2, 1e-4, 20, {0.1, 0.05, 0.01}), GradientLineSearch);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  auto result = optimizer.LineSearchOptimize(test_tps_,
                                             [this](const SITPST &state) { return MockEnergyEvaluator(state); });

  EXPECT_TRUE(result.converged);
  EXPECT_FALSE(result.energy_trajectory.empty());
  EXPECT_FALSE(result.gradient_norms.empty());

  // Energy should be finite
  EXPECT_TRUE(std::isfinite(result.final_energy));
  EXPECT_TRUE(std::isfinite(result.min_energy));
}

// Test iterative optimization
TEST_F(OptimizerTest, IterativeOptimization) {
  test_params_ = OptimizerParams(
    OptimizerParams::BaseParams(1000, 1e-15, 1e-15, 1000, {0.1, 0.05, 0.01}),
    StochasticGradient);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  auto result = optimizer.IterativeOptimize(test_tps_,
                                            [this](const SITPST &state) { return MockEnergyEvaluator(state); });

  EXPECT_FALSE(result.converged);
  EXPECT_FALSE(result.energy_trajectory.empty());
  EXPECT_FALSE(result.gradient_norms.empty());

  // Energy should be finite
  EXPECT_TRUE(std::isfinite(result.final_energy));
  EXPECT_TRUE(std::isfinite(result.min_energy));
}

// Test optimization with callbacks
TEST_F(OptimizerTest, OptimizationWithCallbacks) {
  test_params_ = OptimizerParams::CreateStochasticGradient({0.1, 0.05}, 1000);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  typename OptimizerT::OptimizationCallback callback;
  std::vector<double> callback_energies;
  std::vector<size_t> callback_iterations;

  callback.on_iteration = [&](size_t iteration, double energy, double energy_error, double gradient_norm) {
    callback_iterations.push_back(iteration);
    callback_energies.push_back(energy);
  };

  callback.on_best_state_found = [](const SITPST &state, double energy) {
    // This should be called when a better state is found
  };

  auto result = optimizer.IterativeOptimize(test_tps_,
                                            [this](const SITPST &state) { return MockEnergyEvaluator(state); },
                                            callback);

  EXPECT_EQ(callback_iterations.size(), callback_energies.size());
}

// Test time logging functionality
TEST_F(OptimizerTest, TimeLogging) {
  test_params_ = OptimizerParams::CreateStochasticGradient({0.1, 0.05, 0.01}, 1000);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Capture output to verify time logging
  std::stringstream captured_output;
  std::streambuf *original_cout = nullptr;

  if (rank_ == kMPIMasterRank) {
    original_cout = std::cout.rdbuf();
    std::cout.rdbuf(captured_output.rdbuf());
  }

  auto result = optimizer.IterativeOptimize(test_tps_,
                                            [this](const SITPST &state) { return MockEnergyEvaluator(state); });

  if (rank_ == kMPIMasterRank) {
    std::cout.rdbuf(original_cout);

    // Verify that the output contains time logging
    std::string output = captured_output.str();
    EXPECT_FALSE(result.converged);

    // Check that time logging appears in the output
    // The format should be "TotT = X.XXs" where X.XX is the time
    EXPECT_TRUE(output.find("TotT =") != std::string::npos);
    EXPECT_TRUE(output.find("s") != std::string::npos);

    // Verify that each iteration has time logging
    size_t totT_count = 0;
    size_t pos = 0;
    while ((pos = output.find("TotT =", pos)) != std::string::npos) {
      totT_count++;
      pos += 6; // Move past "TotT ="
    }

    // With advanced stop functionality, the optimizer can run for max_iterations
    // if convergence criteria are not met. The time logging should match iterations.
    EXPECT_GT(totT_count, 0);
    EXPECT_EQ(totT_count, result.total_iterations);
    EXPECT_LE(result.total_iterations, test_params_.base_params.max_iterations);
  }
}

TEST_F(OptimizerTest, TimeLoggingImposeStoppingAdvance) {
  test_params_ = OptimizerParams::CreateStochasticGradient({0.1, 0.05, 0.01, 0}, 1000);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Capture output to verify time logging
  std::stringstream captured_output;
  std::streambuf *original_cout = nullptr;

  if (rank_ == kMPIMasterRank) {
    original_cout = std::cout.rdbuf();
    std::cout.rdbuf(captured_output.rdbuf());
  }

  auto result = optimizer.IterativeOptimize(test_tps_,
                                            [this](const SITPST &state) { return MockEnergyEvaluator(state); });

  if (rank_ == kMPIMasterRank) {
    std::cout.rdbuf(original_cout);

    // Verify that the output contains time logging
    std::string output = captured_output.str();
    EXPECT_FALSE(result.converged);

    // Check that time logging appears in the output
    // The format should be "TotT = X.XXs" where X.XX is the time
    EXPECT_TRUE(output.find("TotT =") != std::string::npos);
    EXPECT_TRUE(output.find("s") != std::string::npos);

    // Verify that each iteration has time logging
    size_t totT_count = 0;
    size_t pos = 0;
    while ((pos = output.find("TotT =", pos)) != std::string::npos) {
      totT_count++;
      pos += 6; // Move past "TotT ="
    }

    // With advanced stop functionality, the optimizer can run for max_iterations
    // if convergence criteria are not met. The time logging should match iterations.
    EXPECT_GT(totT_count, 0);
    EXPECT_EQ(totT_count, result.total_iterations);
    EXPECT_LE(result.total_iterations, test_params_.base_params.max_iterations);
  }
}

// Test different optimization schemes
TEST_F(OptimizerTest, DifferentOptimizationSchemes) {
  std::vector<WAVEFUNCTION_UPDATE_SCHEME> schemes = {
    StochasticGradient,
    RandomStepStochasticGradient,
    RandomGradientElement,
    BoundGradientElement
  };

  for (auto scheme : schemes) {
    test_params_ = OptimizerParams(OptimizerParams::BaseParams(1000, 1e-15, 1e-15, 1000, {0.1}), scheme);

    OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

    auto result = optimizer.IterativeOptimize(test_tps_,
                                              [this](const SITPST &state) { return MockEnergyEvaluator(state); });

    EXPECT_FALSE(result.converged);
    EXPECT_TRUE(std::isfinite(result.final_energy));
  }
}

// Test error handling for invalid schemes
TEST_F(OptimizerTest, ErrorHandling) {
  test_params_.update_scheme = static_cast<WAVEFUNCTION_UPDATE_SCHEME>(999); // Invalid scheme

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  EXPECT_THROW(
    optimizer.IterativeOptimize(test_tps_,
      [this](const SITPST &state) { return MockEnergyEvaluator(state); }),
    std::runtime_error
  );
}

// Test stochastic reconfiguration structure (basic test)
TEST_F(OptimizerTest, StochasticReconfigurationStructure) {
  test_params_ =
      OptimizerParams::CreateStochasticReconfiguration({0.1}, ConjugateGradientParams(100, 1e-6, 0, 10), 1000);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Test that the optimizer can be constructed with stochastic reconfiguration
  // We'll skip the actual optimization since it requires proper gradient samples
  EXPECT_TRUE(true); // Just test that construction succeeds
}

// Test bounded gradient update
TEST_F(OptimizerTest, BoundedGradientUpdate) {
  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  SITPST initial_state = test_tps_;
  SITPST gradient = test_tps_;

  // Fill gradient with some values
  for (size_t row = 0; row < gradient.rows(); ++row) {
    for (size_t col = 0; col < gradient.cols(); ++col) {
      for (size_t i = 0; i < gradient({row, col}).size(); ++i) {
        QNT qn0 = QNT();
        gradient({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
        gradient({row, col})[i].Random(qn0);
      }
    }
  }
  gradient *= 100.0; //Make gradient large enough to trigger bounded update.

  double step_length = 0.1;
  SITPST updated_state = optimizer.BoundedGradientUpdate(initial_state, gradient, step_length);

  EXPECT_NE(GetMaxAbs(updated_state), GetMaxAbs(initial_state));
}

// Test random gradient update
TEST_F(OptimizerTest, RandomGradientUpdate) {
  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  SITPST initial_state = test_tps_;
  SITPST gradient = test_tps_;

  // Fill gradient with some values
  for (size_t row = 0; row < gradient.rows(); ++row) {
    for (size_t col = 0; col < gradient.cols(); ++col) {
      for (size_t i = 0; i < gradient({row, col}).size(); ++i) {
        QNT qn0 = QNT();
        gradient({row, col})[i] = Tensor(gradient({row, col})[i].GetIndexes());
        gradient({row, col})[i].Random(qn0);
      }
    }
  }

  double step_length = 0.1;
  SITPST updated_state = optimizer.RandomGradientUpdate(initial_state, gradient, step_length);

  EXPECT_NE(GetMaxAbs(updated_state), GetMaxAbs(initial_state));
}

// Test advanced stop functionality - Gradient convergence
TEST_F(OptimizerTest, AdvancedStopGradientConvergence) {
  // Set very low gradient tolerance to trigger gradient convergence
  test_params_ = OptimizerParams(OptimizerParams::BaseParams(1000, 1e-2, 1e-10, 20, {0.1, 0.05, 0.01}),
                                 StochasticGradient);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Create a mock evaluator that returns very small gradients after first iteration
  auto mock_evaluator =
      [this, iteration_count = 0](const SITPST &state) mutable -> std::tuple<TenElemT, SITPST, double> {
    TenElemT energy = 1.0;
    SITPST gradient = state;
    double error = 0.1;

    // Return very small gradient after first iteration to trigger convergence
    double gradient_scale = (iteration_count > 0) ? 1e-11 : 1.0;
    iteration_count++;

    for (size_t row = 0; row < state.rows(); ++row) {
      for (size_t col = 0; col < state.cols(); ++col) {
        for (size_t i = 0; i < state({row, col}).size(); ++i) {
          gradient({row, col})[i] = gradient_scale * state({row, col})[i];
        }
      }
    }

    return {energy, gradient, error};
  };

  auto result = optimizer.IterativeOptimize(test_tps_, mock_evaluator);

  EXPECT_TRUE(result.converged);
  // Should stop early due to gradient convergence
  EXPECT_LT(result.total_iterations, test_params_.base_params.step_lengths.size());
  EXPECT_GT(result.total_iterations, 0);
}

// Test advanced stop functionality - Energy convergence
TEST_F(OptimizerTest, AdvancedStopEnergyConvergence) {
  // Set very low energy tolerance to trigger energy convergence
  test_params_ = OptimizerParams(OptimizerParams::BaseParams(1000, 1e-10, 1e-6, 20, {0.1, 0.05, 0.01}),
                                 StochasticGradient);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Create a mock evaluator that returns very small energy changes after first iteration
  auto mock_evaluator =
      [this, iteration_count = 0](const SITPST &state) mutable -> std::tuple<TenElemT, SITPST, double> {
    TenElemT energy = 1.0 + (iteration_count > 0 ? 1e-11 : 0.1); // Very small change after first iteration
    SITPST gradient = state;
    double error = 0.1;
    iteration_count++;

    for (size_t row = 0; row < state.rows(); ++row) {
      for (size_t col = 0; col < state.cols(); ++col) {
        for (size_t i = 0; i < state({row, col}).size(); ++i) {
          gradient({row, col})[i] = state({row, col})[i];
        }
      }
    }

    return {energy, gradient, error};
  };

  auto result = optimizer.IterativeOptimize(test_tps_, mock_evaluator);

  EXPECT_TRUE(result.converged);
  // Should stop early due to energy convergence
  EXPECT_GT(result.total_iterations, 0);
}

// Test advanced stop functionality - Plateau detection
TEST_F(OptimizerTest, AdvancedStopPlateauDetection) {
  // Set very low plateau patience to trigger plateau detection
  test_params_ = OptimizerParams(OptimizerParams::BaseParams(1000, 1e-2, 1e-4, 2, {0.1, 0.05, 0.01}),
                                 StochasticGradient);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Create a mock evaluator that returns increasing energy (no improvement)
  auto mock_evaluator =
      [this, iteration_count = 0](const SITPST &state) mutable -> std::tuple<TenElemT, SITPST, double> {
    TenElemT energy = 1.0 + iteration_count * 0.1; // Always increasing energy
    SITPST gradient = state;
    double error = 0.1;
    iteration_count++;

    for (size_t row = 0; row < state.rows(); ++row) {
      for (size_t col = 0; col < state.cols(); ++col) {
        for (size_t i = 0; i < state({row, col}).size(); ++i) {
          gradient({row, col})[i] = state({row, col})[i];
        }
      }
    }

    return {energy, gradient, error};
  };

  auto result = optimizer.IterativeOptimize(test_tps_, mock_evaluator);

  EXPECT_TRUE(result.converged);
  // Should stop after plateau_patience iterations without improvement
  EXPECT_EQ(result.total_iterations, test_params_.base_params.plateau_patience + 1);
}

// Test advanced stop functionality - ShouldStop method directly
TEST_F(OptimizerTest, ShouldStopMethod) {
  test_params_ = OptimizerParams(
    OptimizerParams::BaseParams(1000, 1e-15, 1e-30, 20, {0.1}),
    StochasticGradient);
  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Test gradient convergence
  EXPECT_TRUE(optimizer.ShouldStop(1.0, 1.0, 1e-31, 0)); // gradient_norm < tolerance

  // Test energy convergence
  EXPECT_TRUE(optimizer.ShouldStop(1.0, 1.0 + 1e-16, 1.0, 0)); // |ΔE| < tolerance

  // Test plateau detection
  EXPECT_TRUE(optimizer.ShouldStop(1.0, 1.0, 1.0, 20)); // iterations_without_improvement >= patience

  // Test normal case (should not stop)
  EXPECT_FALSE(optimizer.ShouldStop(1.0, 0.9, 1.0, 0));
}

// Test actual gradient convergence with rapidly decreasing gradient
TEST_F(OptimizerTest, ActualGradientConvergence) {
  // Use standard tolerances for genuine convergence testing
  test_params_ = OptimizerParams(OptimizerParams::BaseParams(1000, 1e-3, 1e-4, 20, {0.1, 0.05, 0.01}),
                                 StochasticGradient);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Create a mock evaluator that rapidly decreases gradient to trigger convergence
  auto fast_converging_evaluator =
      [this, iteration_count = 0](const SITPST &state) mutable -> std::tuple<TenElemT, SITPST, double> {
    TenElemT energy = 1.0 - iteration_count * 0.01; // Decreasing energy
    SITPST gradient = state;
    double error = 0.1;

    // Gradient decreases rapidly: reaches by iteration 10
    double gradient_scale = std::max(1e-8, 1.0 - iteration_count * 0.1);
    gradient *= gradient_scale;

    iteration_count++;
    return {energy, gradient, error};
  };

  auto result = optimizer.IterativeOptimize(test_tps_, fast_converging_evaluator);

  EXPECT_TRUE(result.converged); // Should converge due to gradient tolerance
  EXPECT_LT(result.total_iterations, 20); // Should converge quickly
  EXPECT_GT(result.total_iterations, 8); // But not immediately
  EXPECT_FALSE(result.energy_trajectory.empty());
  EXPECT_FALSE(result.gradient_norms.empty());
}

// Test actual energy convergence with stable energy after initial improvement
TEST_F(OptimizerTest, ActualEnergyConvergence) {
  // Use strict energy tolerance for energy convergence testing
  test_params_ = OptimizerParams(OptimizerParams::BaseParams(1000, 1e-6, 1e-2, 20, {0.1, 0.05, 0.01}),
                                 StochasticGradient);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Create a mock evaluator that stabilizes energy after initial decrease
  auto energy_converging_evaluator =
      [this, iteration_count = 0](const SITPST &state) mutable -> std::tuple<TenElemT, SITPST, double> {
    TenElemT energy;
    if (iteration_count < 10) {
      energy = 1.0 - iteration_count * 0.1; // Initial energy improvement
    } else {
      energy = 0.0 + (iteration_count - 10) * 1e-8; // Very small changes (< 1e-6)
    }

    SITPST gradient = state;
    gradient *= 0.1; // Keep gradient above tolerance to test energy convergence specifically
    double error = 0.1;

    iteration_count++;
    return {energy, gradient, error};
  };

  auto result = optimizer.IterativeOptimize(test_tps_, energy_converging_evaluator);

  EXPECT_TRUE(result.converged); // Should converge due to energy tolerance
  EXPECT_LT(result.total_iterations, 50); // Should converge reasonably quickly
  EXPECT_GT(result.total_iterations, 10); // But after initial improvement phase
  EXPECT_FALSE(result.energy_trajectory.empty());
  EXPECT_FALSE(result.gradient_norms.empty());
}

// Test actual plateau detection with no improvement in energy
TEST_F(OptimizerTest, ActualPlateauDetection) {
  // Use small plateau patience for plateau detection testing
  test_params_ = OptimizerParams(OptimizerParams::BaseParams(1000, 1e-10, 1e-10, 5, {0.1, 0.05, 0.01}),
                                 StochasticGradient);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Create a mock evaluator that plateaus (no improvement) after initial decrease
  auto plateau_evaluator =
      [this, iteration_count = 0](const SITPST &state) mutable -> std::tuple<TenElemT, SITPST, double> {
    TenElemT energy;
    if (iteration_count < 3) {
      energy = 1.0 - iteration_count * 0.1; // Initial improvement
    } else {
      energy = 0.7 + (iteration_count % 2) * 0.01; // No net improvement, just fluctuation
    }

    SITPST gradient = state;
    gradient *= 0.1; // Keep gradient and energy changes above tolerance
    double error = 0.1;

    iteration_count++;
    return {energy, gradient, error};
  };

  auto result = optimizer.IterativeOptimize(test_tps_, plateau_evaluator);

  EXPECT_TRUE(result.converged); // Should converge due to plateau detection
  // Should stop after plateau_patience iterations without improvement (plus initial improvement phase)
  EXPECT_LE(result.total_iterations, test_params_.base_params.plateau_patience + 5);
  EXPECT_GT(result.total_iterations, test_params_.base_params.plateau_patience);
  EXPECT_FALSE(result.energy_trajectory.empty());
  EXPECT_FALSE(result.gradient_norms.empty());
}

// Test that optimization continues when no stopping criteria are met
TEST_F(OptimizerTest, NoEarlyStop) {
  // Set very strict tolerances and high patience to prevent early stopping
  test_params_ = OptimizerParams(OptimizerParams::BaseParams(1000, 1e-20, 1e-20, 1000, {0.1, 0.05, 0.01}),
                                 StochasticGradient);

  OptimizerT optimizer(test_params_, comm_, rank_, mpi_size_);

  // Create a mock evaluator that returns decreasing energy (improvement)
  auto mock_evaluator =
      [this, iteration_count = 0](const SITPST &state) mutable -> std::tuple<TenElemT, SITPST, double> {
    TenElemT energy = 1.0 - iteration_count * 0.01; // Always decreasing energy
    SITPST gradient = state;
    double error = 0.1;
    iteration_count++;

    for (size_t row = 0; row < state.rows(); ++row) {
      for (size_t col = 0; col < state.cols(); ++col) {
        for (size_t i = 0; i < state({row, col}).size(); ++i) {
          gradient({row, col})[i] = state({row, col})[i];
        }
      }
    }

    return {energy, gradient, error};
  };

  auto result = optimizer.IterativeOptimize(test_tps_, mock_evaluator);

  // With very strict tolerances, should NOT converge early - should run full iterations
  EXPECT_FALSE(result.converged);
  EXPECT_EQ(result.total_iterations, test_params_.base_params.max_iterations);
}
} // namespace qlpeps

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}
