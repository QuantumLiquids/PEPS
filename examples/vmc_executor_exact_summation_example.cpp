/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-18
*
* Description: QuantumLiquids/PEPS project.
* 
* Example demonstrating VMCPEPSOptimizer integration with exact summation energy evaluator.
* 
* This example shows the "full system integration" approach with VMCPEPSOptimizer,
* which includes Monte Carlo sampling infrastructure, state normalization, data collection,
* and file I/O operations - even when using exact summation for energy evaluation.
* 
* 🎯 USE CASE: When you need the complete VMC infrastructure (state saving, data collection,
*             acceptance rate monitoring, etc.) but want exact gradient computation for small systems.
* 
* 📊 COMPARISON WITH PURE OPTIMIZER TESTS:
* - VMCPEPSOptimizer: Full integration with all VMC features
* - Pure Optimizer tests: Focus only on optimization algorithm correctness
* 
* Design Philosophy: "Use the right tool for the job"
* - For algorithm verification: Use pure Optimizer tests (see tests/test_optimizer/test_optimizer_adagrad_exact_sum.cpp)
* - For system integration: Use VMCPEPSOptimizer (this example)
*/

#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/optimizer/optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/simple_update/square_lattice_nnn_simple_update.h"

using namespace qlten;
using namespace qlpeps;
using qlten::special_qn::fZ2QN;
using qlten::special_qn::TrivialRepQN;

/**
 * @brief Demonstrates VMCPEPSOptimizer with exact summation for 2x2 Heisenberg model
 * 
 * This example shows how to:
 * 1. Set up VMCPEPSOptimizer with full VMC infrastructure
 * 2. Replace Monte Carlo energy evaluator with exact summation
 * 3. Monitor convergence with detailed callbacks
 * 4. Save optimization trajectory and states
 */
void DemonstrateVMCExecutorExactSummation() {
    // System parameters: 2x2 Heisenberg model
    size_t Lx = 2, Ly = 2;
    double J = 1.0;
    auto energy_exact = -2.0; // 2x2 Heisenberg XY model exact energy
    
    // Physical indices for spin-1/2 model with trivial quantum numbers  
    using QNT = TrivialRepQN;
    using TenElemT = QLTEN_Double;
    using SITPST = SplitIndexTPS<TenElemT, QNT>;
    using Model = SquareHeisenberg;
    using MCUpdater = MCUpdateSquareNNExchange;
    
    using IndexT = Index<QNT>;
    using QNSctT = QNSector<QNT>;
    IndexT loc_phy = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::IN);
    
    // Initialize PEPS state
    SITPST split_index_tps(Ly, Lx);
    split_index_tps.Initial(loc_phy);
    
    // Generate all configurations for exact summation (2^4 = 16 configs for 2x2 system)
    std::vector<Configuration> all_configs;
    Configuration config(Lx, Ly);
    for (size_t i = 0; i < (1ULL << (Lx * Ly)); ++i) {
        for (size_t x = 0; x < Lx; ++x) {
            for (size_t y = 0; y < Ly; ++y) {
                config({x, y}) = (i >> (x * Ly + y)) & 1;
            }
        }
        all_configs.push_back(config);
    }
    
    // Truncation parameters
    auto trun_para = BMPSTruncatePara(8, 8, 1e-16, CompressMPSScheme::SVD_COMPRESS, 
                                      std::optional<double>(), std::optional<size_t>());
    
    // Create Heisenberg model
    Model heisenberg_model(J, J, 0); // Jx = Jy = J, Jz = 0 (XY model)
    
    // VMC optimization parameters with full feature set
    qlpeps::OptimizerParams::BaseParams base_params(100, 1e-15, 1e-30, 20, 1.0);
    qlpeps::AdaGradParams adagrad_params(1e-8, 0.0);
    qlpeps::OptimizerParams opt_params(base_params, adagrad_params);
    
    // Monte Carlo parameters (not used for sampling but required for VMCPEPSOptimizer)
    Configuration init_config(Ly, Lx);
    init_config.Random(std::vector<size_t>{2, 2}); // 2 up, 2 down
    qlpeps::MonteCarloParams mc_params(1, 0, 1, init_config, true);
    
    // PEPS parameters  
    qlpeps::PEPSParams peps_params(trun_para);
    
    // Combined VMC parameters
    qlpeps::VMCPEPSOptimizerParams optimize_para(opt_params, mc_params, peps_params);
    
    // 🎯 KEY FEATURE: VMCPEPSOptimizer with full infrastructure
    VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
        optimize_para, split_index_tps, MPI_COMM_WORLD, heisenberg_model);
    
    // 🔧 EXACT SUMMATION INTEGRATION: Replace Monte Carlo with exact computation (unified interface)
    auto exact_energy_evaluator = [&](const SITPST &state) -> std::tuple<TenElemT, SITPST, double> {
        auto [energy, gradient, error] = ExactSumEnergyEvaluator(
            state, all_configs, trun_para, heisenberg_model, Ly, Lx);
        
        // Check if we're in MPI environment and print only from master rank
        int rank = 0;
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized) {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        }
        
        if (rank == 0) {
            std::cout << "  [Exact] Energy: " << energy << ", Gradient norm: " << gradient.NormSquare() << std::endl;
        }
        return {energy, gradient, error};
    };
    
    executor.SetEnergyEvaluator(exact_energy_evaluator);
    
    // 📊 ADVANCED MONITORING: Detailed callback with convergence tracking
    typename VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model>::OptimizerT::OptimizationCallback callback;
    callback.on_iteration = [&](size_t iteration, double energy, double energy_error, double gradient_norm) {
        std::cout << "VMC-Executor Step " << iteration 
                  << ": E=" << std::fixed << std::setprecision(12) << energy
                  << " (exact=" << energy_exact << ")"
                  << " ||∇||=" << std::scientific << std::setprecision(3) << gradient_norm
                  << " error=" << energy_error << std::endl;
                  
        // Convergence check
        if (std::abs(energy - energy_exact) < 1e-10) {
            std::cout << "🎯 Converged to exact energy!" << std::endl;
        }
    };
    
    callback.on_completion = [&](const auto& result) {
        std::cout << "✅ Optimization completed!" << std::endl;
        std::cout << "   Final energy: " << result.final_energy << std::endl;
        std::cout << "   Total iterations: " << result.iteration_count << std::endl;
        std::cout << "   Convergence reason: " << 
            (result.converged ? "Gradient tolerance reached" : "Maximum iterations reached") << std::endl;
    };
    
    executor.SetOptimizationCallback(callback);
    
    // 🚀 EXECUTE FULL VMC INTEGRATION with exact summation
    std::cout << "🔧 Starting VMCPEPSOptimizer with exact summation..." << std::endl;
    std::cout << "   System: 2x2 Heisenberg XY model" << std::endl;
    std::cout << "   Exact energy: " << energy_exact << std::endl;
    std::cout << "   Features: State saving, data collection, acceptance monitoring" << std::endl;
    std::cout << "   Energy evaluation: Exact summation (deterministic)" << std::endl;
    std::cout << std::endl;
    
    executor.Execute();
    
    // 📈 RESULTS ANALYSIS 
    double final_energy = executor.GetMinEnergy();
    std::cout << std::endl;
    std::cout << "🎯 FINAL RESULTS:" << std::endl;
    std::cout << "   Final energy: " << std::fixed << std::setprecision(12) << final_energy << std::endl;
    std::cout << "   Exact energy:  " << std::fixed << std::setprecision(12) << energy_exact << std::endl;
    std::cout << "   Error:        " << std::scientific << std::setprecision(3) 
              << std::abs(final_energy - energy_exact) << std::endl;
    
    // Verify convergence
    if (std::abs(final_energy - energy_exact) < 1e-5) {
        std::cout << "✅ SUCCESS: Converged to exact energy within tolerance!" << std::endl;
    } else {
        std::cout << "❌ WARNING: Did not converge to exact energy!" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI for VMCPEPSOptimizer
    MPI_Init(&argc, &argv);
    
    std::cout << "===========================================" << std::endl;
    std::cout << "VMCPEPSOptimizer + Exact Summation" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << std::endl;
    
    try {
        DemonstrateVMCExecutorExactSummation();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    MPI_Finalize();
    return 0;
}
