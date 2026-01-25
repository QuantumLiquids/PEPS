// SPDX-License-Identifier: LGPL-3.0-only
/*
* Migration Example: From VMCPEPSExecutor to VMCPEPSOptimizer
* 
* This file demonstrates the complete migration process with working code examples.
*/

#include <mpi.h>
#include <iostream>

// OLD: Legacy includes
// #include "qlpeps/algorithm/vmc_update/vmc_peps.h"
// #include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"

// NEW: Modern includes
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/optimizer/optimizer_params.h"

// Common includes
#include "qlpeps/algorithm/vmc_update/model_solvers/build_in_model_solvers_all.h"
#include "qlpeps/vmc_basic/configuration_update_strategies/monte_carlo_sweep_updater_all.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/consts.h"
#include "qlten/qlten.h"

using namespace qlpeps;
using namespace qlten;

// Type aliases for clarity
using QNT = qlten::special_qn::TrivialRepQN;
using TenElemT = TEN_ELEM_TYPE;
using IndexT = Index<QNT>;
using QNSctT = QNSector<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;
using TPST = TPS<TenElemT, QNT>;
using SITPST = SplitIndexTPS<TenElemT, QNT>;

// ============================================================================
// OLD CODE: Using VMCPEPSExecutor (Legacy)
// ============================================================================

void demonstrate_legacy_problems() {
    std::cout << "=== WHY LEGACY APPROACH WAS PROBLEMATIC ===" << std::endl;
    
    std::cout << "❌ Legacy VMCPEPSExecutor problems:" << std::endl;
    std::cout << "  - Monolithic parameter structure (VMCOptimizePara)" << std::endl;
    std::cout << "  - Mixed concerns in single executor class" << std::endl;
    std::cout << "  - Hardcoded optimization algorithms" << std::endl;
    std::cout << "  - Difficult to extend with new optimizers" << std::endl;
    std::cout << "  - Poor separation between MC, PEPS, and optimization logic" << std::endl;
    std::cout << "  - Legacy enum-based algorithm selection" << std::endl;
    std::cout << "" << std::endl;
    
    std::cout << "✅ Modern VMCPEPSOptimizer solutions:" << std::endl;
    std::cout << "  - Clean separation of parameter types" << std::endl;
    std::cout << "  - Modular Optimizer class design" << std::endl;
    std::cout << "  - Type-safe algorithm selection with std::variant" << std::endl;
    std::cout << "  - Easy to add new optimization algorithms" << std::endl;
    std::cout << "  - Clear responsibility boundaries" << std::endl;
    std::cout << "  - Factory methods and builder patterns" << std::endl;
}

// ============================================================================
// NEW CODE: Using VMCPEPSOptimizer (Modern)
// ============================================================================

void new_modern_approach() {
    std::cout << "\n=== NEW MODERN APPROACH ===" << std::endl;
    
    // NEW: Separate parameter structures for better modularity
    
    // 1. Monte Carlo parameters
    MonteCarloParams mc_params(
        1000,  // num_samples
        100,   // num_warmup_sweeps
        10,    // sweeps_between_samples
        "config_path",  // config_path
        Configuration(4, 4)  // alternative_init_config
    );
    
    // 2. PEPS parameters
    PEPSParams peps_params(
        BMPSTruncateParams<qlten::QLTEN_Double>(8, 1e-12, 1000),  // truncation parameters
        "wavefunction_data"  // wavefunction_path
    );
    
    // 3. Optimizer parameters
    OptimizerParams opt_params;
    opt_params.core_params.step_lengths = {0.01, 0.01, 0.01};
    opt_params.update_scheme = StochasticReconfiguration;
    opt_params.cg_params = ConjugateGradientParams(1e-6, 1000, 1e-8);
    
    // 4. Combine all parameters
    VMCPEPSOptimizerParams params(opt_params, mc_params, peps_params);
    
    // NEW: Create executor with modern parameters
    using Model = SquareSpinOneHalfXXZModelOBC;
    using MCUpdater = MCUpdateSquareNNExchange;
    Model model;
    
    // Note: This would be the new executor (commented out for demonstration)
    // VMCPEPSOptimizer<TenElemT, QNT, MCUpdater, Model> executor(
    //     params, 4, 4, MPI_COMM_WORLD, model);
    
    std::cout << "Modern parameter structure created successfully" << std::endl;
    std::cout << "  - MC samples: " << mc_params.num_samples << std::endl;
    std::cout << "  - Step lengths: " << opt_params.core_params.step_lengths.size() << " steps" << std::endl;
    std::cout << "  - Update scheme: " << opt_params.update_scheme << std::endl;
}

// ============================================================================
// MIGRATION HELPER FUNCTIONS
// ============================================================================

// Modern parameter creation with factory methods
VMCPEPSOptimizerParams create_modern_parameters() {
    std::cout << "\n=== CREATING MODERN PARAMETERS WITH FACTORY METHODS ===" << std::endl;
    
    // Create Monte Carlo parameters
    MonteCarloParams mc_params(
        1000,  // num_samples
        100,   // num_warmup_sweeps
        10,    // sweeps_between_samples
        {1, 1, 1, 1}  // occupancy configuration
    );
    
    // Create PEPS parameters
    BMPSTruncateParams<qlten::QLTEN_Double> truncate_para(8, 1e-12, 1000);
    PEPSParams peps_params(truncate_para, 4, 4, "wavefunction_data");
    
    // Create optimizer parameters using factory method
    ConjugateGradientParams cg_params(1000, 1e-6, 20, 1e-8);
    OptimizerParams opt_params = OptimizerParams::CreateStochasticReconfiguration(
        100, cg_params, 0.01);
    
    VMCPEPSOptimizerParams modern_params(opt_params, mc_params, peps_params);
    
    std::cout << "✅ Successfully created modern parameter structure" << std::endl;
    return modern_params;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== VMC PEPS Migration Tutorial ===" << std::endl;
        std::cout << "MPI Size: " << size << std::endl;
        std::cout << "=====================================" << std::endl;
        
        // Explain why legacy approach was problematic
        demonstrate_legacy_problems();
        
        // Demonstrate modern approach
        new_modern_approach();
        
        // Demonstrate modern parameter creation with factory methods
        VMCPEPSOptimizerParams modern_params = create_modern_parameters();
        
        std::cout << "\n=== MIGRATION COMPLETE ===" << std::endl;
        std::cout << "You can now use VMCPEPSOptimizer with the converted parameters!" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
