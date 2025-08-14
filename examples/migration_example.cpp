// SPDX-License-Identifier: LGPL-3.0-only
/*
* Migration Example: From VMCPEPSExecutor to VMCPEPSOptimizerExecutor
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

void old_legacy_approach() {
    std::cout << "=== OLD LEGACY APPROACH ===" << std::endl;
    
    // OLD: Single parameter structure
    VMCOptimizePara optimize_para(
        BMPSTruncatePara(8, 1e-12, 1000),  // truncation parameters
        1000,  // num_samples
        100,   // num_warmup_sweeps
        10,    // sweeps_between_samples
        {1, 1, 1, 1},  // occupancy
        4, 4,  // ly, lx
        {0.01, 0.01, 0.01},  // step_lengths
        StochasticReconfiguration,  // update_scheme
        ConjugateGradientParams(1e-6, 1000, 1e-8),  // cg_params
        "wavefunction_data"  // wavefunction_path
    );

    // OLD: Create executor with legacy parameters
    using Model = SquareSpinOneHalfXXZModel;
    using MCUpdater = MCUpdateSquareNNExchange;
    Model model;
    
    // Note: This would be the old executor (commented out for migration)
    // VMCPEPSExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    //     optimize_para, 4, 4, MPI_COMM_WORLD, model);
    
    std::cout << "Legacy parameter structure created successfully" << std::endl;
    std::cout << "  - MC samples: " << optimize_para.mc_samples << std::endl;
    std::cout << "  - Step lengths: " << optimize_para.step_lens.size() << " steps" << std::endl;
    std::cout << "  - Update scheme: " << optimize_para.update_scheme << std::endl;
}

// ============================================================================
// NEW CODE: Using VMCPEPSOptimizerExecutor (Modern)
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
        BMPSTruncatePara(8, 1e-12, 1000),  // truncation parameters
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
    using Model = SquareSpinOneHalfXXZModel;
    using MCUpdater = MCUpdateSquareNNExchange;
    Model model;
    
    // Note: This would be the new executor (commented out for demonstration)
    // VMCPEPSOptimizerExecutor<TenElemT, QNT, MCUpdater, Model> executor(
    //     params, 4, 4, MPI_COMM_WORLD, model);
    
    std::cout << "Modern parameter structure created successfully" << std::endl;
    std::cout << "  - MC samples: " << mc_params.num_samples << std::endl;
    std::cout << "  - Step lengths: " << opt_params.core_params.step_lengths.size() << " steps" << std::endl;
    std::cout << "  - Update scheme: " << opt_params.update_scheme << std::endl;
}

// ============================================================================
// MIGRATION HELPER FUNCTIONS
// ============================================================================

// Helper function to convert legacy parameters to new structure
VMCPEPSOptimizerParams convert_legacy_to_modern(const VMCOptimizePara& legacy_params) {
    std::cout << "\n=== CONVERTING LEGACY TO MODERN ===" << std::endl;
    
    // Convert Monte Carlo parameters
    MonteCarloParams mc_params(
        legacy_params.mc_samples,
        legacy_params.mc_warm_up_sweeps,
        legacy_params.mc_sweeps_between_sample,
        "config_path",  // You may need to set this appropriately
        legacy_params.init_config  // Use the legacy configuration
    );
    
    // Convert PEPS parameters
    PEPSParams peps_params(
        legacy_params.bmps_trunc_para,
        legacy_params.wavefunction_path
    );
    
    // Convert optimizer parameters
    OptimizerParams opt_params;
    opt_params.core_params.step_lengths = legacy_params.step_lens;
    opt_params.update_scheme = legacy_params.update_scheme;
    opt_params.cg_params = legacy_params.cg_params;
    
    // Create combined parameters
    VMCPEPSOptimizerParams modern_params(opt_params, mc_params, peps_params);
    
    std::cout << "Successfully converted legacy parameters to modern structure" << std::endl;
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
        
        // Demonstrate old approach
        old_legacy_approach();
        
        // Demonstrate new approach
        new_modern_approach();
        
        // Demonstrate conversion
        VMCOptimizePara legacy_params(
            BMPSTruncatePara(8, 1e-12, 1000),
            1000, 100, 10,
            {1, 1, 1, 1}, 4, 4,
            {0.01, 0.01, 0.01},
            StochasticReconfiguration,
            ConjugateGradientParams(1e-6, 1000, 1e-8),
            "wavefunction_data"
        );
        
        VMCPEPSOptimizerParams converted_params = convert_legacy_to_modern(legacy_params);
        
        std::cout << "\n=== MIGRATION COMPLETE ===" << std::endl;
        std::cout << "You can now use VMCPEPSOptimizerExecutor with the converted parameters!" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
