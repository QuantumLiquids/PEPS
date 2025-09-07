// SPDX-License-Identifier: LGPL-3.0-only

/**
 * @file qlpeps.h
 * @brief Main header file for the PEPS (Projected Entangled Pair States) library
 * @author Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * @date 2024-09-25
 * 
 * @mainpage PEPS Library Documentation
 * 
 * @section intro_sec Introduction
 * 
 * The PEPS library is a high-performance C++ implementation for simulating 2D strongly 
 * correlated electron systems using tensor network methods. It provides efficient 
 * implementations of finite-size PEPS algorithms, variational Monte Carlo optimization, 
 * and quantum many-body calculations.
 * 
 * @section features_sec Key Features
 * 
 * - **Finite-Size PEPS**: Efficient implementation for finite 2D quantum systems
 * - **Multiple Update Algorithms**: Simple update, loop update, and variational methods
 * - **Variational Monte Carlo**: VMC-based optimization by AdaGrad, Adam or stochastic reconfiguration
 * - **Model-Specific Solvers**: Built-in implementations for common quantum models, and can expand to your own model.
 * - **MPI Parallelization**: Distributed computing support
 * - **Header-Only Design**: Easy integration into existing projects
 * 
 * @section usage_sec Usage
 * 
 * Include this header to access all PEPS functionality:
 * @code
 * #include "qlpeps/qlpeps.h"
 * @endcode
 * 
 * @section structure_sec Library Structure
 * 
 * The library is organized into several main components:
 * 
 * - **@ref algorithm**: PEPS update algorithms and optimization methods
 * - **@ref two_dim_tn**: 2D tensor network framework and PEPS implementations
 * - **@ref vmc_basic**: Variational Monte Carlo infrastructure
 * - **@ref optimizer**: Optimization algorithms 
 * - **@ref utility**: Helper functions and utilities
 * 
 * @section examples_sec Examples
 * 
 * See the tutorial examples and test cases for usage examples:
 * - Tutorial examples in the `tutorial/examples/` directory
 * - Test cases in the `tests/` directory
 * 
 * @section dependencies_sec Dependencies
 * 
 * - **C++20** compiler
 * - **MPI** for parallel computing
 * - **TensorToolkit** for tensor operations
 * - **CMake** for building
 * 
 * @section license_sec License
 * 
 * This library is licensed under LGPL-3.0-only.
 * 
 * @section contact_sec Contact
 * 
 * For questions and contributions, please contact:
 * - Hao-Xin Wang: wanghaoxin1996@gmail.com
 * - GitHub: https://github.com/QuantumLiquids/PEPS
 */

#ifndef QLPEPS_QLPEPS_H
#define QLPEPS_QLPEPS_H

/**
 * @brief Main namespace for the PEPS library
 * 
 * All PEPS library components are contained within this namespace.
 */
namespace qlpeps {
    // Main library components are included below
}

#include "qlpeps/algorithm/algorithm_all.h"
#include "qlpeps/api/vmc_api.h"
#include "qlpeps/api/conversions.h"

#endif //QLPEPS_QLPEPS_H
