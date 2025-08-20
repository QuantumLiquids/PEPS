/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-10-23
*
* Description: QuantumLiquids/PEPS project. Exact summation energy evaluator.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_ENERGY_EVALUATOR_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_ENERGY_EVALUATOR_H

#include "qlten/qlten.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer.h"
#include <mpi.h>
#include <numeric>  // for std::accumulate

namespace qlpeps {
using namespace qlten;

/**
 * @brief Helper function to evaluate local psi partial psi dagger for fermion systems
 * 
 * This function is used in fermion gradient calculations to handle the complex
 * tensor contractions required for fermion parity operators.
 */
template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> EvaluateLocalPsiPartialPsiDag(
  const QLTensor<TenElemT, QNT> &hole_dag,
  const QLTensor<TenElemT, QNT> &local_psi_ten
) {
  const QLTensor<TenElemT, QNT> hole = Dag(hole_dag);
  QLTensor<TenElemT, QNT> psi_scale_ten, psi_partial_psi_dag;
  Contract(&hole, {1, 2, 3, 4}, &local_psi_ten, {0, 1, 2, 3}, &psi_scale_ten);
  Contract(&hole_dag, {0}, &psi_scale_ten, {0}, &psi_partial_psi_dag);
  return psi_partial_psi_dag;
}

/**
 * @brief Helper function to convert a vector to a Configuration object
 *
 * @param config_vec Vector representation of the configuration
 * @param Lx Number of columns in the lattice
 * @param Ly Number of rows in the lattice
 * @return Configuration object
 */
Configuration Vec2Config(const std::vector<size_t> &config_vec,
                         size_t Lx,
                         size_t Ly) {
  Configuration config(Ly, Lx);
  for (size_t i = 0; i < config_vec.size(); i++) {
    const size_t row = i / Lx;
    const size_t col = i % Lx;
    config({row, col}) = config_vec[i];
  }
  return config;
}

/**
 * @brief Helper function to generate all possible configurations for a given system
 *
 * This function generates all possible configurations for a system with given
 * particle numbers/spin. Equivalent to ExchangeMCUpdater.
 * So It should not work for Transverse-field Ising model and Hubbard model.
 * It's useful for exact summation when the
 * Hilbert space is small enough to enumerate.
 *
 * @param particle_counts Vector of particle counts for each configuration type
 * @param Lx Number of columns in the lattice
 * @param Ly Number of rows in the lattice
 * @return Vector of all possible configurations
 */
std::vector<Configuration> GenerateAllPermutationConfigs(
  const std::vector<size_t> &particle_counts,
  size_t Lx,
  size_t Ly
) {
  std::vector<Configuration> all_configs;

  // Create the base configuration vector
  std::vector<size_t> config_vec;
  for (size_t i = 0; i < particle_counts.size(); ++i) {
    for (size_t j = 0; j < particle_counts[i]; ++j) {
      config_vec.push_back(i);
    }
  }

  // Generate all permutations
  do {
    all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
  } while (std::next_permutation(config_vec.begin(), config_vec.end()));

  return all_configs;
}

// TODO: MPI configuration distribution removed - caused memory leaks in single-process mode
// DistributeConfigurations function deleted due to untested MPI code that crashed during testing
// When MPI parallel evaluation is needed, rewrite with proper memory management and testing

// TODO: ExactSumEnergyEvaluatorMPI function removed - caused 6-7GB memory leaks
// 
// MEMORY LEAK INVESTIGATION RESULTS:
// - This 164-line MPI function caused severe memory explosions in single-process environments
// - Root cause: Forcing MPI operations in single-process mode creates unnecessary overhead
// - Function included complex MPI_Bcast, MPI_Reduce, MPI_Send/Recv operations without proper testing
//
// REQUIRED FUTURE WORK:
// 1. Design proper single-process vs multi-process API selection
// 2. Implement memory-safe MPI gradient reduction 
// 3. Add comprehensive unit tests for MPI communication patterns
// 4. Consider using MPI collective operations instead of manual Send/Recv loops
//
// The function signature was:
// template<typename ModelEnergySolver, typename TenElemT, typename QNT>
// std::tuple<TenElemT, SplitIndexTPS<TenElemT, QNT>, double> ExactSumEnergyEvaluatorMPI(...)
//
// DO NOT REIMPLEMENT without proper memory leak testing and single-process compatibility
} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_ENERGY_EVALUATOR_H
