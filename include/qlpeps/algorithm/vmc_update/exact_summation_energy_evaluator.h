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

/**
 * @brief Distribute configurations among MPI ranks for parallel computation
 *
 * Uses round-robin distribution to ensure balanced workload and deterministic results.
 * Each rank gets approximately all_configs.size() / mpi_size configurations.
 * This maintains identical behavior regardless of number of MPI ranks.
 *
 * @param all_configs Vector of all configurations to distribute
 * @param rank Current MPI rank
 * @param mpi_size Total number of MPI ranks
 * @return Vector of configurations assigned to this rank
 */
std::vector<Configuration> DistributeConfigurations(
  const std::vector<Configuration> &all_configs,
  int rank,
  int mpi_size
) {
  std::vector<Configuration> local_configs;

  // Round-robin distribution: rank i gets configurations i, i+mpi_size, i+2*mpi_size, ...
  for (size_t i = rank; i < all_configs.size(); i += mpi_size) {
    local_configs.push_back(all_configs[i]);
  }

  return local_configs;
}

/**
 * @brief MPI-aware exact summation energy evaluator
 *
 * @warning MEMORY SAFETY: This function creates multiple copies of large SplitIndexTPS objects
 * during MPI communication. Total memory usage scales as O(mpi_size × state_size).
 * Monitor memory usage carefully for large systems or high MPI rank counts.
 *
 * @warning DEADLOCK PROTECTION: Fixed version eliminates previous MPI deadlock risks
 * from excessive barriers and manual Send/Recv loops.
 *
 * Distributes exact configuration summation across MPI ranks while maintaining
 * identical MPI communication patterns to Monte Carlo evaluators for drop-in compatibility.
 *
 * Algorithm flow:
 * 1. INPUT: state valid ONLY on master rank (from Optimizer)
 * 2. BROADCAST: state to all ranks for parallel computation
 * 3. DISTRIBUTE: configurations round-robin among ranks
 * 4. COMPUTE: each rank processes assigned configurations independently
 * 5. REDUCE: gather partial energy/gradient sums to master rank
 * 6. BROADCAST: final energy to all ranks for convergence checks
 *
 * This eliminates Monte Carlo noise while maintaining identical MPI communication
 * patterns to standard energy evaluators for drop-in compatibility.
 *
 * @tparam ModelEnergySolver Model type with CalEnergyAndHoles method
 * @tparam TenElemT Tensor element type (double or complex)
 * @tparam QNT Quantum number type
 * @param split_index_tps PEPS state (valid ONLY on master rank from Optimizer)
 * @param all_configs All configurations to sum over
 * @param trun_para BMPS truncation parameters
 * @param model Physical model
 * @param Ly Lattice height
 * @param Lx Lattice width
 * @param comm MPI communicator
 * @param rank Current MPI rank
 * @param mpi_size Total MPI ranks
 * @return (energy, gradient, error) where:
 *         - energy: valid on ALL ranks (broadcast for convergence checks)
 *         - gradient: valid ONLY on master rank (gathered for optimization)
 *         - error: always 0.0, valid ONLY on master rank
 */
template<typename ModelEnergySolver, typename TenElemT, typename QNT>
std::tuple<TenElemT, SplitIndexTPS<TenElemT, QNT>, double> ExactSumEnergyEvaluatorMPI(
  const SplitIndexTPS<TenElemT, QNT> &split_index_tps,
  const std::vector<Configuration> &all_configs,
  const BMPSTruncatePara &trun_para,
  ModelEnergySolver &model,
  size_t Ly,
  size_t Lx,
  MPI_Comm comm,
  int rank,
  int mpi_size
) {
  using WaveFunctionType = SplitIndexTPS<TenElemT, QNT>;

  // MPI state distribution: Energy evaluator's core responsibility
  // INPUT contract: state valid ONLY on master rank (from Optimizer)
  WaveFunctionType local_state;

  if (rank == kMPIMasterRank) {
    local_state = split_index_tps;
  } else {
    // Non-master ranks: empty state, will be overwritten by broadcast
    local_state = WaveFunctionType(Ly, Lx);
  }

  // Broadcast state to all ranks for parallel computation (only needed in multi-process environment)
  if (mpi_size > 1) {
    MPI_Bcast(local_state, comm, kMPIMasterRank);
  }

  // Distribute configurations among ranks
  std::vector<Configuration> local_configs = DistributeConfigurations(all_configs, rank, mpi_size);

  // Initialize local computation variables
  std::vector<double> local_weights;
  std::vector<TenElemT> local_e_loc_set;
  WaveFunctionType local_g_weighted_sum(Ly, Lx, local_state.PhysicalDim());
  WaveFunctionType local_g_times_e_weighted_sum(Ly, Lx, local_state.PhysicalDim());

  // Process assigned configurations
  for (auto &config : local_configs) {
    TPSWaveFunctionComponent<TenElemT, QNT> tps_sample(local_state, config, trun_para);
    local_weights.push_back(std::norm(tps_sample.amplitude));

    TensorNetwork2D<TenElemT, QNT> holes_dag(Ly, Lx);
    TenElemT e_loc = model.template CalEnergyAndHoles<TenElemT, QNT, true>(
      &local_state,
      &tps_sample,
      holes_dag);
    local_e_loc_set.push_back(e_loc);

    WaveFunctionType gradient_sample(Ly, Lx, local_state.PhysicalDim());
    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        size_t basis = tps_sample.config({row, col});

        if constexpr (Index<QNT>::IsFermionic()) {
          auto psi_partial_psi_dag = EvaluateLocalPsiPartialPsiDag(holes_dag({row, col}), tps_sample.tn({row, col}));
          gradient_sample({row, col})[basis] = psi_partial_psi_dag;
        } else {
          gradient_sample({row, col})[basis] = tps_sample.amplitude * holes_dag({row, col});
        }
      }
    }
    local_g_weighted_sum += gradient_sample;
    local_g_times_e_weighted_sum += e_loc * gradient_sample;
  }

  // Compute local weighted sums
  double local_weight_sum = 0.0;
  TenElemT local_e_loc_sum = TenElemT(0.0);

  for (size_t j = 0; j < local_e_loc_set.size(); j++) {
    local_e_loc_sum += local_e_loc_set[j] * local_weights[j];
    local_weight_sum += local_weights[j];
  }

  // Reduce scalar values to master rank
  double global_weight_sum;
  TenElemT global_e_loc_sum;

  if (mpi_size > 1) {
    MPI_Barrier(comm);
    MPI_Reduce(&local_weight_sum, &global_weight_sum, 1, MPI_DOUBLE, MPI_SUM, kMPIMasterRank, comm);

    if constexpr (std::is_same_v<TenElemT, double>) {
      MPI_Reduce(&local_e_loc_sum, &global_e_loc_sum, 1, MPI_DOUBLE, MPI_SUM, kMPIMasterRank, comm);
    } else {
      MPI_Reduce(&local_e_loc_sum, &global_e_loc_sum, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, kMPIMasterRank, comm);
    }
  } else {
    // Single process: no reduction needed
    global_weight_sum = local_weight_sum;
    global_e_loc_sum = local_e_loc_sum;
  }

  // Reduce gradient tensors to master rank using collective operations
  WaveFunctionType global_g_weighted_sum(Ly, Lx, local_state.PhysicalDim());
  WaveFunctionType global_g_times_e_weighted_sum(Ly, Lx, local_state.PhysicalDim());

  if (mpi_size > 1) {
    if (rank == kMPIMasterRank) {
      global_g_weighted_sum = local_g_weighted_sum;
      global_g_times_e_weighted_sum = local_g_times_e_weighted_sum;

      for (int source_rank = 1; source_rank < mpi_size; source_rank++) {
        WaveFunctionType remote_g_weighted_sum(Ly, Lx, local_state.PhysicalDim());
        WaveFunctionType remote_g_times_e_weighted_sum(Ly, Lx, local_state.PhysicalDim());

        // Blocking receive with unique tags to prevent deadlock
        MPI_Status status1, status2;
        MPI_Recv(remote_g_weighted_sum, source_rank, comm, 2 * source_rank);
        MPI_Recv(remote_g_times_e_weighted_sum, source_rank, comm, 2 * source_rank + 1);

        // Accumulate results
        global_g_weighted_sum += remote_g_weighted_sum;
        global_g_times_e_weighted_sum += remote_g_times_e_weighted_sum;
      }
    } else {
      // Non-master ranks: send gradients and exit communication phase
      MPI_Send(local_g_weighted_sum, kMPIMasterRank, comm, 2 * rank);
      MPI_Send(local_g_times_e_weighted_sum, kMPIMasterRank, comm, 2 * rank + 1);
    }
  } else {
    // Single process: no communication needed
    global_g_weighted_sum = local_g_weighted_sum;
    global_g_times_e_weighted_sum = local_g_times_e_weighted_sum;
  }

  // Ensure all gradient communication is complete (only needed in multi-process environment)
  if (mpi_size > 1) {
    MPI_Barrier(comm);
  }

  // Compute final results on master rank
  TenElemT energy = TenElemT(0.0);
  WaveFunctionType gradient(Ly, Lx, local_state.PhysicalDim());

  if (rank == kMPIMasterRank) {
    energy = global_e_loc_sum / global_weight_sum;
    gradient = (global_g_times_e_weighted_sum - energy * global_g_weighted_sum) * (1.0 / global_weight_sum);

    if constexpr (Index<QNT>::IsFermionic()) {
      gradient.ActFermionPOps();
    }
  }

  // OUTPUT contract: Energy must be available on ALL ranks for Optimizer convergence checks (only needed in multi-process environment)
  if (mpi_size > 1) {
    if constexpr (std::is_same_v<TenElemT, double>) {
      MPI_Bcast(&energy, 1, MPI_DOUBLE, kMPIMasterRank, comm);
    } else {
      MPI_Bcast(&energy, 1, MPI_DOUBLE_COMPLEX, kMPIMasterRank, comm);
    }
  }

  // Final MPI distribution:
  // - energy: valid on ALL ranks (broadcast)
  // - gradient: valid ONLY on master rank (gathered)
  // - error: 0.0 for exact computation, valid ONLY on master rank
  double error = (rank == kMPIMasterRank) ? 0.0 : 0.0;

  return {energy, gradient, error};
}
} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_ENERGY_EVALUATOR_H
