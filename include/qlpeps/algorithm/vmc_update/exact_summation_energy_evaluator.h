/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2025-08-10
*
* Description: QuantumLiquids/PEPS project. Exact summation energy evaluator.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_ENERGY_EVALUATOR_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_ENERGY_EVALUATOR_H

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/vmc_basic/wave_function_component.h"
#include "qlpeps/two_dim_tn/tensor_network_2d/tensor_network_2d.h"
#include "qlpeps/ond_dim_tn/boundary_mps/bmps.h"
#include "qlpeps/utility/helpers.h" // ComplexConjugate
#include <mpi.h>
#include <numeric>  // for std::accumulate

namespace qlpeps {

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
 * @brief Single-process exact summation energy evaluator (RESTORED after MPI cleanup)
 *
 * Mathematical convention (consistent with the MC-based evaluator):
 * \f[
 *   \Psi(S;\,\theta),\quad w(S) = \frac{|\Psi(S)|^2}{Z},\quad Z = \sum_S |\Psi(S)|^2.
 * \f]
 * Local energy
 * \f[
 *   E_{\mathrm{loc}}(S) = \sum_{S'} \frac{\Psi^*(S')}{\Psi^*(S)}\, \langle S'|H|S \rangle.
 * \f]
 * Log-derivative operator
 * \f[
 *   O_i^*(S) = \frac{\partial \ln \Psi^*(S)}{\partial \theta_i^*}.
 * \f]
 * Energy and gradient (treating \(\theta\) and \(\theta^*\) as independent):
 * \f[
 *   E = \langle E_{\mathrm{loc}} \rangle_{w},\quad
 *   \frac{\partial E}{\partial \theta_i^*} = \langle E_{\mathrm{loc}}^* O_i^* \rangle_{w}
 *   - \langle E_{\mathrm{loc}} \rangle_{w}^*\, \langle O_i^* \rangle_{w}.
 * \f]
 * Implementation mapping (using raw weights \(w_{\text{raw}}(S)=|\Psi(S)|^2\)):
 * \f[
 *   S_{O} = \sum_S w_{\text{raw}}(S)\, O^*(S),\quad
 *   S_{EO} = \sum_S w_{\text{raw}}(S)\, E_{\mathrm{loc}}^*(S)\, O^*(S),
 * \f]
 * \f[
 *   E = \frac{\sum_S w_{\text{raw}}(S)\, E_{\mathrm{loc}}(S)}{\sum_S w_{\text{raw}}(S)},\quad
 *   \nabla_{\theta^*} E = \frac{S_{EO} - E^*\, S_{O}}{\sum_S w_{\text{raw}}(S)}.
 * \f]
 * Boson vs Fermion:
 * - Boson: \(O^*(S)\) is formed from the amplitude and \(\text{hole\_dag}\) tensors (consistent with \(\Psi^*\)).
 * - Fermion: \(O^*(S)\) is built via EvaluateLocalPsiPartialPsiDag, and \c gradient.ActFermionPOps() is applied once
 *   at the end to enforce parity.
 *
 * Notes:
 * - Mirrors VMCPEPSOptimizerExecutor::SampleEnergyAndHoles_: use \(E_{\mathrm{loc}}^*\) when multiplying gradient
 *   samples and use \(E^*\) in the covariance subtraction.
 * - For real types, \c ComplexConjugate is a no-op.
 * - Symbol ↔ Code mapping in this function:
 *   - \(S_O\)  ↔  Ostar_weighted_sum
 *   - \(S_{EO}\)  ↔  ELocConj_Ostar_weighted_sum
 *
 * @tparam ModelT The model type (must have CalEnergyAndHoles method)
 * @tparam TenElemT Tensor element type (double or complex)
 * @tparam QNT Quantum number type (fZ2QN-like for fermions, TrivialRepQN/U1 variants for bosons)
 * @param split_index_tps The current PEPS state
 * @param all_configs All possible configurations to sum over
 * @param trun_para BMPS truncation parameters
 * @param model The physical model
 * @param Ly Number of rows in the lattice
 * @param Lx Number of columns in the lattice
 * @return Tuple of (energy, gradient, error) where error is always 0 for exact summation
 */
template<typename ModelT, typename TenElemT, typename QNT>
std::tuple<TenElemT, SplitIndexTPS<TenElemT, QNT>, double> ExactSumEnergyEvaluator(
    const SplitIndexTPS<TenElemT, QNT> &split_index_tps,
    const std::vector<Configuration> &all_configs,
    const BMPSTruncatePara &trun_para,
    ModelT &model,
    size_t Ly, size_t Lx
) {
  using SplitIndexTPSType = SplitIndexTPS<TenElemT, QNT>;

  std::vector<double> weights;
  std::vector<TenElemT> e_loc_set;
  // Accumulators: S_O and S_{EO} (see Doxygen above)
  SplitIndexTPSType Ostar_weighted_sum(Ly, Lx, split_index_tps.PhysicalDim());            // S_O
  SplitIndexTPSType ELocConj_Ostar_weighted_sum(Ly, Lx, split_index_tps.PhysicalDim());   // S_{EO}

  // Exact summation over all configurations
  for (auto &config : all_configs) {
    TPSWaveFunctionComponent<TenElemT, QNT>
        tps_sample(split_index_tps, config, trun_para);
    weights.push_back(std::norm(tps_sample.amplitude));
    TensorNetwork2D<TenElemT, QNT> holes_dag(Ly, Lx);// \partial_{theta^*} \Psi^*
    TenElemT e_loc =
        model.template CalEnergyAndHoles<TenElemT, QNT, true>(
            &split_index_tps, &tps_sample, holes_dag);
    e_loc_set.push_back(e_loc);

    // Per-configuration increment to S_O (see Doxygen mapping): O^*(S) weighted by |Ψ(S)|^2
    SplitIndexTPSType Ostar_weighted_increment(Ly, Lx, split_index_tps.PhysicalDim());
    for (size_t row = 0; row < Ly; row++) {
      for (size_t col = 0; col < Lx; col++) {
        size_t basis = tps_sample.config({row, col});

        // Handle gradient calculation based on particle type
        if constexpr (Index<QNT>::IsFermionic()) {
          // Fermion system: Use complex tensor contractions
          auto psi_partial_psi_dag = EvaluateLocalPsiPartialPsiDag(holes_dag({row, col}), tps_sample.tn({row, col}));
          Ostar_weighted_increment({row, col})[basis] = psi_partial_psi_dag;
        } else {
          // Boson system: |Psi|^2 * \Delta, where \Delta = \partial_{\theta^*} ln(\Psi^*)
          Ostar_weighted_increment({row, col})[basis] = tps_sample.amplitude * holes_dag({row, col});
        }
      }
    }
    // S_O += w_raw(S) · O*(S)
    Ostar_weighted_sum += Ostar_weighted_increment;
    // S_{EO} += w_raw(S) · E_loc*(S) · O*(S)
    ELocConj_Ostar_weighted_sum += ComplexConjugate(e_loc) * Ostar_weighted_increment;
  }

  // Calculate weighted averages
  double weight_sum = 0.0;
  TenElemT e_loc_sum = TenElemT(0.0);
  for (size_t j = 0; j < e_loc_set.size(); j++) {
    e_loc_sum += e_loc_set[j] * weights[j];
    weight_sum += weights[j];
  }
  TenElemT energy = e_loc_sum / weight_sum;

  // Calculate gradient
  // (S_{EO} − E^* S_O) / W_sum
  SplitIndexTPSType gradient = (ELocConj_Ostar_weighted_sum - ComplexConjugate(energy) * Ostar_weighted_sum) * (1.0 / weight_sum);

  // Apply fermion parity operations only for fermion systems
  if constexpr (Index<QNT>::IsFermionic()) {
    gradient.ActFermionPOps();
  }

  return {energy, gradient, 0.0}; // Error is 0 for exact summation
}

/**
 * @brief MPI-enabled exact summation energy evaluator (Fake-parallel per contract)
 *
 * Contract (per docs/dev/design/arch/mpi-contracts.md):
 * - Input state is valid only on the master rank. The evaluator broadcasts the state to all ranks.
 * - Return values (energy, gradient, error) are valid only on the master rank.
 * - For mpi_size == 1, this function degenerates to the single-process evaluator.
 *
 * Implementation (fake-parallel for testing):
 * - Broadcast the state once to satisfy "who uses, who distributes".
 * - Only master rank performs the full exact summation by calling the single-process evaluator.
 * - Non-master ranks return placeholder gradient tensors (zeros). Index compatibility is not guaranteed on placeholders,
 *   because downstream Optimizer contracts require gradient validity only on master rank.
 * - No static partitioning or point-to-point communication is used here.
 *
 * Rationale:
 * - This helper is designed for tests to verify Evaluator correctness independently of MC sampling.
 * - It enforces the MPI contracts without adding performance complexity, minimizing maintenance risk.
 * - For true distributed enumeration, implement a separate parallel version guarded by tests and profiling.
 *
 * @tparam ModelT The model type (must have CalEnergyAndHoles method)
 * @tparam TenElemT Tensor element type (double or complex)
 * @tparam QNT Quantum number type
 * @param split_index_tps_master_only State valid only on master
 * @param all_configs All configurations to enumerate
 * @param trun_para BMPS truncation parameters
 * @param model Physical model
 * @param Ly Lattice rows
 * @param Lx Lattice cols
 * @param comm MPI communicator
 * @param rank MPI rank
 * @param mpi_size MPI world size
 * @return (energy, gradient, error) valid only at master rank. Others return placeholders.
 */
template<typename ModelT, typename TenElemT, typename QNT>
std::tuple<TenElemT, SplitIndexTPS<TenElemT, QNT>, double> ExactSumEnergyEvaluatorMPI(
    const SplitIndexTPS<TenElemT, QNT> &split_index_tps_master_only,
    const std::vector<Configuration> &all_configs,
    const BMPSTruncatePara &trun_para,
    ModelT &model,
    size_t Ly,
    size_t Lx,
    const MPI_Comm &comm,
    int rank,
    int mpi_size
) {
  using SplitIndexTPSType = SplitIndexTPS<TenElemT, QNT>;

  // Degenerate to single-process evaluator when mpi_size == 1
  if (mpi_size == 1) {
    return ExactSumEnergyEvaluator<ModelT, TenElemT, QNT>(
        split_index_tps_master_only, all_configs, trun_para, model, Ly, Lx);
  }

  // Broadcast input state from master to all ranks (to satisfy contract; computation remains master-only)
  SplitIndexTPSType split_index_tps_bcast;
  if (rank == qlten::kMPIMasterRank) {
    split_index_tps_bcast = split_index_tps_master_only;
  }
  qlpeps::MPI_Bcast(split_index_tps_bcast, comm, qlten::kMPIMasterRank);

  if (rank == qlten::kMPIMasterRank) {
    // Master computes using single-process evaluator on the broadcasted state
    return ExactSumEnergyEvaluator<ModelT, TenElemT, QNT>(
        split_index_tps_bcast, all_configs, trun_para, model, Ly, Lx);
  } else {
    // Non-master returns placeholders(zeros)
    SplitIndexTPSType zero_grad(Ly, Lx, split_index_tps_bcast.PhysicalDim());
    return {TenElemT(0.0), zero_grad, 0.0};
  }
}
} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_ENERGY_EVALUATOR_H
