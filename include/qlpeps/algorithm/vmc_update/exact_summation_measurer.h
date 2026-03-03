// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-03-02
*
* Description: QuantumLiquids/PEPS project.
* Exact summation measurement evaluator — iterates all configurations to produce
* deterministic measurement results from EvaluateObservables().
* Test-only utility, analogous to ExactSumEnergyEvaluatorMPI.
*/

#ifndef QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_MEASURER_H
#define QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_MEASURER_H

#include "qlten/qlten.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlpeps/vmc_basic/wave_function_component.h"
#include "qlpeps/algorithm/vmc_update/model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h" // Vec2Config, GenerateAllPermutationConfigs
#include <mpi.h>
#include <unordered_map>
#include <string>
#include <algorithm>  // std::sort
#include <numeric>    // std::iota
#include <limits>
#include <stdexcept>

namespace qlpeps {

/**
 * @brief Generate all 2^N binary configurations for an Lx x Ly lattice.
 *
 * Used for models without conserved quantum numbers (e.g., TFIM where
 * the transverse field breaks Sz conservation). Each site takes value 0 or 1.
 *
 * @param Lx Number of columns
 * @param Ly Number of rows
 * @return All 2^(Lx*Ly) configurations in lexicographic bit order
 */
inline std::vector<Configuration> GenerateAllBinaryConfigs(size_t Lx, size_t Ly) {
  if (Ly != 0 && Lx > std::numeric_limits<size_t>::max() / Ly) {
    throw std::invalid_argument("GenerateAllBinaryConfigs: Lx*Ly overflows size_t");
  }
  const size_t N = Lx * Ly;
  if (N >= std::numeric_limits<size_t>::digits) {
    throw std::invalid_argument("GenerateAllBinaryConfigs: Lx*Ly must be < size_t bit width");
  }
  const size_t num_configs = 1ULL << N;
  std::vector<Configuration> all_configs;
  all_configs.reserve(num_configs);
  for (size_t i = 0; i < num_configs; ++i) {
    std::vector<size_t> config_vec(N);
    for (size_t bit = 0; bit < N; ++bit) {
      config_vec[bit] = (i >> bit) & 1;
    }
    all_configs.push_back(Vec2Config(config_vec, Lx, Ly));
  }
  return all_configs;
}

/**
 * @brief Exact-summation measurement evaluator (MPI-parallel).
 *
 * Iterates over ALL configurations in the Hilbert space (or a specified subset),
 * constructs TPSWaveFunctionComponent for each, calls the measurement solver's
 * EvaluateObservables(), and accumulates weighted-average observables.
 *
 * This is a test-only utility for golden-regression of measurement observables.
 * It produces deterministic results (no MC noise).
 *
 * MPI pattern: same as ExactSumEnergyEvaluatorMPI — broadcast state, stripe-distribute
 * configs, reduce weighted sums.
 *
 * @tparam MeasurementSolverT  Measurement solver type (e.g., SquareSpinlessFermion)
 * @tparam TenElemT  Tensor element type (QLTEN_Double or QLTEN_Complex)
 * @tparam QNT  Quantum number type
 * @tparam ContractorT  Contractor template (default: BMPSContractor)
 *
 * @param split_index_tps_master_only  TPS state (valid on master rank only)
 * @param all_configs  All configurations to sum over
 * @param trun_para  BMPS truncation parameters
 * @param solver  Measurement solver instance (mutable for caching)
 * @param Ly  Number of rows
 * @param Lx  Number of columns
 * @param comm  MPI communicator
 * @param rank  MPI rank
 * @param mpi_size  Number of MPI processes
 *
 * @return ObservableMap with exact weighted-average observables (valid on master only)
 */
template<typename MeasurementSolverT, typename TenElemT, typename QNT,
         template<typename, typename> class ContractorT = BMPSContractor>
ObservableMap<TenElemT> ExactSumMeasurerMPI(
    const SplitIndexTPS<TenElemT, QNT> &split_index_tps_master_only,
    const std::vector<Configuration> &all_configs,
    const typename ContractorT<TenElemT, QNT>::TruncateParams &trun_para,
    MeasurementSolverT &solver,
    size_t Ly,
    size_t Lx,
    const MPI_Comm &comm,
    int rank,
    int mpi_size) {
  if (all_configs.empty()) {
    throw std::invalid_argument("ExactSumMeasurerMPI: all_configs must not be empty");
  }

  using SplitIndexTPSType = SplitIndexTPS<TenElemT, QNT>;

  // 1. Broadcast state from master to all ranks
  SplitIndexTPSType split_index_tps_bcast(Ly, Lx);
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    split_index_tps_bcast = split_index_tps_master_only;
  }
  if (mpi_size > 1) {
    qlpeps::MPI_Bcast(split_index_tps_bcast, comm, qlten::hp_numeric::kMPIMasterRank);
  }

  // 2. Stripe-distribute configs and accumulate weighted observables
  double weight_rank = 0.0;
  ObservableMap<TenElemT> weighted_obs_rank;  // key -> weighted sum of values

  for (size_t i = rank; i < all_configs.size(); i += mpi_size) {
    const auto &config = all_configs[i];

    TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, ContractorT>
        tps_sample(split_index_tps_bcast, config, trun_para);
    const double weight = std::norm(tps_sample.amplitude);
    weight_rank += weight;

    // Call the measurement solver
    auto obs = solver.template EvaluateObservables<TenElemT, QNT>(
        &split_index_tps_bcast, &tps_sample);

    // Accumulate weighted observables
    for (auto &[key, values] : obs) {
      auto &accum = weighted_obs_rank[key];
      if (accum.empty()) {
        accum.resize(values.size(), TenElemT(0));
      }
      for (size_t j = 0; j < values.size(); ++j) {
        accum[j] += weight * values[j];
      }
    }
  }

  // 3. Build deterministic key order for MPI collectives.
  //    All configs produce the same observable keys from the same solver,
  //    but some ranks may have processed zero configs (when mpi_size > num_configs).
  //    Strategy: broadcast key metadata from rank 0 of those who processed at least
  //    one config. Since config 0 is always assigned to rank 0, and we require
  //    all_configs.size() >= 1, rank 0 always has at least one config.
  //    Use sorted keys so all ranks iterate in identical order.

  // Rank 0 extracts sorted key list and sizes
  std::vector<std::string> sorted_keys;
  std::vector<size_t> key_sizes;  // number of TenElemT values per key
  for (const auto &[key, values] : weighted_obs_rank) {
    sorted_keys.push_back(key);
    key_sizes.push_back(values.size());
  }
  // Sort by key name for deterministic ordering across ranks
  {
    std::vector<size_t> perm(sorted_keys.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&](size_t a, size_t b) { return sorted_keys[a] < sorted_keys[b]; });
    std::vector<std::string> sk(sorted_keys.size());
    std::vector<size_t> ks(key_sizes.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      sk[i] = sorted_keys[perm[i]];
      ks[i] = key_sizes[perm[i]];
    }
    sorted_keys = std::move(sk);
    key_sizes = std::move(ks);
  }

  // Broadcast key count and metadata from rank 0 so idle ranks know what to reduce
  if (mpi_size > 1) {
    int num_keys = static_cast<int>(sorted_keys.size());
    HANDLE_MPI_ERROR(::MPI_Bcast(&num_keys, 1, MPI_INT,
                                  qlten::hp_numeric::kMPIMasterRank, comm));

    if (rank != qlten::hp_numeric::kMPIMasterRank) {
      sorted_keys.resize(num_keys);
      key_sizes.resize(num_keys);
    }

    // Broadcast key names: serialize as lengths + chars
    for (int k = 0; k < num_keys; ++k) {
      int len = static_cast<int>(sorted_keys[k].size());
      HANDLE_MPI_ERROR(::MPI_Bcast(&len, 1, MPI_INT,
                                    qlten::hp_numeric::kMPIMasterRank, comm));
      if (rank != qlten::hp_numeric::kMPIMasterRank) {
        sorted_keys[k].resize(len);
      }
      HANDLE_MPI_ERROR(::MPI_Bcast(sorted_keys[k].data(), len, MPI_CHAR,
                                    qlten::hp_numeric::kMPIMasterRank, comm));
    }

    // Broadcast value sizes per key
    // (reinterpret size_t as unsigned long long for portability)
    static_assert(sizeof(size_t) == sizeof(unsigned long long) ||
                  sizeof(size_t) == sizeof(unsigned long));
    HANDLE_MPI_ERROR(::MPI_Bcast(key_sizes.data(),
                                  static_cast<int>(key_sizes.size()),
                                  sizeof(size_t) == 8 ? MPI_UNSIGNED_LONG_LONG : MPI_UNSIGNED_LONG,
                                  qlten::hp_numeric::kMPIMasterRank, comm));
  }

  // 4. MPI reduction with deterministic key order
  double weight_sum = 0.0;
  ObservableMap<TenElemT> result;

  if (mpi_size > 1) {
    HANDLE_MPI_ERROR(::MPI_Reduce(&weight_rank, &weight_sum, 1,
                                   MPI_DOUBLE, MPI_SUM,
                                   qlten::hp_numeric::kMPIMasterRank, comm));

    for (size_t k = 0; k < sorted_keys.size(); ++k) {
      const auto &key = sorted_keys[k];
      const size_t n = key_sizes[k];

      // Local buffer: use accumulated values if this rank has them, else zeros
      std::vector<TenElemT> local_buf(n, TenElemT(0));
      auto it = weighted_obs_rank.find(key);
      if (it != weighted_obs_rank.end()) {
        local_buf = std::move(it->second);
      }

      std::vector<TenElemT> global_buf(n, TenElemT(0));
      HANDLE_MPI_ERROR(::MPI_Reduce(local_buf.data(), global_buf.data(),
                                     static_cast<int>(n),
                                     qlten::hp_numeric::GetMPIDataType<TenElemT>(),
                                     MPI_SUM,
                                     qlten::hp_numeric::kMPIMasterRank, comm));
      if (rank == qlten::hp_numeric::kMPIMasterRank) {
        result[key] = std::move(global_buf);
      }
    }
  } else {
    weight_sum = weight_rank;
    result = std::move(weighted_obs_rank);
  }

  // 5. Master divides by total weight to get exact means
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    if (!(weight_sum > 0.0)) {
      throw std::runtime_error("ExactSumMeasurerMPI: total weight must be positive");
    }
    for (auto &[key, values] : result) {
      for (auto &v : values) {
        v /= weight_sum;
      }
    }
  }

  return result;
}

} // namespace qlpeps

#endif // QLPEPS_ALGORITHM_VMC_UPDATE_EXACT_SUMMATION_MEASURER_H
