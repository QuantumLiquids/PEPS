/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-06
*
* Description: QuantumLiquids/PEPS project. Mean, Variance, etc.
*/

#ifndef QLPEPS_MONTE_CARLO_TOOLS_STATISTICS_H
#define QLPEPS_MONTE_CARLO_TOOLS_STATISTICS_H

#include <vector>
#include <fstream>
#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <cmath>
#include "qlten/framework/hp_numeric/mpi_fun.h"

namespace qlpeps {
namespace hp_numeric = qlten::hp_numeric;
using qlten::hp_numeric::kMPIMasterRank;

template<typename DataType>
void DumpVecData(
    const std::string &filename,
    const std::vector<DataType> &data
) {
  std::ofstream ofs(filename, std::ofstream::binary);
  if (!ofs.is_open()) {
    throw std::ios_base::failure("Failed to open file: " + filename);
  }
  
  for (auto datum : data) {
    ofs << datum << '\n';
    if (ofs.fail()) {
      throw std::ios_base::failure("Failed to write to file: " + filename);
    }
  }
  ofs << std::endl;
  if (ofs.fail()) {
    throw std::ios_base::failure("Failed to write endl to file: " + filename);
  }
  
  ofs.close();
  if (ofs.fail()) {
    throw std::ios_base::failure("Failed to close file: " + filename);
  }
}

template<typename T>
T Mean(const std::vector<T> &data) {
  if (data.empty()) {
    return T(0);
  }
  auto const count = static_cast<T>(data.size());
//  return std::reduce(data.begin(), data.end()) / count;
  return std::accumulate(data.begin(), data.end(), T(0)) / count;
}

template<typename T>
typename qlten::RealTypeTrait<T>::type Variance(const std::vector<T> &data,
                                                const T &mean) {
  using RealT = typename qlten::RealTypeTrait<T>::type;
  size_t data_size = data.size();
  std::vector<T> diff(data_size);
  std::transform(data.begin(), data.end(), diff.begin(), [mean](T x) { return x - mean; });
#if __cplusplus < 202002L
  RealT sq_sum = 0.0;
  for (const auto &num : diff) {
    sq_sum += RealT(std::norm(num));
  }
#else
  RealT sq_sum = std::transform_reduce(diff.begin(), diff.end(), RealT(0.0), std::plus{},
                                        [](const T &num) {
                                          return static_cast<RealT>(std::norm(num));
                                        });
#endif
  auto const count = static_cast<RealT>(data_size);
  return sq_sum / count;
}

template<typename T>
typename qlten::RealTypeTrait<T>::type Variance(const std::vector<T> &data) {
  return Variance(data, Mean(data));
}

template<typename T>
typename qlten::RealTypeTrait<T>::type StandardError(const std::vector<T> &data,
                                                     const T &mean) {
  using RealT = typename qlten::RealTypeTrait<T>::type;
  if (data.size() == 1) {
    return std::numeric_limits<RealT>::infinity();
  }
  return std::sqrt(Variance(data, mean) / (static_cast<RealT>(data.size()) - RealT(1.0)));
}

template<typename T>
std::vector<T> AveListOfData(
    const std::vector<std::vector<T> > &data //outside idx: sample index; inside idx: something like site/bond
) {
  using RealT = typename qlten::RealTypeTrait<T>::type;
  if (data.size() == 0) {
    return std::vector<T>();
  }
  const size_t N = data[0].size();
  if (N == 0) {
    return std::vector<T>();
  }
  const size_t sample_size = data.size();
  std::vector<T> sum(N, T(0)), ave(N);
  for (size_t sample_idx = 0; sample_idx < sample_size; sample_idx++) {
    for (size_t i = 0; i < N; i++) {
      sum[i] += data[sample_idx][i];
    }
  }
  for (size_t i = 0; i < N; i++) {
    ave[i] = sum[i] / static_cast<RealT>(sample_size);
  }
  return ave;
}

/**
 * @brief Mean and standard error via sqrt(N)-binning across MPI ranks.
 *
 * Method:
 * - Each rank holds a local time series local_samples of length N.
 * - Assumption: N is uniform across ranks in comm.
 * - Bin size b is chosen as floor(sqrt(N)). Trailing samples that do not form a full
 *   bin are discarded per rank. Each rank computes per-bin means; the master rank
 *   gathers all bin means and computes the overall mean and standard error of the mean.
 *
 * Notes:
 * - Different N across ranks is NOT supported by this function by contract. If in
 *   the future N differs, this function should be updated to choose a uniform bin
 *   size across ranks (e.g., b = floor(sqrt(min_i N_i))).
 * - Only the master rank (kMPIMasterRank) returns the computed mean and error;
 *   other ranks return default-initialized values.
 * - Edge cases: If total gathered bins == 0, error remains 0.0; if == 1, error
 *   is set to infinity because it cannot be estimated.
 *
 * Requirements:
 * - ElemT supports accumulation from ElemT(0) and has an MPI datatype mapping via
 *   hp_numeric::GetMPIDataType<ElemT>().
 */
template<typename ElemT>
std::pair<ElemT, typename qlten::RealTypeTrait<ElemT>::type> MeanAndBinnedErrorSqrtNUniformBin(
    const std::vector<ElemT> &local_samples,
    const MPI_Comm &comm) {
  using RealT = typename qlten::RealTypeTrait<ElemT>::type;
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);

  std::vector<ElemT> local_bin_means;
  const size_t num_local_samples = local_samples.size();
  if (num_local_samples > 0) {
    // Determine bin size using sqrt(N) heuristic. Ensure at least 1.
    const size_t bin_size = std::max(1UL, static_cast<size_t>(std::sqrt(num_local_samples)));
    // Integer division is intentional. We discard trailing samples that don't
    // form a full bin. This is a standard and robust practice.
    const size_t num_bins = num_local_samples / bin_size;

    // Only proceed if at least one full bin can be formed.
    if (num_bins > 0) {
      local_bin_means.reserve(num_bins);
      for (size_t i = 0; i < num_bins; ++i) {
        // Calculate sum over the bin
        ElemT bin_sum = std::accumulate(local_samples.begin() + i * bin_size,
                                      local_samples.begin() + (i + 1) * bin_size,
                                      ElemT(0));
        local_bin_means.push_back(bin_sum / static_cast<RealT>(bin_size));
      }
    }
  }

  // --- MPI Communication ---
  // 1. Gather the number of bins from each rank to the master.
  int local_bin_count = static_cast<int>(local_bin_means.size());
  std::vector<int> bin_counts_per_rank;
  if (comm_rank == kMPIMasterRank) {
    bin_counts_per_rank.resize(comm_size);
  }

  MPI_Gather(&local_bin_count, 1, MPI_INT,
             comm_rank == kMPIMasterRank ? bin_counts_per_rank.data() : nullptr, 1, MPI_INT,
             kMPIMasterRank, comm);

  // 2. Use Gatherv to collect all bin means on the master rank.
  std::vector<ElemT> all_bin_means;
  std::vector<int> displacements;
  int total_bins = 0;

  if (comm_rank == kMPIMasterRank) {
    displacements.resize(comm_size);
    for (int i = 0; i < comm_size; ++i) {
      displacements[i] = (i == 0) ? 0 : displacements[i - 1] + bin_counts_per_rank[i - 1];
      total_bins += bin_counts_per_rank[i];
    }
    all_bin_means.resize(total_bins);
  }

  MPI_Gatherv(local_bin_means.data(), local_bin_count, hp_numeric::GetMPIDataType<ElemT>(),
              comm_rank == kMPIMasterRank ? all_bin_means.data() : nullptr,
              comm_rank == kMPIMasterRank ? bin_counts_per_rank.data() : nullptr,
              comm_rank == kMPIMasterRank ? displacements.data() : nullptr,
              hp_numeric::GetMPIDataType<ElemT>(), kMPIMasterRank, comm);

  // --- Final computation on master rank ---
  ElemT mean(0);
  RealT standard_err(0.0);

  if (comm_rank == kMPIMasterRank) {
    if (total_bins > 0) {
      mean = Mean(all_bin_means);
      if (total_bins > 1) {
        standard_err = StandardError(all_bin_means, mean);
      } else {
        // Cannot compute error from a single bin.
        standard_err = std::numeric_limits<RealT>::infinity();
      }
    }
  }
  return std::make_pair(mean, standard_err);
}

///< only rank 0 obtained the result.
template<typename ElemT>
std::pair<ElemT, typename qlten::RealTypeTrait<ElemT>::type> GatherStatisticSingleData(
    ElemT data,
    const MPI_Comm &comm) {
  using RealT = typename qlten::RealTypeTrait<ElemT>::type;
  int comm_rank, comm_size;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm_size(comm, &comm_size);
  if (comm_size == 1) {
    return std::make_pair(data, std::numeric_limits<RealT>::infinity());
  }

  ElemT mean(0);
  RealT standard_err(0);

  std::unique_ptr<ElemT[]> gather_data;
  if (comm_rank == qlten::hp_numeric::kMPIMasterRank) {
    gather_data = std::make_unique<ElemT[]>(comm_size);
  }

  HANDLE_MPI_ERROR(::MPI_Gather(&data,
                                1,
                                hp_numeric::GetMPIDataType<ElemT>(),
                                (void *) gather_data.get(),
                                1,
                                hp_numeric::GetMPIDataType<ElemT>(),
                                qlten::hp_numeric::kMPIMasterRank,
                                comm));

  if (comm_rank == qlten::hp_numeric::kMPIMasterRank) {
    ElemT sum = 0.0;
    for (size_t i = 0; i < comm_size; i++) {
      sum += gather_data[i];
    }
    mean = sum / static_cast<RealT>(comm_size);
    
    if (comm_size > 1) {
      RealT sum_square = 0.0;
      for (size_t i = 0; i < comm_size; i++) {
        sum_square += static_cast<RealT>(std::norm(gather_data[i]));
      }
      RealT variance = sum_square / static_cast<RealT>(comm_size) - static_cast<RealT>(std::norm(mean));
      standard_err = std::sqrt(variance / (static_cast<RealT>(comm_size) - 1));

      // Check if standard error is infinite and output warning with data
      if (std::isinf(standard_err)) {
        std::cerr << "Warning: Infinite standard error detected with " 
                  << comm_size << " processes.\n"
                  << "Gathered data values:\n";
        for (size_t i = 0; i < comm_size; i++) {
          std::cerr << "Process " << i << ": " << gather_data[i] << "\n";
        }
        std::cerr << "Mean value: " << mean << std::endl;
      }
    }

  }
  return std::make_pair(mean, standard_err);
}

template<typename ElemT>
void GatherStatisticListOfData(
    const std::vector<ElemT> &data,
    const MPI_Comm &comm,
    std::vector<ElemT> &avg, //output
    std::vector<double> &std_err//output
) {
  using RealT = typename qlten::RealTypeTrait<ElemT>::type;
  const size_t data_size = data.size(); // number of data
  if (data_size == 0) {
    avg = std::vector<ElemT>();
    std_err = std::vector<double>();
    return;
  }
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  const size_t world_size = mpi_size;
  if (world_size == 1) {
    avg = data;
    std_err = std::vector<double>();
    return;
  }
  std::vector<ElemT> all_data;
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    all_data.resize(world_size * data_size);
  }
  HANDLE_MPI_ERROR(::MPI_Gather(data.data(),
                                data_size,
                                hp_numeric::GetMPIDataType<ElemT>(),
                                rank == qlten::hp_numeric::kMPIMasterRank ? all_data.data() : nullptr,
                                data_size,
                                hp_numeric::GetMPIDataType<ElemT>(),
                                qlten::hp_numeric::kMPIMasterRank,
                                comm));

  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    std::vector<std::vector<ElemT>> data_gather_transposed(data_size, std::vector<ElemT>(world_size));
    for (size_t i = 0; i < world_size; i++) {
      for (size_t j = 0; j < data_size; j++) {
        data_gather_transposed[j][i] = all_data[i * data_size + j];
      }
    }
    avg.resize(data_size);
    std_err.resize(data_size);
    for (size_t i = 0; i < data_size; i++) {
      avg[i] = Mean(data_gather_transposed[i]);
      std_err[i] = static_cast<double>(StandardError(data_gather_transposed[i], avg[i]));
    }
  }
  return;
}

/**
 * @brief Compute short-window IPS (Initial Positive Sequence) variance inflation factor g.
 *
 * The function is not used in the current implementation. But may used in the future.
 * Possble issues: 1. Magic number 
 * 2. May not reuse the mean results
 * 3. biased autocovariance.
 * 
 * Statistical intent: For a weakly correlated MCMC time series {x_t}, the Standard Error of the mean
 *   is inflated by g = 1 + 2 sum_{k=1..K} rho_k^{(+)}, where rho_k are normalized autocorrelations.
 * We adopt the IPS truncation: accumulate only within the longest initial window where
 *   gamma_k = Gamma_k + Gamma_{k+1} > 0, with biased autocovariances Gamma_k estimated with
 * denominator N. When correlations are negligible, g -> 1.
 *
 * Implementation details:
 * - Input values may be real or complex; only their real parts are used for autocorrelation.
 * - Uses a small cap on window size and the IPS criterion to stop early; returns 1 if unstable.
 * - Time complexity O(N*K_cap), suitable for per-rank local estimation before MPI aggregation.
 *
 * @tparam ElemT            Numeric type, supports std::real(ElemT).
 * @param values            Sequence of samples from a single MC chain.
 * @param precomputed_mean  Optional pre-computed mean of the real part of the values. If not provided,
 *                          it's computed internally to avoid redundant calculations.
 * @return double Variance inflation factor g >= 1.0.
 *
 */
template<typename ElemT>
double IPSInflationFactor(const std::vector<ElemT> &values, std::optional<double> precomputed_mean = std::nullopt) {
  constexpr size_t kMaxAutocorrelationLag = 20;
  const size_t N = values.size();
  if (N < 3) { return 1.0; }
  std::vector<double> x(N);
  for (size_t i = 0; i < N; ++i) { x[i] = static_cast<double>(std::real(values[i])); }
  double mu;
  if (precomputed_mean) {
    mu = *precomputed_mean;
  } else {
    mu = 0.0;
    for (double v : x) { mu += v; }
    mu /= static_cast<double>(N);
  }
  for (double &v : x) { v -= mu; }
  auto gamma_hat = [&](size_t k) -> double {
    double acc = 0.0;
    const size_t M = N - k;
    for (size_t t = 0; t < M; ++t) { acc += x[t] * x[t + k]; }
    return acc / static_cast<double>(N); // biased autocovariance with denom N
    // Question: should we use N-k as denominator?
  };
  const double gamma0 = gamma_hat(0);
  if (gamma0 <= 0.0) { return 1.0; }
  const size_t K_max = (N > 2 ? std::min(kMaxAutocorrelationLag, N - 2) : 0);
  double sum_rho_pos = 0.0;
  for (size_t k = 1; k <= K_max; ++k) {
    const double gk = gamma_hat(k);
    const double gk1 = gamma_hat(k + 1);
    const double gamma_seq = gk + gk1; // IPS criterion
    if (gamma_seq <= 0.0) { break; }
    const double rho_k = gk / gamma0;
    sum_rho_pos += rho_k;
  }
  double g = 1.0 + 2.0 * sum_rho_pos;
  if (!(g >= 1.0)) { g = 1.0; }
  return g;
}

}//qlpeps
#endif //QLPEPS_MONTE_CARLO_TOOLS_STATISTICS_H
