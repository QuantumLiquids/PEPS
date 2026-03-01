// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-28
*
* Description: QuantumLiquids/PEPS project. MinSR T-matrix construction via
*              MPI ring exchange with four-term centering formula.
*/

#ifndef QLPEPS_OPTIMIZER_MINSR_TMATRIX_H
#define QLPEPS_OPTIMIZER_MINSR_TMATRIX_H

#include <vector>
#include <complex>
#include <cassert>
#include <numeric>
#include "mpi.h"
#include "qlpeps/two_dim_tn/tps/split_index_tps.h"

namespace qlpeps {

/**
 * @brief MinSR T-matrix: Ns x Ns centered Gram matrix stored as distributed row-blocks.
 *
 * Each MPI rank holds Ns_local contiguous rows of the matrix (row-major dense layout).
 * The matrix T_ij = (1/Ns) * <O_bar_i | O_bar_j> is computed via:
 *   1. Ring exchange of O* sample batches to compute raw inner products.
 *   2. Four-term centering formula (no explicit centered O storage).
 *   3. 1/Ns normalization to match the existing SR convention.
 *
 * The ring exchange uses a pipeline-safe pattern with blocking SITPS MPI_Send/MPI_Recv:
 * rank (P-1) sends first in each round to break the circular wait.
 *
 * @tparam TenElemT Tensor element type (double or std::complex<double>)
 * @tparam QNT Quantum number type
 */
template<typename TenElemT, typename QNT>
class MinSRTMatrix {
 public:
  using SITPST = SplitIndexTPS<TenElemT, QNT>;

  MinSRTMatrix() = default;

  /**
   * @brief Construct T via ring exchange + four-term centering.
   *
   * After this call, each rank holds its centered, (1/Ns)-normalized row-block of T.
   * Centering terms (m_i, c) are derived from Gram matrix row averages,
   * so O_mean is NOT needed here.
   *
   * @param Ostar_samples  Local O* samples on this rank (size Ns_local)
   * @param comm           MPI communicator
   */
  void Construct(const std::vector<SITPST>& Ostar_samples, MPI_Comm comm) {
    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    ns_local_ = Ostar_samples.size();
    ns_global_ = ns_local_ * static_cast<size_t>(world_size);
    comm_ = comm;

    // Allocate row-block storage (row-major: ns_local_ rows x ns_global_ cols)
    row_block_.assign(ns_local_ * ns_global_, TenElemT(0));
    std::vector<TenElemT> row_sums(ns_local_, TenElemT(0));

    // --- Round 0: local inner products (no MPI) ---
    const size_t local_col_start = static_cast<size_t>(rank) * ns_local_;
    for (size_t i = 0; i < ns_local_; ++i) {
      for (size_t j = 0; j < ns_local_; ++j) {
        TenElemT ip = Ostar_samples[i] * Ostar_samples[j];
        row_block_[i * ns_global_ + local_col_start + j] = ip;
        row_sums[i] += ip;
      }
    }

    // --- Rounds 1..P-1: ring exchange ---
    if (world_size > 1) {
      const int next_rank = (rank + 1) % world_size;
      const int prev_rank = (rank - 1 + world_size) % world_size;

      // Double-buffered: send_ptr points to what we send, recv_buf receives
      const std::vector<SITPST>* send_ptr = &Ostar_samples;
      std::vector<SITPST> owned_send_buf;
      std::vector<SITPST> recv_buf(ns_local_);

      for (int round = 1; round < world_size; ++round) {
        // Source rank of the batch we are about to receive
        const int source_rank = ((rank - round) % world_size + world_size) % world_size;
        const size_t col_start = static_cast<size_t>(source_rank) * ns_local_;

        // Pipeline-safe ring exchange:
        // Rank (P-1) sends first, all others receive first.
        // This breaks the circular dependency in the ring.
        if (rank == world_size - 1) {
          SendBatch_(*send_ptr, next_rank, comm);
          RecvBatch_(recv_buf, prev_rank, comm);
        } else {
          RecvBatch_(recv_buf, prev_rank, comm);
          SendBatch_(*send_ptr, next_rank, comm);
        }

        // Compute cross inner products
        for (size_t i = 0; i < ns_local_; ++i) {
          for (size_t j = 0; j < ns_local_; ++j) {
            TenElemT ip = Ostar_samples[i] * recv_buf[j];
            row_block_[i * ns_global_ + col_start + j] = ip;
            row_sums[i] += ip;
          }
        }

        // Next round: send what we just received
        owned_send_buf = std::move(recv_buf);
        send_ptr = &owned_send_buf;
        recv_buf.resize(ns_local_);
      }
    }

    // --- Derive centering terms from row averages ---
    const double ns_d = static_cast<double>(ns_global_);

    // m_i = row_sum_i / Ns
    m_.resize(ns_local_);
    for (size_t i = 0; i < ns_local_; ++i) {
      m_[i] = row_sums[i] / ns_d;
    }

    // Allgather all m values (Ns scalars total)
    all_m_.resize(ns_global_);
    MPI_Allgather(m_.data(), static_cast<int>(ns_local_),
                  GetMPIDataType_(), all_m_.data(),
                  static_cast<int>(ns_local_), GetMPIDataType_(), comm);

    // c = sum_j m_j / Ns
    c_ = TenElemT(0);
    for (size_t j = 0; j < ns_global_; ++j) {
      c_ += all_m_[j];
    }
    c_ /= ns_d;

    // --- Apply four-term centering in-place and 1/Ns normalization ---
    for (size_t i = 0; i < ns_local_; ++i) {
      for (size_t j = 0; j < ns_global_; ++j) {
        TenElemT& entry = row_block_[i * ns_global_ + j];
        entry = (entry - m_[i] - Conj_(all_m_[j]) + c_) / ns_d;
      }
    }
  }

  /// Dense row-block data pointer (row-major, ns_local x ns_global)
  TenElemT* RowBlockData() { return row_block_.data(); }
  const TenElemT* RowBlockData() const { return row_block_.data(); }

  /// Number of local rows (Ns_local)
  size_t LocalRows() const { return ns_local_; }

  /// Total matrix dimension (Ns)
  size_t GlobalSize() const { return ns_global_; }

  /// Local centering dots m_i (size Ns_local)
  const std::vector<TenElemT>& LocalCenteringDots() const { return m_; }

  /// All centering dots (size Ns, available after Construct)
  const std::vector<TenElemT>& AllCenteringDots() const { return all_m_; }

  /// Grand centering constant c
  TenElemT CenteringNormSq() const { return c_; }

 private:
  size_t ns_local_ = 0;
  size_t ns_global_ = 0;
  MPI_Comm comm_ = MPI_COMM_NULL;

  std::vector<TenElemT> row_block_;   ///< Dense row-block (ns_local x ns_global, row-major)
  std::vector<TenElemT> m_;           ///< Local centering dots (ns_local)
  std::vector<TenElemT> all_m_;       ///< All centering dots (ns_global)
  TenElemT c_ = TenElemT(0);         ///< Grand centering constant

  /// Send a batch of SITPS objects to dest rank.
  static void SendBatch_(const std::vector<SITPST>& batch, int dest, MPI_Comm comm) {
    for (size_t i = 0; i < batch.size(); ++i) {
      ::qlpeps::MPI_Send(batch[i], dest, comm);
    }
  }

  /// Receive a batch of SITPS objects from source rank.
  static void RecvBatch_(std::vector<SITPST>& batch, int source, MPI_Comm comm) {
    for (size_t i = 0; i < batch.size(); ++i) {
      ::qlpeps::MPI_Recv(batch[i], source, comm);
    }
  }

  /// Conjugate helper: identity for real, complex conjugate for complex.
  static TenElemT Conj_(TenElemT val) {
    if constexpr (std::is_same_v<TenElemT, double>) {
      return val;
    } else {
      return std::conj(val);
    }
  }

  /// Get MPI datatype for TenElemT.
  static MPI_Datatype GetMPIDataType_() {
    if constexpr (std::is_same_v<TenElemT, double>) {
      return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<TenElemT, std::complex<double>>) {
      return MPI_CXX_DOUBLE_COMPLEX;
    } else {
      static_assert(sizeof(TenElemT) == 0, "Unsupported TenElemT for MPI");
      return MPI_DATATYPE_NULL;
    }
  }
};

} // namespace qlpeps

#endif // QLPEPS_OPTIMIZER_MINSR_TMATRIX_H
