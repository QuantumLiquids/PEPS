/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-18
*
* Description: QuantumLiquids/PEPS project. SMatrix in Stochastic Reconfiguration. Especially define the multiplication on vector.
*/


#ifndef QLPEPS_VMC_PEPS_STOCHASTIC_RECONFIGURATION_SMATRIX_H
#define QLPEPS_VMC_PEPS_STOCHASTIC_RECONFIGURATION_SMATRIX_H

#include "qlpeps/two_dim_tn/tps/split_index_tps.h"
#include "qlten/framework/hp_numeric/mpi_fun.h"    // qlten::hp_numeric::MPI_Bcast

namespace qlpeps {

/*
 * SRSMatrix implements the SR covariance matrix-vector product:
 *   Sv = (1/N) Σ_i O*_i · δ_i + diag_shift · v
 * where δ_i = (O_i · v) − (Ō · v) is a centered scalar projection.
 *
 * This avoids catastrophic cancellation by moving the mean subtraction
 * from TPS-level (large − large = small) to scalar-level, where it
 * is numerically benign.
 *
 * Ostar_samples, Ostar_mean are expected to be in the physical O* representation:
 *   O^*(S) = Pi(R^*(S))
 * as prepared by the energy evaluators.
 *
 * MPI contract:
 * - Ostar_mean_ is non-null only on master (rank 0). Master computes
 *   mean_dot_v = Ō · v and broadcasts the scalar to all ranks each
 *   matvec. diag_shift is also applied only on master.
 * - When world_size == 1, no MPI communication occurs.
 */
template<typename TenElemT, typename QNT>
class SRSMatrix {
  using SITPS = SplitIndexTPS<TenElemT, QNT>;
 public:
  SRSMatrix(std::vector<SITPS> *Ostar_samples, SITPS *Ostar_mean,
            size_t world_size, MPI_Comm comm) :
      Ostar_samples_(Ostar_samples), Ostar_mean_(Ostar_mean),
      world_size_(world_size), comm_(comm) {}

  SITPS operator*(const SITPS &v0) const {
    // Option 2: centered scalar projection.
    // δ_i = (O_i · v) − (Ō · v) is a scalar subtraction,
    // avoiding catastrophic cancellation of two large TPS objects.

    // Master computes mean_dot_v; broadcast to all ranks.
    TenElemT mean_dot_v(0.0);
    if (Ostar_mean_ != nullptr) {
      mean_dot_v = (*Ostar_mean_) * v0;
    }
    if (world_size_ > 1) {
      qlten::hp_numeric::MPI_Bcast(&mean_dot_v, 1,
                                   qlten::hp_numeric::kMPIMasterRank, comm_);
    }

    TenElemT delta_0 = (*Ostar_samples_)[0] * v0 - mean_dot_v;
    SITPS res = delta_0 * (*Ostar_samples_)[0];
    for (size_t i = 1; i < Ostar_samples_->size(); i++) {
      TenElemT delta_i = (*Ostar_samples_)[i] * v0 - mean_dot_v;
      res += delta_i * (*Ostar_samples_)[i];
    }
    res *= 1.0 / double(Ostar_samples_->size() * world_size_);

#ifndef NDEBUG
    // Verify ⟨δ⟩ ≈ 0 (equivalence of Option 2 and Option 3).
    // In exact arithmetic, ⟨δ⟩ = ⟨O·v⟩ − Ō·v = 0.
    // Only meaningful in single-rank mode; with multiple ranks each rank
    // holds a subset of samples so local ⟨δ⟩ ≠ 0 is expected.
    if (world_size_ == 1) {
      TenElemT sum_delta(0.0);
      for (size_t i = 0; i < Ostar_samples_->size(); i++) {
        sum_delta += (*Ostar_samples_)[i] * v0 - mean_dot_v;
      }
      double mean_delta_abs = std::abs(sum_delta / double(Ostar_samples_->size()));
      if (mean_delta_abs > 1e-10) {
        std::cerr << "[SRSMatrix] Warning: |<delta>| = " << std::scientific << mean_delta_abs
                  << " (expected ~ 0). Option 2/3 equivalence may be degraded." << std::endl;
      }
    }
#endif

    if (Ostar_mean_ != nullptr && diag_shift != 0.0) {
      res += (diag_shift * v0);
    }

    return res;
  }

  TenElemT diag_shift = 0.0;
 private:
  std::vector<SITPS> *Ostar_samples_;
  SITPS *Ostar_mean_;
  size_t world_size_;
  MPI_Comm comm_;
};

}//qlpeps

#endif //QLPEPS_VMC_PEPS_STOCHASTIC_RECONFIGURATION_SMATRIX_H
