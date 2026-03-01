// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-02-28
*
* Description: QuantumLiquids/PEPS project. MinSR eigensolve and pseudo-inverse.
*              Path B: replicated LAPACK eigensolve via MPI_Allgather.
*/

#ifndef QLPEPS_OPTIMIZER_MINSR_EIGENSOLVE_H
#define QLPEPS_OPTIMIZER_MINSR_EIGENSOLVE_H

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include "mpi.h"
#include "qlten/framework/hp_numeric/backend_selector.h"
#include "qlpeps/optimizer/optimizer_params.h"  // MinSRSolverMode

// Path A (ScaLAPACK distributed eigensolve) — included at file scope
// to avoid nested namespace issues. Self-contained with own namespace blocks.
#ifdef QLPEPS_HAS_SCALAPACK
#include "qlpeps/optimizer/minsr_scalapack.h"
#endif

namespace qlpeps {

namespace minsr_detail {

/// Apply pseudo-inverse cutoff to eigenvalues (Chen & Heyl Eq. 22-23).
///
/// Soft cutoff:  lambda_plus = lambda^5 / (lambda^6 + cutoff^6)
/// Hard cutoff:  lambda_plus = 1/lambda if lambda > cutoff, else 0
///
/// @param eigenvalues  Eigenvalues (ascending order from LAPACK)
/// @param r_pinv       Relative cutoff factor
/// @param a_pinv       Absolute cutoff
/// @param soft_cutoff  Use soft cutoff (true) or hard cutoff (false)
/// @return Filtered inverse eigenvalues (same size)
inline std::vector<double> ApplyPseudoInverseCutoff(
    const std::vector<double>& eigenvalues,
    double r_pinv, double a_pinv, bool soft_cutoff) {
  const size_t n = eigenvalues.size();
  if (n == 0) return {};

  // Find |lambda_max| (eigenvalues from dsyev/zheev are in ascending order)
  double lambda_max_abs = 0.0;
  for (const auto& ev : eigenvalues) {
    lambda_max_abs = std::max(lambda_max_abs, std::abs(ev));
  }

  const double cutoff = r_pinv * lambda_max_abs + a_pinv;
  std::vector<double> lambda_plus(n, 0.0);

  if (soft_cutoff) {
    const double cutoff6 = cutoff * cutoff * cutoff * cutoff * cutoff * cutoff;
    for (size_t i = 0; i < n; ++i) {
      const double lam = eigenvalues[i];
      const double lam2 = lam * lam;
      const double lam5 = lam2 * lam2 * lam;
      const double lam6 = lam5 * lam;
      const double denom = lam6 + cutoff6;
      // Guard against 0/0 when both lam and cutoff are zero
      lambda_plus[i] = (denom == 0.0) ? 0.0 : lam5 / denom;
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      if (std::abs(eigenvalues[i]) > cutoff) {
        lambda_plus[i] = 1.0 / eigenvalues[i];
      }
    }
  }
  return lambda_plus;
}

// =============================================================================
// Path B: Replicated eigensolve (LAPACK) — real case
// =============================================================================

/// Replicated eigensolve for real symmetric T.
///
/// 1. Allgather row-blocks to form full T on all ranks.
/// 2. dsyev eigendecomposition (all ranks, identical result).
/// 3. Pseudo-inverse cutoff.
/// 4. y = Z * diag(lambda_plus) * Z^T * rhs (local BLAS).
///
/// @param T_rowblock  Local row-block data (ns_local x ns_global, row-major)
/// @param ns_local    Number of local rows
/// @param ns_global   Total matrix dimension
/// @param rhs         Right-hand side vector (epsilon_bar, length ns_global)
/// @param comm        MPI communicator
/// @param r_pinv      Relative pseudo-inverse cutoff
/// @param a_pinv      Absolute pseudo-inverse cutoff
/// @param soft_cutoff Use soft cutoff
/// @return Solution y (length ns_global, identical on all ranks)
inline std::vector<double> ReplicatedEigenSolveReal(
    const double* T_rowblock, size_t ns_local, size_t ns_global,
    const std::vector<double>& rhs,
    MPI_Comm comm,
    double r_pinv, double a_pinv, bool soft_cutoff) {

  // 1. Allgather to form full T
  std::vector<double> T_full(ns_global * ns_global);
  MPI_Allgather(T_rowblock, static_cast<int>(ns_local * ns_global), MPI_DOUBLE,
                T_full.data(), static_cast<int>(ns_local * ns_global), MPI_DOUBLE,
                comm);

  // LAPACKE_dsyev expects column-major (Fortran layout).
  // Our T_full is row-major. Since T is symmetric, row-major == column-major.
  // So we can pass it directly with LAPACK_COL_MAJOR.
  const lapack_int n = static_cast<lapack_int>(ns_global);
  std::vector<double> eigenvalues(ns_global);
  lapack_int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n,
                                   T_full.data(), n, eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error("LAPACKE_dsyev failed with info = " + std::to_string(info));
  }
  // T_full now contains eigenvectors (columns of Z, column-major)

  // 2. Pseudo-inverse cutoff
  auto lambda_plus = ApplyPseudoInverseCutoff(eigenvalues, r_pinv, a_pinv, soft_cutoff);

  // 3. Compute y = Z * diag(lambda_plus) * Z^T * rhs
  // Step 1: tmp = Z^T * rhs (length ns_global)
  std::vector<double> tmp(ns_global, 0.0);
  // Z is stored column-major in T_full: Z[i,k] = T_full[i + k*n]
  for (size_t k = 0; k < ns_global; ++k) {
    double dot = 0.0;
    for (size_t i = 0; i < ns_global; ++i) {
      dot += T_full[i + k * ns_global] * rhs[i];
    }
    tmp[k] = dot;
  }

  // Step 2: tmp[k] *= lambda_plus[k]
  for (size_t k = 0; k < ns_global; ++k) {
    tmp[k] *= lambda_plus[k];
  }

  // Step 3: y = Z * tmp
  std::vector<double> y(ns_global, 0.0);
  for (size_t i = 0; i < ns_global; ++i) {
    double val = 0.0;
    for (size_t k = 0; k < ns_global; ++k) {
      val += T_full[i + k * ns_global] * tmp[k];
    }
    y[i] = val;
  }

  return y;
}

// =============================================================================
// Path B: Replicated eigensolve (LAPACK) — complex case
// =============================================================================

/// Replicated eigensolve for complex Hermitian T.
///
/// Same algorithm as real case but using zheev for Hermitian eigendecomposition.
/// y = Z * diag(lambda_plus) * Z^dagger * rhs.
inline std::vector<std::complex<double>> ReplicatedEigenSolveComplex(
    const std::complex<double>* T_rowblock, size_t ns_local, size_t ns_global,
    const std::vector<std::complex<double>>& rhs,
    MPI_Comm comm,
    double r_pinv, double a_pinv, bool soft_cutoff) {

  using ComplexT = std::complex<double>;

  // 1. Allgather to form full T
  std::vector<ComplexT> T_full(ns_global * ns_global);
  MPI_Allgather(T_rowblock, static_cast<int>(ns_local * ns_global),
                MPI_CXX_DOUBLE_COMPLEX,
                T_full.data(), static_cast<int>(ns_local * ns_global),
                MPI_CXX_DOUBLE_COMPLEX, comm);

  // T is Hermitian, stored row-major after Allgather.
  // For complex Hermitian matrices, we cannot use the trick of passing
  // row-major data directly to a col-major LAPACK routine: row-major H
  // read as col-major gives H^T = conj(H) != H. LAPACKE_ROW_MAJOR also
  // uses a real transpose internally, producing the same problem.
  //
  // Solution: explicitly convert to column-major, then use LAPACK_COL_MAJOR.
  const lapack_int n = static_cast<lapack_int>(ns_global);
  std::vector<ComplexT> Z(ns_global * ns_global);
  for (size_t i = 0; i < ns_global; ++i) {
    for (size_t j = 0; j < ns_global; ++j) {
      Z[i + j * ns_global] = T_full[i * ns_global + j];
    }
  }

  std::vector<double> eigenvalues(ns_global);
  lapack_int info = LAPACKE_zheev(
      LAPACK_COL_MAJOR, 'V', 'U', n,
      reinterpret_cast<lapack_complex_double*>(Z.data()),
      n, eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error("LAPACKE_zheev failed with info = " + std::to_string(info));
  }
  // Z now holds eigenvectors as columns (col-major): Z[i + k*n] = v_k[i]

  // 2. Pseudo-inverse cutoff (eigenvalues are real for Hermitian matrices)
  auto lambda_plus = ApplyPseudoInverseCutoff(eigenvalues, r_pinv, a_pinv, soft_cutoff);

  // 3. Compute y = Z * diag(lambda_plus) * Z^H * rhs
  // Z is column-major: Z[i,k] = Z[i + k*n] = v_k[i]

  // Step 1: tmp[k] = (Z^H * rhs)[k] = sum_i conj(v_k[i]) * rhs[i]
  std::vector<ComplexT> tmp(ns_global, ComplexT(0));
  for (size_t k = 0; k < ns_global; ++k) {
    ComplexT dot(0);
    for (size_t i = 0; i < ns_global; ++i) {
      dot += std::conj(Z[i + k * ns_global]) * rhs[i];
    }
    tmp[k] = dot;
  }

  // Step 2: tmp[k] *= lambda_plus[k]
  for (size_t k = 0; k < ns_global; ++k) {
    tmp[k] *= lambda_plus[k];
  }

  // Step 3: y[i] = (Z * tmp)[i] = sum_k v_k[i] * tmp[k]
  std::vector<ComplexT> y(ns_global, ComplexT(0));
  for (size_t i = 0; i < ns_global; ++i) {
    ComplexT val(0);
    for (size_t k = 0; k < ns_global; ++k) {
      val += Z[i + k * ns_global] * tmp[k];
    }
    y[i] = val;
  }

  return y;
}

} // namespace minsr_detail

/// Dispatch to real or complex replicated eigensolve based on TenElemT.
template<typename TenElemT>
std::vector<TenElemT> ReplicatedEigenSolveWithPseudoInverse(
    const TenElemT* T_rowblock, size_t ns_local, size_t ns_global,
    const std::vector<TenElemT>& rhs,
    MPI_Comm comm,
    double r_pinv, double a_pinv, bool soft_cutoff) {
  if constexpr (std::is_same_v<TenElemT, double>) {
    return minsr_detail::ReplicatedEigenSolveReal(
        T_rowblock, ns_local, ns_global, rhs, comm, r_pinv, a_pinv, soft_cutoff);
  } else if constexpr (std::is_same_v<TenElemT, std::complex<double>>) {
    return minsr_detail::ReplicatedEigenSolveComplex(
        T_rowblock, ns_local, ns_global, rhs, comm, r_pinv, a_pinv, soft_cutoff);
  } else {
    static_assert(sizeof(TenElemT) == 0, "Unsupported TenElemT");
  }
}

// =============================================================================
// Path A/B dispatch based on MinSRSolverMode
// =============================================================================

namespace minsr_detail {
/// Auto-dispatch threshold: above this Ns, prefer Path A if available.
constexpr size_t kAutoDispatchThreshold = 5000;
} // namespace minsr_detail

/**
 * @brief Dispatch MinSR eigensolve to Path A (ScaLAPACK) or Path B (replicated LAPACK).
 *
 * Selection logic:
 * - kReplicated: always Path B.
 * - kDistributed: always Path A (throws if ScaLAPACK not available).
 * - kAuto: Path A if ScaLAPACK available AND N > kAutoDispatchThreshold, else Path B.
 */
template<typename TenElemT>
std::vector<TenElemT> MinSREigenSolveDispatch(
    const TenElemT* T_rowblock, size_t local_rows, size_t N,
    const std::vector<TenElemT>& rhs, MPI_Comm comm,
    double r_pinv, double a_pinv, bool soft_cutoff,
    MinSRSolverMode solver_mode) {

  bool use_scalapack = false;

  switch (solver_mode) {
    case MinSRSolverMode::kReplicated:
      use_scalapack = false;
      break;
    case MinSRSolverMode::kDistributed:
#ifdef QLPEPS_HAS_SCALAPACK
      use_scalapack = true;
#else
      throw std::runtime_error(
          "MinSRSolverMode::kDistributed requested but ScaLAPACK is not available. "
          "Rebuild with -DQLPEPS_USE_SCALAPACK=ON.");
#endif
      break;
    case MinSRSolverMode::kAuto:
#ifdef QLPEPS_HAS_SCALAPACK
      use_scalapack = (N > minsr_detail::kAutoDispatchThreshold);
#else
      use_scalapack = false;
#endif
      break;
  }

  if (use_scalapack) {
#ifdef QLPEPS_HAS_SCALAPACK
    return minsr_detail::ScaLAPACKEigenSolveWithPseudoInverse<TenElemT>(
        T_rowblock, local_rows, N, rhs, comm, r_pinv, a_pinv, soft_cutoff);
#else
    // Unreachable — guarded by switch above.
    throw std::runtime_error("ScaLAPACK path selected but not compiled in.");
#endif
  } else {
    return ReplicatedEigenSolveWithPseudoInverse<TenElemT>(
        T_rowblock, local_rows, N, rhs, comm, r_pinv, a_pinv, soft_cutoff);
  }
}

} // namespace qlpeps

#endif // QLPEPS_OPTIMIZER_MINSR_EIGENSOLVE_H
