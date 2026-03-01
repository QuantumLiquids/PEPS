// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2026-03-01
*
* Description: QuantumLiquids/PEPS project. ScaLAPACK distributed eigensolve
*              for MinSR Path A. Reduces per-rank memory from O(Ns^2) to O(Ns^2/P).
*/

#ifndef QLPEPS_OPTIMIZER_MINSR_SCALAPACK_H
#define QLPEPS_OPTIMIZER_MINSR_SCALAPACK_H

#ifdef QLPEPS_HAS_SCALAPACK

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "mpi.h"

// =============================================================================
// Section A: extern "C" declarations for BLACS and ScaLAPACK
// =============================================================================

extern "C" {

// BLACS C interface
void Cblacs_pinfo(int* mypnum, int* nprocs);
void Cblacs_get(int context, int request, int* value);
void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
void Cblacs_gridexit(int context);
int Csys2blacs_handle(MPI_Comm comm);

// ScaLAPACK Fortran API (trailing underscores)
int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);

void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
               const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

// Redistribute: general dense matrix copy between BLACS grids
void pdgemr2d_(const int* m, const int* n,
               const double* A, const int* ia, const int* ja, const int* descA,
               double* B, const int* ib, const int* jb, const int* descB,
               const int* ictxt);

void pzgemr2d_(const int* m, const int* n,
               const void* A, const int* ia, const int* ja, const int* descA,
               void* B, const int* ib, const int* jb, const int* descB,
               const int* ictxt);

// Eigensolve: symmetric / Hermitian
void pdsyev_(const char* jobz, const char* uplo, const int* n,
             double* A, const int* ia, const int* ja, const int* descA,
             double* W, double* Z, const int* iz, const int* jz, const int* descZ,
             double* work, const int* lwork, int* info);

void pzheev_(const char* jobz, const char* uplo, const int* n,
             void* A, const int* ia, const int* ja, const int* descA,
             double* W, void* Z, const int* iz, const int* jz, const int* descZ,
             void* work, const int* lwork, double* rwork, const int* lrwork, int* info);

// indxl2g: convert local index to global (1-based)
int indxl2g_(const int* indxloc, const int* nb, const int* iproc,
             const int* isrcproc, const int* nprocs);

} // extern "C"

namespace qlpeps {
namespace minsr_detail {

// Forward declaration — defined in minsr_eigensolve.h
std::vector<double> ApplyPseudoInverseCutoff(
    const std::vector<double>& eigenvalues,
    double r_pinv, double a_pinv, bool soft_cutoff);

// =============================================================================
// Section B: ScaLAPACKContext RAII struct
// =============================================================================

/**
 * @brief RAII wrapper for a BLACS 2D process grid context.
 *
 * Created per eigensolve call. Cost of Cblacs_gridinit is negligible
 * compared to the O(Ns^3/P) eigensolve.
 */
struct ScaLAPACKContext {
  int ictxt = -1;
  int nprow = 0, npcol = 0;
  int myrow = -1, mycol = -1;
  int nb = 64;            ///< Block size for block-cyclic distribution
  bool valid = false;
  MPI_Comm dup_comm = MPI_COMM_NULL;  ///< Owned duplicated communicator

  ScaLAPACKContext() = default;

  // Non-copyable, non-movable (RAII resource — moving would require nullifying source handles)
  ScaLAPACKContext(const ScaLAPACKContext&) = delete;
  ScaLAPACKContext& operator=(const ScaLAPACKContext&) = delete;
  ScaLAPACKContext(ScaLAPACKContext&&) = delete;
  ScaLAPACKContext& operator=(ScaLAPACKContext&&) = delete;

  /// Initialize a near-square 2D process grid.
  void Init(MPI_Comm comm, int nprocs, int my_rank) {
    MPI_Comm_dup(comm, &dup_comm);
    ictxt = Csys2blacs_handle(dup_comm);

    // Near-square grid via MPI_Dims_create
    int dims[2] = {0, 0};
    MPI_Dims_create(nprocs, 2, dims);
    nprow = dims[0];
    npcol = dims[1];

    Cblacs_gridinit(&ictxt, "R", nprow, npcol);  // Row-major mapping
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    valid = (myrow >= 0 && mycol >= 0);
  }

  /// Initialize a Px1 column grid (for row-block source layout).
  void InitColumn(MPI_Comm comm, int nprocs) {
    MPI_Comm_dup(comm, &dup_comm);
    ictxt = Csys2blacs_handle(dup_comm);
    nprow = nprocs;
    npcol = 1;
    Cblacs_gridinit(&ictxt, "R", nprow, npcol);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);
    valid = (myrow >= 0 && mycol >= 0);
  }

  ~ScaLAPACKContext() {
    if (valid) {
      Cblacs_gridexit(ictxt);
    }
    if (dup_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&dup_comm);
    }
  }
};

// =============================================================================
// Section C: Helper functions
// =============================================================================

/// Compute local dimension for block-cyclic distribution (wraps numroc_).
inline int LocalBlockCyclicDim(int n, int nb, int iproc, int nprocs) {
  const int isrcproc = 0;
  return numroc_(&n, &nb, &iproc, &isrcproc, &nprocs);
}

/// Initialize a ScaLAPACK array descriptor (wraps descinit_).
inline void InitDescriptor(int* desc, int m, int n, int mb, int nb,
                           int ictxt, int lld) {
  int info = 0;
  const int irsrc = 0, icsrc = 0;
  descinit_(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);
  if (info != 0) {
    throw std::runtime_error("descinit_ failed with info = " + std::to_string(info));
  }
}

/// Convert local block-cyclic index to global index (0-based).
inline int BlockCyclicLocalToGlobal(int local_idx, int nb, int iproc, int nprocs) {
  // indxl2g_ is 1-based
  int loc1 = local_idx + 1;
  const int isrcproc = 0;
  return indxl2g_(&loc1, &nb, &iproc, &isrcproc, &nprocs) - 1;
}

// =============================================================================
// Section D: Distributed eigensolve — real symmetric case
// =============================================================================

/**
 * @brief Path A distributed eigensolve for real symmetric T.
 *
 * Algorithm:
 * 1. Create source context (Px1) and target context (near-square 2D).
 * 2. pdgemr2d_: redistribute row-block -> block-cyclic.
 * 3. pdsyev_: distributed eigendecomposition.
 * 4. Pseudo-inverse cutoff on eigenvalues.
 * 5. Distributed matvec: y = Z * diag(lambda_plus) * Z^T * rhs via MPI_Allreduce.
 *
 * @note The local ns_local x N row-block is rectangular, so row-major != col-major
 *       even for a globally symmetric matrix. Explicit row-to-column-major conversion
 *       is required before passing to pdgemr2d_ (which expects Fortran column-major).
 */
inline std::vector<double> ScaLAPACKEigenSolveReal(
    const double* T_rowblock, size_t local_rows, size_t N,
    const std::vector<double>& rhs, MPI_Comm comm,
    double r_pinv, double a_pinv, bool soft_cutoff) {

  if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("ScaLAPACKEigenSolveReal: N exceeds MPI/LAPACK int limit");
  }

  int nprocs, my_rank;
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &my_rank);

  const int n = static_cast<int>(N);
  const int ns_local = static_cast<int>(local_rows);

  // --- Transpose local row-block to column-major ---
  // Row-major: T_rowblock[i*N + j], Column-major: T_src_colmaj[j*ns_local + i]
  // The global matrix is symmetric, but the local ns_local x N block is rectangular,
  // so row-major != col-major. ScaLAPACK (Fortran) expects column-major.
  std::vector<double> T_src_colmaj(static_cast<size_t>(ns_local) * N);
  for (int i = 0; i < ns_local; ++i) {
    for (size_t j = 0; j < N; ++j) {
      T_src_colmaj[j * ns_local + i] = T_rowblock[i * N + j];
    }
  }

  // --- Step 1: Create BLACS contexts ---
  ScaLAPACKContext src_ctx;
  src_ctx.InitColumn(comm, nprocs);

  ScaLAPACKContext tgt_ctx;
  tgt_ctx.Init(comm, nprocs, my_rank);

  const int nb = tgt_ctx.nb;

  // --- Step 2: Set up source descriptor (Px1 grid, row-block layout) ---
  // Source: each rank owns ns_local rows x N cols. MB=ns_local, NB=N.
  int desc_src[9];
  int src_lld = std::max(1, ns_local);
  InitDescriptor(desc_src, n, n, ns_local, n, src_ctx.ictxt, src_lld);

  // --- Step 3: Set up target descriptor (2D grid, block-cyclic) ---
  int local_rows_bc = LocalBlockCyclicDim(n, nb, tgt_ctx.myrow, tgt_ctx.nprow);
  int local_cols_bc = LocalBlockCyclicDim(n, nb, tgt_ctx.mycol, tgt_ctx.npcol);

  int desc_tgt[9];
  int tgt_lld = std::max(1, local_rows_bc);
  InitDescriptor(desc_tgt, n, n, nb, nb, tgt_ctx.ictxt, tgt_lld);

  // --- Step 4: Redistribute column-major row-block -> block-cyclic ---
  std::vector<double> T_bc(static_cast<size_t>(local_rows_bc) * local_cols_bc, 0.0);
  const int one = 1;
  // Union context: both src_ctx (Px1) and tgt_ctx (2D grid) cover the same P
  // processes in comm, so tgt_ctx.ictxt is a valid union context for pdgemr2d_.
  pdgemr2d_(&n, &n,
            T_src_colmaj.data(), &one, &one, desc_src,
            T_bc.data(), &one, &one, desc_tgt,
            &tgt_ctx.ictxt);

  // --- Step 5: pdsyev_ eigendecomposition ---
  std::vector<double> eigenvalues(N);
  std::vector<double> Z_bc(static_cast<size_t>(local_rows_bc) * local_cols_bc, 0.0);

  int desc_z[9];
  InitDescriptor(desc_z, n, n, nb, nb, tgt_ctx.ictxt, tgt_lld);

  // Workspace query
  double work_query = 0.0;
  int lwork = -1;
  int info = 0;
  const char jobz = 'V', uplo = 'U';
  pdsyev_(&jobz, &uplo, &n,
          T_bc.data(), &one, &one, desc_tgt,
          eigenvalues.data(),
          Z_bc.data(), &one, &one, desc_z,
          &work_query, &lwork, &info);
  if (info != 0) {
    throw std::runtime_error("pdsyev_ workspace query failed, info = " + std::to_string(info));
  }

  lwork = static_cast<int>(work_query) + 1;
  std::vector<double> work(lwork);
  pdsyev_(&jobz, &uplo, &n,
          T_bc.data(), &one, &one, desc_tgt,
          eigenvalues.data(),
          Z_bc.data(), &one, &one, desc_z,
          work.data(), &lwork, &info);
  if (info != 0) {
    throw std::runtime_error("pdsyev_ failed with info = " + std::to_string(info));
  }

  // --- Step 6: Pseudo-inverse cutoff ---
  auto lambda_plus = ApplyPseudoInverseCutoff(
      std::vector<double>(eigenvalues.begin(), eigenvalues.end()),
      r_pinv, a_pinv, soft_cutoff);

  // --- Step 7: Distributed matvec y = Z * diag(lambda_plus) * Z^T * rhs ---
  // Z_bc is block-cyclic (column-major local storage).
  // Z_bc[local_i + local_j * tgt_lld] = Z[global_i, global_j]

  // Step 7a: tmp[k] = sum_i Z[i,k] * rhs[i]  (Z^T * rhs)
  std::vector<double> local_tmp(N, 0.0);
  for (int lj = 0; lj < local_cols_bc; ++lj) {
    int gj = BlockCyclicLocalToGlobal(lj, nb, tgt_ctx.mycol, tgt_ctx.npcol);
    for (int li = 0; li < local_rows_bc; ++li) {
      int gi = BlockCyclicLocalToGlobal(li, nb, tgt_ctx.myrow, tgt_ctx.nprow);
      local_tmp[gj] += Z_bc[li + lj * tgt_lld] * rhs[gi];
    }
  }
  std::vector<double> tmp(N, 0.0);
  MPI_Allreduce(local_tmp.data(), tmp.data(), static_cast<int>(N),
                MPI_DOUBLE, MPI_SUM, comm);

  // Step 7b: tmp[k] *= lambda_plus[k]
  for (size_t k = 0; k < N; ++k) {
    tmp[k] *= lambda_plus[k];
  }

  // Step 7c: y[i] = sum_k Z[i,k] * tmp[k]  (Z * scaled_tmp)
  std::vector<double> local_y(N, 0.0);
  for (int lj = 0; lj < local_cols_bc; ++lj) {
    int gj = BlockCyclicLocalToGlobal(lj, nb, tgt_ctx.mycol, tgt_ctx.npcol);
    double tmp_gj = tmp[gj];
    for (int li = 0; li < local_rows_bc; ++li) {
      int gi = BlockCyclicLocalToGlobal(li, nb, tgt_ctx.myrow, tgt_ctx.nprow);
      local_y[gi] += Z_bc[li + lj * tgt_lld] * tmp_gj;
    }
  }
  std::vector<double> y(N, 0.0);
  MPI_Allreduce(local_y.data(), y.data(), static_cast<int>(N),
                MPI_DOUBLE, MPI_SUM, comm);

  return y;
}

// =============================================================================
// Section E: Distributed eigensolve — complex Hermitian case
// =============================================================================

/**
 * @brief Path A distributed eigensolve for complex Hermitian T.
 *
 * Same algorithm as real case, with differences:
 * - Local row-block must be transposed to column-major before pzgemr2d_
 *   (complex Hermitian: row-major != col-major).
 * - Uses pzheev_ (needs extra rwork array).
 * - Back-multiply uses conj(Z[i,k]) for Z^H step.
 * - MPI type: MPI_CXX_DOUBLE_COMPLEX.
 */
inline std::vector<std::complex<double>> ScaLAPACKEigenSolveComplex(
    const std::complex<double>* T_rowblock, size_t local_rows, size_t N,
    const std::vector<std::complex<double>>& rhs, MPI_Comm comm,
    double r_pinv, double a_pinv, bool soft_cutoff) {

  if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("ScaLAPACKEigenSolveComplex: N exceeds MPI/LAPACK int limit");
  }

  using ComplexT = std::complex<double>;

  int nprocs, my_rank;
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &my_rank);

  const int n = static_cast<int>(N);
  const int ns_local = static_cast<int>(local_rows);

  // --- Transpose local row-block to column-major ---
  // Row-major: T_rowblock[i*N + j], Column-major: T_src_colmaj[j*ns_local + i]
  // For complex Hermitian, row-major != col-major, so explicit transpose needed.
  std::vector<ComplexT> T_src_colmaj(static_cast<size_t>(ns_local) * N);
  for (int i = 0; i < ns_local; ++i) {
    for (size_t j = 0; j < N; ++j) {
      T_src_colmaj[j * ns_local + i] = T_rowblock[i * N + j];
    }
  }

  // --- Create BLACS contexts ---
  ScaLAPACKContext src_ctx;
  src_ctx.InitColumn(comm, nprocs);

  ScaLAPACKContext tgt_ctx;
  tgt_ctx.Init(comm, nprocs, my_rank);

  const int nb = tgt_ctx.nb;

  // --- Source descriptor (Px1 grid) ---
  int desc_src[9];
  int src_lld = std::max(1, ns_local);
  InitDescriptor(desc_src, n, n, ns_local, n, src_ctx.ictxt, src_lld);

  // --- Target descriptor (2D grid, block-cyclic) ---
  int local_rows_bc = LocalBlockCyclicDim(n, nb, tgt_ctx.myrow, tgt_ctx.nprow);
  int local_cols_bc = LocalBlockCyclicDim(n, nb, tgt_ctx.mycol, tgt_ctx.npcol);

  int desc_tgt[9];
  int tgt_lld = std::max(1, local_rows_bc);
  InitDescriptor(desc_tgt, n, n, nb, nb, tgt_ctx.ictxt, tgt_lld);

  // --- Redistribute ---
  std::vector<ComplexT> T_bc(static_cast<size_t>(local_rows_bc) * local_cols_bc,
                             ComplexT(0));
  const int one = 1;
  // Union context: both src_ctx (Px1) and tgt_ctx (2D grid) cover the same P
  // processes in comm, so tgt_ctx.ictxt is a valid union context for pzgemr2d_.
  pzgemr2d_(&n, &n,
            T_src_colmaj.data(), &one, &one, desc_src,
            T_bc.data(), &one, &one, desc_tgt,
            &tgt_ctx.ictxt);

  // --- pzheev_ eigendecomposition ---
  std::vector<double> eigenvalues(N);
  std::vector<ComplexT> Z_bc(static_cast<size_t>(local_rows_bc) * local_cols_bc,
                             ComplexT(0));

  int desc_z[9];
  InitDescriptor(desc_z, n, n, nb, nb, tgt_ctx.ictxt, tgt_lld);

  // Workspace query
  ComplexT work_query(0);
  int lwork = -1;
  double rwork_query = 0.0;
  int lrwork = -1;
  int info = 0;
  const char jobz = 'V', uplo = 'U';

  pzheev_(&jobz, &uplo, &n,
          T_bc.data(), &one, &one, desc_tgt,
          eigenvalues.data(),
          Z_bc.data(), &one, &one, desc_z,
          &work_query, &lwork, &rwork_query, &lrwork, &info);
  if (info != 0) {
    throw std::runtime_error("pzheev_ workspace query failed, info = " + std::to_string(info));
  }

  lwork = static_cast<int>(work_query.real()) + 1;
  lrwork = static_cast<int>(rwork_query) + 1;
  std::vector<ComplexT> work(lwork);
  std::vector<double> rwork(lrwork);

  pzheev_(&jobz, &uplo, &n,
          T_bc.data(), &one, &one, desc_tgt,
          eigenvalues.data(),
          Z_bc.data(), &one, &one, desc_z,
          work.data(), &lwork, rwork.data(), &lrwork, &info);
  if (info != 0) {
    throw std::runtime_error("pzheev_ failed with info = " + std::to_string(info));
  }

  // --- Pseudo-inverse cutoff ---
  auto lambda_plus = ApplyPseudoInverseCutoff(
      std::vector<double>(eigenvalues.begin(), eigenvalues.end()),
      r_pinv, a_pinv, soft_cutoff);

  // --- Distributed matvec y = Z * diag(lambda_plus) * Z^H * rhs ---

  // Step 1: tmp[k] = sum_i conj(Z[i,k]) * rhs[i]  (Z^H * rhs)
  std::vector<ComplexT> local_tmp(N, ComplexT(0));
  for (int lj = 0; lj < local_cols_bc; ++lj) {
    int gj = BlockCyclicLocalToGlobal(lj, nb, tgt_ctx.mycol, tgt_ctx.npcol);
    for (int li = 0; li < local_rows_bc; ++li) {
      int gi = BlockCyclicLocalToGlobal(li, nb, tgt_ctx.myrow, tgt_ctx.nprow);
      local_tmp[gj] += std::conj(Z_bc[li + lj * tgt_lld]) * rhs[gi];
    }
  }
  std::vector<ComplexT> tmp(N, ComplexT(0));
  MPI_Allreduce(local_tmp.data(), tmp.data(), static_cast<int>(N),
                MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm);

  // Step 2: tmp[k] *= lambda_plus[k]
  for (size_t k = 0; k < N; ++k) {
    tmp[k] *= lambda_plus[k];
  }

  // Step 3: y[i] = sum_k Z[i,k] * tmp[k]  (Z * scaled_tmp)
  std::vector<ComplexT> local_y(N, ComplexT(0));
  for (int lj = 0; lj < local_cols_bc; ++lj) {
    int gj = BlockCyclicLocalToGlobal(lj, nb, tgt_ctx.mycol, tgt_ctx.npcol);
    ComplexT tmp_gj = tmp[gj];
    for (int li = 0; li < local_rows_bc; ++li) {
      int gi = BlockCyclicLocalToGlobal(li, nb, tgt_ctx.myrow, tgt_ctx.nprow);
      local_y[gi] += Z_bc[li + lj * tgt_lld] * tmp_gj;
    }
  }
  std::vector<ComplexT> y(N, ComplexT(0));
  MPI_Allreduce(local_y.data(), y.data(), static_cast<int>(N),
                MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm);

  return y;
}

// =============================================================================
// Section F: Template dispatcher
// =============================================================================

/**
 * @brief ScaLAPACK distributed eigensolve with pseudo-inverse cutoff.
 *
 * Dispatches to real (pdsyev) or complex (pzheev) based on TenElemT.
 * Reduces per-rank memory from O(Ns^2) to O(Ns^2/P).
 *
 * @param T_rowblock  Local row-block data (ns_local x N, row-major)
 * @param local_rows  Number of local rows owned by this rank
 * @param N           Global matrix dimension (Ns)
 * @param rhs         Right-hand side vector (length N)
 * @param comm        MPI communicator
 * @param r_pinv      Relative pseudo-inverse cutoff
 * @param a_pinv      Absolute pseudo-inverse cutoff
 * @param soft_cutoff Use soft cutoff
 * @return Solution y (length N, identical on all ranks)
 */
template<typename TenElemT>
std::vector<TenElemT> ScaLAPACKEigenSolveWithPseudoInverse(
    const TenElemT* T_rowblock, size_t local_rows, size_t N,
    const std::vector<TenElemT>& rhs, MPI_Comm comm,
    double r_pinv, double a_pinv, bool soft_cutoff) {
  if constexpr (std::is_same_v<TenElemT, double>) {
    return ScaLAPACKEigenSolveReal(
        T_rowblock, local_rows, N, rhs, comm, r_pinv, a_pinv, soft_cutoff);
  } else if constexpr (std::is_same_v<TenElemT, std::complex<double>>) {
    return ScaLAPACKEigenSolveComplex(
        T_rowblock, local_rows, N, rhs, comm, r_pinv, a_pinv, soft_cutoff);
  } else {
    static_assert(sizeof(TenElemT) == 0, "Unsupported TenElemT for ScaLAPACK eigensolve");
  }
}

} // namespace minsr_detail
} // namespace qlpeps

#endif // QLPEPS_HAS_SCALAPACK

#endif // QLPEPS_OPTIMIZER_MINSR_SCALAPACK_H
