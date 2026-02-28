// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-17
*
* Description: QuantumLiquids/PEPS project. Implementation for conjugate gradient solver
*/

#ifndef QLPEPS_VMC_PEPS_CONJUGATE_GRADIENT_SOLVER_H
#define QLPEPS_VMC_PEPS_CONJUGATE_GRADIENT_SOLVER_H

#include <algorithm>  // std::max
#include <cmath>      // std::sqrt
#include <complex>    // std::complex
#include <concepts>   // std::convertible_to, std::same_as
#include <cstddef>    // size_t
#include <iostream>   // cout, endl
#include <limits>     // std::numeric_limits
#include <type_traits>  // std::remove_cvref_t
#include <utility>      // std::declval
#include <vector>
#include "mpi.h"
#include "qlpeps/optimizer/optimizer_params.h"  // ConjugateGradientParams
#ifndef  NDEBUG

#include <cassert>

#endif

#ifdef QLPEPS_TIMING_MODE

#include "qlten/utility/timer.h"
#endif
#include "qlten/framework/hp_numeric/mpi_fun.h"

namespace qlpeps {
using qlten::hp_numeric::kMPIMasterRank;
namespace hp_numeric = qlten::hp_numeric;

template<typename InnerType>
concept CGInnerProductType =
    std::same_as<std::remove_cvref_t<InnerType>, double> ||
    std::same_as<std::remove_cvref_t<InnerType>, std::complex<double>>;

/**
 * @brief Compile-time requirements for the vector type used by CG.
 *
 * Required operations:
 * - NormSquare(): returns squared 2-norm ||v||^2 = v^\dagger v
 * - Inner product: u * v returns v-space scalar
 * - Linear algebra: u + v, u - v, and u += v
 * - Scalar-vector multiply: alpha * v and beta * v
 */
template<typename VectorType>
concept CGVectorType = requires(
    const VectorType &lhs,
    const VectorType &rhs,
    VectorType v,
    const double beta
) {
  { lhs.NormSquare() } -> std::convertible_to<double>;
  { lhs * rhs } -> CGInnerProductType;
  { lhs + rhs } -> std::convertible_to<VectorType>;
  { lhs - rhs } -> std::convertible_to<VectorType>;
  { v += rhs } -> std::convertible_to<VectorType>;
  { beta * lhs } -> std::convertible_to<VectorType>;
  { (std::declval<double>() / (lhs * rhs)) * lhs } -> std::convertible_to<VectorType>;
};

/**
 * @brief Reason the CG solver terminated.
 */
enum class CGTerminationReason {
  kConverged,           ///< ||r|| <= tolerance
  kMaxIterations,       ///< Hit max_iter without convergence
  kIndefiniteMatrix,    ///< p^T A p <= 0 detected
  kNumericalBreakdown,  ///< NaN or Inf detected in residual, alpha, or beta
  kStagnated            ///< Step norm < eps * ||x|| for 3 consecutive iterations
};

inline const char* to_string(CGTerminationReason reason) {
  switch (reason) {
    case CGTerminationReason::kConverged:          return "kConverged";
    case CGTerminationReason::kMaxIterations:      return "kMaxIterations";
    case CGTerminationReason::kIndefiniteMatrix:   return "kIndefiniteMatrix";
    case CGTerminationReason::kNumericalBreakdown: return "kNumericalBreakdown";
    case CGTerminationReason::kStagnated:          return "kStagnated";
    default:                                       return "Unknown";
  }
}

/**
 * @brief Result of conjugate gradient solver.
 *
 * @tparam VectorType The vector type used in the linear system.
 */
template<typename VectorType>
struct CGResult {
  VectorType x;                    ///< Solution (best iterate when not converged)
  double residual_norm;            ///< Final residual 2-norm ||r||
  size_t iterations;               ///< Number of CG iterations performed
  CGTerminationReason reason;      ///< Why CG stopped

  bool converged() const { return reason == CGTerminationReason::kConverged; }
};

/**
 * @brief Compile-time requirements for the matrix/operator type used by CG.
 */
template<typename MatrixType, typename VectorType>
concept CGMatrixType = CGVectorType<VectorType> &&
    requires(const MatrixType &matrix_a, const VectorType &v) {
      { matrix_a * v } -> std::convertible_to<VectorType>;
    };

/**
 * @brief Compile-time requirements for MPI communication hooks used by parallel CG.
 *
 * Required user-provided free functions (via ADL):
 * - MPI_Bcast(VectorType&, MPI_Comm)
 * - MPI_Send(const VectorType&, int, MPI_Comm, int)
 * - MPI_Recv(VectorType&, int, MPI_Comm, int)
 */
template<typename VectorType>
concept CGMPICommunicationVectorType =
    CGVectorType<VectorType> && std::default_initializable<VectorType> &&
    requires(
        VectorType &v,
        const VectorType &const_v,
        const MPI_Comm &comm,
        const int peer,
        const int tag
    ) {
      { MPI_Bcast(v, comm) } -> std::same_as<void>;
      { MPI_Send(const_v, peer, comm, tag) } -> std::same_as<void>;
      { MPI_Recv(v, peer, comm, tag) } -> std::same_as<MPI_Status>;
    };

namespace detail {

inline bool pap_is_valid(double pap) {
  return pap > 0.0;
}

inline bool pap_is_valid(std::complex<double> pap) {
  return pap.real() > 0.0 && std::abs(pap.imag()) < 1e-10;
}

inline double real_part(double x) { return x; }
inline double real_part(std::complex<double> x) { return x.real(); }

inline double abs_sq(double x) { return x * x; }
inline double abs_sq(std::complex<double> x) { return std::norm(x); }

}  // namespace detail

/**
 * @brief Serial conjugate gradient solver.
 *
 * Solves the linear system A * x = b where A is a self-adjoint positive-definite
 * matrix/operator, using the conjugate gradient method.
 *
 * Features:
 * - Best-iterate tracking: non-converged returns use the x with smallest residual norm
 * - NaN/Inf detection: terminates with kNumericalBreakdown
 * - Stagnation detection: terminates with kStagnated after 3 consecutive no-progress steps
 * - Orthogonality-based restart: resets search direction when residuals lose orthogonality
 * - Periodic residual recomputation: reduces floating-point drift
 *
 * Reference: https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *
 * @tparam MatrixType Type satisfying CGMatrixType (matrix_a * vector operation).
 * @tparam VectorType Type satisfying CGVectorType.
 * @param matrix_a Self-adjoint positive-definite matrix/operator
 * @param b Right-hand side vector
 * @param x0 Initial guess
 * @param params CG solver parameters (tolerances, max iterations, etc.)
 * @return CGResult containing solution, residual norm, iteration count, and termination reason
 */
template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType>
CGResult<VectorType> ConjugateGradientSolver(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0,
    const ConjugateGradientParams &params) {
  const size_t max_iter = params.max_iter;
  const double relative_tolerance = params.relative_tolerance;
  const double absolute_tolerance = params.absolute_tolerance;
  const int residual_recompute_interval = params.residual_recompute_interval;
  const double orthogonality_threshold = params.orthogonality_threshold;

  const double rhs_norm_sq = b.NormSquare();
  const double tol_sq = std::max(relative_tolerance * relative_tolerance * rhs_norm_sq,
                                 absolute_tolerance * absolute_tolerance);
  VectorType r = b - matrix_a * x0;
  double r_norm_sq = r.NormSquare();
  if (r_norm_sq <= tol_sq) {
    return {x0, std::sqrt(r_norm_sq), 0, CGTerminationReason::kConverged};
  }

  VectorType p = r;
  VectorType x = x0;
  VectorType best_x = x0;
  double best_residual_norm_sq = r_norm_sq;
  VectorType r_prev = r;  // copy-init avoids requiring default construction
  double rkp1_2norm = r_norm_sq;
  size_t stagnation_count = 0;
  constexpr double eps = std::numeric_limits<double>::epsilon();

  for (size_t k = 0; k < max_iter; k++) {
    double rk_2norm = rkp1_2norm;
    VectorType ap = matrix_a * p;
    auto pap = p * ap;
    if (!detail::pap_is_valid(pap)) {
      return {best_x, std::sqrt(best_residual_norm_sq), k, CGTerminationReason::kIndefiniteMatrix};
    }
    auto alpha = rk_2norm / pap;
    x = x + alpha * p;

    // Stagnation detection
    double alpha_abs_sq = detail::abs_sq(alpha);
    double step_norm_sq = alpha_abs_sq * p.NormSquare();
    double x_norm_sq = x.NormSquare();
    if (step_norm_sq < eps * eps * x_norm_sq) {
      if (++stagnation_count >= 3) {
        return {best_x, std::sqrt(best_residual_norm_sq), k + 1, CGTerminationReason::kStagnated};
      }
    } else {
      stagnation_count = 0;
    }

    if (residual_recompute_interval > 0 && (k % residual_recompute_interval) == (residual_recompute_interval - 1)) {
      r = b - matrix_a * x;
    } else {
      r = r - alpha * ap;
    }
    rkp1_2norm = r.NormSquare();

    // NaN/Inf detection
    if (!std::isfinite(rkp1_2norm)) {
      return {best_x, std::sqrt(best_residual_norm_sq), k + 1, CGTerminationReason::kNumericalBreakdown};
    }

    // Best-iterate tracking
    if (rkp1_2norm < best_residual_norm_sq) {
      best_x = x;
      best_residual_norm_sq = rkp1_2norm;
    }

    if (rkp1_2norm <= tol_sq) {
      return {x, std::sqrt(rkp1_2norm), k + 1, CGTerminationReason::kConverged};
    }

    // Orthogonality-based restart
    if (k > 0) {
      double ortho = std::abs(detail::real_part(r_prev * r));
      if (ortho > orthogonality_threshold * rkp1_2norm) {
        p = r;
        r_prev = r;
        continue;
      }
    }
    r_prev = r;

    double beta = rkp1_2norm / rk_2norm;
    if (!std::isfinite(beta)) {
      return {best_x, std::sqrt(best_residual_norm_sq), k + 1, CGTerminationReason::kNumericalBreakdown};
    }
    p = r + beta * p;
  }
  std::cout << "warning: convergence may fail in conjugate gradient solver. residual_norm = "
            << std::scientific << std::sqrt(rkp1_2norm) << std::endl;
  return {best_x, std::sqrt(best_residual_norm_sq), max_iter, CGTerminationReason::kMaxIterations};
}

//forward declaration

template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
void MatrixMultiplyVectorSlave(
    const MatrixType &mat,
    const MPI_Comm &comm
);

template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
VectorType MatrixMultiplyVectorMaster(
    const MatrixType &mat,
    const VectorType &v,
    const MPI_Comm &comm
);

template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
void ConjugateGradientSolverSlave(
    const MatrixType &matrix_a,
    const MPI_Comm &comm
);

template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
CGResult<VectorType> ConjugateGradientSolverMaster(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0,
    const ConjugateGradientParams &params,
    const MPI_Comm &comm
);

/**
 * @brief User-defined MPI communication functions required for VectorType.
 *
 * For parallel conjugate gradient solver to work, users must provide the following
 * MPI communication functions for their VectorType:
 *
 * @code
 * template<typename VectorType>
 * void MPI_Bcast(VectorType &vector, const MPI_Comm &comm);
 *
 * template<typename VectorType>
 * void MPI_Send(const VectorType &vector, int dest, const MPI_Comm &comm, int tag = 0);
 *
 * template<typename VectorType>
 * MPI_Status MPI_Recv(VectorType &vector, int src, const MPI_Comm &comm, int tag = 0);
 * @endcode
 *
 * These functions must handle:
 * - Broadcasting vector data from master to all ranks
 * - Sending vector data to specified destination rank
 * - Receiving vector data from specified source rank (supports MPI_ANY_SOURCE)
 */

/**
 * @brief Parallel conjugate gradient solver
 *
 * Solves the linear system A * x = b using the conjugate gradient method
 * in a distributed MPI environment. The matrix A is distributed across processors.
 *
 * @warning Result distribution behavior:
 *   - Master rank (rank 0): Returns the actual CGResult with solution
 *   - Slave ranks: Return a placeholder CGResult with x0
 *   - If you need the solution on all ranks, manually broadcast after calling this function
 *
 * @tparam MatrixType Type satisfying CGMatrixType (matrix_a * vector operation).
 * @tparam VectorType Type satisfying CGMPICommunicationVectorType.
 * @param matrix_a Self-adjoint matrix/operator (distributed)
 * @param b Right-hand side vector
 * @param x0 Initial guess
 * @param params CG solver parameters (tolerances, max iterations, etc.)
 * @param comm MPI communicator
 * @return CGResult containing solution, residual norm, iteration count, and termination reason
 */
template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
CGResult<VectorType> ConjugateGradientSolver(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0,
    const ConjugateGradientParams &params,
    const MPI_Comm &comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    return ConjugateGradientSolverMaster(
        matrix_a, b, x0, params, comm
    );
  } else {
    ConjugateGradientSolverSlave<MatrixType, VectorType>(
        matrix_a, comm
    );
    return {x0, 0.0, 0, CGTerminationReason::kMaxIterations};  // Slave returns placeholder
  }
}

enum ConjugateGradientSolverInstruction {
  start,
  multiplication,
  finish
};

inline void MasterBroadcastInstruction(
    const ConjugateGradientSolverInstruction instruction,
    const MPI_Comm &comm) {
  HANDLE_MPI_ERROR(::MPI_Bcast(const_cast<ConjugateGradientSolverInstruction *>(&instruction),
                               1,
                               MPI_INT,
                               qlten::hp_numeric::kMPIMasterRank,
                               comm));
}

inline ConjugateGradientSolverInstruction
SlaveReceiveBroadcastInstruction(const MPI_Comm &comm) {
  ConjugateGradientSolverInstruction instruction;
  HANDLE_MPI_ERROR(::MPI_Bcast(&instruction,
                               1,
                               MPI_INT,
                               qlten::hp_numeric::kMPIMasterRank,
                               comm));
  return instruction;
}

/**
 * @brief MPI master-side conjugate gradient solver.
 *
 * Same algorithm as the serial solver, but uses distributed matrix-vector products
 * via MatrixMultiplyVectorMaster. Every early-return path broadcasts a finish
 * instruction to prevent slave deadlocks.
 */
template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
CGResult<VectorType> ConjugateGradientSolverMaster(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0,
    const ConjugateGradientParams &params,
    const MPI_Comm &comm
) {
  const size_t max_iter = params.max_iter;
  const double relative_tolerance = params.relative_tolerance;
  const double absolute_tolerance = params.absolute_tolerance;
  const int residual_recompute_interval = params.residual_recompute_interval;
  const double orthogonality_threshold = params.orthogonality_threshold;

  MasterBroadcastInstruction(start, comm);

  const double rhs_norm_sq = b.NormSquare();
  const double tol_sq = std::max(relative_tolerance * relative_tolerance * rhs_norm_sq,
                                 absolute_tolerance * absolute_tolerance);

  VectorType ax0 = MatrixMultiplyVectorMaster(matrix_a, x0, comm);
  VectorType r = b - ax0;
  double r_norm_sq = r.NormSquare();
  if (r_norm_sq <= tol_sq) {
    MasterBroadcastInstruction(finish, comm);
    return {x0, std::sqrt(r_norm_sq), 0, CGTerminationReason::kConverged};
  }
  VectorType p = r;
  VectorType x = x0;
  VectorType best_x = x0;
  double best_residual_norm_sq = r_norm_sq;
  VectorType r_prev;
  double rkp1_2norm = r_norm_sq;
  size_t stagnation_count = 0;
  constexpr double eps = std::numeric_limits<double>::epsilon();

  for (size_t k = 0; k < max_iter; k++) {
    double rk_2norm = rkp1_2norm;
    MasterBroadcastInstruction(multiplication, comm);
    VectorType ap = MatrixMultiplyVectorMaster(matrix_a, p, comm);
    auto pap = (p * ap);
    if (!detail::pap_is_valid(pap)) {
      MasterBroadcastInstruction(finish, comm);
      return {best_x, std::sqrt(best_residual_norm_sq), k, CGTerminationReason::kIndefiniteMatrix};
    }
    auto alpha = rk_2norm / pap; //auto is double or complex
    x += alpha * p;

    // Stagnation detection
    double alpha_abs_sq = detail::abs_sq(alpha);
    double step_norm_sq = alpha_abs_sq * p.NormSquare();
    double x_norm_sq = x.NormSquare();
    if (step_norm_sq < eps * eps * x_norm_sq) {
      if (++stagnation_count >= 3) {
        MasterBroadcastInstruction(finish, comm);
        return {best_x, std::sqrt(best_residual_norm_sq), k + 1, CGTerminationReason::kStagnated};
      }
    } else {
      stagnation_count = 0;
    }

    if (residual_recompute_interval > 0 && (k % residual_recompute_interval) == (residual_recompute_interval - 1)) {
      MasterBroadcastInstruction(multiplication, comm);
      VectorType ax = MatrixMultiplyVectorMaster(matrix_a, x, comm);
      r = b - ax;
    } else {
      r += (-alpha) * ap;
    }
    rkp1_2norm = r.NormSquare();

    // NaN/Inf detection
    if (!std::isfinite(rkp1_2norm)) {
      MasterBroadcastInstruction(finish, comm);
      return {best_x, std::sqrt(best_residual_norm_sq), k + 1, CGTerminationReason::kNumericalBreakdown};
    }

    // Best-iterate tracking
    if (rkp1_2norm < best_residual_norm_sq) {
      best_x = x;
      best_residual_norm_sq = rkp1_2norm;
    }

    if (rkp1_2norm <= tol_sq) {
      MasterBroadcastInstruction(finish, comm);
      return {x, std::sqrt(rkp1_2norm), k + 1, CGTerminationReason::kConverged};
    }

    // Orthogonality-based restart
    if (k > 0) {
      double ortho = std::abs(detail::real_part(r_prev * r));
      if (ortho > orthogonality_threshold * rkp1_2norm) {
        p = r;
        r_prev = r;
        rk_2norm = rkp1_2norm;
        continue;
      }
    }
    r_prev = r;

    double beta = rkp1_2norm / rk_2norm;
    if (!std::isfinite(beta)) {
      MasterBroadcastInstruction(finish, comm);
      return {best_x, std::sqrt(best_residual_norm_sq), k + 1, CGTerminationReason::kNumericalBreakdown};
    }
#if VERBOSE_MODE == 1
    std::cout << "k = " << k << "\t residual norm = " << std::scientific << rkp1_2norm
                << "\t beta = " << std::fixed << beta << "."
                << std::endl;
#endif
    p = r + beta * p;
  }
  std::cout << "warning: convergence may fail in conjugate gradient solver. residual_norm = "
            << std::scientific << std::sqrt(rkp1_2norm) << std::endl;
  MasterBroadcastInstruction(finish, comm);
  return {best_x, std::sqrt(best_residual_norm_sq), max_iter, CGTerminationReason::kMaxIterations};
}

template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
void ConjugateGradientSolverSlave(
    const MatrixType &matrix_a,
    const MPI_Comm &comm
) {
  auto instrct = SlaveReceiveBroadcastInstruction(comm);
  assert(instrct == start);
  MatrixMultiplyVectorSlave<MatrixType, VectorType>(matrix_a, comm);
  instrct = SlaveReceiveBroadcastInstruction(comm);
  while (instrct != finish) {
    //instrct == multiplication
    MatrixMultiplyVectorSlave<MatrixType, VectorType>(matrix_a, comm);
    instrct = SlaveReceiveBroadcastInstruction(comm);
  }
}

template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
VectorType MatrixMultiplyVectorMaster(
    const MatrixType &mat,
    const VectorType &v,
    const MPI_Comm &comm
) {
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);
#ifdef QLPEPS_TIMING_MODE
  qlten::Timer cg_mat_vec_mult_timer("conjugate gradient matrix vector multiplication");
  qlten::Timer cg_broadcast_vec_timer("conjugate gradient broadcast vector");
#endif
  MPI_Bcast(const_cast<VectorType &>(v), comm); //defined by user
  #ifdef QLPEPS_MPI_DEBUG
  int dbg_rank_master = -1; MPI_Comm_rank(comm, &dbg_rank_master);
  if (dbg_rank_master == qlten::hp_numeric::kMPIMasterRank) {
    std::cerr << "[MPI DEBUG][CG] master broadcasted vector" << std::endl;
  }
  #endif
#ifdef QLPEPS_TIMING_MODE
  cg_broadcast_vec_timer.PrintElapsed();
#endif
  std::vector<VectorType> res_list(mpi_size);
  VectorType res = mat * v;
#ifdef QLPEPS_TIMING_MODE
  qlten::Timer cg_gather_vec_timer("conjugate gradient gather vector");
#endif
  for (size_t i = 1; i < mpi_size; i++) {
    #ifdef QLPEPS_MPI_DEBUG
    std::cerr << "[MPI DEBUG][CG] master waiting recv i=" << i << " tag=0 from ANY" << std::endl;
    #endif
    MPI_Recv(res_list[i], MPI_ANY_SOURCE, comm, 0);
  }
#ifdef QLPEPS_TIMING_MODE
  cg_gather_vec_timer.PrintElapsed();
  qlten::Timer cg_gather_reduce_vec_timer("conjugate gradient gather reduce (summation) vector");
#endif
  for (size_t i = 1; i < mpi_size; i++) {
    res += res_list[i];
  }
#ifdef QLPEPS_TIMING_MODE
  cg_gather_reduce_vec_timer.PrintElapsed();
  cg_mat_vec_mult_timer.PrintElapsed();
#endif
  return res;
}

template<typename MatrixType, typename VectorType>
requires CGMatrixType<MatrixType, VectorType> && CGMPICommunicationVectorType<VectorType>
void MatrixMultiplyVectorSlave(
    const MatrixType &mat,
    const MPI_Comm &comm
) {
  VectorType v;
  MPI_Bcast(v, comm);
  #ifdef QLPEPS_MPI_DEBUG
  int dbg_rank_slave = -1; MPI_Comm_rank(comm, &dbg_rank_slave);
  std::cerr << "[MPI DEBUG][CG] rank " << dbg_rank_slave << " received broadcast vector" << std::endl;
  #endif
  VectorType res = mat * v;
  #ifdef QLPEPS_MPI_DEBUG
  std::cerr << "[MPI DEBUG][CG] rank " << dbg_rank_slave << " sending result to master tag=0" << std::endl;
  #endif
  MPI_Send(res, qlten::hp_numeric::kMPIMasterRank, comm, 0);
}

}//qlpeps

#endif //QLPEPS_VMC_PEPS_CONJUGATE_GRADIENT_SOLVER_H
