// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-17
*
* Description: QuantumLiquids/PEPS project. Implementation for conjugate gradient solver
*/

#ifndef QLPEPS_VMC_PEPS_CONJUGATE_GRADIENT_SOLVER_H
#define QLPEPS_VMC_PEPS_CONJUGATE_GRADIENT_SOLVER_H

#include <cstddef>   //size_t
#include <iostream>  //cout, endl
#include "mpi.h"
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
/**
 * solve the equation
 *          A * x = b
 * where A is self-conjugated matrix/operator, denoted by matrix_a.
 *
 * code write according to https://en.wikipedia.org/wiki/Conjugate_gradient_method
 * @tparam MatrixType
 * @tparam VectorType
 *          has the following methods
 *          NormSquare(), return the 2-norm, v^dag*v
 *          operator*(VectorType v2), return v^dag * v2
 * @param matrix_a
 * @param b
 * @param x0
 * @param tolerance
 * @return
 */
template<typename MatrixType, typename VectorType>
VectorType ConjugateGradientSolver(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0, //initial guess
    const size_t max_iter,
    const double tolerance,
    size_t &iter   //return
) {
  VectorType r = b - matrix_a * x0;
  if (r.NormSquare() < tolerance) {
    iter = 0;
    return x0;
  }
  VectorType p = r;
  VectorType x = x0;
  double rkp1_2norm = r.NormSquare();
  for (size_t k = 0; k < max_iter; k++) {
    double rk_2norm = rkp1_2norm;
    VectorType ap = matrix_a * p;
    auto alpha = rk_2norm / (p * ap);
    x = x + alpha * p;
    r = r - alpha * ap;
    rkp1_2norm = r.NormSquare();   // return value of norm has definitely double type.
    if (rkp1_2norm < tolerance) {
      iter = k + 1;
      return x;
    }
    double beta = rkp1_2norm / rk_2norm;
    assert(beta <= 10);
    p = r + beta * p;
  }
  iter = max_iter;
  std::cout << "warning: convergence may fail on gradient solving linear equation. rkp1_2norm = " << std::scientific
            << rkp1_2norm
            << std::endl;
  return x;
}

//forward declaration

template<typename MatrixType, typename VectorType>
void MatrixMultiplyVectorSlave(
    const MatrixType &mat,
    const MPI_Comm &comm
);

template<typename MatrixType, typename VectorType>
VectorType MatrixMultiplyVectorMaster(
    const MatrixType &mat,
    const VectorType &v,
    const MPI_Comm &comm
);

template<typename MatrixType, typename VectorType>
void ConjugateGradientSolverSlave(
    const MatrixType &matrix_a,
    const MPI_Comm &comm
);

template<typename MatrixType, typename VectorType>
VectorType ConjugateGradientSolverMaster(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0, //initial guess
    size_t max_iter,
    double tolerance,
    int,
    size_t &iter,
    const MPI_Comm &comm
);

/**
 * @brief User-defined MPI communication functions required for VectorType
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
 *   - Master rank (rank 0): Returns the actual solution vector
 *   - Slave ranks: Return the initial guess x0 (unchanged)
 *   - If you need the solution on all ranks, manually broadcast after calling this function
 * 
 * @tparam MatrixType Matrix type supporting operator*(VectorType)
 * @tparam VectorType Vector type with required MPI communication functions
 * @param matrix_a Self-adjoint matrix/operator (distributed)
 * @param b Right-hand side vector
 * @param x0 Initial guess
 * @param max_iter Maximum number of iterations
 * @param tolerance Convergence tolerance
 * @param residue_restart_step Residue restart interval (0 to disable)
 * @param iter [out] Number of iterations performed (valid on master rank only)
 * @param comm MPI communicator
 * @return Solution vector on master rank, initial guess x0 on slave ranks
 */
template<typename MatrixType, typename VectorType>
VectorType ConjugateGradientSolver(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0, //initial guess
    const size_t max_iter,
    const double tolerance,
    const int residue_restart_step,
    size_t &iter,    //return value
    const MPI_Comm &comm
) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == qlten::hp_numeric::kMPIMasterRank) {
    return ConjugateGradientSolverMaster(
        matrix_a, b, x0, max_iter, tolerance, residue_restart_step, iter, comm
    );
  } else {
    ConjugateGradientSolverSlave<MatrixType, VectorType>(
        matrix_a, comm
    );
    return x0;
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

inline bool pap_check(const double &pap) {
  return pap > 0;
}

inline bool pap_check(const std::complex<double> &pap) {
  if (pap.real() <= 0) {
    std::cout << pap.real() << std::endl;
    throw std::runtime_error("pap.real() <= 0");
  }
  if (std::abs(pap.imag()) > 1e-10) {
    std::cout << pap.imag() << std::endl;
    throw std::runtime_error("pap.imag() != 0");
  }
  return pap.real() > 0 && std::abs(pap.imag()) < 1e-10; // Adjust tolerance as needed
}

template<typename MatrixType, typename VectorType>
VectorType ConjugateGradientSolverMaster(
    const MatrixType &matrix_a,
    const VectorType &b,
    const VectorType &x0, //initial guess
    size_t max_iter,
    double tolerance,
    int residue_restart_step,
    size_t &iter,
    const MPI_Comm &comm
) {
  MasterBroadcastInstruction(start, comm);

  double tol = b.NormSquare() * tolerance;

  VectorType ax0 = MatrixMultiplyVectorMaster(matrix_a, x0, comm);
  VectorType r = b - ax0;
  double rk_2norm = r.NormSquare();
  if (rk_2norm < tol) {
    iter = 0;
    MasterBroadcastInstruction(finish, comm);
    return x0;
  }
  VectorType p = r;
  VectorType x = x0;
  double rkp1_2norm;
  for (size_t k = 0; k < max_iter; k++) {
    MasterBroadcastInstruction(multiplication, comm);
    VectorType ap = MatrixMultiplyVectorMaster(matrix_a, p, comm);
    auto pap = (p * ap);
    auto alpha = rk_2norm / pap; //auto is double or complex
#ifndef NDEBUG
    assert(pap_check(pap));
//    if (!std::isnormal(alpha.real())) {
//      std::cout << "k : " << k << "\t pap : " << std::scientific << pap
//                << "\t rk_2norm : " << std::scientific << rk_2norm
//                << "\t alpha : " << std::scientific << alpha << std::endl;
//      exit(1);
//    }
#endif
    x += alpha * p;

    if (residue_restart_step > 0 && (k % residue_restart_step) == (residue_restart_step - 1)) {
      MasterBroadcastInstruction(multiplication, comm);
      VectorType ax = MatrixMultiplyVectorMaster(matrix_a, x, comm);
      r = b - ax;
    } else {
      r += (-alpha) * ap;
    }
    rkp1_2norm = r.NormSquare();   // return value of norm has definitely double type.

    if (rkp1_2norm < tol) {
      iter = k + 1;
      MasterBroadcastInstruction(finish, comm);
      return x;
    }
    double beta = rkp1_2norm / rk_2norm;
#if VERBOSE_MODE == 1
    std::cout << "k = " << k << "\t residue norm = " << std::scientific << rkp1_2norm
                << "\t beta = " << std::fixed << beta << "."
                << std::endl;
#endif
#ifndef NDEBUG
    if (beta > 1.0) {
      std::cout << "k = " << k << "\t residue norm = " << std::scientific << rkp1_2norm
                << "\t pap = " << std::scientific << pap
                << "\t beta = " << std::fixed << beta << "."
                << std::endl;
    }
#endif
    p = r + beta * p;
    rk_2norm = rkp1_2norm;
  }
  iter = max_iter;
  std::cout << "warning: convergence may fail on gradient solving linear equation. rkp1_2norm = " << std::scientific
            << rkp1_2norm
            << std::endl;
  MasterBroadcastInstruction(finish, comm);
  return x;
}

template<typename MatrixType, typename VectorType>
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
