// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-17
*
* Description: GraceQ/VMC-PEPS project. Implementation for conjugate gradient solver
*/

#ifndef GRACEQ_VMC_PEPS_CONJUGATE_GRADIENT_SOLVER_H
#define GRACEQ_VMC_PEPS_CONJUGATE_GRADIENT_SOLVER_H

#include <stddef.h>   //size_t
#include <boost/mpi.hpp>
#include "gqpeps/consts.h"    //kMasterProc

#ifdef GQPEPS_TIMING_MODE

#include "gqten/utility/timer.h"

using gqten::Timer;
#endif

#ifndef  NDEBUG

#include <cmath>

#endif

namespace gqpeps {

/**
 * solve the equation
 *          A * x = b
 * where A is self-conjugated matrix/operator, denoted by matrix_a.
 *
 * code write according to https://en.wikipedia.org/wiki/Conjugate_gradient_method
 * @tparam MatrixType
 * @tparam VectorType
 *          has the following methods
 *          Norm(), return the 2-norm, v^dag*v
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
  if (r.Norm() < tolerance) {
    iter = 0;
    return x0;
  }
  VectorType p = r;
  VectorType x = x0;
  double rkp1_2norm = r.Norm();
  for (size_t k = 0; k < max_iter; k++) {
    double rk_2norm = rkp1_2norm;
    VectorType ap = matrix_a * p;
    auto alpha = rk_2norm / (p * ap);
    x = x + alpha * p;
    r = r - alpha * ap;
    rkp1_2norm = r.Norm();   // return value of norm has definitely double type.
    if (rkp1_2norm < tolerance) {
      iter = k + 1;
      return x;
    }
    double beta = rkp1_2norm / rk_2norm;
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
    boost::mpi::communicator &world
);

template<typename MatrixType, typename VectorType>
VectorType MatrixMultiplyVectorMaster(
    const MatrixType &mat,
    const VectorType &v,
    boost::mpi::communicator &world
);

template<typename MatrixType, typename VectorType>
void ConjugateGradientSolverSlave(
    const MatrixType &matrix_a,
    boost::mpi::communicator &world
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
    boost::mpi::communicator &world
);

// virtual forward declaration
// NB! user should define the following functions by himself/herself
//template<typename VectorType>
//void CGSolverBroadCastVector(
//    VectorType &x0,
//    boost::mpi::communicator &world
//);
//
//template<typename VectorType>
//void CGSolverSendVector(
//    boost::mpi::communicator &world,
//    const VectorType &v,
//    const size_t dest,
//    const int tag
//);
//
//template<typename VectorType>
//size_t CGSolverRecvVector(
//    boost::mpi::communicator &world,
//    VectorType &v,
//    const size_t src,
//    const int tag
//);

/**
 * Parallel version. matrix_a is stored distributed in different processor
 *
 * @tparam MatrixType
 * @tparam VectorType
 * @param matrix_a
 * @param b
 * @param x0
 * @param max_iter
 * @param tolerance
 * @param world
 * @return  only return in proc 0 is valid
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
    boost::mpi::communicator &world
) {
  if (world.rank() == kMasterProc) {
    return ConjugateGradientSolverMaster(
        matrix_a, b, x0, max_iter, tolerance, residue_restart_step, iter, world
    );
  } else {
    ConjugateGradientSolverSlave<MatrixType, VectorType>(
        matrix_a, world
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
    const ConjugateGradientSolverInstruction order,
    boost::mpi::communicator &world) {
  boost::mpi::broadcast(world, const_cast<ConjugateGradientSolverInstruction &>(order), kMasterProc);
}

inline ConjugateGradientSolverInstruction
SlaveReceiveBroadcastInstruction(boost::mpi::communicator world) {
  ConjugateGradientSolverInstruction instruction;
  boost::mpi::broadcast(world, instruction, kMasterProc);
  return instruction;
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
    boost::mpi::communicator &world
) {
  MasterBroadcastInstruction(start, world);

  double tol = b.Norm() * tolerance;

  VectorType ax0 = MatrixMultiplyVectorMaster(matrix_a, x0, world);
  VectorType r = b - ax0;
  double rk_2norm = r.Norm();
  if (rk_2norm < tol) {
    iter = 0;
    MasterBroadcastInstruction(finish, world);
    return x0;
  }
  VectorType p = r;
  VectorType x = x0;
  double rkp1_2norm;
  for (size_t k = 0; k < max_iter; k++) {
    MasterBroadcastInstruction(multiplication, world);
    VectorType ap = MatrixMultiplyVectorMaster(matrix_a, p, world);
    auto pap = (p * ap);
    auto alpha = rk_2norm / pap; //auto is double or complex
#ifndef NDEBUG
    assert(pap > 0);
    if (!std::isnormal(alpha)) {
      std::cout << "k : " << k << "\t pap : " << std::scientific << pap
                << "\t rk_2norm : " << std::scientific << rk_2norm
                << "\t alpha : " << std::scientific << alpha << std::endl;
      exit(1);
    }
#endif
    x += alpha * p;

    if (residue_restart_step > 0 && (k % residue_restart_step) == (residue_restart_step - 1)) {
      MasterBroadcastInstruction(multiplication, world);
      VectorType ax = MatrixMultiplyVectorMaster(matrix_a, x, world);
      r = b - ax;
    } else {
      r += (-alpha) * ap;
    }
    rkp1_2norm = r.Norm();   // return value of norm has definitely double type.

    if (rkp1_2norm < tol) {
      iter = k + 1;
      MasterBroadcastInstruction(finish, world);
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
  MasterBroadcastInstruction(finish, world);
  return x;
}

template<typename MatrixType, typename VectorType>
void ConjugateGradientSolverSlave(
    const MatrixType &matrix_a,
    boost::mpi::communicator &world
) {
  auto instrct = SlaveReceiveBroadcastInstruction(world);
  assert(instrct == start);
  MatrixMultiplyVectorSlave<MatrixType, VectorType>(matrix_a, world);
  instrct = SlaveReceiveBroadcastInstruction(world);
  while (instrct != finish) {
    //instrct == multiplication
    MatrixMultiplyVectorSlave<MatrixType, VectorType>(matrix_a, world);
    instrct = SlaveReceiveBroadcastInstruction(world);
  }
}

template<typename MatrixType, typename VectorType>
VectorType MatrixMultiplyVectorMaster(
    const MatrixType &mat,
    const VectorType &v,
    boost::mpi::communicator &world
) {
#ifdef GQPEPS_TIMING_MODE
  Timer cg_mat_vec_mult_timer("conjugate gradient matrix vector multiplication");
  Timer cg_broadcast_vec_timer("conjugate gradient broadcast vector");
#endif
  CGSolverBroadCastVector(const_cast<VectorType &>(v), world); //defined by user
#ifdef GQPEPS_TIMING_MODE
  cg_broadcast_vec_timer.PrintElapsed();
#endif
  std::vector<VectorType> res_list(world.size());
  VectorType res = mat * v;
#ifdef GQPEPS_TIMING_MODE
  Timer cg_gather_vec_timer("conjugate gradient gather vector");
#endif
  for (size_t i = 1; i < world.size(); i++) {
    CGSolverRecvVector(world, res_list[i], boost::mpi::any_source, boost::mpi::any_tag);
  }
#ifdef GQPEPS_TIMING_MODE
  cg_gather_vec_timer.PrintElapsed();
  Timer cg_gather_reduce_vec_timer("conjugate gradient gather reduce (summation) vector");
#endif
  for (size_t i = 1; i < world.size(); i++) {
    res += res_list[i];
  }
#ifdef GQPEPS_TIMING_MODE
  cg_gather_reduce_vec_timer.PrintElapsed();
  cg_mat_vec_mult_timer.PrintElapsed();
#endif
  return res;
}

template<typename MatrixType, typename VectorType>
void MatrixMultiplyVectorSlave(
    const MatrixType &mat,
    boost::mpi::communicator &world
) {
  VectorType v;
  CGSolverBroadCastVector(v, world);
  VectorType res = mat * v;
  CGSolverSendVector(world, res, kMasterProc, world.rank());
  //communicator, data, dest, tag
}

}//gqpeps

#endif //GRACEQ_VMC_PEPS_CONJUGATE_GRADIENT_SOLVER_H
