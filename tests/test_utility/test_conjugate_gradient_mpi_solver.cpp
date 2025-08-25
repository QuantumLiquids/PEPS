// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-17
*
* Description: QuantumLiquids/PEPS project. Unittests for conjugate gradient solver, mpi version
*/


#include "gtest/gtest.h"
#include "my_vector_matrix.h"
#include "qlpeps/utility/conjugate_gradient_solver.h"

using namespace qlten;
using namespace qlpeps;

template<typename ElemT>
void RunTestPlainCGSolverParallelCase(
    const MySquareMatrix<ElemT> &mat,
    const MyVector<ElemT> &b,
    const MyVector<ElemT> &x0,
    const MyVector<ElemT> x_res,
    const MPI_Comm &comm
) {
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  size_t iter;
  auto x = ConjugateGradientSolver(mat, b, x0, 100, 1e-16, 20, iter, comm);
  if (rank == hp_numeric::kMPIMasterRank) {
    x.Print();
    auto diff_vec = x - x_res;
    EXPECT_NEAR(diff_vec.NormSquare(), 0.0, 1e-13);
  }
}

TEST(TestPlainCGSolver, ParallelDouble) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }

  MySquareMatrix<double> dmat1;
  if (rank == 0) {
    dmat1 = MySquareMatrix<double>({{0.0, 2.0, 2.0},
                                    {2.0, 0.0, 7.0},
                                    {2.0, 7.0, 8.0}});
  } else if (rank == 1) {
    dmat1 = MySquareMatrix<double>({{1.0, 0.0, 0.0},
                                    {0.0, 5.0, 0.0},
                                    {0.0, 0.0, 7.0}});
  } else if (rank == 2) {
    dmat1 = MySquareMatrix<double>({{0.0, 0.0, 1.0},
                                    {0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0}});
  }

  MyVector<double> db1({11.0,
                        12.0,
                        13.0});
  MyVector<double> dx01({-1.0, 1.0, 0.0});
  MyVector<double> dx_res1({33.0, -8.0, -2.0});
  RunTestPlainCGSolverParallelCase(dmat1, db1, dx01, dx_res1, comm);
 }

TEST(TestPlainCGSolver, ParallelComplexSameAsDouble) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }

  MySquareMatrix<QLTEN_Complex> cmat1;
  if (rank == 0) {
    cmat1 = MySquareMatrix<QLTEN_Complex>({
                                              {QLTEN_Complex(0.0, 0.0), QLTEN_Complex(2.0, 0.0), QLTEN_Complex(2.0, 0.0)},
                                              {QLTEN_Complex(2.0, 0.0), QLTEN_Complex(0.0, 0.0), QLTEN_Complex(7.0, 0.0)},
                                              {QLTEN_Complex(2.0, 0.0), QLTEN_Complex(7.0, 0.0), QLTEN_Complex(8.0, 0.0)}
                                          });
  } else if (rank == 1) {
    cmat1 = MySquareMatrix<QLTEN_Complex>({
                                              {QLTEN_Complex(1.0, 0.0), QLTEN_Complex(0.0, 0.0), QLTEN_Complex(0.0, 0.0)},
                                              {QLTEN_Complex(0.0, 0.0), QLTEN_Complex(5.0, 0.0), QLTEN_Complex(0.0, 0.0)},
                                              {QLTEN_Complex(0.0, 0.0), QLTEN_Complex(0.0, 0.0), QLTEN_Complex(7.0, 0.0)}
                                          });
  } else if (rank == 2) {
    cmat1 = MySquareMatrix<QLTEN_Complex>({
                                              {QLTEN_Complex(0.0, 0.0), QLTEN_Complex(0.0, 0.0), QLTEN_Complex(1.0, 0.0)},
                                              {QLTEN_Complex(0.0, 0.0), QLTEN_Complex(0.0, 0.0), QLTEN_Complex(0.0, 0.0)},
                                              {QLTEN_Complex(1.0, 0.0), QLTEN_Complex(0.0, 0.0), QLTEN_Complex(0.0, 0.0)}
                                          });
  }

  MyVector<QLTEN_Complex> cb1({
                                  QLTEN_Complex(11.0, 0.0),
                                  QLTEN_Complex(12.0, 0.0),
                                  QLTEN_Complex(13.0, 0.0)
                              });

  MyVector<QLTEN_Complex> cx01({
                                   QLTEN_Complex(-1.0, 0.0),
                                   QLTEN_Complex(1.0, 0.0),
                                   QLTEN_Complex(0.0, 0.0)
                               });

  MyVector<QLTEN_Complex> cx_res1({
                                      QLTEN_Complex(33.0, 0.0),
                                      QLTEN_Complex(-8.0, 0.0),
                                      QLTEN_Complex(-2.0, 0.0)
                                  });

  RunTestPlainCGSolverParallelCase(cmat1, cb1, cx01, cx_res1, comm);
}

TEST(TestPlainCGSolver, ParallelComplexImaginaryOffDiagonal) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (rank != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }

  MySquareMatrix<QLTEN_Complex> cmat2;
  if (rank == 0) {
    cmat2 = MySquareMatrix<QLTEN_Complex>({
                                              {QLTEN_Complex(2.0, 0.0), QLTEN_Complex(0.5, 1.0), QLTEN_Complex(0.0, -1.5)},
                                              {QLTEN_Complex(0.5, -1.0), QLTEN_Complex(2.0, 0.0), QLTEN_Complex(1.0, 0.5)},
                                              {QLTEN_Complex(0.0, 1.5), QLTEN_Complex(1.0, -0.5), QLTEN_Complex(3.0, 0.0)}
                                          });
  } else if (rank == 1) {
    cmat2 = MySquareMatrix<QLTEN_Complex>({
                                              {QLTEN_Complex(1.0, 0.0), QLTEN_Complex(0.3, 0.6), QLTEN_Complex(0.0, -1.0)},
                                              {QLTEN_Complex(0.3, -0.6), QLTEN_Complex(1.5, 0.0), QLTEN_Complex(0.5, 0.3)},
                                              {QLTEN_Complex(0.0, 1.0), QLTEN_Complex(0.5, -0.3), QLTEN_Complex(2.0, 0.0)}
                                          });
  } else if (rank == 2) {
    cmat2 = MySquareMatrix<QLTEN_Complex>({
                                              {QLTEN_Complex(1.3, 0.0), QLTEN_Complex(0.2, 0.4), QLTEN_Complex(0.0, -0.5)},
                                              {QLTEN_Complex(0.2, -0.4), QLTEN_Complex(1.5, 0.0), QLTEN_Complex(0.5, 0.2)},
                                              {QLTEN_Complex(0.0, 0.5), QLTEN_Complex(0.5, -0.2), QLTEN_Complex(1.0, 0.0)}
                                          });
  }


  MyVector<QLTEN_Complex> cb2({
                                  QLTEN_Complex(9.3, -7.35),
                                  QLTEN_Complex(0.0, -5.0),
                                  QLTEN_Complex(7.0, 7.0)
                              });

  MyVector<QLTEN_Complex> cx02({
                                   QLTEN_Complex(0.0, 0.0),
                                   QLTEN_Complex(0.0, 0.0),
                                   QLTEN_Complex(0.0, 0.0)
                               });

  MyVector<QLTEN_Complex> cx_res2({
                                      QLTEN_Complex(1.0, 0.5),
                                      QLTEN_Complex(-1.0, -1.5),
                                      QLTEN_Complex(2.0, 1.0)
                                  });

  RunTestPlainCGSolverParallelCase(cmat2, cb2, cx02, cx_res2, comm);
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  MPI_Init(nullptr, nullptr);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}
