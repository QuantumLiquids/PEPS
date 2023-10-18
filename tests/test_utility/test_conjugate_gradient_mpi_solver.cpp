// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2023-10-17
*
* Description: GraceQ/VMC-PEPS project. Unittests for conjugate gradient solver, mpi version
*/


#include "gtest/gtest.h"
#include "my_vector_matrix.h"
#include "gqpeps/consts.h"    //kMasterProc

using namespace gqten;

using namespace boost::mpi;

namespace gqpeps {
template<typename ElemT>
void CGSolverBroadCastVector(
    MyVector<ElemT> &x0,
    boost::mpi::communicator &world
) {
  broadcast(world, x0.GetElements(), kMasterProc);
}

template<typename ElemT>
void CGSolverSendVector(
    boost::mpi::communicator &world,
    const MyVector<ElemT> &v,
    const size_t dest,
    const int tag
) {
  world.send(dest, tag, v.GetElements());
}

template<typename ElemT>
size_t CGSolverRecvVector(
    boost::mpi::communicator &world,
    MyVector<ElemT> &v,
    const size_t src,
    const int tag
) {
  boost::mpi::status status = world.recv(src, tag, v.GetElements());
  return status.source();
}
}

#include "gqpeps/utility/conjugate_gradient_solver.h"

using namespace gqpeps;

template<typename ElemT>
void RunTestPlainCGSolverParallelCase(
    const MySquareMatrix<ElemT> &mat,
    const MyVector<ElemT> &b,
    const MyVector<ElemT> &x0,
    const MyVector<ElemT> x_res,
    communicator &world
) {
  size_t iter;
  auto x = ConjugateGradientSolver(mat, b, x0, 100, 1e-16, iter, world);
  if (world.rank() == kMasterProc) {
    x.Print();
    auto diff_vec = x - x_res;
    EXPECT_NEAR(diff_vec.Norm(), 0.0, 1e-13);
  }
}


TEST(TestPlainCGSolver, Parallel) {
  boost::mpi::communicator world;
  ::testing::TestEventListeners &listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }

  MySquareMatrix<double> dmat1;
  if (world.rank() == 0) {
    dmat1 = MySquareMatrix<double>({{0.0, 2.0, 2.0},
                                    {2.0, 0.0, 7.0},
                                    {2.0, 7.0, 8.0}});
  } else if (world.rank() == 1) {
    dmat1 = MySquareMatrix<double>({{1.0, 0.0, 0.0},
                                    {0.0, 5.0, 0.0},
                                    {0.0, 0.0, 7.0}});
  } else if (world.rank() == 2) {
    dmat1 = MySquareMatrix<double>({{0.0, 0.0, 1.0},
                                    {0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0}});
  }

  MyVector<double> db1({11.0,
                        12.0,
                        13.0});
  MyVector<double> dx01({-1.0, 1.0, 0.0});
  MyVector<double> dx_res1({33.0, -8.0, -2.0});
  RunTestPlainCGSolverParallelCase(dmat1, db1, dx01, dx_res1, world);

//  MySquareMatrix<GQTEN_Complex> zmat1({{1.0, 2.0, 3.0},
//                                       {2.0, 5.0, 7.0},
//                                       {3.0, 7.0, 15.0}});
//  MyVector<GQTEN_Complex> zb1({11.0, 12.0, 13.0});
//  MyVector<GQTEN_Complex> zx01({-1.0, 1.0, 0.0});
//  MyVector<GQTEN_Complex> zx_res1({33.0, -8.0, -2.0});
//  RunTestPlainCGSolverParallelCase(zmat1, zb1, zx01, zx_res1, world);
}


int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  boost::mpi::environment env(boost::mpi::threading::multiple);
  return RUN_ALL_TESTS();
}
