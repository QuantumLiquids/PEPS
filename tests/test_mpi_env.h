#ifndef PEPS_TESTS_TEST_MPI_ENV_H
#define PEPS_TESTS_TEST_MPI_ENV_H

#include "gtest/gtest.h"
#include "mpi.h"

class MPIEnvironment : public ::testing::Environment {
 public:
  virtual void SetUp() {
    int mpi_error = MPI_Init(nullptr, nullptr);
    ASSERT_FALSE(mpi_error);
  }
  virtual void TearDown() {
    int mpi_error = MPI_Finalize();
    ASSERT_FALSE(mpi_error);
  }
  virtual ~MPIEnvironment() {}
};

class MPITest : public testing::Test {
 public:
  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }
};
#endif //PEPS_TESTS_TEST_MPI_ENV_H
