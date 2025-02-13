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

#endif //PEPS_TESTS_TEST_MPI_ENV_H
