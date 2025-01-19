/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-07
*
* Description: QuantumLiquids/PEPS project. Unittests for Monte-Carlo Statistic MPI functions.
*/

#include "gtest/gtest.h"
#include <complex>    //std::norm
#include "qlpeps/monte_carlo_tools/statistics.h"

class StatisticsTest : public ::testing::Test {
 protected:
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  void SetUp() override {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    if (rank != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }

  void TearDown() override {
    // Clean up any resources after each test
  }
};

TEST_F(StatisticsTest, GatherStatisticSingleDataTest) {
  double single_data = rank;
  std::pair<double, double> actual_result = qlpeps::GatherStatisticSingleData(single_data, comm);

  if (rank == 0) {
    double expected_mean = static_cast<double>(mpi_size - 1) / 2.0;
    double n = static_cast<double>(mpi_size);
    // Variance formula: 0^2 + 1^2 + ... + (n-1)^2 = (n-1) * n * (2n-1) / 6
    double expected_std_err = std::sqrt((n - 1) * (2 * n - 1) / 6 - expected_mean * expected_mean) / std::sqrt(n - 1);

    EXPECT_EQ(actual_result.first, expected_mean);
    EXPECT_NEAR(actual_result.second, expected_std_err, 1e-15);
  }
}

TEST_F(StatisticsTest, GatherStatisticListOfDataTest) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  for (auto &value : data) {
    value += rank;
  }

  std::vector<double> actual_avgs;
  std::vector<double> actual_std_errs;
  qlpeps::GatherStatisticListOfData(data, comm, actual_avgs, actual_std_errs);

  if (rank == 0) {
    std::vector<double> expected_avgs = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (auto &avg : expected_avgs) {
      avg += static_cast<double>(mpi_size - 1) / 2.0;
    }
    double n = static_cast<double>(mpi_size);
    double expected_std_err = std::sqrt((n - 1) * (2 * n - 1) / 6 - (n - 1) / 2.0 * (n - 1) / 2.0) / std::sqrt(n - 1);
    std::vector<double> expected_std_errs(data.size(), expected_std_err);

    for (size_t i = 0; i < expected_avgs.size(); ++i) {
      EXPECT_EQ(expected_avgs[i], actual_avgs[i]);
      EXPECT_NEAR(expected_std_errs[i], actual_std_errs[i], 1e-15);
    }
  }
}

TEST_F(StatisticsTest, GatherStatisticComplexSingleDataTest) {
  std::complex<double> single_data(double(rank), double(rank + 1));
  std::pair<std::complex<double>, double>
      actual_result = qlpeps::GatherStatisticSingleData(single_data, comm);

  if (rank == 0) {
    std::complex<double> expected_mean =
        std::complex<double>(static_cast<double>(mpi_size - 1) / 2.0, static_cast<double>(mpi_size + 1) / 2.0);
    double n = static_cast<double>(mpi_size);
    double expected_std_err =
        std::sqrt((n - 1) * (2 * n - 1) / 6 - (n - 1) / 2.0 * (n - 1) / 2.0) * std::sqrt(2.0) / std::sqrt(n - 1);

    EXPECT_EQ(actual_result.first, expected_mean);
    EXPECT_NEAR(actual_result.second, expected_std_err, 1e-15);
  }
}

TEST_F(StatisticsTest, GatherStatisticListOfComplexDataTest) {
  std::vector<std::complex<double>>
      data = {std::complex<double>(1.0, 2.0), std::complex<double>(3.0, 4.0), std::complex<double>(5.0, 6.0)};
  for (auto &value : data) {
    value += std::complex<double>(rank, rank + 1);
  }

  std::vector<std::complex<double>> actual_avgs;
  std::vector<double> actual_std_errs;
  qlpeps::GatherStatisticListOfData(data, comm, actual_avgs, actual_std_errs);

  if (rank == 0) {
    std::vector<std::complex<double>> expected_avgs =
        {std::complex<double>(1.0, 2.0), std::complex<double>(3.0, 4.0), std::complex<double>(5.0, 6.0)};
    for (auto &avg : expected_avgs) {
      avg += std::complex<double>(static_cast<double>(mpi_size - 1) / 2.0, static_cast<double>(mpi_size + 1) / 2.0);
    }
    double n = static_cast<double>(mpi_size);
    double expected_std_err =
        std::sqrt((n - 1) * (2 * n - 1) / 6 - (n - 1) / 2.0 * (n - 1) / 2.0) * std::sqrt(2.0) / std::sqrt(n - 1);

    for (size_t i = 0; i < expected_avgs.size(); ++i) {
      EXPECT_EQ(expected_avgs[i], actual_avgs[i]);
      EXPECT_NEAR(actual_std_errs[i], expected_std_err, 1e-15);
    }
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  testing::InitGoogleTest(&argc, argv);
  auto test_err = RUN_ALL_TESTS();
  MPI_Finalize();
  return test_err;
}