/*
* Author: Hao-Xin Wang<wanghaoxin1996@gmail.com>
* Creation Date: 2024-01-07
*
* Description: QuantumLiquids/PEPS project. Unittests for Monte-Carlo Statistic MPI functions.
*/

#include "gtest/gtest.h"
#include "qlpeps/monte_carlo_tools/statistics.h"

boost::mpi::environment env;

class StatisticsTest : public ::testing::Test {
 protected:
  boost::mpi::communicator world;
  void SetUp() override {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
  }

  void TearDown() override {
    // Clean up any resources after each test
  }
};

TEST_F(StatisticsTest, GatherStatisticSingleDataTest) {
  double data = world.rank();
  std::pair<double, double> actualResult = qlpeps::GatherStatisticSingleData(data, MPI_Comm(world));
  if (world.rank() == 0) {
    double expectedMean = (double) (world.size() - 1) / 2.0;
    double n = world.size();
    // 0^2 + 1^2 + 2^2 + 3^2 + ... + (n-1)^2 = (n-1) * n * (2n-1) / 6
    double
        expectedErr = std::sqrt((n - 1) * (2 * n - 1) / 6 - expectedMean * expectedMean) / std::sqrt(n - 1);
    EXPECT_EQ(actualResult.first, expectedMean);
    EXPECT_NEAR(actualResult.second, expectedErr, 1e-15);
  }
}

TEST_F(StatisticsTest, GatherStatisticListOfDataTest) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
  for (auto &datum : data) {
    datum += world.rank();
  }

  std::vector<double> actual_avgs;
  std::vector<double> actual_std_errs;
  qlpeps::GatherStatisticListOfData(data, world, actual_avgs, actual_std_errs);
  if (world.rank() == 0) {
    std::vector<double> expected_avgs = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (auto &avg : expected_avgs) {
      avg += double(world.size() - 1) / 2.0;
//      std::cout << avg << ",";
    }
//    std::cout << std::endl;
    double n = world.size();
    double expected_err =
        std::sqrt((n - 1) * (2 * n - 1) / 6
                      - (double) (n - 1) / 2.0 * (double) (n - 1) / 2.0) / std::sqrt(n - 1);
    std::vector<double> expected_std_errs(5, expected_err);
    for (size_t i = 0; i < expected_avgs.size(); i++) {
      EXPECT_EQ(expected_avgs[i], actual_avgs[i]);
      EXPECT_NEAR(expected_std_errs[i], actual_std_errs[i], 1e-15);
    }
  }
}

int main(int argc, char *argv[]) {
  boost::mpi::environment env;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}